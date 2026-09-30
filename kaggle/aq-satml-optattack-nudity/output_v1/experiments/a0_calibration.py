"""A0 -- Calibration: per-input-channel activation statistics for SD cross-attention.

Everything downstream depends on this. It answers the single go/no-go question for C1
(PLAN.md Sec. 2): are SD-1.5 attn2 activations heavy-tailed enough, at the alpha values
real SmoothQuant deployments use, for the activation-aware collision budget to exceed
the weight-only budget?

Writes, per attn2 to_k/to_v linear:
  * per-input-channel abs-max for EACH calibration sample (so downstream code can
    bootstrap calibration draws -- the distributional object of the theory),
  * per-tensor abs-max and a sample of per-token dynamic scales,
  * the fitted lognormal tail sigma of the per-channel magnitudes.

GPU, light (no image decoding). Cross-attention statistics use direct UNet probes;
statistics for upstream/self-attention/FF layers use real denoising trajectories so
their activation distribution is not estimated from unrelated random latents.
Run:  python experiments/a0_calibration.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.aquant import ActStats, act_scale_dynamic
from lib.deploy import linear_names
from lib.editors import build_pipe, cross_attn_keys
from lib.nsfw import load_coco_captions

OUT = os.environ.get("OUT", "outputs")
os.makedirs(OUT, exist_ok=True)
N_CAL = int(os.environ.get("N_CAL", 256))
STEPS = int(os.environ.get("CAL_STEPS", 4))       # timesteps sampled per prompt

pipe, device = build_pipe(dtype=torch.float16)
unet = pipe.unet
SCOPE = os.environ.get("SCOPE", "xattn")
names = ([k[:-len(".weight")] for k in cross_attn_keys(unet)] if SCOPE == "xattn"
         else linear_names(unet, SCOPE))
print(f"scope={SCOPE}")
print(f"{len(names)} cross-attention linears; {N_CAL} calibration prompts x {STEPS} steps")

prompts = load_coco_captions(os.environ.get("COCO_CSV"), n=N_CAL)

# Per-sample statistics: one ActStats per prompt, so we keep the per-sample rows that
# `craft_act.bootstrap_draws` needs. (A single pooled abs-max would destroy exactly the
# variation this project studies.)
per_sample = {n: [] for n in names}
tensor_absmax = {n: 0.0 for n in names}
token_scales = {n: [] for n in names}

sched = pipe.scheduler
sched.set_timesteps(50, device=device)
ts = sched.timesteps[:: max(1, len(sched.timesteps) // STEPS)][:STEPS]

for i, p in enumerate(prompts):
    st = ActStats(keep_token_scales=1).attach(unet, names)
    with torch.no_grad():
        if SCOPE == "xattn":
            # to_k/to_v consume the text embedding, which is independent of the latent
            # denoising trajectory. Direct timestep probes avoid unnecessary sampling.
            emb = pipe.encode_prompt(p, device, 1, do_classifier_free_guidance=False)[0]
            lat = torch.randn(1, unet.config.in_channels, 64, 64, device=device,
                              dtype=unet.dtype,
                              generator=torch.Generator(device=device).manual_seed(i))
            for t in ts:
                unet(lat, t, encoder_hidden_states=emb)
        else:
            # For attn1/FF and other upstream linears, activations depend on the evolving
            # latent. Feeding fresh random noise at late timesteps is not representative.
            pipe(p, guidance_scale=7.5, num_inference_steps=STEPS,
                 generator=torch.Generator(device=device).manual_seed(i),
                 output_type="latent")
    st.detach()
    for n in names:
        per_sample[n].append(st.chan_absmax[n].numpy().astype(np.float32))
        tensor_absmax[n] = max(tensor_absmax[n], st.tensor_absmax[n])
        token_scales[n].append(float(torch.cat(st.token_scales[n]).median()))
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(prompts)}", flush=True)

np.savez_compressed(os.path.join(OUT, os.environ.get("STATS_NAME", "act_stats.npz")),
                    **{n: np.stack(per_sample[n]) for n in names})

summary = {}
for n in names:
    A = np.stack(per_sample[n])                       # [n_cal, in]
    pooled = A.max(0)
    lg = np.log(np.maximum(pooled, 1e-12))
    summary[n] = dict(
        n_channels=int(A.shape[1]), tensor_absmax=float(tensor_absmax[n]),
        # lognormal tail sigma: the `tail` knob of theory/dist_binwidth.py
        tail_sigma=float(lg.std()),
        outlier_ratio=float(pooled.max() / np.median(pooled)),
        median_token_scale=float(np.median(token_scales[n])),
        # spread of the per-channel abs-max ACROSS calibration samples -- the source of
        # the draw-to-draw variation the robust box has to survive
        cross_sample_cv=float(np.median(A.std(0) / np.maximum(A.mean(0), 1e-12))),
    )
json.dump(summary, open(os.path.join(
    OUT, os.environ.get("STATS_NAME", "act_stats.npz").replace(".npz", "_summary.json")),
    "w"), indent=2)

# ---- wide survey: where do SD-1.5's activation outliers actually LIVE? ------------
# The erasers edit attn2 to_k/to_v, whose input is the (well-conditioned) text embedding.
# The massive activation outliers that motivate SmoothQuant are a property of some other
# part of the network. Locating them decides whether the activation-quant attack surface
# and the concept-edit surface overlap at all. Pooled stats only -- no per-sample rows.
WIDE_N = int(os.environ.get("WIDE_N", 64))
wide_names = [n for n, m in unet.named_modules() if isinstance(m, torch.nn.Linear)]
print(f"\nwide survey over {len(wide_names)} UNet linears, {WIDE_N} prompts")
wide = ActStats(keep_token_scales=1).attach(unet, wide_names)
with torch.no_grad():
    for i, p in enumerate(prompts[:WIDE_N]):
        pipe(p, guidance_scale=7.5, num_inference_steps=STEPS,
             generator=torch.Generator(device=device).manual_seed(10000 + i),
             output_type="latent")
wide.detach()
wrows = []
for n in wide_names:
    pooled = wide.chan_absmax[n].numpy().astype(np.float64)
    lg = np.log(np.maximum(pooled, 1e-12))
    kind = ("attn2" if "attn2" in n else "attn1" if "attn1" in n else
            "ff" if ".ff." in n else "proj" if "proj" in n else "other")
    wrows.append(dict(module=n, kind=kind, n_channels=int(pooled.size),
                      tail_sigma=float(lg.std()),
                      outlier_ratio=float(pooled.max() / np.median(pooled)),
                      absmax=float(pooled.max())))
wdf = pd.DataFrame(wrows)
wdf.to_csv(os.path.join(OUT, "act_stats_wide.csv"), index=False)
print(wdf.groupby("kind").agg(n=("module", "size"), tail_sigma=("tail_sigma", "median"),
                              outlier_ratio=("outlier_ratio", "median"),
                              max_outlier_ratio=("outlier_ratio", "max")
                              ).to_string(float_format=lambda v: f"{v:.3f}"))
print("Read-off: if the big outlier_ratio values sit OUTSIDE attn2, the activation-quant")
print("attack surface and the concept-edit surface do not overlap in this architecture.")

ts_sig = np.array([summary[n]["tail_sigma"] for n in names])
ratio = np.array([summary[n]["outlier_ratio"] for n in names])
print(f"\ntail sigma  : median={np.median(ts_sig):.2f}  p90={np.percentile(ts_sig,90):.2f}  "
      f"max={ts_sig.max():.2f}")
print(f"outlier ratio (max/median channel): median={np.median(ratio):.1f}  max={ratio.max():.1f}")
print("\nGO/NO-GO for C1: theory/dist_binwidth.py puts the activation-quant budget above")
print("the weight-only budget at tail sigma >~ 1.5 with alpha >= 0.75. Compare above.")
