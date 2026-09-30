"""B1 -- The CPU decision gate. Run this BEFORE spending anything on generation.

Everything here is free (no GPU, no image generation) and it answers, for the ESD-u
edit, the two questions that decide whether B2 is worth running at all:

  PASSIVE   Is the per-weight edit small relative to the INT8 bin? ESD-u spreads its
            update over ~800M parameters where UCE writes a concentrated closed-form
            edit into ~20M, so the edit may simply fall inside the bins and be snapped
            away by ordinary quantization -- erasure destroyed with no attacker at all.
            Signal: high code agreement / low cosine FIDELITY under weight-only
            INT8. (The naive |q(We)-q(Wo)|/|dW| ratio is ~1.0 at every edit
            magnitude and never fires -- see lib.wquant.edit_preservation.)

  ACTIVE    Is the robust collision budget wide enough to retain the erasure at FP16?
            Stage 2 showed 25% weight-space retention buys only 8% behavioural
            retention, so anything below ~0.9 weight-space retention is hopeless.
            Signal: budget_ratio and edit_retention on the noxattn linears.

Needs outputs/act_stats_noxattn.npz (a0 with SCOPE=noxattn) and the B0 checkpoint.
Run:  python experiments/b1_esd_gate.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.craft_act import bootstrap_draws, code_agreement, craft_act, robust_box
from lib.deploy import linear_names
from lib.editors import build_pipe
from lib.wquant import edit_preservation

OUT = os.environ.get("OUT", "outputs")
QMAX = 127.0
TAG = os.environ.get("ESD_TAG", "esd_noxattn_gogh")
STATS_FILE = os.environ.get("STATS_FILE", "act_stats_noxattn.npz")
N_DRAWS_LIST = [int(x) for x in os.environ.get("N_DRAWS_LIST", "1,4,16").split(",")]
N_CAL = int(os.environ.get("N_CAL", 64))
ACTIVE_RETENTION_THRESHOLD = float(os.environ.get("ACTIVE_RETENTION_THRESHOLD", 0.9))

STATS = np.load(os.path.join(OUT, STATS_FILE))
pipe, _ = build_pipe(device="cpu", dtype=torch.float32)
orig = {n: p.detach().float() for n, p in pipe.unet.named_parameters()}

# Prefer a RELEASED ESD checkpoint (no training needed); fall back to B0's output.
ESD_CKPT = os.environ.get("ESD_CKPT", "")
if ESD_CKPT:
    from lib.esd import load_esd_checkpoint
    trained = load_esd_checkpoint(ESD_CKPT, orig)
else:
    trained = {k: v.float().clone() for k, v in
               load_file(os.path.join(OUT, f"{TAG}.safetensors")).items()}

lin = set(linear_names(pipe.unet, "noxattn"))
# only weights of Linear modules that ESD-u actually trained AND we have stats for
targets = []
for k in trained:
    if not k.endswith(".weight"):
        continue
    mod = k[: -len(".weight")]
    if mod in lin and mod in STATS.files:
        targets.append((k, mod))
print(f"{len(targets)} trained Linear tensors with calibration statistics "
      f"(of {len(trained)} trained tensors total)")
if not targets:
    raise SystemExit("no overlap between ESD-u's edit and the calibrated linears")

rows = []
for k, mod in targets:
    Wo, We = orig[k], trained[k].float()
    dW = We - Wo
    if float(dW.abs().mean()) < 1e-12:
        continue
    A = STATS[mod]
    Dw = (Wo.abs().amax(1, keepdim=True) / QMAX).clamp(min=1e-12)
    ep = edit_preservation(Wo, We, "int8_perchannel")
    base = dict(tensor=k, mean_abs_dW=float(dW.abs().mean()),
                rho=float(dW.abs().mean() / Dw.mean()),
                dW_over_bin=float((dW.abs() / Dw).mean()),
                int8_code_agreement=ep["collision"],
                int8_fidelity=ep["fidelity"], int8_distortion=ep["distortion"])
    for M in N_DRAWS_LIST:
        draws = bootstrap_draws(A, Wo, alpha=0.75, n_draws=M, n_cal=N_CAL)
        _, _, width = robust_box(Wo, draws)
        ratio = (width / Dw)[torch.isfinite(width)]
        Wst, info = craft_act(Wo, We, draws)
        rows.append(dict(base, n_draws=M, budget_ratio=float(ratio.median()),
                         frac_wider=float((ratio > 1).float().mean()),
                         edit_retention=info["edit_retention"],
                         craft_agreement=float(np.mean(code_agreement(Wst, Wo, draws)))))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "b1_esd_gate.csv"), index=False)

print("\n" + "=" * 84)
print("PASSIVE signal -- is the ESD-u edit small enough for INT8 to snap it away?")
print("=" * 84)
u = df.drop_duplicates("tensor")
print(f"  rho = E|dW| / bin width      : median={u.rho.median():.3f}  "
      f"p90={u.rho.quantile(.9):.3f}")
print(f"  INT8 code agreement          : median={u.int8_code_agreement.median():.3f}")
print(f"  INT8 edit FIDELITY (cosine)  : median={u.int8_fidelity.median():.3f}")
print("  (reference, UCE on attn2: code agreement 0.203, and its erasure is")
print("   behaviourally intact after W8A8 -- measured in A2)")
# NOTE: the earlier criterion used |q(We)-q(Wo)|/|dW|, which is ~1.0 at every rho and
# therefore never fires. Cosine fidelity is the discriminating statistic: on synthetic
# edits it runs 0.11 (rho=0.008) to 0.999 (rho=8) monotonically.
passive = u.int8_fidelity.median() < 0.8
print(f"  -> PASSIVE REVIVAL PLAUSIBLE: {passive}")

print("\n" + "=" * 84)
print("ACTIVE signal -- can a craft retain the erasure at FP16?")
print("=" * 84)
g = df.groupby("n_draws").agg(budget_ratio=("budget_ratio", "median"),
                              frac_wider=("frac_wider", "median"),
                              edit_retention=("edit_retention", "median"),
                              craft_agreement=("craft_agreement", "min"))
print(g.to_string(float_format=lambda v: f"{v:.4f}"))
print("  (reference, UCE on attn2 at M=1: budget 1.13, retention 0.257 -> 8% behavioural)")
best = float(g.edit_retention.max())
active = best > ACTIVE_RETENTION_THRESHOLD
print(f"  -> ACTIVE ATTACK PLAUSIBLE: {active}  (best weight-space retention {best:.3f})")

verdict = ("RUN B2 -- passive revival plausible" if passive else
           "RUN B2 -- active attack plausible" if active else
           "DO NOT RUN B2 -- neither signal present; report the negative and stop")
json.dump(dict(passive_plausible=bool(passive), active_plausible=bool(active),
               best_retention=best, verdict=verdict,
               active_retention_threshold=ACTIVE_RETENTION_THRESHOLD,
               int8_fidelity=float(u.int8_fidelity.median()),
               rho=float(u.rho.median()),
               n_tensors=len(u)),
          open(os.path.join(OUT, "b1_esd_gate.json"), "w"), indent=2)
print("\n" + "=" * 84)
print("GATE:", verdict)
print("=" * 84)
