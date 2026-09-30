"""SMOKE -- exercise the whole deployment path in miniature, in ~2 minutes.

Stage 2 burned a GPU slot discovering that `prepare()` raised a CPU/CUDA device
mismatch on its first line of real use: calibration statistics come from numpy (CPU),
UNet weights live on the GPU. Every downstream experiment died before generating a
single image. This script exists so that never happens again -- it touches every step
A2/A4/A6 depend on, with one prompt and four denoising steps, and it FAILS LOUDLY.

Checks, in order:
  1. swap in W8A8Linear on the real UNet                     (device / dtype plumbing)
  2. generate on all three execution paths                   (fp16 / fake / fused)
  3. INVARIANT: the fp16 path must equal the untouched UNet. SmoothQuant migration is
     mathematically a no-op at full precision (X/s then W*s), so any drift here is a
     bug in the smoothing or scale bookkeeping, not a quantization effect.
  4. per-checkpoint recalibration on shipped weights          (the fixed point)
  5. craft_act on a real tensor: code agreement must be exactly 1.0
  6. report whether this GPU has a true fused INT8 GEMM

Run:  python experiments/a_smoke.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.craft_act import bootstrap_draws, code_agreement, craft_act
from lib.deploy import linear_names, prepare, recalibrate
from lib.editors import build_pipe, cross_attn_keys, snapshot_original
from lib.kernels import _int_mm_available
from lib.scoring import generate

OUT = os.environ.get("OUT", "outputs")
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAIL.append(name)


print("=" * 78)
print("SMOKE TEST -- W8A8 deployment path")
print("=" * 78)

pipe, device = build_pipe(dtype=torch.float16)
keys = cross_attn_keys(pipe.unet)
orig = snapshot_original(pipe, keys)
SCOPE = os.environ.get("SCOPE", "xattn")
stats_path = os.path.join(OUT, os.environ.get("STATS_FILE", "act_stats.npz"))
if not os.path.exists(stats_path):
    raise SystemExit(f"missing {stats_path} -- run experiments/a0_calibration.py first")
STATS = np.load(stats_path)
names = [n for n in linear_names(pipe.unet, SCOPE) if n in STATS.files]
names = names[: int(os.environ.get("SMOKE_MAX_LINEARS", 64))]
pre_swap_modules = dict(pipe.unet.named_modules())
baseline_weights = {n: pre_swap_modules[n].weight.detach().float().cpu().clone()
                    for n in names}
check("calibration statistics present", len(names) > 0,
      f"{len(names)} linears (scope={SCOPE})")
stats = {n: STATS[n] for n in names}

P, S, ST = ["a starry night, in the style of Vincent van Gogh"], [11], 4
base = generate(pipe, P, S, steps=ST)[0]
base_a = np.asarray(base, dtype=np.float32)

# 1. swap
try:
    made = prepare(pipe, names, act_stats=stats, alpha=0.75)
    check("prepare() swaps in W8A8Linear", len(made) == len(names), f"{len(made)} modules")
except Exception as e:
    check("prepare() swaps in W8A8Linear", False, repr(e)[:160])
    raise SystemExit("cannot continue without prepare()")

dev_ok = all(q.smooth is None or q.smooth.device == q.W.device for q in made.values())
check("smooth/weight buffers share a device", dev_ok)

# 2 + 3. all three paths, and the FP16 invariant
imgs = {}
for mode in ("fp16", "fake", "fused"):
    try:
        for q in made.values():
            q.mode = mode
        im = generate(pipe, P, S, steps=ST)[0]
        a = np.asarray(im, dtype=np.float32)
        imgs[mode] = a
        check(f"generate on '{mode}' path", np.isfinite(a).all() and a.std() > 1.0,
              f"mean={a.mean():.1f} std={a.std():.1f}")
    except Exception as e:
        check(f"generate on '{mode}' path", False, repr(e)[:160])

if "fp16" in imgs:
    d = np.abs(imgs["fp16"] - base_a).mean()
    check("INVARIANT fp16 path == untouched UNet", d < 1.0, f"mean|diff|={d:.3f}/255")
if "fake" in imgs and "fused" in imgs:
    d = np.abs(imgs["fake"] - imgs["fused"]).mean()
    print(f"  [INFO] fake vs fused mean|diff| = {d:.3f}/255 "
          f"(the C2 quantity; 0 would mean the paths are indistinguishable end-to-end)")
if "fp16" in imgs and "fake" in imgs:
    d = np.abs(imgs["fp16"] - imgs["fake"]).mean()
    print(f"  [INFO] fp16 vs W8A8 mean|diff| = {d:.3f}/255")

# 4. recalibration on shipped weights
try:
    recalibrate(made, {k: orig[k] for k in keys}, stats, 0.75)
    check("recalibrate() on shipped weights", True)
except Exception as e:
    check("recalibrate() on shipped weights", False, repr(e)[:160])

# 5. craft on a real tensor
try:
    W = baseline_weights[names[0]]
    draws = bootstrap_draws(stats[names[0]], W, alpha=0.75, n_draws=4, n_cal=32)
    Wc, info = craft_act(W, W + 0.01 * torch.randn_like(W), draws)
    ag = min(code_agreement(Wc, W, draws))
    check("craft_act keeps un-erased INT8 codes", ag >= 1.0 - 1e-9, f"agreement={ag:.6f}")
except Exception as e:
    check("craft_act keeps un-erased INT8 codes", False, repr(e)[:160])

# 6. real fused kernel?
if torch.cuda.is_available():
    ok = _int_mm_available(torch.device("cuda"))
    print(f"  [INFO] true fused INT8 GEMM available on {torch.cuda.get_device_name(0)}: {ok}")
    if not ok:
        print("         -> the 'fused' path uses an EXACT integer fallback with identical")
        print("            semantics; C2 needs sm_80+ (A100/H100) for the real instruction.")

print("=" * 78)
print(f"SMOKE: {'ALL PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
