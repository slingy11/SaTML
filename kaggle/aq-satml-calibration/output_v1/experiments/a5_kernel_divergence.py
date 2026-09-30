"""A5 -- fake-quant vs a TRUE fused INT8 GEMM (PLAN.md Sec. 3, claim C2).

This experiment measures how far a fake-quant reference and native integer GEMM differ
on SD-1.5 cross-attention. Only K1/K2 below describe the implemented paths:

  K1  int32 accumulation vs FP16/FP32 accumulation over K terms
  K2  order of scale application (per-element dequant before the sum vs one rescale)
Two additional sensitivity rows simulate accumulator saturation and per-tile activation
scaling. They are hypothetical and are not attributed to torch._int_mm or to a deployed
backend.

Part 1 is a pure microbenchmark on real weights and calibration activations. Part 2
checks that `torch._int_mm` agrees with the exact integer reference and records whether
the native instruction was actually available.

Needs outputs/act_stats.npz from A0 for realistic activation scales.
Run:  python experiments/a5_kernel_divergence.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.aquant import act_scale_dynamic, quant_act, quant_weight, weight_scale_pc
from lib.editors import build_pipe, cross_attn_keys, snapshot_original
from lib.kernels import (_int_mm_available, divergence_report, fake_quant_linear,
                         fused_int8_linear)

OUT = os.environ.get("OUT", "outputs")
os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
N_TOK = int(os.environ.get("N_TOK", 512))
REQUIRE_REAL_INT8 = os.environ.get("REQUIRE_REAL_INT8", "0") == "1"
real_int8 = bool(device == "cuda" and _int_mm_available(torch.device(device)))
print(f"native torch._int_mm execution available: {real_int8}")
if REQUIRE_REAL_INT8 and not real_int8:
    raise SystemExit("REQUIRE_REAL_INT8=1 but this device cannot execute torch._int_mm")

pipe, _ = build_pipe(device="cpu", dtype=torch.float32)
keys = cross_attn_keys(pipe.unet)
orig = snapshot_original(pipe, keys)
stats_path = os.path.join(OUT, "act_stats.npz")
STATS = np.load(stats_path) if os.path.exists(stats_path) else None
if STATS is None:
    print("! outputs/act_stats.npz not found; using unit-scale synthetic activations")

# ---- 1. per-lever divergence on real cross-attention weights --------------------
rows = []
g = torch.Generator().manual_seed(0)
for k in keys:
    mod = k[: -len(".weight")]
    W = orig[k].float().to(device)
    K = W.shape[1]
    if STATS is not None and mod in STATS.files:
        # reproduce the measured per-channel activation profile (this is what makes the
        # accumulation error realistic -- outlier channels dominate the sum)
        chan = torch.as_tensor(STATS[mod].max(0)).float().to(device)
        X = torch.randn(N_TOK, K, generator=g).to(device) * chan.view(1, -1) / 3.0
    else:
        X = torch.randn(N_TOK, K, generator=g).to(device)
    # saturation bound for K3: a typical requantizing epilogue keeps int16 headroom
    rep = divergence_report(X, W, acc_clamp=2 ** 15 - 1)
    for lever, d in rep.items():
        rows.append(dict(tensor=k, lever=lever, **d))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "a5_kernel_divergence.csv"), index=False)

print("=" * 78)
print("1. Reference-path differences on SD cross-attention (median over tensors)")
print("=" * 78)
gsum = df.groupby("lever").agg(max_abs=("max", "median"), mean_abs=("mean", "median"),
                               rel_to_output=("rel", "median"))
print(gsum.to_string(float_format=lambda v: f"{v:.3e}"))

floor = float(gsum.loc["quant_error_vs_fp32", "mean_abs"])
k1 = float(gsum.loc["K1_fused_vs_fake_fp16", "mean_abs"])
print(f"\nfake-vs-fused gap / honest quantization floor = {k1 / max(floor, 1e-30):.4f}")
print("Only K1/K2 compare the specified fake-quant and torch._int_mm semantics.")
print("The saturation and per-tile rows are hypothetical sensitivity analyses, not")
print("deployment claims, unless a named backend with those semantics is evaluated.")

# ---- 2. is our 'fused' path the same function the GPU runs? ---------------------
print("\n" + "=" * 78)
print("2. torch._int_mm agreement with the exact integer reference")
print("=" * 78)
check = {"device": device, "has_int_mm": hasattr(torch, "_int_mm"),
         "native_path_available": real_int8}
if device == "cuda" and hasattr(torch, "_int_mm"):
    W = orig[keys[0]].float().cuda()
    X = torch.randn(N_TOK, W.shape[1], generator=g).cuda()
    s_x, s_w = act_scale_dynamic(X), weight_scale_pc(W)
    q_x, q_w = quant_act(X, s_x), quant_weight(W, s_w)
    try:
        gpu = torch._int_mm(q_x.to(torch.int8).contiguous(),
                            q_w.t().contiguous().to(torch.int8))
        ref = (q_x.double() @ q_w.t().double()).round().to(torch.int32)
        check.update(int_mm_ok=True, max_int_diff=int((gpu - ref).abs().max()))
        print(f"  max |torch._int_mm - exact| = {check['max_int_diff']}  "
              f"({'EXACT' if check['max_int_diff'] == 0 else 'MISMATCH'})")
    except Exception as e:                       # shape constraints, unsupported arch
        check.update(int_mm_ok=False, error=str(e))
        print(f"  torch._int_mm unavailable for these shapes: {e}")
        print("  -> falls back to the exact integer reference; the 'fused' semantics")
        print("     (int32 accumulate, single rescale) are still what is evaluated.")
else:
    print("  no CUDA / no torch._int_mm: exact integer reference used for the fused path.")

json.dump(dict(levers=gsum.reset_index().to_dict("records"),
               fake_vs_fused_over_floor=k1 / max(floor, 1e-30), int_mm=check),
          open(os.path.join(OUT, "a5_kernel_divergence.json"), "w"), indent=2, default=str)
