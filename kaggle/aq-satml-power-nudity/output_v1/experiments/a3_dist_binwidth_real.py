"""A3 -- The distributional bin-width model on REAL SD activation statistics.

theory/dist_binwidth.py derived and validated the model on synthetic activations:

    D_rob(r,c) = min_m (1/2 + r_m)/u_m + min_m (1/2 - r_m)/u_m     (exact)
    E[D_rob]   ~ 2 D_smooth / (M + 1)                              (1/M decay law)

This script re-runs all three checks against the activation statistics A0 measured on
SD-1.5, which is what turns the theory from a synthetic exercise into a claim about a
real model. It also reports the go/no-go quantity for C1: the fraction of cross-
attention weights whose activation-quant collision budget EXCEEDS the weight-only bin.

Needs outputs/act_stats.npz from A0. CPU only.
Run:  python experiments/a3_dist_binwidth_real.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.craft_act import bootstrap_draws, robust_box
from lib.editors import build_pipe, cross_attn_keys, snapshot_original

OUT = os.environ.get("OUT", "outputs")
os.makedirs(OUT, exist_ok=True)
QMAX = 127.0
STATS = np.load(os.path.join(OUT, "act_stats.npz"))

pipe, _ = build_pipe(device="cpu", dtype=torch.float32)
keys = cross_attn_keys(pipe.unet)
orig = snapshot_original(pipe, keys)
tensors = [(k, k[: -len(".weight")]) for k in keys if k[: -len(".weight")] in STATS.files]
print(f"{len(tensors)} tensors with calibration statistics")


def closed_form(W, draws, qmax=QMAX):
    """The exact D_rob (with the row abs-max cap) and the mean u, without
    materialising the per-draw boxes. Mirrors lib/craft_act.bin_bounds."""
    A, B, U = [], [], []
    for s in draws:
        sv = s.to(W.dtype).view(1, -1).clamp(min=1e-12)
        s_w = ((W * sv).abs().amax(1, keepdim=True) / qmax).clamp(min=1e-12)
        u = sv / s_w
        k = torch.round(W * u)
        r = W * u - k
        A.append(torch.minimum((0.5 + r) / u, W + qmax / u))   # row abs-max cap
        B.append(torch.minimum((0.5 - r) / u, qmax / u - W))
        U.append(u)
    return torch.stack(A).amin(0) + torch.stack(B).amin(0), torch.stack(U).mean(0)


rows = []
for alpha in [0.5, 0.75, 0.85]:
    for k, mod in tensors:
        # float64: the robust box is a DIFFERENCE of two nearly equal numbers, so a
        # float32 check reports spurious mismatches (0.13 exact-match) where the model
        # is in fact exact. The budget magnitudes are unaffected; only the check is.
        W = orig[k].double()
        A = STATS[mod]
        Dw0 = (W.abs().amax(1, keepdim=True) / QMAX).clamp(min=1e-12)
        draws = bootstrap_draws(A, W, alpha=alpha, n_draws=16, n_cal=64)
        _, _, bf = robust_box(W, draws)
        cf, ubar = closed_form(W, draws)
        m = torch.isfinite(bf) & torch.isfinite(cf)
        rel = ((bf[m] - cf[m]).abs() / bf[m].abs().clamp(min=1e-30))
        ratio = (bf / Dw0)[torch.isfinite(bf)]
        rows.append(dict(alpha=alpha, tensor=k,
                         closed_form_exact_frac=float((rel < 1e-6).float().mean()),
                         median_budget_ratio=float(ratio.median()),
                         frac_wider=float((ratio > 1).float().mean()),
                         frac_10x=float((ratio > 10).float().mean())))
    print(f"alpha={alpha} done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "a3_binwidth_real.csv"), index=False)

print("\n" + "=" * 78)
print("1. Exact closed form vs brute-force intersection, on real activation statistics")
print("=" * 78)
print(f"  fraction of coordinates matching to 1e-6: {df.closed_form_exact_frac.min():.5f} "
      f"(min over {len(df)} tensor x alpha cells)")

print("\n" + "=" * 78)
print("2. Activation-quant budget vs the weight-only INT8 bin  (the C1 go/no-go)")
print("=" * 78)
g = df.groupby("alpha").agg(median_budget_ratio=("median_budget_ratio", "median"),
                            frac_wider=("frac_wider", "median"),
                            frac_10x=("frac_10x", "median"))
print(g.to_string(float_format=lambda v: f"{v:.3f}"))

# ---- 3. the 1/M decay law on real statistics -----------------------------------
print("\n" + "=" * 78)
print("3. Decay with the number of calibration draws M   (law: E[D_rob * u_bar] = 2/(M+1))")
print("=" * 78)
k0, mod0 = tensors[0]
W0, A0 = orig[k0].double(), STATS[mod0]
decay = []
for M in [1, 2, 4, 8, 16, 32]:
    draws = bootstrap_draws(A0, W0, alpha=0.75, n_draws=M, n_cal=64)
    cf, ubar = closed_form(W0, draws)
    m = torch.isfinite(cf)
    obs = float((cf[m] * ubar[m]).mean())
    decay.append(dict(M=M, observed=obs, law=2.0 / (M + 1)))
    print(f"  M={M:<4} observed={obs:.4f}   law 2/(M+1)={2.0/(M+1):.4f}")
o = np.array([d["observed"] for d in decay]); l = np.array([d["law"] for d in decay])
r2 = float(1 - ((o - l) ** 2).sum() / ((o - o.mean()) ** 2).sum())
print(f"  -> R2 of the 1/M law on real statistics: {r2:.4f}")

# ---- 4. THE decisive sweep: budget vs the weight-only bin as a function of M -------
# M is the number of independent calibration draws the attacker must survive, i.e. the
# threat model. M=1 means the attacker knows the deployer's exact calibration set.
print("\n" + "=" * 78)
print("4. Budget vs weight-only INT8 bin by threat model (M = draws attacker must survive)")
print("=" * 78)
msweep = []
for alpha in [0.5, 0.75, 0.85]:
    for M in [1, 2, 4, 8, 16, 32]:
        rr, fw = [], []
        for k, mod in tensors[::4]:      # subsample: float64 x M draws is memory-heavy
            W = orig[k].double()
            Dw0 = (W.abs().amax(1, keepdim=True) / QMAX).clamp(min=1e-12)
            draws = bootstrap_draws(STATS[mod], W, alpha=alpha, n_draws=M, n_cal=64, seed=M)
            _, _, bf = robust_box(W, [d.double() for d in draws])
            ratio = (bf / Dw0)[torch.isfinite(bf)]
            rr.append(float(ratio.median())); fw.append(float((ratio > 1).double().mean()))
        msweep.append(dict(alpha=alpha, M=M, median_budget_ratio=float(np.median(rr)),
                           frac_wider=float(np.median(fw))))
        print(f"  alpha={alpha:<5} M={M:<3} median D_rob/D_w={msweep[-1]['median_budget_ratio']:.3f}  "
              f"frac wider={msweep[-1]['frac_wider']:.4f}", flush=True)
pd.DataFrame(msweep).to_csv(os.path.join(OUT, "a3_threat_model_sweep.csv"), index=False)
print("\nRead-off: the attack has a WIDER box than weight-only PTQ only where")
print("median D_rob/D_w > 1. If that holds only at small M, the attack requires knowing")
print("the deployer's calibration draw -- and randomising it is a complete defense.")

json.dump(dict(by_alpha=g.reset_index().to_dict("records"), decay=decay, decay_r2=r2,
               threat_model_sweep=msweep),
          open(os.path.join(OUT, "a3_binwidth_real.json"), "w"), indent=2)
