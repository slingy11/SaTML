"""Distributional bin-width model -- CPU/numpy only, no torch, no GPU.

The paper's theoretical spine.

Weight-only craft solves a FIXED box constraint: keep w inside its INT8 bin, width
D_w = s_w(r) = amax_c|W[r,c]| / 127. Under an activation-aware W8A8 recipe
(SmoothQuant/AWQ) a per-input-channel factor s(c), estimated from CALIBRATION
ACTIVATIONS, is folded into the weights. Two things change:

  (i)  the bin width becomes HETEROGENEOUS across input channels,
           D_smooth(r,c) = s_w'(r) / s(c),
       so channels the calibration set saw as quiet get a much WIDER bin than the
       weight-only bin -- this is the extra collision budget the attack lives on;
  (ii) s(c) is a random variable over the calibration draw, so the attacker must sit
       in the INTERSECTION of the draws' bins, not in one fixed bin.

EXACT result (no assumption on code stability). Writing u_m = s_m(c)/s_w_m(r) and the
rounding residual r_m = w*u_m - round(w*u_m) in [-1/2, 1/2], the bin containing w under
draw m is [w - (1/2 + r_m)/u_m,  w + (1/2 - r_m)/u_m], so the robust box is

    D_rob(r,c) = min_m (1/2 + r_m)/u_m  +  min_m (1/2 - r_m)/u_m .

Consequences:
  * the intersection is never empty (it always contains the original weight) -- the
    right question is never "is it feasible" but "how much budget is left";
  * with r_m ~ U[-1/2, 1/2] i.i.d. over M draws and u_m ~ u_bar,
        E[D_rob]  ~  2 / ((M + 1) u_bar)  =  2 D_smooth / (M + 1),
    a 1/M decay law: robustness to more calibration draws costs the attacker budget;
  * because D_smooth is heavy-tailed across channels, a nontrivial fraction of weights
    still have D_rob > D_w even after that decay. That fraction IS the new attack
    surface, and it is what an audit can be designed to shrink.

Run:  python theory/dist_binwidth.py
"""
import json
import os

import numpy as np

OUT = os.environ.get("OUT", "outputs")
os.makedirs(OUT, exist_ok=True)
QMAX = 127.0


# ---- synthetic stand-ins for SD-1.5 cross-attention statistics -----------------
def make_weight(out=320, cin=768, seed=0):
    """Gaussian weights with roughly the scale of SD-1.5 attn2.to_k / to_v."""
    return np.random.default_rng(seed).normal(0, 0.03, size=(out, cin))


def make_act_pool(n_samples=1024, cin=768, tail=1.0, seed=1):
    """Per-input-channel abs-max contributed by each calibration sample. `tail` is the
    lognormal sigma: larger = heavier-tailed activations (a few massive outlier
    channels), the regime real UNet / transformer activations live in."""
    g = np.random.default_rng(seed)
    base = np.exp(g.normal(0, tail, size=cin))
    return base[None, :] * np.exp(g.normal(0, 0.5 * tail, size=(n_samples, cin)))


def smooth_scales(act_absmax, W, alpha=0.5, eps=1e-5):
    a = np.maximum(act_absmax, eps)
    w = np.maximum(np.abs(W).max(0), eps)
    return np.maximum(a ** alpha / w ** (1 - alpha), eps)


def draws_from_pool(pool, W, alpha=0.5, n_draws=24, n_cal=64, seed=2):
    g = np.random.default_rng(seed)
    return np.stack([smooth_scales(pool[g.integers(0, len(pool), n_cal)].max(0), W, alpha)
                     for _ in range(n_draws)])


def weight_only_bin(W, qmax=QMAX):
    """The prior paper's budget: INT8 per-output-channel bin width, no activations."""
    return np.broadcast_to(np.abs(W).max(1, keepdims=True) / qmax, W.shape)


# ---- brute force (ground truth) ------------------------------------------------
def robust_box_bruteforce(W, S, qmax=QMAX):
    LO, HI = [], []
    for s in S:
        sv = s[None, :]
        Ws = W * sv
        s_w = np.maximum(np.abs(Ws).max(1, keepdims=True) / qmax, 1e-12)
        k = np.clip(np.round(Ws / s_w), -qmax, qmax)
        cap = s_w * qmax          # a weight past the row abs-max would move s_w itself
        LO.append(np.maximum((k - 0.5) * s_w, -cap) / sv)
        HI.append(np.minimum((k + 0.5) * s_w, cap) / sv)
    return np.stack(HI).min(0) - np.stack(LO).max(0)


# ---- the closed form -----------------------------------------------------------
def robust_box_closedform(W, S, qmax=QMAX):
    """D_rob = min_m (1/2 + r_m)/u_m + min_m (1/2 - r_m)/u_m, plus u statistics."""
    A, B, U = [], [], []
    for s in S:
        sv = s[None, :]
        Ws = W * sv
        s_w = np.maximum(np.abs(Ws).max(1, keepdims=True) / qmax, 1e-12)
        u = sv / s_w
        k = np.round(W * u)
        r = W * u - k
        # distance to each side, capped by the row abs-max (which sits at qmax/u)
        A.append(np.minimum((0.5 + r) / u, W + qmax / u))
        B.append(np.minimum((0.5 - r) / u, qmax / u - W))
        U.append(u)
    U = np.stack(U)
    return np.stack(A).min(0) + np.stack(B).min(0), U.mean(0), U


def r2(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(1 - np.sum((obs - pred) ** 2) / max(np.sum((obs - obs.mean()) ** 2), 1e-30))


W = make_weight()
Dw0 = weight_only_bin(W)          # prior paper's budget, per coordinate
res = {}

# ---- 1. exact closed form vs brute force ---------------------------------------
print("=" * 78)
print("1. Closed form  D_rob = min_m (1/2+r_m)/u_m + min_m (1/2-r_m)/u_m  vs brute force")
print("=" * 78)
rows = []
for tail in [0.5, 1.0, 1.5]:
    pool = make_act_pool(tail=tail)
    for alpha in [0.5, 0.75, 0.85]:
        S = draws_from_pool(pool, W, alpha=alpha)
        bf = robust_box_bruteforce(W, S)
        cf, ubar, U = robust_box_closedform(W, S)
        m = np.isfinite(bf) & np.isfinite(cf)
        err = np.abs(bf[m] - cf[m])
        rel = err / np.maximum(np.abs(bf[m]), 1e-30)
        rows.append(dict(tail=tail, alpha=alpha, n=int(m.sum()), max_abs_err=float(err.max()),
                         median_rel_err=float(np.median(rel)),
                         frac_exact=float((rel < 1e-9).mean()), r2=r2(bf[m], cf[m])))
        r = rows[-1]
        print(f"  tail={tail:<4} alpha={alpha:<5} n={m.sum():>7}  exact-match={r['frac_exact']:.5f}  "
              f"median rel err={r['median_rel_err']:.2e}  R2={r['r2']:.6f}")
res["closed_form"] = rows

# ---- 2. is the activation-quant budget WIDER than the weight-only budget? -------
print()
print("=" * 78)
print("2. Budget vs the prior weight-only budget   (the 'wider collision budget' claim)")
print("=" * 78)
rows = []
for tail in [0.5, 1.0, 1.5]:
    pool = make_act_pool(tail=tail)
    for alpha in [0.5, 0.75, 0.85]:
        S = draws_from_pool(pool, W, alpha=alpha, n_draws=24, n_cal=64)
        bf = robust_box_bruteforce(W, S)
        m = np.isfinite(bf)
        ratio = bf[m] / Dw0[m]
        rows.append(dict(tail=tail, alpha=alpha, median_ratio=float(np.median(ratio)),
                         p90_ratio=float(np.percentile(ratio, 90)),
                         frac_wider=float((ratio > 1).mean()),
                         frac_10x=float((ratio > 10).mean())))
        r = rows[-1]
        print(f"  tail={tail:<4} alpha={alpha:<5} median D_rob/D_w={r['median_ratio']:.2f}  "
              f"p90={r['p90_ratio']:.2f}  frac wider={r['frac_wider']:.3f}  "
              f"frac >10x={r['frac_10x']:.3f}")
res["vs_weight_only"] = rows

# ---- 3. the 1/M decay law -------------------------------------------------------
print()
print("=" * 78)
print("3. Decay with the number of calibration draws M   (predicted E[D_rob] ~ 2/(M+1))")
print("=" * 78)
pool = make_act_pool(tail=1.0)
rows = []
for M in [1, 2, 4, 8, 16, 32, 64]:
    S = draws_from_pool(pool, W, alpha=0.5, n_draws=M, n_cal=64)
    bf = robust_box_bruteforce(W, S)
    _, ubar, _ = robust_box_closedform(W, S)
    m = np.isfinite(bf)
    obs = float(np.mean(bf[m] * ubar[m]))            # in units of D_smooth
    rows.append(dict(M=M, mean_norm_budget=obs, law=2.0 / (M + 1),
                     frac_wider_than_weight_only=float((bf[m] / Dw0[m] > 1).mean())))
    print(f"  M={M:<4} E[D_rob * u_bar]={obs:.4f}   law 2/(M+1)={2.0/(M+1):.4f}   "
          f"frac wider than weight-only={rows[-1]['frac_wider_than_weight_only']:.3f}")
res["decay_law"] = rows
print(f"  -> R2 of the 1/M law: "
      f"{r2([x['mean_norm_budget'] for x in rows], [x['law'] for x in rows]):.4f}")

# ---- 4. the audit lever: calibration set SIZE and diversity ----------------------
print()
print("=" * 78)
print("4. Audit lever: does a bigger calibration set shrink the attack surface?")
print("=" * 78)
rows = []
for n_cal in [8, 16, 64, 256, 1024]:
    S = draws_from_pool(pool, W, alpha=0.5, n_draws=24, n_cal=n_cal)
    bf = robust_box_bruteforce(W, S)
    m = np.isfinite(bf)
    rows.append(dict(n_cal=n_cal, median_ratio=float(np.median(bf[m] / Dw0[m])),
                     frac_wider=float((bf[m] / Dw0[m] > 1).mean())))
    print(f"  n_cal={n_cal:<5} median D_rob/D_w={rows[-1]['median_ratio']:.2f}  "
          f"frac wider than weight-only={rows[-1]['frac_wider']:.3f}")
res["ncal"] = rows

json.dump(res, open(os.path.join(OUT, "dist_binwidth_theory.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'dist_binwidth_theory.json')}")
