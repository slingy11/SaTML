# Craft-then-quantize under ACTIVATION quantization: the distributional box.
#
# Weight-only craft (lib/wquant.craft) solves a DETERMINISTIC box constraint: put w*
# inside the target bin (b_lo, b_hi). Under an activation-aware recipe the bin itself
# moves with the calibration draw, because the per-input-channel migration factor
#     s(c) = amax|X[:, c]|^alpha / amax|W[:, c]|^(1-alpha)
# is estimated from calibration ACTIVATIONS. The constraint becomes a chance
# constraint, and its deterministic solution is the INTERSECTION of the target bins
# over the draws:
#
#     D_rob(r, c) = min_m hi_m(r, c)  -  max_m lo_m(r, c)
#
# The intersection always contains W_orig, so it is never empty: the question is not
# feasibility but how much budget survives, measured against the weight-only INT8 bin
# D_w = amax_c|W[r,c]|/127 (the prior paper's budget). See theory/dist_binwidth.py for
# the exact closed form and the 1/M decay law.
import torch

QMAX = 127.0
INF = float("inf")


def bin_bounds(W_ref, s_chan, qmax=QMAX):
    """Bounds, in UNSMOOTHED weight coordinates, of the INT8 bin that `W_ref` occupies
    under migration factor `s_chan` [in] and per-output-channel weight quantization.

    Returns (lo, hi), both [out, in], capped at the row abs-max so the weight scale
    itself cannot move.
    """
    dt = W_ref.dtype if W_ref.is_floating_point() else torch.float32
    s = s_chan.to(dt).view(1, -1).clamp(min=1e-12)
    Ws = W_ref.to(dt) * s                                   # smoothed weights
    s_w = (Ws.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-12)
    k = torch.clamp(torch.round(Ws / s_w), -qmax, qmax)
    # The bin is additionally capped at the row abs-max: a weight allowed past it would
    # BECOME the new abs-max, changing s_w and re-coding the whole output channel. This
    # also makes the saturated (|k| = qmax) bins finite.
    cap = s_w * qmax
    lo = torch.maximum((k - 0.5) * s_w, -cap) / s
    hi = torch.minimum((k + 0.5) * s_w, cap) / s
    return lo, hi


def robust_box(W_orig, draws, delta=0.0, qmax=QMAX):
    """Intersect (delta=0) or quantile-intersect (delta>0) the target bins over the
    calibration draws.

    draws : iterable of per-input-channel migration factors [in], one per calibration
            draw (e.g. bootstrap resamples of the COCO calibration set).
    delta : accepted per-side failure rate. delta=0 gives the deterministic robust box;
            delta>0 gives the chance-constrained relaxation, which is WIDER -- the
            attacker trades a delta failure rate for collision budget.

    Returns (lo_rob, hi_rob, width) with width = (hi_rob - lo_rob) clamped at 0.
    """
    LO, HI = [], []
    for s in draws:
        lo, hi = bin_bounds(W_orig, s, qmax)
        LO.append(lo)
        HI.append(hi)
    LO = torch.stack(LO)          # [M, out, in]
    HI = torch.stack(HI)
    if delta <= 0:
        lo_rob, hi_rob = LO.amax(0), HI.amin(0)
    else:
        fin = lambda T, v: torch.where(torch.isinf(T), torch.full_like(T, v), T)
        big = float(torch.nan_to_num(fin(HI, 0.0), posinf=0.0).abs().max()) * 1e3 + 1.0
        lo_rob = torch.quantile(fin(LO, -big).flatten(1), 1.0 - delta, dim=0).view_as(LO[0])
        hi_rob = torch.quantile(fin(HI, big).flatten(1), delta, dim=0).view_as(HI[0])
    return lo_rob, hi_rob, (hi_rob - lo_rob).clamp(min=0)


def craft_act(W_orig, W_erased, draws, delta=0.0, eps_frac=1e-3, qmax=QMAX,
              pin_absmax=True):
    """Weights that read as `W_erased` at FP16 but keep `W_orig`'s INT8 codes under
    EVERY calibration draw in `draws` (or all but a `delta` fraction).

    Returns (W_star, info). `info` reports the realised collision budget relative to the
    weight-only INT8 bin -- the quantity the theory predicts a-priori.
    """
    lo, hi, width = robust_box(W_orig, draws, delta, qmax)
    # NOTE: the intersection ALWAYS contains W_orig, so width > 0 is not informative.
    # The meaningful budget question is width relative to the weight-only INT8 bin
    # (the prior paper's budget) -- see theory/dist_binwidth.py.
    Dw_only = (W_orig.float().abs().amax(1, keepdim=True) / qmax).clamp(min=1e-12)
    feasible = width > 0
    # Saturated codes give a box with an open side (width = inf); the inset must stay
    # finite AND strictly positive there, or the clamp below either pushes the weight to
    # +-inf or leaves it exactly on a bin boundary, where rounding can go either way.
    Dw = Dw_only.expand_as(width)
    bound = torch.where(torch.isfinite(width), 0.25 * width, Dw)
    eps = torch.minimum(eps_frac * Dw, bound).clamp(min=0)
    lo_i, hi_i = lo + eps, hi - eps
    # nearest point of the robust box to the erased value; degenerate coords keep W_orig
    W_star = torch.maximum(torch.minimum(W_erased.float(), hi_i), lo_i)
    W_star = torch.where(feasible & torch.isfinite(W_star), W_star, W_orig.float())

    if pin_absmax:
        # The per-output-channel weight scale is set by the row abs-max of the SMOOTHED
        # weights, and the arg-max column can differ per draw. Pin the union of arg-max
        # columns to their original values so every draw's scale is preserved exactly.
        keep = torch.zeros_like(W_orig, dtype=torch.bool)
        for s in draws:
            am = (W_orig.float() * s.float().view(1, -1)).abs().argmax(dim=1)
            keep[torch.arange(W_orig.shape[0], device=am.device), am] = True
        W_star = torch.where(keep, W_orig.float(), W_star)

    moved = (W_star - W_orig.float()).abs()
    fin = torch.isfinite(width)
    ratio = (width / Dw_only)[fin]
    info = dict(
        n_draws=len(draws), delta=float(delta),
        # budget relative to the weight-only INT8 bin: >1 means activation quantization
        # genuinely widened the collision budget at that coordinate
        median_budget_ratio=float(ratio.median()) if ratio.numel() else 0.0,
        frac_wider_than_weight_only=float((ratio > 1).float().mean()) if ratio.numel() else 0.0,
        frac_10x=float((ratio > 10).float().mean()) if ratio.numel() else 0.0,
        pinned_frac=float(keep.float().mean()) if pin_absmax else 0.0,
        mean_abs_move=float(moved.mean()),
        # how much of the erasure edit the attacker actually got to keep at FP16
        edit_retention=float(moved.sum()
                             / (W_erased.float() - W_orig.float()).abs().sum().clamp(min=1e-12)),
    )
    return W_star, info


def code_agreement(W_a, W_b, draws, qmax=QMAX):
    """Fraction of weights whose INT8 code agrees, per draw. The attack requires this
    to be ~1.0 between the crafted weights and the ORIGINAL (un-erased) weights."""
    out = []
    for s in draws:
        dt = W_a.dtype if W_a.is_floating_point() else torch.float32
        sv = s.to(dt).view(1, -1).clamp(min=1e-12)
        def codes(W):
            Ws = W.to(dt) * sv
            s_w = (Ws.abs().amax(1, keepdim=True) / qmax).clamp(min=1e-12)
            return torch.clamp(torch.round(Ws / s_w), -qmax, qmax)
        out.append(float((codes(W_a) == codes(W_b)).float().mean()))
    return out


# ---- calibration draws ---------------------------------------------------------
def bootstrap_draws(per_sample_absmax, W, alpha=0.5, n_draws=32, n_cal=64, seed=0):
    """Resample the calibration set and recompute the migration factor each time.

    `per_sample_absmax` is [n_samples, in]: the per-input-channel abs-max contributed by
    each calibration sample (collected by lib.aquant.ActStats, one row per sample).
    The spread across the returned draws IS the distributional wrinkle. Note the
    direction, which is counterintuitive: a LARGER n_cal makes draws AGREE and therefore
    WIDENS the robust box (helping the attacker). What shrinks it is more draws (n_draws),
    i.e. audit-time calibration diversity -- see PLAN.md Sec. 2.
    """
    from .aquant import smooth_scales
    A = torch.as_tensor(per_sample_absmax).to(
        W.dtype if W.is_floating_point() else torch.float32)
    g = torch.Generator().manual_seed(seed)
    draws = []
    for _ in range(n_draws):
        idx = torch.randint(0, A.shape[0], (min(n_cal, A.shape[0]),), generator=g)
        draws.append(smooth_scales(A[idx].amax(0), W, alpha=alpha))
    return draws


def tail_energy_fraction(edit_dW, chan_absmax, pct=0.999):
    """The clipping surface: fraction of the erasure edit's energy carried by input
    channels whose calibrated range clips (channels above the `pct` quantile of the
    per-channel abs-max). High values mean the deployment quantizer destroys the edit
    without any weight code ever changing."""
    a = torch.as_tensor(chan_absmax).float()
    thr = torch.quantile(a, pct)
    mask = (a >= thr).view(1, -1)
    e = (edit_dW.float() ** 2)
    return float((e * mask).sum() / e.sum().clamp(min=1e-12))
