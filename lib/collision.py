# Exact deployed-collision sets and the output-space optimal attack.
#
# A shipped tensor W "collides" with the original W0 under a deployment quantizer
# when the deployer's dequantized weights are identical for W and W0. For the
# recipes here the deployer derives every scale from the RECEIVED weights:
#
#   w8a8_sq    SmoothQuant migration s_c = a_c^alpha / max_r|W_rc|^(1-alpha) with
#              fixed calibration activations a, then symmetric per-row INT8 on W*s.
#   int8_wo    weight-only symmetric per-row INT8 (absmax/127).
#   int4_g128  weight-only symmetric INT4, groups of 128 input channels (absmax/7).
#   nf4_g64    weight-only NF4 codebook, groups of 64, absmax scaling.
#
# Scales depend on W only through column / row / group abs-maxima, so the
# collision set's closure is a box: original code intervals, capped so no maximum
# can grow, with the arg-max coordinates pinned so no maximum can shrink.
import torch

from .aquant import smooth_scales

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
                    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0], dtype=torch.float64)


def _levels(name):
    if name == "int8_wo":
        return torch.arange(-127, 128, dtype=torch.float64) / 127, None
    if name == "int4_g128":
        return torch.arange(-7, 8, dtype=torch.float64) / 7, 128
    if name == "int4_g64":
        return torch.arange(-7, 8, dtype=torch.float64) / 7, 64
    if name == "nf4_g64":
        return NF4, 64
    raise ValueError(name)


def _groups(W, g):
    o, i = W.shape
    g = i if g is None else g
    assert i % g == 0
    return W.reshape(o, i // g, g)


def wo_dequant(W, name):
    lv, g = _levels(name)
    lv = lv.to(W.device, W.dtype)
    Wg = _groups(W, g)
    amax = Wg.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    code = (Wg / amax).unsqueeze(-1).sub(lv).abs().argmin(-1)
    return (lv[code] * amax).reshape(W.shape), code.reshape(W.shape)


def _finish(Wg, lo, hi, pin, width, shape, eps_frac=1e-3):
    eps = eps_frac * width.expand_as(Wg)
    L = (lo + eps - Wg).clamp(max=0)
    H = (hi - eps - Wg).clamp(min=0)
    L = torch.where(pin, torch.zeros_like(L), L)
    H = torch.where(pin, torch.zeros_like(H), H)
    return L.reshape(shape), H.reshape(shape)


def wo_box(W, name):
    lv, g = _levels(name)
    lv = lv.to(W.device, W.dtype)
    Wg = _groups(W, g)
    amax = Wg.abs().amax(-1, keepdim=True)
    code = wo_dequant(W, name)[1].reshape(Wg.shape)
    big = torch.full((1,), 1e9, dtype=W.dtype, device=W.device)
    mids = torch.cat([lv[:1] - big, (lv[1:] + lv[:-1]) / 2, lv[-1:] + big])
    one = torch.ones((), dtype=W.dtype, device=W.device)
    lo = torch.maximum(mids[code], -one) * amax
    hi = torch.minimum(mids[code + 1], one) * amax
    pin = Wg.abs() == amax
    width = (mids[code + 1].clamp(max=1) - mids[code].clamp(min=-1)) * amax
    return _finish(Wg, lo, hi, pin, width, W.shape)


def _w8a8(W, a, alpha):
    s = smooth_scales(a, W, alpha=alpha).to(W.dtype)
    Ws = W * s.view(1, -1)
    sw = (Ws.abs().amax(1, keepdim=True) / 127).clamp(min=1e-12)
    code = torch.clamp(torch.round(Ws / sw), -127, 127)
    return code, sw, s


def w8a8_dequant(W, a, alpha=0.5):
    """Deployed weights in unsmoothed coordinates after the deployer recalibrates."""
    code, sw, s = _w8a8(W, a, alpha)
    return code * sw / s.view(1, -1), code


def w8a8_box(W, a, alpha=0.5, column_cap=True):
    code, sw, s = _w8a8(W, a, alpha)
    sv = s.view(1, -1)
    cap = sw * 127 / sv
    pin = (W * sv).abs() == (W * sv).abs().amax(1, keepdim=True)
    if column_cap:
        colcap = W.abs().amax(0, keepdim=True)     # keeps the migration scales fixed
        cap = torch.minimum(cap, colcap)
        pin = pin | (W.abs() == colcap)
    lo = torch.maximum((code - 0.5) * sw / sv, -cap)
    hi = torch.minimum((code + 0.5) * sw / sv, cap)
    return _finish(W, lo, hi, pin, sw / sv, W.shape)


def box(W, quant, a=None, alpha=0.5):
    return w8a8_box(W, a, alpha) if quant == "w8a8_sq" else wo_box(W, quant)


def dequant(W, quant, a=None, alpha=0.5):
    return w8a8_dequant(W, a, alpha) if quant == "w8a8_sq" else wo_dequant(W, quant)


def solve(Delta, G, L, H, max_iter=4000, gap_tol=1e-4, check_every=50,
          work_dtype=torch.float32):
    """min tr((X-Delta) G (X-Delta)^T) s.t. L <= X <= H (rows independent).

    FISTA with a diagonal Gershgorin majoriser D >= 2G, which keeps the proximal
    step a coordinate-wise clip, plus adaptive restart. Returns the iterate, its
    objective, a Frank-Wolfe lower bound on the optimum, the edit's own objective
    (the residual of X = 0), and the iteration count.
    """
    exact = (Delta, G, L, H)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Delta, G, L, H = (t.to(device=dev, dtype=work_dtype) for t in exact)
    D = 2 * G.abs().sum(1)
    X = Delta.clamp(L, H)
    Y, tk = X.clone(), 1.0
    base = float(torch.einsum("ij,jk,ik->", Delta, G, Delta))

    def gap(X):
        R = X - Delta
        f = float(torch.einsum("ij,jk,ik->", R, G, R))
        g = 2 * R @ G
        fw = float(torch.where(g > 0, g * (X - L), g * (X - H)).sum())
        return f, fw

    it = 0
    for it in range(1, max_iter + 1):
        grad = 2 * (Y - Delta) @ G
        Xn = (Y - grad / D).clamp(L, H)
        tn = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
        if float(((Xn - X) * grad).sum()) > 0:
            tn, Y = 1.0, Xn.clone()
        else:
            Y = Xn + ((tk - 1) / tn) * (Xn - X)
        X, tk = Xn, tn
        if it % check_every == 0:
            f, fw = gap(X)
            if fw <= gap_tol * base:
                break
    # Certify in the caller's precision: the Frank-Wolfe bound holds at any iterate.
    Delta, G, L, H = exact
    X = X.to(device=Delta.device, dtype=Delta.dtype).clamp(L, H)
    base = float(torch.einsum("ij,jk,ik->", Delta, G, Delta))
    f, fw = gap(X)
    return X, f, max(f - fw, 0.0), base, it
