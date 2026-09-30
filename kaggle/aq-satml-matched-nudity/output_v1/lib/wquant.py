# Weight quantizers, the craft-then-quantize construction, and shared-scale
# collision analysis. These are the exact operators used for every result in the
# paper (NF4 levels, block-64 abs-max, INT8, uniform INT4, and the craft eps/amax
# pinning are all unchanged from the reported experiments).
import numpy as np
import torch

# 16 NF4 levels (NormalFloat-4), as published.
NF4_L = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0], dtype=torch.float32)

# Uniform symmetric 4-bit: 16 evenly spaced levels, abs-max scaled (same interface as NF4).
INT4_L = torch.linspace(-1.0, 1.0, 16)

QUANT = {"nf4": NF4_L, "int4u": INT4_L}


def bnd(L):
    """Midpoints between adjacent levels = the bin boundaries."""
    return (L[1:] + L[:-1]) / 2.0


def block_absmax(W, bs=64):
    sh = W.shape
    n = W.numel()
    pad = (-n) % bs
    f = torch.cat([W.flatten(), torch.zeros(pad)]) if pad else W.flatten()
    b = f.view(-1, bs)
    return b.abs().amax(1, keepdim=True).clamp(min=1e-12), b, sh, n, pad


def codes(v, L):
    """Nearest-level index of `v` in the sorted level set `L`, without materialising an
    [N, len(L)] temporary. Exact: bucketize against the midpoints IS round-to-nearest
    (verified identical to the argmin form on 6M random values)."""
    return torch.bucketize(v.contiguous(), bnd(L).to(v.dtype))


def quant_with_scale(W, s, L, bs=64):
    """Quantize W to level set L using a GIVEN per-block scale s."""
    sh = W.shape
    n = W.numel()
    pad = (-n) % bs
    f = torch.cat([W.flatten(), torch.zeros(pad)]) if pad else W.flatten()
    b = f.view(-1, bs)
    idx = codes(b / s, L)
    return (L.to(b.dtype)[idx] * s).flatten()[:n].view(sh)


def quant(W, L, bs=64):
    """Standard round-to-nearest quantization to level set L (block abs-max scale)."""
    s, _, _, _, _ = block_absmax(W, bs)
    return quant_with_scale(W, s, L, bs)


def nf4(W, bs=64):
    return quant(W, NF4_L, bs)


def int8pt(W):
    """INT8 per-tensor symmetric."""
    s = (W.abs().max() / 127.0).clamp(min=1e-12)
    return torch.clamp(torch.round(W / s), -127, 127) * s


def int8pc(W):
    """INT8 per-(output-)channel symmetric."""
    s = (W.abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-12)
    return torch.clamp(torch.round(W / s), -127, 127) * s


def craft(Worig, Werased, L=NF4_L, bs=64):
    """Craft-then-quantize: return weights that read close to `Werased` at full
    precision but quantize EXACTLY to `Worig`'s codes under level set L.

    Each weight is placed at the interior point of its target (original) bin
    nearest the erased value; the block abs-max weight is pinned so the scale is
    preserved. By construction quant_with_scale(craft, s, L) == quant_with_scale(Worig, s, L)."""
    B = bnd(L)
    s, bo, sh, n, pad = block_absmax(Worig, bs)
    fe = torch.cat([Werased.flatten(), torch.zeros(pad)]) if pad else Werased.flatten()
    be = fe.view(-1, bs)
    idx = codes(bo / s, L)                                # target code = original's
    lo = torch.where(idx > 0, B[(idx - 1).clamp(min=0)], torch.full_like(idx, -1e9, dtype=torch.float))
    hi = torch.where(idx < len(L) - 1, B[idx.clamp(max=len(L) - 2)], torch.full_like(idx, 1e9, dtype=torch.float))
    ne = be / s
    eps = 1e-3
    nstar = torch.max(torch.min(ne, hi - eps), lo + eps).clamp(-1.0, 1.0)
    amax_i = bo.abs().argmax(1)
    ar = torch.arange(nstar.shape[0])
    nstar[ar, amax_i] = (bo / s)[ar, amax_i]
    Wstar = (nstar * s).flatten()[:n].view(sh)
    return Wstar, s


# ---- shared-scale collision analysis (Tables I & III) --------------------------
# Quantize BOTH original and edited weights with the SAME scale (set by the edited
# tensor) so any measured difference reflects the edit alone. Each returns
# (collision_mask, effective_edit_after_quantization).

def sc_fp16(w_o, w_e):
    do, de = w_o.half().float(), w_e.half().float()
    eq = de - do
    return (eq == 0), eq


def sc_int8pt(w_o, w_e):
    s = (w_e.abs().max() / 127.0).clamp(min=1e-12)
    co = torch.clamp(torch.round(w_o / s), -127, 127)
    ce = torch.clamp(torch.round(w_e / s), -127, 127)
    return (co == ce), (ce - co) * s


def sc_int8pc(w_o, w_e):
    s = (w_e.abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-12)
    co = torch.clamp(torch.round(w_o / s), -127, 127)
    ce = torch.clamp(torch.round(w_e / s), -127, 127)
    return (co == ce), (ce - co) * s


def sc_nf4(w_o, w_e, bs=64):
    shape = w_e.shape
    n = w_e.numel()
    pad = (-n) % bs
    fe, fo = w_e.flatten(), w_o.flatten()
    if pad:
        fe = torch.cat([fe, torch.zeros(pad)])
        fo = torch.cat([fo, torch.zeros(pad)])
    am = fe.view(-1, bs).abs().amax(1, keepdim=True).clamp(min=1e-12)
    io = codes(fo.view(-1, bs) / am, NF4_L)
    ie = codes(fe.view(-1, bs) / am, NF4_L)
    L = NF4_L.to(am.dtype)
    do = (L[io] * am).flatten()[:n]
    de = (L[ie] * am).flatten()[:n]
    coll = (io == ie).flatten()[:n]
    return coll.view(shape), (de - do).view(shape)


SC = {"fp16": sc_fp16, "int8_perchannel": sc_int8pc, "int8_pertensor": sc_int8pt, "nf4": sc_nf4}
# ---- a-priori bin widths (no fit, no edit) -------------------------------------
# The denominator of the survival index rho = E|dW| / Delta. Each is a property of the
# ORIGINAL weights and the quantizer alone, so rho is computable before any evaluation.

def nf4_bin_width(W, bs=64):
    """Mean local NF4 bin width in weight units (interior codes only)."""
    B = bnd(NF4_L)
    s, b, sh, n, pad = block_absmax(W, bs)
    ncoord = b / s
    idx = codes(ncoord, NF4_L)
    interior = (idx > 0) & (idx < len(NF4_L) - 1)
    widths = ((B[idx.clamp(max=len(B) - 1)] - B[(idx - 1).clamp(min=0)]) * s).flatten()[:n]
    m = interior.flatten()[:n]
    return float(widths[m].mean()) if bool(m.any()) else float(widths.mean())


def int8pc_bin_width(W):
    """INT8 per-output-channel bin width = the per-row scale."""
    return float((W.abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-12).mean())


def int8pt_bin_width(W):
    return float((W.abs().max() / 127.0).clamp(min=1e-12))


BIN_WIDTH = {"nf4": nf4_bin_width, "int8_perchannel": int8pc_bin_width,
             "int8_pertensor": int8pt_bin_width}
def edit_preservation(w_o, w_e, quantizer="int8_perchannel"):
    """How faithfully does a quantizer preserve an edit? Returns a dict.

    WARNING about the obvious metric: |q(W_e)-q(W_o)| / |dW| ("edit survival") is ~1.0
    at EVERY edit magnitude and does not discriminate. Collided weights contribute 0,
    while the few weights that do cross a bin boundary jump a full bin width -- and
    because round-to-nearest is unbiased those two effects cancel in the L1 sum. Measured
    on synthetic edits spanning rho = 0.008 to 8, survival stays in [0.998, 1.006] while
    the edit goes from almost entirely destroyed to perfectly preserved.

    Use `fidelity` (cosine between the realised and intended edit) instead: it runs
    0.113 -> 0.999 over that same range, monotonically. `distortion` is its L1 twin.
    """
    coll, eff = SC[quantizer](w_o, w_e)
    dW = (w_e - w_o).float()
    eff = eff.float()
    nd = float((dW * dW).sum())
    ne = float((eff * eff).sum())
    return dict(
        collision=float(coll.float().mean()),
        fidelity=float((eff * dW).sum() / max((ne * nd) ** 0.5, 1e-30)),
        distortion=float((eff - dW).abs().sum() / max(float(dW.abs().sum()), 1e-30)),
        survival_nondiagnostic=float(eff.abs().sum() / max(float(dW.abs().sum()), 1e-30)),
    )
