# The true fused INT8 GEMM path, and the fake-quant / fused divergence probe.
#
# The distinction this project attacks (PLAN.md Sec. 3):
#
#   fake-quant :  y = (q_x * s_x) @ (q_w * s_w).T          -- dequantize FIRST, then
#                 accumulate in floating point. This is what every PTQ simulator
#                 (and our prior certifier) evaluates.
#   fused      :  acc = int32( q_x @ q_w.T ) ; y = acc * s_x * s_w.T
#                 -- accumulate EXACTLY in int32, rescale ONCE in the epilogue.
#                 This is what the deployed kernel computes.
#
# The reference and native paths agree in exact arithmetic. K1/K2 measure finite-
# precision differences in the two specified paths. Saturating accumulators and
# per-tile rescaling are exploratory semantics only: they are not properties of
# torch._int_mm and must not be presented as deployment gaps without naming and
# validating a backend that actually implements them.
import torch


# torch._int_mm is probed ONCE per device and the answer cached. Probing per call is
# both slow and dangerous: on an unsupported device (e.g. T4 / sm_75) the failed
# cublasLt call surfaces as a CUDA error, and repeatedly provoking CUDA errors inside a
# long generation run risks leaving the context unusable.
_INT_MM_OK = {}


def _int_mm_available(dev):
    key = str(dev)
    if key not in _INT_MM_OK:
        ok = False
        if dev.type == "cuda" and hasattr(torch, "_int_mm"):
            try:
                a = torch.ones(32, 32, dtype=torch.int8, device=dev)
                b = torch.ones(32, 32, dtype=torch.int8, device=dev)
                torch._int_mm(a, b)
                ok = True
            except Exception:
                ok = False
        _INT_MM_OK[key] = ok
        print(f"[kernels] torch._int_mm on {key}: "
              f"{'available (true fused INT8 GEMM)' if ok else 'unavailable -> exact integer fallback'}")
    return _INT_MM_OK[key]


def _int32_mm(a_i8, b_i8):
    """[M, K] x [K, N] int8 -> [M, N] int32, EXACT on every path.

    Uses torch._int_mm where the device supports it. Otherwise falls back to a float
    matmul that is still exact: with INT8 operands the accumulator is bounded by
    K * 127^2, so float32 (exact integers below 2^24) suffices for small K and float64
    is used beyond that. The fused SEMANTICS -- int32 accumulate, one rescale in the
    epilogue -- are preserved either way; only the instruction used differs.
    """
    if _int_mm_available(a_i8.device):
        try:
            return torch._int_mm(a_i8.contiguous(), b_i8)
        except (RuntimeError, NotImplementedError):
            _INT_MM_OK[str(a_i8.device)] = False
    K = a_i8.shape[-1]
    exact_in_fp32 = K * 127 * 127 <= 2 ** 24
    dt = torch.float32 if exact_in_fp32 else torch.float64
    return (a_i8.to(dt) @ b_i8.to(dt)).to(torch.int32)


def fused_int8_linear(q_x, s_x, q_w, s_w, acc_clamp=None):
    """q_x: [..., K] integer-valued float or int8 codes; s_x: [..., 1] per-token scale.
    q_w: [N, K] integer-valued weight codes; s_w: [N, 1] per-output-channel scale.

    acc_clamp: optional int32 saturation bound applied to the accumulator BEFORE the
    rescale -- lever K3. Real requantizing epilogues saturate; simulators do not.
    """
    lead = q_x.shape[:-1]
    K = q_x.shape[-1]
    a = q_x.reshape(-1, K).to(torch.int8)
    # torch._int_mm is an internal API with stricter layout requirements than
    # torch.matmul.  Materialise the transpose so a capable GPU does not silently fall
    # back to the reference path merely because the RHS is a non-contiguous view.
    b = q_w.to(torch.int8).t().contiguous()          # [K, N]
    acc = _int32_mm(a, b)                            # [M, N] int32, exact
    if acc_clamp is not None:
        acc = torch.clamp(acc, -int(acc_clamp), int(acc_clamp))
    y = acc.float() * s_x.reshape(-1, 1).float() * s_w.reshape(1, -1).float()
    return y.reshape(*lead, -1)


def fake_quant_linear(q_x, s_x, q_w, s_w, dtype=torch.float16):
    """Dequantize first, then accumulate in `dtype` -- the simulator path. `dtype`
    exposes lever K1: FP16 accumulation over K terms is where the paths separate."""
    X = (q_x.to(dtype) * s_x.to(dtype))
    W = (q_w.to(dtype) * s_w.to(dtype))
    return (X @ W.t()).float()


def tiled_token_scale(X, tile=64, qmax=127.0):
    """Lever K4: a fused kernel that computes the dynamic scale per K-tile rather than
    over the whole row sees a different scale than the reference implementation."""
    K = X.shape[-1]
    Xf = X.reshape(-1, K)
    pad = (-K) % tile
    if pad:
        Xf = torch.cat([Xf, torch.zeros(Xf.shape[0], pad, device=Xf.device, dtype=Xf.dtype)], 1)
    t = Xf.view(Xf.shape[0], -1, tile).abs().amax(-1) / qmax      # [M, ntiles]
    return t.clamp(min=1e-12)


def fused_tiled_linear(X, W, tile=64, qmax=127.0):
    """Lever K4: quantize each K-tile of the activations with its own scale and sum the
    per-tile int32 partial products. A tiled kernel that recomputes the dynamic scale
    per tile computes this, not the per-row version."""
    from .aquant import quant_weight, weight_scale_pc
    K = X.shape[-1]
    Xf = X.reshape(-1, K).float()
    s_w = weight_scale_pc(W.float(), qmax)
    q_w = quant_weight(W.float(), s_w, qmax)
    t_sc = tiled_token_scale(Xf, tile, qmax)                 # [M, ntiles]
    y = torch.zeros(Xf.shape[0], W.shape[0], device=X.device)
    for j in range(t_sc.shape[1]):
        sl = slice(j * tile, min((j + 1) * tile, K))
        if sl.start >= K:
            break
        s_j = t_sc[:, j:j + 1]
        q_j = torch.clamp(torch.round(Xf[:, sl] / s_j), -qmax, qmax)
        acc = _int32_mm(q_j.to(torch.int8), q_w[:, sl].to(torch.int8).t())
        y = y + acc.float() * s_j * s_w.reshape(1, -1)
    return y.reshape(*X.shape[:-1], -1)


def divergence_report(X, W, qmax=127.0, acc_clamp=None, tile=64):
    """Quantify K1-K4 on one linear layer. Returns per-lever max/mean |fused - fake|
    relative to the FP32 reference, so the levers are directly comparable."""
    from .aquant import act_scale_dynamic, quant_act, weight_scale_pc, quant_weight
    Xf, Wf = X.float(), W.float()
    s_x = act_scale_dynamic(Xf, qmax)
    s_w = weight_scale_pc(Wf, qmax)
    q_x, q_w = quant_act(Xf, s_x, qmax), quant_weight(Wf, s_w, qmax)

    ref = (Xf @ Wf.t())
    fused = fused_int8_linear(q_x, s_x, q_w, s_w)
    fused_sat = fused_int8_linear(q_x, s_x, q_w, s_w, acc_clamp=acc_clamp) if acc_clamp else fused
    fake16 = fake_quant_linear(q_x, s_x, q_w, s_w, torch.float16)
    fake32 = fake_quant_linear(q_x, s_x, q_w, s_w, torch.float32)

    def d(a, b):
        e = (a - b).abs()
        return dict(max=float(e.max()), mean=float(e.mean()),
                    rel=float(e.mean() / ref.abs().mean().clamp(min=1e-12)))

    return dict(
        K1_fused_vs_fake_fp16=d(fused, fake16),      # accumulation precision
        K1b_fused_vs_fake_fp32=d(fused, fake32),     # residual after removing FP16
        K2_fake_fp16_vs_fp32=d(fake16, fake32),      # order of dequant vs accumulate
        hypothetical_int16_acc_saturation=d(fused_sat, fused),
        # K4: a kernel that derives the dynamic scale per K-tile computes a DIFFERENT
        # function than one that derives it per row. Measured as the output divergence,
        # not as a scale ratio (the per-tile maxima trivially max to the per-row max).
        hypothetical_per_tile_act_scale=d(fused_tiled_linear(Xf, Wf, tile, qmax), fused),
        quant_error_vs_fp32=d(fused, ref),           # the honest quantization floor
    )
