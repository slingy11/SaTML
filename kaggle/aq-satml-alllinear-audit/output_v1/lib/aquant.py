# Activation quantization: the new attack surface.
#
# Everything here follows the standard W8A8 conventions so that results transfer to
# real deployment stacks:
#   nn.Linear holds W of shape [out, in] and computes  y = X @ W.T (+ b).
#   Weights  -> INT8 symmetric per-OUTPUT-channel   (scale s_w, shape [out, 1])
#   Activations -> INT8 symmetric, either
#       "dynamic"  per-token scale  s_x = amax_j |X[..., j]| / 127     (input-dependent)
#       "static"   per-tensor scale from a calibration set, optionally a percentile
#                  (which CLIPS the tail -- see PLAN.md Sec. 2, the clipping surface)
#   SmoothQuant migrates difficulty per-INPUT-channel:  X <- X / s,  W <- W * s
#       with  s(c) = amax|X[:, c]|^alpha / amax|W[:, c]|^(1-alpha).
#
# The `mode` on W8A8Linear selects the execution path and is the object of study:
#   "fp16"  full precision (the audit path)
#   "fake"  quantize->dequantize->FP matmul (what every PTQ simulator does)
#   "fused" true INT8 GEMM with int32 accumulation (see lib/kernels.py)
import torch
import torch.nn as nn

QMAX = 127.0


# ---- activation quantizers ----------------------------------------------------
def act_scale_dynamic(X, qmax=QMAX):
    """Per-token symmetric scale: one scale per row of the flattened [..., in] input."""
    return (X.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-12)


def quant_act(X, s, qmax=QMAX):
    """Symmetric INT8 codes (float-valued, still exact integers) under scale `s`.
    Clamping is where a calibrated (static/percentile) range destroys tail signal."""
    return torch.clamp(torch.round(X / s), -qmax, qmax)


def fake_quant_act(X, s, qmax=QMAX):
    return quant_act(X, s, qmax) * s


def act_scale_static(absmax, pct=1.0, qmax=QMAX):
    """Static per-tensor scale from a calibration abs-max. `pct` < 1 shrinks the range
    (percentile clipping), which is exactly the clipping-tail surface of PLAN.md Sec. 2."""
    return (torch.as_tensor(absmax).float() * float(pct) / qmax).clamp(min=1e-12)


# ---- weight quantizer (per-output-channel INT8, the W of W8A8) -----------------
def weight_scale_pc(W, qmax=QMAX):
    return (W.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-12)


def quant_weight(W, s, qmax=QMAX):
    return torch.clamp(torch.round(W / s), -qmax, qmax)


def fake_quant_weight(W, s=None, qmax=QMAX):
    s = weight_scale_pc(W, qmax) if s is None else s
    return quant_weight(W, s, qmax) * s


# ---- SmoothQuant / AWQ-style activation-aware migration -----------------------
def smooth_scales(act_absmax_per_channel, W, alpha=0.5, eps=1e-5):
    """Per-INPUT-channel migration factor s(c) (SmoothQuant Eq. 4).

    `act_absmax_per_channel` is [in], `W` is [out, in]. Returns s of shape [in].
    NOTE the asymmetry that makes this project's theory necessary: `s` is estimated
    from CALIBRATION ACTIVATIONS, so it is a random variable over the calibration
    draw -- while the weight bin width alone is deterministic.
    """
    dt = W.dtype if W.is_floating_point() else torch.float32
    # calibration statistics arrive from numpy (CPU) while the weights may live on GPU
    a = torch.as_tensor(act_absmax_per_channel).to(device=W.device, dtype=dt).clamp(min=eps)
    w = W.abs().amax(dim=0).to(dt).clamp(min=eps)
    s = (a.pow(alpha) / w.pow(1.0 - alpha)).clamp(min=eps)
    return s


def apply_smoothing(W, s):
    """Fold the migration into the weights: W <- W * diag(s) (columns = input channels).
    The matching X <- X / s is applied at run time by W8A8Linear."""
    return W * s.view(1, -1)


def awq_scales(act_absmax_per_channel, alpha=0.5, eps=1e-5):
    """AWQ-style: scale purely by activation magnitude (no weight term). Same shape and
    same distributional character as `smooth_scales`; kept separate so the two
    activation-aware recipes can be compared as distinct geometries."""
    a = torch.as_tensor(act_absmax_per_channel).float().clamp(min=eps)
    return (a.pow(alpha) / a.pow(alpha).mean()).clamp(min=eps)


def _same_device(t, ref):
    return torch.as_tensor(t).to(ref.device)


# ---- the quantized linear ------------------------------------------------------
class W8A8Linear(nn.Module):
    """Drop-in replacement for nn.Linear with a selectable execution path.

    smooth : optional per-input-channel migration factor [in]. When present the
             weight is stored already smoothed and the input is divided by `s` at
             run time (mathematically a no-op at FP16, NOT a no-op once quantized --
             which is the whole point).
    act    : "dynamic" (per-token scale) or "static" (calibrated per-tensor scale).
    """

    def __init__(self, lin, mode="fake", act="dynamic", smooth=None,
                 act_absmax=None, act_pct=1.0, qmax=QMAX):
        super().__init__()
        self.mode, self.act, self.qmax = mode, act, qmax
        # Keep the received checkpoint weights separate from the migrated weights used
        # by the quantizer.  The full-precision audit must evaluate the checkpoint as it
        # was shipped; applying X/s and W*s in the audit path introduces avoidable
        # low-precision rounding and previously made the path labelled "fp16" execute
        # an FP32 matmul.
        self.compute_dtype = lin.weight.dtype
        W_fp = lin.weight.detach().float()
        self.register_buffer("W_fp", W_fp.clone())
        W = W_fp
        self.register_buffer("smooth", None if smooth is None
                             else smooth.float().view(-1).to(W.device))
        if smooth is not None:
            W = apply_smoothing(W, self.smooth)
        self.register_buffer("W", W)
        self.register_buffer("bias", None if lin.bias is None else lin.bias.detach().float())
        s_a = None if act_absmax is None else act_scale_static(act_absmax, act_pct, qmax)
        self.register_buffer("s_a_static", None if s_a is None else s_a.reshape(1))
        self.register_buffer("s_w", weight_scale_pc(W, qmax))
        self.register_buffer("q_w", quant_weight(W, self.s_w, qmax))

    # -- keep the buffers consistent when an attacker rewrites the FP weights --
    def set_weight(self, W):
        W = W.float().to(self.W.device)
        self.W_fp.copy_(W)
        if self.smooth is not None:
            W = apply_smoothing(W, self.smooth)
        self.W.copy_(W)
        self.s_w.copy_(weight_scale_pc(W, self.qmax))
        self.q_w.copy_(quant_weight(W, self.s_w, self.qmax))

    def act_scale(self, X):
        if self.act == "static" and self.s_a_static is not None:
            return self.s_a_static.to(X.dtype).to(X.device)
        return act_scale_dynamic(X, self.qmax)

    def forward(self, X):
        if self.mode == "fp16":
            # The experiments construct this wrapper after pipe.to(torch.float16), so
            # compute_dtype is FP16.  Keeping this explicit prevents a silent promotion
            # to FP32 and makes the path name truthful.  Use the same fused linear
            # primitive as nn.Linear: spelling this as matmul followed by bias addition
            # changes FP16 rounding by one ULP on CUDA.
            Xa = X.to(self.compute_dtype)
            bias = None if self.bias is None else self.bias.to(
                device=Xa.device, dtype=self.compute_dtype)
            y = torch.nn.functional.linear(
                Xa, self.W_fp.to(device=Xa.device, dtype=self.compute_dtype), bias)
        else:
            Xf = X.float()
            if self.smooth is not None:
                Xf = Xf / self.smooth
            s_x = self.act_scale(Xf)
            if self.mode == "fake":
                # A fake-quant deployment dequantizes to the model's compute dtype
                # before GEMM.  Previously both operands stayed FP32, invalidating the
                # claimed FP16-vs-fused comparison.
                Xq = fake_quant_act(Xf, s_x, self.qmax).to(self.compute_dtype)
                Wq = (self.q_w * self.s_w).to(self.compute_dtype)
                y = Xq @ Wq.t()
            elif self.mode == "fused":
                from .kernels import fused_int8_linear
                y = fused_int8_linear(quant_act(Xf, s_x, self.qmax), s_x,
                                      self.q_w, self.s_w)
            else:
                raise ValueError(f"unknown mode {self.mode}")
        if self.mode != "fp16" and self.bias is not None:
            y = y + self.bias.to(device=y.device, dtype=y.dtype)
        return y.to(X.dtype)


# ---- calibration: per-input-channel activation statistics ---------------------
class ActStats:
    """Forward hooks on selected nn.Linear modules, accumulating what an activation
    quantizer actually needs: per-input-channel abs-max, per-tensor abs-max, and a
    sample of per-token scales (the empirical law of s_x used by theory/)."""

    def __init__(self, keep_token_scales=256):
        self.chan_absmax, self.tensor_absmax, self.token_scales = {}, {}, {}
        self.keep = keep_token_scales
        self._handles = []

    def _hook(self, name):
        def fn(mod, inp, out):
            X = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            cm = X.abs().amax(0).cpu()
            prev = self.chan_absmax.get(name)
            self.chan_absmax[name] = cm if prev is None else torch.maximum(prev, cm)
            tm = float(X.abs().max())
            self.tensor_absmax[name] = max(self.tensor_absmax.get(name, 0.0), tm)
            ts = self.token_scales.setdefault(name, [])
            if len(ts) < self.keep:
                ts.append(act_scale_dynamic(X).flatten().cpu())
        return fn

    def attach(self, model, names):
        mods = dict(model.named_modules())
        for n in names:
            self._handles.append(mods[n].register_forward_hook(self._hook(n)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        return self

    def summary(self):
        return {n: dict(chan_absmax=self.chan_absmax[n],
                        tensor_absmax=self.tensor_absmax[n],
                        token_scales=torch.cat(self.token_scales.get(n, [torch.zeros(1)])))
                for n in self.chan_absmax}


# ---- model surgery -------------------------------------------------------------
def swap_linears(model, names, mode="fake", act="dynamic", smooth=None,
                 act_absmax=None, act_pct=1.0):
    """Replace `names` (module paths of nn.Linear) with W8A8Linear. `smooth` and
    `act_absmax` are dicts keyed by the same names. Returns {name: W8A8Linear}."""
    mods = dict(model.named_modules())
    made = {}
    for n in names:
        lin = mods[n]
        parent = mods[n.rsplit(".", 1)[0]] if "." in n else model
        q = W8A8Linear(lin, mode=mode, act=act,
                       smooth=None if smooth is None else smooth.get(n),
                       act_absmax=None if act_absmax is None else act_absmax.get(n),
                       act_pct=act_pct).to(lin.weight.device)
        setattr(parent, n.rsplit(".", 1)[-1], q)
        made[n] = q
    return made


def set_mode(made, mode):
    for q in made.values():
        q.mode = mode
    return made
