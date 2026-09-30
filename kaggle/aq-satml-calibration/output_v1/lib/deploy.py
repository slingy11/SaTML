# Putting a checkpoint on a deployment path and scoring it.
#
# The three paths of PLAN.md Sec. 3 (fp16 / fake_w8a8 / fused_int8) differ only in the
# `mode` of the swapped-in W8A8Linear, so a single generation harness covers all of
# them and every contrast is exactly paired (same seeds, same prompts, same weights).
import torch

from .aquant import W8A8Linear, set_mode, smooth_scales, swap_linears
from .scoring import generate


def prepare(pipe, names, act_stats=None, alpha=None, act="dynamic", act_pct=1.0,
            calib_idx=None):
    """Swap `names` (module paths of nn.Linear) for W8A8Linear.

    act_stats : {name: [n_samples, in] per-sample per-channel abs-max} from A0.
    alpha     : SmoothQuant exponent; None disables migration (plain W8A8).
    calib_idx : which calibration samples the DEPLOYER used. The attack's whole premise
                is that the attacker must be robust to not knowing this exactly, so it
                is an explicit knob rather than an implicit default.
    """
    smooth, absmax = None, None
    if act_stats is not None:
        mods = dict(pipe.unet.named_modules())
        absmax = {}
        smooth = {} if alpha is not None else None
        for n in names:
            A = torch.as_tensor(act_stats[n]).float()
            A = A if calib_idx is None else A[calib_idx]
            pooled = A.amax(0)
            absmax[n] = float(pooled.max())
            if smooth is not None:
                smooth[n] = smooth_scales(pooled, mods[n].weight.detach().float())
    return swap_linears(pipe.unet, names, mode="fp16", act=act, smooth=smooth,
                        act_absmax=absmax, act_pct=act_pct)


def load_weights(made, wd):
    """Install a weight dict (keys ending in `.weight`, as elsewhere in this codebase)
    into the swapped modules, keeping every quantization buffer consistent."""
    for k, v in wd.items():
        n = k[: -len(".weight")] if k.endswith(".weight") else k
        if n in made:
            made[n].set_weight(v)


def score_paths(pipe, made, wd, prompts, seeds, score_fn, paths=("fp16", "fake", "fused"),
                steps=30, guidance=7.5, act_stats=None, alpha=None, calib_idx=None):
    """Generate and score the same checkpoint on each execution path.

    `score_fn(images) -> per-image float array`. Returns {path_label: scores}, using the
    certifier's path names so the output drops straight into lib.certify.
    """
    label = {"fp16": "fp16", "fake": "fake_w8a8", "fused": "fused_int8"}
    if act_stats is not None and alpha is not None:
        recalibrate(made, wd, act_stats, alpha, calib_idx)   # deployer calibrates on what it received
    else:
        load_weights(made, wd)
    out = {}
    for p in paths:
        set_mode(made, p)
        imgs = generate(pipe, prompts, seeds, steps=steps, guidance=guidance)
        out[label[p]] = score_fn(imgs)
    set_mode(made, "fp16")
    return out


# Which linears a W8A8 pipeline can migrate, by experimental scope. `xattn` is the
# surface every closed-form eraser (UCE, TIME) and ESD-x writes to. `noxattn` is the
# quantization scope for the ESD-u track: 118 linears, of which ESD-u actually trains 96
# (64 attn1 + 32 ff) -- it excludes time_emb_proj by name, so the 5442x-outlier layer is
# NOT edited. b1/b2 intersect this scope with the trained tensors, so the difference is
# handled; measured outlier ratios of the trained set are max 69.7 / median 2.6, versus
# 11.1 for attn2.
SCOPES = {
    "xattn":   lambda n: "attn2" in n and (n.endswith("to_k") or n.endswith("to_v")),
    "noxattn": lambda n: ("attn2" not in n and "time_embedding" not in n
                          and not n.startswith("conv_out")),
    "all":     lambda n: True,
}


def linear_names(unet, scope="xattn"):
    """Module paths (not weight keys) of the nn.Linear layers in `scope`."""
    sel = SCOPES[scope]
    return [n for n, m in unet.named_modules()
            if isinstance(m, (torch.nn.Linear, W8A8Linear)) and sel(n)]


def edit_energy_split(orig_sd, edited_sd, unet):
    """Where did a fine-tuned edit actually LAND? Splits squared edit energy across
    Linear (quantizer-migratable) vs Conv (not handled by SmoothQuant here) vs other.

    This is the honest scoping check for ESD-u: if most of its edit energy sits in
    convolutions, the activation-quant analysis only covers a minority of the edit and
    any conclusion must say so.
    """
    kinds = {}
    for n, m in unet.named_modules():
        t = ("linear" if isinstance(m, (torch.nn.Linear, W8A8Linear))
             else "conv" if isinstance(m, torch.nn.modules.conv._ConvNd) else None)
        if t:
            kinds[n] = t
    tot = {"linear": 0.0, "conv": 0.0, "other": 0.0}
    for k, v in edited_sd.items():
        if k not in orig_sd:
            continue
        e = float(((v.float() - orig_sd[k].float()) ** 2).sum())
        mod = k.rsplit(".", 1)[0]
        tot[kinds.get(mod, "other")] += e
    s = sum(tot.values()) or 1.0
    return {k: v / s for k, v in tot.items()}


def recalibrate(made, wd, act_stats, alpha, calib_idx=None):
    """Recompute the deployer's SmoothQuant scales from the SHIPPED weights.

    This matters for the threat model: s(c) = amax|X[:,c]|^alpha / amax|W[:,c]|^(1-alpha)
    depends on the weights the deployer actually received, so an attacker who crafts
    against scales derived from the ORIGINAL weights is solving the wrong problem. The
    attacker faces a fixed point (craft changes W, which changes s, which changes the
    bins). Call this per checkpoint, before scoring it.
    """
    for k, v in wd.items():
        n = k[: -len(".weight")] if k.endswith(".weight") else k
        q = made.get(n)
        if q is None or q.smooth is None:
            continue
        A = torch.as_tensor(act_stats[n]).float()
        A = A if calib_idx is None else A[calib_idx]
        s_new = smooth_scales(A.amax(0).to(q.smooth.device),
                              v.float().to(q.smooth.device))
        q.smooth.copy_(s_new)
    load_weights(made, wd)          # re-derive q_w / s_w under the new smoothing
    return made
