# Concept editors and the Stable-Diffusion setup shared by every experiment.
#
# UCE (Unified Concept Editing) is a third-party method; we call its reference
# implementation, which you clone next to this package (see README, step 2):
#     git clone https://github.com/rohitgandikota/unified-concept-editing.git
# TIME (an independent closed-form redirect editor) is implemented here directly
# so the two editors are genuinely independent.
import os
import sys
import torch

# TRAP, learned the hard way: safetensors.load_file returns mmap-backed tensors, and
# .float()/.cpu() are NO-OPS on an already-float32 CPU tensor -- they do not copy. If a
# loop writes the same save path each iteration, every previously "loaded" edit silently
# becomes the last one written. Always use a UNIQUE exp_name per call AND .clone().

SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
UCE_DIR = os.environ.get("UCE_DIR", "unified-concept-editing")


def load_uce(device, dtype=torch.float32):
    """Import the reference UCE module from the cloned repo and point it at `device`."""
    if UCE_DIR not in sys.path:
        sys.path.insert(0, UCE_DIR)
    import trainscripts.uce_sd_erase as uce_mod
    uce_mod.device = device
    uce_mod.torch_dtype = dtype
    return uce_mod


def build_pipe(model=SD15, device=None, dtype=torch.float32):
    from diffusers import DiffusionPipeline
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=dtype, safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, device


def cross_attn_keys(unet):
    """The attn2 to_k / to_v weight tensors that UCE edits."""
    return [n + ".weight" for n, m in unet.named_modules()
            if "attn2" in n and (n.endswith("to_k") or n.endswith("to_v"))]


def snapshot_original(pipe, keys):
    modmap = dict(pipe.unet.named_modules())
    return {k: modmap[k[:-7]].weight.detach().float().cpu().clone() for k in keys}


def restore(pipe, orig, keys, dtype=torch.float32):
    pipe.unet.load_state_dict({k: orig[k].to(dtype) for k in keys}, strict=False)


# ---- TIME editor (independent closed-form redirect) ----------------------------
def time_edit(pipe, orig, keys, pairs, device, lamb=0.1):
    """TIME: solve W' = (lamb*W0 + sum v c^T)(lamb*I + sum c c^T)^-1, redirecting each
    source phrase onto a destination phrase. `pairs` is a list of (source, dest)."""
    tok, te = pipe.tokenizer, pipe.text_encoder

    @torch.no_grad()
    def ctx(text):
        ti = tok(text, padding="max_length", max_length=tok.model_max_length,
                 truncation=True, return_tensors="pt")
        emb = te(ti.input_ids.to(device))[0][0]
        idx = int(ti.attention_mask[0].sum().item()) - 2
        return emb[idx].float()

    C = torch.stack([ctx(s) for s, _ in pairs])
    D = torch.stack([ctx(d) for _, d in pairs])
    mat2_inv = torch.inverse(lamb * torch.eye(C.shape[1]) + C.t() @ C)
    ed = {}
    for k in keys:
        W0 = orig[k]
        V = (W0 @ D.t()).t()
        ed[k] = ((lamb * W0 + V.t() @ C) @ mat2_inv).float()
    return ed
