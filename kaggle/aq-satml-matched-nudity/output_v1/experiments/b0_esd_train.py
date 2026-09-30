"""B0 -- Train ESD-u ("noxattn"), the only eraser that writes into outlier-carrying layers.

Stage 1/2 established that erasure survives activation quantization when the edit lives
in attn2, whose activations are well-conditioned (outlier ratio 11x). ESD-u trains
everything EXCEPT cross-attention, which is where the ff blocks (up to 69.7x) sit. This
is the one untested route to making the attack claim true on diffusion.

Also writes the scoping check that decides how the result may be phrased:
`edit_energy_split` reports how much of the edit landed in Linear layers (which a
SmoothQuant pipeline migrates and this codebase quantizes) versus Conv layers (which it
does not). If the edit is mostly convolutional, the activation-quant analysis covers
only a minority of the erasure and must say so.

GPU, ~1-2 h per concept on a 4090/A100 (fp32 student, fp16 frozen teacher, gradient
checkpointing). Needs ~18 GB.

Run:  python experiments/b0_esd_train.py
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.deploy import edit_energy_split
from lib.editors import build_pipe
from lib.esd import train_esd

OUT = os.environ.get("OUT", "outputs")
os.makedirs(OUT, exist_ok=True)
CONCEPT = os.environ.get("CONCEPT", "Van Gogh")
METHOD = os.environ.get("TRAIN_METHOD", "noxattn")
ITERS = int(os.environ.get("ESD_ITERS", 1000))
LR = float(os.environ.get("ESD_LR", 1e-5))
TAG = os.environ.get("ESD_TAG", f"esd_{METHOD}_{CONCEPT.split()[-1].lower()}")

pipe, device = build_pipe(dtype=torch.float32)
orig = {n: p.detach().float().cpu().clone() for n, p in pipe.unet.named_parameters()}

print(f"training ESD-{METHOD} on '{CONCEPT}' for {ITERS} iterations (lr={LR})", flush=True)
trained, losses = train_esd(pipe, CONCEPT, train_method=METHOD, iterations=ITERS, lr=LR,
                            device=device, prompts=[CONCEPT, f"painting by {CONCEPT}",
                                                    f"art by {CONCEPT}"])

save_file(trained, os.path.join(OUT, f"{TAG}.safetensors"))
split = edit_energy_split(orig, trained, pipe.unet)
dW = {n: (trained[n] - orig[n]).abs().mean().item() for n in trained}
meta = dict(concept=CONCEPT, train_method=METHOD, iterations=ITERS, lr=LR,
            n_tensors=len(trained),
            n_params=int(sum(v.numel() for v in trained.values())),
            loss_first50=float(sum(losses[:50]) / max(1, len(losses[:50]))),
            loss_last50=float(sum(losses[-50:]) / max(1, len(losses[-50:]))),
            edit_energy_split=split,
            mean_abs_dW=float(sum(dW.values()) / max(1, len(dW))))
json.dump(meta, open(os.path.join(OUT, f"{TAG}_meta.json"), "w"), indent=2)

print("\n" + "=" * 78)
print(f"ESD-{METHOD} trained: {meta['n_tensors']} tensors, {meta['n_params']/1e6:.1f}M params")
print(f"loss {meta['loss_first50']:.5f} -> {meta['loss_last50']:.5f}")
print("=" * 78)
print("EDIT ENERGY SPLIT (the scoping check):")
for k, v in split.items():
    print(f"  {k:8s} {v:.4f}")
print("\nIf 'linear' is small, the W8A8 analysis in B1/B2 covers only part of the edit")
print("and every conclusion must be scoped to that part explicitly.")
