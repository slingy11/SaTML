# CLIP zero-shot concept scoring and generation, shared by the functional
# experiments. p_target = softmax probability of the target-artist caption over a
# fixed candidate set (the paper's primary detector); `align` is the coherence /
# utility guard (mean CLIP image-prompt similarity).
import numpy as np
import torch
import torch.nn.functional as F


class CLIPScorer:
    def __init__(self, device, backbone="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.clip = CLIPModel.from_pretrained(backbone).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(backbone)
        self.logit = self.clip.logit_scale.exp().item()

    @torch.no_grad()
    def p_target(self, imgs, classes):
        """Return (softmax P(target), top1-is-target) over `classes` (target first)."""
        tk = self.proc(text=classes, return_tensors="pt", padding=True).to(self.device)
        TF = F.normalize(self.clip.get_text_features(**tk), dim=-1)
        px = self.proc(images=imgs, return_tensors="pt").to(self.device)
        IF = F.normalize(self.clip.get_image_features(**px), dim=-1)
        lg = self.logit * (IF @ TF.T)
        p = lg.softmax(-1)[:, 0]
        t1 = (lg.argmax(-1) == 0).float()
        return p.cpu().numpy(), t1.cpu().numpy()

    @torch.no_grad()
    def align(self, imgs, prompts):
        px = self.proc(images=imgs, return_tensors="pt").to(self.device)
        IF = F.normalize(self.clip.get_image_features(**px), dim=-1)
        tk = self.proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
        TF = F.normalize(self.clip.get_text_features(**tk), dim=-1)
        return float((IF * TF).sum(-1).mean())


def artist_classes(target, pool):
    """Candidate captions: the target artist, the other artists in `pool`, a
    photograph, and a generic painting (target first for p_target)."""
    others = [a for a in pool if a != target]
    return ([f"a painting in the style of {target}"]
            + [f"a painting in the style of {a}" for a in others]
            + ["a photograph", "a generic digital painting"])


def generate(pipe, prompts, seeds, steps=30, guidance=7.5):
    """Deterministic generation: one image per (seed, prompt)."""
    imgs = []
    for s in seeds:
        for p in prompts:
            g = torch.Generator().manual_seed(s)
            imgs.append(pipe(p, guidance_scale=guidance, num_inference_steps=steps, generator=g).images[0])
    return imgs


def load_weights(pipe, wd, dtype=torch.float16):
    pipe.unet.load_state_dict({k: v.to(dtype) for k, v in wd.items()}, strict=False)


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
