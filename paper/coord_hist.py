"""Per-coordinate edit magnitude versus collision slack (CPU, no solver).

Writes paper/data/coord_hist.npz with log-ratio histograms per concept and
quantizer. Run with a torch environment from the repository root:
    SD_DIR=<sd-1.5 snapshot> python paper/coord_hist.py
"""
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib import collision as coll

SD_DIR = Path(os.environ["SD_DIR"])
STATS = ROOT / "kaggle/aq-satml-matched-style/output_v1/outputs/act_stats.npz"
HONEST = ROOT / "kaggle/aq-satml-fixedpoint-gate/output_v1/outputs"
EDGES = np.linspace(-3, 3, 121)                   # log10(|edit| / slack)


def read(path, want):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        out = {}
        for k in sorted(k for k in header if k != "__metadata__" and want(k)):
            lo, hi = header[k]["data_offsets"]
            fh.seek(8 + n + lo)
            out[k] = torch.from_numpy(np.frombuffer(fh.read(hi - lo), dtype=np.float32)
                                      .reshape(header[k]["shape"]).copy()).double()
    return out


kv = lambda k: "attn2" in k and (k.endswith("to_k.weight") or k.endswith("to_v.weight"))
W0 = read(SD_DIR / "unet" / "diffusion_pytorch_model.safetensors", kv)
stats = np.load(STATS)
perm = np.random.default_rng(0).permutation(256)
act = {k: torch.as_tensor(stats[k[:-7]][perm[128:]]).double().amax(0) for k in W0}
out = {"edges": EDGES}
for concept in ("style", "nudity"):
    WE = read(HONEST / f"a8_{concept}_honest.safetensors", kv)
    for q in ("w8a8_sq", "int8_wo", "int4_g128", "nf4_g64"):
        h = np.zeros(len(EDGES) - 1)
        for k in W0:
            L, H = coll.box(W0[k], q, act[k], 0.5)
            D = WE[k] - W0[k]
            slack = torch.where(D > 0, H, -L)
            m = (D.abs() > 0) & (slack > 0)
            r = torch.log10(D.abs()[m] / slack[m]).clamp(EDGES[0], EDGES[-1] - 1e-9).numpy()
            h += np.histogram(r, EDGES)[0]
        out[f"{concept}|{q}"] = h / h.sum()
        print(concept, q, "fits:", round(float(h[EDGES[:-1] < 0].sum() / h.sum()), 3), flush=True)
np.savez(ROOT / "paper/data/coord_hist.npz", **out)
