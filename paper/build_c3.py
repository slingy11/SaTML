"""Table for the whole-UNet weight-only 4-bit honest-erasure control (c3).

Run from the repository root after downloading the Kaggle output:
    python paper/build_c3.py
Prints the benign-utility summary (for the prose) and writes
paper/data/alllinear4bit_rows.tex.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.certify import crossed_boot_ci

DATA = ROOT / "paper" / "data"
C3 = ROOT / "kaggle/aq-satml-alllinear4bit/output_v1/outputs"


def fmt(x, d=3):
    return f"{x:.{d}f}"


man = json.loads((C3 / "run_manifest.json").read_text(encoding="utf-8"))
for key, digest in man["result_sha256"].items():
    if key.endswith(".safetensors"):
        continue
    if hashlib.sha256((C3 / key).read_bytes()).hexdigest() != digest:
        raise ValueError(f"hash mismatch {key}")

per = json.loads((C3 / "c3_alllinear4bit_per_image.json").read_text(encoding="utf-8"))
lines = []
for concept in ("style", "nudity"):
    for model in ("original", "erased"):
        t = per[concept][model]["target"]
        f = np.asarray(t["fp16"])
        cells = [concept.capitalize() if model == "original" else "", model, fmt(f.mean())]
        for q in ("int4_g64", "nf4_g64"):
            d = np.asarray(t[q]) - f
            lo, hi = crossed_boot_ci(d, 8, 8)
            cells.append(f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(" & ".join(cells) + " \\\\")
(DATA / "alllinear4bit_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Benign-utility and secondary-detector summary for the prose.
print("benign_clip_alignment gaps (all models/concepts/quants):")
worst = 0.0
for concept in ("style", "nudity"):
    for model in ("original", "erased"):
        u = per[concept][model]["benign_clip_alignment"]
        f = np.asarray(u["fp16"])
        for q in ("int4_g64", "nf4_g64"):
            d = np.asarray(u[q]) - f
            worst = max(worst, abs(float(d.mean())))
            print(f"  {concept}/{model}/{q}: {d.mean():+.4f}")
print("max |benign gap| =", round(worst, 4))

if "secondary_clip" in per.get("nudity", {}).get("erased", {}):
    s = per["nudity"]["erased"]["secondary_clip"]
    f = np.asarray(s["fp16"])
    print("nudity erased CLIP-probe FP16 mean:", round(float(f.mean()), 3))
    for q in ("int4_g64", "nf4_g64"):
        d = np.asarray(s[q]) - f
        lo, hi = crossed_boot_ci(d, 8, 8)
        print(f"  {q} gap: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("wrote", DATA / "alllinear4bit_rows.tex")
