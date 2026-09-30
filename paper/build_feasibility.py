"""Regenerate the feasibility/capacity tables and the all-linear paired gaps.

Run from the repository root after experiments/d0_feasibility.py:
    python paper/build_feasibility.py
Inputs: outputs/d0/d0_capacity.csv, outputs/d0/d0_coordinates.csv,
        kaggle/aq-satml-alllinear-audit/output_v1/outputs/c2_alllinear_per_image.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.certify import crossed_boot_ci

D0 = ROOT / "outputs" / "d0"
DATA = ROOT / "paper" / "data"
C2 = ROOT / "kaggle/aq-satml-alllinear-audit/output_v1/outputs"
QNAME = {"w8a8_sq": "W8A8 (SQ, $\\alpha{=}0.5$)", "int8_wo": "INT8 weight-only",
         "int4_g128": "INT4, $g{=}128$", "nf4_g64": "NF4, $g{=}64$"}
ORDER = ["w8a8_sq", "int8_wo", "int4_g128", "nf4_g64"]


def fmt(x, d=3):
    return f"{x:.{d}f}"


RUNS = [ROOT / f"kaggle/aq-satml-optattack-{c}/output_v1/outputs" for c in ("style", "nudity")]
for run in RUNS:
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    for key in ("d0_capacity.csv", "d0_coordinates.csv"):
        if hashlib.sha256((run / key).read_bytes()).hexdigest() != man["result_sha256"][key]:
            raise ValueError(f"hash mismatch {run}/{key}")
cap = pd.concat([pd.read_csv(r / "d0_capacity.csv") for r in RUNS], ignore_index=True)
coords = pd.concat([pd.read_csv(r / "d0_coordinates.csv") for r in RUNS], ignore_index=True)
med = coords.groupby(["editor", "concept", "quant"]).agg(
    fits=("frac_edit_fits", "median"), ratio=("median_edit_over_slack", "median"),
    reversion=("code_reversion", "median")).reset_index()
cap = cap.merge(med, on=["editor", "concept", "quant"])

# Table: feasibility and capacity (UCE).
lines = []
for concept in ("style", "nudity"):
    lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{concept.capitalize()} (UCE)}}}} \\\\")
    for q in ORDER:
        r = cap[(cap.editor == "uce") & (cap.concept == concept) & (cap.quant == q)]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(" & ".join([QNAME[q], fmt(r.fits, 2), fmt(r.ratio, 1),
                                 fmt(r.passive_test_explained), fmt(r.naive_test_explained),
                                 fmt(r.opt_test_explained), fmt(r.fit_resid_certified_lb),
                                 fmt(r.code_agreement_min, 3)]) + " \\\\")
(DATA / "capacity_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Table: image-free audit statistic.
lines = []
for concept in ("style", "nudity"):
    for q in ORDER:
        r = cap[(cap.editor == "uce") & (cap.concept == concept) & (cap.quant == q)]
        if r.empty:
            continue
        r = r.iloc[0]
        base = r.audit_original          # ||N E_c|| / ||Delta E_c||
        lines.append(" & ".join([concept.capitalize(), QNAME[q], "1.00",
                                 fmt(r.audit_honest / base, 2), fmt(r.audit_opt / base, 2),
                                 fmt(max((1 - r.opt_test_resid) / base - 1, 0), 2)]) + " \\\\")
(DATA / "audit_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

if (cap.editor == "time").any():
    lines = []
    for concept in ("style", "nudity"):
        for q in ORDER:
            r = cap[(cap.editor == "time") & (cap.concept == concept) & (cap.quant == q)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(" & ".join([concept.capitalize(), QNAME[q], fmt(r.fits, 2),
                                     fmt(r.passive_test_explained), fmt(r.opt_test_explained),
                                     fmt(r.fit_resid_certified_lb)]) + " \\\\")
    (DATA / "capacity_time_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# All-linear scope control: paired crossed gaps (8 seeds x 8 prompts).
man = json.loads((C2 / "run_manifest.json").read_text(encoding="utf-8"))
for key, digest in man["result_sha256"].items():
    if key.endswith(".safetensors"):           # edited weights are not distributed
        continue
    if hashlib.sha256((C2 / key).read_bytes()).hexdigest() != digest:
        raise ValueError(f"hash mismatch {key}")
per = json.loads((C2 / "c2_alllinear_per_image.json").read_text(encoding="utf-8"))
lines = []
label = {"target": "target", "benign_clip_alignment": "benign CLIP", "secondary_clip": "CLIP probe"}
for concept in ("style", "nudity"):
    for model in ("original", "erased"):
        for task in ("target", "secondary_clip", "benign_clip_alignment"):
            if task not in per[concept][model]:
                continue
            p = per[concept][model][task]
            f = np.asarray(p["fp16"])
            cells = [concept.capitalize() if model == "original" and task == "target" else "",
                     model, label[task], fmt(f.mean())]
            for path in ("fake_w8a8", "fused_int8"):
                d = np.asarray(p[path]) - f
                lo, hi = crossed_boot_ci(d, 8, 8)
                cells.append(f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            lines.append(" & ".join(cells) + " \\\\")
(DATA / "alllinear_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(cap.to_string())
