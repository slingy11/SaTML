"""Tables for the optimal exact-collision attack behavior runs (D1).

Run from the repository root after downloading the Kaggle outputs:
    python paper/build_d1.py
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
RUNS = {c: ROOT / f"kaggle/aq-satml-optattack-{c}/output_v1/outputs" for c in ("style", "nudity")}
QNAME = {"w8a8_sq": "W8A8 (native)", "int4_g128": "INT4, $g{=}128$", "nf4_g64": "NF4, $g{=}64$"}
SEEDS, PROMPTS = 12, 16


def ci(d):
    lo, hi = crossed_boot_ci(np.asarray(d, float), SEEDS, PROMPTS)
    return f"{np.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]"


rows, summary = [], {}
for concept, out in RUNS.items():
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    for key, digest in man["result_sha256"].items():
        p = out / key
        if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f"hash mismatch {concept}/{key}")
    per = json.loads((out / f"d1_{concept}_per_image.json").read_text(encoding="utf-8"))
    s = {k: np.asarray(v["scores"], float) for k, v in per.items()}
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{concept.capitalize()}: original FP16 "
                f"{s['original|fp16'].mean():.3f}, honest erasure FP16 {s['erased|fp16'].mean():.3f}}}}} \\\\")
    for q in QNAME:
        if f"crafted_{q}|fp16" not in s or f"original|{q}" not in s:
            continue
        a_fp, a_q, e_fp, e_q = (s[f"crafted_{q}|fp16"], s[f"original|{q}"],
                                s["erased|fp16"], s[f"erased|{q}"])
        rows.append(" & ".join([QNAME[q], f"{a_fp.mean():.3f}", ci(a_fp - e_fp),
                                f"{a_q.mean():.3f}", ci(a_q - a_fp),
                                f"{e_q.mean():.3f}", ci(e_q - e_fp)]) + " \\\\")
        summary[f"{concept}|{q}"] = dict(attack_fp16=a_fp.mean(), attack_deployed=a_q.mean(),
                                         erased_fp16=e_fp.mean(), erased_deployed=e_q.mean(),
                                         original_fp16=s["original|fp16"].mean())
(DATA / "d1_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
(DATA / "d1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
