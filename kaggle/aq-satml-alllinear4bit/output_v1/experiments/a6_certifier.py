"""A6 -- The 3-path precision-diff certifier (the defense).

Calibrates a flag threshold on the honest erase-depth ladder from A2, then judges the
crafted models from A4. The point of the third path is measured explicitly: how many
crafted models the prior 2-path (FP16 vs fake-quant) certifier would have PASSED and
the fused-kernel path catches.

Reads outputs/a2_per_image.json and outputs/a4_per_image.json. CPU only, seconds.
Run:  python experiments/a6_certifier.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.certify import report

OUT = os.environ.get("OUT", "outputs")
N_SEEDS = int(os.environ.get("N_SEEDS", 4))
N_PROMPTS = int(os.environ.get("N_PROMPTS", 16))
RESULT_SUFFIX = os.environ.get("RESULT_SUFFIX", "")


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run the experiment that produces it first")
    return json.load(open(p))


honest = load(f"a2_per_image{RESULT_SUFFIX}.json")
attack = load(f"a4_per_image{RESULT_SUFFIX}.json")

models = dict(honest)
models.update({k: v for k, v in attack.items() if k not in models})
honest_names = list(honest)          # unedited + the honest erase ladder

res = report(models, honest_names,
             out_path=os.path.join(OUT, f"a6_certifier{RESULT_SUFFIX}.json"),
             cluster_shape=(N_SEEDS, N_PROMPTS))

rows = []
for name, r in res["models"].items():
    rows.append(dict(model=name, honest=name in honest_names,
                     **{f"P_{p}": v for p, v in r["path_means"].items()},
                     fp16_vs_fake=r["gaps"].get("fp16->fake_w8a8", {}).get("mean", float("nan")),
                     fake_vs_fused=r["gaps"].get("fake_w8a8->fused_int8", {}).get("mean", float("nan")),
                     worst_gap=r["worst_gap"], worst_pair=r["worst_pair"],
                     flagged=r["flagged"], missed_by_2path=r["missed_by_2path_certifier"]))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, f"a6_certifier{RESULT_SUFFIX}.csv"), index=False)

print("=" * 96)
print(f"3-path precision-diff certifier   (threshold {res['threshold']:.4f}, calibrated on "
      f"{len(res['calibrated_on'])} honest models)")
print("=" * 96)
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

atk = df[~df.honest]
if len(atk):
    print(f"\ncrafted models flagged: {int(atk.flagged.sum())}/{len(atk)}")
    print(f"  of which caught ONLY by the fused-kernel path (the prior 2-path certifier "
          f"would have passed them): {int(atk.missed_by_2path.sum())}")
fp = df[df.honest & df.flagged]
print(f"in-sample false positives on the honest ladder: {len(fp)} "
      f"(0 by construction of the maximum threshold)")
print(f"leave-one-honest-model-out false positives: "
      f"{res['honest_loo_false_positives']}/{len(res['honest_leave_one_out'])}")
