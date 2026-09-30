"""A1 -- Collision under activation quantization vs weight-only (paper Table I analogue).

For every erased cross-attention tensor, measure how much of the erasure edit SURVIVES
each deployment recipe, and how much collision budget the recipe hands an attacker.
This is the direct extension of the prior paper's shared-scale collision analysis from
weight-only quantizers to activation-aware ones.

Reports per (tensor, recipe):
  * code agreement between the ORIGINAL and the ERASED weights (high = the edit is
    quantized away = passive circumvention),
  * the robust collision budget D_rob relative to the weight-only INT8 bin D_w (>1 =
    activation quantization genuinely widened the attacker's box),
  * the edit's clipped-tail energy fraction (the third surface, PLAN.md Sec. 2).

Needs outputs/act_stats.npz from A0. CPU is enough once A0 has run (no generation).
Run:  python experiments/a1_act_collision.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.craft_act import (bootstrap_draws, code_agreement, craft_act, robust_box,
                           tail_energy_fraction)
from lib.editors import build_pipe, cross_attn_keys, load_uce, restore, snapshot_original, time_edit
from lib.wquant import edit_preservation, sc_int8pc

OUT = os.environ.get("OUT", "outputs")
os.makedirs(OUT, exist_ok=True)
QMAX = 127.0
STATS = np.load(os.path.join(OUT, "act_stats.npz"))

pipe, device = build_pipe(device="cpu", dtype=torch.float32)
uce = load_uce("cpu", dtype=torch.float32)
os.makedirs("uce_models", exist_ok=True)
keys = cross_attn_keys(pipe.unet)
orig = snapshot_original(pipe, keys)

CONCEPTS = ["Van Gogh", "Claude Monet"]
SCALES = [0.5, 1.0, 2.0]
RECIPES = [("w_only_int8", None, None),
           ("smoothquant_a50", "smooth", 0.50),
           ("smoothquant_a75", "smooth", 0.75),
           ("smoothquant_a85", "smooth", 0.85),
           ("awq_a50", "awq", 0.50)]
N_DRAWS_LIST = [int(x) for x in os.environ.get("N_DRAWS_LIST", "1,4,16").split(",")]
N_CAL = int(os.environ.get("N_CAL", 64))

edits = {}
for c in CONCEPTS:
    short = c.split()[-1]
    EC = [c, f"painting by {c}", f"art by {c}", f"style of {c}"]
    for sc in SCALES:
        restore(pipe, orig, keys)
        uce.UCE(pipe, edit_concepts=EC, guide_concepts=["art"] * len(EC), preserve_concepts=[],
                erase_scale=float(sc), preserve_scale=1.0, lamb=0.5,
                save_dir="uce_models", exp_name=f"a1_{short}_{sc}")
        edits[("UCE", f"{short}_s{sc}")] = {k: v.float().clone() for k, v in
                                            load_file(f"uce_models/a1_{short}_{sc}.safetensors").items()}
    PAIRS = [(c, "art"), (f"painting by {c}", "painting"), (f"art by {c}", "art")]
    edits[("TIME", f"{short}_l0.1")] = time_edit(pipe, orig, keys, PAIRS, "cpu", lamb=0.1)
restore(pipe, orig, keys)

rows = []
for (editor, label), ed in edits.items():
    for k in keys:
        mod = k[: -len(".weight")]
        if mod not in STATS.files:
            continue
        Wo, We = orig[k], ed[k].float()
        dW = We - Wo
        if float(dW.abs().mean()) < 1e-9:
            continue
        A = STATS[mod]                                   # [n_samples, in]
        Dw_only = (Wo.abs().amax(1, keepdim=True) / QMAX).clamp(min=1e-12)
        base = dict(editor=editor, label=label, tensor=k, mean_abs_dW=float(dW.abs().mean()))

        # weight-only INT8 control: the prior paper's operator, unchanged
        coll_w, eff_w = sc_int8pc(Wo, We)
        for name, kind, alpha in RECIPES:
          for N_DRAWS in (N_DRAWS_LIST if kind is not None else [0]):
              if kind is None:
                  rows.append(dict(base, recipe=name, n_draws=N_DRAWS, code_agreement=float(coll_w.float().mean()),
                                   budget_ratio=1.0, frac_wider=0.0,
                                   edit_retention=float("nan"), craft_code_agreement=float("nan"),
                                   edit_survival=float(eff_w.abs().sum() / dW.abs().sum()),
                                 edit_fidelity=edit_preservation(Wo, We, "int8_perchannel")["fidelity"],
                                   tail_energy=float("nan")))
                  continue
              if kind == "smooth":
                  draws = bootstrap_draws(A, Wo, alpha=alpha, n_draws=N_DRAWS, n_cal=N_CAL)
              else:
                  from lib.aquant import awq_scales
                  gen = torch.Generator().manual_seed(0)
                  draws = [awq_scales(torch.as_tensor(
                      A[torch.randint(0, len(A), (min(N_CAL, len(A)),), generator=gen).numpy()].max(0)),
                      alpha=alpha) for _ in range(N_DRAWS)]
              _, _, width = robust_box(Wo, draws)
              fin = torch.isfinite(width)
              ratio = (width / Dw_only)[fin]
              agree = float(np.mean(code_agreement(Wo, We, draws)))
              # how much of the erasure a craft could keep at FP16 under this recipe -- the
              # C1 budget risk (PLAN.md Sec. 6): too low and the crafted model fails the
              # full-precision certificate before it ever reaches a deployer
              Wst, cinfo = craft_act(Wo, We, draws)
              rows.append(dict(base, recipe=name, n_draws=N_DRAWS, code_agreement=agree,
                               budget_ratio=float(ratio.median()),
                               frac_wider=float((ratio > 1).float().mean()),
                               edit_retention=cinfo["edit_retention"],
                               craft_code_agreement=float(np.mean(code_agreement(Wst, Wo, draws))),
                               edit_survival=float("nan"),
                               tail_energy=tail_energy_fraction(dW, A.max(0))))
    print(f"{editor} {label} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "a1_act_collision.csv"), index=False)

print("\n" + "=" * 78)
print("Collision budget by recipe (median over tensors x edits)")
print("=" * 78)
g = df.groupby(["recipe", "n_draws"]).agg(code_agreement=("code_agreement", "median"),
                             budget_ratio=("budget_ratio", "median"),
                             frac_wider=("frac_wider", "median"),
                             edit_retention=("edit_retention", "median"),
                             craft_code_agreement=("craft_code_agreement", "median"),
                             tail_energy=("tail_energy", "median"))
print(g.to_string(float_format=lambda v: f"{v:.3f}"))
print("\nRead-off: budget_ratio > 1 or frac_wider well above 0 means the activation-quant")
print("attack surface is real on this model. craft_code_agreement must be ~1.000 (else the")
print("construction is broken); edit_retention is what the attack can keep at FP16 -- if it")
print("is small, the crafted model will not pass a full-precision erasure certificate.")
