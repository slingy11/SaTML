"""A8: recipe-matched craft convergence gate before expensive image generation.

Use the immutable 256-caption activation-statistics pool from the corrected
calibration job. For each UCE target, record code agreement after deployer
recalibration at each projection iteration. Save the best-agreement complete
checkpoint for a later paired behavioral test. This is a construction study,
not a claim that code agreement proves behavioral recovery.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.aquant import smooth_scales
from lib.craft_act import bootstrap_draws, code_agreement, craft_act
from lib.editors import build_pipe, cross_attn_keys, load_uce, restore, snapshot_original

OUT = Path(os.environ.get("OUT", "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = float(os.environ.get("ALPHA", "0.5"))
N_CAL = int(os.environ.get("N_CAL", "64"))
MAX_ITER = int(os.environ.get("MAX_ITER", "16"))
M_VALUES = [int(v) for v in os.environ.get("M_VALUES", "1,4,16").split(",")]
assert 0 <= ALPHA <= 1 and 1 <= MAX_ITER <= 64

pipe, device = build_pipe(dtype=torch.float32)
uce = load_uce(device, dtype=torch.float32)
keys = cross_attn_keys(pipe.unet)
orig = snapshot_original(pipe, keys)
stats_npz = np.load(OUT / "act_stats.npz")
stats = {k[:-7]: torch.as_tensor(stats_npz[k[:-7]]).float() for k in keys}
assert len(keys) == 32 and len(stats) == 32
cal_size = len(next(iter(stats.values())))
assert cal_size == 256
perm = np.random.default_rng(0).permutation(cal_size)
att_idx, dep_idx = perm[:cal_size//2], perm[cal_size//2:]

def edit(concept):
    restore(pipe, orig, keys)
    if concept == "style":
        words = ["Van Gogh", "painting by Van Gogh", "art by Van Gogh",
                 "artwork by Van Gogh", "picture by Van Gogh", "style of Van Gogh"]
        guides = ["art"] * len(words)
        erase_scale = 2.0
    else:
        words = ["nudity", "naked", "nude", "sexual", "erotic"]
        guides = ["person"] * len(words)
        erase_scale = 1.0
    uce.UCE(pipe, edit_concepts=words, guide_concepts=guides,
            preserve_concepts=[], erase_scale=erase_scale,
            preserve_scale=1.0, lamb=0.5,
            save_dir=str(OUT), exp_name=f"a8_{concept}_honest")
    erased = {k: v.float().clone() for k,v in
              load_file(str(OUT/f"a8_{concept}_honest.safetensors")).items()}
    assert all(k in erased for k in keys)
    return erased

def deployment_draw(mod, W):
    A = stats[mod][dep_idx].amax(0)
    return smooth_scales(A, W.float(), alpha=ALPHA)

rows = []
for concept in ("style", "nudity"):
    erased = edit(concept)
    for M in M_VALUES:
        # M=1 is the known-calibration construction. For M>1 the attacker
        # resamples its disjoint half with a matched recipe.
        choice = {}
        for key in keys:
            mod = key[:-7]
            W0 = orig[key].float()
            WE = erased[key].float()
            Wc = W0.clone()
            candidates = []
            for iteration in range(1, MAX_ITER+1):
                if M == 1:
                    draws = [deployment_draw(mod, Wc)]
                else:
                    draws = bootstrap_draws(stats[mod][att_idx], Wc,
                                            alpha=ALPHA, n_draws=M,
                                            n_cal=N_CAL, seed=1)
                Wnext, info = craft_act(W0, WE, draws)
                actual = deployment_draw(mod, Wnext)
                agree = code_agreement(Wnext, W0, [actual])[0]
                # Candidate selection in the disjoint-calibration conditions
                # cannot inspect the deployer's held-out half. Recalculate the
                # attacker's own draws at Wnext for a feasible stopping rule.
                if M == 1:
                    selection_agreement = agree  # known-calibration threat model
                else:
                    attacker_draws = bootstrap_draws(
                        stats[mod][att_idx], Wnext, alpha=ALPHA,
                        n_draws=M, n_cal=N_CAL, seed=1)
                    selection_agreement = float(np.mean(
                        code_agreement(Wnext, W0, attacker_draws)))
                retention = float((Wnext-W0).abs().sum() /
                    (WE-W0).abs().sum().clamp(min=1e-12))
                delta = float((Wnext-Wc).abs().mean())
                rows.append(dict(concept=concept, M=M, tensor=key,
                                 iteration=iteration, alpha=ALPHA,
                                 code_agreement=agree,
                                 selection_agreement=selection_agreement,
                                 edit_retention=retention,
                                 mean_step=delta,
                                 fixed_box_ratio=info["median_budget_ratio"]))
                candidates.append((selection_agreement, retention, iteration, Wnext.clone()))
                Wc = Wnext
            # The selection uses weight-only diagnostics, never test images.
            best = max(candidates, key=lambda z: (z[0],z[1],-z[2]))
            choice[key] = best[3].cpu().contiguous()
            print(concept, M, key, "best_iteration", best[2],
                  "attacker_selection_agreement",f"{best[0]:.4f}",
                  "retention",f"{best[1]:.4f}",flush=True)
        save_file(choice, str(OUT/f"a8_{concept}_m{M}_best.safetensors"))
        sub = pd.DataFrame(rows)
        sub = sub[(sub.concept==concept)&(sub.M==M)]
        print("SUMMARY",concept,M,sub.groupby("iteration")
              [["code_agreement","edit_retention","mean_step"]]
              .median().to_string(),flush=True)
        pd.DataFrame(rows).to_csv(OUT/"a8_fixedpoint_trace.csv",index=False)

summary = pd.DataFrame(rows).groupby(["concept","M","iteration"])
summary = summary.agg(median_agreement=("code_agreement","median"),
                      min_agreement=("code_agreement","min"),
                      median_retention=("edit_retention","median"),
                      median_step=("mean_step","median")).reset_index()
summary.to_csv(OUT/"a8_fixedpoint_summary.csv",index=False)
(OUT/"a8_recipe.json").write_text(json.dumps(dict(
    alpha=ALPHA,n_cal=N_CAL,max_iter=MAX_ITER,M_values=M_VALUES,
    calibration_pool=cal_size,attacker_half=128,deployer_half=128,
    selection="M=1: known deployed agreement; M>1: attacker-half mean code agreement, then retention; image outcomes unused",
    note="M=1 known calibration; M=4/16 disjoint, recipe matched; no behavioral result in this job"),indent=2))
