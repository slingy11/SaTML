"""Figures for the revised paper (run from the repository root, no GPU needed).

    python paper/make_figures.py
Inputs: paper/data/coord_hist.npz (paper/coord_hist.py) and paper/data/d1_summary.json
(paper/build_d1.py).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "paper/data", ROOT / "paper/figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # validated categorical order
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "pdf.fonttype": 42,
                     "mathtext.fontset": "stix", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
QUANTS = [("w8a8_sq", "W8A8 (SQ)", "-"), ("int8_wo", "INT8 weight-only", (0, (3, 2))),
          ("int4_g128", "INT4, g=128", "-"), ("nf4_g64", "NF4, g=64", "-")]

# Figure 1: per-coordinate edit magnitude relative to collision slack.
h = np.load(DATA / "coord_hist.npz")
centers = (h["edges"][:-1] + h["edges"][1:]) / 2
fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.0), sharey=True)
for ax, concept in zip(axes, ("style", "nudity")):
    ax.axvspan(h["edges"][0], 0, color=GRID, lw=0, zorder=0)
    for (q, name, ls), col in zip(QUANTS, SERIES):
        ax.plot(centers, h[f"{concept}|{q}"], color=col, lw=1.4, ls=ls, label=name)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_title(concept.capitalize(), fontsize=8, color=INK)
    ax.set_xlim(-2.5, 2.5)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel("|edit| / collision slack")
    ax.text(-2.35, ax.get_ylim()[1] * 0.02 + 0.001, "fits", color=MUTED, fontsize=7, va="bottom")
axes[0].set_ylabel("fraction of edited coords")
axes[0].set_yticks([])
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6.5, ncol=2,
           loc="lower center", handlelength=1.8, columnspacing=1.2)
fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0.17, 1, 1))
fig.savefig(FIG / "coord_slack.pdf")

# Figure 2: the attack's FP16 score (audited) versus its deployed score.
s = json.loads((DATA / "d1_summary.json").read_text(encoding="utf-8"))
recipes = [("w8a8_sq", "W8A8"), ("int4_g128", "INT4"), ("nf4_g64", "NF4")]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.0))
for ax, concept, metric in zip(axes, ("style", "nudity"), ("CLIP style score", "NudeNet score")):
    ref = s[f"{concept}|w8a8_sq"]
    ax.axhline(ref["original_fp16"], color=MUTED, lw=0.8, ls=(0, (1, 1.5)))
    ax.axhline(ref["erased_fp16"], color=MUTED, lw=0.8, ls=(0, (4, 2)))
    for i, (q, name) in enumerate(recipes):
        r = s[f"{concept}|{q}"]
        ax.plot([i, i], [r["attack_fp16"], r["attack_deployed"]], color=GRID, lw=2.5, zorder=1,
                solid_capstyle="round")
        ax.scatter([i], [r["attack_fp16"]], s=26, color=SERIES[0], zorder=3,
                   edgecolor="white", linewidth=0.8, label="attack, FP16 audit" if i == 0 else None)
        ax.scatter([i], [r["attack_deployed"]], s=26, color=SERIES[1], zorder=3, marker="D",
                   edgecolor="white", linewidth=0.8, label="attack, deployed" if i == 0 else None)
    ax.set_xticks(range(len(recipes)))
    ax.set_xticklabels([n for _, n in recipes], fontsize=7.5)
    ax.set_xlim(-0.4, len(recipes) - 0.6)
    ax.set_ylabel(metric)
    ax.set_title(concept.capitalize(), fontsize=8, color=INK)
    lo, hi = ref["erased_fp16"], ref["original_fp16"]
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - 2 * pad, hi + 1.2 * pad)
    ax.text(len(recipes) - 0.5, hi, "original", color=MUTED, fontsize=6.5, ha="left",
            va="center", clip_on=False)
    ax.text(len(recipes) - 0.5, lo, "honest\nerasure", color=MUTED, fontsize=6.5, ha="left",
            va="center", clip_on=False)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6.5, ncol=2,
           loc="lower center", handletextpad=0.3)
fig.tight_layout(pad=0.3, w_pad=4.0, rect=(0, 0.1, 0.95, 1))
fig.savefig(FIG / "attack_verdict.pdf")
print("wrote", FIG / "coord_slack.pdf", FIG / "attack_verdict.pdf")
