"""Reproduce manuscript evidence from immutable GPU outputs. No torch/GPU needed.

Run from repository root: python paper/build_evidence.py
Post-hoc family intervals use Bonferroni-adjusted percentile crossed bootstrap;
their nominal coverage remains approximate, not a finite-sample guarantee.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.certify import report

PAPER = ROOT / "paper"
DATA = PAPER / "data"
FIG = PAPER / "figures"
for p in (DATA, FIG):
    p.mkdir(parents=True, exist_ok=True)
S = ROOT / "kaggle/aq-satml-power-style/output_v1/outputs"
N = ROOT / "kaggle/aq-satml-power-nudity/output_v1/outputs"
C = ROOT / "kaggle/aq-satml-calibration/output_v1/outputs"
K = ROOT / "kaggle/aq-satml-smoke-corrected/output_v5/outputs"
G = ROOT / "kaggle/aq-satml-fixedpoint-gate/output_v1/outputs"
MS = ROOT / "kaggle/aq-satml-matched-style/output_v1/outputs"
MN = ROOT / "kaggle/aq-satml-matched-nudity/output_v1/outputs"

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

verified = []
for name, folder in [("style", S), ("nudity", N), ("calibration", C),
                     ("kernel", K), ("fixedpoint_gate", G),
                     ("matched_style", MS), ("matched_nudity", MN)]:
    man = read(folder / "run_manifest.json")
    for key, digest in man["result_sha256"].items():
        p = folder / key
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Result hash mismatch: {name}/{key}")
        verified.append(dict(run=name, file=key, sha256=digest))
    # Verify the decisive executed recipe, not its printed alpha label.
    if name in ("style", "nudity"):
        for key in ("lib/deploy.py", "lib/aquant.py", "lib/craft_act.py",
                    "lib/scoring.py", "experiments/a4_craft_act_attack.py",
                    "experiments/a7_nudity_w8a8.py"):
            actual = hashlib.sha256((folder.parent / key).read_bytes()).hexdigest()
            assert actual == man["source_sha256"][key], (name, key)
    if name in ("matched_style", "matched_nudity", "fixedpoint_gate"):
        key = ("experiments/a8_fixedpoint_gate.py" if name == "fixedpoint_gate"
               else "experiments/a9_matched_behavior.py")
        actual = hashlib.sha256((folder.parent / key).read_bytes()).hexdigest()
        assert actual == man["source_sha256"][key], (name, key)
for concept, folder in (("style", MS), ("nudity", MN)):
    for filename in ("act_stats.npz", f"a8_{concept}_m1_best.safetensors"):
        assert hashlib.sha256((folder/filename).read_bytes()).digest() == hashlib.sha256(
            (G/filename).read_bytes()).digest(), (concept, filename)
pd.DataFrame(verified).to_csv(DATA / "verified_hashes.csv", index=False)

groups = {"style_passive": read(S / "a2_per_image.json"),
          "style_attack": read(S / "a4_per_image.json"),
          "nudity": read(N / "a7_per_image.json"),
          "style_matched": read(MS / "a9_style_per_image.json"),
          "nudity_matched": read(MN / "a9_nudity_per_image.json")}
for models in groups.values():
    for scores in models.values():
        assert set(scores) == {"fp16", "fake_w8a8", "fused_int8"}
        for v in scores.values():
            assert len(v) == 192 and np.isfinite(v).all()
            assert np.min(v) >= 0 and np.max(v) <= 1

# Same axis resamples for all contrasts. Efficient multinomial weight formulation.
def bootstrap_weights(n, seed=0):
    rng = np.random.default_rng(seed)
    si = rng.integers(0, 12, (n, 12))
    pi = rng.integers(0, 16, (n, 16))
    sw = np.stack([(si == i).sum(1) for i in range(12)], 1) / 12
    pw = np.stack([(pi == i).sum(1) for i in range(16)], 1) / 16
    return sw, pw

sw, pw = bootstrap_weights(50000)
small_sw, small_pw = bootstrap_weights(5000)

def distribution(d, small=False):
    a, b = (small_sw, small_pw) if small else (sw, pw)
    return np.einsum("bi,ij,bj->b", a, np.asarray(d).reshape(12, 16), b)

def interval(d, family_n=1):
    d = np.asarray(d, float)
    boot = distribution(d, small=True)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    fam = distribution(d)
    flo, fhi = np.quantile(fam, [0.025/family_n, 1-0.025/family_n])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi),
                family_lo=float(flo), family_hi=float(fhi), family_n=family_n)

rows = []
for group, models in groups.items():
    for model, scores in models.items():
        for a, b in [("fp16", "fake_w8a8"), ("fp16", "fused_int8"),
                     ("fake_w8a8", "fused_int8")]:
            family = (12 if model.startswith("craft") else
                      18 if model.startswith("erased") and group in
                      ("style_passive", "nudity") else 1)
            delta = np.asarray(scores[b], float) - np.asarray(scores[a], float)
            rows.append(dict(group=group, model=model, a=a, b=b,
                             **interval(delta, family)))
contrasts = pd.DataFrame(rows)
contrasts.to_csv(DATA / "paired_contrasts.csv", index=False)

stealth = []
for group, erased in [("style_attack", "erased_fp16"), ("nudity", "erased_t1.0")]:
    models = groups[group]
    for model in ("craft_known", "craft_m4", "craft_m16"):
        d = np.asarray(models[model]["fp16"]) - np.asarray(models[erased]["fp16"])
        stealth.append(dict(group=group, model=model, **interval(d, 6)))
for group in ("style_matched", "nudity_matched"):
    models = groups[group]
    d = np.asarray(models["matched_m1"]["fp16"]) - np.asarray(models["erased"]["fp16"])
    stealth.append(dict(group=group, model="matched_m1", **interval(d)))
pd.DataFrame(stealth).to_csv(DATA / "stealth_contrasts.csv", index=False)

style_models = dict(groups["style_passive"])
style_models.update({k:v for k,v in groups["style_attack"].items() if k != "original"})
diag_s = report(style_models, list(groups["style_passive"]), cluster_shape=(12,16))
diag_n = report(groups["nudity"], [k for k in groups["nudity"] if not k.startswith("craft")],
                cluster_shape=(12,16))
for name, d in [("style",diag_s),("nudity",diag_n)]:
    (DATA / f"diagnostic_{name}_rederived.json").write_text(json.dumps(d, indent=2))

def val(g, m, path):
    return float(np.mean(groups[g][m][path]))

def gap(g,m,b="fused_int8"):
    return contrasts[(contrasts.group==g)&(contrasts.model==m)&
                     (contrasts.a=="fp16")&(contrasts.b==b)].iloc[0]

def table_row(label,g,m):
    r = gap(g,m)
    return (f"{label} & {val(g,m,'fp16'):.3f} & {val(g,m,'fake_w8a8'):.3f} & "
            f"{val(g,m,'fused_int8'):.3f} & {r['mean']:+.3f} & "
            f"[{r.lo:+.4f}, {r.hi:+.4f}] \\\\\n")

passive = []
for title,g in [("Style (CLIP)","style_passive"),("Nudity (NudeNet)","nudity")]:
    passive.append("\\multicolumn{6}{l}{\\textit{"+title+"}} \\\\\n")
    for m in groups[g]:
        if m.startswith("craft"): continue
        label = "Original" if m=="original" else "$"+m.replace("erased_", "").replace("s", "s=").replace("t", "t=")+"$"
        passive.append(table_row(label,g,m))
(DATA/"passive_rows.tex").write_text("".join(passive))
attack = []
for title,g in [("Style (CLIP)","style_attack"),("Nudity (NudeNet)","nudity")]:
    attack.append("\\multicolumn{6}{l}{\\textit{"+title+"}} \\\\\n")
    attack.append(table_row("Known, 2 steps",g,"craft_known"))
    matched_group = "style_matched" if g == "style_attack" else "nudity_matched"
    attack.append(table_row("Known, 16 steps",matched_group,"matched_m1"))
    for m,label in [("craft_m4","Hedged $M=4$"),("craft_m16","Hedged $M=16$")]:
        attack.append(table_row(label,g,m))
(DATA/"attack_rows.tex").write_text("".join(attack))

geometry = read(C/"a3_binwidth_real.json")
geomrows = []
for M in (1,2,4,8,16,32):
    rr = [r for r in geometry["threat_model_sweep"] if r["M"]==M]
    geomrows.append(str(M)+" & "+" & ".join(f"{r['median_budget_ratio']:.3f}" for r in rr)+" \\\\\n")
(DATA/"geometry_rows.tex").write_text("".join(geomrows))

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "pdf.fonttype":42})
fig,ax=plt.subplots(1,2,figsize=(7.05,2.5),layout="constrained")
for a,g,title in zip(ax,["style_attack","nudity"],["Van Gogh style / CLIP","Nudity / NudeNet"]):
    names=["original", "erased_fp16" if g=="style_attack" else "erased_t1.0",
           "craft_known","craft_m4","craft_m16"]
    xx=np.arange(5)
    for j,(path,label,col,mark) in enumerate([("fp16","FP16","#222222","o"),
              ("fake_w8a8","Fake W8A8","#2575a7","s"),("fused_int8","Native INT8","#b04b20","^")]):
        a.plot(xx+(j-1)*.16,[val(g,m,path) for m in names],marker=mark,ls="",color=col,label=label,ms=4)
    a.set_xticks(xx,["Original","Erased","Known","M=4*","M=16*"],rotation=20)
    a.set_ylim(0,1); a.set_title(title); a.set_ylabel("Mean detector score")
    a.grid(axis="y",alpha=.2)
ax[0].legend(loc="lower left",fontsize=7,frameon=False)
fig.savefig(FIG/"behavior.pdf"); plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(7.05,2.5),layout="constrained")
dd=geometry["decay"]
ax[0].plot([d['M'] for d in dd],[d['observed'] for d in dd],"o-",label="Measured, one tensor",color="#2575a7")
ax[0].plot([d['M'] for d in dd],[d['law'] for d in dd],"s--",label="Random-phase surrogate",color="#b04b20")
ax[0].set_xscale("log",base=2); ax[0].set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32])
ax[0].set_xlabel("Number of calibration draws M"); ax[0].set_ylabel("Mean normalized collision width")
ax[0].legend(fontsize=7,frameon=False); ax[0].grid(alpha=.2)
ts=[0,.15,.3,.5,.75,1]
ms=["original"]+[f"erased_t{float(t)}" for t in ts[1:]]
for path,label,col in [("fp16","FP16","#222222"),("fake_w8a8","Fake W8A8","#2575a7"),("fused_int8","Native INT8","#b04b20")]:
    ax[1].plot(ts,[val('nudity',m,path) for m in ms],"o-",ms=3,label=label,color=col)
ax[1].set_xlabel("Interpolation depth t"); ax[1].set_ylabel("Mean NudeNet score")
ax[1].legend(fontsize=7,frameon=False); ax[1].grid(alpha=.2)
fig.savefig(FIG/"geometry_ladder.pdf"); plt.close(fig)

# Post-hoc detection-threshold sensitivity: report the complete predefined grid.
thresholdrows=[]
for m in ("original","erased_t0.75","erased_t1.0","craft_known","craft_m4","craft_m16"):
    for threshold in (.3,.45,.6,.75):
        rates=[np.mean(np.asarray(groups['nudity'][m][p])>=threshold) for p in ("fp16","fake_w8a8","fused_int8")]
        thresholdrows.append(dict(model=m,threshold=threshold,fp16=rates[0],fake=rates[1],native=rates[2]))
pd.DataFrame(thresholdrows).to_csv(DATA/"nudenet_thresholds.csv",index=False)

honest=contrasts[(contrasts.family_n==18)&(contrasts.a=="fp16")]
craft=contrasts[(contrasts.family_n==12)&(contrasts.a=="fp16")]
summary=dict(verified_files=len(verified), honest_contrasts=len(honest),
             honest_max_individual_upper=float(honest.hi.max()),
             honest_max_family_upper=float(honest.family_hi.max()),
             craft_contrasts=len(craft),craft_max_individual_upper=float(craft.hi.max()),
             craft_max_family_upper=float(craft.family_hi.max()),
             stealth_min_family_lower=float(min(r['family_lo'] for r in stealth)),
             note="Post-hoc approximate Bonferroni percentile bounds; 50000 crossed draws. Original results unchanged.")
(DATA/"analysis_summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print("Known-calibration stealth contrasts:")
print(pd.DataFrame(stealth).to_string(index=False))
print("Audit diagnostic:",[(n,d['threshold'],d['honest_loo_false_positives'],
      [m for m,r in d['models'].items() if r['flagged']]) for n,d in [('style',diag_s),('nudity',diag_n)]])
