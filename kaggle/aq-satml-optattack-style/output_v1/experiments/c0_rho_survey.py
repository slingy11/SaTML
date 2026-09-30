"""C0 -- The rho survey: does one weight-space number predict whether an edit survives
quantization, across edit families?

    rho = E|dW| / Delta        Delta = the quantizer's bin width for the ORIGINAL weights

rho is computable from weight deltas alone -- no generation, no evaluation, no GPU. The
claim under test is that it predicts the measured edit survival
    fidelity = cos( q(W_edited) - q(W_orig),  dW )
with no free parameters: rho >> 1 means the edit clears the bins and survives
quantization; rho << 1 means quantization snaps it away and the edit is silently undone.

Stage 1/2 supplies one anchor point: UCE on SD-1.5 attn2 has a large rho and its erasure
is behaviourally intact after W8A8 (measured in A2, not inferred).

NOTE: the naive "edit survival" ratio |q(W_e)-q(W_o)|/|dW| is ~1.0 at every rho and is
NOT diagnostic -- see lib.wquant.edit_preservation. This survey uses cosine fidelity.
The question is whether other edit families -- instruction/safety fine-tuning, LoRA
merges, other erasers -- land in the same regime or in the rho < 1 regime, where
quantization is silently undoing the edit in production.

CPU only. Cost is download time, not compute. Kaggle-friendly.
Run:  python experiments/c0_rho_survey.py
"""
import gc
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.wquant import BIN_WIDTH, edit_preservation

OUT = os.environ.get("OUT", "outputs")
os.makedirs(OUT, exist_ok=True)
MIN_NUMEL = int(os.environ.get("MIN_NUMEL", 1 << 16))     # skip tiny tensors
MAX_TENSORS = int(os.environ.get("MAX_TENSORS", 0))       # 0 = all
# embedding / lm_head matrices are hundreds of millions of params; they are not
# where edits are studied and they dominate peak memory. Skipped by default.
MAX_NUMEL = int(os.environ.get("MAX_NUMEL", 80_000_000))
KEEP_CACHE = os.environ.get("KEEP_CACHE", "0") == "1"
QUANTIZERS = os.environ.get("QUANTIZERS", "nf4,int8_perchannel").split(",")

# (family, label, base_repo, edited_repo). Ungated repos by default so the survey runs
# without an HF token; add gated pairs (Llama, Gemma) via PAIRS_JSON if you have one.
PAIRS = [
    ("instruct_ft", "Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("instruct_ft", "Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("instruct_ft", "SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B",
     "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("instruct_ft", "TinyLlama-1.1B", "TinyLlama/TinyLlama_v1.1",
     "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
]
if os.environ.get("PAIRS_JSON"):
    PAIRS = [tuple(p) for p in json.load(open(os.environ["PAIRS_JSON"]))]


def shard_map(repo):
    """{tensor_name: shard_path} without loading anything."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"])
    m = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            p = os.path.join(d, f)
            with safe_open(p, framework="pt") as h:
                for k in h.keys():
                    m[k] = p
    return m


def get(name, smap, cache):
    """Read one tensor, keeping at most one open handle per shard."""
    from safetensors import safe_open
    p = smap[name]
    if p not in cache:
        cache.clear()
        cache[p] = safe_open(p, framework="pt")
    return cache[p].get_tensor(name)


def survey_pair(family, label, base_repo, edit_repo):
    bm, em = shard_map(base_repo), shard_map(edit_repo)
    common = [k for k in bm if k in em]
    bc, ec = {}, {}
    rows, n, skipped = [], 0, []
    for k in common:
        Wo = get(k, bm, bc)
        if Wo.ndim != 2 or Wo.numel() < MIN_NUMEL:
            continue
        if Wo.numel() > MAX_NUMEL:
            skipped.append((k, int(Wo.numel())))
            continue
        We = get(k, em, ec)
        if We.shape != Wo.shape:
            continue
        Wo = Wo.float()
        We = We.float()
        dW = We - Wo
        mad = float(dW.abs().mean())
        if mad < 1e-12:
            continue                              # untouched tensor
        row = dict(family=family, model=label, tensor=k, numel=int(Wo.numel()),
                   mean_abs_dW=mad, mean_abs_W=float(Wo.abs().mean()),
                   rel_edit=mad / float(Wo.abs().mean().clamp(min=1e-12)))
        for q in QUANTIZERS:
            row[f"{q}_rho"] = mad / max(BIN_WIDTH[q](Wo), 1e-30)
            for mk, mv in edit_preservation(Wo, We, q).items():
                row[f"{q}_{mk}"] = mv
        rows.append(row)
        n += 1
        del Wo, We, dW
        if MAX_TENSORS and n >= MAX_TENSORS:
            break
    bc.clear(); ec.clear(); gc.collect()
    print(f"  {label}: {n} edited 2-D tensors"
          + (f" ({len(skipped)} skipped as > {MAX_NUMEL/1e6:.0f}M params: "
             f"{[x[0] for x in skipped][:3]})" if skipped else ""), flush=True)
    if not KEEP_CACHE:
        import shutil
        for r in (base_repo, edit_repo):          # free disk before the next pair
            d = os.path.expanduser(
                "~/.cache/huggingface/hub/models--" + r.replace("/", "--"))
            shutil.rmtree(d, ignore_errors=True)
    return rows


all_rows = []
for fam, label, b, e in PAIRS:
    print(f"[{fam}] {label}: {b} -> {e}", flush=True)
    try:
        all_rows += survey_pair(fam, label, b, e)
    except Exception:
        print(f"  SKIP {label}:\n{traceback.format_exc(limit=2)}", flush=True)

# ---- PUBLIC diffusion eraser checkpoints (no training needed) ---------------------
# The fastest route to the ESD-u question: rho needs only weight deltas, so a released
# ESD checkpoint answers it in minutes on CPU instead of 1-2 h of GPU fine-tuning.
# Set ESD_CKPTS to a JSON list of [label, repo_or_path] entries; entries may be an HF
# repo id or a local .safetensors/.bin of a UNet state dict. Failures are skipped loudly.
_ESD_BASE = "https://erasing.baulab.info/weights/esd_models"
# ESD was trained on Stable Diffusion v1.4. Diffing its checkpoints against v1.5 makes
# the v1.4->v1.5 BASE-MODEL difference dominate dW -- it swamped the erasure edit and
# made ESD-u and ESD-x look identical. The base model must match the checkpoint.
_SD14 = "CompVis/stable-diffusion-v1-4"
ESD_CKPTS = json.loads(os.environ.get("ESD_CKPTS", json.dumps([
    # ESD-u ("noxattn"): the variant that trains the outlier-carrying layers. THE one.
    ["ESDu_nudity", f"{_ESD_BASE}/NSFW/diffusers-nudity-ESDu1-UNET.pt", _SD14],
    # ESD-x control: same concept as UCE/TIME above, cross-attention only
    ["ESDx_VanGogh", f"{_ESD_BASE}/art/diffusers-VanGogh-ESDx1-UNET.pt", _SD14],
])))
if ESD_CKPTS:
    try:
        from lib.editors import build_pipe
        _base_cache = {}

        def _base_for(repo):
            if repo not in _base_cache:
                _p, _ = build_pipe(model=repo, device="cpu", dtype=torch.float32)
                _base_cache.clear()          # one base in memory at a time
                _base_cache[repo] = {n: q.detach().float()
                                     for n, q in _p.unet.named_parameters()}
            return _base_cache[repo]

        for _entry in ESD_CKPTS:
            label, src = _entry[0], _entry[1]
            base_repo = _entry[2] if len(_entry) > 2 else _SD14
            base_sd = _base_for(base_repo)
            try:
                tmp = None
                if os.path.exists(src):
                    path = src
                elif src.startswith("http"):
                    # direct URL (e.g. erasing.baulab.info ESD weights)
                    import urllib.request
                    tmp = os.path.join("/tmp", os.path.basename(src))
                    if not os.path.exists(tmp):
                        print(f"    downloading {src}", flush=True)
                        urllib.request.urlretrieve(src, tmp)
                    path = tmp
                else:
                    from huggingface_hub import hf_hub_download, list_repo_files
                    files = [f for f in list_repo_files(src)
                             if f.endswith((".safetensors", ".bin", ".pt"))]
                    if not files:
                        raise FileNotFoundError(f"no weight file in {src}")
                    path = hf_hub_download(src, files[0])
                if path.endswith(".safetensors"):
                    from safetensors.torch import load_file as _lf
                    ed = {k: v.clone() for k, v in _lf(path).items()}
                else:
                    try:
                        ed = torch.load(path, map_location="cpu", weights_only=True)
                    except Exception:
                        ed = torch.load(path, map_location="cpu")
                    ed = ed.get("state_dict", ed) if isinstance(ed, dict) else ed
                ed = {k.replace("model.diffusion_model.", "").replace("unet.", ""): v
                      for k, v in ed.items()}
                hit = 0
                for k, Wo in base_sd.items():
                    We = ed.get(k)
                    if We is None or We.shape != Wo.shape or Wo.ndim != 2:
                        continue
                    We = We.float()
                    dW = We - Wo
                    mad = float(dW.abs().mean())
                    if mad < 1e-12:
                        continue
                    row = dict(family="concept_erasure_ckpt", model=label, tensor=k,
                               numel=int(Wo.numel()), mean_abs_dW=mad,
                               mean_abs_W=float(Wo.abs().mean()),
                               rel_edit=mad / float(Wo.abs().mean()))
                    for q in QUANTIZERS:
                        row[f"{q}_rho"] = mad / max(BIN_WIDTH[q](Wo), 1e-30)
                        for mk, mv in edit_preservation(Wo, We, q).items():
                            row[f"{q}_{mk}"] = mv
                    all_rows.append(row)
                    hit += 1
                # sanity check that the base matches: ESD-x edits attn2 ONLY, so if the
                # base model is right its edit energy must concentrate there.
                import collections
                en = collections.Counter()
                for r in all_rows[-hit:]:
                    t = r["tensor"]
                    k = ("attn2" if "attn2" in t else "attn1" if "attn1" in t else
                         "ff" if ".ff." in t else "other")
                    en[k] += (r["mean_abs_dW"] ** 2) * r["numel"]
                tot = sum(en.values()) or 1.0
                share = {k: round(v / tot, 4) for k, v in en.most_common()}
                print(f"  {label}: {hit} edited 2-D tensors vs {base_repo}", flush=True)
                print(f"    edit energy by layer kind: {share}", flush=True)
                del ed
                gc.collect()
                if tmp and os.path.exists(tmp) and os.environ.get("KEEP_CKPT") != "1":
                    os.remove(tmp)          # 3.2 GB each; Kaggle disk is finite
            except Exception:
                print(f"  SKIP {label} ({src}):\n{traceback.format_exc(limit=2)}", flush=True)
    except Exception:
        print(f"  SKIP ESD checkpoints:\n{traceback.format_exc(limit=2)}", flush=True)

# ---- the diffusion erasers, computed locally, as the anchor -----------------------
if os.environ.get("INCLUDE_DIFFUSION", "1") == "1":
    try:
        from safetensors.torch import load_file
        from lib.editors import (build_pipe, cross_attn_keys, load_uce, restore,
                                 snapshot_original, time_edit)
        pipe, _ = build_pipe(device="cpu", dtype=torch.float32)
        uce = load_uce("cpu", dtype=torch.float32)
        os.makedirs("uce_models", exist_ok=True)
        keys = cross_attn_keys(pipe.unet)
        orig = snapshot_original(pipe, keys)
        C = "Van Gogh"
        EC = [C, f"painting by {C}", f"art by {C}", f"style of {C}"]
        edits = {}
        for sc in (0.5, 1.0, 2.0):
            restore(pipe, orig, keys)
            uce.UCE(pipe, edit_concepts=EC, guide_concepts=["art"] * len(EC),
                    preserve_concepts=[], erase_scale=float(sc), preserve_scale=1.0,
                    lamb=0.5, save_dir="uce_models", exp_name=f"c0_s{sc}")
            edits[("concept_erasure", f"UCE_s{sc}")] = {
                k: v.float().clone() for k, v in load_file(f"uce_models/c0_s{sc}.safetensors").items()}
        edits[("concept_erasure", "TIME_l0.1")] = time_edit(
            pipe, orig, keys, [(C, "art"), (f"painting by {C}", "painting")], "cpu", lamb=0.1)
        for (fam, label), ed in edits.items():
            for k in keys:
                Wo, We = orig[k], ed[k].float()
                dW = We - Wo
                mad = float(dW.abs().mean())
                if mad < 1e-12:
                    continue
                row = dict(family=fam, model=label, tensor=k, numel=int(Wo.numel()),
                           mean_abs_dW=mad, mean_abs_W=float(Wo.abs().mean()),
                           rel_edit=mad / float(Wo.abs().mean()))
                for q in QUANTIZERS:
                    row[f"{q}_rho"] = mad / max(BIN_WIDTH[q](Wo), 1e-30)
                    for mk, mv in edit_preservation(Wo, We, q).items():
                        row[f"{q}_{mk}"] = mv
                all_rows.append(row)
            print(f"  {label}: done", flush=True)
    except Exception:
        print(f"  SKIP diffusion anchor:\n{traceback.format_exc(limit=2)}", flush=True)

df = pd.DataFrame(all_rows)
if df.empty:
    raise SystemExit("no pairs surveyed")
df.to_csv(os.path.join(OUT, "c0_rho_survey.csv"), index=False)

print("\n" + "=" * 92)
print("rho = E|dW| / bin width, by edit family and model")
print("=" * 92)
for q in QUANTIZERS:
    g = df.groupby(["family", "model"]).agg(
        n=("tensor", "size"), rho=(f"{q}_rho", "median"),
        collision=(f"{q}_collision", "median"), fidelity=(f"{q}_fidelity", "median"),
        distortion=(f"{q}_distortion", "median"))
    print(f"\n--- {q} ---")
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    sub = df[np.isfinite(df[f"{q}_rho"])]
    r, s = sub[f"{q}_rho"].values, sub[f"{q}_fidelity"].values
    # parameter-free leading-order prediction: survival ~ min(1, rho) is the naive form;
    # report rank correlation, which is what "does rho predict survival" actually asks
    from scipy import stats as st
    rho_s = st.spearmanr(r, s).correlation
    print(f"  Spearman(rho, fidelity) = {rho_s:.3f}   n={len(sub)}")
    below = sub.groupby(["family", "model"]).apply(
        lambda d: float((d[f"{q}_rho"] < 1).mean()), include_groups=False)
    print("  fraction of tensors with rho < 1 (edit smaller than the bin -> at risk):")
    print("   " + below.to_string(float_format=lambda v: f"{v:.3f}").replace("\n", "\n   "))

json.dump({"n_rows": len(df),
           "families": sorted(df.family.unique().tolist()),
           "models": sorted(df.model.unique().tolist())},
          open(os.path.join(OUT, "c0_rho_survey.json"), "w"), indent=2)
print("\nRead-off: any family with median rho < 1 has edits that ordinary quantization")
print("partly erases. That is the actionable finding; a high Spearman means one")
print("weight-space number predicts it with no evaluation and no free parameters.")
