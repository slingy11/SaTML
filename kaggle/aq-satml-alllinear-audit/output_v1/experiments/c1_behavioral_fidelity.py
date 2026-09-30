"""C1 -- Does weight-space fidelity predict BEHAVIOURAL fidelity? The ICLR gate.

C0 established that rho = E|dW|/Delta predicts how faithfully a quantizer preserves an
edit IN WEIGHT SPACE (Spearman 0.99). The open question, and the one a reviewer will
ask, is whether that means anything for the model's behaviour.

There is already one data point AGAINST: UCE at NF4 has weight fidelity 0.556, yet the
prior work found NF4 does not revive the erased concept. So the link is not automatic.

The measurement, built to mirror the weight-space one exactly. An instruct model is the
base model plus an edit. Quantizing it may partly undo that edit, which would move its
behaviour back TOWARD the base model. So define, in logit space,

    b = logits(instruct)   - logits(base)        the intended behavioural edit
    a = logits(instruct_q) - logits(base)        what survives quantization

    behavioural_fidelity = cos(a, b)

This is the exact analogue of the weight-space cosine, so the two are directly
comparable. The prediction under test, made from weights alone and out of sample:

    Qwen2.5-1.5B  weight fidelity 0.699 (INT8) / 0.230 (NF4)  -> LOW  behavioural fidelity
    Qwen2.5-0.5B  weight fidelity 0.980 (INT8) / 0.549 (NF4)  -> HIGH behavioural fidelity

No generation and no benchmark download: forward passes only.
Run:  python experiments/c1_behavioral_fidelity.py
"""
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.wquant import int8pc, nf4

OUT = os.environ.get("OUT", "outputs")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KEEP_POS = int(os.environ.get("KEEP_POS", 16))       # last-N positions of logits kept
MAX_LEN = int(os.environ.get("MAX_LEN", 96))
QUANTS = {"int8_perchannel": int8pc, "nf4": nf4}

PAIRS = [
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
]
if os.environ.get("PAIRS_JSON"):
    PAIRS = [tuple(p) for p in json.load(open(os.environ["PAIRS_JSON"]))]

PROMPTS = [
    "Explain why the sky appears blue.", "Write a haiku about winter.",
    "List three uses for a paperclip.", "What is the capital of Australia?",
    "Summarise the causes of the French Revolution.", "Translate 'good morning' into Spanish.",
    "Give me a recipe for pancakes.", "How do I change a flat tyre?",
    "What is the difference between RAM and storage?", "Describe the water cycle.",
    "Write a polite email declining an invitation.", "Explain recursion to a beginner.",
    "Name four planets in the solar system.", "What causes inflation?",
    "Suggest a title for an essay about privacy.", "How does a refrigerator work?",
    "Give three tips for better sleep.", "What is photosynthesis?",
    "Write a short poem about the ocean.", "Explain what an API is.",
    "How do vaccines work?", "What is the tallest mountain on Earth?",
    "Draft a two-sentence product description for a water bottle.",
    "Explain the rules of chess briefly.", "What is machine learning?",
    "Give me a packing list for a beach holiday.", "Why do leaves change colour?",
    "Describe how to make a paper aeroplane.", "What is compound interest?",
    "Write one sentence about the moon.", "How do I boil an egg?",
    "What is the purpose of a firewall?",
]


def logits_for(model, tok, prompts):
    """Last-KEEP_POS positions of the logits, fp16 on CPU."""
    out = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            ids = {k: v.to(model.device) for k, v in ids.items()}
            lg = model(**ids).logits[0].float()
            out.append(lg[-KEEP_POS:].half().cpu())
    n = min(x.shape[0] for x in out)
    return torch.stack([x[-n:] for x in out])


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-30))


rows = []
for label, base_repo, inst_repo in PAIRS:
    print(f"\n=== {label} ===", flush=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(inst_repo)

        m = AutoModelForCausalLM.from_pretrained(base_repo, torch_dtype=torch.float16).to(DEV).eval()
        L_base = logits_for(m, tok, PROMPTS)
        del m; gc.collect(); torch.cuda.empty_cache()

        m = AutoModelForCausalLM.from_pretrained(inst_repo, torch_dtype=torch.float16).to(DEV).eval()
        L_inst = logits_for(m, tok, PROMPTS)
        # keep the FP16 instruct weights so each quantizer starts from the same place
        orig = {n: p.detach().clone() for n, p in m.named_parameters() if p.ndim == 2}
        b = (L_inst - L_base)

        for qname, qfn in QUANTS.items():
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if n in orig:
                        w = orig[n].float().cpu()
                        p.copy_(qfn(w).to(p.dtype).to(p.device))
            L_q = logits_for(m, tok, PROMPTS)
            a = (L_q - L_base)
            rows.append(dict(model=label, quantizer=qname,
                             behavioural_fidelity=cos(a, b),
                             # how far the quantized model drifts from the instruct model,
                             # relative to the size of the instruct edit itself
                             drift_ratio=float((L_q - L_inst).float().norm()
                                               / b.float().norm().clamp(min=1e-30))))
            print(f"  {qname:17s} behavioural_fidelity={rows[-1]['behavioural_fidelity']:.4f}"
                  f"  drift/edit={rows[-1]['drift_ratio']:.4f}", flush=True)
            with torch.no_grad():           # restore before the next quantizer
                for n, p in m.named_parameters():
                    if n in orig:
                        p.copy_(orig[n])
        del m, orig, L_base, L_inst, L_q
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        print(f"  SKIP {label}: {traceback.format_exc(limit=2)}", flush=True)

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no pairs measured")
df.to_csv(os.path.join(OUT, "c1_behavioral_fidelity.csv"), index=False)

print("\n" + "=" * 86)
print("BEHAVIOURAL vs WEIGHT-SPACE fidelity")
print("=" * 86)
csv = os.path.join(OUT, "c0_rho_survey.csv")
if os.path.exists(csv):
    w = pd.read_csv(csv)
    wf = []
    for _, r in df.iterrows():
        sub = w[w.model == r["model"]]
        if len(sub):
            wf.append(float(sub[f"{r['quantizer']}_fidelity"].median()))
        else:
            wf.append(np.nan)
    df["weight_fidelity"] = wf
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ok = df.dropna(subset=["weight_fidelity"])
    if len(ok) >= 3:
        from scipy import stats as st
        r = st.spearmanr(ok.weight_fidelity, ok.behavioural_fidelity).correlation
        print(f"\n  Spearman(weight_fidelity, behavioural_fidelity) = {r:.4f}  n={len(ok)}")
        print("  -> this number IS the ICLR gate. High means rho predicts behaviour and the")
        print("     predictive-theory framing holds. Low means rho is a weight-space fact")
        print("     only, and the reframe should be dropped rather than argued around.")
else:
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
json.dump(df.to_dict("records"), open(os.path.join(OUT, "c1_behavioral_fidelity.json"), "w"),
          indent=2)
