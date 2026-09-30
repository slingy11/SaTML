"""Inline section files and generated tables into one submission-ready main.tex.

Run from paper/:  python make_submission.py  -> writes paper/submission/
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "submission"


def inline(text):
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("\\input{sec_") and s.endswith("}"):
            out.append(inline((HERE / (s[len("\\input{"):-1] + ".tex")).read_text(encoding="utf-8")))
        elif "\\tableinput{" in line:
            pre, rest = line.split("\\tableinput{", 1)
            path, post = rest.split("}", 1)
            out.append(pre + (HERE / path).read_text(encoding="utf-8").rstrip("\n") + post)
        else:
            out.append(line)
    return "\n".join(out)


text = inline((HERE / "main.tex").read_text(encoding="utf-8"))
for drop in ("\\makeatletter\n\\newcommand{\\tableinput}[1]{\\@@input #1 }\n\\makeatother\n",
             "\\newcommand{\\TBD}[1]{\\textcolor{red}{[#1]}}\n"):
    text = text.replace(drop, "")
assert "\\input{" not in text and "tableinput" not in text and "\\TBD" not in text
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "main.tex").write_text(text + "\n", encoding="utf-8")
shutil.copy(HERE / "references.bib", OUT / "references.bib")
for f in ("coord_slack.pdf", "attack_verdict.pdf"):
    shutil.copy(HERE / "figures" / f, OUT / "figures" / f)
print("wrote", OUT / "main.tex", len(text.splitlines()), "lines")
