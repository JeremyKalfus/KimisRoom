"""The eleven preregistered tests, labelled with the LessWrong post's condition names.

Same plot as runs/confirm_figures.py's figS1 (read from results/confirm/analysis/hypotheses.csv), with the
post's labels: "No argument", "Convinced by Argument" and the two "Fine tune on ..." conditions.

Usage: python post/fig3_preregistered_tests.py post/3-preregistered-tests.png
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "PREREG.md").exists())
OUT = Path(sys.argv[1])
H = {r["id"]: r for r in csv.DictReader(open(ROOT / "results/confirm/analysis/hypotheses.csv"))}

INK, MUTED, RULE = "#16181c", "#59606b", "#d6dae0"
YES, WEAK, REV = "#22684a", "#9a7614", "#9b3b1c"
plt.rcParams.update({
    "font.family": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 11, "mathtext.fontset": "stix",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "figure.dpi": 100, "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.25})

ARG, NOARG = "Convinced by Argument", "No argument"
ROWS = [("H1q", f"Qwen3-32B, {ARG}: emotions − controls"), ("H1k", f"Kimi K2.6, {ARG}: emotions − controls"),
        ("H2q", f"Qwen3-32B, {NOARG}: emotions − controls"), ("H2k", f"Kimi K2.6, {NOARG}: emotions − controls"),
        ("H3q", f"Qwen3-32B: {ARG} − {NOARG}"), ("H3k", f"Kimi K2.6: {ARG} − {NOARG}"),
        ("H5q", "Qwen3-32B: Fine tune on \"Can't feel emotion\" − on \"Can't count letters\""),
        ("H6q", f"Qwen3-32B, {ARG}: who was feeling, vector on answer − on task"),
        ("H6k", f"Kimi K2.6, {ARG}: who was feeling, vector on answer − on task"),
        ("H4q", f"Qwen3-32B, {ARG}: which emotion, self − user"),
        ("H4k", f"Kimi K2.6, {ARG}: which emotion, self − user")]

fig, ax = plt.subplots(figsize=(8.2, 5.4))
ax.axvspan(-0.1, 0.1, color="#eef0f3", zorder=0)
ax.axvline(0, color=MUTED, lw=1, zorder=1)
for k, (i, _) in enumerate(ROWS):
    r = H[i]
    e, lo, hi, p = float(r["estimate"]), float(r["ci95_lo"]), float(r["ci95_hi"]), float(r["p_holm"])
    sup = r["supported"] == "TRUE"
    col = (YES if p < 0.01 else WEAK) if sup else REV
    y = len(ROWS) - 1 - k
    ax.errorbar(e, y, xerr=[[e - lo], [hi - e]], fmt="o", color=col, ms=6, lw=2, capsize=0, zorder=3)
    verdict = ("supported" if p < 0.01 else f"supported, p = {p:.2f}") if sup else "reversed"
    ax.text(0.62, y, f"{e:+.2f}   {verdict}", va="center", fontsize=9.5, color=col)
    if k == 1:
        ax.axhline(y - 0.5, color=RULE, lw=1)
ax.set_yticks(range(len(ROWS)), [lab for _, lab in reversed(ROWS)])
ax.set_xlim(-0.45, 0.6)
ax.set_xlabel("estimate, log-odds (95% CI); shaded band = ±0.10, the smallest effect of interest")
ax.set_title("Preregistered tests: 10 of 11 in the predicted direction, Kimi's naming test reversed", pad=12)
ax.text(-0.44, len(ROWS) - 2.0, "primary pair above the line", fontsize=8.5, color=MUTED, va="center")
fig.savefig(OUT)
plt.close(fig)
