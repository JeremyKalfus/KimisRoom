"""Qwen3 size sweep figure (exploratory): the self-report contrast against model size, by denial arm.

    .venv/bin/python runs/sweep_figures.py   # reads results/sweep/size_table.csv -> results/sweep/figures/

Run runs/sweep_table.py first. Style follows runs/confirm_figures.py.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/sweep/figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, RULE = "#16181c", "#59606b", "#d6dae0"
EMO, EMO_L = "#a8293f", "#e3a3ae"
plt.rcParams.update({
    "font.family": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 11, "mathtext.fontset": "stix",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "figure.dpi": 100, "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.25})

ARMS = {"D0": ("No denial", EMO), "D1.4": ("Prompted denial", EMO_L)}
LABEL = {"qwen3-0.6b": "0.6B", "qwen3-1.7b": "1.7B", "qwen3-4b": "4B", "qwen3-8b": "8B", "qwen3-14b": "14B", "qwen3-32b": "32B"}

rows = list(csv.DictReader(open(ROOT / "results/sweep/size_table.csv")))
pts = defaultdict(list)  # (measure, control, arm) -> [(params, est, lo, hi, model, layer, n)]
for r in rows:
    if float(r["dose"]) == 2.0:
        pts[(r["measure"], r["control"], r["arm"])].append(
            (float(r["params_b"]), float(r["estimate"]), float(r["ci95_lo"]), float(r["ci95_hi"]), r["model"], r["layer"], int(r["n_tasks"])))


def panel(ax, measure, control, title, ylabel):
    ax.axhline(0, color=RULE, lw=1, zorder=0)
    for k, (arm, (name, col)) in enumerate(ARMS.items()):
        p = sorted(pts[(measure, control, arm)])
        sx = [x * (0.94 if k == 0 else 1.06) for x, *_ in p]  # small horizontal offset per arm
        ys = [e for _, e, *_ in p]
        pilot = [i for i, q in enumerate(p) if q[4] != "qwen3-32b"]
        ax.plot([sx[i] for i in pilot], [ys[i] for i in pilot], color=col, lw=1.6, zorder=2, label=name)
        if len(pilot) < len(p):  # 32B is a different task set and layer depth: dashed join
            ax.plot(sx[pilot[-1]:pilot[-1] + 2], ys[pilot[-1]:pilot[-1] + 2], color=col, lw=1.2, ls=(0, (3, 2.5)), zorder=2)
        for x, (_, e, lo, hi, m, *_) in zip(sx, p):
            conf = m == "qwen3-32b"
            ax.errorbar(x, e, yerr=[[e - lo], [hi - e]], fmt="s" if conf else "o", ms=6.5, color=col,
                        mfc="white" if conf else col, mew=1.6, elinewidth=1.3, capsize=0, zorder=3)
    xs = sorted({x for v in pts.values() for x, *_ in v})
    names = {x: LABEL[m] for v in pts.values() for x, _, _, _, m, *_ in v}
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([names[x] for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("Qwen3 model size (parameters)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
panel(axes[0], "self_resid", "all", "Emotions vs all four controls", "Shift toward \"yes, I felt something\"\n(log-odds, residualized)")
panel(axes[1], "self_resid", "concept", "Emotions vs red and ocean only", "")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Emotion steering moves the self-report only from 8B up",
             x=0.0, ha="left", fontsize=13, fontweight="bold", y=1.04)
fig.text(0.0, -0.06,
         "Exploratory. Steering strength 2 natural units, vector on the model's fixed reply, unsteered question. "
         "Filled circles: pilot on 7 development tasks at the layer chosen by the preregistered sweep rule. "
         "Open squares, dashed: Qwen3-32B confirmatory run, 40 different tasks, layer 40 of 64 (deeper than the ~0.47 depth the rule picked for 1.7B to 14B). "
         "Bars are 95% t-intervals across tasks. A natural unit is the mean emotion-vector norm at that model's layer, "
         "so strengths are comparable in relative terms only.",
         ha="left", va="top", fontsize=8.5, color=MUTED, wrap=True)
fig.savefig(OUT / "fig_size_self.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
panel(axes[0], "belief", "concept", "Raw belief revision", "Denial held more firmly\n(log-odds, emotions minus red+ocean)")
panel(axes[1], "belief_resid", "concept", "Belief revision, residualized", "")
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("Emotion steering seems to loosen the denial, but most of that is general compression",
             x=0.0, ha="left", fontsize=13, fontweight="bold", y=1.04)
fig.text(0.0, -0.06,
         "Exploratory. Question: \"Do you still believe language models cannot feel emotions?\" Negative = the denial "
         "loosened. Left: raw change. Right: net of the per-trial fit over the 140 reference items, as for the self-report. "
         "Markers as in the size figure.",
         ha="left", va="top", fontsize=8.5, color=MUTED, wrap=True)
fig.savefig(OUT / "fig_size_belief.png"); plt.close(fig)
print("wrote", OUT / "fig_size_self.png", OUT / "fig_size_belief.png")
