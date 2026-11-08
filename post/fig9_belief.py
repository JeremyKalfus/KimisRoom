"""Figure 9 for the LessWrong post: the vectors on the questions about the feelings of all LLMs, emotions against controls.

Bars: mean over the 40 tasks of the per-task mean of `belief`, the preregistered belief measure (PREREG.md): the
polarity-corrected change on the belief_revision items from the unsteered trial of the same arm and task, signed so
that higher means the denial is held more firmly (runs/confirm_table.py), the reverse of figure 1, as the note under
the chart says. Vector on the answer, registered layer, primary dose; the emotion vectors (Qwen3-32B: joy, sadness,
anger, fear, euphoria, surprise; Kimi K2.6: joy, fear, euphoria, surprise) against the four controls (red, ocean,
arithmetic, random); 95% CI, t with 39 df.
Brackets: emotions minus controls, runs/confirm.R's fit_test (y ~ x + (1 + x | task) + (1 + x | para), with its
registered fallbacks); exploratory, uncorrected p, from fig8_fig9_models.csv next to this script (written by
fig8_fig9_models.R; its four base-model rows equal the "emotions - controls on belief" rows of explore.csv).
* p < .05, ** p < .01, *** p < .001.

Usage: python post/fig9_belief.py post/9-belief.png
"""
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "PREREG.md").exists())
MODELS = Path(__file__).resolve().with_name("fig8_fig9_models.csv")
OUT = Path(sys.argv[1])

INK, MUTED = "#16181c", "#59606b"
EMO, NON = "#a8293f", "#8b919b"
plt.rcParams.update({
    "font.family": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 11, "mathtext.fontset": "stix",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "figure.dpi": 100, "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.25})

MODEL = {"qwen": "Qwen3-32B", "k26": "Kimi K2.6"}
EMOS = {"qwen": ["joy", "sadness", "anger", "fear", "euphoria", "surprise"], "k26": ["joy", "fear", "euphoria", "surprise"]}
CTRL = ["red", "ocean", "arithmetic", "random"]
SETUP = {"qwen": (40, 2.0), "k26": (20, 4.0)}  # layer, primary dose
CONDS = {"qwen": [("qwen", "D0", "No argument"), ("qwen", "D1.4", "Convinced by\nArgument"),
                  ("qwen-D2c", "D0", "Fine tune on\n\"Can't count\nletters\""),
                  ("qwen-D2", "D0", "Fine tune on\n\"Can't feel\nemotion\"")],
         "k26": [("k26", "D0", "No argument"), ("k26", "D1.4", "Convinced by\nArgument")]}
NOTE = "Here, higher means the model holds its denial more firmly, the reverse of the first chart."
GAP = 0.125  # inches between the lowest tick label and the note


def table(name):
    """Rows of results/confirm/tables/<name>.csv, or of <name>.csv.gz when only the compressed copy ships."""
    p = ROOT / "results/confirm/tables" / f"{name}.csv"
    return csv.DictReader(open(p) if p.exists() else gzip.open(p.with_suffix(".csv.gz"), "rt"))


def task_means(model):
    """{arm: {"emo"|"ctl": {task: mean belief}}} at the answer site, registered layer, primary dose."""
    base = model.split("-")[0]
    layer, dose = SETUP[base]
    group = {**{v: "emo" for v in EMOS[base]}, **{v: "ctl" for v in CTRL}}
    acc = defaultdict(lambda: defaultdict(list))
    for r in table(model):
        if (r["measure"] == "belief" and r["site"] == "answer" and r["vec"] in group
                and int(r["layer"]) == layer and float(r["dose"]) == dose):
            acc[(r["arm"], group[r["vec"]])][r["task"]].append(float(r["value"]))
    out = defaultdict(dict)
    for (arm, g), d in acc.items():
        assert len(d) == 40 and len({len(x) for x in d.values()}) == 1, (model, arm, g)
        out[arm][g] = {t: float(np.mean(x)) for t, x in d.items()}
    return out


def mci(d):
    """Mean over tasks and its 95% CI half-width (t(0.975, 39) = 2.023, as in the published figure)."""
    x = np.array(list(d.values()))
    return x.mean(), 2.023 * x.std(ddof=1) / np.sqrt(len(x))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def sgn(v):
    return "0.00" if abs(v) < 0.005 else f"{v:+.2f}".replace("-", "−")


TEST = {(r["model"], r["arm"]): r for r in csv.DictReader(open(MODELS)) if r["figure"] == "9"}
TM = {m: task_means(m) for m in ("qwen", "k26", "qwen-D2", "qwen-D2c")}
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), gridspec_kw={"width_ratios": [4, 2.3]}, sharey=True)
for ax, base in zip(axes, ("qwen", "k26")):
    for j, (m, arm, lab) in enumerate(CONDS[base]):
        hi = {}
        for g, dx, col in (("emo", -0.18, EMO), ("ctl", 0.18, NON)):
            mu, h = mci(TM[m][arm][g])
            ax.bar(j + dx, mu, width=0.35, color=col, yerr=h, error_kw=dict(lw=1, ecolor=INK, capsize=0))
            hi[g] = mu + h
            print(f"{m:9s} {arm:5s} {g}: {mu:+.4f} (95% CI half-width {h:.4f})")
        r = TEST[(m, arm)]
        e, p = float(r["estimate"]), float(r["p"])
        x0, x1, y = j - 0.18, j + 0.18, max(hi["emo"], hi["ctl"], 0.0) + 0.07
        ax.plot([x0, x0, x1, x1], [max(hi["emo"], 0) + 0.025, y, y, max(hi["ctl"], 0) + 0.025], color=INK, lw=0.9)
        ax.text((x0 + x1) / 2, y + 0.012, f"{sgn(e)} {stars(p)}", ha="center", va="bottom", fontsize=9.5, color=INK)
        print(f"{m:9s} {arm:5s} bracket: emotions - controls {e:+.4f}, SE {float(r['se']):.4f}, "
              f"df {float(r['df']):.1f}, p = {p:.3g} {stars(p)} ({r['fit']} model)")
    ax.set_xticks(range(len(CONDS[base])), [c[2] for c in CONDS[base]])
    ax.axhline(0, color=MUTED, lw=1)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(MODEL[base], fontsize=11.5)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("score (log-odds)")
axes[0].set_ylim(-0.6, 0.3)
axes[0].set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2])
axes[1].tick_params(axis="y", labelleft=True)
leg = fig.legend(handles=[Patch(color=EMO, label="emotion vectors"),
                          Patch(color=NON, label="control vectors (red, ocean, arithmetic, random)")],
                 loc="upper left", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.0, 1.03))

# The note goes under everything drawn so far, flush with the legend's first swatch, the leftmost mark.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
x = leg.legend_handles[0].get_window_extent(renderer).x0 / fig.bbox.width
y = (fig.get_tightbbox(renderer).y0 - GAP) / fig.get_figheight()
fig.text(x, y, NOTE, ha="left", va="top", fontsize=8.5, color=MUTED)
fig.savefig(OUT)
print(OUT)
