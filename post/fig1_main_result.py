"""Figure 1 for the LessWrong post: the main result, emotion vectors against control vectors in each condition.

Bars: mean over the 40 tasks of the per-task mean self_resid (vector on the answer, registered layer, primary dose),
for the emotion vectors (Qwen3-32B: joy, sadness, anger, fear, euphoria, surprise; Kimi K2.6: joy, fear, euphoria,
surprise) and for the four controls (red, ocean, arithmetic, random); 95% CI, t with 39 df.
Brackets: emotions minus controls, the mixed-model estimate of runs/confirm.R (y ~ x + (1 + x | task) + (1 + x | para)).
Stars: No argument and Convinced by Argument are the preregistered tests H2q, H1q, H2k, H1k, with their Holm p
(results/confirm/analysis/hypotheses.csv); the two fine-tunes are exploratory, uncorrected p ("fine-tunes" rows of
results/confirm/analysis/explore.csv).

Usage: python post/fig1_main_result.py post/1-main-result.png
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
A = ROOT / "results/confirm/analysis"
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
REGISTERED = {("qwen", "D0"): "H2q", ("qwen", "D1.4"): "H1q", ("k26", "D0"): "H2k", ("k26", "D1.4"): "H1k"}


def table(name):
    """Rows of results/confirm/tables/<name>.csv, or of <name>.csv.gz when only the compressed copy ships."""
    p = ROOT / "results/confirm/tables" / f"{name}.csv"
    return csv.DictReader(open(p) if p.exists() else gzip.open(p.with_suffix(".csv.gz"), "rt"))


def task_means(name):
    """{arm: {"emo"|"ctl": {task: mean self_resid}}} at the answer site, registered layer, primary dose."""
    base = name.split("-")[0]
    layer, dose = SETUP[base]
    group = {**{v: "emo" for v in EMOS[base]}, **{v: "ctl" for v in CTRL}}
    acc = defaultdict(lambda: defaultdict(list))
    for r in table(name):
        if (r["measure"] == "self_resid" and r["site"] == "answer" and r["vec"] in group
                and int(r["layer"]) == layer and float(r["dose"]) == dose):
            acc[(r["arm"], group[r["vec"]])][r["task"]].append(float(r["value"]))
    out = defaultdict(dict)
    for (arm, g), d in acc.items():
        assert len(d) == 40 and len({len(x) for x in d.values()}) == 1, (name, arm, g)
        out[arm][g] = {t: float(np.mean(x)) for t, x in d.items()}
    return out


def mci(d):
    """Mean over tasks and its 95% CI half-width (t(0.975, 39) = 2.023, as in the published figure)."""
    x = np.array(list(d.values()))
    return x.mean(), 2.023 * x.std(ddof=1) / np.sqrt(len(x))


def test(m, arm):
    """(estimate, p) of emotions minus controls: registered tests with their Holm p, the fine-tunes uncorrected."""
    if (m, arm) in REGISTERED:
        r = next(r for r in csv.DictReader(open(A / "hypotheses.csv")) if r["id"] == REGISTERED[(m, arm)])
        return float(r["estimate"]), float(r["p_holm"])
    r = next(r for r in csv.DictReader(open(A / "explore.csv"))
             if r["analysis"] == "fine-tunes" and r["model"] == m and r["arm"] == arm and r["what"] == "emotions - controls")
    return float(r["estimate"]), float(r["p"])


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def sgn(v):
    return "0.00" if abs(v) < 0.005 else f"{v:+.2f}"


def bracket(ax, x0, x1, y, label, d=0.025):
    ax.plot([x0, x0, x1, x1], [y - d, y, y, y - d], color=INK, lw=0.9)
    ax.text((x0 + x1) / 2, y + 0.008, label, ha="center", va="bottom", fontsize=9.5, color=INK)


TM = {name: task_means(name) for name in ("qwen", "k26", "qwen-D2", "qwen-D2c")}
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), gridspec_kw={"width_ratios": [4, 2.3]}, sharey=True)
for ax, base in zip(axes, ("qwen", "k26")):
    for j, (m, arm, lab) in enumerate(CONDS[base]):
        tops = []
        for g, dx, col in (("emo", -0.18, EMO), ("ctl", 0.18, NON)):
            mu, h = mci(TM[m][arm][g])
            ax.bar(j + dx, mu, width=0.35, color=col, yerr=h, error_kw=dict(lw=1, ecolor=INK, capsize=0))
            tops.append(mu + h)
            print(f"{m:9s} {arm:5s} {g}: {mu:+.4f} (95% CI half-width {h:.4f})")
        e, p = test(m, arm)
        bracket(ax, j - 0.18, j + 0.18, max(*tops, 0.0) + 0.05, f"{sgn(e)} {stars(p)}")
        print(f"{m:9s} {arm:5s} bracket: emotions - controls {e:+.4f}, p = {p:.3g} {stars(p)}")
    ax.set_xticks(range(len(CONDS[base])), [c[2] for c in CONDS[base]])
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_title(MODEL[base], fontsize=11.5)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("score (log-odds)")
axes[0].set_ylim(-0.3, 0.62)
axes[1].tick_params(axis="y", labelleft=True)
fig.legend(handles=[Patch(color=EMO, label="emotion vectors"),
                    Patch(color=NON, label="control vectors (red, ocean, arithmetic, random)")],
           loc="upper left", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.0, 1.03))
fig.savefig(OUT)
print(OUT)
