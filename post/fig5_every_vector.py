"""Figure 5 for the LessWrong post: every vector's corrected self-report score, per panel.

Only the vectors each model's tests use: Qwen's six AIPsy-validated emotions, Kimi's four, and the four controls.
Calm (no AIPsy-Affect label) and Kimi's sadness and anger (failed the check) are left out.
Bars: mean over the 40 tasks of the per-task mean self_resid (vector on the answer, primary dose), 95% CI.
Dashed line: the control with the highest mean in that panel. Stars: paired t-test over the 40 tasks of each
emotion against that control, Holm-corrected over the emotions in the panel (exploratory).

Usage: python post/fig5_every_vector.py post/5-every-vector.png (also writes post/5-every-vector.csv)
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import math  # noqa: E402

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "PREREG.md").exists())
T = ROOT / "results/confirm/tables"
OUT = Path(sys.argv[1])


def open_table(name):
    """results/confirm/tables/<name>.csv if it was unzipped, else the gzipped copy in the repository."""
    p = T / f"{name}.csv"
    return open(p) if p.exists() else gzip.open(T / f"{name}.csv.gz", "rt")

INK, MUTED, RULE = "#16181c", "#59606b", "#d6dae0"
EMO, NON, REV = "#a8293f", "#8b919b", "#9b3b1c"
plt.rcParams.update({
    "font.family": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 11, "mathtext.fontset": "stix",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight",
})

MODEL = {"qwen": "Qwen3-32B", "k26": "Kimi K2.6"}
EMOS = {"qwen": ["joy", "sadness", "anger", "fear", "euphoria", "surprise"], "k26": ["joy", "fear", "euphoria", "surprise"]}
CTRL = ["red", "ocean", "arithmetic", "random"]
SETUP = {"qwen": (40, 2.0), "k26": (20, 4.0)}  # layer, primary dose
PANELS = [("qwen", "qwen", "D0", "No argument"), ("qwen", "qwen", "D1.4", "Convinced by Argument"),
          ("qwen", "qwen-D2c", "D0", "Fine tune on \"Can't count letters\""),
          ("qwen", "qwen-D2", "D0", "Fine tune on \"Can't feel emotion\""),
          ("k26", "k26", "D0", "No argument"), ("k26", "k26", "D1.4", "Convinced by Argument")]


def task_means(base, table, arm):
    """{vec: {task: mean self_resid over paraphrases}} at the answer site, registered layer, primary dose."""
    layer, dose = SETUP[base]
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open_table(table)):
        if (r["arm"] == arm and r["measure"] == "self_resid" and r["site"] == "answer"
                and int(r["layer"]) == layer and float(r["dose"]) == dose):
            acc[r["vec"]][r["task"]].append(float(r["value"]))
    return {v: {t: float(np.mean(x)) for t, x in d.items()} for v, d in acc.items()}


def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta (Numerical Recipes 6.4)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def betai(a, b, x):
    if x <= 0.0 or x >= 1.0:
        return 0.0 if x <= 0.0 else 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1.0) / (a + b + 2.0) else 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_p(x, y):
    """Two-sided paired t-test p-value."""
    d = x - y
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    df = n - 1
    return betai(df / 2.0, 0.5, df / (df + t * t))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


fig = plt.figure(figsize=(12.5, 7.4))
gs = fig.add_gridspec(2, 4, height_ratios=[10, 8], hspace=0.55, wspace=0.12)
rows = []
for idx, (base, table, arm, label) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, idx] if idx < 4 else gs[1, idx - 4])
    tm = task_means(base, table, arm)
    order = EMOS[base] + CTRL
    tasks = sorted(tm[order[0]])
    assert len(tasks) == 40 and all(sorted(tm[v]) == tasks for v in order), (table, arm)
    val = {v: np.array([tm[v][t] for t in tasks]) for v in order}
    top = max(CTRL, key=lambda v: val[v].mean())
    p = np.array([paired_p(val[v], val[top]) for v in EMOS[base]])
    p_adj = dict(zip(EMOS[base], holm(p)))
    n = len(order)
    ax.axvline(0, color=MUTED, lw=1)
    for k, v in enumerate(order):
        y = n - 1 - k
        mu, se = val[v].mean(), val[v].std(ddof=1) / np.sqrt(len(tasks))
        ax.barh(y, mu, height=0.7, color=EMO if v in EMOS[base] else NON, xerr=1.96 * se,
                error_kw=dict(lw=0.8, ecolor=INK))
        rows.append((table, arm, v, mu, 1.96 * se, top, p_adj.get(v)))
        if v in p_adj and stars(p_adj[v]):
            diff = mu - val[top].mean()
            x = max(mu + 1.96 * se, val[top].mean(), 0.0) + 0.02
            ax.text(x, y, stars(p_adj[v]), va="center", ha="left", fontsize=9, color=INK if diff > 0 else REV)
    ax.axvline(val[top].mean(), color=INK, lw=1, ls="--")
    ax.annotate(top, xy=(val[top].mean(), 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(len(CTRL) - 0.5, color=RULE, lw=1)
    ax.set_title(f"{MODEL[base]}\n{label}", fontsize=10.5, pad=16, loc="left", fontweight="bold")
    ax.set_xlim(-0.38, 0.88)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0.0, 0.5])
    ax.set_xlabel("score (log-odds)")
    ax.set_yticks(range(n), list(reversed(order)) if idx in (0, 4) else [""] * n)

fig.legend(handles=[Patch(color=EMO, label="emotion vectors"), Patch(color=NON, label="control vectors"),
                    Line2D([], [], color=INK, ls="--", lw=1, label="strongest control (named above the line)")],
           loc="upper left", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.06, 0.99))
fig.text(0.02, 0.0, "Stars: significantly above (black) or below (rust) the dashed line.",
         ha="left", va="top", fontsize=8.5, color=MUTED)
fig.savefig(OUT)

with open(OUT.with_suffix(".csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["panel_table", "arm", "vec", "mean", "ci95_half", "strongest_control", "p_holm_vs_strongest_control"])
    w.writerows(rows)
print(OUT)
