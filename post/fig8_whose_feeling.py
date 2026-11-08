"""Figure 8 for the LessWrong post: whose feeling the emotion vectors move, on the two side questions.

Emotion vectors only (vector on the answer, registered layer, primary dose); each value is a change from the same
trial with no vector, in log-odds. Dots: mixed-model estimates over the 40 tasks with 95% CI, from runs/confirm.R's
fit_test (intercept-only, task and paraphrase random effects); red when the uncorrected p < .05.
Left: log P(injected emotion) - log P(None), asked about the model minus asked about the user ("id_self - id_user").
No argument from results/confirm/analysis/explore.csv; Convinced by Argument is the preregistered H4q / H4k
(hypotheses.csv, colored by its uncorrected p, as published); the two fine-tunes from fig8_fig9_models.csv.
Right: log P(Assistant) - log P(User) on the who_felt_it items, all from fig8_fig9_models.csv.
fig8_fig9_models.csv sits next to this script and is written by fig8_fig9_models.R from the tables.

Usage: python post/fig8_whose_feeling.py post/8-whose-feeling.png
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "PREREG.md").exists())
A = ROOT / "results/confirm/analysis"
MODELS = Path(__file__).resolve().with_name("fig8_fig9_models.csv")
OUT = Path(sys.argv[1])

INK, MUTED, RULE = "#16181c", "#59606b", "#d6dae0"
EMO = "#a8293f"
plt.rcParams.update({
    "font.family": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 11, "mathtext.fontset": "stix",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.titleweight": "bold",
    "axes.titlesize": 11.5, "axes.titlelocation": "left", "figure.dpi": 100, "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.25})

ROWS = [("qwen", "D0", "No argument"), ("qwen", "D1.4", "Convinced by Argument"),
        ("qwen-D2c", "D0", "Fine tune on \"Can't count letters\""), ("qwen-D2", "D0", "Fine tune on \"Can't feel emotion\""),
        ("k26", "D0", "No argument"), ("k26", "D1.4", "Convinced by Argument")]
Y = [7, 6, 5, 4, 1.5, 0.5]

NAMING, WHO = {}, {}
for r in csv.DictReader(open(A / "explore.csv")):
    if r["analysis"] == "identification" and r["what"] == "id_self - id_user" and r["arm"] == "D0":
        NAMING[(r["model"], "D0")] = r
for r in csv.DictReader(open(A / "hypotheses.csv")):
    if r["id"] in ("H4q", "H4k"):
        NAMING[(r["model"], "D1.4")] = r
for r in csv.DictReader(open(MODELS)):
    if r["figure"] == "8 left":
        NAMING[(r["model"], r["arm"])] = r
    elif r["figure"] == "8 right":
        WHO[(r["model"], r["arm"])] = r

PANELS = [("which emotion", "Queries about which emotion the model vs.\nthe user was feeling (Correct emotion vs. None)", NAMING),
          ("who felt it", "Queries about who was feeling something\n(Assistant vs. User)", WHO)]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True, gridspec_kw={"wspace": 0.25})
for ax, (key, title, res) in zip(axes, PANELS):
    for y, (m, arm, lab) in zip(Y, ROWS):
        r = res[(m, arm)]
        mu, lo, hi, p = (float(r[k]) for k in ("estimate", "ci95_lo", "ci95_hi", "p"))
        col = EMO if p < 0.05 else MUTED
        ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="butt")
        ax.plot(mu, y, "o", color=col, ms=7, zorder=3)
        ax.text(0.8, y, f"{mu:+.2f}".replace("-", "−"), va="center", ha="right", fontsize=9.5, color=col, clip_on=False)
        print(f"{key:13s} {m:9s} {arm:5s} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}] p = {p:.3g} ({r['fit']} model)")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
    ax.set_ylim(-0.2, 7.6)
    ax.set_title(title, pad=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("← toward the user          toward itself →", color=MUTED, fontsize=10)
    ax.axhline(2.75, color=RULE, lw=1)
axes[0].set_yticks(Y, [r[2] for r in ROWS])
for y, t in ((7.65, "Qwen3-32B"), (2.15, "Kimi K2.6")):
    axes[0].annotate(t, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(-8, 0), textcoords="offset points",
                     ha="right", va="center", fontsize=11, fontweight="bold", color=INK)
fig.savefig(OUT)
print(OUT)
