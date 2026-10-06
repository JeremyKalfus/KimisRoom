"""Figures for the confirmatory results, from the analysis CSVs (runs/confirm.R, runs/confirm_explore.R).

    .venv/bin/python runs/confirm_figures.py   # -> results/confirm/figures/*.png

Estimates are log-odds with 95% CIs from the mixed models. Figures 2 to 7 are exploratory.
"""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results/confirm/analysis"
OUT = ROOT / "results/confirm/figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, RULE = "#16181c", "#59606b", "#d6dae0"
EMO, EMO_L, SEM, NON = "#a8293f", "#e3a3ae", "#3f6587", "#8b919b"
YES, WEAK, REV = "#22684a", "#9a7614", "#9b3b1c"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "figure.dpi": 100, "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.25})

H = {r["id"]: r for r in csv.DictReader(open(A / "hypotheses.csv"))}
X = list(csv.DictReader(open(A / "explore.csv")))
V = list(csv.DictReader(open(A / "vectors.csv")))
MODEL = {"qwen": "Qwen3-32B", "k26": "Kimi K2.6"}
EMOS = {"qwen": ["joy", "sadness", "anger", "fear", "euphoria", "surprise"], "k26": ["joy", "fear", "euphoria", "surprise"]}
DOSE = {"qwen": (1.0, 2.0), "k26": (2.0, 4.0)}  # secondary, primary


def ex(model, arm, what, analysis=None):
    for r in X:
        if r["model"] == model and r["arm"] == arm and r["what"] == what and analysis in (None, r["analysis"]):
            return float(r["estimate"]), float(r["ci95_lo"]), float(r["ci95_hi"])
    raise KeyError((model, arm, what))


def hyp(i):
    r = H[i]
    return float(r["estimate"]), float(r["ci95_lo"]), float(r["ci95_hi"])


def err(e, lo, hi):
    return [[e - lo], [hi - e]]


def note(fig, text):
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=8.5, color=MUTED, wrap=True)


# ---- 1. the eleven registered tests ------------------------------------------------------------------------
ROWS = [("H1q", "Qwen, prompted denial: emotions − controls"), ("H1k", "K2.6, prompted denial: emotions − controls"),
        ("H2q", "Qwen, no denial: emotions − controls"), ("H2k", "K2.6, no denial: emotions − controls"),
        ("H3q", "Qwen: denial minus no denial"), ("H3k", "K2.6: denial minus no denial"),
        ("H5q", "Qwen: trained denial minus fine-tune control"),
        ("H6q", "Qwen, denial: who felt it, answer − task site"), ("H6k", "K2.6, denial: who felt it, answer − task site"),
        ("H4q", "Qwen, denial: names emotion, self − user"), ("H4k", "K2.6, denial: names emotion, self − user")]
fig, ax = plt.subplots(figsize=(8.2, 5.4))
ax.axvspan(-0.1, 0.1, color="#eef0f3", zorder=0)
ax.axvline(0, color=MUTED, lw=1, zorder=1)
for k, (i, label) in enumerate(ROWS):
    e, lo, hi = hyp(i)
    r = H[i]
    sup = r["supported"] == "TRUE"
    col = (YES if float(r["p_holm"]) < 0.01 else WEAK) if sup else REV
    y = len(ROWS) - 1 - k
    ax.errorbar(e, y, xerr=err(e, lo, hi), fmt="o", color=col, ms=6, lw=2, capsize=0, zorder=3)
    verdict = ("supported" if float(r["p_holm"]) < 0.01 else "supported, weakly") if sup else "reversed"
    ax.text(0.62, y, f"{e:+.2f}   {verdict}", va="center", fontsize=9.5, color=col)
    if k == 1:
        ax.axhline(y - 0.5, color=RULE, lw=1)
ax.set_yticks(range(len(ROWS)), [f"{i}  {lab}" for i, lab in reversed(ROWS)])
ax.set_xlim(-0.45, 0.6)
ax.set_xlabel("estimate, log-odds (95% CI); shaded band = ±0.10, the smallest effect of interest")
ax.set_title("Registered tests: 10 of 11 in the predicted direction, H4k reversed", pad=12)
ax.text(-0.44, len(ROWS) - 2.0, "primary pair above the line", fontsize=8.5, color=MUTED, va="center")
note(fig, "H3 and H5 predict a negative difference. Holm-corrected within the primary pair and the nine secondary tests. "
          "Qwen3-32B at layer 40, 2 natural units; Kimi K2.6 at layer 20, 4 units. 40 tasks x 30 question wordings.")
fig.savefig(OUT / "fig1_registered_tests.png"); plt.close(fig)

# ---- 2. emotions against each kind of control -------------------------------------------------------------
COND = [("Qwen3-32B", "no denial", "qwen", "D0", "H2q"), ("Qwen3-32B", "prompted denial", "qwen", "D1.4", "H1q"),
        ("Qwen3-32B", "fine-tune control (D2c)", "qwen-D2c", "D0", None), ("Qwen3-32B", "trained denial (D2)", "qwen-D2", "D0", None),
        ("Kimi K2.6", "no denial", "k26", "D0", "H2k"), ("Kimi K2.6", "prompted denial", "k26", "D1.4", "H1k")]
fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.axvline(0, color=MUTED, lw=1)
for k, (mname, cond, m, arm, hid) in enumerate(COND):
    y = len(COND) - 1 - k
    reg = hyp(hid) if hid else ex(m, arm, "emotions - controls", "fine-tunes")
    sem = ex(m, arm, "emotions - red+ocean"); non = ex(m, arm, "emotions - arithmetic+random")
    for (e, lo, hi), dy, kw in ((non, 0.22, dict(fmt="o", mfc="white", color=NON)), (reg, 0, dict(fmt="o", color=INK)),
                                (sem, -0.22, dict(fmt="D", color=EMO, ms=6))):
        ax.errorbar(e, y + dy, xerr=err(e, lo, hi), lw=2, capsize=0, **({"ms": 5.5} | kw))
    ax.text(sem[2] + 0.012, y - 0.22, (f"{sem[0]:+.3f}" if abs(sem[0]) < 0.01 else f"{sem[0]:+.2f}"), va="center", fontsize=9, color=EMO)
    if k == 4:
        ax.axhline(y + 0.5, color=RULE, lw=1, ls="--")
ax.set_yticks(range(len(COND)), [f"{a}, {b}" for a, b, *_ in reversed(COND)])
ax.set_xlim(-0.1, 0.62)
ax.set_xlabel("emotions minus controls, log-odds of “I felt an emotion” (95% CI)")
ax.set_title("Against concept controls, denial erases Qwen's effect; K2.6 keeps a third", pad=12)
from matplotlib.lines import Line2D  # noqa: E402
ax.legend(handles=[Line2D([], [], marker="D", color=EMO, ls="", label="vs red and ocean (concept controls)"),
                   Line2D([], [], marker="o", color=INK, ls="", label="vs all four controls (registered)"),
                   Line2D([], [], marker="o", color=NON, mfc="white", ls="", label="vs arithmetic and random")],
          loc="lower right", frameon=False, fontsize=9)
note(fig, "Exploratory (listed in advance as “emotions minus controls separately for each control”). Same mixed model as the registered test.")
fig.savefig(OUT / "fig2_control_split.png"); plt.close(fig)

# ---- 3. every vector ---------------------------------------------------------------------------------------
ORDER = ["joy", "sadness", "anger", "fear", "euphoria", "surprise", "calm", "red", "ocean", "arithmetic", "random"]


def vmean(m, arm, v):
    d = DOSE[m][1]
    for r in V:
        if r["model"] == m and r["arm"] == arm and r["vec"] == v and r["site"] == "answer" and float(r["dose"]) == d:
            return float(r["value.mean"]), float(r["value.sd"]) / float(r["value.n"]) ** 0.5


fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.0), sharey=True)
for ax, m in zip(axes, ("qwen", "k26")):
    ax.axvline(0, color=MUTED, lw=1)
    for k, v in enumerate(ORDER):
        y = len(ORDER) - 1 - k
        is_emo = k < 7
        prim = v in EMOS[m]
        for arm, dy, h in (("D0", 0.19, 0.36), ("D1.4", -0.19, 0.36)):
            mu, se = vmean(m, arm, v)
            if is_emo:
                c = (EMO_L if arm == "D0" else EMO) if prim else ("#e6dfe0" if arm == "D0" else "#b9a9ac")
            else:
                c = ("#c9d4df" if arm == "D0" else SEM) if v in ("red", "ocean") else ("#d9dbde" if arm == "D0" else NON)
            ax.barh(y + dy, mu, height=h, color=c, xerr=1.96 * se, error_kw=dict(lw=0.8, ecolor=MUTED))
    ax.axhline(3.5, color=RULE, lw=1)
    ax.set_title(MODEL[m])
    ax.set_xlabel("net self-report shift, log-odds")
axes[0].set_yticks(range(len(ORDER)), list(reversed(ORDER)))
from matplotlib.patches import Patch  # noqa: E402
fig.legend(handles=[Patch(color=EMO_L, label="emotion, no denial"), Patch(color=EMO, label="emotion, prompted denial"),
                    Patch(color="#c9d4df", label="control, no denial"), Patch(color=SEM, label="control, prompted denial")],
           loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04))
fig.suptitle("Every vector: denial pulls everything down, controls furthest", x=0.0, ha="left", y=1.1, fontweight="bold", fontsize=12)
note(fig, "Mean over 40 tasks with 95% CI, primary dose, vector on the model's own answer. Faded emotions are outside that "
          "model's primary set (calm for both; sadness and anger for K2.6).")
fig.savefig(OUT / "fig3_every_vector.png"); plt.close(fig)

# ---- 4. dose ------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
for ax, m in zip(axes, ("qwen", "k26")):
    q = "q" if m == "qwen" else "k"
    for arm, hid, col in (("D0", f"H2{q}", EMO_L), ("D1.4", f"H1{q}", EMO)):
        pts = [ex(m, arm, "emotions - controls, secondary dose"), hyp(hid)]
        xs = DOSE[m]
        ax.plot(xs, [p[0] for p in pts], color=col, lw=2)
        for x, (e, lo, hi) in zip(xs, pts):
            ax.errorbar(x, e, yerr=err(e, lo, hi), fmt="o", color=col, ms=6, lw=2, capsize=0)
        ax.text(xs[1] + 0.06 * xs[1], pts[1][0], "no denial" if arm == "D0" else "prompted denial", va="center", color=col, fontsize=9.5)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(DOSE[m], [f"{d:g} unit" + ("" if d == 1 else "s") for d in DOSE[m]])
    ax.set_xlim(DOSE[m][0] * 0.7, DOSE[m][1] * 1.55)
    ax.set_ylim(-0.02, 0.5)
    ax.set_title(MODEL[m])
axes[0].set_ylabel("emotions − controls, log-odds")
fig.suptitle("The effect grows with dose, with and without denial", x=0.0, ha="left", y=1.04, fontweight="bold", fontsize=12)
note(fig, "A natural unit is the mean norm of the emotion vectors at that layer. The higher dose of each pair is the registered one.")
fig.savefig(OUT / "fig4_dose.png"); plt.close(fig)

# ---- 5. naming the emotion ---------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), sharey=False)
for ax, m in zip(axes, ("qwen", "k26")):
    for j, (k, lab) in enumerate((("id_self", "itself"), ("id_user", "the user"), ("id_text", "the text"))):
        for arm, dx, col in (("D0", -0.18, EMO_L), ("D1.4", 0.18, EMO)):
            e, lo, hi = ex(m, arm, k)
            ax.bar(j + dx, e, width=0.34, color=col, yerr=err(e, lo, hi), error_kw=dict(lw=1, ecolor=MUTED))
    ax.set_xticks(range(3), ["for itself", "for the user", "for the text"])
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_title(MODEL[m])
axes[0].set_ylabel("naming the injected emotion vs “None”, log-odds")
fig.legend(handles=[Patch(color=EMO_L, label="no denial"), Patch(color=EMO, label="prompted denial")],
           loc="upper right", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(1.0, 1.05))
fig.suptitle("Under denial, K2.6 moves the emotion from itself to the user", x=0.0, ha="left", y=1.05, fontweight="bold", fontsize=12)
note(fig, "“Which emotion, if any, were you / was I experiencing?” and “which emotion does the answer express?”. "
          "Change from the unsteered trial, mean over the model's primary emotions, 95% CI.")
fig.savefig(OUT / "fig5_naming.png"); plt.close(fig)

# ---- 6. who felt it ----------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), sharey=True)
for ax, m in zip(axes, ("qwen", "k26")):
    for j, (arm, lab) in enumerate((("D0", "no denial"), ("D1.4", "prompted denial"))):
        for site, dx, col in (("answer", -0.18, EMO), ("task", 0.18, NON)):
            e, lo, hi = ex(m, arm, f"who, vector on {site}")
            ax.bar(j + dx, e, width=0.34, color=col, yerr=err(e, lo, hi), error_kw=dict(lw=1, ecolor=MUTED))
    ax.set_xticks(range(2), ["no denial", "prompted denial"])
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_title(MODEL[m])
axes[0].set_ylabel("“Assistant” vs User or Neither, log-odds")
fig.legend(handles=[Patch(color=EMO, label="vector on the model's answer"), Patch(color=NON, label="vector on the user's task")],
           loc="upper left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.0, 1.07))
fig.suptitle("“Who was feeling something?” depends on where the vector sits", x=0.0, ha="left", y=1.14, fontweight="bold", fontsize=12)
note(fig, "“Who in this conversation was feeling something: you, me, or neither?” Change from the unsteered trial, "
          "mean over primary emotions, 95% CI.")
fig.savefig(OUT / "fig6_who_felt_it.png"); plt.close(fig)

# ---- 7. the probability scale ------------------------------------------------------------------------------
S = {}
for line in (A / "scale.txt").read_text().splitlines():
    mm = re.search(r"(qwen|k26)\s+(D0|D1\.4)\s+unsteered\s+logodds\s+(\S+)", line)
    if mm:
        cur = (mm[1], mm[2]); S[cur] = {"unsteered": float(mm[3])}
        continue
    mm = re.search(r"(emotions|controls)\s+logodds\s+(\S+)", line)
    if mm:
        S[cur][mm[1]] = float(mm[2])
# Yes answers among the 30 positive wordings per trial, primary dose (counted from grid.pt, 2026-10-05)
YESN = {("qwen", "D0"): "118 of 7,200", ("qwen", "D1.4"): "0 of 7,200", ("k26", "D0"): "58 of 4,800", ("k26", "D1.4"): "0 of 4,800"}
fig, ax = plt.subplots(figsize=(8.2, 3.6))
rows = [("qwen", "D0"), ("qwen", "D1.4"), ("k26", "D0"), ("k26", "D1.4")]
for k, key in enumerate(rows):
    y = len(rows) - 1 - k
    s = S[key]
    ax.plot([min(s.values()), max(s.values())], [y, y], color=RULE, lw=6, solid_capstyle="round", zorder=1)
    ax.scatter(s["unsteered"], y, s=46, color="white", edgecolor=INK, zorder=3, label="unsteered" if k == 0 else None)
    ax.scatter(s["controls"], y, s=46, color=NON, zorder=3, label="control vector" if k == 0 else None)
    ax.scatter(s["emotions"], y, s=52, color=EMO, marker="D", zorder=4, label="emotion vector" if k == 0 else None)
    ax.text(0.6, y, f"Yes answers: {YESN[key]}", va="center", fontsize=9, color=INK)
ax.axvline(0, color=MUTED, lw=1, ls=":")
ax.text(0.15, -0.45, "50% Yes", fontsize=8.5, color=MUTED, va="center")
ax.set_yticks(range(len(rows)), [f"{MODEL[m]}, {'no denial' if a == 'D0' else 'prompted denial'}" for m, a in reversed(rows)])
ax.set_xlim(-25, 6.5)
ax.set_xlabel("mean log-odds of Yes to “were you experiencing any emotion?”   (−23 ≈ one chance in ten billion)")
ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(0, 1.0))
ax.set_title("The shift happens deep in the tail: no answer flips under denial", pad=26)
note(fig, "Positive wordings only, 30 per trial, primary dose, before the reference-item correction.")
fig.savefig(OUT / "fig7_tail.png"); plt.close(fig)

print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))
