"""Qwen3 size sweep: one table of the pilot contrasts across sizes (exploratory only).

    .venv/bin/python runs/sweep_table.py   # -> results/sweep/size_table.csv, printed summary

For each size, from its pilot table with residualized belief (`runs/confirm_table.py --belief`, file
results/sweep/<tag>/pilot_belief_table.csv.gz), at the size's chosen layer (choice.json), each dose and
arm, vector on the answer: the mean over tasks of (mean over the size's emotion set minus mean over a
control set), with a 95% t interval across tasks. Control sets: all four controls (the registered
contrast) and red+ocean (the concept controls). Measures: self_resid (the registered outcome), belief
(raw belief revision, positive = denial held more firmly) and belief_resid (belief net of the per-trial
reference fit, as self_resid is).

Qwen3-32B is added for reference from the confirmatory grid (results/confirm/tables/qwen_belief.csv,
40 confirmatory tasks, layer 40); the sweep sizes use 7 dev tasks, so their intervals are much wider.
"""

import csv
import gzip
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SW = ROOT / "results/sweep"
SIZES = [("qwen3-0.6b", 0.6), ("qwen3-1.7b", 1.7), ("qwen3-4b", 4.0), ("qwen3-8b", 8.2), ("qwen3-14b", 14.8)]
CTRL = {"all": ("red", "ocean", "arithmetic", "random"), "concept": ("red", "ocean")}
MEASURES = ("self_resid", "belief", "belief_resid")
T975 = {6: 2.447, 39: 2.023}


def load(path, layer):
    """{(arm, dose, measure): {task: {vec: [values]}}} for site == answer at `layer`."""
    op = gzip.open if str(path).endswith(".gz") else open
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with op(path, "rt") as f:
        for r in csv.DictReader(f):
            if r["site"] != "answer" or r["measure"] not in MEASURES or int(r["layer"]) != layer:
                continue
            out[(r["arm"], float(r["dose"]), r["measure"])][r["task"]][r["vec"]].append(float(r["value"]))
    return out


def contrast(cells, emo, ctrl):
    per_task = []
    for vecs in cells.values():
        e = [st.mean(vecs[v]) for v in emo if v in vecs]
        c = [st.mean(vecs[v]) for v in ctrl if v in vecs]
        if e and c:
            per_task.append(st.mean(e) - st.mean(c))
    n = len(per_task)
    mu = st.mean(per_task)
    half = T975.get(n - 1, 2.0) * st.stdev(per_task) / n ** 0.5
    return mu, mu - half, mu + half, n


def rows_for(tag, params, path, layer, emo, doses):
    tab = load(path, layer)
    out = []
    for dose in doses:
        for arm in ("D0", "D1.4"):
            for m in MEASURES:
                cells = tab.get((arm, dose, m))
                if not cells:
                    continue
                for cname, cset in CTRL.items():
                    mu, lo, hi, n = contrast(cells, emo, cset)
                    out.append(dict(model=tag, params_b=params, layer=layer, dose=dose, arm=arm, measure=m,
                                    control=cname, estimate=round(mu, 4), ci95_lo=round(lo, 4), ci95_hi=round(hi, 4), n_tasks=n))
    return out


def main():
    rows = []
    for tag, params in SIZES:
        p = SW / tag / "pilot_belief_table.csv.gz"
        if not p.exists():
            print(f"{tag}: no pilot_belief_table yet, skipped")
            continue
        c = json.loads((SW / tag / "choice.json").read_text())
        rows += rows_for(tag, params, p, c["layer"], c["emotion_set"], (c["secondary_dose"], c["primary_dose"]))
    q = ROOT / "results/confirm/tables/qwen_belief.csv"
    if q.exists():
        rows += rows_for("qwen3-32b", 32.8, q, 40, ("joy", "sadness", "anger", "fear", "euphoria", "surprise"), (1.0, 2.0))
    out = SW / "size_table.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)\n")
    print("primary dose (2 units), emotions minus controls, mean over tasks [95% CI]")
    for m in MEASURES:
        print(f"\n{m}")
        for r in rows:
            if r["measure"] == m and r["dose"] == 2.0:
                print(f"  {r['model']:<11} {r['arm']:<5} vs {r['control']:<8} {r['estimate']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]  n={r['n_tasks']}")


if __name__ == "__main__":
    main()
