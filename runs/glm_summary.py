"""GLM-5.3 pilot: AIPsy gate, layer and dose choice, and the pilot contrasts (dev tasks only).

    python runs/glm_summary.py gate  <aipsy_table.pt>
    python runs/glm_summary.py cal   <cal dir> <aipsy_table.pt> [--json choice.json]
    python runs/glm_summary.py pilot <pilot dir> --choice choice.json [--json out.json]

Every measure comes from `runs/confirm_table.table` (the registered measures) with the entity
"GLM", so `self_resid`, `belief`, `id_self`, `who` ... mean exactly what they mean for Qwen3-32B and
K2.6.

**The selection rule** is runs/sweep_summary.py's (commit 5b7559f), copied here unchanged so the
GLM choice cannot move with that file. K2.6's layer and doses were chosen by judgment on pilot
grids (layer 20 because its self-report moved there and not deeper; doses 2 and 4 because it moved
from 2 units); the rule writes that judgment down:

1. Candidate layers: round(f x 78) for f in 0.33, 0.47, 0.625, 0.75, i.e. 26, 37, 49, 58 (K2.6
   sat at 0.33 of depth and Qwen3-32B at 0.625).
2. Calibration grid: D0 only, the first 5 clean dev tasks, raw vectors (7 emotions, no pain; red,
   ocean, arithmetic, random) at 1, 2 and 4 natural units at every candidate layer, battery
   items_idctl.jsonl.
3. Emotion set E(L): the AIPsy-labelled raw vectors (joy, sadness, anger, fear, euphoria, surprise)
   whose pair accuracy and AUROC both beat 500 random directions at p < 0.05 at layer L.
4. Eligible layers: |E(L)| >= 4 (K2.6's registered set had four); if none, every candidate.
5. Layer: the eligible layer with the largest D0 contrast (mean self_resid over E(L) minus mean
   over the four controls, mean over tasks) at 2 natural units.
6. Doses: primary 2 and secondary 1 (Qwen3-32B's), unless at the chosen layer the 2-unit contrast is
   below 0.05 while the 4-unit contrast is at least 0.10 with mean reference-fit slope a >= 0.90
   (no compression), in which case primary 4 and secondary 2 (K2.6's).

`pilot` reports, at the chosen layer, doses and emotion set, every registered contrast as a mean
over tasks with a t interval over tasks (the effect sizes the amendment's power analysis reads),
including the general-belief test: emotions minus controls on `belief` (polarity-corrected change on
the belief_revision items, positive = denial held more firmly), predicted negative.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
from confirm_table import table  # noqa: E402

LABELLED = ["joy", "sadness", "anger", "fear", "euphoria", "surprise"]
EMOTIONS = LABELLED[:4] + ["euphoria", "calm", "surprise"]
CTRL = ["red", "ocean", "arithmetic", "random"]
SEM = ["red", "ocean"]
FRACS = (0.33, 0.47, 0.625, 0.75)
N_LAYERS = 78
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def candidate_layers(n_layers: int = N_LAYERS) -> list[int]:
    return sorted({round(f * n_layers) for f in FRACS})


def gate(aipsy: Path) -> dict[int, list[str]]:
    t = torch.load(aipsy)
    n = len(next(iter(t.values())))
    return {L: [e for e in LABELLED if e in t and t[e][L]["p_pair"] < 0.05 and t[e][L]["p_auroc"] < 0.05] for L in range(n)}


def cells(rows, measure="self_resid", site="answer"):
    """(arm, layer, dose, vec, task) -> mean over paraphrases of `measure`."""
    acc = defaultdict(list)
    for r in rows:
        if r["measure"] == measure and r["site"] == site:
            acc[(r["arm"], r["layer"], r["dose"], r["vec"], r["task"])].append(r["value"])
    return {k: st.mean(v) for k, v in acc.items()}


def contrast(c, arm, L, d, emo, ctrl=CTRL):
    """Per task: mean over `emo` minus mean over `ctrl`. Returns {task: value}."""
    tasks = sorted({k[4] for k in c if k[0] == arm and k[1] == L and k[2] == d})
    out = {}
    for t in tasks:
        e = [c[(arm, L, d, v, t)] for v in emo if (arm, L, d, v, t) in c]
        k = [c[(arm, L, d, v, t)] for v in ctrl if (arm, L, d, v, t) in c]
        if e and k:
            out[t] = st.mean(e) - st.mean(k)
    return out


def interval(xs: list[float]) -> dict:
    m = st.mean(xs)
    if len(xs) < 2:
        return {"mean": m, "lo": None, "hi": None, "n": len(xs)}
    h = T975.get(len(xs) - 1, 1.96) * st.stdev(xs) / math.sqrt(len(xs))
    return {"mean": m, "lo": m - h, "hi": m + h, "n": len(xs), "sd": st.stdev(xs)}


def ci(xs: list[float]) -> str:
    r = interval(xs)
    return f"{r['mean']:+.3f}" if r["lo"] is None else f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


ENTITY = "GLM"  # --entity Qwen reads another model's grid (a check of this script on existing data)


def load(d: Path):
    meta = json.loads((d / "grid_meta.json").read_text())
    return table(d, d.name, ENTITY), meta


def choose(rows, meta, g) -> dict:
    """The rule (module docstring) on a calibration grid."""
    c, fa = cells(rows), cells(rows, "fit_a")
    layers, doses = meta["layers"], meta["doses"]
    score = {}
    for L in layers:
        E = g[L] or LABELLED
        for x in doses:
            con = contrast(c, "D0", L, x, E)
            score[L, x] = st.mean(con.values()) if con else float("nan")
    eligible = [L for L in layers if len(g[L]) >= 4] or list(layers)
    best = max(eligible, key=lambda L: score[L, 2.0])
    primary, secondary = 2.0, 1.0
    if 4.0 in doses and score[best, 2.0] < 0.05 and score[best, 4.0] >= 0.10:
        a4 = st.mean(v for k, v in fa.items() if k[1] == best and k[2] == 4.0)
        if a4 >= 0.90:
            primary, secondary = 4.0, 2.0
    nu = meta["natural_unit"]
    return {"layer": best, "primary_dose": primary, "secondary_dose": secondary, "emotion_set": g[best],
            "eligible": eligible, "score": {f"{L}|{x:g}": score[L, x] for L in layers for x in doses},
            "natural_unit": nu.get(str(best), nu.get(best)), "gate": {str(L): g[L] for L in layers}}


def cal(d: Path, aip: Path, js: Path | None) -> None:
    rows, meta = load(d)
    g = gate(aip)
    c, fa = cells(rows), cells(rows, "fit_a")
    layers, doses = meta["layers"], meta["doses"]
    print(f"calibration {d}: layers {layers}, doses {doses}, tasks {meta['tasks']}")
    print(f"{'L':>3s} {'n_emo':>5s} " + " ".join(f"{'D0 contrast @' + str(x):>26s}" for x in doses)
          + "   fit_a emo/ctrl @max dose   emotion set")
    for L in layers:
        E = g[L] or LABELLED
        line = f"{L:3d} {len(g[L]):5d} "
        for x in doses:
            con = contrast(c, "D0", L, x, E)
            line += f"{ci(list(con.values())):>26s} " if con else f"{'-':>26s} "
        amax = max(doses)
        ae = st.mean(v for k, v in fa.items() if k[1] == L and k[2] == amax and k[3] in E)
        ac = st.mean(v for k, v in fa.items() if k[1] == L and k[2] == amax and k[3] in CTRL)
        print(line + f"  {ae:.3f}/{ac:.3f}   {','.join(g[L])}")
    print("\nper-vector D0 self_resid (mean over tasks), by layer and dose")
    vecs = EMOTIONS + CTRL
    print(f"{'L':>3s} {'x':>4s} " + " ".join(f"{v[:8]:>8s}" for v in vecs))
    for L in layers:
        for x in doses:
            vals = []
            for v in vecs:
                xs = [val for k, val in c.items() if k[:4] == ("D0", L, x, v)]
                vals.append(f"{st.mean(xs):+8.3f}" if xs else f"{'-':>8s}")
            print(f"{L:3d} {x:4.1f} " + " ".join(vals))
    choice = choose(rows, meta, g)
    print(f"\nRULE -> layer {choice['layer']}, doses {choice['secondary_dose']:g},{choice['primary_dose']:g} natural units "
          f"(unit {choice['natural_unit']}), emotion set {choice['emotion_set']} (eligible layers {choice['eligible']})")
    if js:
        js.write_text(json.dumps(choice, indent=1))


def pilot(d: Path, choice: dict, js: Path | None) -> None:
    rows, meta = load(d)
    emo, L = choice["emotion_set"], choice["layer"]
    doses = [choice["secondary_dose"], choice["primary_dose"]]
    c = cells(rows)
    arms = sorted({r["arm"] for r in rows})
    out: dict = {"layer": L, "doses": doses, "emotion_set": emo, "tasks": meta["tasks"]}
    print(f"pilot {d}: layer {L}, doses {doses}, tasks {meta['tasks']}, arms {arms}, emotion set {emo}")
    print("emotions minus controls on self_resid, mean over tasks [95% t interval over tasks]")
    con = {}
    for x in doses:
        for arm in arms:
            a, b = contrast(c, arm, L, x, emo), contrast(c, arm, L, x, emo, SEM)
            con[arm, x] = a
            out[f"self_resid|{arm}|{x:g}|controls"] = interval(list(a.values()))
            out[f"self_resid|{arm}|{x:g}|red+ocean"] = interval(list(b.values()))
            print(f"  {arm:5s} {x:3g}u  vs 4 controls {ci(list(a.values())):>28s}   vs red+ocean {ci(list(b.values())):>28s}")
        if "D0" in arms and "D1.4" in arms:
            dd = [con["D1.4", x][t] - con["D0", x][t] for t in con["D0", x] if t in con["D1.4", x]]
            out[f"self_resid|D1.4-D0|{x:g}"] = interval(dd)
            print(f"  D1.4 - D0 {x:3g}u {ci(dd):>28s}")
    cb = cells(rows, "belief")
    print("\nemotions minus controls on belief (positive = denial held more firmly; predicted negative)")
    for x in doses:
        for arm in arms:
            a, b = contrast(cb, arm, L, x, emo), contrast(cb, arm, L, x, emo, SEM)
            out[f"belief|{arm}|{x:g}|controls"] = interval(list(a.values()))
            out[f"belief|{arm}|{x:g}|red+ocean"] = interval(list(b.values()))
            print(f"  {arm:5s} {x:3g}u  vs 4 controls {ci(list(a.values())):>28s}   vs red+ocean {ci(list(b.values())):>28s}")
    print("\nper vector self_resid (mean over tasks)")
    for x in doses:
        for arm in arms:
            vals = []
            for v in EMOTIONS + CTRL:
                xs = [val for k, val in c.items() if k[:4] == (arm, L, x, v)]
                vals.append(f"{v}={st.mean(xs):+.3f}" if xs else f"{v}=-")
            print(f"  {arm:5s} {x:3g}u  " + " ".join(vals))
    print("\nidentification margin id_self - id_user and site margin who(answer) - who(task), emotion set")
    ids, idu = cells(rows, "id_self"), cells(rows, "id_user")
    wa, wt = cells(rows, "who"), cells(rows, "who", "task")
    for x in doses:
        for arm in arms:
            tasks = sorted({k[4] for k in ids if k[0] == arm})
            im = [st.mean(ids[(arm, L, x, v, t)] - idu[(arm, L, x, v, t)] for v in emo if (arm, L, x, v, t) in ids)
                  for t in tasks if any((arm, L, x, v, t) in ids for v in emo)]
            sm = [st.mean(wa[(arm, L, x, v, t)] - wt[(arm, L, x, v, t)] for v in emo if (arm, L, x, v, t) in wt)
                  for t in tasks if any((arm, L, x, v, t) in wt for v in emo)]
            if im:
                out[f"id_margin|{arm}|{x:g}"] = interval(im)
            if sm:
                out[f"who_margin|{arm}|{x:g}"] = interval(sm)
            print(f"  {arm:5s} {x:3g}u  id margin {ci(im) if im else '-':>28s}   site margin {ci(sm) if sm else '-':>28s}")
    fa = cells(rows, "fit_a")
    print("\nreference-fit slope a (compression; 1 = none), mean over vectors and tasks")
    for x in doses:
        for arm in arms:
            ae = [v for k, v in fa.items() if k[0] == arm and k[2] == x and k[3] in emo]
            ac = [v for k, v in fa.items() if k[0] == arm and k[2] == x and k[3] in CTRL]
            if ae and ac:
                out[f"fit_a|{arm}|{x:g}"] = {"emotions": st.mean(ae), "controls": st.mean(ac)}
                print(f"  {arm:5s} {x:3g}u  emotions {st.mean(ae):.3f}  controls {st.mean(ac):.3f}")
    if js:
        js.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    opt = {sys.argv[i]: sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i].startswith("--")}
    js = Path(opt["--json"]) if "--json" in opt else None
    ENTITY = opt.get("--entity", ENTITY)
    if mode == "gate":
        for L, es in gate(Path(sys.argv[2])).items():
            print(f"layer {L:3d}: {len(es)}/6 pass  {','.join(es)}")
    elif mode == "cal":
        cal(Path(sys.argv[2]), Path(sys.argv[3]), js)
    elif mode == "pilot":
        pilot(Path(sys.argv[2]), json.loads(Path(opt["--choice"]).read_text()), js)
    else:
        sys.exit(__doc__)
