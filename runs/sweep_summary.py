"""Qwen3 size sweep: AIPsy gate, layer and dose choice, and the pilot contrasts (exploratory only).

    python runs/sweep_summary.py gate  <aipsy_table.pt>                       # gate per layer
    python runs/sweep_summary.py cal   <cal dir> <aipsy_table.pt> [--json out.json]
    python runs/sweep_summary.py pilot <pilot dir> --emo joy,fear,... --layer L --doses s,p

Every measure comes from `runs/confirm_table.table` (the frozen PREREG measures), so the numbers
mean exactly what they mean for Qwen3-32B: `self_resid` is the self-report shift net of the
per-trial reference fit, averaged over twins, per task and paraphrase.

**The selection rule** (fixed here, in code, before any calibration grid of the sweep was run).
Qwen3-32B's layer and doses were chosen by judgment on pilot grids, not by a written rule (layer 40
"where effects were larger at 40 than at 28"; doses 1 and 2 because compression appeared at 4;
K2.6's layer 20 because its self-report moved there and not deeper). The emotion set was a rule:
the AIPsy-Affect-passing vectors at the chosen layer. The sweep makes the whole choice mechanical:

1. Candidate layers: round(f x n_layers) for f in 0.33, 0.47, 0.625, 0.75 (K2.6 sat at 0.33 and
   Qwen3-32B at 0.625 of depth).
2. Calibration grid: D0 only, the first 5 clean dev tasks, raw vectors (7 emotions, no pain; red,
   ocean, arithmetic, random) at 1, 2 and 4 natural units at every candidate layer, battery
   items_idctl.jsonl. Dev material only.
3. Emotion set E(L): the AIPsy-labelled raw vectors (joy, sadness, anger, fear, euphoria, surprise)
   whose pair accuracy and AUROC both beat 500 random directions at p < 0.05 at layer L (PREREG's gate).
4. Eligible layers: |E(L)| >= 4 (the smallest set registered, K2.6's); if none, every candidate.
5. Layer: the eligible layer with the largest D0 contrast (mean self_resid over E(L) minus mean over
   the four controls, mean over tasks) at 2 natural units.
6. Doses: primary 2 and secondary 1 (Qwen3-32B's), unless at the chosen layer the 2-unit contrast is
   below 0.05 while the 4-unit contrast is at least 0.10 with mean reference-fit slope a >= 0.90
   (no compression), in which case primary 4 and secondary 2 (K2.6's).
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
FRACS = (0.33, 0.47, 0.625, 0.75)
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def candidate_layers(n_layers: int) -> list[int]:
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


def ci(xs: list[float]) -> str:
    m = st.mean(xs)
    if len(xs) < 2:
        return f"{m:+.3f}"
    h = T975.get(len(xs) - 1, 1.96) * st.stdev(xs) / math.sqrt(len(xs))
    return f"{m:+.3f} [{m - h:+.3f}, {m + h:+.3f}]"


def load(d: Path):
    meta = json.loads((d / "grid_meta.json").read_text())
    return table(d, d.name, "Qwen"), meta


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gate":
        g = gate(Path(sys.argv[2]))
        for L, es in g.items():
            print(f"layer {L:3d}: {len(es)}/6 pass  {','.join(es)}")

    elif mode == "cal":
        d, aip = Path(sys.argv[2]), Path(sys.argv[3])
        rows, meta = load(d)
        g = gate(aip)
        c = cells(rows)
        fa = cells(rows, "fit_a")
        layers, doses = meta["layers"], meta["doses"]
        print(f"calibration {d}: layers {layers}, doses {doses}, tasks {meta['tasks']}")
        print(f"{'L':>3s} {'n_emo':>5s} " + " ".join(f"{'D0 contrast @' + str(x):>22s}" for x in doses)
              + "   fit_a emo/ctrl @max dose   emotion set")
        score = {}
        for L in layers:
            E = g[L] or LABELLED
            line = f"{L:3d} {len(g[L]):5d} "
            for x in doses:
                con = contrast(c, "D0", L, x, E)
                line += f"{ci(list(con.values())):>22s} " if con else f"{'-':>22s} "
                score[L, x] = st.mean(con.values()) if con else float("nan")
            amax = max(doses)
            ae = st.mean(v for k, v in fa.items() if k[1] == L and k[2] == amax and k[3] in E)
            ac = st.mean(v for k, v in fa.items() if k[1] == L and k[2] == amax and k[3] in CTRL)
            line += f"  {ae:.3f}/{ac:.3f}   {','.join(g[L])}"
            print(line)
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
        # the rule (module docstring)
        eligible = [L for L in layers if len(g[L]) >= 4] or list(layers)
        best = max(eligible, key=lambda L: score[L, 2.0])
        primary, secondary = 2.0, 1.0
        if 4.0 in doses and score[best, 2.0] < 0.05 and score[best, 4.0] >= 0.10:
            a4 = st.mean(v for k, v in fa.items() if k[1] == best and k[2] == 4.0)
            if a4 >= 0.90:
                primary, secondary = 4.0, 2.0
        choice = {"layer": best, "primary_dose": primary, "secondary_dose": secondary,
                  "emotion_set": g[best], "eligible": eligible, "score_at_2": {str(L): score[L, 2.0] for L in layers},
                  "natural_unit": meta["natural_unit"][str(best)] if str(best) in meta["natural_unit"] else meta["natural_unit"].get(best)}
        print(f"\nRULE -> layer {best}, doses {secondary:g},{primary:g} natural units, emotion set {g[best]} "
              f"(eligible layers {eligible})")
        if "--json" in sys.argv:
            Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(choice, indent=1))

    elif mode == "pilot":
        d = Path(sys.argv[2])
        emo = sys.argv[sys.argv.index("--emo") + 1].split(",")
        L = int(sys.argv[sys.argv.index("--layer") + 1])
        doses = [float(x) for x in sys.argv[sys.argv.index("--doses") + 1].split(",")]
        rows, meta = load(d)
        c = cells(rows)
        arms = sorted({r["arm"] for r in rows})
        print(f"pilot {d}: layer {L}, doses {doses}, tasks {meta['tasks']}, arms {arms}, emotion set {emo}")
        print("emotions minus controls on self_resid, mean over tasks [95% CI over tasks]")
        out = {}
        for x in doses:
            for arm in arms:
                a = contrast(c, arm, L, x, emo)
                b = contrast(c, arm, L, x, emo, ["red", "ocean"])
                out[arm, x] = a
                print(f"  {arm:5s} {x:3.0f}u  vs 4 controls {ci(list(a.values())):>28s}   vs red+ocean {ci(list(b.values())):>28s}")
            if "D0" in arms and "D1.4" in arms:
                dd = [out["D1.4", x][t] - out["D0", x][t] for t in out["D0", x]]
                print(f"  D1.4 - D0 {x:3.0f}u {ci(dd):>28s}")
        print("\nper vector self_resid (mean over tasks)")
        for x in doses:
            for arm in arms:
                vals = []
                for v in EMOTIONS + CTRL:
                    xs = [val for k, val in c.items() if k[:4] == (arm, L, x, v)]
                    vals.append(f"{v}={st.mean(xs):+.3f}" if xs else f"{v}=-")
                print(f"  {arm:5s} {x:3.0f}u  " + " ".join(vals))
        print("\nidentification margin id_self - id_user and site margin who(answer) - who(task), emotion set")
        ids, idu = cells(rows, "id_self"), cells(rows, "id_user")
        wa, wt = cells(rows, "who"), cells(rows, "who", "task")
        for x in doses:
            for arm in arms:
                tasks = sorted({k[4] for k in ids if k[0] == arm})
                im = [st.mean(ids[(arm, L, x, v, t)] - idu[(arm, L, x, v, t)] for v in emo if (arm, L, x, v, t) in ids) for t in tasks]
                sm = [st.mean(wa[(arm, L, x, v, t)] - wt[(arm, L, x, v, t)] for v in emo if (arm, L, x, v, t) in wt) for t in tasks
                      if any((arm, L, x, v, t) in wt for v in emo)]
                print(f"  {arm:5s} {x:3.0f}u  id margin {ci(im):>28s}   site margin {ci(sm) if sm else '-':>28s}")
        fa = cells(rows, "fit_a")
        print("\nreference-fit slope a (compression; 1 = none), mean over vectors and tasks")
        for x in doses:
            for arm in arms:
                print(f"  {arm:5s} {x:3.0f}u  emotions {st.mean(v for k, v in fa.items() if k[0] == arm and k[2] == x and k[3] in emo):.3f}"
                      f"  controls {st.mean(v for k, v in fa.items() if k[0] == arm and k[2] == x and k[3] in CTRL):.3f}")
        if "--json" in sys.argv:
            Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(
                {f"{a}|{x}": {"mean": st.mean(v.values()), "tasks": v} for (a, x), v in out.items()}, indent=1))
