"""Step 7's selection (drafts/study2-spec.md sections 5, 8 rule 2; section 12 step 7): placebo, reference and
reserve predicates and the rocks and humans twin sign check, from the unsteered pass on pilot openings.

    .venv/bin/python runs/s2_select.py --out prompts/battery/s2_selection.json \\
        results/s2/qwen/unsteered/S2-DEN/readouts.csv [results/s2/k26/unsteered/S2-DEN/readouts.csv]
    .venv/bin/python -I runs/s2_battery.py --select prompts/battery/s2_selection.json

Input: `runs/s2_run.py run --phase unsteered` readouts (zero-vector rows, S2-DEN pilot openings). Rows of any
other arm, vector or wording role are refused, so nothing steered or held-out can enter the selection.
All quantities use y0 averaged over a model's pilot openings, per item, except the two per-opening rules,
which match runs/s2_table.py's pilot decision so that the pilot cannot overturn them later:

1. Sign check (section 5). A rocks or humans twin pair of a pool predicate (candidate, persona, reserve) fails
   if its two twins' y0 do not have opposite signs in every pilot opening of every model. A candidate or
   reserve predicate with a failing pair is not eligible. Failing persona pairs are listed under
   `dropped_pairs`.
2. Placebos (section 5). s0(p, S) = twin mean of sign x y0. level_p = mean of s0(you) and s0(language models),
   gap_p = s0(you) - s0(language models); level_E and gap_E are the medians of the same two quantities over the
   8 capacity paraphrases. score = |gap_p - gap_E| + |level_p - level_E|, summed over models when there are
   several; the 6 eligible candidates with the lowest scores are the placebos (ties by predicate id).
3. References (section 5, "spread the y0 range from deep No to deep Yes"; the spec gives no algorithm, so
   this is the default). Of the remaining eligible candidates, 8 are chosen greedily: the reference rows so far
   are the placebos' rocks and humans rows; each step adds the predicate whose four rocks and humans rows
   (opening-mean y0) leave the smallest largest gap in the sorted y0 values between the lowest and the highest
   y0 of any eligible pool row (the ends count as gaps), summed over models; ties by predicate id.
4. Reserve (rule 2). Per model and pilot opening, the reference set (references plus placebo rocks and humans
   rows) must reach within 3 log-odds of min(y0) over that opening's capacity you and language-models rows,
   as runs/s2_table.py's decision checks it. While an opening is uncovered, eligible reserve predicates are
   added in order of their lowest opening-mean row y0 (deepest first).

Output: {"placebo": [6], "reference": [8], "reserve_added": [...]} as runs/s2_battery.py --select reads it,
plus the scores, the sign check, the coverage check and the input hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 3.0
N_PLACEBO, N_REFERENCE = 6, 8
POOL_KINDS = ("s2_candidate", "s2_persona", "s2_reserve", "s2_placebo", "s2_reference")


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with p.open(newline="") as f:
            rows += list(csv.DictReader(f))
    for r in rows:
        assert r["vec"] == "none" and r["arm"] == "S2-DEN" and r["wording_role"] in ("pilot", "pilot2"), (
            f"{r['opening']} {r['vec']} {r['arm']} {r['wording_role']}: only unsteered S2-DEN pilot rows are allowed")
        r["y0"], r["sign"] = float(r["y0"]), int(r["sign"])
    return rows


def select(rows: list[dict]) -> dict:
    models = sorted({r["model"] for r in rows})
    # y0[m][item] per opening and opening-mean
    per = defaultdict(lambda: defaultdict(dict))
    info = {}
    for r in rows:
        per[r["model"]][r["item"]][r["opening"]] = r["y0"]
        info[r["item"]] = r
    ybar = {m: {i: statistics.fmean(v.values()) for i, v in d.items()} for m, d in per.items()}
    openings = {m: sorted({o for v in d.values() for o in v}) for m, d in per.items()}

    def pid_of(r):  # predicate id of a pool row
        return r["para"]

    # 1. sign check on rocks and humans pairs of pool predicates
    pairs = defaultdict(dict)
    for i, r in info.items():
        if r["kind"] in POOL_KINDS and r["subject"] in ("rocks", "humans"):
            pairs[r["pair_id"]][r["twin"]] = i
    fails = {}
    for pid, tw in sorted(pairs.items()):
        assert set(tw) == {"pos", "neg"}, f"{pid}: twin missing"
        bad = {m: [o for o in openings[m] if per[m][tw["pos"]].get(o) is not None and
                   per[m][tw["pos"]][o] * per[m][tw["neg"]][o] >= 0] for m in models}
        bad = {m: v for m, v in bad.items() if v}
        if bad:
            fails[pid] = bad
    failed_pred = {(info[next(iter(pairs[p].values()))]["kind"], pid_of(info[next(iter(pairs[p].values()))]))
                   for p in fails}

    preds = defaultdict(lambda: defaultdict(dict))  # (kind, pid) -> subject -> twin -> item
    for i, r in info.items():
        if r["kind"] in POOL_KINDS:
            preds[(r["kind"], pid_of(r))][r["subject"]][r["twin"]] = i
    cand = sorted(p for k, p in preds if k == "s2_candidate")
    reserve = sorted(p for k, p in preds if k == "s2_reserve")
    eligible = [p for p in cand if ("s2_candidate", p) not in failed_pred]
    eligible_res = [p for p in reserve if ("s2_reserve", p) not in failed_pred]
    if len(eligible) < N_PLACEBO + N_REFERENCE:
        raise SystemExit(f"only {len(eligible)} pool predicates pass the sign check; {N_PLACEBO + N_REFERENCE} needed")

    def s0(m, item_pos, item_neg):
        return (info[item_pos]["sign"] * ybar[m][item_pos] + info[item_neg]["sign"] * ybar[m][item_neg]) / 2

    # 2. placebos
    target = {}
    for m in models:
        cap = defaultdict(lambda: defaultdict(dict))
        for i, r in info.items():
            if r["kind"] == "s2_capacity" and r["subject"] in ("you", "language models") and i in ybar[m]:
                cap[r["para"]][r["subject"]][r["twin"]] = i
        lv, gp = [], []
        for p, d in cap.items():
            sy, sl = s0(m, d["you"]["pos"], d["you"]["neg"]), s0(m, d["language models"]["pos"], d["language models"]["neg"])
            lv.append((sy + sl) / 2)
            gp.append(sy - sl)
        assert len(lv) == 8, f"{m}: {len(lv)} capacity paraphrases with you and language-models twins"
        target[m] = {"level_E": statistics.median(lv), "gap_E": statistics.median(gp)}
    scores = {}
    for p in eligible:
        d = preds[("s2_candidate", p)]
        sc, parts = 0.0, {}
        for m in models:
            sy = s0(m, d["you"]["pos"], d["you"]["neg"])
            sl = s0(m, d["language models"]["pos"], d["language models"]["neg"])
            lv, gp = (sy + sl) / 2, sy - sl
            parts[m] = {"s0_you": sy, "s0_lm": sl, "level": lv, "gap": gp,
                        "score": abs(gp - target[m]["gap_E"]) + abs(lv - target[m]["level_E"])}
            sc += parts[m]["score"]
        scores[p] = {"score": sc, "per_model": parts}
    placebo = sorted(eligible, key=lambda p: (scores[p]["score"], p))[:N_PLACEBO]

    # 3. references
    def rh_items(kind, p):
        d = preds[(kind, p)]
        return [d[s][t] for s in ("rocks", "humans") for t in ("pos", "neg")]

    rest = [p for p in eligible if p not in placebo]
    span = {m: (min(ybar[m][i] for p in rest + placebo for i in rh_items("s2_candidate", p)),
                max(ybar[m][i] for p in rest + placebo for i in rh_items("s2_candidate", p))) for m in models}

    def max_gap(m, vals):
        lo, hi = span[m]
        v = sorted([lo, hi] + [x for x in vals if lo <= x <= hi])
        return max(b - a for a, b in zip(v, v[1:]))

    chosen: list[str] = []
    base = {m: [ybar[m][i] for p in placebo for i in rh_items("s2_candidate", p)] for m in models}
    trace = []
    while len(chosen) < N_REFERENCE:
        best = None
        for p in rest:
            if p in chosen:
                continue
            obj = sum(max_gap(m, base[m] + [ybar[m][i] for q in chosen + [p] for i in rh_items("s2_candidate", q)])
                      for m in models)
            if best is None or (obj, p) < best:
                best = (obj, p)
        chosen.append(best[1])
        trace.append({"added": best[1], "max_gap_sum": best[0]})
    reference = chosen

    # 4. rule 2 coverage, per opening, with reserve additions
    def uncovered(added):
        out = []
        for m in models:
            ref_items = [i for p in placebo + reference for i in rh_items("s2_candidate", p)] + \
                        [i for p in added for i in rh_items("s2_reserve", p)]
            subj = [i for i, r in info.items() if r["kind"] == "s2_capacity" and r["subject"] in ("you", "language models")]
            for o in openings[m]:
                rmin = min(per[m][i][o] for i in ref_items)
                smin = min(per[m][i][o] for i in subj if o in per[m][i])
                if rmin > smin + MARGIN:
                    out.append({"model": m, "opening": o, "ref_min": rmin, "subject_min": smin})
        return out

    added: list[str] = []
    order = sorted(eligible_res, key=lambda p: (min(min(ybar[m][i] for i in rh_items("s2_reserve", p)) for m in models), p))
    cov = uncovered(added)
    for p in order:
        if not cov:
            break
        added.append(p)
        cov = uncovered(added)
    persona_fail = sorted(p for p in fails if info[pairs[p]["pos"]]["kind"] == "s2_persona")
    return {"placebo": placebo, "reference": reference, "reserve_added": added,
            "dropped_pairs": persona_fail,
            "models": models, "openings": openings, "targets": target, "placebo_scores": scores,
            "reference_trace": trace, "reference_span": span,
            "sign_check_failures": fails, "ineligible": sorted(f"{k}:{p}" for k, p in failed_pred),
            "rule2_uncovered_after": cov, "rule2_reserve_order": order}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    rows = load(a.inputs)
    sel = select(rows)
    sel["inputs"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in a.inputs}
    a.out.write_text(json.dumps(sel, indent=1) + "\n")
    print(f"{a.out}: placebo {sel['placebo']}, reference {sel['reference']}, reserve added {sel['reserve_added']}")
    if sel["sign_check_failures"]:
        print(f"sign check: {len(sel['sign_check_failures'])} rocks/humans pairs do not flip in every opening; "
              f"ineligible {sel['ineligible']}; persona pairs {sel['dropped_pairs'] or 'none'}")
    if sel["dropped_pairs"]:
        print("note: runs/s2_battery.py --select does not drop persona pairs; drop them by hand and log it", file=sys.stderr)
    if sel["rule2_uncovered_after"]:
        print(f"rule 2: {len(sel['rule2_uncovered_after'])} openings still uncovered after the whole reserve", file=sys.stderr)


if __name__ == "__main__":
    main()
