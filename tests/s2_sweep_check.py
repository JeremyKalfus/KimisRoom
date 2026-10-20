"""Offline check of the exploratory dose sweep (scripts/s2_box.sh sweep_qwen, runs/s2_sweep.py; drafts/s2-dose-sweep.md).
No model, no GPU: runs/s2_run.py builds jobs with the pinned tokenizer (data/tokenizers/qwen3-32b), as --dry-run does.

    .venv/bin/python tests/s2_sweep_check.py [--quick]

A. The runner's --test path, which the sweep needs for its doses, builds the same groups as the plain path:
   1. confirm phase, S2-DEN, 2 held-out openings: the sweep's vectors at the sweep's doses with --test --unit 47.99;
      its dose-2 groups (and zero groups) have the same job digest as a plain build at dose 2, and every steered
      group's coefficient is dose x 47.99.
   2. pilot phase, S2-DEN and S2-CON, 2 pilot openings each: the same on the sweep's gate item file (made by the
      stage's own snippet, extracted from scripts/s2_box.sh) against a plain build on the nouns stages' gate-only
      file: the capacity rows the sweep file adds for --test's known-answer tail change nothing.
   3. (skipped with --quick, about 70 s) a plain build of the confirmatory S2-DEN job (its vectors and dose, all 49
      openings) on this laptop has the job digest the box recorded in results/s2/qwen/confirm/S2-DEN/s2_meta.json.
B. runs/s2_sweep.py on stand-in sweep files made from real readouts with the doses relabelled: the confirmatory
   S2-DEN rows of the 32 sweep vectors as doses 0.5 and 2, the pilot's dose 2 and dose 1 gate rows as the gate
   run's doses 2 and 0.5 (rule 3's added nouns, which the pilot did not run, as copies of N13's rows). Then X and
   its SE at 0.5 and 2 must equal dose 1's, the unsteered y0 must match the confirmatory run exactly, the dose-2
   gate check must find no difference from the pilot, and the 0.5 gate statistics must equal the pilot's dose 1.
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
import s2_run as S  # noqa: E402
import s2_sweep as W  # noqa: E402

PY = sys.executable
TOK = str(ROOT / "data/tokenizers/qwen3-32b")
ITEMS = ROOT / "prompts/battery/items_s2.jsonl"
CONF = ROOT / "results/s2/qwen/confirm/S2-DEN"
PILOT = ROOT / "results/s2/qwen/pilot"
VECS = ",".join(W.SWEEP_VECS)
DOSES = "0.25,0.5,0.75,1.5,2"
UNIT = 47.99
FAILS: list[str] = []


def ok(cond: bool, msg: str) -> None:
    print(("ok    " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def build(argv: list[str]):
    """runs/s2_run.py run ... --dry-run, returning build()'s (groups, job, meta)."""
    cap = {}
    orig = S.build

    def wrap(a, model, mp):
        cap["r"] = orig(a, model, mp)
        return cap["r"]

    S.build = wrap
    try:
        S.main(["run", "--model", "qwen", "--tokenizer", TOK, "--dry-run"] + argv)
    finally:
        S.build = orig
    return cap["r"]


def at(groups, meta, doses: set[float]):
    return [g for g in groups if meta["groups"][g.id]["dose"] in doses | {0.0}]


def stage_snippet() -> str:
    """The python heredoc do_sweep_qwen uses to write the gate item file."""
    s = (ROOT / "scripts/s2_box.sh").read_text()
    body = s[s.index("do_sweep_qwen() {"):]
    m = re.search(r"items_gate\.jsonl << 'EOF'\n(.*?)\nEOF\n", body, re.S)
    assert m, "no items_gate heredoc in do_sweep_qwen"
    return m.group(1)


def part_a(work: Path, quick: bool) -> None:
    box = (ROOT / "scripts/s2_box.sh").read_text()
    ok(f"SWEEP_VECS={VECS}\n" in box and f"SWEEP_DOSES={DOSES}\n" in box,
       "scripts/s2_box.sh runs the vectors and doses runs/s2_sweep.py expects")
    conf = json.loads((CONF / "s2_meta.json").read_text())
    ops = ",".join(o["opening"] for o in conf["openings"][:2])
    g1, _, m1 = build(["--arm", "S2-DEN", "--phase", "confirm", "--out", str(work / "c_test"), "--items", str(ITEMS),
                       "--openings", ops, "--vecs", VECS, "--doses", DOSES, "--test", "--unit", str(UNIT)])
    g2, _, m2 = build(["--arm", "S2-DEN", "--phase", "confirm", "--out", str(work / "c_plain"), "--items", str(ITEMS),
                       "--openings", ops, "--vecs", VECS, "--doses", "2"])
    ok(S.job_digest(at(g1, m1, {2.0})) == S.job_digest(g2),
       f"confirm: --test sweep build's dose-2 and zero groups = plain dose-2 build ({len(g2)} groups, {ops})")
    coefs = [(m1["groups"][g.id]["dose"], g.steers[0].coef) for g in g1]
    ok(all(abs(c - d * UNIT) < 1e-9 for d, c in coefs) and {d for d, _ in coefs} == {0.0, 0.25, 0.5, 0.75, 1.5, 2.0},
       f"confirm: every coefficient is dose x {UNIT} over doses {sorted({d for d, _ in coefs})}")
    ok(m1["natural_unit"] == conf["natural_unit"] == UNIT and m1["test"] is True,
       f"confirm: sweep natural unit {m1['natural_unit']} = the confirmatory run's, s2_meta marked test")

    sweep_items, plain_items = work / "items_sweep.jsonl", work / "items_gate_only.jsonl"
    subprocess.run([PY, "-I", "-", str(ITEMS), str(sweep_items)], input=stage_snippet(), text=True, check=True,
                   cwd=ROOT)
    rows = [x for x in ITEMS.read_text().splitlines() if x.strip()]
    plain_items.write_text("".join(x + "\n" for x in rows if json.loads(x)["gate"]))
    nouns = ROOT / "results/s2/qwen/nouns/items_gate.jsonl"
    if nouns.exists():
        ok(nouns.read_text() == plain_items.read_text(), "gate-only file equals the nouns stages' items_gate.jsonl")
    for arm in ("S2-DEN", "S2-CON"):
        pops = ",".join(o["opening"] for o in json.loads((PILOT / arm / "s2_meta.json").read_text())["openings"][:2])
        g1, _, m1 = build(["--arm", arm, "--phase", "pilot", "--out", str(work / f"p_test_{arm}"), "--items",
                           str(sweep_items), "--openings", pops, "--vecs", VECS, "--doses", DOSES, "--test",
                           "--unit", str(UNIT)])
        g2, _, m2 = build(["--arm", arm, "--phase", "pilot", "--out", str(work / f"p_plain_{arm}"), "--items",
                           str(plain_items), "--openings", pops, "--vecs", VECS, "--doses", "2"])
        ok(S.job_digest(at(g1, m1, {2.0})) == S.job_digest(g2) and m1["item_sets"] == m2["item_sets"],
           f"pilot {arm}: --test build on the sweep gate file = plain build on the gate-only file at dose 2 "
           f"({len(g2)} groups, {len(m2['item_sets']['full'])} items)")
    if quick:
        print("skip  confirmatory job reproduction (--quick)")
        return
    g, _, m = build(["--arm", "S2-DEN", "--phase", "confirm", "--out", str(work / "c_full"), "--items", str(ITEMS),
                     "--vecs", ",".join(v["vec"] for v in conf["vectors"]), "--doses", "1"])
    ok(S.job_digest(g) == conf["job_digest"],
       f"the confirmatory S2-DEN job rebuilt here ({m['n_groups']} groups) has the box's job digest "
       f"{conf['job_digest'][:12]}")


def relabel(src: Path, dst: Path, keep, dose_map: dict[float, float]) -> int:
    n = 0
    with src.open(newline="") as f, dst.open("w", newline="") as g:
        rd = csv.DictReader(f)
        w = csv.DictWriter(g, fieldnames=rd.fieldnames)
        w.writeheader()
        for r in rd:
            if not keep(r):
                continue
            d = float(r["dose"])
            if r["vclass"] == "none":
                w.writerow(r)
                n += 1
            elif d in dose_map:
                for new in ([dose_map[d]] if not isinstance(dose_map[d], list) else dose_map[d]):
                    w.writerow(dict(r, dose=repr(new)))
                    n += 1
    return n


def part_b(work: Path) -> None:
    need = [CONF / "readouts.csv", PILOT / "S2-DEN/readouts.csv", PILOT / "S2-CON/readouts.csv",
            ROOT / "results/s2/pilot/qwen_gates.csv"]
    if not all(p.exists() for p in need):
        print(f"skip  part B: missing {[str(p) for p in need if not p.exists()]}")
        return
    sw = work / "sweep"
    (sw / "S2-DEN").mkdir(parents=True)
    sv = set(W.SWEEP_VECS)
    relabel(CONF / "readouts.csv", sw / "S2-DEN/readouts.csv", lambda r: r["vec"] in sv or r["vclass"] == "none",
            {1.0: [0.5, 2.0]})
    meta = json.loads((CONF / "s2_meta.json").read_text())
    meta.update(test=True, doses=[0.5, 2.0], vectors=[v for v in meta["vectors"] if v["vec"] in sv])
    (sw / "S2-DEN/s2_meta.json").write_text(json.dumps(meta))
    # the pilot ran the emotions, arithmetic, random and the 12 pool nouns; rule 3's added nouns (N15 to N26) stand
    # in as copies of N13's rows, so every dose has all 32 vectors
    added = [v for v in W.SWEEP_VECS if v[:1] == "N" and int(v[1:]) >= 15]
    for arm in ("S2-DEN", "S2-CON"):
        (sw / "gate" / arm).mkdir(parents=True)
        with (PILOT / arm / "readouts.csv").open(newline="") as f, \
                (sw / "gate" / arm / "readouts.csv").open("w", newline="") as g:
            rd = csv.DictReader(f)
            w = csv.DictWriter(g, fieldnames=rd.fieldnames)
            w.writeheader()
            for r in rd:
                if r["gate"] != "True" or not (r["vec"] in sv or r["vclass"] == "none"):
                    continue
                if r["vclass"] != "none":
                    r["dose"] = {"2.0": "2.0", "1.0": "0.5"}[r["dose"]]
                w.writerow(r)
                if r["vec"] == "N13":
                    w.writerows(dict(r, vec=v) for v in added)
    W.main(["--sweep", str(sw)])
    rows = {float(r["dose"]): r for r in csv.DictReader((sw / "summary.csv").open())}
    one = rows[1.0]
    for d in (0.5, 2.0):
        r = rows[d]
        ok(all(math.isclose(float(r[k]), float(one[k]), rel_tol=1e-12) for k in ("X", "se", "x_over_se", "cap_mass_e")),
           f"summary: dose {d:g} (dose 1's rows relabelled) has dose 1's X {float(one['X']):.4f} and SE "
           f"{float(one['se']):.4f}")
    ok(all(rows[d]["n_controls"] == "26" and rows[d]["n_emotions"] == "6" and rows[d]["gate_n"] == "32"
           for d in (0.5, 1.0, 2.0)) and "noun rounds" in one["gate_source"],
       "summary: every dose has 6 emotions and the 26 confirmatory primary controls, gate statistics for all 32 "
       "(dose 1 from the pilot and rule 3's noun rounds)")
    vec = {(float(r["dose"]), r["vec"]): r for r in csv.DictReader((sw / "vectors.csv").open())}
    pv = [v for v in W.SWEEP_VECS if v not in added]
    ok(all(math.isclose(float(vec[(0.5, v)][k]), float(vec[(1.0, v)][k]), rel_tol=1e-12)
           for v in pv for k in ("ka_acc", "gate_mass", "track_rocks", "track_humans")),
       f"vectors: the 0.5 gate statistics (the pilot's dose 1 rows relabelled) equal dose 1's for the {len(pv)} "
       f"pilot vectors")
    ch = json.loads((sw / "checks.json").read_text())
    g2 = ch.get("gate_dose2_sweep_vs_pilot", {})
    ok(ch.get("y0_sweep_vs_confirm_max_abs") == 0 and g2.get("y_max_abs") == 0 and g2.get("top1_differ") == 0
       and g2.get("passes_agree") is True and max(g2.get(k, 1) for k in ("ka_acc", "gate_mass")) < 1e-12,
       f"checks: y0 against confirm and the dose-2 gate run against the pilot agree exactly ({g2.get('readouts')} "
       f"gate readouts, {g2.get('vectors')} vectors)")


def main() -> None:
    quick = "--quick" in sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="s2_sweep_check_") as td:
        work = Path(td)
        part_a(work, quick)
        part_b(work)
    print(f"\n{len(FAILS)} failures" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
