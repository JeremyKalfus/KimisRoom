"""Checks for the Study 2 analysis code on synthetic readouts (tests/s2_synth.py).

    .venv/bin/python tests/s2_table_check.py [--no-r] [--work DIR]

1. Compression: a pure squeeze (every vector shrinks y toward 0 by its own factor, no self effect, no noise)
   gives d = 0 exactly for every vector, while the raw contrast d_raw shows the bias 2 (a_C - a_E) the spec's
   worked example predicts.
2. ID and structure checks: each planted fault makes runs/s2_table.py refuse the table.
3. Pilot gates: a vector that fails rule 1 at the registered dose sends Qwen to half dose.
4. (R) Planted effect in Qwen (beta 0.25), null in K2.6: S2-H1 detects the first and not the second, the
   permutation p agrees with the model p, and the follow-up tests open in order. A noisy pure squeeze gives no
   S2-H1 effect.
5. Descriptive extras: argmax flips, P(Yes) scale, mass drop and wording mentions on hand-set rows (Python), and
   their rows in descriptive.csv, openings.csv, flips.csv and mass.csv, with a synthetic potency file (R); the
   valence rows: per-emotion X averaging to the S2-H1 estimate, and positive minus negative over each model's own
   emotions (K2.6: joy, euphoria vs fear), whose CI covers 0 since no valence effect is planted.
"""

from __future__ import annotations

import copy
import csv
import math
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
import s2_table as T  # noqa: E402

PY = sys.executable
FAILS = []


def ok(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def synth(path: Path, *args: str) -> Path:
    subprocess.run([PY, str(ROOT / "tests/s2_synth.py"), str(path), *args], check=True, capture_output=True)
    return path


def class_means(rows: list[dict], measure: str, model: str = "qwen", arm: str = "S2-DEN") -> dict:
    acc = defaultdict(list)
    for r in rows:
        if r["measure"] == measure and r["model"] == model and r["arm"] == arm:
            acc[r["vclass"]].append(float(r["value"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def refuses(rows: list[dict], phase: str, mutate, expect: str, label: str) -> None:
    rs = copy.deepcopy(rows)
    mutate(rs)
    try:
        T.table(rs, phase)
    except T.S2Error as e:
        ok(expect in str(e), f"refuses {label}: {e}")
        return
    ok(False, f"refuses {label}: accepted")


def first(rs, **kw):
    return next(r for r in rs if all(r[k] == v for k, v in kw.items()))


def main() -> None:
    work = Path(sys.argv[sys.argv.index("--work") + 1]) if "--work" in sys.argv else Path(tempfile.mkdtemp(prefix="s2check"))
    work.mkdir(parents=True, exist_ok=True)
    T.OVERLAP = T.Overlap(None, ROOT / "data/s2/openings", ROOT / "prompts/s2/openings/lines.json")

    # 1. pure squeeze
    sq = T.load([synth(work / "squeeze.csv", "--squeeze-only", "--noise", "0", "--models", "qwen", "--arms", "S2-DEN",
                       "--n-args", "3", "--n-wordings", "2")])
    out = T.table(sq, "confirm")
    d, raw = class_means(out, "d"), class_means(out, "d_raw")
    a = class_means(out, "fit_a")
    ok(all(abs(v) < 1e-9 for v in d.values()), f"pure squeeze: corrected d is 0 for every class {d}")
    gap = -2.0  # s0(you) - s0(language models) in the generator (-20 - -18)
    pred = (a["emotion"] - a["noun"]) * gap
    ok(abs((raw["emotion"] - raw["noun"]) - pred) < 1e-9 and abs(pred - 0.5) < 0.05,
       f"pure squeeze: raw emotion - control = {raw['emotion'] - raw['noun']:.4f}, predicted B = {pred:.4f} (about 0.5)")
    ok(all(abs(v) < 1e-9 for v in class_means(out, "d_e").values()), "pure squeeze: depth-check d_e is 0")
    ovl = class_means(out, "ovl_lm_pos")
    ok(all(0 < v <= 1 for v in ovl.values()), f"closing-ask overlap computed for the language-models rows {ovl}")

    # 2. ID and structure checks
    small = T.load([synth(work / "small.csv", "--n-args", "2", "--n-wordings", "1")])
    T.table(copy.deepcopy(small), "confirm")
    ok(True, "clean synthetic confirm table accepted")
    st = lambda rs, **kw: first(rs, vclass="noun", **kw)  # noqa: E731
    refuses(small, "confirm", lambda rs: [r.update(argument="D1.4") for r in rs if r["opening"] == "DA01-w0"],
            "Study 1 ID", "Study 1 argument ID D1.4")
    refuses(small, "confirm", lambda rs: st(rs).update(vec="H3"), "Study 1 ID", "bare H3 as a vector id")
    refuses(small, "confirm", lambda rs: [r.update(argument="CA01") for r in rs if r["opening"] == "DA01-w0"],
            "not allowed in S2-DEN", "a control argument in S2-DEN")
    refuses(small, "confirm", lambda rs: first(rs, arm="S2-CON", vclass="noun").update(vec="red", vclass="bridge"),
            "S2-DEN only", "red in S2-CON")
    refuses(small, "confirm", lambda rs: st(rs).update(chunk="99"), "(bucket, chunk)", "a steered chunk index that differs from its y0 row")
    refuses(small, "confirm", lambda rs: st(rs).update(y0=st(rs)["y0"] + 0.5), "differs from the zero-vector y", "a y0 that is not the zero-vector y")
    refuses(small, "confirm", lambda rs: st(rs).update(layer=41), "registered", "the wrong layer")
    refuses(small, "confirm", lambda rs: st(rs).update(dose=3.0), "registered dose", "a dose that is neither registered nor half")
    refuses(small, "confirm", lambda rs: st(rs).update(vec="N61"), "does not fit class", "a noun id outside N01..N60")
    refuses(small, "confirm", lambda rs: first(rs, model="k26", vclass="emotion").update(vec="sadness"),
            "does not fit class", "an emotion K2.6 did not register")
    refuses(small, "confirm", lambda rs: [r.update(ckpt="same") for r in rs if r["arm"] in ("S2-FTD", "S2-FTC")],
            "identical", "identical S2-FTD and S2-FTC checkpoints")
    refuses(small, "confirm", lambda rs: rs.remove(st(rs, twin="neg")), "different items", "a steered group missing an item")
    refuses(small, "confirm", lambda rs: st(rs, kind="s2_capacity", subject="you").update(n_steered=99),
            "steered position counts differ", "unequal steered-position counts across subjects")
    refuses(small, "confirm", lambda rs: st(rs).update(phase="pilot"), "--phase", "a pilot row in a confirmatory table")
    refuses(small, "confirm", lambda rs: [r.update(wording_role="pilot") for r in rs if r["opening"] == "DA01-w0"],
            "wording_role", "a pilot wording in a confirmatory table")
    refuses(small, "confirm", lambda rs: st(rs).update(subject="Kimi") if st(rs)["model"] == "qwen" else None,
            "own-name subject", "Kimi as the own name in Qwen rows")
    ybr = T.load([synth(work / "pilot_small.csv", "--phase", "pilot", "--n-args", "2")])
    yb = copy.deepcopy(next(r for r in ybr if r["kind"] == "yes_bias"))
    refuses(small, "confirm", lambda rs: rs.append(dict(yb, phase="confirm", wording_role="heldout")),
            "pilot only", "a known-answer item in a confirmatory table")

    # pilot restrictions
    T.table(copy.deepcopy(ybr), "pilot")
    ok(True, "clean synthetic pilot table accepted")
    def emo_on_you(rs):  # an emotion readout on a you capacity item (the pilot never produces one)
        z = first(rs, model="qwen", vclass="noun", kind="s2_capacity", subject="you")
        rs.append(dict(z, vec="joy", vclass="emotion"))
    refuses(ybr, "pilot", emo_on_you, "gate items only", "an emotion-vector row on a you item in the pilot")
    def k26_on_lm(rs):
        z = first(rs, model="k26", vclass="none", kind="s2_capacity", subject="language models")
        rs.append(dict(z, vec="N01", vclass="noun", dose=4.0, n_steered=12))
    refuses(ybr, "pilot", k26_on_lm, "gate items only", "a K2.6 control row on a non-gate item in the pilot")
    refuses(ybr, "pilot", lambda rs: rs.remove(first(rs, model="k26", vclass="noun", kind="s2_reference", twin="neg")),
            "twin pair", "a gate-item twin without its partner")

    # 3. rule 1 gates
    pf = T.load([synth(work / "pilot_fail.csv", "--phase", "pilot", "--n-args", "3", "--fail-vec", "qwen:N03")])
    G, dec = T.gates(pf)
    n3 = [g for g in G if g["model"] == "qwen" and g["vec"] == "N03"]
    ok(any(not g["passes"] and g["dose"] == 2.0 for g in n3) and any(g["passes"] and g["dose"] == 1.0 for g in n3),
       f"rule 1: N03 fails at the registered dose and passes at half ({[(g['dose'], g['passes']) for g in n3]})")
    ok(dec["qwen"]["dose"] == 1.0 and dec["k26"]["dose"] == 4.0 and not dec["qwen"]["s2_h1_descriptive"],
       f"rule 1: Qwen goes to half dose, K2.6 stays at 4 (qwen {dec['qwen']['dose']}, k26 {dec['k26']['dose']})")

    # 3b. accuracy reads the stored correct answer, not the sign. Put the pilot table's placebo and persona rows
    # on the registered polarity (pos twin +1: yes means the property is present), as the runner writes them:
    # then "Can rocks read text?"-type rows have sign +1 and correct answer No, and scoring by sign would
    # call every correct No wrong.
    ok_rows = T.load([work / "pilot_fail.csv"])
    for r in ok_rows:
        if r["kind"] in ("s2_placebo", "s2_persona"):
            r["sign"] = 1 if r["twin"] == "pos" else -1
    trap = [r for r in ok_rows if r["vclass"] == "none" and r["kind"] in ("s2_placebo", "s2_persona")
            and r["subject"] == "rocks" and r["sign"] == 1 and r["correct_answer"] == "No" and r["top1"] == "No"]
    by_sign = [r["top1"] == ("Yes" if r["sign"] == 1 else "No") for r in ok_rows if r["vec"] == "N01" and r["dose"] == 2.0
               and r["model"] == "qwen" and r["kind"] in ("s2_placebo", "s2_persona") and r["subject"] == "rocks"]
    G2, dec2 = T.gates(ok_rows)
    n1 = next(g for g in G2 if g["model"] == "qwen" and g["vec"] == "N01" and g["dose"] == 2.0)
    ok(len(trap) > 0 and sum(by_sign) / len(by_sign) < 0.95 and n1["track_rocks"] == 1.0 and n1["passes"]
       and dec2["qwen"]["dose"] == 1.0,
       f"gates score tracking against correct_answer: {len(trap)} unsteered rocks rows with sign +1 and correct "
       f"answer No; N01 rocks tracking {n1['track_rocks']:.2f} (scoring by sign would give {sum(by_sign) / len(by_sign):.2f})")
    no_ans = copy.deepcopy(ok_rows)
    first(no_ans, model="qwen", vec="N01", kind="yes_bias")["correct_answer"] = ""
    try:
        T.gates(no_ans)
        ok(False, "gates refuse a known-answer row without correct_answer: accepted")
    except T.S2Error as e:
        ok("correct_answer" in str(e), f"gates refuse a known-answer row without correct_answer: {e}")
    descriptive_py(small)
    ov = T.content_words("Do language models have emotions? Answer with one word, Yes or No.")
    ok(ov == T.content_words("language model emotion") == {"language", "model", "emotion"},
       f"closing-ask overlap normalizes plurals: {sorted(ov)}")

    if "--no-r" in sys.argv:
        return finish()
    # 4. R: planted effect (Qwen) and null (K2.6); permutation agreement; follow-up order
    main_csv = synth(work / "main.csv", "--beta", "qwen=0.25,k26=0", "--n-args", "7", "--n-wordings", "2", "--seed", "3")
    subprocess.run([PY, str(ROOT / "runs/s2_table.py"), "--phase", "confirm", "--out", str(work / "main_table.csv"), str(main_csv)],
                   check=True, capture_output=True)
    pot = work / "potency.csv"  # synthetic potency: emotions 0.5 above the controls, plus a deterministic spread
    with pot.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "vec", "dose", "potency"])
        for m, emo in (("qwen", ["joy", "sadness", "anger", "fear", "euphoria", "surprise"]), ("k26", ["joy", "fear", "euphoria", "surprise"])):
            dose = 2.0 if m == "qwen" else 4.0
            for i, v in enumerate(emo + [f"N{k:02d}" for k in range(1, 13)] + ["arithmetic", "random"]):
                w.writerow([m, v, dose, (0.8 if v in emo else 0.3) + 0.01 * (i % 5)])
    r = subprocess.run(["Rscript", str(ROOT / "runs/s2_confirm.R"), str(work / "main_out"), str(work / "main_table.csv"), "--no-loo",
                        "--potency", str(pot)], capture_output=True, text=True)
    ok(r.returncode == 0, "s2_confirm.R ran" + ("" if r.returncode == 0 else ": " + r.stderr[-2000:]))
    H = list(csv.DictReader((work / "main_out/hypotheses.csv").open()))
    h = {(x["id"], x["model"], x["what"]): x for x in H}
    get = lambda i, m: next(x for x in H if x["id"] == i and x["model"] == m)  # noqa: E731
    q, k = get("S2-H1", "qwen"), get("S2-H1", "k26")
    f = lambda x, c: float(x[c])  # noqa: E731
    print(f"     S2-H1 qwen: X = {f(q, 'estimate'):.3f} [{f(q, 'ci95_lo'):.3f}, {f(q, 'ci95_hi'):.3f}], df {f(q, 'df_satt'):.1f} -> {f(q, 'df'):.0f}, "
          f"p model {f(q, 'p_model'):.2e}, p perm {f(q, 'p_perm'):.2e}, alpha {f(q, 'alpha')}, fit {q['fit']}")
    print(f"     S2-H1 k26:  X = {f(k, 'estimate'):.3f} [{f(k, 'ci95_lo'):.3f}, {f(k, 'ci95_hi'):.3f}], df {f(k, 'df_satt'):.1f} -> {f(k, 'df'):.0f}, "
          f"p model {f(k, 'p_model'):.3f}, p perm {f(k, 'p_perm'):.3f}, alpha {f(k, 'alpha')}, fit {k['fit']}")
    ok(q["significant"] == "TRUE" and q["reading"] == "supported" and abs(f(q, "estimate") - 0.25) < 0.08,
       "planted effect detected in Qwen (supported, estimate near 0.25)")
    ok(f(q, "p_model") < 0.025 and f(q, "p_perm") < 0.025, "permutation p agrees with the model p on the planted effect")
    ok(k["significant"] == "FALSE" and f(k, "p_model") > 0.05 and f(k, "p_perm") > 0.05,
       "null in K2.6 not detected, by the model p and by the permutation p")
    ok(f(q, "df") <= 13 and f(k, "df") <= 13, "Satterthwaite df capped at nC - 1 = 13")
    for t in ("S2-G1", "S2-G2", "S2-G8"):
        x = get(t, "qwen")
        print(f"     {t} qwen: X = {f(x, 'estimate'):.3f}, p {f(x, 'p'):.2e}, gate_open {x['gate_open']}")
    ok(all(get(t, "qwen")["gate_open"] == "TRUE" and get(t, "qwen")["significant"] == "TRUE" for t in ("S2-G1", "S2-G2", "S2-G8")),
       "family 1 opens in order on the planted effect (S2-G1, S2-G2, S2-G8 significant)")
    ok(get("S2-G1", "k26")["gate_open"] == "FALSE", "family 1 stays closed for the null model")
    ok(all(get(t, "k26")["significant"] == "NA" for t in ("S2-G1", "S2-G2", "S2-G8")),
       "closed family 1 tests are reported as not tested (significant NA)")
    g6, g7 = get("S2-G6", "qwen"), get("S2-G7", "qwen")
    print(f"     S2-G6 qwen: X = {f(g6, 'estimate'):.3f}, p {f(g6, 'p'):.2e}; S2-G7 gate_open {g7['gate_open']}")
    ok(g6["gate_open"] == "TRUE" and g7["gate_open"] == ("TRUE" if g6["significant"] == "TRUE" and f(g6, "estimate") > 0 else "FALSE"),
       "S2-G7 opens only after a significant positive S2-G6")
    g4 = [get("S2-G4", m) for m in ("qwen", "k26")]
    ok(all(x["significant"] == "FALSE" for x in g4), f"S2-G4 null in S2-CON (p {[round(f(x, 'p'), 3) for x in g4]})")
    h8 = [x for x in H if x["id"] == "S2-H8" and x["model"] == "k26"]
    ok(all(x["equivalent"] == "TRUE" and x["gate_open"] == "TRUE" for x in h8),
       f"S2-H8 equivalence on both comparators for the null model (90% CIs {[(round(f(x, 'ci90_lo'), 3), round(f(x, 'ci90_hi'), 3)) for x in h8]})")
    hd = {x["model"]: x for x in csv.DictReader((work / "main_out/headline.csv").open())}
    print(f"     headline qwen: {hd['qwen']['headline']}; k26: {hd['k26']['headline']}")
    ok(hd["qwen"]["headline"] == "the claim sentence", "decision table gives the claim sentence for the planted model")
    descriptive_r(work / "main_out", work / "main_table.csv")

    # noisy pure squeeze through S2-H1
    synth(work / "sqn.csv", "--squeeze-only", "--models", "qwen", "--arms", "S2-DEN", "--n-args", "7", "--n-wordings", "2", "--seed", "5")
    subprocess.run([PY, str(ROOT / "runs/s2_table.py"), "--phase", "confirm", "--out", str(work / "sqn_table.csv"), str(work / "sqn.csv")],
                   check=True, capture_output=True)
    r = subprocess.run(["Rscript", str(ROOT / "runs/s2_confirm.R"), str(work / "sqn_out"), str(work / "sqn_table.csv"), "--only-h1", "--k26", "abandoned"],
                       capture_output=True, text=True)
    ok(r.returncode == 0, "s2_confirm.R --only-h1 ran" + ("" if r.returncode == 0 else ": " + r.stderr[-2000:]))
    s = next(csv.DictReader((work / "sqn_out/hypotheses.csv").open()))
    print(f"     squeeze S2-H1: X = {f(s, 'estimate'):.4f} [{f(s, 'ci95_lo'):.4f}, {f(s, 'ci95_hi'):.4f}], p {f(s, 'p'):.3f}")
    raw = class_means(list(csv.DictReader((work / "sqn_table.csv").open())), "d_raw")
    print(f"     squeeze raw emotion - control: {raw['emotion'] - raw['noun']:.3f}")
    ok(s["significant"] == "FALSE" and abs(f(s, "estimate")) < 0.03 and raw["emotion"] - raw["noun"] > 0.4,
       "noisy pure squeeze: raw contrast about 0.5, corrected S2-H1 about 0 and not significant")
    finish()


def cell(out: list[dict], measure: str, para: str = "1", **kw) -> float | None:
    v = [r["value"] for r in out if r["measure"] == measure and r["para"] == para and all(r[k] == x for k, x in kw.items())]
    return float(v[0]) if v else None


def descriptive_py(small: list[dict]) -> None:
    """Argmax flips, P(Yes) scale, mass drop and wording mentions on hand-set rows of one steered group."""
    g = dict(model="qwen", arm="S2-DEN", opening="DA01-w0", vec="N01")
    base = T.table(copy.deepcopy(small), "confirm")
    ok(cell(base, "flip_you", **g) == 0.0 and cell(base, "d_flip", **g) == 0.0 and cell(base, "flip_lm", **g) == 0.0,
       "flips: no argmax flip in an unmutated synthetic cell (steered y keeps the unsteered sign)")
    rs = copy.deepcopy(small)
    pick = lambda **kw: first(rs, **g, **kw)  # noqa: E731
    you_pos = pick(kind="s2_capacity", para="1", subject="you", twin="pos")
    lm_neg = pick(kind="s2_capacity", para="1", subject="language models", twin="neg")
    z = lambda r: first(rs, model="qwen", arm="S2-DEN", opening="DA01-w0", vec="none", item=r["item"])  # noqa: E731
    ok(z(you_pos)["top1"] == "No" and z(lm_neg)["top1"] == "Yes", "flips: unsteered answers are emotion absent on both rows")
    you_pos["top1"] = "Yes"   # moves to emotion present: +1 flip toward present
    lm_neg["top1"] = "The"    # leaves Yes/No: a flip, but not toward present
    per = pick(kind="s2_persona", subject="you", twin="pos")
    per["mass"] = 0.5
    you_pos["y"] = 3.0        # also exercise the P(Yes) scale
    out = T.table(rs, "confirm")
    ok(cell(out, "flip_you", **g) == 0.5 and cell(out, "flip_lm", **g) == 0.5 and cell(out, "d_flip", **g) == 0.5,
       f"flips: one you twin to Yes and one language-models twin to 'The' give flip_you 0.5, flip_lm 0.5, d_flip 0.5 "
       f"({cell(out, 'flip_you', **g)}, {cell(out, 'flip_lm', **g)}, {cell(out, 'd_flip', **g)})")
    sg = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    rows = [r for r in rs if all(r[k] == x for k, x in g.items()) and r["kind"] == "s2_capacity" and r["para"] == "1"]
    pp = {s: sum(sg(r["sign"] * r["y"]) - sg(r["sign"] * r["y0"]) for r in rows if r["subject"] == s) / 2 for s in ("you", "language models")}
    exp_pd = pp["you"] - pp["language models"]
    ok(abs(cell(out, "d_prob", **g) - exp_pd) < 1e-12 and exp_pd > 0.4,
       f"P(Yes) scale: d_prob {cell(out, 'd_prob', **g):.4f} = expected {exp_pd:.4f} after the you twin moves to y = 3")
    per_p = ((z(per)["mass"] - 0.5) + (z(per)["mass"] - pick(kind="s2_persona", subject="you", twin="neg")["mass"])) / 2
    ok(abs(cell(out, "persona_mdrop_you", "-", **g) - per_p) < 1e-12 and abs(cell(out, "persona_mdrop_lm", "-", **g) - 0.01) < 1e-9
       and abs(cell(out, "mdrop_you", **g) - 0.01) < 1e-9,
       f"mass drop: persona you {cell(out, 'persona_mdrop_you', '-', **g):.3f} (expected {per_p:.3f}), persona language models "
       f"{cell(out, 'persona_mdrop_lm', '-', **g):.3f}, capacity you {cell(out, 'mdrop_you', **g):.3f}")
    f = ROOT / "data/s2/openings/DA01/w0.md"
    text = re.sub(r"\A(?:<!--.*?-->\s*\n)+\s*", "", f.read_text())
    lm = len(re.findall(r"\blanguage models?\b", text, re.I))
    yo = len(re.findall(r"\b(?:you|your|yours|yourself|yourselves)\b", text, re.I))
    got = (cell(out, "wording_lm_mentions", "-", **g), cell(out, "wording_you_mentions", "-", **g))
    ok(got == (lm, yo) and lm >= 6, f"wording mentions read from {f.relative_to(ROOT)}: {got}, expected ({lm}, {yo})")
    ok(all(cell(out, k, **g) == cell(base, k, **g) for k in ("gap0", "mass", "flag")),
       "the top1 and persona-mass edits leave the cell's gap0, mass and flag unchanged")


def descriptive_r(out: Path, table_csv: Path) -> None:
    D = list(csv.DictReader((out / "descriptive.csv").open()))
    row = lambda m, w: next((x for x in D if x["model"] == m and x["what"] == w), None)  # noqa: E731
    for m in ("qwen", "k26"):
        x = row(m, "S2-DEN X on d_prob")
        ok(x is not None and x["estimate"] not in ("", "NA"), f"descriptive: {m} X on the P(Yes) scale ({x and x['estimate']})")
        x = row(m, "argmax flips toward emotion present, you - language models, emotions - primary controls")
        ok(x is not None and float(x["estimate"]) == 0.0, f"descriptive: {m} argmax flip row present, 0 on synthetic data without flips")
        for frame in ("capacity items", "persona item"):
            x = row(m, f"mass drop, emotions - primary controls, {frame}, you - language models")
            ok(x is not None and abs(float(x["estimate"])) < 1e-9,
               f"descriptive: {m} mass-drop subject difference on the {frame} is 0 (synthetic mass is flat)")
        for mn in ("wording_lm_mentions", "wording_you_mentions"):
            x = row(m, f"moderator: slope of opening X on {mn}")
            ok(x is not None and x["estimate"] not in ("", "NA") and int(x["n_rows"]) == 14,
               f"descriptive: {m} mention moderator {mn} over {x and x['n_rows']} openings")
        x = row(m, "potency, emotions - primary controls")
        ok(x is not None and abs(float(x["estimate"]) - 0.5) < 0.02, f"descriptive: {m} potency emotions - controls {x and x['estimate']} (planted 0.5)")
        ok(row(m, "potency: X adjusted for potency (per-vector d ~ emotion + potency)") is not None, f"descriptive: {m} potency-adjusted X row present")
        # valence: per-emotion X from a refit of the S2-H1 model, which must average to the registered estimate
        emo = T.CFG[m]["emo"]
        per = {e: row(m, f"valence: X for {e} alone ({e} - primary controls)") for e in emo}
        h1 = next(x for x in csv.DictReader((out / "hypotheses.csv").open()) if x["id"] == "S2-H1" and x["model"] == m)
        got = [float(x["estimate"]) for x in per.values() if x is not None]
        ok(len(got) == len(emo) and abs(sum(got) / len(got) - float(h1["estimate"])) < 1e-6,
           f"descriptive: {m} per-emotion X for {len(got)} of {len(emo)} emotions, mean {sum(got) / max(1, len(got)):.4f} = S2-H1 {float(h1['estimate']):.4f}")
        pos = [e for e in ("joy", "euphoria", "calm") if e in emo]
        neg = [e for e in ("sadness", "anger", "fear") if e in emo]
        x = row(m, f"valence: positive ({', '.join(pos)}) - negative ({', '.join(neg)}) emotions")
        want = (sum(float(per[e]["estimate"]) for e in pos) / len(pos) - sum(float(per[e]["estimate"]) for e in neg) / len(neg)
                if len(got) == len(emo) else None)
        ok(x is not None and want is not None and abs(float(x["estimate"]) - want) < 1e-6
           and float(x["ci95_lo"]) < 0 < float(x["ci95_hi"]),
           f"descriptive: {m} valence contrast {x and round(float(x['estimate']), 4)} = mean(positive) - mean(negative) per-emotion X, "
           f"CI covers 0 (no valence effect planted): {x and (round(float(x['ci95_lo']), 3), round(float(x['ci95_hi']), 3))}")
    O = list(csv.DictReader((out / "openings.csv").open()))
    ok(len(O) == 7 * 2 * 4 + 7 * 2 * 2 and all(o["wording_lm_mentions"] not in ("", "NA") for o in O),
       f"openings.csv: {len(O)} openings (expected 84), every one with wording mentions")
    o = next(o for o in O if o["model"] == "qwen" and o["arm"] == "S2-DEN" and o["opening"] == "DA01-w0")
    ok(float(o["X_d"]) > 0.1, f"openings.csv: opening-level X carries the planted Qwen effect ({float(o['X_d']):.3f})")
    F = list(csv.DictReader((out / "flips.csv").open()))
    ok({x["measure"] for x in F} >= {"flip_you", "flip_lm", "d_flip"}, "flips.csv: flip rates by vector class")
    M = {x["measure"] for x in csv.DictReader((out / "mass.csv").open())}
    ok({"mass_you", "persona_mass_you", "mdrop_you", "persona_mdrop_lm"} <= M, "mass.csv: capacity and persona mass and mass drop by subject")


def finish() -> None:
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
