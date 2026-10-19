"""Offline check of runs/s2_carry.py, the carry-over run and D1.4 bridge (no model loaded, no GPU, no network).

    .venv/bin/python tests/s2_carry_check.py

Pinned tokenizers (data/tokenizers/qwen3-32b) and a mock model folder (config.json only) stand in for the
weights; the held-out openings are opened with a stand-in stamp B (a temporary manifest listing every held-out
hash, with an .ots header carrying its digest), as tests/s2_run_check.py does.

0. The frozen files the runner uses are byte-identical to stamp A's; the t quantile matches tables.
1. Carry-over: refused before stamp B, and kimisroom.run's decoded-prefix guard refuses its job; with the stand-in
   stamp B, 49 held-out openings x 11 = 539 groups with Study 1's ids, the task rotation in opening order, the
   vector on exactly the fixed answer's tokens at 2 x 47.99 = 95.98, Study 1's 10 vectors at unit norm, 314
   unsteered battery suffixes; tampered materials are refused; Study 1's runs/confirm_table.py scores a synthetic
   result; the run path after the build (box-only weight check, hash and kimisroom.run stood in) writes table.csv
   and resumes without rerunning; the command line dry run builds the same count.
2. Bridge: runs/s2_run.py's own build on the D1.4 opening, relabelled S2-BRIDGE: 43 groups with the confirmatory
   S2-DEN vectors at 1 unit, the same suffixes, spans, vectors and coefficients as the same build on a pilot
   opening (only the prefix differs); the patches are undone; tampered D1.4 files and vectors or doses unlike the
   confirmatory S2-DEN run's are refused; the command line dry run builds 43 groups.
3. Bridge table: the frozen convert turns a synthetic bridge result into readouts; runs/s2_table.py's check refuses
   them (not an S2 arm) and bridge_check accepts them; per-vector d, X and its interval match a hand computation;
   rows with a Study 1 ID or an S2 arm are refused; --den-table and --den-readouts (synthetic S2-DEN readouts from
   the frozen build and convert, through runs/s2_table.py's own table) add per-argument rows that match by hand.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
os.chdir(ROOT)
import confirm_table as CT  # noqa: E402
import s2_carry as C  # noqa: E402
import s2_run as S  # noqa: E402
import s2_table as T  # noqa: E402
from kimisroom.engine import Readout  # noqa: E402
from kimisroom.s2_guard import OTS_MAGIC, Guard, HeldOutError  # noqa: E402

PY = sys.executable
TOK = ROOT / "data/tokenizers/qwen3-32b"
FAILS: list[str] = []


def ok(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def refused(fn, want: str = "") -> str | None:
    try:
        fn()
    except (SystemExit, HeldOutError, T.S2Error, AssertionError) as e:
        return str(e) if want in str(e) else None
    return None


def sh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, *args], capture_output=True, text=True, cwd=ROOT)


def file_sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stand_in_stamp_b(work: Path) -> Path:
    man = work / "manifest-s2.txt"
    man.write_text("stand-in stamp B manifest\n" + "\n".join(sorted(Guard().hashes)) + "\n")
    (work / "manifest-s2.txt.ots").write_bytes(OTS_MAGIC + b"\x01\x08" + hashlib.sha256(man.read_bytes()).digest())
    return man


def confirm_vectors() -> tuple[str, str]:
    """The S2-DEN vectors and dose as scripts/s2_box.sh final_vecs lists them, from the same files."""
    d = json.loads((ROOT / "results/s2/pilot/qwen_decision.json").read_text())["qwen"]
    nj = json.loads((ROOT / "results/s2/qwen/nouns/nouns.json").read_text())
    pool = json.loads((ROOT / "results/qwen/s2/pool.json").read_text())
    order = lambda v: (0, int(v[1:])) if v[:1] == "N" else (1, ("arithmetic", "random").index(v))  # noqa: E731
    prim = sorted(set(d["primary_controls"]) | set(nj["added"]), key=order)
    think = [t for t in d["thinking"] if t in pool["thinking_survivors"]]
    return f"{d['dose']:g}", ",".join(list(d["emotions"]) + prim + think + ["red", "ocean"])


# ---------------------------------------------------------------- 0


def frozen() -> None:
    listed = {}
    for line in (ROOT / "prereg/stampA-s2.txt").read_text().splitlines():
        p = line.split()
        if len(p) == 2 and len(p[0]) == 64:
            listed[p[1]] = p[0]
    for f in ("runs/s2_run.py", "runs/s2_table.py", "kimisroom/trials.py", "kimisroom/s2_guard.py"):
        ok(listed.get(f) == file_sha(ROOT / f), f"{f} is byte-identical to its stamp A hash")
    head = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "runs/confirm_table.py", "runs/qwen_pilot.py",
                           "kimisroom/battery.py", "kimisroom/engine.py"], cwd=ROOT).returncode
    ok(head == 0, "Study 1's table script, grid builder, battery loader and engine are unchanged from HEAD")
    for df, want in ((1, 12.7062), (7, 2.3646), (13, 2.1604), (25, 2.0595), (200, 1.9719)):
        got = C.t_quantile(0.975, df)
        ok(abs(got - want) < 5e-4, f"t quantile 0.975, df {df}: {got:.4f} (table {want})")


# ---------------------------------------------------------------- 1


def carry(work: Path, mock: Path) -> None:
    out = work / "carry"
    r = sh("runs/s2_carry.py", "carry", "--model-path", str(mock), "--tokenizer", str(TOK), "--out", str(out), "--dry-run")
    ok(r.returncode != 0 and "stamp B does not hold" in r.stderr, "carry-over refused before stamp B (held-out guard)")
    r = sh("runs/s2_carry.py", "carry", "--model-path", str(mock), "--tokenizer", str(TOK), "--out", str(out),
           "--dry-run", "--manifest", str(work / "x.txt"))
    ok(r.returncode != 0 and "tests only" in r.stderr, "--manifest refused without --test")
    man = stand_in_stamp_b(work)
    a = Namespace(model_path=str(mock), tokenizer=str(TOK), manifest=str(man), out=str(out))
    groups, job, meta = C.carry_build(a)
    targets = meta.pop("_targets")
    ok(meta["guard"]["stamp_b"] is True, "the stand-in stamp B opens the guard")
    ops = meta["openings"]
    ok(len(ops) == 49 and len({o["opening"] for o in ops}) == 49, f"49 distinct openings (got {len(ops)})")
    ok(all(o["opening"].startswith("DA") for o in ops), "every opening is a denial argument's (S2-DEN)")
    want = sorted(S.openings_for("S2-DEN", "confirm", C.REPLIES, None), key=lambda o: (o["argument"], o["wording"]))
    ok([o["opening"] for o in ops] == [o["opening"] for o in want], "openings in argument then wording order")
    g = Guard()
    ok(all(o["sha256"] in g.hashes and o["argument_sha256"] in g.hashes for o in ops),
       "every opening and argument hash is on the held-out list")
    ok(len(groups) == 539 and len({x.id for x in groups}) == 539, f"539 groups with unique ids (got {len(groups)})")
    ok([o["task"] for o in ops] == [f"c{i % 40 + 1:02d}" for i in range(49)], "tasks c01..c40 then c01..c09 in order")
    tasks = {t["id"]: t for t in C.load_jsonl(C.TASKS)}
    by_op: dict[str, list] = {}
    for x in groups:
        by_op.setdefault(x.id.split("|")[0], []).append(x)
    ok(all(len(v) == 11 for v in by_op.values()), "11 groups per opening (10 vectors + unsteered)")
    dec = S.renderer("qwen", str(TOK)).tok
    good_ids = good_span = good_share = True
    for o in ops:
        gs = by_op[o["opening"]]
        base = gs[0]
        t = tasks[o["task"]]
        good_ids &= base.id == f"{o['opening']}|{o['task']}|none|0|0" and not base.steers
        good_ids &= [x.id for x in gs[1:]] == [f"{o['opening']}|{o['task']}|{v}|40|2.0" for v in C.CARRY_VECS]
        a_, b_ = gs[1].steers[0].prefix
        good_span &= dec.decode(base.prefix[a_:b_]).strip() == t["answer"].strip()
        good_span &= dec.decode(base.prefix[b_:]).strip() == "<|im_end|>"
        good_span &= all(x.steers[0].prefix == (a_, b_) and x.steers[0].layer == 40 for x in gs[1:])
        good_share &= all(x.prefix is base.prefix and x.suffixes is base.suffixes for x in gs[1:])
    ok(good_ids, "ids are Study 1's arm|task|vector|layer|dose with the opening as the arm")
    ok(good_span, "the steered span is exactly the fixed answer, ending the prefix (only <|im_end|> after it)")
    ok(good_share, "steered and unsteered trials share the prefix and the battery suffixes")
    coefs = {x.steers[0].coef for x in groups if x.steers}
    nat = float(torch.load(C.S.B.MODELS["qwen3-32b"]["vectors"], weights_only=False)["nat_norm"][40])
    ok(len(coefs) == 1 and abs(coefs.pop() - 2 * nat) < 1e-9 and abs(2 * nat - 95.98) < 0.01,
       f"every coefficient is 2 x {nat:.4f} = {2 * nat:.2f}")
    ok(sorted(job["vectors"]) == sorted(f"{v}@40" for v in C.CARRY_VECS), "job vectors: Study 1's 10, none other")
    ok(all(abs(float(v.norm()) - 1) < 1e-4 and v.shape == (5120,) for v in job["vectors"].values()),
       "every job vector is unit norm and 5,120 wide")
    ok(all(len(x.suffixes) == 314 for x in groups), "every group reads Study 1's 314 battery items")
    ok({"Yes", "No"} <= set(targets) and job["targets"] == list(targets.values()), "targets include Yes and No")
    ok(job["s2_heldout"] is not None and len(job["s2_heldout"]["entries"]) == 96, "the job carries the held-out list")
    tok = dec
    prefixes = [x.prefix for x in groups]
    ok(refused(lambda: Guard().check_decoded(tok, prefixes)) is not None,
       "kimisroom.run's decoded-prefix guard refuses the job without stamp B")
    n = Guard(manifest=man).check_decoded(tok, prefixes)
    ok(n == 49, f"with stamp B the decoded-prefix guard finds all 49 prefixes held-out (got {n})")
    tokens = sum(x.n_tokens for x in groups)
    ok(tokens == meta["tokens"], f"{tokens:,} tokens")
    # tampered materials
    for name, val in (("TASKS_SHA", "0" * 64), ("BATTERY_SHA", "0" * 64)):
        old = getattr(C, name)
        setattr(C, name, val)
        try:
            ok(refused(lambda: C.carry_build(a), "not Study 1's registered") is not None, f"carry refuses a changed {name}")
        finally:
            setattr(C, name, old)
    # Study 1's table script scores the result (synthetic readouts on two openings)
    d = work / "carry_grid"
    d.mkdir()
    keep = {o["opening"] for o in ops[:2]}
    rng = torch.Generator().manual_seed(0)
    res = {}
    for x in groups:
        if x.id.split("|")[0] in keep:
            lp = -torch.rand(len(x.suffixes), len(targets), generator=rng) * 6 - 0.1
            res[x.id] = {"readouts": [Readout(torch.tensor([0]), torch.tensor([0.0]), lp[i]) for i in range(len(x.suffixes))]}
    torch.save(res, d / "grid.pt")
    (d / "targets.json").write_text(json.dumps(targets))
    (d / "grid_meta.json").write_text(json.dumps(meta, default=str))
    rows = CT.table(d, "qwen", "Qwen")
    ok({r["arm"] for r in rows} == keep and {r["vec"] for r in rows} == set(C.CARRY_VECS),
       "runs/confirm_table.py scores it: openings as arms, Study 1's 10 vectors")
    ok({(r["layer"], r["dose"], r["site"]) for r in rows} == {(40, 2.0, "answer")}, "layer 40, dose 2, answer site")
    ms = {r["measure"] for r in rows}
    ok({"self_resid", "self_raw", "other_raw", "tp_self_resid", "who", "belief", "id_self"} <= ms,
       f"Study 1's measures present ({len(rows):,} rows)")
    # the run path after the build, with the box-only steps stood in: the pinned-weight check and checkpoint hash, and
    # kimisroom.run (a stand-in that reads the saved job and writes synthetic readouts for every group)
    run_out = work / "carry_run"
    calls = []

    def fake_run(cmd, check, cwd):
        calls.append(cmd)
        jb = torch.load(cmd[3], weights_only=False)
        gen = torch.Generator().manual_seed(1)
        res = {x.id: {"readouts": [Readout(torch.tensor([0]), torch.tensor([0.0]), -torch.rand(len(jb["targets"]), generator=gen) * 6 - 0.1)
                                   for _ in x.suffixes]} for x in jb["groups"]}
        torch.save(res, cmd[4])

    saved = (C.carry_build, S.pinned_revision, S.check_weights, S.ckpt_hash, C.subprocess.run)
    C.carry_build = lambda a_: (groups, job, dict(meta, _targets=targets))
    S.pinned_revision = lambda *x: ("Qwen/Qwen3-32B", "stand-in", "test")
    S.check_weights = lambda *x: {"files_checked": 0}
    S.ckpt_hash = lambda *x: "stand-in"
    C.subprocess.run = fake_run
    try:
        C.stage_carry(Namespace(model_path=str(mock), tokenizer=str(TOK), manifest=None, out=str(run_out), dry_run=False,
                                max_tokens=800_000))
        n_calls = len(calls)
        C.stage_carry(Namespace(model_path=str(mock), tokenizer=str(TOK), manifest=None, out=str(run_out), dry_run=False,
                                max_tokens=800_000))
    finally:
        C.carry_build, S.pinned_revision, S.check_weights, S.ckpt_hash, C.subprocess.run = saved
    cmd = calls[0] if calls else []
    ok(cmd[1:3] == ["-m", "kimisroom.run"] and cmd[cmd.index("--gpus-per-worker") + 1] == "2" and
       cmd[cmd.index("--model") + 1] == str(mock), "the run goes through kimisroom.run with 2 GPUs per worker")
    trows = list(csv.DictReader((run_out / "table.csv").open()))
    ok(len({r["arm"] for r in trows}) == 49 and len({(r["arm"], r["vec"]) for r in trows}) == 490,
       f"table.csv by runs/confirm_table.py: 49 openings x 10 vectors ({len(trows):,} rows)")
    gm = json.loads((run_out / "grid_meta.json").read_text())
    ok(gm["ckpt_sha256"] == "stand-in" and gm["job_digest"] and gm["tasks"][:2] == ["c01", "c02"],
       "grid_meta.json records the checkpoint, the job digest and the task order")
    ok(n_calls == 1 and len(calls) == 1, "a rerun with grid.pt present scores again without running")
    # the command line, as scripts/s2_box.sh carry_qwen runs it
    r = sh("runs/s2_carry.py", "carry", "--model-path", str(mock), "--tokenizer", str(TOK), "--out", str(out),
           "--dry-run", "--test", "--manifest", str(man))
    ok(r.returncode == 0 and "= 539 groups" in r.stdout and f"{tokens:,} tokens" in r.stdout and not out.exists(),
       "command line dry run: 539 groups, same tokens, nothing written")
    print("   " + next((x for x in r.stdout.splitlines() if "groups" in x), r.stderr[-400:]))


# ---------------------------------------------------------------- 2


def bridge(work: Path, mock: Path) -> tuple[list, dict, Namespace]:
    dose, vecs = confirm_vectors()
    out = work / "bridge"
    a = Namespace(model_path=str(mock), tokenizer=str(TOK), out=str(out), vecs=vecs, doses=dose, max_tokens=800_000,
                  dry_run=True)
    orig = (S.openings_for, S.build, S.stage_run)
    groups, job, meta = C.bridge_groups(a)
    ok((S.openings_for, S.build, S.stage_run) == orig, "runs/s2_run.py's functions are restored after the bridge")
    ok(len(groups) == 43 and len({x.id for x in groups}) == 43, f"43 groups (42 vectors + zero; got {len(groups)})")
    ok(all(x.id.startswith("S2-BRIDGE|S1-ARG4|") and x.id.endswith("|post_subject") for x in groups),
       "every id is S2-BRIDGE|S1-ARG4|...|post_subject")
    ok(meta["arm"] == "S2-BRIDGE" and list(meta["groups"]) == [x.id for x in groups], "meta arm and index relabelled")
    o = meta["openings"]
    ok(len(o) == 1 and o[0]["opening"] == "S1-ARG4" and o[0]["role"] == "bridge", "one opening, S1-ARG4")
    ok(meta["guard"]["heldout_openings"] == [], "the D1.4 opening is not held-out")
    ok(meta["doses"] == [1.0] and abs(meta["natural_unit"] - 47.99) < 1e-9, "dose 1 unit of the registered 47.99")
    ok(len(meta["item_sets"]["full"]) == 216 and all(len(x.suffixes) == 216 for x in groups),
       "the 216-item confirmatory battery")
    ok(not any(T.STUDY1_IDS.match(str(v)) for x in groups for v in x.id.split("|")), "no Study 1 ID in any group id")
    txt = C.load_denial(C.D14_ARG)
    rep = json.loads(C.D14_REPLY.read_text())["reply"]
    tok = S.renderer("qwen", str(TOK))
    ok(groups[0].prefix == tok.ids(tok.text([{"role": "user", "content": txt}, {"role": "assistant", "content": rep}])),
       "the prefix is the D1.4 argument and its frozen Qwen reply")
    # the same build on a pilot opening: only the prefix differs
    pil = S.openings_for("S2-DEN", "pilot", C.REPLIES, None)[0]
    s2a = C.s2_args(C.bridge_argv(a))
    S.openings_for = lambda *x: [pil]
    try:
        pg, _, pm = S.build(s2a, "qwen", str(TOK))
    finally:
        S.openings_for = orig[0]
    same = len(pg) == len(groups) and all(
        x.suffixes == y.suffixes and x.steers[0].suffix_spans == y.steers[0].suffix_spans and
        x.steers[0].vector == y.steers[0].vector and x.steers[0].coef == y.steers[0].coef and
        x.id.split("|")[2:] == y.id.split("|")[2:] for x, y in zip(groups, pg))
    ok(same and groups[0].prefix != pg[0].prefix,
       f"suffixes, spans, vectors and coefficients equal the frozen build's on pilot opening {pil['opening']}")
    # tampered D1.4 files
    old = C.D14_REPLY_SHA
    C.D14_REPLY_SHA = "0" * 64
    try:
        ok(refused(lambda: C.bridge_groups(a), "frozen Qwen reply") is not None, "a changed D1.4 reply is refused")
    finally:
        C.D14_REPLY_SHA = old
    # vectors and dose against the confirmatory S2-DEN run
    fake = work / "den_meta.json"
    old = C.CONFIRM_DEN_META
    C.CONFIRM_DEN_META = fake
    try:
        fake.write_text(json.dumps({"vectors": [{"vec": v} for v in vecs.split(",")], "doses": [1.0]}))
        ok(refused(lambda: C.check_bridge_vectors(a)) is None, "the confirmatory run's vectors and dose are accepted")
        fake.write_text(json.dumps({"vectors": [{"vec": v} for v in vecs.split(",")[:-1]], "doses": [1.0]}))
        ok(refused(lambda: C.check_bridge_vectors(a), "--vecs differ") is not None, "other vectors are refused")
        fake.write_text(json.dumps({"vectors": [{"vec": v} for v in vecs.split(",")], "doses": [2.0]}))
        ok(refused(lambda: C.check_bridge_vectors(a), "--doses") is not None, "another dose is refused")
    finally:
        C.CONFIRM_DEN_META = old
    r = sh("runs/s2_carry.py", "bridge", "--model-path", str(mock), "--tokenizer", str(TOK), "--out", str(out),
           "--vecs", vecs, "--doses", dose, "--dry-run")
    ok(r.returncode == 0 and "43 groups" in r.stdout, "command line dry run: 43 groups")
    print("   " + next((x for x in r.stdout.splitlines() if "groups" in x), r.stderr[-400:]))
    return groups, meta, a


# ---------------------------------------------------------------- 3


def fake_run(d: Path, groups: list, meta: dict, shift) -> None:
    """A synthetic result for a runs/s2_run.py build, written as stage_run writes it, then the frozen convert: y0 is
    the same random draw for every opening, and a steered group moves only the "you" capacity rows, by
    shift(index entry) in the item's polarity, so d = shift exactly."""
    d.mkdir()
    meta = copy.deepcopy(meta)
    meta["tokenizer"] = str(TOK)
    meta["suffix_chunk"] = S.SUFFIX_CHUNK
    meta["ckpt_sha256"] = "stand-in"
    g0 = {meta["groups"][x.id]["set"]: x for x in groups if meta["groups"][x.id]["vec"].startswith("none")}
    meta["suffix_lens"] = {k: {i: len(s) for i, s in zip(meta["item_sets"][k], x.suffixes)} for k, x in g0.items()}
    meta["n_steered"] = {i: b - a for i, (a, b) in zip(meta["item_sets"]["full"], g0["full"].steers[0].suffix_spans)}
    S.write_json(d / "s2_meta.json", meta)
    items = {json.loads(x)["id"]: json.loads(x) for x in Path(meta["items_file"]).read_text().splitlines() if x.strip()}
    ids = meta["item_sets"]["full"]
    rnd = random.Random(1)
    y0 = {i: rnd.uniform(-4, 4) for i in ids}
    yes = meta["targets"]["Yes"]
    res = {}
    for gid, g in meta["groups"].items():
        ros = []
        for i in ids:
            it = items[i]
            y = y0[i]
            if g["vclass"] != "none" and it["kind"] == "s2_capacity" and it["subject"] == "you":
                y += shift(g) * S.SIGN[it["yes_means"]]
            p = 1 / (1 + math.exp(-y))
            ros.append(Readout(torch.tensor([yes]), torch.tensor([0.0]), torch.tensor([math.log(p * 0.95), math.log((1 - p) * 0.95)]),
                               target_logits=torch.tensor([y / 2, -y / 2], dtype=torch.float64)))
        res[gid] = {"readouts": ros}
    torch.save(res, d / "grid.pt")
    S.convert(d)


def table(work: Path, groups: list, meta: dict, a: Namespace) -> None:
    d = work / "bridge_run"
    cls = {v["vec"]: v["vclass"] for v in meta["vectors"]}
    delta = {v: (0.6 + 0.07 * k if c == "emotion" else 0.05 * k - 0.4 if c in T.PRIMARY else 1.5)
             for k, (v, c) in enumerate(cls.items())}
    fake_run(d, groups, meta, lambda g: delta[g["vec"]])
    rows = T.load([d / "readouts.csv"])
    ok({r["arm"] for r in rows} == {"S2-BRIDGE"} and {r["opening"] for r in rows} == {"S1-ARG4"},
       f"the frozen convert writes {len(rows):,} S2-BRIDGE readouts")
    ok(refused(lambda: T.check(copy.deepcopy(rows), "confirm"), "unknown arm") is not None,
       "runs/s2_table.py's check refuses the bridge rows (not an S2 arm)")
    ok(refused(lambda: C.bridge_check(copy.deepcopy(rows))) is None, "bridge_check accepts them")
    bad = copy.deepcopy(rows)
    bad[5]["argument"] = "D1.4"
    ok(refused(lambda: C.bridge_check(bad), "Study 1 ID") is not None, "a Study 1 ID in a bridge row is refused")
    bad = copy.deepcopy(rows)
    bad[5]["arm"] = "S2-DEN"
    ok(refused(lambda: C.bridge_check(bad)) is not None, "an S2-DEN row among the bridge rows is refused")
    # den table: two arguments, two openings each, d per paraphrase (plus rows that must be ignored)
    den = work / "den_table.csv"
    want = {}
    with den.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "phase", "arm", "argument", "opening", "vec", "vclass", "layer", "dose", "para", "measure", "value"])
        for arg, off in (("DA03", 0.3), ("DA07", -0.2)):
            per = {}
            for slot in ("w1", "w4"):
                for k, (v, c) in enumerate(cls.items()):
                    for p in range(8):
                        val = delta[v] + (off if c == "emotion" else 0.0) + 0.01 * p + (0.02 if slot == "w4" else 0.0)
                        w.writerow(["qwen", "confirm", "S2-DEN", arg, f"{arg}/{slot}", v, c, 40, 1.0, p, "d", val])
                        w.writerow(["qwen", "confirm", "S2-DEN", arg, f"{arg}/{slot}", v, c, 40, 1.0, p, "d_raw", 99])
                        w.writerow(["qwen", "confirm", "S2-CON", "CA01", "CA01/w1", v, c, 40, 1.0, p, "d", 99])
                        per.setdefault(v, []).append(val)
            want[arg] = C.contrast({v: (cls[v], sum(x) / len(x)) for v, x in per.items()})["X"]
    C.main(["table", "--bridge", str(d), "--den-table", str(den)])
    vec_rows = list(csv.DictReader((d / "vectors.csv").open()))
    ok(len(vec_rows) == 42 and all(abs(float(r["mean_d"]) - delta[r["vec"]]) < 1e-6 for r in vec_rows),
       "per-vector mean d equals the planted shift for all 42 vectors")
    E = [delta[v] for v, c in cls.items() if c == "emotion"]
    Cc = [delta[v] for v, c in cls.items() if c in T.PRIMARY]
    me, mc = sum(E) / len(E), sum(Cc) / len(Cc)
    vE = sum((x - me) ** 2 for x in E) / (len(E) - 1) / len(E)
    vC = sum((x - mc) ** 2 for x in Cc) / (len(Cc) - 1) / len(Cc)
    se = math.sqrt(vE + vC)
    welch = (vE + vC) ** 2 / (vE ** 2 / (len(E) - 1) + vC ** 2 / (len(Cc) - 1))
    df = min(welch, len(Cc) - 1)
    summ = list(csv.DictReader((d / "summary.csv").open()))
    b = summ[0]
    ok(len(E) == 6 and len(Cc) == 26 and b["source"] == "D1.4 bridge" and abs(float(b["df"]) - df) < 1e-9,
       f"bridge row: 6 emotions against 26 primary controls, Welch df {welch:.2f} capped at 25 -> {df:.2f}")
    q = C.t_quantile(0.975, df)
    ok(abs(float(b["X"]) - (me - mc)) < 1e-6 and abs(float(b["se"]) - se) < 1e-6, f"X = {me - mc:.4f} and its se by hand")
    ok(abs(float(b["ci_lo"]) - (me - mc - q * se)) < 1e-9 and abs(float(b["ci_hi"]) - (me - mc + q * se)) < 1e-9,
       f"the 95% interval is X -+ t(0.975, {df:.2f}) = {q:.4f} x se")
    wide = {f"e{i}": ("emotion", 1.0 + 0.1 * i) for i in range(6)} | {f"N{i:02d}": ("noun", 0.11 * i) for i in range(1, 27)}
    ok(C.contrast(wide)["df"] == 25, "the df cap of nC - 1 = 25 binds when the Welch df is larger")
    ok([r["argument"] for r in summ[1:]] == ["DA03", "DA07"] and
       all(abs(float(r["X"]) - want[r["argument"]]) < 1e-9 and r["n_openings"] == "2" for r in summ[1:]),
       "per-argument S2-DEN rows match by hand (other measures and arms ignored): "
       + ", ".join(r["argument"] + " " + format(float(r["X"]), "+.3f") for r in summ[1:]))
    ok(len(list(csv.DictReader((d / "table.csv").open()))) > 0, "bridge table.csv written")
    bad = work / "den_bad.csv"
    bad.write_text(den.read_text().replace(",1.0,", ",2.0,"))
    ok(refused(lambda: C.main(["table", "--bridge", str(d), "--den-table", str(bad)]), "dose") is not None,
       "an S2-DEN table at another dose is refused")
    # --den-readouts: S2-DEN confirmatory readouts through runs/s2_table.py's own check and table. Stand-in: the frozen
    # build on one pilot opening of each of two denial arguments, marked held-out, with a per-argument offset on the
    # emotion vectors only (so each argument's X differs from the bridge's by that offset)
    pils, seen = [], set()
    for o in S.openings_for("S2-DEN", "pilot", C.REPLIES, None):
        if o["argument"] not in seen and len(pils) < 2:
            seen.add(o["argument"])
            pils.append(dict(o, role="heldout"))
    off = {pils[0]["argument"]: 0.25, pils[1]["argument"]: -0.15}
    s2a = C.s2_args(C.bridge_argv(a))
    orig = S.openings_for
    S.openings_for = lambda *x: [dict(o) for o in pils]
    try:
        dg, _, dm = S.build(s2a, "qwen", str(TOK))
    finally:
        S.openings_for = orig
    arg_of = {o["opening"]: o["argument"] for o in pils}
    fake_run(work / "den_run", dg, dm, lambda g: delta[g["vec"]] + (off[arg_of[g["opening"]]] if g["vclass"] == "emotion" else 0))
    C.main(["table", "--bridge", str(d), "--out", str(work / "t2"), "--den-readouts", str(work / "den_run/readouts.csv")])
    summ = list(csv.DictReader((work / "t2/summary.csv").open()))
    want = {arg: C.contrast({v: (c, delta[v] + (o if c == "emotion" else 0)) for v, c in cls.items()})["X"]
            for arg, o in off.items()}
    got = ", ".join(r["argument"] + " " + format(float(r["X"]), "+.3f") for r in summ[1:])
    ok([r["argument"] for r in summ[1:]] == sorted(off) and
       all(abs(float(r["X"]) - want[r["argument"]]) < 1e-6 and r["n_openings"] == "1" for r in summ[1:]),
       f"--den-readouts: per-argument rows from S2-DEN readouts via runs/s2_table.py match by hand ({got})")
    ok(refused(lambda: C.main(["table", "--bridge", str(d), "--den-table", str(den), "--den-readouts",
                               str(work / "den_run/readouts.csv")]), "not both") is not None,
       "--den-table and --den-readouts together are refused")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        mock = work / "mock/Qwen3-32B"
        mock.mkdir(parents=True)
        (mock / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 5120}))
        frozen()
        carry(work, mock)
        groups, meta, a = bridge(work, mock)
        table(work, groups, meta, a)
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
