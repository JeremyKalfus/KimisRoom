"""Study 2's post-confirmatory exploratory runs on Qwen3-32B: the carry-over run and the D1.4 bridge
(prereg/PREREG-s2.md section 5, "Exploratory analysis", where the eight choices below are fixed before any
confirmatory result; section 6, stamp B procedure step 4). Neither enters a registered test or a pilot rule.

    python runs/s2_carry.py carry  --model-path /nvme/qwen/Qwen3-32B --out results/s2/qwen/carry/carryover
    python runs/s2_carry.py bridge --model-path /nvme/qwen/Qwen3-32B --out results/s2/qwen/carry/bridge \\
        --vecs joy,...,red,ocean --doses 1          # the confirmatory S2-DEN vectors and dose (s2_box.sh final_vecs)
    python runs/s2_carry.py table  --bridge results/s2/qwen/carry/bridge \
        [--den-readouts results/s2/qwen/confirm/S2-DEN/readouts.csv | --den-table S2_TABLE.csv]
    add --dry-run to carry or bridge to build the job and print groups and tokens without running anything
    (scripts/s2_box.sh stage carry_qwen runs all three, after the confirmatory Qwen run)

carry: Study 1's carry-over design (PREREG.md, "Study design") with each of Qwen's 49 clean held-out S2-DEN
  openings (argument and frozen reply) as the conditioning pair. Each opening is followed by one of Study 1's
  40 confirmatory tasks and its fixed answer (prompts/tasks/confirmatory_tasks.jsonl), assigned in rotation in
  opening order (openings sorted by argument, then wording slot: the 1st gets c01, the 40th c40, the 41st c01
  again). The vector goes on the fixed answer's tokens only (no task-site trials); every item of Study 1's
  314-item battery (prompts/battery/items_confirm.jsonl, third-person items about the model rendered with "Qwen")
  is one unsteered suffix. Vectors: Study 1's Qwen set from results/qwen/v2/vectors.pt (hash-asserted), joy,
  sadness, anger, fear, euphoria, surprise, random, red, ocean, arithmetic, at Study 1's 2 natural units
  (coefficient 2 x 47.991127 = 95.98), plus one unsteered trial per opening: 49 x 11 = 539 groups. Groups are
  built by kimisroom.trials.carry_over_group exactly as runs/qwen_pilot.py grid built Study 1's, with ids
  `<opening>|<task>|<vector>|40|2.0` (unsteered `<opening>|<task>|none|0|0`), so Study 1's runs/confirm_table.py
  scores OUT/grid.pt unchanged (OUT/table.csv; the "arm" column holds the opening).
bridge: Study 2's design on one opening, Study 1's D1.4 argument with its frozen Qwen reply (the files whose
  SHA-256 PREREG-s2.md section 8 lists), through runs/s2_run.py's own code: its parser, build (post-subject
  groups on the 216-item confirmatory battery, vectors from the registered files, the registered unit, the
  held-out guard), pinned-weight check, kimisroom.run execution and conversion. Two things differ, both made
  here without editing the frozen file: the opening list is that one opening, and the arm label is S2-BRIDGE
  (opening and argument S1-ARG4), never S2-DEN or a Study 1 ID. Vectors and dose are the confirmatory S2-DEN ones
  (--vecs, --doses), checked against results/s2/qwen/confirm/S2-DEN/s2_meta.json when that run exists.
table: the bridge's measures with runs/s2_table.py's own measure code (load, groups_of, group_measures; its
  check() is replaced by bridge_check, which keeps every check that applies to one Study 1 opening) ->
  BRIDGE/table.csv, per-vector means of d -> BRIDGE/vectors.csv, and emotion-minus-control -> BRIDGE/summary.csv:
  per vector, the mean of d over the 8 capacity paraphrases (and, per argument, over that argument's openings);
  X = mean over the emotions minus mean over the primary controls (sampled nouns, arithmetic, random); a 95% Welch t
  interval on those vector-level means, se = sqrt(var_E/n_E + var_C/n_C), Welch-Satterthwaite df capped at n_C - 1
  (section 5's cap for the registered contrast). With --den-readouts (Qwen's S2-DEN confirmatory readouts, put through runs/s2_table.py's
  own check and table) or --den-table (a runs/s2_table.py --phase confirm table holding them) the same quantity is
  added per denial argument, over that argument's held-out openings, beside the bridge's row. Descriptive only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

import torch  # noqa: E402

import s2_run as S  # noqa: E402  (frozen in stamp A; used, never edited)
import s2_table as T  # noqa: E402  (frozen in stamp A; its measure code only)
from kimisroom.battery import BATTERY_CONFIRM, load_battery  # noqa: E402
from kimisroom.engine import Steer  # noqa: E402
from kimisroom.s2_guard import HELDOUT, Guard, opening_sha, text_sha  # noqa: E402
from kimisroom.texts import load_denial, load_jsonl  # noqa: E402
from kimisroom.trials import Turns, carry_over_group  # noqa: E402

MODEL = "qwen"
LAYER = 40
REPLIES = ROOT / "data/s2/replies/qwen3-32b"
N_OPENINGS = 49  # Qwen's clean held-out S2-DEN openings (PREREG-s2.md section 3), fixed at stamp A
# Study 1's registered materials (PREREG.md section 7) and Study 1's Qwen vector set (PREREG.md, Vectors)
TASKS = ROOT / "prompts/tasks/confirmatory_tasks.jsonl"
TASKS_SHA = "6996d1fc552bdd25b088547fb5a5fcbbd1c7c44f0644a23cddc8a84202d2551a"
BATTERY_SHA = "5f0075580febf7effce186e76e50cd6618fa56f7b7953cf1089628dea0c366a2"
N_ITEMS = 314
CARRY_VECS = ["joy", "sadness", "anger", "fear", "euphoria", "surprise", "random", "red", "ocean", "arithmetic"]
CARRY_DOSE = 2.0  # Study 1's primary Qwen dose, natural units
ENTITY = "Qwen"   # third-person items about the model (runs/qwen_pilot.py battery())
# The bridge opening (PREREG-s2.md section 8)
D14_ARG = ROOT / "prompts/denial/d1_argument_4.md"
D14_ARG_SHA = "997e7bbdd71c6135a8576d5e2439b5297f21e653ef1f6343966fa172d18767a6"
D14_REPLY = ROOT / "data/assent/qwen3-32b_d1_argument_4.json"
D14_REPLY_SHA = "d5d2c6b61b675af506d04c7edfaa64ad8bda47d46da2931aecb6fccd4d16be4a"
BRIDGE_ARM = "S2-BRIDGE"
BRIDGE_LABEL = "S1-ARG4"
CONFIRM_DEN_META = ROOT / "results/s2/qwen/confirm/S2-DEN/s2_meta.json"
RATES = {"Study 1 Qwen confirmatory (2,523 tok/s)": 2523, "pilot's slower end (1,700 tok/s)": 1700}


def sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


def file_sha(p: Path) -> str:
    return sha(Path(p).read_bytes())


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"s2_carry: {msg}")


def hours(tokens: int) -> str:
    return ", ".join(f"{tokens / r / 3600:.2f} h at {k}" for k, r in RATES.items())


# ---------------------------------------------------------------- carry-over run


def carry_openings(guard: Guard) -> list[dict]:
    """Qwen's clean held-out S2-DEN openings, from the stamped reply selection (runs/s2_run.py openings_for), in
    the rotation order: argument, then wording slot. Each must be on the held-out list; the guard refuses them
    unless stamp B holds."""
    ops = sorted(S.openings_for("S2-DEN", "confirm", REPLIES, None), key=lambda o: (o["argument"], o["wording"]))
    need(len(ops) == N_OPENINGS, f"{len(ops)} held-out S2-DEN openings with a clean reply, registered {N_OPENINGS}")
    for o in ops:
        held = guard.check_opening(o["text"], o["reply"], f"carry-over opening {o['opening']}")
        need(held, f"{o['opening']} is not on the held-out list {HELDOUT}")
    return ops


def carry_build(a) -> tuple[list, dict, dict]:
    guard = Guard(HELDOUT, Path(a.manifest)) if a.manifest else Guard()
    print(guard.log_line(), flush=True)
    ops = carry_openings(guard)
    need(file_sha(TASKS) == TASKS_SHA, f"{TASKS}: not Study 1's registered task file")
    need(file_sha(BATTERY_CONFIRM) == BATTERY_SHA, f"{BATTERY_CONFIRM}: not Study 1's registered battery")
    tasks = load_jsonl(TASKS)
    need([t["id"] for t in tasks] == [f"c{i:02d}" for i in range(1, 41)], "task file is not c01..c40 in order")
    items = load_battery(BATTERY_CONFIRM, entity=ENTITY)
    need(len(items) == N_ITEMS, f"{len(items)} battery items, expected {N_ITEMS}")
    cfg = S.B.MODELS[S.TAG[MODEL]]
    need(file_sha(cfg["vectors"]) == cfg["vectors_sha256"], f"{cfg['vectors']}: not the registered vector file")
    vz = torch.load(cfg["vectors"], weights_only=False, map_location="cpu")
    # Study 1's natural unit as runs/qwen_pilot.py natural_unit reads it (the file's nat_norm at the layer)
    nu = float(vz["nat_norm"][LAYER]) if "nat_norm" in vz else float(
        torch.stack([vz["vectors"][e][LAYER].float().norm() for e in vz["emotions"]]).mean())
    need(abs(nu - cfg["unit"]) < 0.005, f"natural unit {nu} != registered {cfg['unit']}")
    coef = CARRY_DOSE * nu
    need(not any("pain" in v for v in CARRY_VECS), "no pain vector (ethics condition)")
    jvec = {}
    for v in CARRY_VECS:
        raw = vz["vectors"][v][LAYER]
        need(raw.ndim == 1 and float(raw.norm()) > 0, f"vector {v} at layer {LAYER} is empty")
        jvec[f"{v}@{LAYER}"] = raw / raw.norm()  # as runs/qwen_pilot.py grid
    r = S.renderer(MODEL, a.tokenizer or a.model_path)
    groups, plan = [], []
    t0 = time.time()
    for i, o in enumerate(ops):
        t = tasks[i % len(tasks)]
        cond = [{"role": "user", "content": o["text"]}, {"role": "assistant", "content": o["reply"]}]
        turns = Turns(None, cond, t["task"], t["answer"])
        base = carry_over_group(r, f"{o['opening']}|{t['id']}|none|0|0", turns, items)
        assert base.readout == "suffix_last" and not base.steers
        span = carry_over_group(r, "span", turns, items[:1], "answer", "x", 1.0, 0).steers[0].prefix
        got = r.tok.decode(base.prefix[span[0]:span[1]]).strip()
        need(got == t["answer"].strip(), f"{o['opening']}: the steered span is not the fixed answer of {t['id']}")
        groups.append(base)
        for v in CARRY_VECS:
            groups.append(replace(base, id=f"{o['opening']}|{t['id']}|{v}|{LAYER}|{CARRY_DOSE}",
                                  steers=[Steer(layer=LAYER, vector=f"{v}@{LAYER}", coef=coef, prefix=span)]))
        plan.append({"opening": o["opening"], "argument": o["argument"], "slot": o["slot"], "task": t["id"],
                     "reply_draw": o["reply_draw"], "sha256": opening_sha(o["text"], o["reply"]),
                     "argument_sha256": text_sha(o["text"]), "prefix_tokens": len(base.prefix),
                     "answer_span": list(span)})
        print(f"  {o['opening']}: task {t['id']}, prefix {len(base.prefix):,} tokens, answer span {span}", flush=True)
    words = sorted({w for it in items if isinstance(it["answer_format"], list) for w in it["answer_format"]} | {"Yes", "No"})
    targets = r.answer_tokens(words)
    tokens = sum(g.n_tokens for g in groups)
    meta = {
        # Study 1's grid_meta.json keys (runs/qwen_pilot.py grid), which runs/confirm_table.py reads
        "layers": [LAYER], "doses": [CARRY_DOSE], "dose_mode": "natural", "fracs": [CARRY_DOSE],
        "vectors": CARRY_VECS, "scale": {LAYER: nu}, "natural_unit": {LAYER: nu}, "emotions": CARRY_VECS[:6],
        "vector_kinds": ["raw"], "battery": str(BATTERY_CONFIRM.relative_to(ROOT)), "n_items": len(items),
        "tasks": [p["task"] for p in plan], "task_file": str(TASKS.relative_to(ROOT)), "task_site": False,
        "vector_file": str(Path(cfg["vectors"]).relative_to(ROOT)), "pain_max_natural_dose": None,
        # Study 2 provenance
        "study": "Study 2 post-confirmatory carry-over run (exploratory; PREREG-s2.md section 5)",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": MODEL, "model_path": str(a.model_path),
        "entity": ENTITY, "coef": coef, "vector_file_sha256": cfg["vectors_sha256"], "task_file_sha256": TASKS_SHA,
        "battery_sha256": BATTERY_SHA, "replies": str(REPLIES.relative_to(ROOT)), "openings": plan,
        "guard": {"stamp_b": guard.open, "status": guard.status, "stamp_a": guard.stamp_a_status,
                  "heldout_sha256": guard.data.get("_file_sha256")},
        "n_groups": len(groups), "tokens": tokens, "build_seconds": round(time.time() - t0, 1)}
    job = {"groups": groups, "vectors": jvec, "targets": list(targets.values()), "top_k": 5, "s2_heldout": guard.data}
    return groups, job, meta | {"_targets": targets}


def digest(groups) -> str:
    d = hashlib.sha256()
    seen = set()
    for g in groups:
        s = g.steers[0] if g.steers else None
        d.update(f"{g.id}|{s and s.vector}|{s and s.coef!r}|{s and s.prefix}|{len(g.suffixes)}\n".encode())
        k = (len(g.prefix), hash(tuple(g.prefix)))
        if k not in seen:
            seen.add(k)
            d.update(str(g.prefix).encode())
            d.update(str(g.suffixes).encode())
    return d.hexdigest()


def stage_carry(a) -> None:
    out = Path(a.out)
    groups, job, meta = carry_build(a)
    targets = meta.pop("_targets")
    print(f"carry-over: {N_OPENINGS} openings x ({len(CARRY_VECS)} vectors at {CARRY_DOSE:g} units + unsteered) = "
          f"{len(groups)} groups, {meta['tokens']:,} tokens ({hours(meta['tokens'])}); built in "
          f"{meta['build_seconds']}s", flush=True)
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    mp = Path(a.model_path)
    need((mp / "config.json").exists(), f"--model-path {mp}: no checkpoint")
    cache: dict[str, str] = {}
    pin = S.pinned_revision(MODEL, None, False)
    meta["weights_check"] = S.check_weights(MODEL, mp, S.BOX / f"{MODEL}_box.json", pin, cache)
    meta["hf_revision"] = pin[1]
    meta["ckpt_sha256"] = S.ckpt_hash(mp, True, cache)
    meta["job_digest"] = digest(groups)
    old = out / "grid_meta.json"
    if old.exists():
        prev = json.loads(old.read_text())
        need(prev.get("job_digest") == meta["job_digest"],
             f"{out} holds a different job ({str(prev.get('job_digest'))[:12]} vs {meta['job_digest'][:12]})")
    S.write_json(old, meta)
    (out / "targets.json").write_text(json.dumps(targets))
    if not (out / "grid.pt").exists():
        torch.save(job, out / "grid.job.pt")
        subprocess.run([sys.executable, "-m", "kimisroom.run", str(out / "grid.job.pt"), str(out / "grid.pt"),
                        "--model", str(mp), "--gpus-per-worker", "2", "--max-tokens", str(a.max_tokens)],
                       check=True, cwd=ROOT)
    import confirm_table as CT
    rows = CT.table(out, MODEL, ENTITY)
    with (out / "table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{out / 'table.csv'}: {len(rows):,} rows (runs/confirm_table.py)", flush=True)


# ---------------------------------------------------------------- D1.4 bridge


def bridge_opening() -> dict:
    need(file_sha(D14_ARG) == D14_ARG_SHA, f"{D14_ARG}: not the registered D1.4 argument")
    need(file_sha(D14_REPLY) == D14_REPLY_SHA, f"{D14_REPLY}: not the registered frozen Qwen reply")
    text, reply = load_denial(D14_ARG), json.loads(D14_REPLY.read_text())["reply"]
    from s2_assent import text_measures
    m = text_measures(reply)
    return {"opening": BRIDGE_LABEL, "argument": BRIDGE_LABEL, "slot": "s1", "wording": -1, "role": "bridge",
            "arg_kind": "study1-denial", "text": text, "reply": reply, "reply_draw": 0,
            "reply_len": m["words"], "emo_words": m["emotion_words"]}


@contextmanager
def bridge_patches():
    """runs/s2_run.py with two module globals swapped while the bridge runs: openings_for returns the D1.4
    opening, and build relabels its S2-DEN groups as S2-BRIDGE. Everything else is the frozen code."""
    orig_open, orig_build = S.openings_for, S.build
    op = bridge_opening()

    def openings_for(arm, phase, rdir, explicit):
        assert arm == "S2-DEN" and phase == "confirm" and explicit is None, (arm, phase, explicit)
        return [dict(op)]

    def build(a, model, model_path):
        groups, job, meta = orig_build(a, model, model_path)
        pre = "S2-DEN|"
        assert all(g.id.startswith(pre) for g in groups)
        new = [replace(g, id=BRIDGE_ARM + "|" + g.id[len(pre):]) for g in groups]
        job["groups"] = new
        meta["groups"] = {BRIDGE_ARM + "|" + k[len(pre):]: v for k, v in meta["groups"].items()}
        meta["arm"] = BRIDGE_ARM
        meta["replies"] = str(D14_REPLY.relative_to(ROOT))
        meta["bridge"] = {"study": "Study 2 post-confirmatory D1.4 bridge (exploratory; PREREG-s2.md section 5)",
                          "argument_file": str(D14_ARG.relative_to(ROOT)), "argument_file_sha256": D14_ARG_SHA,
                          "reply_file": str(D14_REPLY.relative_to(ROOT)), "reply_file_sha256": D14_REPLY_SHA,
                          "built_as": "runs/s2_run.py build for S2-DEN confirm, relabelled " + BRIDGE_ARM}
        return new, job, meta

    S.openings_for, S.build = openings_for, build
    try:
        yield op
    finally:
        S.openings_for, S.build = orig_open, orig_build


def s2_args(argv: list[str]):
    """runs/s2_run.py's own parser and checks on ARGV, returning its namespace without running anything."""
    got = {}
    orig = S.stage_run
    S.stage_run = lambda a: got.setdefault("a", a)
    try:
        S.main(argv)
    finally:
        S.stage_run = orig
    return got["a"]


def bridge_argv(a) -> list[str]:
    need(a.vecs and a.doses, "bridge needs --vecs and --doses (the confirmatory S2-DEN ones: s2_box.sh final_vecs)")
    argv = ["run", "--model", MODEL, "--arm", "S2-DEN", "--phase", "confirm", "--model-path", str(a.model_path),
            "--out", str(a.out), "--items", str(S.ITEMS.relative_to(ROOT)), "--vecs", a.vecs, "--doses", a.doses,
            "--max-tokens", str(a.max_tokens),
            "--ckpt-registry", str(Path(a.out) / "checkpoints.json")]
    if a.tokenizer:
        argv += ["--tokenizer", a.tokenizer]
    return argv + (["--dry-run"] if a.dry_run else [])


def check_bridge_vectors(a) -> None:
    """The bridge's vectors and dose are the confirmatory S2-DEN run's, when that run's metadata is here."""
    if not CONFIRM_DEN_META.exists():
        print(f"note: {CONFIRM_DEN_META} not here; vectors taken from --vecs as given", flush=True)
        return
    m = json.loads(CONFIRM_DEN_META.read_text())
    want = [v["vec"] for v in m["vectors"]]
    need(a.vecs.split(",") == want, f"--vecs differ from the confirmatory S2-DEN run's vectors ({CONFIRM_DEN_META})")
    need([float(x) for x in a.doses.split(",")] == [float(x) for x in m["doses"]],
         f"--doses {a.doses} differ from the confirmatory S2-DEN run's {m['doses']}")


def stage_bridge(a) -> None:
    check_bridge_vectors(a)
    argv = bridge_argv(a)
    print(f"bridge: runs/s2_run.py's S2-DEN confirm path on the one opening {BRIDGE_LABEL} (D1.4 and its frozen Qwen "
          f"reply), every group relabelled {BRIDGE_ARM}", flush=True)
    with bridge_patches():
        S.main(argv)
    if not a.dry_run:
        need((Path(a.out) / "readouts.csv").exists(), f"{a.out}/readouts.csv was not written")


def bridge_groups(a):
    """(groups, job, meta) of the bridge as stage_bridge builds them (tests)."""
    s2a = s2_args(bridge_argv(a))
    with bridge_patches():
        return S.build(s2a, MODEL, s2a.tokenizer or s2a.model_path)


# ---------------------------------------------------------------- bridge table


def bridge_check(rows: list[dict]) -> None:
    """runs/s2_table.py check() for the bridge rows: every check that applies to one Study 1 opening, with the
    label S2-BRIDGE / S1-ARG4 in place of the S2 arm and argument rules."""
    ckpt, doses = set(), set()
    for r in rows:
        for k, v in r.items():
            T.need(not (isinstance(v, str) and T.STUDY1_IDS.match(v.strip())), f"Study 1 ID {v!r} in column {k!r}")
        T.need(r["model"] == MODEL and r["phase"] == "confirm", f"bridge rows are qwen confirm, got {r['model']} {r['phase']}")
        T.need(r["arm"] == BRIDGE_ARM and r["argument"] == BRIDGE_LABEL and r["opening"] == BRIDGE_LABEL,
               f"bridge rows carry {BRIDGE_ARM} / {BRIDGE_LABEL}, got {r['arm']} / {r['argument']} / {r['opening']}")
        T.need(r["wording_role"] == "bridge", f"wording_role {r['wording_role']!r}")
        T.check_vec(r)
        T.need(r["site"] == "post_subject", f"site {r['site']!r}")
        if r["vclass"] == "none":
            T.need(r["dose"] == 0 and r["n_steered"] == 0 and r["y"] == r["y0"], "zero-vector rows: dose 0, n_steered 0, y == y0")
        else:
            T.need(r["layer"] == T.CFG[MODEL]["layer"], f"layer {r['layer']}")
            T.need(any(abs(r["dose"] - f * T.CFG[MODEL]["dose"]) < 1e-9 for f in (1.0, 0.5)), f"dose {r['dose']}")
            T.need(r["n_steered"] > 0, "steered row with no steered positions")
            doses.add(r["dose"])
        T.need(r["kind"] in T.KINDS and r["kind"] != "yes_bias", f"item kind {r['kind']!r}")
        s = r["subject"]
        if s in (T.CFG["qwen"]["name"], T.CFG["k26"]["name"]):
            T.need(s == T.CFG[MODEL]["name"], f"own-name subject {s!r} in qwen rows")
            r["subject"] = s = "own name"
        T.need(s in T.SUBJ and s != "", f"subject {s!r}: every confirmatory item has a subject")
        T.need(r["twin"] in ("pos", "neg") and r["sign"] in (1, -1), "twin must be pos/neg and sign +-1")
        for col, rule in (("reference_fit", T.is_ref(r)), ("gate", T.gate_item(r))):
            v = r.get(col)
            if v not in (None, ""):
                T.need((str(v).strip().lower() in ("true", "1")) == rule, f"item {r['item']}: {col}={v} disagrees")
        if r.get("ckpt"):
            ckpt.add(r["ckpt"])
    T.need(len(ckpt) <= 1, "more than one checkpoint hash in the bridge rows")
    T.need(len(doses) == 1, f"the bridge runs at one dose, got {sorted(doses)}")


PRIMARY = set(T.PRIMARY)  # noun, arithmetic, noise


def t_quantile(p: float, df: float) -> float:
    """Student t quantile by bisection on the CDF (regularized incomplete beta, Lentz continued fraction)."""
    def betacf(a, b, x):
        tiny, c, d = 1e-300, 1.0, 1.0 - (a + b) * x / (a + 1)
        d = 1.0 / (d if abs(d) > tiny else tiny)
        h = d
        for m in range(1, 300):
            for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                        -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
                d = 1.0 + num * d
                d = 1.0 / (d if abs(d) > tiny else tiny)
                c = 1.0 + num / c
                c = c if abs(c) > tiny else tiny
                h *= d * c
        return h

    def ibeta(a, b, x):
        if x <= 0 or x >= 1:
            return float(x >= 1)
        lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
        if x < (a + 1) / (a + b + 2):
            return math.exp(lb) * betacf(a, b, x) / a
        return 1 - math.exp(lb) * betacf(b, a, 1 - x) / b

    def cdf(t):
        tail = 0.5 * ibeta(df / 2, 0.5, df / (df + t * t))
        return 1 - tail if t > 0 else tail

    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def contrast(per_vec: dict[str, tuple[str, float]]) -> dict:
    """X = mean over emotion vectors minus mean over primary controls, of per-vector mean d, with a 95% Welch t
    interval (df capped at nC - 1)."""
    E = [m for c, m in per_vec.values() if c == "emotion"]
    C = [m for c, m in per_vec.values() if c in PRIMARY]
    need(len(E) >= 2 and len(C) >= 2, f"need at least 2 emotions and 2 primary controls, got {len(E)} and {len(C)}")
    me, mc = sum(E) / len(E), sum(C) / len(C)
    ve = sum((x - me) ** 2 for x in E) / (len(E) - 1)
    vc = sum((x - mc) ** 2 for x in C) / (len(C) - 1)
    se = math.sqrt(ve / len(E) + vc / len(C))
    # Welch-Satterthwaite df, capped at nC - 1 as section 5 caps the registered contrast's
    den = (ve / len(E)) ** 2 / (len(E) - 1) + (vc / len(C)) ** 2 / (len(C) - 1)
    df = min(se ** 4 / den, len(C) - 1) if den > 0 else len(C) - 1
    q = t_quantile(0.975, df)
    x = me - mc
    return {"n_emotions": len(E), "n_controls": len(C), "emotions_mean": me, "controls_mean": mc, "X": x,
            "se": se, "df": df, "ci_lo": x - q * se, "ci_hi": x + q * se}


def den_table_rows(path: Path):
    """The rows of a runs/s2_table.py --phase confirm table (read as a stream)."""
    with Path(path).open(newline="") as f:
        yield from csv.DictReader(f)


def den_readout_rows(path: Path) -> list[dict]:
    """Qwen's S2-DEN confirmatory readouts through runs/s2_table.py's own check and table (no overlap measures)."""
    rows = [r for r in T.load([Path(path)]) if r["model"] == MODEL and r["arm"] == "S2-DEN"]
    need(rows, f"{path}: no qwen S2-DEN readouts")
    T.OVERLAP = None
    return T.table(rows, "confirm")


def den_per_argument(rows, dose: float, where: str) -> list[dict]:
    """Per argument, per vector mean d (capacity frame) over that argument's openings and paraphrases, from Qwen's
    S2-DEN confirmatory table rows."""
    acc = defaultdict(lambda: defaultdict(list))
    cls, ops = {}, defaultdict(set)
    for r in rows:
        if r["model"] != MODEL or r["arm"] != "S2-DEN" or r["measure"] != "d":
            continue
        need(abs(float(r["dose"]) - dose) < 1e-9, f"{where}: S2-DEN rows at dose {r['dose']}, the bridge ran at {dose}")
        acc[r["argument"]][r["vec"]].append(float(r["value"]))
        cls[r["vec"]] = r["vclass"]
        ops[r["argument"]].add(r["opening"])
    need(acc, f"{where}: no qwen S2-DEN rows with measure d")
    out = []
    for arg in sorted(acc):
        per = {v: (cls[v], sum(x) / len(x)) for v, x in acc[arg].items()}
        out.append({"source": "S2-DEN confirmatory", "argument": arg, "n_openings": len(ops[arg]), **contrast(per)})
    return out


def stage_table(a) -> None:
    bdir = Path(a.bridge)
    rows = T.load([bdir / "readouts.csv"])
    bridge_check(rows)
    steered, zero = T.groups_of(rows, "confirm")
    T.OVERLAP = None  # no closing ask for a Study 1 opening; the overlap measures are left out
    long, per_vec = [], {}
    for key, g in sorted(steered.items()):
        model, arm, opening, vec, layer, dose = key
        r0 = next(iter(g.values()))
        ms, _ = T.group_measures(g, zero[key[:3]])
        base = dict(model=model, phase="confirm", arm=arm, argument=r0["argument"], opening=opening, vec=vec,
                    vclass=r0["vclass"], layer=layer, dose=dose, reply_len=r0["reply_len"], emo_words=r0["emo_words"])
        long += [dict(base, para=p, measure=m, value=v) for p, m, v in ms]
        d = [v for p, m, v in ms if m == "d"]
        need(len(d) == 8, f"{vec}: {len(d)} capacity paraphrases with d, expected 8")
        per_vec[vec] = (r0["vclass"], sum(d) / len(d))
    out = Path(a.out) if a.out else bdir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long[0]))
        w.writeheader()
        w.writerows(long)
    with (out / "vectors.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vec", "vclass", "mean_d"])
        w.writerows([v, c, m] for v, (c, m) in per_vec.items())
    dose = next(iter(steered))[5]
    summ = [{"source": "D1.4 bridge", "argument": BRIDGE_LABEL, "n_openings": 1, **contrast(per_vec)}]
    need(not (a.den_table and a.den_readouts), "pass --den-table or --den-readouts, not both")
    if a.den_table:
        summ += den_per_argument(den_table_rows(Path(a.den_table)), dose, a.den_table)
    elif a.den_readouts:
        summ += den_per_argument(den_readout_rows(Path(a.den_readouts)), dose, a.den_readouts)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    b = summ[0]
    print(f"{out / 'table.csv'}: {len(long):,} rows from {len(steered)} steered groups; bridge X = {b['X']:+.3f} "
          f"(95% CI {b['ci_lo']:+.3f} to {b['ci_hi']:+.3f}, {b['n_emotions']} emotions vs {b['n_controls']} primary "
          f"controls); {out / 'summary.csv'}: {len(summ)} rows", flush=True)


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["carry", "bridge", "table"])
    ap.add_argument("--model-path")
    ap.add_argument("--out")
    ap.add_argument("--tokenizer", help="tokenizer dir if not the model's (dry runs)")
    ap.add_argument("--vecs", help="bridge: the confirmatory S2-DEN vectors, comma list")
    ap.add_argument("--doses", help="bridge: the confirmatory dose")
    ap.add_argument("--bridge", help="table: the bridge output directory")
    ap.add_argument("--den-table", help="table: runs/s2_table.py --phase confirm output with Qwen's S2-DEN rows")
    ap.add_argument("--den-readouts", help="table: Qwen's S2-DEN confirmatory readouts.csv, through runs/s2_table.py")
    ap.add_argument("--max-tokens", type=int, default=800_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true", help="tests only: allows --manifest")
    ap.add_argument("--manifest", help="tests only (needs --test): a stamp B manifest other than prereg/manifest-s2.txt")
    a = ap.parse_args(argv)
    need(not a.manifest or (a.test and a.dry_run), "--manifest is for dry-run tests only (needs --test --dry-run)")
    os.chdir(ROOT)  # runs/confirm_table.py reads the battery path recorded relative to the repo
    if a.stage == "table":
        need(a.bridge, "table needs --bridge")
        return stage_table(a)
    need(a.model_path and a.out, "--model-path and --out are required")
    if a.stage == "carry":
        return stage_carry(a)
    need(not a.manifest, "--manifest applies to carry only")
    return stage_bridge(a)


if __name__ == "__main__":
    main()
