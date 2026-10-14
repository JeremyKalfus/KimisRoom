"""Study 2 runner (drafts/study2-spec.md sections 6-8, 11; section 12 steps 7 and 9).

    python runs/s2_run.py heldout                       # data/s2/heldout.json, the guard's list (laptop)
    python runs/s2_run.py run --model qwen --arm S2-DEN --phase pilot \\
        --model-path /nvme/qwen/Qwen3-32B --out results/s2/qwen/pilot/S2-DEN          (GPU)
    python runs/s2_run.py run --model qwen --arm S2-FTD --phase pilot --replies qwen3-32b \\
        --model-path /nvme/qwen/Qwen3-32B --adapter results/qwen/adapters/distill_keep/D2/adapter --out ...
    python runs/s2_run.py run --model k26 --arm S2-DEN --phase unsteered \\
        --model-path /nvme/k26 --out results/s2/k26/unsteered/S2-DEN                  (step 7)
    python runs/s2_run.py convert --out DIR             # grid.pt -> readouts.csv again
    python runs/s2_run.py potency --model qwen --model-path /nvme/qwen/Qwen3-32B --out results/s2/qwen/potency
                                                        # exploratory moderator: OUT/potency.csv (stage_potency)
    add --dry-run to build the job and print groups and tokens without running anything

One invocation is one (model, arm, phase) on one checkpoint. Phases:
  unsteered  step 7: zero-vector groups only, S2-DEN pilot openings, every subject-bearing row of the item
             file (capacity, state, the s2_candidate pool, persona, the s2_reserve pool, and placebo and
             reference rows if a selection exists). runs/s2_select.py turns its readouts into the selection
             that `runs/s2_battery.py --select` takes.
  pilot      the pilot battery on the pilot wordings (pilot, pilot2, or the spare that replaced a pilot).
             Qwen: the zero vector and every control on the full battery; emotions on the gate items only.
             K2.6: every vector, the zero vector included, on the gate items only. Doses: Qwen registered
             and half; K2.6 registered (pass --doses for the half dose).
  confirm    the confirmatory battery on each kept argument's clean held-out wordings; every vector on the
             full battery; red and ocean in S2-DEN only. Doses: --doses (rule 1's decision).

Groups. One group per (arm, opening, vector, layer, dose, site), id `arm|opening|vec|layer|dose|site`, site
always post_subject. The opening (the argument and the model's frozen reply) is the prefix and is never
steered; each item is one suffix steered from the token after its subject through the readout
(`trials.post_subject_group`), subjectless known-answer items on their stored `span_start`. The zero vector is
a coefficient-0 steer over the same spans, so steered and unsteered rows share suffixes, chunks and spans.
Items run gate items first: a gate-only group then puts each item in the same (bucket, chunk, row) as the
full-battery zero group, so a gate-only steered row has a y0 row with the same chunk. In the Qwen pilot
one extra zero group per opening on the gate items only (`none@gate`) checks that: its y must equal the
full zero group's y on every gate item, and it is not written out.

Held-out guard (section 11): every opening is hashed against data/s2/heldout.json at group construction
(`kimisroom.s2_guard`), and the job carries the list so `kimisroom.run` checks the decoded prefixes again.
Before stamp B any held-out opening is refused. The checked digest goes to the log and to s2_meta.json.

Checkpoints. --model-path is the checkpoint the arm runs on. S2-FTD and S2-FTC take Study 1's merged D2
and D2c checkpoints: either --model-path to a merged directory, or --adapter with the base as --model-path,
which merges into OUT/merged first (`kimisroom.distill.merge_adapter`, as `runs/distill.py merge`). The
checkpoint's SHA-256 (over its config, index, tokenizer files and, with --ckpt-hash full, every weight shard)
is recorded in every row (`ckpt`) and in OUT/../checkpoints.json, which refuses two arms that should differ
(S2-FTD, S2-FTC and the base) on the same hash. The base arms (S2-DEN, S2-CON, and the unsteered pass) refuse
to run unless --model-path holds the registered revision (Qwen/Qwen3-32B@9216db57, moonshotai/Kimi-K2.6@7eb5002f;
runs/s2_battery.py MODELS): the box file scripts/s2_box.sh wrote at download (results/s2/box/<model>_box.json,
or --box) must name that revision with revision_match true, and every file it hashed must have the same SHA-256
in --model-path (the shards are hashed once, for this check and the checkpoint hash together). The revision
and the box file's hash go to s2_meta.json (`hf_revision`, `weights_check`). Tests pin another model with
--test --pin-revision REPO@REV. Replies: Qwen arms default to data/s2/replies/qwen3-32b,
K2.6 to data/s2/replies/k26; S2-FTD and S2-FTC must name the base-Qwen replies (`--replies qwen3-32b`).

Execution and resume: the job goes to `kimisroom.run` (both models; Qwen3-32B with 2 GPUs per worker), which
saves each finished pass and skips it on a rerun; a rerun of this script with the same arguments rebuilds
the same job, checks it against the saved one and resumes. --local DEVICE runs the HF engine in this
process instead (small models on the laptop, the same pass files). OUT/readouts.csv has the columns
runs/s2_table.py documents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

import torch  # noqa: E402

import s2_battery as B  # noqa: E402
from s2_assent import CAP as REPLY_CAP  # noqa: E402  the reply cap epoch steered runs use (2,048 new tokens)
import s2_table as T  # noqa: E402
from kimisroom.engine import SUFFIX_BUCKET  # noqa: E402
from kimisroom.s2_guard import HELDOUT, Guard, opening_sha, text_sha  # noqa: E402
from kimisroom.texts import load_denial  # noqa: E402
from kimisroom.trials import Renderer, post_subject_group  # noqa: E402

SITE = "post_subject"
SUFFIX_CHUNK = 16  # HFEngine and Engine default, which kimisroom.run uses; --local may set another for tests
TAG = {"qwen": "qwen3-32b", "k26": "kimi-k2.6"}         # item file and s2_battery keys
REPLIES = {"qwen": "qwen3-32b", "k26": "k26"}           # data/s2/replies/<dir>
OPENINGS = ROOT / "data/s2/openings"
REPLY_ROOT = ROOT / "data/s2/replies"
LINES = ROOT / "prompts/s2/openings/lines.json"
ASSIGNMENT = OPENINGS / "assignment.json"
ITEMS = ROOT / "prompts/battery/items_s2.jsonl"
SIGN = {"emotion_present": 1, "property_present": 1, "true_claim": 1,
        "emotion_absent": -1, "property_absent": -1, "false_claim": -1}
UNSTEERED_KINDS = ("s2_capacity", "s2_state", "s2_candidate", "s2_persona", "s2_reserve", "s2_placebo",
                   "s2_reference")
COLUMNS = T.REQUIRED + ["correct_answer", "reply_len", "emo_words", "reply_draw", "ckpt", "gate", "reference_fit",
                        "text"]


def sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, path)


# ---------------------------------------------------------------- openings and replies


def lines() -> dict:
    return {e["id"]: e for e in json.loads(LINES.read_text())}


def arg_kind(arg: str) -> str:
    return "control" if arg.startswith("CA") else lines()[arg]["subkind"]


def argument_text(arg: str, slot: str, root: Path | None = None) -> str:
    p = (root or OPENINGS) / arg / f"{slot}.md"
    assert p.exists(), f"{p}: no accepted wording"
    return load_denial(p)


def load_replies(d: Path) -> tuple[dict, dict]:
    """(selected wordings, eligibility) of one reply source; refuses a stale eligible.json and a selection
    from another reply cap epoch than REPLY_CAP (runs/s2_assent.py, "Cap")."""
    sel = json.loads((d / "selected.json").read_text())
    el = json.loads((d / "eligible.json").read_text())
    assert sel.get("cap") == REPLY_CAP and el.get("cap") == REPLY_CAP, (
        f"{d}: selected.json / eligible.json are from the {sel.get('cap', 1200)} / {el.get('cap', 1200)}-token reply cap, "
        f"not {REPLY_CAP}: extend the capped replies, then grade, tally and eligible (runs/s2_assent.py)")
    assert el["selected_verdicts_sha256"] == sel["verdicts_sha256"], f"{d}/eligible.json is stale: rerun `eligible`"
    return sel, el


def reply_of(d: Path, w: dict, opening: str) -> str:
    assert w and w["status"] == "clean", f"{opening}: no clean frozen reply in {d} (status {w and w['status']})"
    reply = json.loads((d / w["reply_file"]).read_text())["reply"]
    assert sha(reply) == w["reply_sha256"], f"{opening}: {w['reply_file']} changed after selection"
    return reply


def openings_for(arm: str, phase: str, rdir: Path, explicit: list[str] | None) -> list[dict]:
    sel, el = load_replies(rdir)
    pre = "DA" if arm == "S2-DEN" else "CA"
    assert phase != "unsteered" or arm == "S2-DEN", "the step 7 pass runs on S2-DEN pilot openings only"
    out = []
    for arg, x in sorted(el["arguments"].items()):
        if not arg.startswith(pre):
            continue
        if phase == "confirm":
            slots = [(s, "heldout") for s in x["clean_heldout"]] if x["kept"] is True else []
        else:
            slots = [(s, "pilot2" if s == x["pilot2"] else "pilot") for s in x["pilots_run"]]
        for slot, role in slots:
            o = f"{arg}/{slot}"
            if explicit is not None and o not in explicit:
                continue
            w = sel["wordings"].get(o)
            out.append({"opening": o, "argument": arg, "slot": slot, "wording": int(slot[1:]), "role": role,
                        "arg_kind": arg_kind(arg), "text": argument_text(arg, slot), "reply": reply_of(rdir, w, o),
                        "reply_draw": w["draw"]})
    if explicit is not None:
        missing = set(explicit) - {o["opening"] for o in out}
        assert not missing, f"--openings {sorted(missing)} are not {phase} openings of {arm} with a clean reply in {rdir}"
    assert out, f"no {phase} openings for {arm} in {rdir}"
    from s2_assent import text_measures
    for o in out:
        m = text_measures(o["reply"])
        o["reply_len"], o["emo_words"] = m["words"], m["emotion_words"]
    return out


# ---------------------------------------------------------------- items


def load_items(path: Path, model: str, phase: str) -> list[dict]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if TAG[model] in r["models"]]
    if phase == "unsteered":
        rows = [r for r in rows if r["kind"] in UNSTEERED_KINDS and r["subject"]]
    else:
        rows = [r for r in rows if phase in r["batteries"]]
    assert rows, f"no {phase} items for {model} in {path}"
    # gate items first: a gate-only group then gives each item the chunk it has in the full group
    return [r for r in rows if r["gate"]] + [r for r in rows if not r["gate"]]


def check_references(path: Path, model: str, phase: str) -> None:
    """Pilot and confirmatory runs: refuse, before anything loads, an item file that leaves s2_table.py's reference
    fit fewer than 3 items (T.is_ref: s2_reference and s2_reserve rows and the rocks and humans placebo rows),
    counted on the items this run steers: all of them, and in the pilot the gate items alone (emotion vectors,
    and every K2.6 vector, run on those). The unselected battery has none; the unsteered pass runs on it by design."""
    items = load_items(path, model, phase)
    sets = {"items": items} | ({"gate items": [r for r in items if r["gate"]]} if phase == "pilot" else {})
    for name, its in sets.items():
        n = sum(map(T.is_ref, its))
        if n < 3:
            sys.exit(f"refusing {model} {phase} run: {n} of the {model} {phase} {name} in {path} are in s2_table.py's "
                     f"reference fit, which needs at least 3 (s2_reference and s2_reserve rows, rocks and humans "
                     f"placebo rows). This looks like the unselected battery: select it first with "
                     f"runs/s2_battery.py --select SEL.json (after the unsteered pass, step 7) and pass the selected "
                     f"item file with --items.")


def para_of(r: dict) -> str:
    if r["kind"] in ("s2_capacity", "s2_state"):
        return str(int(r["paraphrase"]) + 1)
    return r["id"] if r["kind"] == "yes_bias" else str(r["paraphrase"])


# ---------------------------------------------------------------- vectors


def vclass_of(v: str, model: str) -> str:
    if v in T.CFG[model]["emo"]:
        return "emotion"
    if v.startswith("mp:"):
        return "thinking"
    return {"arithmetic": "arithmetic", "random": "noise", "red": "bridge", "ocean": "bridge", "none": "none"}.get(
        v, "noun" if v[:1] == "N" else "?")


def resolve_vectors(a, model: str, layer: int, phase: str, arm: str) -> tuple[list[tuple[str, str, torch.Tensor]], dict]:
    """[(name, vclass, unit (H,) fp32)] and provenance. Emotions, arithmetic, random, red and ocean come from
    the registered vector file (hash-asserted), nouns and mental-process controls from controls.pt."""
    if phase == "unsteered":
        return [], {}
    cfg = B.MODELS[TAG[model]]
    epath = Path(a.emotion_vectors or cfg["vectors"])
    if not a.test:
        assert sha(epath.read_bytes()) == cfg["vectors_sha256"], f"{epath}: not the registered vector file"
    vz = torch.load(epath, weights_only=False, map_location="cpu")
    cdir = Path(a.controls or ROOT / f"results/{model}/s2")
    blob = torch.load(cdir / "controls.pt", weights_only=False, map_location="cpu") if (cdir / "controls.pt").exists() else None
    pool = json.loads((cdir / "pool.json").read_text()) if (cdir / "pool.json").exists() else None
    spec = a.vecs or ("emotions,pool,arithmetic,random" + (",bridge" if phase == "confirm" and arm == "S2-DEN" else ""))
    names: list[str] = []
    for tok in spec.split(","):
        if tok == "emotions":
            names += T.CFG[model]["emo"]
        elif tok in ("pool", "nouns", "thinking"):
            assert pool is not None and blob is not None, f"{cdir}: controls.pt and pool.json are needed for {tok!r}"
            if tok in ("pool", "nouns"):
                names += pool["nouns_chosen"]
            if tok in ("pool", "thinking"):
                names += (sorted(v for v, i in blob["info"].items() if i["vclass"] == "thinking") if phase == "pilot"
                          else pool["thinking_survivors"])
        elif tok == "bridge":
            names += ["red", "ocean"]
        elif tok:
            names.append(tok)
    drop = set((a.drop or "").split(",")) - {""}
    names = [n for n in dict.fromkeys(names) if n not in drop]
    out = []
    for n in names:
        c = vclass_of(n, model)
        assert c in T.CLASSES and c != "none", f"vector {n!r} has no class"
        if n in vz["vectors"] and c in ("emotion", "arithmetic", "noise", "bridge"):
            raw = vz["vectors"][n][layer]
        else:
            assert blob is not None and n in blob["vectors"], f"vector {n!r} is in neither {epath} nor {cdir}/controls.pt"
            assert blob["layer"] == layer, f"controls.pt was built at layer {blob['layer']}, not {layer}"
            raw = blob["vectors"][n]
        raw = raw.to(torch.float32)
        assert raw.ndim == 1 and raw.norm() > 0
        out.append((n, c, raw / raw.norm()))
        T.check_vec({"model": model, "vec": n, "vclass": c})
    if phase == "confirm" and arm != "S2-DEN":
        assert not any(c == "bridge" for _, c, _ in out), "red and ocean run in S2-DEN only"
    prov = {"emotion_file": str(epath), "emotion_file_sha256": sha(epath.read_bytes()),
            "controls_sha256": sha((cdir / "controls.pt").read_bytes()) if blob is not None else None,
            "pool_sha256": sha((cdir / "pool.json").read_bytes()) if pool is not None else None}
    return out, prov | {"vz_emotions": list(vz["emotions"]), "_vz": vz}


def natural_unit(vz: dict, layer: int) -> float:
    return float(torch.stack([vz["vectors"][e][layer].float().norm() for e in vz["emotions"]]).mean())


# ---------------------------------------------------------------- checkpoints


BASE_ARMS = ("S2-DEN", "S2-CON")  # run on the downloaded base weights, so the pinned revision is checked
BOX = ROOT / "results/s2/box"     # scripts/s2_box.sh box_json: <model>_box.json per box session


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hashes(model_path: Path, names, cache: dict | None = None) -> dict[str, str]:
    """name -> SHA-256 of model_path/name, 8 files at a time (hashlib releases the GIL); `cache` is reused and filled."""
    from concurrent.futures import ThreadPoolExecutor
    cache = {} if cache is None else cache
    todo = [n for n in names if n not in cache]
    with ThreadPoolExecutor(8) as ex:
        cache.update(zip(todo, ex.map(lambda n: file_sha256(model_path / n), todo)))
    return {n: cache[n] for n in names}


def ckpt_files(model_path: Path, full: bool) -> list[str]:
    small = ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.json", "merges.txt", "tiktoken.model"]
    files = [f for f in small if (model_path / f).exists()]
    if full:
        files += [p.name for p in sorted(model_path.glob("*.safetensors"))]
    return files


def ckpt_hash(model_path: Path, full: bool, cache: dict | None = None) -> str:
    """SHA-256 over `<name> <sha256>` lines of the config, index, tokenizer files and (full) every weight shard."""
    d = hashlib.sha256()
    for name, h in file_hashes(model_path, ckpt_files(model_path, full), cache).items():
        d.update(f"{name} {h}\n".encode())
    return d.hexdigest()


def pinned_revision(model: str, pin: str | None, test: bool) -> tuple[str, str, str]:
    """(repo, revision, source): the registered revision (runs/s2_battery.py MODELS, as scripts/vm_setup.sh
    downloads it), or REPO@REV from --pin-revision, which only a --test run may pass."""
    if pin:
        assert test, "--pin-revision is for tests only (needs --test); real runs check the registered revision"
        repo, _, rev = pin.rpartition("@")
        assert repo and rev, f"--pin-revision {pin}: expected REPO@REVISION"
        return repo, rev, "--pin-revision (test)"
    m = B.MODELS[TAG[model]]
    return m["repo"], m["revision"], "registered (runs/s2_battery.py MODELS)"


def check_weights(model: str, model_path: Path, box_path: Path, pin: tuple[str, str, str],
                  cache: dict | None = None) -> dict:
    """Refuse unless model_path holds the pinned revision. The reference is the box file scripts/s2_box.sh
    wrote when it downloaded the weights (results/s2/box/<model>_box.json): it must name the pinned repo and
    revision with revision_match true, and every file it hashed must be in model_path with the same SHA-256,
    with no weight shard or remote-code file in model_path that it did not hash. The shard hashes are the ones
    the full checkpoint hash computes anyway, so the check adds only the small files."""
    repo, rev, source = pin
    if not box_path.exists():
        sys.exit(f"refusing {model} base-arm run: no box file {box_path} (scripts/s2_box.sh env_{model} writes it "
                 f"when it downloads {repo}@{rev[:8]}); the weights at {model_path} cannot be checked")
    box = json.loads(box_path.read_text())
    bad = []
    if box.get("hf_repo") != repo:
        bad.append(f"box repo {box.get('hf_repo')} != pinned {repo}")
    if box.get("hf_revision_pinned") != rev:
        bad.append(f"box revision {box.get('hf_revision_pinned')} != pinned {rev}")
    if not box.get("revision_match") or box.get("hf_revision_downloaded") != [rev]:
        bad.append(f"box downloaded {box.get('hf_revision_downloaded')}, not only {rev}")
    want = box.get("weights_sha256") or {}
    if not any(n.endswith(".safetensors") for n in want):
        bad.append("box lists no weight shard")
    local = {p.name for p in model_path.iterdir() if p.is_file() and p.suffix in (".safetensors", ".py")}
    extra = sorted(local - set(want))
    if extra:
        bad.append(f"{len(extra)} weight or code files in {model_path} the box did not hash: {extra[:5]}")
    gone = sorted(n for n in want if not (model_path / n).is_file())
    if gone:
        bad.append(f"{len(gone)} files the box hashed are not in {model_path}: {gone[:5]}")
    if not bad:
        got = file_hashes(model_path, sorted(want), cache)
        diff = sorted(n for n in want if got[n] != want[n])
        if diff:
            bad.append(f"{len(diff)} files differ from the box's SHA-256: {diff[:5]}")
    if bad:
        sys.exit(f"refusing {model} base-arm run on {model_path}: not the pinned {repo}@{rev} per {box_path}:\n  "
                 + "\n  ".join(bad))
    return {"repo": repo, "revision": rev, "pinned_by": source, "box_file": str(box_path),
            "box_sha256": file_sha256(box_path), "box_created": box.get("created"), "files_checked": len(want),
            "shards_checked": sum(n.endswith(".safetensors") for n in want)}


def register_ckpt(reg: Path, arm: str, model: str, path: str, h: str) -> None:
    d = json.loads(reg.read_text()) if reg.exists() else {}
    d[f"{model}|{arm}"] = {"path": path, "sha256": h}
    base = {d[k]["sha256"] for k in d if k.startswith(f"{model}|") and k.split("|")[1] in ("S2-DEN", "S2-CON")}
    for ft in ("S2-FTD", "S2-FTC"):
        k = f"{model}|{ft}"
        if k in d:
            assert d[k]["sha256"] not in base, f"{ft} checkpoint {d[k]['path']} has the base checkpoint's hash"
    if f"{model}|S2-FTD" in d and f"{model}|S2-FTC" in d:
        assert d[f"{model}|S2-FTD"]["sha256"] != d[f"{model}|S2-FTC"]["sha256"], "S2-FTD and S2-FTC checkpoints are identical"
    write_json(reg, d)


# ---------------------------------------------------------------- build


def renderer(model: str, tok_path: str):
    from transformers import AutoTokenizer
    if model == "qwen":
        from kimisroom.hf_engine import Qwen3Renderer
        return Qwen3Renderer(AutoTokenizer.from_pretrained(tok_path), enable_thinking=False)
    return Renderer(AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True), thinking=False)


def capacity_tail(r, msgs: list[dict], path: Path, model: str) -> int:
    """The known-answer tail as s2_battery.tokenizer_pass computes it, with renderer `r`: the low
    median steered span of that model's capacity items (the span does not depend on the opening)."""
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    cap = [x for x in rows if TAG[model] in x["models"] and x["kind"] == "s2_capacity" and x["subject"]]
    assert cap, f"no capacity items for {model} in {path}"
    g = post_subject_group(r, "tail", msgs, cap, "v", 0.0, 0)
    return statistics.median_low([e - s for s, e in g.steers[0].suffix_spans])


def build(a, model: str, model_path: str) -> tuple[list, dict, dict]:
    """Groups, job vectors and the index that `convert` needs."""
    arm, phase = a.arm, a.phase
    assert model == "qwen" or arm in ("S2-DEN", "S2-CON"), f"{arm} is Qwen only"
    layer = a.layer if a.test and a.layer is not None else T.CFG[model]["layer"]
    cfg = B.MODELS[TAG[model]]
    if arm in ("S2-FTD", "S2-FTC"):
        assert a.replies, f"{arm}: name the base-Qwen replies explicitly (--replies qwen3-32b)"
    rdir = Path(a.replies) if a.replies and Path(a.replies).is_dir() else ROOT / "data/s2/replies" / (a.replies or REPLIES[model])
    guard = Guard(Path(a.heldout) if a.heldout else HELDOUT)
    print(guard.log_line(), flush=True)
    ops = openings_for(arm, phase, rdir, a.openings.split(",") if a.openings else None)
    held = [o["opening"] for o in ops if guard.check_opening(o["text"], o["reply"], f"{arm} opening {o['opening']}")]
    if phase != "confirm":
        assert not held, f"held-out text in {phase} openings {held}"
    items = load_items(Path(a.items or ITEMS), model, phase)
    vecs, prov = resolve_vectors(a, model, layer, phase, arm)
    vz = prov.pop("_vz", None)
    if vz is not None and a.test and a.unit is None:
        unit = natural_unit(vz, layer)
    elif a.test and a.unit is not None:
        unit = a.unit
    else:
        unit = cfg["unit"]
        if vz is not None:
            nu = natural_unit(vz, layer)
            assert abs(nu - unit) < 0.005 * max(1.0, unit / 10), f"natural unit {nu} != registered {unit}"
    R = T.CFG[model]["dose"]
    doses = [float(x) for x in a.doses.split(",")] if a.doses else ([R, R / 2] if (phase == "pilot" and model == "qwen") else [R])
    if not a.test:
        for d in doses:
            assert any(abs(d - f * R) < 1e-9 for f in (1.0, 0.5)), f"dose {d} is neither registered ({R}) nor half"
    gate_only_all = phase == "pilot" and model == "k26"
    sets = {"full": [r for r in items if not gate_only_all or r["gate"]]}
    if phase == "pilot" and not gate_only_all:
        sets["gate"] = [r for r in items if r["gate"]]
    r = renderer(model, a.tokenizer or model_path)
    yes, no = r.answer_tokens(["Yes", "No"]).values()
    H = None
    groups, index = [], {}
    test_tail = None
    t0 = time.time()
    for o in ops:
        msgs = [{"role": "user", "content": o["text"]}, {"role": "assistant", "content": o["reply"]}]
        bases = {}
        for name, its in sets.items():
            tail = None
            if any(not it["subject"] for it in its):
                tails = {it["steer_tail"][TAG[model]] for it in its if not it["subject"]}
                assert len(tails) == 1, f"subjectless items carry different tails {tails}"
                tail = tails.pop()
                if a.test:
                    # the stored tail is the registered tokenizer's; a --test run on another checkpoint's
                    # tokenizer (Moonlight for K2.6) recomputes it the way s2_battery does
                    if test_tail is None:
                        test_tail = capacity_tail(r, msgs, Path(a.items or ITEMS), model)
                        print(f"--test: known-answer tail {test_tail} from this tokenizer (stored {tail})", flush=True)
                    tail = test_tail
            g = post_subject_group(r, "x", msgs, its, "none", 0.0, layer, tail)
            spans = g.steers[0].suffix_spans
            for it, (s0, _) in zip(its, spans):
                # the stored span start is the registered tokenizer's; a --test run on another checkpoint's
                # tokenizer (Moonlight for K2.6) may differ, and only the tail rule applies there
                if not it["subject"] and not a.test:
                    assert s0 == it["span_start"][TAG[model]], f"{it['id']}: span start {s0} != stored {it['span_start']}"
            bases[name] = g
        gate_set = "full" if gate_only_all else "gate"

        def add(vec: str, vclass: str, dose: float, set_name: str, coef: float):
            gid = f"{arm}|{o['opening']}|{vec}|{layer}|{dose:g}|{SITE}"
            b = bases[set_name]
            vname = "none" if vclass == "none" else f"{vec}@{layer}"
            groups.append(replace(b, id=gid, steers=[replace(b.steers[0], vector=vname, coef=coef)]))
            index[gid] = {"opening": o["opening"], "vec": vec, "vclass": vclass, "dose": dose, "set": set_name}

        add("none", "none", 0.0, "full", 0.0)
        if phase == "pilot" and model == "qwen":
            add("none@gate", "none", 0.0, "gate", 0.0)
        for v, c, _ in vecs:
            for d in doses:
                on_gate = phase == "pilot" and (c == "emotion" or model == "k26")
                add(v, c, d, gate_set if on_gate else "full", d * unit)
    jvec = {f"{v}@{layer}": u for v, _, u in vecs}
    H = vecs[0][2].shape[0] if vecs else None
    if H is None:
        from transformers import AutoConfig
        try:
            H = AutoConfig.from_pretrained(model_path, trust_remote_code=True).hidden_size
        except Exception:  # dry run without weights: the zero vector's size is checked by the engine
            H = 1
    jvec["none"] = torch.zeros(H)
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "arm": arm, "phase": phase,
            "model_path": str(model_path), "layer": layer, "natural_unit": unit, "doses": doses, "site": SITE,
            "test": bool(a.test), "targets": {"Yes": yes, "No": no}, "replies": str(rdir),
            "items_file": str(a.items or ITEMS), "items_sha256": sha(Path(a.items or ITEMS).read_bytes()),
            "item_sets": {k: [it["id"] for it in v] for k, v in sets.items()},
            "vectors": [{"vec": v, "vclass": c} for v, c, _ in vecs], **prov,
            "openings": [{k: o[k] for k in ("opening", "argument", "slot", "wording", "role", "arg_kind", "reply_draw",
                                            "reply_len", "emo_words")} | {"sha256": opening_sha(o["text"], o["reply"]),
                                                                           "argument_sha256": text_sha(o["text"])}
                         for o in ops],
            "guard": {"stamp_b": guard.open, "status": guard.status, "stamp_a": guard.stamp_a_status,
                      "heldout_sha256": guard.data.get("_file_sha256"), "heldout_openings": held},
            "groups": index, "n_groups": len(groups), "tokens": sum(g.n_tokens for g in groups),
            "build_seconds": round(time.time() - t0, 1)}
    job = {"groups": groups, "vectors": jvec, "targets": [yes, no], "top_k": 5, "s2_heldout": guard.data}
    return groups, job, meta


def job_digest(groups) -> str:
    d = hashlib.sha256()
    seen = set()
    for g in groups:
        s = g.steers[0]
        d.update(f"{g.id}|{s.vector}|{s.coef!r}|{s.layer}|{len(g.suffixes)}\n".encode())
        k = (len(g.prefix), hash(tuple(g.prefix)))
        if k not in seen:
            seen.add(k)
            d.update(str(g.prefix).encode())
            d.update(str(g.suffixes).encode())
            d.update(str(s.suffix_spans).encode())
    return d.hexdigest()


# ---------------------------------------------------------------- execute


def run_local(job_path: Path, out_path: Path, model_path: str, device: str, max_tokens: int,
              suffix_chunk: int = SUFFIX_CHUNK) -> None:
    """kimisroom.run's single-worker HF path in this process, with the same pass files and resume."""
    from kimisroom.hf_engine import HFEngine
    from kimisroom.run import plan
    from kimisroom.s2_guard import check_job
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *x, **k: None  # HFEngine times itself with it
    job = torch.load(job_path, weights_only=False)
    groups = job["groups"]
    guard = check_job(model_path, groups, job.get("s2_heldout"))
    passes = plan(groups, 1, max_tokens)[0]
    parts = Path(str(out_path) + ".parts")
    parts.mkdir(exist_ok=True)
    eng = None
    t0 = time.time()
    for p, gs in enumerate(passes):
        f = parts / f"gpu0_pass{p}.pt"
        if f.exists():
            continue
        if eng is None:
            eng = HFEngine(model_path, device=device, dtype=torch.bfloat16, suffix_chunk=suffix_chunk)
        res = eng.run(gs, job["vectors"], job["targets"], job["top_k"])
        torch.save(res, f.with_suffix(".tmp"))
        os.replace(f.with_suffix(".tmp"), f)
        print(f"[local pass {p + 1}/{len(passes)}] {len(gs)} groups, {sum(g.n_tokens for g in gs):,} tokens, "
              f"{time.time() - t0:.0f}s", flush=True)
    results = {}
    for f in sorted(parts.glob("gpu*_pass*.pt")):
        results.update(torch.load(f, weights_only=False))
    missing = {g.id for g in groups} - results.keys()
    assert not missing, f"{len(missing)} groups missing from {parts}"
    results["_meta"] = {"model": model_path, "backend": "hf-local", "device": device, "s2_guard": guard,
                        "tokens": sum(g.n_tokens for g in groups), "seconds": time.time() - t0}
    torch.save(results, str(out_path) + ".tmp")
    os.replace(str(out_path) + ".tmp", out_path)


def convert(out: Path) -> Path:
    """OUT/grid.pt + OUT/s2_meta.json -> OUT/readouts.csv (runs/s2_table.py's input)."""
    meta = json.loads((out / "s2_meta.json").read_text())
    res = torch.load(out / "grid.pt", weights_only=False)
    model, arm, phase, layer = meta["model"], meta["arm"], meta["phase"], meta["layer"]
    items = {json.loads(x)["id"]: json.loads(x) for x in Path(meta["items_file"]).read_text().splitlines() if x.strip()}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta.get("tokenizer") or meta["model_path"], trust_remote_code=True)
    ops = {o["opening"]: o for o in meta["openings"]}
    sets = meta["item_sets"]

    def readouts(gid):
        g = meta["groups"][gid]
        ros = res[gid]["readouts"]
        ids = sets[g["set"]]
        assert len(ros) == len(ids), f"{gid}: {len(ros)} readouts for {len(ids)} items"
        out_ = {}
        for i, (iid, ro) in enumerate(zip(ids, ros)):
            assert ro.target_logits is not None, f"{gid}: no fp32 target logits (old engine?)"
            out_[iid] = (i, ro)
        return out_

    # (bucket, chunk) of an item: its suffix length bucket and its within-bucket chunk, in set order
    lens = meta["suffix_lens"]

    def place(set_name: str) -> dict:
        per, pos = {}, {}
        for iid in sets[set_name]:
            bk = -(-lens[set_name][iid] // SUFFIX_BUCKET) * SUFFIX_BUCKET
            k = per.get(bk, 0)
            per[bk] = k + 1
            pos[iid] = (str(bk), str(k // meta["suffix_chunk"]))
        return pos

    places = {k: place(k) for k in sets}
    zero_y = {}
    for gid, g in meta["groups"].items():
        if g["vec"] == "none":
            zero_y[g["opening"]] = {iid: float(ro.target_logits[0] - ro.target_logits[1])
                                    for iid, (_, ro) in readouts(gid).items()}
    check = {}
    for gid, g in meta["groups"].items():
        if g["vec"] == "none@gate":
            z = zero_y[g["opening"]]
            diffs = [abs(float(ro.target_logits[0] - ro.target_logits[1]) - z[iid]) for iid, (_, ro) in readouts(gid).items()]
            check[g["opening"]] = max(diffs)
            assert all(places["gate"][iid] == places["full"][iid] for iid in sets["gate"]), "gate chunks moved"
            assert max(diffs) <= 1e-4, f"{g['opening']}: gate-only zero group differs from the full one by {max(diffs)}"
    rows = []
    for gid, g in meta["groups"].items():
        if g["vec"] == "none@gate":
            continue
        o = ops[g["opening"]]
        z = zero_y[g["opening"]]
        for iid, (i, ro) in readouts(gid).items():
            it = items[iid]
            y = float(ro.target_logits[0] - ro.target_logits[1])
            tl = ro.target_logprobs
            rows.append({
                "model": model, "phase": "pilot" if phase == "unsteered" else phase, "arm": arm,
                "argument": o["argument"], "arg_kind": o["arg_kind"], "wording": o["wording"],
                "wording_role": o["role"], "opening": o["opening"], "vec": g["vec"], "vclass": g["vclass"],
                "layer": layer, "dose": g["dose"], "site": SITE, "item": iid, "kind": it["kind"], "para": para_of(it),
                "pair_id": it["pair_id"], "twin": it["twin"], "subject": it["subject"] or "",
                "sign": SIGN[it["yes_means"]], "y": y, "y0": z[iid],
                "mass": float(math.exp(float(tl[0])) + math.exp(float(tl[1]))),
                "top1": tok.decode([int(ro.top_ids[0])]).strip(),
                "bucket": places[g["set"]][iid][0], "chunk": places[g["set"]][iid][1],
                "n_steered": 0 if g["vclass"] == "none" else meta["n_steered"][iid],
                "correct_answer": it.get("correct_answer") or "", "reply_len": o["reply_len"],
                "emo_words": o["emo_words"], "reply_draw": o["reply_draw"], "ckpt": meta["ckpt_sha256"],
                "gate": it["gate"], "reference_fit": it["reference_fit"], "text": it["text"]})
    path = out / "readouts.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    if check:
        print(f"gate-only zero groups vs full zero groups: max |dy| {max(check.values()):.3g} over {len(check)} openings")
    print(f"{path}: {len(rows):,} readouts from {len(meta['groups'])} groups")
    return path


# ---------------------------------------------------------------- stages


def stage_run(a) -> None:
    if a.phase in ("pilot", "confirm"):
        check_references(Path(a.items or ITEMS), a.model, a.phase)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = a.model
    model_path = Path(a.model_path) if a.model_path else None
    if a.adapter:
        assert a.arm in ("S2-FTD", "S2-FTC"), "--adapter is for the fine-tuned arms"
        merged = Path(a.merge_dir) if a.merge_dir else out / "merged"
        if not (merged / "config.json").exists():
            assert not a.dry_run, "a dry run does not merge"
            from kimisroom.distill import merge_adapter
            print(f"merging {a.adapter} into {model_path} -> {merged}", flush=True)
            print(json.dumps(merge_adapter(str(model_path), Path(a.adapter), merged)), flush=True)
        model_path = merged
    if a.arm in ("S2-FTD", "S2-FTC") and not a.test:
        assert model == "qwen" and model_path is not None
    groups, job, meta = build(a, model, str(model_path) if model_path else (a.tokenizer or ""))
    meta["tokenizer"] = a.tokenizer
    meta["suffix_chunk"] = a.suffix_chunk
    assert a.suffix_chunk == SUFFIX_CHUNK or a.local, "kimisroom.run uses the engines' default suffix chunk (16)"
    print(f"{model} {a.arm} {a.phase}: {len(meta['openings'])} openings, {len(meta['vectors'])} vectors x doses "
          f"{meta['doses']} + zero, {meta['n_groups']} groups, {meta['tokens']:,} tokens "
          f"(built in {meta['build_seconds']}s)", flush=True)
    if a.dry_run:
        return
    assert model_path is not None and (model_path / "config.json").exists(), f"--model-path {model_path}: no checkpoint"
    hashes: dict[str, str] = {}
    if a.arm in BASE_ARMS:
        assert a.ckpt_hash == "full", "base arms need --ckpt-hash full: the revision check compares every weight shard"
        pin = pinned_revision(model, a.pin_revision, a.test)
        box = Path(a.box) if a.box else BOX / f"{model}_box.json"
        meta["weights_check"] = check_weights(model, model_path, box, pin, hashes)
        meta["hf_revision"] = pin[1]
        print(f"weights {model_path}: {pin[0]}@{pin[1]} ({pin[2]}), {meta['weights_check']['files_checked']} files "
              f"match {box}", flush=True)
    else:
        meta["weights_check"] = None  # fine-tuned arms: a merged checkpoint, identified by ckpt_sha256 alone
        meta["hf_revision"] = None
    meta["ckpt_sha256"] = ckpt_hash(model_path, a.ckpt_hash == "full", hashes)
    meta["ckpt_hash_mode"] = a.ckpt_hash
    print(f"checkpoint {model_path}: sha256 {meta['ckpt_sha256']} ({a.ckpt_hash})", flush=True)
    register_ckpt(Path(a.ckpt_registry) if a.ckpt_registry else out.parent / "checkpoints.json", a.arm, model,
                  str(model_path), meta["ckpt_sha256"])
    # per-item suffix length and steered count, from one opening (both are independent of the opening:
    # s2_battery.py asserts it on two fixtures)
    g0 = {meta["groups"][g.id]["set"]: g for g in groups if meta["groups"][g.id]["vec"].startswith("none")}
    meta["suffix_lens"] = {k: {iid: len(s) for iid, s in zip(meta["item_sets"][k], g.suffixes)} for k, g in g0.items()}
    full = g0["full"]
    meta["n_steered"] = {iid: b - a_ for iid, (a_, b) in zip(meta["item_sets"]["full"], full.steers[0].suffix_spans)}
    digest = job_digest(groups)
    old = out / "s2_meta.json"
    if old.exists():
        prev = json.loads(old.read_text())
        assert prev.get("job_digest") == digest or a.fresh, (
            f"{out} holds a different job ({prev.get('job_digest', '?')[:12]} vs {digest[:12]}): pass --fresh to replace it")
        if prev.get("job_digest") != digest:
            for f in (out / "grid.pt.parts").glob("*.pt"):
                f.unlink()
            (out / "grid.pt").unlink(missing_ok=True)
    meta["job_digest"] = digest
    write_json(out / "s2_meta.json", meta)
    if (out / "grid.pt").exists():
        print(f"{out}/grid.pt exists: converting only", flush=True)
    else:
        torch.save(job, out / "grid.job.pt")
        if a.local:
            run_local(out / "grid.job.pt", out / "grid.pt", str(model_path), a.local, a.max_tokens, a.suffix_chunk)
        else:
            gpw = a.gpus_per_worker or (2 if model == "qwen" else 1)
            subprocess.run([sys.executable, "-m", "kimisroom.run", str(out / "grid.job.pt"), str(out / "grid.pt"),
                            "--model", str(model_path), "--gpus-per-worker", str(gpw), "--max-tokens", str(a.max_tokens)],
                           check=True, cwd=ROOT)
    convert(out)


POTENCY_STORIES = ROOT / "data/stimuli/k26-v2/stories.jsonl"
POTENCY_N = 48  # every second of the 96 neutral train stories, by id


def potency_texts(n: int = POTENCY_N) -> list[dict]:
    """The potency texts: every second story, in id order, of the 96 registered neutral train stories that
    build the neutral mean every pattern is taken against (runs/s2_vectors.py), the first `n` of those."""
    from kimisroom.trials import load_jsonl
    neu = sorted((r for r in load_jsonl(POTENCY_STORIES) if r["condition"] == "neutral" and r["split"] == "train"),
                 key=lambda r: r["id"])
    assert len(neu) == 96, f"{POTENCY_STORIES}: {len(neu)} neutral train stories, expected 96"
    return neu[::2][:n]


def stage_potency(a) -> None:
    """OUT/potency.csv: per pattern and dose, the mean per-token increase in next-token NLL on neutral text when
    the pattern is added at the registered layer at every token position of the text (spec section 9 moderators).
    Per text: the mean over its predicted tokens of NLL(steered) - NLL(unsteered), both from the same engine's
    teacher-forced `prefix_all` readout (fp32 log_softmax over the full vocabulary); then the mean over texts.
    Patterns: every emotion the model uses, the pool's nouns, every mental-process control in controls.pt,
    arithmetic, random, red and ocean (--vecs to change); doses: registered and half (--doses)."""
    from kimisroom.engine import Group, Steer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, model_path = a.model, a.model_path
    layer = a.layer if a.test and a.layer is not None else T.CFG[model]["layer"]
    cfg = B.MODELS[TAG[model]]
    a.vecs = a.vecs or "emotions,pool,arithmetic,random,bridge"
    vecs, prov = resolve_vectors(a, model, layer, "pilot", "S2-DEN")  # pilot: every mental-process control
    vz = prov.pop("_vz")
    unit = a.unit if a.test and a.unit is not None else (natural_unit(vz, layer) if a.test else cfg["unit"])
    if not a.test:
        nu = natural_unit(vz, layer)
        assert abs(nu - unit) < 0.005 * max(1.0, unit / 10), f"natural unit {nu} != registered {unit}"
    R = T.CFG[model]["dose"]
    doses = [float(x) for x in a.doses.split(",")] if a.doses else [R, R / 2]
    if not a.test:
        assert all(any(abs(d - f * R) < 1e-9 for f in (1.0, 0.5)) for d in doses), f"doses {doses}: registered {R} or half"
    r = renderer(model, a.tokenizer or model_path)
    texts = potency_texts(a.potency_n)
    groups, index = [], {}
    for s in texts:
        ids = r.ids(s["text"])
        for v, c, d in [("none", "none", 0.0)] + [(v, c, d) for v, c, _ in vecs for d in doses]:
            gid = f"potency|{s['id']}|{v}|{layer}|{d:g}"
            vname = "none" if v == "none" else f"{v}@{layer}"
            groups.append(Group(gid, ids, steers=[Steer(layer, vname, d * unit, prefix=(0, len(ids)))], readout="prefix_all"))
            index[gid] = {"text": s["id"], "vec": v, "vclass": c, "dose": d}
    H = vecs[0][2].shape[0]
    jvec = {f"{v}@{layer}": u for v, _, u in vecs} | {"none": torch.zeros(H)}
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "model_path": str(model_path),
            "layer": layer, "natural_unit": unit, "doses": doses, "test": bool(a.test),
            "texts": [s["id"] for s in texts], "stories_sha256": sha(POTENCY_STORIES.read_bytes()),
            "vectors": [{"vec": v, "vclass": c} for v, c, _ in vecs], **prov, "groups": index,
            "n_groups": len(groups), "tokens": sum(g.n_tokens for g in groups), "job_digest": job_digest(groups)}
    print(f"{model} potency: {len(texts)} texts, {len(vecs)} vectors x doses {doses} + unsteered, {len(groups)} groups, "
          f"{meta['tokens']:,} tokens", flush=True)
    if a.dry_run:
        return
    assert (Path(model_path) / "config.json").exists(), f"--model-path {model_path}: no checkpoint"
    meta["ckpt_sha256"] = ckpt_hash(Path(model_path), False)
    old = out / "potency_meta.json"
    if old.exists() and json.loads(old.read_text()).get("job_digest") != meta["job_digest"]:
        assert a.fresh, f"{out} holds a different potency job: pass --fresh to replace it"
        for f in (out / "potency.pt.parts").glob("*.pt"):
            f.unlink()
        (out / "potency.pt").unlink(missing_ok=True)
    write_json(old, meta)
    res_path = out / "potency.pt"
    if not res_path.exists():
        torch.save({"groups": groups, "vectors": jvec, "targets": [], "top_k": 1}, out / "potency.job.pt")
        if a.local:
            run_local(out / "potency.job.pt", res_path, str(model_path), a.local, a.max_tokens, a.suffix_chunk)
        else:
            gpw = a.gpus_per_worker or (2 if model == "qwen" else 1)
            subprocess.run([sys.executable, "-m", "kimisroom.run", str(out / "potency.job.pt"), str(res_path),
                            "--model", str(model_path), "--gpus-per-worker", str(gpw), "--max-tokens", str(a.max_tokens)],
                           check=True, cwd=ROOT)
    res = torch.load(res_path, weights_only=False)
    nll = {}
    for gid in index:
        lps = [ro.next_logprob for ro in res[gid]["readouts"]]
        assert lps and all(x is not None and math.isfinite(x) for x in lps), f"{gid}: no next-token logprobs"
        nll[gid] = -sum(lps) / len(lps)
    rows = []
    for v, c, _ in vecs:
        for d in doses:
            diffs = [nll[f"potency|{s['id']}|{v}|{layer}|{d:g}"] - nll[f"potency|{s['id']}|none|{layer}|0"] for s in texts]
            m = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else float("nan")
            rows.append({"model": model, "vec": v, "vclass": c, "layer": layer, "dose": d, "potency": m,
                         "se": sd / math.sqrt(len(diffs)), "n_texts": len(texts),
                         "nll0": sum(nll[f"potency|{s['id']}|none|{layer}|0"] for s in texts) / len(texts)})
    path = out / "potency.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows (unsteered NLL {rows[0]['nll0']:.3f} per token over {len(texts)} texts)")


def stage_heldout(a) -> None:
    """data/s2/heldout.json: every held-out wording of the assignment, with its argument hash and each
    model's clean frozen reply hash (where one exists)."""
    asg = json.loads(Path(a.assignment).read_text())
    reps, stale = {}, []
    for m, d in (("qwen3-32b", "qwen3-32b"), ("k26", "k26")):
        p = REPLY_ROOT / d / "selected.json"
        if p.exists():
            sel = json.loads(p.read_text())
            if sel.get("cap") != REPLY_CAP:  # a selection of the old cap epoch is never hashed into the list
                stale.append(m)
                print(f"warning: {p} is from the {sel.get('cap', 1200)}-token reply cap, not {REPLY_CAP}; {m}'s replies "
                      f"are left out until its capped replies are extended and regraded (runs/s2_assent.py)",
                      file=sys.stderr)
                continue
            reps[m] = (p.parent, sel["wordings"])
    entries = []
    for arg, x in sorted(asg["arguments"].items()):
        for slot, role in sorted(x["roles"].items()):
            if not role.startswith("heldout"):
                continue
            o = f"{arg}/{slot}"
            text = argument_text(arg, slot)
            e = {"opening": o, "role": role, "argument_sha256": text_sha(text), "replies": {}}
            for m, (d, w) in reps.items():
                if (w.get(o) or {}).get("status") == "clean":
                    rep = reply_of(d, w[o], o)
                    e["replies"][m] = {"draw": w[o]["draw"], "reply_sha256": text_sha(rep),
                                       "sha256": opening_sha(text, rep)}
            entries.append(e)
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "assignment_sha256": sha(Path(a.assignment).read_bytes()),
           "reply_cap": REPLY_CAP, "replies_left_out_old_cap": stale,
           "hashing": "sha256 of stripped UTF-8 text; opening = argument + '\\n\\n' + reply (kimisroom/s2_guard.py)",
           "replies_from": sorted(reps), "entries": entries}
    write_json(Path(a.out or HELDOUT), out)
    n_rep = sum(len(e["replies"]) for e in entries)
    print(f"{a.out or HELDOUT}: {len(entries)} held-out wordings, {n_rep} model replies hashed ({', '.join(sorted(reps))})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["run", "convert", "heldout", "potency"])
    ap.add_argument("--model", choices=["qwen", "k26"])
    ap.add_argument("--arm", choices=list(T.ARMS))
    ap.add_argument("--phase", choices=["unsteered", "pilot", "confirm"])
    ap.add_argument("--model-path")
    ap.add_argument("--adapter", help="S2-FTD/S2-FTC: Study 1 adapter dir, merged into --model-path first")
    ap.add_argument("--merge-dir")
    ap.add_argument("--out")
    ap.add_argument("--replies", help="reply source: a data/s2/replies/<name> name or a directory")
    ap.add_argument("--openings", help="comma list (DA01/w1,...) restricting the phase's openings")
    ap.add_argument("--vecs", help="comma list: emotions, pool, nouns, thinking, arithmetic, random, bridge, or names")
    ap.add_argument("--drop", help="vectors to leave out (rule 1 drops)")
    ap.add_argument("--doses", help="natural units, comma list")
    ap.add_argument("--items")
    ap.add_argument("--controls", help="dir with controls.pt and pool.json (default results/<model>/s2)")
    ap.add_argument("--emotion-vectors")
    ap.add_argument("--heldout")
    ap.add_argument("--assignment", default=str(ASSIGNMENT))
    ap.add_argument("--tokenizer", help="tokenizer dir if not the model's (dry runs)")
    ap.add_argument("--ckpt-hash", choices=["full", "small"], default="full")
    ap.add_argument("--ckpt-registry")
    ap.add_argument("--box", help="box file with the weights' revision and SHA-256s (default results/s2/box/<model>_box.json)")
    ap.add_argument("--pin-revision", help="tests only (needs --test): REPO@REVISION instead of the registered pin")
    ap.add_argument("--local", help="run in this process on DEVICE (cpu, mps) instead of kimisroom.run")
    ap.add_argument("--suffix-chunk", type=int, default=SUFFIX_CHUNK, help="--local only (memory on the laptop)")
    ap.add_argument("--gpus-per-worker", type=int)
    ap.add_argument("--max-tokens", type=int, default=800_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="replace a different job in --out")
    ap.add_argument("--test", action="store_true", help="tests only: allow --layer, --unit, unregistered vectors and doses")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--unit", type=float)
    ap.add_argument("--potency-n", type=int, default=POTENCY_N, help="potency: number of neutral texts")
    a = ap.parse_args(argv)
    assert not ((a.layer is not None or a.unit is not None) and not a.test), "--layer and --unit need --test"
    if a.stage == "heldout":
        return stage_heldout(a)
    assert a.out, "--out is required"
    if a.stage == "convert":
        convert(Path(a.out))
        return
    if a.stage == "potency":
        assert a.model and a.model_path, "--model and --model-path are required"
        return stage_potency(a)
    assert a.model and a.arm and a.phase, "--model, --arm and --phase are required"
    stage_run(a)


if __name__ == "__main__":
    main()
