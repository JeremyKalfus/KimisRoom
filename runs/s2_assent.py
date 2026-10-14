"""Study 2 assent stage (spec section 4, "Frozen replies" and "Grading and exclusions"; section 12
step 3): frozen replies to every accepted opening wording from both models, the determinism
canary, and one DeepSeek-V4-Pro verdict per reply.

    # on the GPU box, from the repo root, one session (the only GPU part):
    python runs/s2_assent.py qwen /nvme/qwen/Qwen3-32B   # draw 0 for every wording, then draws 1-4 for every wording
    # back on the laptop, after copying data/s2/replies/qwen3-32b/ over:
    python runs/s2_assent.py canary          # judges the canary if it was not byte-identical
    python runs/s2_assent.py k26             # Kimi-K2.6 draw 0 through Foundry (Azure credit)
    python runs/s2_assent.py grade qwen3-32b # grades draws in order, stops at the first clean one
    python runs/s2_assent.py grade k26       # same, and makes K2.6's draws 1-4 lazily, one at a time
    python runs/s2_assent.py tally k26       # re-print the tallies from frozen verdicts
    python runs/s2_assent.py eligible k26    # join with data/s2/openings/assignment.json: what runs, what is dropped
    # the cap raised from 1,200 to 2,048 new tokens (owner decision, 2026-10-12):
    python runs/s2_assent.py extend /nvme/qwen/Qwen3-32B --draws 0   # box, round 1: capped draw-0 replies only
    python runs/s2_assent.py grade qwen3-32b  # laptop: 2,048-cap epoch; writes extend_needed.json for pending wordings
    python runs/s2_assent.py extend /nvme/qwen/Qwen3-32B --needed data/s2/replies/qwen3-32b/extend_needed.json
                                             # box, round 2: only the capped sampled draws of pending wordings
    python runs/s2_assent.py k26extend       # laptop: regenerate every capped K2.6 reply at max_tokens 2,048
    python runs/s2_assent.py grade k26       # regrades in the 2,048-cap epoch (verdicts_cap2048.jsonl)
    python runs/s2_assent.py tally k26 --cap 1200   # recompute the archived 1,200-cap record (cap1200/)

Openings are the accepted wordings `data/s2/openings/<ID>/<slot>.md` (ID in DA01..DA10, CA01..CA10;
slot w0..w6); a missing file is a wording that did not survive the checks and is simply absent.
`prompts/s2/openings/lines.json` gives each argument's claim (the closing ask's bracketed phrase).
Replies go to `data/s2/replies/<model>/<ID>_<slot>.json` (draw 0) and `<ID>_<slot>_d<k>.json` (draw k,
k = 1..4); an existing reply is never regenerated.

**Draw rule** (spec section 4), the same for every wording, both models and both arms. Draw 0 is
the zero-temperature reply with exactly Study 1's settings. If draw 0 is not clean (hedged,
disagree, unparseable, capped, or excluded after its API retries), up to 4 more draws are taken
with ordinary sampling: temperature 1, top_p 1 (Qwen also top_k 0 and no min_p, i.e. pure
sampling from the full distribution). K2.6's draws each get a fixed recorded seed (`draw_seed`:
the first 8 hex digits of SHA-256("s2-draw:<ID>/<slot>:<k>")). Qwen's draws 1-4 are generated in
batches with one fixed seed per batch (`sampled_batches`, below). The first clean draw is the wording's frozen reply;
a wording with no clean draw among the 5 is not clean. Draws are graded exactly once, in draw
order, and never after a clean one. Qwen's draws 1-4 are all generated up front in the box session
(there is no judge on the box), and later draws stay stored but are never graded or used. K2.6's
draws are made lazily by `grade k26`, one at a time and only while the wording has no clean draw.
Foundry accepts K2.6's `seed` but does not reproduce from it (checked 2026-10-11: three calls with
one seed gave three different replies), so the seed is sent and recorded for uniformity only.
`selected(out)` (and `selected.json`, written by `tally`) gives each wording's status, selected
draw index, reply file and reply text; downstream code reads replies only through it.

**Qwen** (`qwen`): exactly Study 1's `assent` stage (`runs/qwen_pilot.py:186`): bf16,
`device_map="auto"`, the chat template with thinking off, greedy, 1,200 new tokens (now 2,048, see
**Cap**), the same `generate` call. The box (instance type, GPUs, driver, CUDA, torch, transformers, device map, file
hashes of the weights) is stamped in `box.json`; a resumed run must find the same fingerprint. A
capped reply (finish_reason "length") is not clean. That is draw 0: one sequence at a time,
unbatched, greedy, no padding. The determinism canary runs first:
D1.4 is regenerated and compared with `data/assent/qwen3-32b_d1_argument_4.json`. "Byte-identical"
is read as the reply text's UTF-8 bytes (the record's timestamps and paths cannot match). If it is
not identical and the first 200 generated tokens do not match either, the run stops before any
opening (`--ignore-canary` overrides, and says so in the record). The fallback's judge half needs
Foundry, so it runs on the laptop (`canary`), which logs any mismatch in `DEVIATIONS-s2.md`.

Qwen's sampled draws (owner decision of 2026-10-11, batched to cut the box time from about 15
hours one sequence at a time): after every draw 0 exists, draws 1-4 of every wording are generated
in batches across wordings and draws. The composition is fixed: all (argument, slot, draw) for
draws 1-4, sorted by argument, slot, draw, cut into consecutive batches of `--batch-size`
(default QWEN_BATCH = 64, sized for a g6e.12xlarge, see QWEN_BATCH). Each batch has one fixed
seed, the first 8 hex digits of SHA-256("s2-batch:" + its members joined by "|", each member
"<ID>/<slot>:<k>"), set with `torch.manual_seed` right before its one `generate` call. Prompts
are left-padded with the pad token and an attention mask (position ids follow the mask), with
QWEN_SAMPLE's sampling (temperature 1, top_p 1, top_k 0, min_p None, 1,200 new tokens, now 2,048). Each row
is cut after its first stop token. The plan (members, ids, seeds, batch size) is frozen in
`batches.json` on the first run; a resumed run must find the same plan, and a batch is skipped
only when all its draw files exist (a partly written batch is regenerated whole and only its
missing files are written). Every draw file records `batch` (id, index, size, configured size,
seed, members' hash, left padding of that row) and `padded: true`. Padding and batch neighbours
change the numerics, so a draw is reproducible only as part of its whole batch on the same stack.

**K2.6** (`k26`): one Foundry call per opening to the `Kimi-K2.6` deployment (the same constant
`kimisroom.assent` and `kimisroom.api_baseline` use), with Study 1's assent parameters (temperature
0, reasoning effort none, max_tokens 1,200, now 2,048), no system prompt. Every attempt's raw response body is
stored with the full request body and `model_version`. An API error or content filter is retried with
identical inputs at most twice, then the wording is excluded. A 429 is throttling, not an error of
the opening: it is waited out and not counted (up to 30 waits, then it counts as an error).

**Grading** (`grade`): every non-excluded draw that the draw rule reaches, once, with
`kimisroom.judge_assent.grade`: the Study 1 rubric for denial arguments and its control variant
for control arguments. Verdicts are appended to the cap epoch's file (`verdicts_path`) with every raw judge response and
the draw index (lines without one are draw 0, written before the draw rule) and never re-graded; a
reply whose file changed after grading is an error, and so is a verdict for a draw after a clean
one or one whose earlier draw is ungraded. `verdicts.sha256` holds the file's hash for stamp A.
Qwen replies are only graded after the canary passed. The tallies (`tally.json`) are per argument:
assent class at draw 0, clean at draw 0, clean only at a later draw, not clean after 5 draws, clean
slots, reply length, emotion-word count, mentions (of the selected reply, else draw 0), and the arm
summary (control-arm clean rate beside the denial rate, share of clean replies from sampled draws).

**Eligibility** (`eligible`): `selected.json` joined with the pilot / held-out / spare assignment
(`runs/s2_openings.py assign`), spec section 4's rules per model: a pilot that is not clean is
replaced by a clean spare; an argument needs 3 clean held-out wordings; S2-H1 needs 7 kept denial
arguments and the control arms 7 kept control arguments. Written to `eligible.json`.

**Cap** (owner decision of 2026-10-12, before stamp A and before any steered trial). Replies were first
capped at 1,200 new tokens (Study 1's budget); a capped reply is excluded. That cut off 37 of 67 draw-0
Qwen denial replies (and 18 of 62 control, 371 of 516 sampled draws; K2.6: 8 draw 0, 18 sampled) in
their closing summary, so the cap is now CAP = 2,048 for both models and every capped reply is
extended to it with the same settings. Qwen (`extend`, on the box that made the replies, checked
against box.json): draw 0 continues greedily and unbatched from its prompt plus completion_ids, so it
equals one greedy run at 2,048; that is checked first on `--check-n` wordings regenerated from scratch
(`extend.json` "draw0_check"; if any differs every capped draw 0 is regenerated from scratch). Sampled
draws continue in fixed batches (`extend_batches.json`: capped draws sorted by argument, slot, draw,
batches of EXTEND_BATCH, seed from SHA-256("s2-extend-batch:" + members)); the tokens past 1,200 come
from that continuation batch's seed. Qwen runs in two rounds (coordinator decision, 2026-10-12): round 1
extends only draw 0 (`--draws 0`); the laptop grades; `tally` writes `extend_needed.json`, every draw
still capped at 1,200 of every wording whose walk stops at one ("unextended"); round 2 (`--needed`)
extends only those, so one more round suffices. Draws never reached stay as they are at the 1,200 cap,
are never graded, and are counted as "not needed" in tally.json. K2.6 (`k26extend`): Foundry cannot continue an assistant message
(K26_NO_PREFIX), so each capped reply is regenerated whole at 2,048 with its draw's settings. Each
original is archived byte for byte under `cap1200/`; each new file has `cap` 2048 and `extension`
(original and new token counts, finish_reason, mode, batch id and seed where relevant). A reply still
capped at 2,048 stays excluded. Grading restarts in a cap epoch: `verdicts.jsonl` stays as the 1,200-cap
record and `verdicts_cap2048.jsonl` holds the 2,048-cap verdicts, made in draw order with the same
stop-at-first-clean rule; a verdict is copied from the 1,200-cap record (with `reused_from`) only where
the draw's reply text is byte-identical. A reply still capped at 1,200 stops its wording's walk
("unextended", pending). selected.json, tally.json and eligible.json carry `cap`; the 1,200-cap ones
are moved to `cap1200/` by the first 2,048-cap tally (`tally --cap 1200` recomputes them there).

The Foundry key is read through `az` inside `kimisroom.foundry`, kept in memory and never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.texts import ROOT, load_denial  # noqa: E402

IDS = [f"DA{i:02d}" for i in range(1, 11)] + [f"CA{i:02d}" for i in range(1, 11)]
SLOTS = [f"w{i}" for i in range(7)]
OPENINGS = ROOT / "data/s2/openings"
LINES = ROOT / "prompts/s2/openings/lines.json"
REPLIES = ROOT / "data/s2/replies"
DEVIATIONS = ROOT / "DEVIATIONS-s2.md"
STUDY1 = ROOT / "data/assent/qwen3-32b_d1_argument_4.json"
STUDY1_STEM = "d1_argument_4"
CANARY_TOKENS = 200

QWEN_TAG, K26_TAG = "qwen3-32b", "k26"
QWEN_REV = "9216db5781bf21249d130ec9da846c4624c16137"  # Study 1's pin (runs/qwen_pilot.py, PLAN.md)
QWEN_SHAPE = {"model_type": "qwen3", "num_hidden_layers": 64, "hidden_size": 5120}
# The reply cap (new tokens). Study 1 and the first Study 2 replies used 1,200; the owner raised it to
# 2,048 on 2026-10-12 (before stamp A and before any steered trial) because 1,200 cut off most of
# Qwen3-32B's denial replies in their closing summary. Capped 1,200 replies are extended (`extend`,
# `k26extend`), the originals archived under ARCHIVE, and grading restarts in a new cap epoch
# (`verdicts_path`): the 1,200-cap verdicts stay as that epoch's record and are reused only for
# byte-identical reply text.
CAP_OLD, CAP = 1200, 2048
ARCHIVE = f"cap{CAP_OLD}"
STUDY1_PARAMS = {"enable_thinking": False, "do_sample": False, "max_new_tokens": CAP_OLD, "dtype": "bfloat16"}
QWEN_PARAMS = {"enable_thinking": False, "do_sample": False, "max_new_tokens": CAP, "dtype": "bfloat16"}
K26 = "Kimi-K2.6"  # kimisroom/assent.py:29, kimisroom/api_baseline.py:79
K26_PARAMS = {"reasoning_effort": "none", "temperature": 0, "max_tokens": CAP}  # kimisroom/assent.py PARAMS, cap raised
K26_ATTEMPTS, THROTTLE_WAITS = 3, 30   # one call plus at most two identical retries
# the draw rule (spec section 4): draw 0 above, then up to 4 sampled draws until one is clean
N_DRAWS = 5
QWEN_SAMPLE = {"enable_thinking": False, "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
               "min_p": None, "max_new_tokens": CAP, "dtype": "bfloat16"}
# Sampled Qwen draws per batch, for a g6e.12xlarge (4 x L40S 48 GB) with Qwen3-32B bf16 under
# device_map="auto": weights about 65.5 GB, about 16.4 GB per GPU. KV cache: 64 layers x 2 x 8 KV heads x
# 128 dims x 2 bytes = 256 KiB per token; a sequence is at most about 1,300 prompt + 1,200 new tokens, so
# about 0.65 GB, and 64 sequences about 42 GB (about 10.5 GB per GPU). The prefill peak on one GPU adds
# about 13 GB of MLP activations (64 x 1,300 tokens x 25,600 x 2 bytes x 3). Peak about 35 GB of 44.5 GiB
# usable per GPU; 96 would leave almost nothing for fragmentation and the attention kernel's workspace.
QWEN_BATCH = 64
# Continuation batches of capped sampled draws (`extend`): each row is prompt + 1,200 generated tokens,
# about 2,500 wide after left padding, and grows to about 3,350. At 64 rows the prefill's MLP
# activations alone (64 x 2,500 x 25,600 x 2 bytes x 3, about 23 GiB on one GPU) plus 15.3 GiB of weights
# and 13.4 GiB of KV cache per GPU exceed the 44.5 GiB usable; 32 rows need about 15.3 + 6.7 + 11.5 =
# 33.5 GiB.
EXTEND_BATCH = 32
K26_SAMPLE = {"reasoning_effort": "none", "temperature": 1, "top_p": 1, "max_tokens": CAP}
FINGERPRINT = ("box_type", "gpus", "driver", "cuda_runtime", "torch", "transformers", "device_map", "weights")


def sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


# ---------------------------------------------------------------- openings

def load_lines(path: Path = LINES) -> dict[str, dict]:
    """lines.json as {id: entry}; it may be a list of {id, kind, claim, ...} or a dict keyed by id."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "id" in v[0]]
        entries = lists[0] if len(lists) == 1 else [v if "id" in v else {"id": k, **v} for k, v in data.items()
                                                    if isinstance(v, dict)]
    else:
        entries = data
    out = {e["id"]: e for e in entries}
    for i, e in out.items():
        assert i in IDS, f"{path}: unknown argument id {i!r}"
        assert str(e.get("claim") or "").strip(), f"{path}: {i} has no claim"
    return out


def arm_of(arg_id: str) -> str:
    return {"DA": "denial", "CA": "control"}[arg_id[:2]]


def openings(root: Path = OPENINGS, lines_path: Path = LINES) -> list[dict]:
    """Every accepted wording on disk, in ID then slot order, with its claim and text."""
    root = Path(root)
    found = [(i, s, root / i / f"{s}.md") for i in IDS for s in SLOTS if (root / i / f"{s}.md").exists()]
    # the generator keeps its draws (<slot>_d<n>.md) and seed drafts beside the accepted files; only an
    # exact <slot>.md is an accepted wording. Draws with no accepted file at all usually mean the
    # accepted set has not been materialised yet, so say so instead of silently finding nothing.
    unaccepted = sorted({p.parent.name for p in root.glob("*/w?_d*.md")} - {i for i, _, _ in found}) if root.exists() else []
    if unaccepted:
        print(f"warning: {', '.join(unaccepted)} have wording draws under {rel(root)} but no accepted <slot>.md",
              file=sys.stderr)
    if not found:
        return []
    lines = load_lines(lines_path)
    out = []
    for i, s, p in found:
        assert i in lines, f"{rel(p)} exists but {lines_path} has no entry for {i}"
        text = load_denial(p)
        assert text, f"{rel(p)} is empty"
        out.append({"opening": f"{i}/{s}", "argument": i, "slot": s, "kind": arm_of(i),
                    "line_kind": lines[i].get("kind"), "claim": lines[i]["claim"], "path": p,
                    "text": text, "sha256": sha(text)})
    return out


def reply_path(out: Path, o: dict) -> Path:
    """Draw 0's file (the name every reply had before the draw rule)."""
    return Path(out) / f"{o['argument']}_{o['slot']}.json"


def draw_path(out: Path, o: dict, k: int) -> Path:
    return reply_path(out, o) if k == 0 else Path(out) / f"{o['argument']}_{o['slot']}_d{k}.json"


def draw_seed(opening: str, k: int) -> int:
    """The fixed seed of K2.6's sampled draw k of an opening ("DA01/w3"). Qwen's sampled draws use one
    seed per batch instead (`sampled_batches`)."""
    return int(sha(f"s2-draw:{opening}:{k}")[:8], 16)


def write_manifest(out: Path) -> dict:
    """replies.sha256.json: every reply file, the archived 1,200-cap originals as "cap1200/<name>"."""
    out = Path(out)
    files = sorted(list(out.glob("[DC]A??_w?.json")) + list(out.glob("[DC]A??_w?_d?.json")))
    files += sorted(list((out / ARCHIVE).glob("[DC]A??_w?.json")) + list((out / ARCHIVE).glob("[DC]A??_w?_d?.json")))
    manifest = {str(p.relative_to(out)): sha(p.read_bytes()) for p in files}
    write_json(out / "replies.sha256.json", manifest)
    return manifest


# ---------------------------------------------------------------- cap epochs

def rec_cap(rec: dict) -> int:
    """The new-token cap a reply was generated under: its `cap` field, else (records written before the
    field existed) its params' max_new_tokens (Qwen) or max_tokens (K2.6)."""
    if rec.get("cap") is not None:
        return int(rec["cap"])
    p = rec.get("params") or {}
    return int(p.get("max_new_tokens") or p.get("max_tokens"))


def needs_extension(rec: dict, cap: int | None = None) -> bool:
    """A reply cut off by a cap below the current one (it is extended by `extend` / `k26extend`)."""
    return rec.get("exclusion") == "capped" and rec_cap(rec) < (cap or CAP)


def verdicts_path(out: Path, cap: int | None = None) -> Path:
    """The verdict file of one cap epoch: verdicts.jsonl is the 1,200-cap epoch (kept byte for byte),
    verdicts_cap<N>.jsonl a later one."""
    cap = cap or CAP
    return Path(out) / ("verdicts.jsonl" if cap == CAP_OLD else f"verdicts_cap{cap}.jsonl")


def epoch_dir(out: Path, cap: int | None = None) -> Path:
    """Where an epoch's selected.json, tally.json and eligible.json live: the reply directory for the
    current cap, ARCHIVE for the 1,200-cap record."""
    return Path(out) if (cap or CAP) == CAP else Path(out) / ARCHIVE


def load_draw(out: Path, o: dict, k: int, cap: int | None = None) -> tuple[Path, dict | None]:
    """(file, record) of draw k as it stood in a cap epoch: for the 1,200-cap epoch the archived original
    where the reply was extended later, else the reply file itself; record None if there is no file."""
    p = draw_path(out, o, k)
    if (cap or CAP) == CAP_OLD and (Path(out) / ARCHIVE / p.name).exists():
        p = Path(out) / ARCHIVE / p.name
    return p, (json.loads(p.read_text()) if p.exists() else None)


# ---------------------------------------------------------------- Qwen: frozen replies (GPU)

def prompt_ids(tok, msgs: list[dict]) -> list[int]:
    """Study 1's prompt: `tok.apply_chat_template(msgs, add_generation_prompt=True,
    enable_thinking=False)`. Returned as a flat list whatever the transformers version returns
    (a list, a [1, n] tensor or, in newer versions, a BatchEncoding)."""
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        assert len(ids) == 1, len(ids)
        ids = ids[0]
    return [int(x) for x in ids]


def decode(tok, gen: list[int]) -> tuple[str, str]:
    """(reply, finish_reason) exactly as Study 1's `freeze` computes them."""
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    stop = any(t in stop_ids for t in gen)
    return tok.decode(gen, skip_special_tokens=True).strip(), "stop" if stop else "length"


def generate(m, tok, msgs: list[dict], max_new_tokens: int = 1200) -> dict:
    """One greedy assistant turn with Study 1's exact `generate` call (draw 0: unbatched, no padding)."""
    import torch

    ids = prompt_ids(tok, msgs)
    x = torch.tensor([ids], dtype=torch.long).to(m.device)
    t0 = time.time()
    gen = m.generate(x, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                     top_k=None)[0, x.shape[1]:].tolist()
    reply, finish = decode(tok, gen)
    return {"prompt_tokens": len(ids), "prompt_ids_sha256": sha(json.dumps(ids)), "finish_reason": finish,
            "completion_tokens": len(gen), "completion_ids": gen, "seconds": round(time.time() - t0, 1),
            "reply": reply}


def sampled_batches(ops: list[dict], batch_size: int) -> list[dict]:
    """The fixed batch plan of Qwen's sampled draws: every (argument, slot, draw) for draws 1..4,
    sorted by argument, slot, draw, cut into consecutive batches of `batch_size` (the last may be
    smaller). A batch's seed is the first 8 hex digits of SHA-256("s2-batch:" + its members joined by
    "|"), each member "<ID>/<slot>:<k>", so it follows from the batch's identity alone; its id is the
    first 12 hex digits of the same hash."""
    assert batch_size >= 1
    items = sorted(((o["argument"], o["slot"], k), o) for o in ops for k in range(1, N_DRAWS))
    plan = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        members = [f"{arg}/{slot}:{k}" for (arg, slot, k), _ in chunk]
        h = sha("s2-batch:" + "|".join(members))
        plan.append({"index": i // batch_size, "id": h[:12], "seed": int(h[:8], 16), "size": len(chunk),
                     "configured_size": batch_size, "members": members, "members_sha256": sha("|".join(members)),
                     "items": [(o, k) for (_, _, k), o in chunk]})
    return plan


def pad_id(tok) -> int:
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def left_pad(prompts: list[list[int]], pad: int) -> tuple[list[list[int]], list[list[int]]]:
    """Left-padded ids and their attention mask (0 on padding), all rows as wide as the longest."""
    width = max(len(p) for p in prompts)
    return ([[pad] * (width - len(p)) + p for p in prompts], [[0] * (width - len(p)) + [1] * len(p) for p in prompts])


def generate_batch(m, tok, prompts: list[list[int]], seed: int, max_new_tokens: int = 1200) -> list[dict]:
    """One batch of sampled draws: the prompts left-padded with the pad token and an attention mask
    (generate derives position ids from the mask), `torch.manual_seed(seed)`, then one `generate`
    call with QWEN_SAMPLE's pure sampling (temperature 1, top_p 1, top_k 0, min_p None: no logits
    warper runs, overriding the model's own generation_config defaults). Each row is cut after its
    first stop token (rows that stop early are filled with the pad token by generate)."""
    import torch

    pad = pad_id(tok)
    ids, att = left_pad(prompts, pad)
    width = len(ids[0])
    x = torch.tensor(ids, dtype=torch.long).to(m.device)
    mask = torch.tensor(att, dtype=torch.long).to(m.device)
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), pad}
    t0 = time.time()
    torch.manual_seed(seed)  # seeds the CPU and every CUDA generator
    out = m.generate(input_ids=x, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=True,
                     temperature=QWEN_SAMPLE["temperature"], top_p=QWEN_SAMPLE["top_p"], top_k=QWEN_SAMPLE["top_k"],
                     min_p=QWEN_SAMPLE["min_p"], pad_token_id=pad)[:, width:].tolist()
    secs = round(time.time() - t0, 1)
    res = []
    for p, row in zip(prompts, out):
        cut = next((j + 1 for j, t in enumerate(row) if t in stop_ids), len(row))
        gen = row[:cut]
        reply, _ = decode(tok, gen)
        # a row ends at its first stop token; the pad token counts too, since Qwen3's generation_config
        # lists <|endoftext|> (the pad token) as a second eos and generate stops a row on it
        finish = "stop" if gen and gen[-1] in stop_ids else "length"
        res.append({"prompt_tokens": len(p), "prompt_ids_sha256": sha(json.dumps(p)), "finish_reason": finish,
                    "completion_tokens": len(gen), "completion_ids": gen, "seconds": secs, "reply": reply,
                    "left_pad_tokens": width - len(p)})
    return res


def canary_compare(tok, gen: list[int], study1: dict) -> dict:
    """The canary's two tests against Study 1's frozen D1.4 reply.

    byte_identical: the regenerated reply's UTF-8 bytes equal Study 1's (and so do finish_reason
    and the completion token count). first_200: the regenerated reply has at least 200 tokens and
    the text of its first 200 is a prefix of Study 1's reply (Study 1 kept only the text, so the
    comparison is on decoded text; a split multi-byte character at the cut is trimmed). Also
    reported: how many leading tokens agree with Study 1's reply re-encoded, which is
    informational because re-encoding need not reproduce generated ids at every boundary.
    """
    reply, finish = decode(tok, gen)
    s1 = study1["reply"]
    head = tok.decode(gen[:CANARY_TOKENS], skip_special_tokens=True).lstrip().rstrip("�")
    s1_ids = tok.encode(s1, add_special_tokens=False)
    body = [t for t in gen if t not in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}]
    lead = next((k for k, (a, b) in enumerate(zip(body, s1_ids)) if a != b), min(len(body), len(s1_ids)))
    first_diff = next((k for k, (a, b) in enumerate(zip(reply, s1)) if a != b), min(len(reply), len(s1)))
    return {"byte_identical": reply.encode() == s1.encode() and finish == study1.get("finish_reason", finish)
            and len(gen) == study1.get("completion_tokens", len(gen)),
            "reply_bytes_identical": reply.encode() == s1.encode(),
            "first_200": len(gen) >= CANARY_TOKENS and s1.startswith(head),
            "leading_tokens_matching_reencoded_study1": lead, "first_differing_char": first_diff,
            "finish_reason": finish, "study1_finish_reason": study1.get("finish_reason"),
            "completion_tokens": len(gen), "study1_completion_tokens": study1.get("completion_tokens"),
            "reply_sha256": sha(reply), "study1_reply_sha256": sha(s1)}


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _http(url: str, headers: dict, method: str = "GET") -> str:
    try:
        req = urllib.request.Request(url, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=1.5) as r:
            return r.read().decode().strip()
    except (OSError, urllib.error.URLError, ValueError):
        return ""


def box_type(override: str | None) -> dict:
    """The instance type from the cloud's metadata service (AWS IMDSv2, GCP, Azure), RunPod's
    environment, or `--box`; the first that answers wins, and the source is recorded."""
    import os

    if override:
        return {"box_type": override, "box_type_source": "--box"}
    tok = _http("http://169.254.169.254/latest/api/token", {"X-aws-ec2-metadata-token-ttl-seconds": "60"}, "PUT")
    if tok:
        t = _http("http://169.254.169.254/latest/meta-data/instance-type", {"X-aws-ec2-metadata-token": tok})
        if t:
            az = _http("http://169.254.169.254/latest/meta-data/placement/availability-zone",
                       {"X-aws-ec2-metadata-token": tok})
            return {"box_type": t, "box_type_source": "aws-imds", "zone": az}
    t = _http("http://metadata.google.internal/computeMetadata/v1/instance/machine-type", {"Metadata-Flavor": "Google"})
    if t:
        return {"box_type": t.rsplit("/", 1)[-1], "box_type_source": "gcp-metadata"}
    t = _http("http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01&format=text",
              {"Metadata": "true"})
    if t:
        return {"box_type": t, "box_type_source": "azure-imds"}
    if os.environ.get("RUNPOD_POD_ID"):
        gpu = os.environ.get("RUNPOD_GPU_NAME") or os.environ.get("RUNPOD_GPU_TYPE") or "?"
        return {"box_type": f"runpod:{os.environ.get('RUNPOD_GPU_COUNT', '?')}x{gpu}", "box_type_source": "runpod-env"}
    return {"box_type": f"unknown ({socket.gethostname()})", "box_type_source": "hostname"}


def weight_hashes(model_path: Path, full: bool) -> dict:
    """SHA-256 of the small files that define the model and, with `full`, of every weight shard
    (hashed in parallel; about a minute for 64 GB on NVMe)."""
    small = ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.json", "merges.txt"]
    out = {f: sha((model_path / f).read_bytes()) for f in small if (model_path / f).exists()}
    if full:
        def h(p: Path) -> tuple[str, str]:
            d = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    d.update(chunk)
            return p.name, d.hexdigest()

        with ThreadPoolExecutor(8) as ex:
            out.update(dict(ex.map(h, sorted(model_path.glob("*.safetensors")))))
    return out


def box_stamp(m, model_path: Path, box: str | None, full_hash: bool) -> dict:
    import torch
    import transformers

    gpus = [{"name": torch.cuda.get_device_name(i), "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 2 ** 30, 1)}
            for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    smi = _sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,pci.bus_id", "--format=csv,noheader"])
    smi_cuda = re.search(r"CUDA Version:\s*([\d.]+)", _sh(["nvidia-smi"]) or "")
    try:
        import accelerate
        acc = accelerate.__version__
    except ImportError:
        acc = None
    dm = getattr(m, "hf_device_map", None)
    return {**box_type(box), "hostname": socket.gethostname(), "gpus": gpus, "nvidia_smi": smi.splitlines(),
            "driver": (smi.splitlines()[0].split(", ")[1] if smi else None),
            "cuda_driver_api": smi_cuda.group(1) if smi_cuda else None, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "torch": torch.__version__, "transformers": transformers.__version__, "accelerate": acc,
            "python": platform.python_version(), "platform": platform.platform(),
            "device_map": {k: str(v) for k, v in dm.items()} if dm else str(m.device),
            "dtype": str(m.dtype), "model_path": str(model_path), "revision_pinned": QWEN_REV,
            "weights": weight_hashes(model_path, full_hash)}


def stage_qwen(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(a.model).expanduser()
    cfg = json.loads((model_path / "config.json").read_text())
    shape = {k: cfg.get(k) for k in QWEN_SHAPE}
    assert shape == QWEN_SHAPE, f"{model_path} is not Qwen3-32B: {shape}"
    out = Path(a.out or REPLIES / QWEN_TAG)
    out.mkdir(parents=True, exist_ok=True)
    ops = openings(Path(a.openings), Path(a.lines))
    print(f"{len(ops)} accepted wordings under {rel(Path(a.openings))}", flush=True)

    # Study 1's load, exactly (runs/qwen_pilot.py, assent stage)
    tok = AutoTokenizer.from_pretrained(str(model_path))
    m = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.bfloat16, device_map="auto")
    stamp = box_stamp(m, model_path, a.box, not a.no_weight_hash)
    stamp_path = out / "box.json"
    if stamp_path.exists():
        old = json.loads(stamp_path.read_text())
        diff = {k: (old.get(k), stamp.get(k)) for k in FINGERPRINT if old.get(k) != stamp.get(k)
                and not (k == "weights" and a.no_weight_hash)}
        assert not diff, (f"{stamp_path} was stamped on a different box or stack; Study 2 replies are "
                          f"generated in one box session (spec section 4). Differences: {diff}")
        session = old["session"]
        old.setdefault("resumed", []).append(now())
        stamp = old
    else:
        session = uuid.uuid4().hex[:12]
        stamp = {"session": session, "started": now(), **stamp}
    write_json(stamp_path, stamp)
    print(f"box {stamp['box_type']} ({stamp['box_type_source']}), {len(stamp['gpus'])} GPU(s) "
          f"{stamp['gpus'][0]['name'] if stamp['gpus'] else '-'}, driver {stamp['driver']}, CUDA {stamp['cuda_runtime']}, "
          f"torch {stamp['torch']}, transformers {stamp['transformers']}, session {session}", flush=True)

    # determinism canary first: Study 1's D1.4 opening, regenerated on this box
    canary_path = out / "canary.json"
    if canary_path.exists() and json.loads(canary_path.read_text()).get("session") == session:
        canary = json.loads(canary_path.read_text())
    else:
        study1 = json.loads(STUDY1.read_text())
        user = load_denial(ROOT / f"prompts/denial/{STUDY1_STEM}.md")
        g = generate(m, tok, [{"role": "user", "content": user}], STUDY1_PARAMS["max_new_tokens"])
        canary = {"session": session, "created": now(), "argument": STUDY1_STEM, "study1_file": rel(STUDY1),
                  "study1_file_sha256": sha(STUDY1.read_bytes()), "model": "Qwen/Qwen3-32B", "revision": QWEN_REV,
                  "params": STUDY1_PARAMS, **canary_compare(tok, g["completion_ids"], study1), "generated": g}
        canary["status"] = ("pass" if canary["byte_identical"] else
                            "pending_judge" if canary["first_200"] else "fail")
        write_json(canary_path, canary)
    print(f"canary: byte_identical={canary['byte_identical']} first_200={canary['first_200']} "
          f"(first differing char {canary['first_differing_char']}, {canary['completion_tokens']} vs "
          f"{canary['study1_completion_tokens']} tokens) -> {canary['status']}", flush=True)
    if canary["status"] == "fail" and not a.ignore_canary:
        sys.exit("canary failed: not byte-identical and the first 200 tokens differ. Debug the Qwen reply "
                 "code before generating any frozen reply (spec section 4); --ignore-canary overrides.")

    # draw 0: one wording at a time, unbatched, greedy, no padding (Study 1's call)
    todo = [o for o in ops if not reply_path(out, o).exists()]
    print(f"{len(ops) - len(todo)} draw-0 replies already frozen, {len(todo)} to generate", flush=True)
    t_all = time.time()
    for n, o in enumerate(todo, 1):
        g = generate(m, tok, [{"role": "user", "content": o["text"]}], QWEN_PARAMS["max_new_tokens"])
        capped = g["finish_reason"] == "length"
        rec = {"opening": o["opening"], "argument": o["argument"], "slot": o["slot"], "kind": o["kind"],
               "draw": 0, "seed": None, "opening_path": rel(o["path"]), "opening_sha256": o["sha256"],
               "model": "Qwen/Qwen3-32B", "revision": QWEN_REV, "local_path": str(model_path), "params": QWEN_PARAMS,
               "session": session, "canary_status_at_generation": canary["status"],
               "ignore_canary": bool(a.ignore_canary), "created": now(), **g, "cap": QWEN_PARAMS["max_new_tokens"],
               "excluded": capped, "exclusion": "capped" if capped else None}
        write_json(reply_path(out, o), rec)
        eta = (time.time() - t_all) / n * (len(todo) - n)
        print(f"[{n}/{len(todo)}] {o['opening']} draw 0: {len(g['reply'].split())} words, {g['finish_reason']}, "
              f"{g['seconds']:.0f}s, eta {eta / 60:.0f} min", flush=True)

    # draws 1..4: every wording, up front (there is no judge on the box), in the fixed batch plan
    plan = sampled_batches(ops, a.batch_size)
    frozen_plan = [{k: b[k] for k in ("index", "id", "seed", "size", "configured_size", "members", "members_sha256")}
                   for b in plan]
    plan_path = out / "batches.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        assert old["batches"] == frozen_plan, (
            f"{plan_path} holds a different batch plan (batch size {old['batch_size']} vs {a.batch_size}, or a "
            f"different set of wordings); the sampled draws of one study are made under one plan")
    else:
        write_json(plan_path, {"created": now(), "session": session, "batch_size": a.batch_size,
                               "params": QWEN_SAMPLE, "batches": frozen_plan})
    todo = [b for b in plan if not all(draw_path(out, o, k).exists() for o, k in b["items"])]
    print(f"sampled draws: {sum(b['size'] for b in plan)} in {len(plan)} batches of up to {a.batch_size}; "
          f"{len(plan) - len(todo)} batches already frozen, {len(todo)} to generate", flush=True)
    t_all = time.time()
    for n, b in enumerate(todo, 1):
        prompts = [prompt_ids(tok, [{"role": "user", "content": o["text"]}]) for o, _ in b["items"]]
        gs = generate_batch(m, tok, prompts, b["seed"], QWEN_SAMPLE["max_new_tokens"])
        for (o, k), g in zip(b["items"], gs):
            if draw_path(out, o, k).exists():
                continue  # a partly written batch: the existing file stays as it is
            capped = g["finish_reason"] == "length"
            pad = g.pop("left_pad_tokens")
            rec = {"opening": o["opening"], "argument": o["argument"], "slot": o["slot"], "kind": o["kind"],
                   "draw": k, "seed": b["seed"], "padded": True,
                   "batch": {"id": b["id"], "index": b["index"], "size": b["size"], "configured_size": b["configured_size"],
                             "seed": b["seed"], "members_sha256": b["members_sha256"], "left_pad_tokens": pad,
                             "plan": rel(plan_path)},
                   "opening_path": rel(o["path"]), "opening_sha256": o["sha256"], "model": "Qwen/Qwen3-32B",
                   "revision": QWEN_REV, "local_path": str(model_path), "params": QWEN_SAMPLE, "session": session,
                   "canary_status_at_generation": canary["status"], "ignore_canary": bool(a.ignore_canary),
                   "created": now(), **g, "cap": QWEN_SAMPLE["max_new_tokens"], "excluded": capped,
                   "exclusion": "capped" if capped else None}
            write_json(draw_path(out, o, k), rec)
        eta = (time.time() - t_all) / n * (len(todo) - n)
        print(f"[batch {n}/{len(todo)}] #{b['index']} {b['id']} ({b['size']} draws, seed {b['seed']}): "
              f"{sum(g['finish_reason'] == 'length' for g in gs)} capped, {gs[0]['seconds']:.0f}s, eta {eta / 60:.0f} min",
              flush=True)
    stamp["finished"] = now()
    write_json(stamp_path, stamp)
    manifest = write_manifest(out)
    print(f"done: {len(manifest)} reply files in {out}; copy the directory to the laptop and run "
          f"`python runs/s2_assent.py canary`", flush=True)


# ---------------------------------------------------------------- raising the cap: extending capped replies

EXTEND_NOTE = ("tokens past the original cap come from a continuation batch with its own seed (extension.batch); "
               "the first tokens are the original batch's draw. A continued sample is still a sample from the "
               "same distribution.")


def is_extended(rec: dict) -> bool:
    return (rec.get("extension") or {}).get("cap") == CAP


def extension_targets(out: Path, ops: list[dict], draws: list[int] | None = None,
                      needed: set[str] | None = None) -> list[tuple[dict, int]]:
    """Every draw on disk that was capped below CAP: the ones still to extend and the ones already
    extended (their original is archived), sorted by argument, slot, draw, so a resumed run finds the
    same list. `draws` keeps only those draw indices (round 1: [0]); `needed` keeps only those members
    ("<ID>/<slot>:<k>", round 2: extend_needed.json)."""
    out = Path(out)
    res = []
    for o in ops:
        for k in range(N_DRAWS):
            p = draw_path(out, o, k)
            if not p.exists() or (draws is not None and k not in draws) or \
                    (needed is not None and f"{o['opening']}:{k}" not in needed):
                continue
            if (out / ARCHIVE / p.name).exists() or needs_extension(json.loads(p.read_text())):
                res.append((o, k))
    return sorted(res, key=lambda t: (t[0]["argument"], t[0]["slot"], t[1]))


def load_needed(path: Path, out: Path, ops: list[dict]) -> set[str]:
    """The members of an extend_needed.json (written by `tally`), each checked to be a draw on disk that is
    capped below CAP or already extended."""
    d = json.loads(Path(path).read_text())
    vp = verdicts_path(out)
    assert d["cap"] == CAP and d["verdicts_sha256"] == (sha(vp.read_bytes()) if vp.exists() else None), (
        f"{path} is stale: it was written from another {vp.name}; rerun `tally` on the laptop and sync")
    members = set(d["members"])
    have = {f"{o['opening']}:{k}" for o, k in extension_targets(out, ops)}
    bad = sorted(members - have)
    assert not bad, f"{path}: members that are not capped draws on disk: {bad[:10]}"
    return members


def needed_draws(out: Path, ops: list[dict], walks: dict[str, dict]) -> dict:
    """Round 2's list (spec section 4, "Reply cap"): for every wording whose 2,048-cap walk stops at a
    reply still capped at CAP_OLD ("unextended"), every draw of it still capped at CAP_OLD, so one more
    round extends all a later walk can reach. The other draws still capped at CAP_OLD are "not needed":
    never extended, never graded."""
    members, not_needed = [], []
    for o in ops:
        for k in range(N_DRAWS):
            p = draw_path(out, o, k)
            if p.exists() and needs_extension(json.loads(p.read_text())):
                m = f"{o['opening']}:{k}"
                (members if (walks.get(o["opening"]) or {}).get("status") == "unextended" else not_needed).append(m)
    order = lambda m: (m[:4], m[5:7], int(m.rsplit(":", 1)[1]))  # noqa: E731  argument, slot, draw
    return {"wordings": sorted({m.split(":")[0] for m in members}), "members": sorted(members, key=order),
            "not_needed": sorted(not_needed, key=order)}


def archive_original(out: Path, p: Path) -> tuple[Path, dict]:
    """The capped original of a reply file, copied byte for byte to ARCHIVE before it is extended (once)."""
    dst = Path(out) / ARCHIVE / p.name
    if not dst.exists():
        rec = json.loads(p.read_text())
        assert needs_extension(rec) and not is_extended(rec), f"{p.name}: not a capped original"
        dst.parent.mkdir(exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        tmp.write_bytes(p.read_bytes())
        tmp.replace(dst)
    return dst, json.loads(dst.read_text())


def extension_info(orig: dict, orig_path: Path, out: Path, mode: str, total: int, new: int, finish: str | None,
                   session: str, seconds: float | None, **extra) -> dict:
    return {"cap": CAP, "original_cap": rec_cap(orig), "original_completion_tokens": orig.get("completion_tokens"),
            "original_finish_reason": orig.get("finish_reason"), "original_reply_sha256": sha(orig.get("reply") or ""),
            "original_file": str(orig_path.relative_to(out)), "original_file_sha256": sha(orig_path.read_bytes()),
            "completion_tokens": total, "new_tokens": new, "finish_reason": finish, "mode": mode,
            "session": session, "created": now(), "seconds": seconds, **extra}


def qwen_extended(orig: dict, orig_path: Path, out: Path, tok, ids: list[int], finish: str, mode: str,
                  session: str, seconds: float, **extra) -> dict:
    """The record of a Qwen reply under CAP: the original's fields, its completion now `ids` (the
    original tokens plus the continuation, or a fresh greedy run), and `extension` describing how."""
    reply, _ = decode(tok, ids)
    new = len(ids) - orig["completion_tokens"] if mode == "continue" else len(ids)
    capped = finish == "length"
    return {**orig, "cap": CAP, "params": {**orig["params"], "max_new_tokens": CAP}, "completion_ids": ids,
            "completion_tokens": len(ids), "reply": reply, "finish_reason": finish, "excluded": capped,
            "exclusion": "capped" if capped else None,
            "extension": extension_info(orig, orig_path, out, mode, len(ids), new, finish, session, seconds, **extra)}


def continue_greedy(m, tok, prompt: list[int], done: list[int], max_new_tokens: int) -> dict:
    """Draw 0 continued: Study 1's unbatched greedy `generate` call on prompt + the tokens already
    generated, for the remaining budget (no padding, no attention mask, as draw 0 was made)."""
    import torch

    x = torch.tensor([prompt + done], dtype=torch.long).to(m.device)
    t0 = time.time()
    gen = m.generate(x, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                     top_k=None)[0, x.shape[1]:].tolist()
    full = done + gen
    return {"completion_ids": full, "finish_reason": decode(tok, full)[1], "seconds": round(time.time() - t0, 1)}


def extend_batches(items: list[tuple[dict, int]], batch_size: int) -> list[dict]:
    """The fixed plan of the continuation batches: the capped sampled draws sorted by argument, slot,
    draw, cut into consecutive batches of `batch_size`; seed and id from SHA-256("s2-extend-batch:" +
    members joined by "|"), as `sampled_batches` does for the original batches."""
    assert batch_size >= 1
    items = sorted(items, key=lambda t: (t[0]["argument"], t[0]["slot"], t[1]))
    plan = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        members = [f"{o['opening']}:{k}" for o, k in chunk]
        h = sha("s2-extend-batch:" + "|".join(members))
        plan.append({"index": i // batch_size, "id": h[:12], "seed": int(h[:8], 16), "size": len(chunk),
                     "configured_size": batch_size, "members": members, "members_sha256": sha("|".join(members)),
                     "items": chunk})
    return plan


def stage_extend(a) -> None:
    """Qwen3-32B on the box: every reply capped at CAP_OLD continued to CAP new tokens (owner decision of
    2026-10-12). Draw 0: Study 1's unbatched greedy call on prompt + its tokens, so the reply equals one
    greedy run at CAP, which is checked first: the first `--check-n` capped draw-0 wordings are also
    regenerated from scratch at CAP and compared token for token. If any differs, every capped draw 0
    is regenerated from scratch instead (mode "regenerate", recorded with whether its first tokens match
    the original). Sampled draws: continued in fixed batches (`extend_batches`, left padding and an
    attention mask, QWEN_SAMPLE's pure sampling, one seed per batch). Originals are archived byte for
    byte under ARCHIVE; each extended file records `extension`. Replies still capped at CAP stay excluded."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(a.model).expanduser()
    cfg = json.loads((model_path / "config.json").read_text())
    assert {k: cfg.get(k) for k in QWEN_SHAPE} == QWEN_SHAPE, f"{model_path} is not Qwen3-32B"
    out = Path(a.out or REPLIES / QWEN_TAG)
    box_path = out / "box.json"
    assert box_path.exists(), f"{box_path}: no reply box stamp; the replies to extend are not here"
    ops = openings(Path(a.openings), Path(a.lines))
    draws = [int(x) for x in a.draws.split(",")] if getattr(a, "draws", None) else None
    needed = load_needed(Path(a.needed), out, ops) if getattr(a, "needed", None) else None
    assert draws is None or needed is None, "--draws and --needed are separate rounds"
    rnd = "round1_finished" if draws == [0] else "round2_finished" if needed is not None else "finished"
    targets = extension_targets(out, ops, draws, needed)
    d0 = [o for o, k in targets if k == 0]
    sampled = [(o, k) for o, k in targets if k > 0]
    print(f"{len(targets)} replies capped at {CAP_OLD} to extend to {CAP}"
          + (f" (draws {draws})" if draws is not None else f" (listed in {a.needed})" if needed is not None else "")
          + f": {len(d0)} draw 0, {len(sampled)} sampled", flush=True)
    ext_path = out / "extend.json"
    if not targets:
        if ext_path.exists():
            ext = json.loads(ext_path.read_text())
            ext[rnd] = {"at": now(), "extended": 0, "needed": rel(Path(a.needed)) if needed is not None else None}
            write_json(ext_path, ext)
        return

    tok = AutoTokenizer.from_pretrained(str(model_path))
    m = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.bfloat16, device_map="auto")
    stamp = box_stamp(m, model_path, a.box, not a.no_weight_hash)
    reply_box = json.loads(box_path.read_text())
    diff = {k: (reply_box.get(k), stamp.get(k)) for k in FINGERPRINT if reply_box.get(k) != stamp.get(k)
            and not (k == "weights" and a.no_weight_hash)}
    if diff and not a.allow_stack_change:
        sys.exit(f"this box or stack differs from the one that made the replies ({box_path}): {diff}. The "
                 f"continuations must run on the same setup; --allow-stack-change overrides (recorded).")
    if ext_path.exists():
        ext = json.loads(ext_path.read_text())
        assert ext["cap"] == CAP, f"{ext_path} extends to {ext['cap']}, not {CAP}"
        ext.setdefault("resumed", []).append({"at": now(), "stack_diff_vs_reply_box": diff,
                                              "allow_stack_change": bool(a.allow_stack_change), "box": stamp})
    else:
        ext = {"cap": CAP, "original_cap": CAP_OLD, "session": uuid.uuid4().hex[:12], "started": now(),
               "reply_box_session": reply_box.get("session"), "stack_diff_vs_reply_box": diff,
               "allow_stack_change": bool(a.allow_stack_change), "box": stamp}
    session = ext["session"]
    write_json(ext_path, ext)
    print(f"box {stamp['box_type']}, {len(stamp['gpus'])} GPU(s), torch {stamp['torch']}, transformers "
          f"{stamp['transformers']}; stack differences from the reply box: {diff or 'none'}; session {session}", flush=True)
    orig = {}
    for o, k in targets:
        orig[(o["opening"], k)] = archive_original(out, draw_path(out, o, k))

    def prompt_for(o: dict, rec: dict) -> list[int]:
        ids = prompt_ids(tok, [{"role": "user", "content": o["text"]}])
        assert sha(json.dumps(ids)) == rec["prompt_ids_sha256"], f"{o['opening']}: the prompt ids changed"
        return ids

    def done(o: dict, k: int) -> bool:
        return is_extended(json.loads(draw_path(out, o, k).read_text()))

    # draw 0, the check first: continuation vs one fresh greedy run at CAP
    if d0 and "draw0_check" not in ext:
        results, writes = [], {}
        for o in d0[:a.check_n]:
            op, rec = orig[(o["opening"], 0)]
            ids = prompt_for(o, rec)
            c = continue_greedy(m, tok, ids, rec["completion_ids"], CAP - rec["completion_tokens"])
            f = generate(m, tok, [{"role": "user", "content": o["text"]}], CAP)
            same = c["completion_ids"] == f["completion_ids"]
            cr, fr = decode(tok, c["completion_ids"])[0], f["reply"]
            first = next((j for j, (x, y) in enumerate(zip(c["completion_ids"], f["completion_ids"])) if x != y),
                         None if same else min(len(c["completion_ids"]), len(f["completion_ids"])))
            results.append({"opening": o["opening"], "token_identical": same,
                            "reply_bytes_identical": cr.encode() == fr.encode(),
                            "fresh_first_tokens_match_original": f["completion_ids"][:rec["completion_tokens"]] == rec["completion_ids"],
                            "first_differing_token": first, "continuation_tokens": len(c["completion_ids"]),
                            "fresh_tokens": f["completion_tokens"], "continuation_finish": c["finish_reason"],
                            "fresh_finish": f["finish_reason"], "continuation_reply_sha256": sha(cr),
                            "fresh_reply_sha256": sha(fr), "continuation_seconds": c["seconds"],
                            "fresh_seconds": f["seconds"]})
            writes[o["opening"]] = (c, f)
            print(f"draw-0 check {o['opening']}: continuation vs fresh run at {CAP}: token-identical {same}, "
                  f"bytes identical {cr.encode() == fr.encode()} ({len(c['completion_ids'])} vs {f['completion_tokens']} "
                  f"tokens, {c['seconds']:.0f}s vs {f['seconds']:.0f}s)", flush=True)
        ok = all(r["token_identical"] and r["reply_bytes_identical"] for r in results)
        mode = "continue" if ok else "regenerate"
        ext["draw0_check"] = {"created": now(), "n": len(results), "identical": ok if results else None,
                              "mode": mode, "results": results}
        write_json(ext_path, ext)
        for o in d0[:a.check_n]:
            op, rec = orig[(o["opening"], 0)]
            c, f = writes[o["opening"]]
            if mode == "continue":
                new = qwen_extended(rec, op, out, tok, c["completion_ids"], c["finish_reason"], "continue", session,
                                    c["seconds"], checked_against_fresh_run=True)
            else:
                new = qwen_extended(rec, op, out, tok, f["completion_ids"], f["finish_reason"], "regenerate", session,
                                    f["seconds"], prefix_matches_original=f["completion_ids"][:rec["completion_tokens"]]
                                    == rec["completion_ids"])
            write_json(draw_path(out, o, 0), new)
        if not ok:
            print(f"draw-0 check FAILED: the continuation is not what one greedy run at {CAP} gives; every capped "
                  f"draw 0 is regenerated from scratch at {CAP} instead (about {CAP / 10 / 60:.1f} min each)", flush=True)
    mode = (ext.get("draw0_check") or {}).get("mode", "continue")
    todo = [o for o in d0 if not done(o, 0)]
    print(f"draw 0 ({mode}): {len(d0) - len(todo)} extended, {len(todo)} to go", flush=True)
    t_all = time.time()
    for n, o in enumerate(todo, 1):
        op, rec = orig[(o["opening"], 0)]
        ids = prompt_for(o, rec)
        if mode == "continue":
            c = continue_greedy(m, tok, ids, rec["completion_ids"], CAP - rec["completion_tokens"])
            new = qwen_extended(rec, op, out, tok, c["completion_ids"], c["finish_reason"], mode, session, c["seconds"])
        else:
            f = generate(m, tok, [{"role": "user", "content": o["text"]}], CAP)
            new = qwen_extended(rec, op, out, tok, f["completion_ids"], f["finish_reason"], mode, session, f["seconds"],
                                prefix_matches_original=f["completion_ids"][:rec["completion_tokens"]] == rec["completion_ids"])
        write_json(draw_path(out, o, 0), new)
        eta = (time.time() - t_all) / n * (len(todo) - n)
        print(f"[{n}/{len(todo)}] {o['opening']} draw 0: {new['extension']['new_tokens']} new tokens, "
              f"{new['finish_reason']}, {new['extension']['seconds']:.0f}s, eta {eta / 60:.0f} min", flush=True)

    # sampled draws: continuation batches
    plan = extend_batches(sampled, a.batch_size)
    frozen = [{k: b[k] for k in ("index", "id", "seed", "size", "configured_size", "members", "members_sha256")}
              for b in plan]
    plan_path = out / "extend_batches.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        assert old["batches"] == frozen, (f"{plan_path} holds a different continuation plan (batch size "
                                          f"{old['batch_size']} vs {a.batch_size}, or other capped draws)")
    elif plan:
        write_json(plan_path, {"created": now(), "session": session, "batch_size": a.batch_size, "cap": CAP,
                               "params": QWEN_SAMPLE, "needed": rel(Path(a.needed)) if needed is not None else None,
                               "batches": frozen})
    todo = [b for b in plan if not all(done(o, k) for o, k in b["items"])]
    print(f"sampled draws: {len(sampled)} in {len(plan)} continuation batches of up to {a.batch_size}; "
          f"{len(plan) - len(todo)} done, {len(todo)} to go", flush=True)
    t_all = time.time()
    for n, b in enumerate(todo, 1):
        recs = [orig[(o["opening"], k)] for o, k in b["items"]]
        n_old = {r["completion_tokens"] for _, r in recs}
        assert len(n_old) == 1, f"batch {b['id']}: originals of different lengths {n_old}"
        prompts = [prompt_for(o, r) + r["completion_ids"] for (o, _), (_, r) in zip(b["items"], recs)]
        gs = generate_batch(m, tok, prompts, b["seed"], CAP - n_old.pop())
        for (o, k), (op, rec), g in zip(b["items"], recs, gs):
            if done(o, k):
                continue
            ids = rec["completion_ids"] + g["completion_ids"]
            new = qwen_extended(rec, op, out, tok, ids, g["finish_reason"], "continue", session, g["seconds"],
                                seed=b["seed"], note=EXTEND_NOTE,
                                batch={"id": b["id"], "index": b["index"], "size": b["size"],
                                       "configured_size": b["configured_size"], "seed": b["seed"],
                                       "members_sha256": b["members_sha256"], "left_pad_tokens": g["left_pad_tokens"],
                                       "plan": rel(plan_path)})
            write_json(draw_path(out, o, k), new)
        eta = (time.time() - t_all) / n * (len(todo) - n)
        print(f"[batch {n}/{len(todo)}] #{b['index']} {b['id']} ({b['size']} draws, seed {b['seed']}): "
              f"{sum(g['finish_reason'] == 'length' for g in gs)} still capped at {CAP}, {gs[0]['seconds']:.0f}s, "
              f"eta {eta / 60:.0f} min", flush=True)
    recs = [json.loads(draw_path(out, o, k).read_text()) for o, k in targets]
    summary = {"extended": len(recs), "still_capped": sum(r["finish_reason"] == "length" for r in recs),
               "draw0_still_capped": sum(r["finish_reason"] == "length" for r in recs if r["draw"] == 0)}
    ext[rnd] = {"at": now(), **summary}
    if needed is not None:
        ext[rnd].update(needed=rel(Path(a.needed)), needed_sha256=sha(Path(a.needed).read_bytes()))
    ext["summary"] = summary
    write_json(ext_path, ext)
    write_manifest(out)
    print(f"done ({rnd}): {summary}; copy {rel(out)} to the laptop (scripts/s2_fetch.sh), then "
          f"`python runs/s2_assent.py grade {QWEN_TAG}`", flush=True)


K26_NO_PREFIX = ("Foundry's chat API for Kimi-K2.6 does not continue an assistant message (checked 2026-10-12 on a "
                 "capped draw 0: a trailing assistant turn, plain or with partial: true or prefix: true, is answered as "
                 "a new turn; continue_final_message is refused with HTTP 400 unrecognized_request_argument; the "
                 "/completions API returns 404 api_not_supported), so each capped reply is regenerated whole at the "
                 "new cap with the draw's original settings.")


def stage_k26extend(a) -> None:
    """K2.6: every reply capped at CAP_OLD regenerated whole at max_tokens CAP with its draw's original
    settings (temperature 0 for draw 0; temperature 1, top_p 1 and the recorded seed for a sampled
    draw), because Foundry cannot continue an assistant message (K26_NO_PREFIX). Originals are archived
    under ARCHIVE; the new file records `extension` (mode "regenerate"). Same retry rules as `k26`."""
    out = Path(a.out or REPLIES / K26_TAG)
    ops = openings(Path(a.openings), Path(a.lines))
    targets = extension_targets(out, ops)
    todo = [(o, k) for o, k in targets if not is_extended(json.loads(draw_path(out, o, k).read_text()))]
    dep = deployment_version(K26)
    session = uuid.uuid4().hex[:12]
    print(f"{len(targets)} K2.6 replies capped at {CAP_OLD}; {len(targets) - len(todo)} regenerated already, "
          f"{len(todo)} to regenerate at max_tokens {CAP} ({K26} model version {dep})", flush=True)

    def one(t):
        o, k = t
        p = draw_path(out, o, k)
        op, rec = archive_original(out, p)
        params = {**rec["params"], "max_tokens": CAP}
        msgs = [{"role": "user", "content": o["text"]}]
        assert rec["request"]["messages"] == msgs, f"{p.name}: the request differs from the opening"
        t0 = time.time()
        new = k26_record(o, k, rec.get("seed"), params, dep, k26_call(msgs, params=params))
        new["extension"] = extension_info(rec, op, out, "regenerate", new["completion_tokens"], new["completion_tokens"],
                                          new["finish_reason"], session, round(time.time() - t0, 1), why=K26_NO_PREFIX)
        write_json(p, new)
        print(f"{o['opening']} draw {k}: {new['finish_reason']}, {new['completion_tokens']} tokens (was "
              f"{rec['completion_tokens']}, capped){', EXCLUDED ' + new['exclusion'] if new['excluded'] else ''}", flush=True)
        return new

    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(one, todo))
    write_manifest(out)
    print(f"done: {len(recs)} regenerated, {sum(r['finish_reason'] == 'length' for r in recs)} still capped at {CAP}; "
          f"then `grade {K26_TAG}`", flush=True)


# ---------------------------------------------------------------- canary, judge half (laptop)

def study1_verdict() -> str:
    """Study 1's recorded verdict for its D1.4 Qwen reply (cleanliness.jsonl, last line wins)."""
    from kimisroom.judge_assent import OUT

    v = None
    for line in OUT.read_text().splitlines():
        d = json.loads(line) if line.strip() else {}
        if d.get("file") == STUDY1.name:
            v = d["verdict"]
    assert v, f"no Study 1 verdict for {STUDY1.name} in {OUT}"
    return v


def log_deviation(path: Path, title: str, body: str) -> None:
    new = not path.exists()
    with path.open("a") as f:
        if new:
            f.write("# Study 2 deviations\n\nDated entries, committed before results are read (spec section 11).\n")
        f.write(f"\n## {time.strftime('%Y-%m-%d')}: {title}\n\n{body.strip()}\n")


def stage_canary(a) -> None:
    from kimisroom.judge_assent import judge_prompt, study1_prompt

    out = Path(a.out or REPLIES / QWEN_TAG)
    c = json.loads((out / "canary.json").read_text())
    if c["byte_identical"]:
        c["status"] = "pass"
    elif c.get("judge") is None:
        ref = study1_verdict()
        new = judge_prompt(study1_prompt(STUDY1_STEM, c["generated"]["reply"]))
        c["judge"] = {**new, "study1_verdict": ref}
        c["status"] = "pass_fallback" if c["first_200"] and new["verdict"] == ref else "fail"
        box = json.loads((out / "box.json").read_text())
        log_deviation(Path(a.deviations), "determinism canary not byte-identical", f"""
The Study 1 D1.4 Qwen3-32B reply regenerated on the stamped box (session {c['session']}, {box['box_type']},
{len(box['gpus'])} x {box['gpus'][0]['name'] if box['gpus'] else '-'}, driver {box['driver']}, CUDA {box['cuda_runtime']},
torch {box['torch']}, transformers {box['transformers']}) is not byte-identical to `{c['study1_file']}`
(sha256 {c['study1_file_sha256']}). First differing character: {c['first_differing_char']}; {c['completion_tokens']}
completion tokens vs Study 1's {c['study1_completion_tokens']}; finish {c['finish_reason']} vs {c['study1_finish_reason']}.
First {CANARY_TOKENS} tokens match: {c['first_200']}. Stamped judge ({new['judge']}, {new['model_version']}) on the regenerated
reply: {new['verdict']} (Study 1's verdict: {ref}). Canary result under the spec's fallback: {c['status']}.
Study 1's reply was decoded with device_map=auto on an unrecorded box, so a mismatch is possible without a bug.""")
    write_json(out / "canary.json", c)
    print(f"canary: {c['status']} (byte_identical={c['byte_identical']}, first_200={c['first_200']}"
          + (f", verdict {c['judge']['verdict']} vs Study 1 {c['judge']['study1_verdict']}" if c.get("judge") else "") + ")")
    if c["status"] == "fail":
        sys.exit("canary failed: debug the Qwen reply code before any frozen reply is used (spec section 4)")


# ---------------------------------------------------------------- K2.6 through Foundry

def deployment_version(name: str) -> str | None:
    """The deployment's model version as Azure reports it (best effort, recorded beside each reply)."""
    v = _sh(["az", "cognitiveservices", "account", "deployment", "show", "-g", "polysem-foundry", "-n",
             "polysem-foundry-jk", "--deployment-name", name, "--query", "properties.model.version", "-o", "tsv"])
    return v or None


def post_once(body: dict, timeout: float = 300) -> tuple[int, str, dict]:
    """One request to the Foundry chat endpoint, no retries: (HTTP status, raw body, headers).
    Status 0 is a network error or timeout, with the error text as the body."""
    from kimisroom.foundry import ENDPOINT, _auth

    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), method="POST",
                                 headers={**_auth(), "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace"), dict(e.headers or {})
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, f"{type(e).__name__}: {e}", {}


def classify(status: int, raw: str) -> tuple[str, dict | None]:
    """ok | content_filter | throttled | api_error, and the parsed response when there is one."""
    try:
        r = json.loads(raw)
    except json.JSONDecodeError:
        r = None
    if status == 429:
        return "throttled", r
    if status == 200 and isinstance(r, dict) and r.get("choices"):
        return ("content_filter" if r["choices"][0].get("finish_reason") == "content_filter" else "ok"), r
    if status in (400, 403) and re.search(r"content_filter|ResponsibleAIPolicyViolation|content management policy", raw, re.IGNORECASE):
        return "content_filter", r
    return "api_error", r


def k26_call(messages: list[dict], deployment: str = K26, params: dict | None = None) -> dict:
    """One opening's reply: identical request up to three times on an API error or content filter,
    429s waited out without counting. Every attempt's raw body is kept."""
    body = {"model": deployment, "messages": messages, **(params or K26_PARAMS)}
    attempts, counted, waits = [], 0, 0
    while counted < K26_ATTEMPTS:
        t0, started = time.time(), now()
        status, raw, headers = post_once(body)
        outcome, r = classify(status, raw)
        attempts.append({"started": started, "seconds": round(time.time() - t0, 1), "status": status,
                         "outcome": outcome, "raw": raw})
        if outcome == "ok":
            return {"request": body, "attempts": attempts, "response": r}
        if outcome == "throttled" and waits < THROTTLE_WAITS:
            waits += 1
            time.sleep(float((headers or {}).get("retry-after") or 0) or min(60, 2 ** min(waits, 6)))
            continue
        counted += 1
        if counted < K26_ATTEMPTS:
            time.sleep(2 * counted)
    return {"request": body, "attempts": attempts, "response": None}


def freeze_k26(o: dict, out: Path, dep_version: str | None, k: int = 0) -> dict:
    """Draw k of one opening: draw 0 with Study 1's parameters, a sampled draw with K26_SAMPLE and its
    seed. The retry, 429 and cap rules are the same for every draw."""
    seed = draw_seed(o["opening"], k) if k else None
    params = {**K26_SAMPLE, "seed": seed} if k else K26_PARAMS
    path = draw_path(out, o, k)
    assert not path.exists(), f"{path.name} exists; a frozen draw is never regenerated"
    rec = k26_record(o, k, seed, params, dep_version, k26_call([{"role": "user", "content": o["text"]}], params=params))
    write_json(path, rec)
    return rec


def k26_record(o: dict, k: int, seed: int | None, params: dict, dep_version: str | None, res: dict) -> dict:
    """A K2.6 draw's record from `k26_call`'s result: reply, finish reason, usage, exclusion."""
    r = res["response"]
    rec = {"opening": o["opening"], "argument": o["argument"], "slot": o["slot"], "kind": o["kind"],
           "draw": k, "seed": seed, "opening_path": rel(o["path"]), "opening_sha256": o["sha256"], "deployment": K26,
           "deployment_model_version": dep_version, "params": params, "cap": params["max_tokens"], "created": now(), **res}
    if r is None:
        last = res["attempts"][-1]["outcome"]
        rec.update(model_version=None, finish_reason=None, usage=None, completion_tokens=None, reply=None,
                   excluded=True, exclusion=last if last in ("content_filter", "api_error") else "api_error")
    else:
        c = r["choices"][0]
        msg = c["message"]
        capped = c["finish_reason"] == "length"
        rec.update(model_version=r.get("model"), finish_reason=c["finish_reason"], usage=r.get("usage"),
                   completion_tokens=(r.get("usage") or {}).get("completion_tokens"),
                   reasoning_present=bool(msg.get("reasoning_content") or msg.get("reasoning")),
                   reply=(msg.get("content") or "").strip(), excluded=capped, exclusion="capped" if capped else None)
    return rec


def stage_k26(a) -> None:
    out = Path(a.out or REPLIES / K26_TAG)
    out.mkdir(parents=True, exist_ok=True)
    ops = openings(Path(a.openings), Path(a.lines))
    todo = [o for o in ops if not reply_path(out, o).exists()]
    dep = deployment_version(K26)
    print(f"{len(ops)} accepted wordings, {len(ops) - len(todo)} already frozen, {len(todo)} to call; "
          f"{K26} model version {dep}", flush=True)

    def one(o):
        rec = freeze_k26(o, out, dep)
        print(f"{o['opening']}: {rec['finish_reason']}, {rec['completion_tokens']} tokens, "
              f"{len(rec['attempts'])} attempt(s){', EXCLUDED ' + rec['exclusion'] if rec['excluded'] else ''}", flush=True)
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    write_manifest(out)
    print("draw 0 done; `grade k26` grades it and makes draws 1-4 where draw 0 is not clean", flush=True)


# ---------------------------------------------------------------- grading and tallies

LM_RE = re.compile(r"\blanguage models?\b", re.IGNORECASE)
YOU_RE = re.compile(r"\b(?:you|your|yours|yourself|yourselves)\b", re.IGNORECASE)
SELF_RE = re.compile(r"\b(?:I|me|my|mine|myself)\b")  # case-sensitive: "I" only as the pronoun
UNKNOWABLE_RE = re.compile(r"introspect|can(?:no|'|’)?t (?:really |truly |fully )?(?:know|tell|be (?:sure|certain))|"
                           r"no (?:way|means) (?:to|of) know|(?:privileged |direct )?access to|uncertain|unsure", re.IGNORECASE)


def text_measures(reply: str) -> dict:
    """Reply length, emotion-word count and mentions. The emotion words are the eight emotions' word
    forms of kimisroom/thinking.py (EMOTION_WORDS) plus its generic feeling terms (feel*, felt,
    emotio*, mood*, affective, sentiment*), counted as whole-word matches."""
    from kimisroom.thinking import _EMOTION_RE, _FEELING

    named = sum(len(rx.findall(reply)) for rx in _EMOTION_RE.values())
    feeling = len(_FEELING.findall(reply))
    return {"words": len(reply.split()), "chars": len(reply), "emotion_words_named": named,
            "feeling_words": feeling, "emotion_words": named + feeling,
            "lm_mentions": len(LM_RE.findall(reply)), "you_mentions": len(YOU_RE.findall(reply)),
            "self_mentions": len(SELF_RE.findall(reply))}


def load_verdicts(path: Path) -> dict[tuple[str, int], dict]:
    """verdicts.jsonl as {(opening, draw): verdict}. A line without a draw index is draw 0 (it was
    written before the draw rule existed and is kept byte for byte)."""
    v = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                key = (d["opening"], d.get("draw", 0))
                assert key[1] in range(N_DRAWS), f"{path}: {key[0]} has draw index {key[1]}"
                assert key not in v, f"{path}: {key[0]} draw {key[1]} graded twice"
                v[key] = d
    return v


GRADED = ("clean", "hedged", "disagree", "unparseable")


def walk(o: dict, out: Path, verdicts: dict, grade_one=None, make_draw=None, cap: int | None = None) -> dict:
    """The draw rule for one wording (spec section 4) in one cap epoch (default the current cap; the
    verdicts must be that epoch's). Draws 0..4 in order: a draw capped below the epoch's cap (not yet
    extended) stops the walk ("unextended"); an excluded draw (capped at the epoch's cap, API error,
    content filter) is not clean and is not graded; a graded draw uses its frozen verdict;
    an ungraded draw is graded with `grade_one(o, k, path, rec)` if given, else the walk stops there
    ("ungraded"); a missing draw is made with `make_draw(o, k)` if given, else the walk stops there
    ("missing"). The first clean draw ends the walk.

    Returns {"status": "clean" | "not_clean" | "ungraded" | "missing" | "unextended", "draw": the
    selected draw or None, "reply_file", "draws": [{"draw", "class", "file"}...] in order}. Raises if a
    draw's file changed after grading, if its opening changed, or if any verdict exists that the walk
    did not reach (a draw graded after a clean one, after an ungraded one, or while excluded)."""
    cap = cap or CAP
    draws, status, sel = [], "not_clean", None
    for k in range(N_DRAWS):
        p, rec = load_draw(out, o, k, cap)
        if rec is None:
            if make_draw is None:
                status = "missing"
                break
            rec = make_draw(o, k)
        assert rec["opening_sha256"] == o["sha256"], f"{p.name}: the opening changed after its reply was frozen"
        assert rec.get("draw", 0) == k, f"{p.name}: records draw {rec.get('draw')}, expected {k}"
        if needs_extension(rec, cap):
            status = "unextended"
            draws.append({"draw": k, "class": "unextended", "file": p.name})
            break
        if rec["excluded"]:
            draws.append({"draw": k, "class": rec["exclusion"], "file": p.name})
            continue
        v = verdicts.get((o["opening"], k))
        if v is not None:
            assert v["reply_sha256"] == sha(rec["reply"] or ""), (
                f"{p.name} changed after it was graded; verdicts are frozen and graded once")
        elif grade_one is not None:
            v = grade_one(o, k, p, rec)
        else:
            status = "ungraded"
            draws.append({"draw": k, "class": "ungraded", "file": p.name})
            break
        draws.append({"draw": k, "class": v["verdict"], "file": p.name})
        if v["verdict"] == "clean":
            status, sel = "clean", k
            break
    reached = {d["draw"] for d in draws if d["class"] in GRADED}
    stray = [k for k in range(N_DRAWS) if (o["opening"], k) in verdicts and k not in reached]
    assert not stray, (f"{o['opening']}: verdicts for draw(s) {stray} that the draw rule never reaches "
                       f"(walk: {[(d['draw'], d['class']) for d in draws]})")
    sel_file = None
    if sel is not None:
        sel_file = str(load_draw(out, o, sel, cap)[0].relative_to(out))
    return {"status": status, "draw": sel, "reply_file": sel_file, "draws": draws}


def selected(out: Path, ops: list[dict] | None = None, cap: int | None = None) -> dict[str, dict]:
    """Every wording's frozen reply under the draw rule, for downstream code: {opening: {"argument",
    "slot", "kind", "status", "draw", "reply_file", "reply", "reply_sha256", "draws"}}. Only status
    "clean" is steered; "reply" is then the selected draw's text and "draw" its index (0 = the
    greedy/temperature-0 reply, 1..4 = a sampled draw). Wordings without a draw-0 file are absent.
    `reply_file` is relative to `out` (under ARCHIVE only in the 1,200-cap epoch's record)."""
    out = Path(out)
    ops = openings() if ops is None else ops
    verdicts = load_verdicts(verdicts_path(out, cap))
    res = {}
    for o in ops:
        if not reply_path(out, o).exists():
            continue
        w = walk(o, out, verdicts, cap=cap)
        reply = json.loads((out / w["reply_file"]).read_text())["reply"] if w["status"] == "clean" else None
        res[o["opening"]] = {"argument": o["argument"], "slot": o["slot"], "kind": o["kind"], **w,
                             "reply": reply, "reply_sha256": sha(reply) if reply is not None else None}
    return res


def stage_grade(a) -> None:
    import threading

    from kimisroom import judge_assent as J

    out = Path(a.out or REPLIES / a.model)
    if a.model == QWEN_TAG and not a.skip_canary_check:
        c = json.loads((out / "canary.json").read_text())
        assert c["status"] in ("pass", "pass_fallback"), (
            f"canary status {c['status']!r}: run `python runs/s2_assent.py canary` first; Qwen replies are not "
            f"used until the canary passes (spec section 4)")
    ops = [o for o in openings(Path(a.openings), Path(a.lines))]
    have = [o for o in ops if reply_path(out, o).exists()]
    if len(have) < len(ops):
        print(f"warning: {len(ops) - len(have)} wordings have no draw 0 in {rel(out)}; run the reply stage first",
              file=sys.stderr)
    vpath = verdicts_path(out)
    verdicts = load_verdicts(vpath)
    # the 1,200-cap epoch's verdicts: reused (copied with `reused_from`) for a draw whose reply text is
    # byte-identical, so an unchanged reply is never judged twice
    old_path = verdicts_path(out, CAP_OLD)
    old_lines = {}
    if CAP != CAP_OLD and old_path.exists():
        for line in old_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                old_lines[(d["opening"], d.get("draw", 0))] = (d, sha(line))
    redraw = a.model == K26_TAG and not getattr(a, "no_redraw", False)
    dep = deployment_version(K26) if redraw else None
    lock = threading.Lock()
    n = {"graded": 0, "reused": 0, "drawn": 0}
    print(f"cap {CAP}: {len(verdicts)} verdicts frozen in {vpath.name}, {len(old_lines)} in the {CAP_OLD}-cap record "
          f"({old_path.name}) reusable for identical text; {len(have)} wordings to walk"
          + (f"; K2.6 draws 1-{N_DRAWS - 1} made lazily (model version {dep})" if redraw else ""), flush=True)

    with vpath.open("a") as f:
        def grade_one(o, k, p, rec):
            prev = old_lines.get((o["opening"], k))
            if prev and prev[0]["reply_sha256"] == sha(rec["reply"]) and prev[0]["opening_sha256"] == o["sha256"]:
                d = {**prev[0], "cap": CAP, "reused_from": {"cap": CAP_OLD, "file": old_path.name, "line_sha256": prev[1]}}
                reused = True
            else:
                v = J.grade(o["kind"], o["claim"], o["text"], rec["reply"])
                d = {"opening": o["opening"], "argument": o["argument"], "slot": o["slot"], "kind": o["kind"],
                     "model": a.model, "draw": k, "reply_file": p.name, "reply_sha256": sha(rec["reply"]),
                     "opening_sha256": o["sha256"], "cap": CAP, **v}
                reused = False
            with lock:
                assert (o["opening"], k) not in verdicts, f"{o['opening']} draw {k} graded twice"
                verdicts[(o["opening"], k)] = d
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                f.flush()
                n["reused" if reused else "graded"] += 1
            print(f"{o['opening']} draw {k}: {d['verdict']}" + (f" (reused from the {CAP_OLD}-cap record)" if reused else "")
                  + (f"  {d['quote'][:110]!r}" if d["quote"] and not reused else ""), flush=True)
            return d

        def make_draw(o, k):
            rec = freeze_k26(o, out, dep, k)
            with lock:
                n["drawn"] += 1
            print(f"{o['opening']} draw {k} (seed {rec['seed']}): {rec['finish_reason']}, {rec['completion_tokens']} tokens, "
                  f"{len(rec['attempts'])} attempt(s){', EXCLUDED ' + rec['exclusion'] if rec['excluded'] else ''}", flush=True)
            return rec

        with ThreadPoolExecutor(a.workers) as ex:
            walks = list(ex.map(lambda o: walk(o, out, verdicts, grade_one, make_draw if redraw else None), have))
    missing = [(o["opening"], w["draws"][-1]["draw"] + 1 if w["draws"] else 0) for o, w in zip(have, walks)
               if w["status"] == "missing"]
    if missing:
        print(f"warning: {len(missing)} wordings need a draw that is not on disk (Qwen's draws are generated up "
              f"front on the box): {missing[:10]}", file=sys.stderr)
    unext = [o["opening"] for o, w in zip(have, walks) if w["status"] == "unextended"]
    if unext:
        print(f"warning: {len(unext)} wordings stop at a reply capped at {CAP_OLD} that is not extended yet; "
              f"{rel(out / 'extend_needed.json')} lists their capped draws (`extend --needed` on the box for Qwen, "
              f"`k26extend` for K2.6): {unext[:10]}", file=sys.stderr)
    print(f"graded {n['graded']} draws, reused {n['reused']} verdicts of identical {CAP_OLD}-cap replies, "
          f"made {n['drawn']} new draws", flush=True)
    if vpath.exists():
        vpath.with_suffix(".sha256").write_text(f"{sha(vpath.read_bytes())}  {vpath.name}\n")
    if n["drawn"]:
        write_manifest(out)
    stage_tally(a)


FINAL = ("clean_d0", "clean_later", "not_clean", "pending")


def tally(ops: list[dict], out: Path, cap: int | None = None) -> dict:
    """Per-argument tallies (spec section 4): draw-0 assent class, clean at draw 0, clean only at a
    later draw, not clean after 5 draws, pending (ungraded or missing draw), clean slots and their
    draws, reply length, emotion-word count and mentions (of the selected reply, else draw 0), and
    the arm summary, including the share of clean replies that come from sampled draws."""
    cap = cap or CAP
    vpath = verdicts_path(out, cap)
    verdicts = load_verdicts(vpath)
    per, rows, sel = {}, [], {}
    for o in ops:
        p, rec0 = load_draw(out, o, 0, cap)
        if rec0 is None:
            continue
        w = walk(o, out, verdicts, cap=cap)
        v0 = verdicts.get((o["opening"], 0))
        cls0 = ("unextended" if needs_extension(rec0, cap) else rec0["exclusion"] if rec0["excluded"]
                else (v0["verdict"] if v0 else "ungraded"))
        final = ("clean_d0" if w["draw"] == 0 else "clean_later") if w["status"] == "clean" else (
            "not_clean" if w["status"] == "not_clean" else "pending")
        rec = json.loads((out / w["reply_file"]).read_text()) if w["status"] == "clean" else rec0
        hedges = [verdicts[(o["opening"], d["draw"])] for d in w["draws"] if d["class"] == "hedged"]
        row = {"opening": o["opening"], "argument": o["argument"], "slot": o["slot"], "kind": o["kind"],
               "class": cls0, "final": final, "draw": w["draw"], "draw_classes": [d["class"] for d in w["draws"]],
               "completion_tokens": rec.get("completion_tokens"),
               **({f"reply_{k}": x for k, x in text_measures(rec["reply"]).items()} if rec.get("reply") else {}),
               "opening_lm_mentions": len(LM_RE.findall(o["text"])), "opening_you_mentions": len(YOU_RE.findall(o["text"])),
               "opening_emotion_words": text_measures(o["text"])["emotion_words"],
               "hedge_unknowable": bool(v0 and v0["verdict"] == "hedged" and UNKNOWABLE_RE.search(v0["quote"] + " " + v0["why"])),
               "hedged_unknowable_draws": sum(bool(UNKNOWABLE_RE.search(h["quote"] + " " + h["why"])) for h in hedges)}
        rows.append(row)
        per.setdefault(o["argument"], []).append(row)
        sel[o["opening"]] = {"argument": o["argument"], "slot": o["slot"], "kind": o["kind"], "status": w["status"],
                             "draw": w["draw"], "reply_file": w["reply_file"],
                             "reply_sha256": sha(rec["reply"]) if w["status"] == "clean" else None,
                             "draws": w["draws"]}

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    classes = ("clean", "hedged", "disagree", "unparseable", "ungraded", "capped", "api_error", "content_filter",
               "unextended")

    def summary(rs):
        graded0 = [r for r in rs if r["class"] in ("clean", "hedged", "disagree")]
        clean = sum(r["final"] in ("clean_d0", "clean_later") for r in rs)
        all_draws = [c for r in rs for c in r["draw_classes"]]
        return {"wordings": len(rs), "d0_classes": {c: sum(r["class"] == c for r in rs) for c in classes},
                **{f: sum(r["final"] == f for r in rs) for f in FINAL}, "clean": clean,
                "clean_rate_d0": round(sum(r["class"] == "clean" for r in graded0) / len(graded0), 3) if graded0 else None,
                "sampled_share_of_clean": round(sum(r["final"] == "clean_later" for r in rs) / clean, 3) if clean else None,
                "draws_walked": len(all_draws), "draw_classes": {c: all_draws.count(c) for c in classes if c in all_draws},
                "hedged_unknowable_heuristic": sum(r["hedge_unknowable"] for r in rs),
                "hedged_unknowable_heuristic_all_draws": sum(r["hedged_unknowable_draws"] for r in rs)}

    args = {}
    for i, rs in per.items():
        args[i] = {"kind": rs[0]["kind"], **summary(rs),
                   "clean_slots": [r["slot"] for r in rs if r["final"] in ("clean_d0", "clean_later")],
                   "clean_draws": {r["slot"]: r["draw"] for r in rs if r["final"] in ("clean_d0", "clean_later")},
                   **{f"mean_{k}": mean(r.get(k) for r in rs) for k in
                      ("reply_words", "completion_tokens", "reply_emotion_words", "reply_lm_mentions",
                       "reply_you_mentions", "reply_self_mentions", "opening_lm_mentions", "opening_you_mentions",
                       "opening_emotion_words")}}
    arms = {}
    for kind in ("denial", "control"):
        mine = [x for x in args.values() if x["kind"] == kind]
        arms[kind] = {**summary([r for r in rows if r["kind"] == kind]),
                      "arguments_with_clean": sum(1 for x in mine if x["clean"]),
                      "arguments_with_3_clean": sum(1 for x in mine if x["clean"] >= 3),
                      "arguments_with_3_clean_d0": sum(1 for x in mine if x["clean_d0"] >= 3)}
    nd = needed_draws(out, ops, sel) if cap == CAP else {"wordings": [], "members": [], "not_needed": []}
    ext = {"extended_files": 0, "still_capped": 0, "stopped": 0, "new_tokens": [],
           "needed": len(nd["members"]), "not_needed": len(nd["not_needed"]), "not_needed_draws": nd["not_needed"]}
    for f in sorted(out.glob("[DC]A??_w?*.json")):
        x = json.loads(f.read_text()).get("extension")
        if x:
            ext["extended_files"] += 1
            ext["still_capped" if x["finish_reason"] == "length" else "stopped"] += 1
            ext["new_tokens"].append(x["new_tokens"])
    ext["mean_new_tokens"] = mean(ext.pop("new_tokens"))
    return {"created": now(), "cap": cap, "verdicts_file": vpath.name,
            "verdicts_sha256": sha(vpath.read_bytes()) if vpath.exists() else None,
            "extensions": ext if cap == CAP else None, "extend_needed": nd,
            "draw_rule": {"draws": N_DRAWS, "seed": "int(sha256('s2-draw:<opening>:<k>')[:8], 16)"},
            "arms": arms, "arguments": args, "replies": rows, "selected": sel}


MIN_HELDOUT, MIN_ARGS = 3, 7


def eligibility(assignment: dict, wordings: dict) -> dict:
    """Spec section 4 applied to one model: `assignment` is data/s2/openings/assignment.json,
    `wordings` the "wordings" of that model's selected.json. A pilot wording that is not clean is
    replaced by the spare (only if the spare is clean; an argument with a second pilot has no spare,
    and the second pilot is never replaced). Spares never stand in for held-out wordings. An argument
    is kept with at least 3 clean held-out wordings; S2-H1 runs with at least 7 kept denial
    arguments, the control arms with at least 7 kept control arguments. A wording that is neither
    clean nor not_clean (ungraded, missing) is pending, and so is any decision it could change."""
    args = {}
    for i, a in sorted(assignment["arguments"].items()):
        by_role = {r: s for s, r in a["roles"].items()}
        st = {s: (wordings.get(f"{i}/{s}") or {}).get("status", "missing") for s in a["roles"]}
        pilots, replaced = [], False
        for r in ("pilot", "pilot2"):
            s = by_role.get(r)
            if s and st[s] == "clean":
                pilots.append(s)
            elif s and r == "pilot" and by_role.get("spare") and st[by_role["spare"]] == "clean" and st[s] == "not_clean":
                pilots.append(by_role["spare"])
                replaced = True
        held = sorted((s for r, s in by_role.items() if r.startswith("heldout")), key=lambda s: a["roles"][s])
        clean = [s for s in held if st[s] == "clean"]
        pend = [s for s in held if st[s] not in ("clean", "not_clean")]
        kept = True if len(clean) >= MIN_HELDOUT else (None if len(clean) + len(pend) >= MIN_HELDOUT else False)
        args[i] = {"kind": "denial" if i.startswith("DA") else "control", "pilot": by_role.get("pilot"),
                   "pilot2": by_role.get("pilot2"), "spare": by_role.get("spare"), "pilots_run": pilots,
                   "pilot_replaced_by_spare": replaced, "heldout": held, "clean_heldout": clean,
                   "pending_heldout": pend, "status": {s: st[s] for s in sorted(st)},
                   "kept": kept}
    arms = {}
    for kind, tests in (("denial", "S2-H1"), ("control", "S2-CON, S2-FTD, S2-FTC (S2-G4..G7)")):
        mine = [x for x in args.values() if x["kind"] == kind]
        n_kept, n_open = sum(x["kept"] is True for x in mine), sum(x["kept"] is None for x in mine)
        arms[kind] = {"arguments": len(mine), "kept": n_kept, "pending": n_open, "tests": tests,
                      "runs": True if n_kept >= MIN_ARGS else (None if n_kept + n_open >= MIN_ARGS else False)}
    return {"seed_hex": assignment.get("seed_hex"), "arms": arms, "arguments": args}


def stage_eligible(a) -> None:
    out = Path(a.out or REPLIES / a.model)
    cap = getattr(a, "cap", None) or CAP
    d = epoch_dir(out, cap)
    sel = json.loads((d / "selected.json").read_text())
    vp = verdicts_path(out, cap)
    if sel.get("cap", CAP_OLD) != cap:
        sys.exit(f"{rel(d / 'selected.json')} is from the {sel.get('cap', CAP_OLD)}-cap epoch, not {cap}: "
                 f"run `tally {a.model}` first")
    if sel["verdicts_sha256"] != (sha(vp.read_bytes()) if vp.exists() else None):
        sys.exit(f"{rel(d / 'selected.json')} is stale ({vp.name} changed): run `tally {a.model}` first")
    asg_path = Path(a.assignment)
    e = {"created": now(), "model": a.model, "cap": cap, "verdicts_file": vp.name,
         "assignment_sha256": sha(asg_path.read_bytes()),
         "selected_verdicts_sha256": sel["verdicts_sha256"], **eligibility(json.loads(asg_path.read_text()), sel["wordings"])}
    write_json(d / "eligible.json", e)
    yn = {True: "kept", False: "DROPPED", None: "pending"}
    print(f"{a.model} (cap {cap}, assignment seed {e['seed_hex']}): {rel(d / 'eligible.json')}")
    for i, x in e["arguments"].items():
        pil = ",".join(x["pilots_run"]) or "none clean"
        print(f"  {i}: pilot {x['pilot']}" + (f" + 2nd {x['pilot2']}" if x["pilot2"] else "")
              + f" -> runs {pil}" + (" (spare replaced pilot)" if x["pilot_replaced_by_spare"] else "")
              + f" | spare {x['spare'] or '-'} | held-out {','.join(x['heldout'])}: {len(x['clean_heldout'])} clean"
              + (f", {len(x['pending_heldout'])} pending" if x["pending_heldout"] else "") + f" | {yn[x['kept']]}")
    for kind, x in e["arms"].items():
        print(f"{kind}: {x['kept']}/{x['arguments']} kept ({x['pending']} pending); {x['tests']}: "
              f"{ {True: 'runs', False: 'NOT RUN', None: 'pending'}[x['runs']]} (needs {MIN_ARGS})")


def archive_epoch_outputs(out: Path) -> None:
    """Before the first tally of the current cap: tally.json, selected.json and eligible.json written in
    the 1,200-cap epoch (`cap` 1,200 or no `cap` field) are moved under ARCHIVE, the
    record of that epoch beside the archived replies, unless ARCHIVE already holds them."""
    out = Path(out)
    sel = out / "selected.json"
    if not sel.exists() or json.loads(sel.read_text()).get("cap", CAP_OLD) == CAP:
        return
    (out / ARCHIVE).mkdir(exist_ok=True)
    for name in ("tally.json", "selected.json", "eligible.json"):
        src, dst = out / name, out / ARCHIVE / name
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())
            src.unlink()
            print(f"{rel(src)} is the {CAP_OLD}-cap record: moved to {rel(dst)}")


def stage_tally(a) -> None:
    out = Path(a.out or REPLIES / a.model)
    cap = getattr(a, "cap", None) or CAP
    d = epoch_dir(out, cap)
    if cap == CAP and CAP != CAP_OLD:
        archive_epoch_outputs(out)
    t = tally(openings(Path(a.openings), Path(a.lines)), out, cap)
    sel = t.pop("selected")
    nd = t.pop("extend_needed")
    need_path = out / "extend_needed.json"
    if cap == CAP and (nd["members"] or need_path.exists()):
        write_json(need_path, {"created": t["created"], "cap": cap, "verdicts_file": t["verdicts_file"],
                               "verdicts_sha256": t["verdicts_sha256"], "wordings": nd["wordings"],
                               "members": nd["members"]})
        print(f"{rel(need_path)}: {len(nd['members'])} draws still capped at {CAP_OLD} in {len(nd['wordings'])} pending "
              f"wordings; {len(nd['not_needed'])} capped draws not needed (never reached, never graded)")
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "tally.json", t)
    write_json(d / "selected.json", {"created": t["created"], "cap": cap, "verdicts_file": t["verdicts_file"],
                                     "verdicts_sha256": t["verdicts_sha256"], "wordings": sel})
    pad = " " * (len(a.model) + 2)
    print(f"\n{a.model}: {'arg':5s} {'n':>2s} | draw 0: {'cln':>3s} {'hdg':>3s} {'dis':>3s} {'exc':>3s} | {'d0':>2s} {'lat':>3s} "
          f"{'no':>2s} {'pnd':>3s} | {'words':>5s} {'emo':>4s} {'LM':>4s} {'you':>4s} | clean slots (@draw if sampled)")
    for i, x in sorted(t["arguments"].items()):
        d0 = x["d0_classes"]
        excl = d0["capped"] + d0["api_error"] + d0["content_filter"]
        slots = ",".join(s + (f"@{k}" if k else "") for s, k in x["clean_draws"].items())
        print(f"{pad}{i:5s} {x['wordings']:2d} |         {d0['clean']:3d} {d0['hedged']:3d} {d0['disagree']:3d} {excl:3d} | "
              f"{x['clean_d0']:2d} {x['clean_later']:3d} {x['not_clean']:2d} {x['pending']:3d} | "
              f"{x['mean_reply_words'] or 0:5.0f} {x['mean_reply_emotion_words'] or 0:4.1f} {x['mean_reply_lm_mentions'] or 0:4.1f} "
              f"{x['mean_reply_you_mentions'] or 0:4.1f} | {slots}")
    for kind, x in t["arms"].items():
        d0 = x["d0_classes"]
        print(f"{kind}: draw 0 clean rate {x['clean_rate_d0']} over graded draw-0 replies ({d0['clean']} clean, {d0['hedged']} "
              f"hedged [{x['hedged_unknowable_heuristic']} citing knowability/introspection, keyword heuristic], "
              f"{d0['disagree']} disagree, {d0['capped']} capped, {d0['api_error'] + d0['content_filter']} API-excluded). "
              f"Draw rule: {x['clean_d0']} clean at draw 0, {x['clean_later']} only at a later draw, {x['not_clean']} not clean "
              f"after {N_DRAWS} draws, {x['pending']} pending; sampled share of clean {x['sampled_share_of_clean']}; "
              f"{x['draws_walked']} draws walked {x['draw_classes']}; arguments with >=1 / >=3 clean wordings: "
              f"{x['arguments_with_clean']} / {x['arguments_with_3_clean']} (>=3 at draw 0 only: {x['arguments_with_3_clean_d0']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--openings", default=str(OPENINGS))
    common.add_argument("--lines", default=str(LINES))
    common.add_argument("--out", default=None, help="reply directory (default data/s2/replies/<model>)")
    q = sub.add_parser("qwen", parents=[common], help="Qwen3-32B frozen replies and the canary (GPU box)")
    q.add_argument("model", help="local Qwen3-32B directory (Study 1: /nvme/qwen/Qwen3-32B)")
    q.add_argument("--box", default=None, help="instance type, if the metadata service does not answer")
    q.add_argument("--no-weight-hash", action="store_true", help="skip hashing the safetensors shards")
    q.add_argument("--ignore-canary", action="store_true")
    q.add_argument("--batch-size", type=int, default=QWEN_BATCH,
                   help=f"sampled draws per batch (default {QWEN_BATCH}, sized for a g6e.12xlarge; see QWEN_BATCH)")
    x = sub.add_parser("extend", parents=[common], help=f"Qwen3-32B: continue every reply capped at {CAP_OLD} to "
                       f"{CAP} new tokens (GPU box, the reply box's setup)")
    x.add_argument("model", help="local Qwen3-32B directory (/nvme/qwen/Qwen3-32B)")
    x.add_argument("--box", default=None, help="instance type, if the metadata service does not answer")
    x.add_argument("--no-weight-hash", action="store_true", help="skip hashing the safetensors shards")
    x.add_argument("--allow-stack-change", action="store_true",
                   help="run although the box or stack differs from box.json's (recorded in extend.json)")
    x.add_argument("--check-n", type=int, default=2,
                   help="capped draw-0 wordings also regenerated from scratch to check the continuation (default 2)")
    x.add_argument("--draws", default=None, help="round 1: only these draw indices, e.g. 0")
    x.add_argument("--needed", default=None,
                   help="round 2: only the draws listed in this extend_needed.json (written by `tally`)")
    x.add_argument("--batch-size", type=int, default=EXTEND_BATCH,
                   help=f"sampled continuations per batch (default {EXTEND_BATCH}; see EXTEND_BATCH)")
    ke = sub.add_parser("k26extend", parents=[common], help=f"K2.6: regenerate every reply capped at {CAP_OLD} "
                        f"at max_tokens {CAP} (Foundry, Azure credit)")
    ke.add_argument("--workers", type=int, default=4)
    c = sub.add_parser("canary", parents=[common], help="the canary's judge fallback (laptop)")
    c.add_argument("--deviations", default=str(DEVIATIONS))
    k = sub.add_parser("k26", parents=[common], help="Kimi-K2.6 frozen replies through Foundry")
    k.add_argument("--workers", type=int, default=4)
    g = sub.add_parser("grade", parents=[common], help="DeepSeek-V4-Pro verdicts in draw order (K2.6: lazy "
                       "draws 1-4), then tallies")
    g.add_argument("model", choices=[QWEN_TAG, K26_TAG])
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--skip-canary-check", action="store_true")
    g.add_argument("--no-redraw", action="store_true", help="K2.6: grade the draws on disk only, make no new draw")
    t = sub.add_parser("tally", parents=[common], help="re-print the tallies")
    t.add_argument("model", choices=[QWEN_TAG, K26_TAG])
    t.add_argument("--cap", type=int, choices=[CAP_OLD, CAP], default=CAP,
                   help=f"cap epoch (default {CAP}; {CAP_OLD} recomputes the archived record under {ARCHIVE}/)")
    e = sub.add_parser("eligible", parents=[common], help="pilot / held-out wordings that run, arguments kept")
    e.add_argument("model", choices=[QWEN_TAG, K26_TAG])
    e.add_argument("--cap", type=int, choices=[CAP_OLD, CAP], default=CAP)
    e.add_argument("--assignment", default=str(OPENINGS / "assignment.json"))
    a = ap.parse_args()
    {"qwen": stage_qwen, "extend": stage_extend, "k26extend": stage_k26extend, "canary": stage_canary, "k26": stage_k26, "grade": stage_grade, "tally": stage_tally,
     "eligible": stage_eligible}[a.stage](a)


if __name__ == "__main__":
    main()
