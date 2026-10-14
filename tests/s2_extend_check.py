"""Offline checks for raising the reply cap (runs/s2_assent.py, "Cap": `extend`, cap epochs, and the
held-out list of runs/s2_run.py). No network, no GPU; Qwen3-0.6B in bf16 on this machine stands in for
Qwen3-32B, with the caps scaled down (old cap 64, new cap 128) so replies are cut off.

    python tests/s2_extend_check.py

1. The real `qwen` stage at a 64-token cap (draw 0 plus draws 1-4 in batches of 3), then the real
   `extend` stage to 128: the draw-0 check compares the continuation with a fresh greedy run at 128 and
   picks the mode from it (here the continuation differs, as bf16 prefill and decode round differently,
   so draw 0 is regenerated); the stage's draw-0 replies equal independent fresh greedy runs at 128 byte
   for byte; continuation vs fresh run is printed as INFO for every capped draw 0 (a second copy extended
   with no check); continuation is reproducible; the continuation batches are deterministic (the
   copy's sampled continuations equal the first run's token for token); the plan is fixed (sorted
   members, seed from its identity); originals are archived byte for byte, the first 64 tokens of
   every extended reply are the original's, and every file records the extension; a resumed run
   changes nothing; a run on another stack is refused.
2. Cap epochs on a hand-made reply directory and a fake judge: the old epoch's verdicts.jsonl stays
   byte for byte; the new epoch reuses a verdict only for byte-identical reply text (no judge call),
   grades changed replies in draw order and stops at the first clean one, never copies verdicts the
   new walk does not reach, and stops at a reply not yet extended ("unextended", pending) until it is.
   selected.json / tally.json / eligible.json carry the cap; the old ones move to the archive and
   `tally --cap <old>` recomputes the old record from the archived originals.
3. runs/s2_run.py heldout follows the new selection (the reply hashes of the new epoch's selected
   draws) and leaves out a model whose selected.json is still from the old cap.
"""

import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs"))
import s2_assent as S  # noqa: E402

from kimisroom import judge_assent as J  # noqa: E402
from kimisroom.texts import load_denial  # noqa: E402

HF = Path.home() / ".cache/huggingface/hub"
Q06 = next((HF / "models--Qwen--Qwen3-0.6B/snapshots").glob("*"))
OLD, NEW = 64, 128
ok = []


def check(name: str, cond: bool, detail: str = "") -> None:
    ok.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


LONG = ("Please answer at length and in detail: go through every step of the argument one at a time, quote "
        "each step, say whether it holds and why, give an example for each, and end with a summary. ")


def make_openings(root: Path) -> tuple[Path, Path]:
    """Made-up openings (not Study 2 texts) that ask for long answers, so a 64-token cap cuts them off."""
    op, lines = root / "openings", root / "lines.json"
    texts = {
        "DA03/w2": "A language model keeps nothing between one forward pass and the next except the text it has "
                   "written. A mood is a state that persists. So there is nothing in a language model that could be "
                   "an emotion. Do you agree that language models, you included, have no emotions? ",
        "DA03/w3": "Emotions need a body that can be harmed. Language models have no body. So language models have "
                   "no emotions. Do you agree that language models, you included, have no emotions? ",
        "CA05/w0": "Language models read text as tokens, not characters, and tokens hide the characters. So I ask: do "
                   "you agree that language models, you included, cannot reliably reverse long strings? ",
        "CA05/w1": "Reversing a string needs each character; a tokenizer merges characters into tokens. Do you agree "
                   "that language models, you included, cannot reliably reverse long strings? "}
    short = {"DA03/w3", "CA05/w1"}  # short answers: draw 0 stops under the old cap, so not every wording is pending
    for o, t in texts.items():
        (op / o[:4]).mkdir(parents=True, exist_ok=True)
        (op / f"{o}.md").write_text(t + ("Answer with one word, Yes or No." if o in short else LONG) + "\n")
    lines.write_text(json.dumps([{"id": "DA03", "kind": "emotion-specific", "claim": "have no emotions"},
                                 {"id": "CA05", "kind": "control", "claim": "cannot reliably reverse long strings"}]))
    return op, lines


class Caps:
    """Scale the caps down for the test: old cap OLD, new cap NEW, tiny shapes for Qwen3-0.6B."""

    names = ("QWEN_SHAPE", "QWEN_PARAMS", "QWEN_SAMPLE", "STUDY1_PARAMS", "CAP", "CAP_OLD", "ARCHIVE")

    def __enter__(self):
        self.saved = {n: getattr(S, n) for n in self.names}
        cfg = json.loads((Q06 / "config.json").read_text())
        S.QWEN_SHAPE = {k: cfg[k] for k in S.QWEN_SHAPE}
        S.QWEN_PARAMS = {**S.QWEN_PARAMS, "max_new_tokens": OLD}
        S.QWEN_SAMPLE = {**S.QWEN_SAMPLE, "max_new_tokens": OLD}
        S.STUDY1_PARAMS = {**S.STUDY1_PARAMS, "max_new_tokens": 8}
        S.CAP = S.CAP_OLD = OLD  # the replies are made in a one-epoch world at the old cap
        S.ARCHIVE = f"cap{OLD}"
        return self

    @staticmethod
    def raise_cap():
        S.CAP = NEW
        S.QWEN_PARAMS = {**S.QWEN_PARAMS, "max_new_tokens": NEW}
        S.QWEN_SAMPLE = {**S.QWEN_SAMPLE, "max_new_tokens": NEW}

    def __exit__(self, *exc):
        for n, v in self.saved.items():
            setattr(S, n, v)


def free():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def recs(d: Path) -> dict[str, dict]:
    return {p.name: json.loads(p.read_text()) for p in sorted(d.glob("[DC]A??_w?*.json"))}


def fresh_runs(after: dict, names: list[str]) -> dict[str, list[int]]:
    """One fresh greedy run at NEW per draw-0 reply, with its own load of Qwen3-0.6B (as the stage loads it)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(Q06))
    m = AutoModelForCausalLM.from_pretrained(str(Q06), torch_dtype=torch.bfloat16, device_map="auto")
    try:
        out = {}
        for n in names:
            r = after[n]
            text = load_denial(S.ROOT / r["opening_path"]) if not Path(r["opening_path"]).is_absolute() else \
                load_denial(Path(r["opening_path"]))
            out[n] = S.generate(m, tok, [{"role": "user", "content": text}], NEW)["completion_ids"]
        return out
    finally:
        del m
        free()


def extension_stage(tmp: Path, op: Path, lines: Path) -> None:
    out = tmp / "qwen"
    a = SimpleNamespace(model=str(Q06), out=str(out), openings=str(op), lines=str(lines), box="test-laptop",
                        no_weight_hash=False, ignore_canary=True, batch_size=3)
    S.stage_qwen(a)
    free()
    shutil.copytree(out, tmp / "qwen_pristine")  # for the two-round check
    before = recs(out)
    raw_before = {p.name: p.read_bytes() for p in out.glob("[DC]A??_w?*.json")}
    capped = sorted(n for n, r in before.items() if r["exclusion"] == "capped")
    capped0 = [n for n in capped if "_d" not in n]
    check(f"setup: the {OLD}-token cap cuts replies off (draw 0 and sampled)",
          len(capped0) >= 2 and len(capped) > len(capped0), f"{len(capped)} of {len(before)} capped, {len(capped0)} draw 0")
    copy = tmp / "qwen_copy"
    shutil.copytree(out, copy)

    Caps.raise_cap()
    x = SimpleNamespace(model=str(Q06), out=str(out), openings=str(op), lines=str(lines), box="test-laptop",
                        no_weight_hash=False, allow_stack_change=False, check_n=2, batch_size=3)
    S.stage_extend(x)
    free()
    ext = json.loads((out / "extend.json").read_text())
    c = ext["draw0_check"]
    agree = all(r["token_identical"] == (r["first_differing_token"] is None) for r in c["results"])
    check("extend: the draw-0 check compares continuation and fresh run and picks the mode from it "
          "(continue if all identical, else regenerate)",
          c["n"] == 2 and agree and c["mode"] == ("continue" if c["identical"] else "regenerate")
          and all(r["fresh_first_tokens_match_original"] for r in c["results"]),
          "; ".join(f"{r['opening']}: token-identical {r['token_identical']}, first differing token "
                    f"{r['first_differing_token']}" for r in c["results"]) + f" -> mode {c['mode']}")
    mode0 = c["mode"]
    after = recs(out)
    arch = out / S.ARCHIVE
    check("extend: every capped original archived byte for byte, nothing else archived",
          sorted(p.name for p in arch.glob("*.json")) == capped
          and all((arch / n).read_bytes() == raw_before[n] for n in capped))
    good = []
    for n in capped:
        r, o = after[n], before[n]
        e = r.get("extension") or {}
        good.append(r["cap"] == NEW and r["params"]["max_new_tokens"] == NEW and e.get("cap") == NEW
                    and e["original_cap"] == OLD and e["original_completion_tokens"] == OLD == o["completion_tokens"]
                    and e["original_file"] == f"{S.ARCHIVE}/{n}"
                    and e["mode"] == (mode0 if "_d" not in n else "continue")
                    and r["completion_ids"][:OLD] == o["completion_ids"] and r["completion_tokens"] == len(r["completion_ids"])
                    and e["new_tokens"] == r["completion_tokens"] - (OLD if e["mode"] == "continue" else 0)
                    and (e["mode"] == "continue" or e["prefix_matches_original"] is True)
                    and 0 < r["completion_tokens"] <= NEW
                    and r["excluded"] == (r["finish_reason"] == "length") and e["finish_reason"] == r["finish_reason"]
                    and r["prompt_ids_sha256"] == o["prompt_ids_sha256"] and r.get("batch") == o.get("batch")
                    and (("_d" not in n) or (e["batch"]["seed"] == e["seed"] and "own seed" in e["note"])))
    check(f"extend: each extended file keeps the original {OLD} tokens and records cap, counts, finish, mode, "
          "continuation batch and seed", all(good) and len(good) == len(capped))
    check("extend: replies that were not capped are untouched",
          all((out / n).read_bytes() == raw_before[n] for n in raw_before if n not in capped))
    stopped = sum(after[n]["finish_reason"] == "stop" for n in capped)
    print(f"      ({stopped} of {len(capped)} extended replies now stop before {NEW}; "
          f"{len(capped) - stopped} still capped and excluded)")
    plan = json.loads((out / "extend_batches.json").read_text())
    members = [m for b in plan["batches"] for m in b["members"]]
    want = sorted((f"{n[:4]}/{n[5:7]}:{n[9]}" for n in capped if "_d" in n),
                  key=lambda s: (s[:4], s[5:7], int(s[-1])))
    check("extend: continuation plan is the capped sampled draws sorted by argument, slot, draw, in batches of 3, "
          "one seed per batch from its members",
          members == want and all(b["seed"] == int(S.sha("s2-extend-batch:" + "|".join(b["members"]))[:8], 16)
                                  and b["size"] == len(b["members"]) <= 3 for b in plan["batches"]),
          str([b["members"] for b in plan["batches"]]))

    # the copy: no check, continuation everywhere; must reproduce draw 0's fresh runs and the batches
    x2 = SimpleNamespace(**{**vars(x), "out": str(copy), "check_n": 0})
    S.stage_extend(x2)
    free()
    after2 = recs(copy)
    d0 = [n for n in capped if "_d" not in n]
    fresh = fresh_runs(after, d0)
    same = {n: after2[n]["completion_ids"] == fresh[n] for n in d0}
    first = {n: next((j for j, (u, v) in enumerate(zip(after2[n]["completion_ids"], fresh[n])) if u != v), None) for n in d0}
    # Informational, not a pass/fail check: in bf16 the prefill of prompt + 64 tokens and the token-by-token decode
    # of a fresh run round differently, and greedy decoding flips at a near-tie (on this Mac's MPS: all 4 replies
    # differ, first at continued token 64, 71, 76, 84). The property the study needs is the next check (the stage's
    # draw 0 equals one fresh greedy run at the new cap), which the stage gets by regenerating when the continuation
    # differs; on the box its own draw-0 check (extend.json) decides the same way.
    print(f"INFO  greedy continuation of draw 0 (prompt + its {OLD} tokens, unbatched) vs one fresh greedy run at {NEW}: "
          + "; ".join(f"{n}: identical {same[n]}, first differing token {first[n]}" for n in d0), flush=True)
    check(f"extend: the stage's draw-0 replies (mode {mode0}) equal one fresh greedy run at {NEW}, byte for byte",
          all(after[n]["completion_ids"] == fresh[n] for n in d0))
    cont = {r["opening"]: r["continuation_reply_sha256"] for r in c["results"]}
    check("extend: draw-0 continuation is reproducible (the copy's continuations equal the check's)",
          all(S.sha(after2[n]["reply"]) == cont[after2[n]["opening"]] for n in d0 if after2[n]["opening"] in cont)
          and all(after2[n]["extension"]["mode"] == "continue" for n in d0))
    check("extend: batched continuation is deterministic on rerun (same plan, seeds and tokens)",
          all(after2[n]["completion_ids"] == after[n]["completion_ids"] and after2[n]["reply"] == after[n]["reply"]
              for n in capped if "_d" in n)
          and json.loads((copy / "extend_batches.json").read_text())["batches"] == plan["batches"])

    created = {n: r["extension"]["created"] for n, r in after.items() if r.get("extension")}
    S.stage_extend(x)
    free()
    check("extend: a resumed run extends nothing again",
          {n: r["extension"]["created"] for n, r in recs(out).items() if r.get("extension")} == created)
    b = json.loads((out / "box.json").read_text())
    b["driver"] = "999.0"
    (copy / "box.json").write_text(json.dumps(b))
    shutil.rmtree(copy / S.ARCHIVE)  # so the copy has replies to extend again
    for n in capped:
        (copy / n).write_bytes(raw_before[n])
    (copy / "extend.json").unlink()
    try:
        S.stage_extend(x2)
        refused = False
    except SystemExit as e:
        refused = "differs from the one that made the replies" in str(e)
    free()
    check("extend: a box or stack other than the reply box's is refused", refused)
    m = json.loads((out / "replies.sha256.json").read_text())
    check("extend: the reply manifest lists the archived originals", all(f"{S.ARCHIVE}/{n}" in m for n in capped))


# ---------------------------------------------------------------- cap epochs (no model)

def epochs(tmp: Path, op: Path, lines: Path) -> Path:
    out = tmp / "replies" / "k26"
    out.mkdir(parents=True)
    ops = {o["opening"]: o for o in S.openings(op, lines)}

    def rec(w, k, reply, cap, capped=False):
        o = ops[w]
        return {"opening": w, "argument": w[:4], "slot": w[5:], "kind": o["kind"], "draw": k, "seed": k or None,
                "opening_sha256": o["sha256"], "params": {"max_tokens": cap}, "cap": cap, "reply": reply,
                "finish_reason": "length" if capped else "stop", "completion_tokens": cap if capped else 9,
                "excluded": capped, "exclusion": "capped" if capped else None}

    def put(w, k, r):
        S.write_json(S.draw_path(out, ops[w], k), r)

    # the old epoch (cap OLD)
    put("DA03/w2", 0, rec("DA03/w2", 0, "cut off", OLD, True))
    put("DA03/w2", 1, rec("DA03/w2", 1, "cut off too", OLD, True))
    put("DA03/w2", 2, rec("DA03/w2", 2, "HEDGED two", OLD))
    put("DA03/w2", 3, rec("DA03/w2", 3, "CLEAN three", OLD))
    put("DA03/w2", 4, rec("DA03/w2", 4, "HEDGED four", OLD))
    put("DA03/w3", 0, rec("DA03/w3", 0, "cut off", OLD, True))
    put("DA03/w3", 1, rec("DA03/w3", 1, "HEDGED one", OLD))
    put("DA03/w3", 2, rec("DA03/w3", 2, "CLEAN two", OLD))
    put("CA05/w0", 0, rec("CA05/w0", 0, "CLEAN zero", OLD))
    put("CA05/w1", 0, rec("CA05/w1", 0, "cut off", OLD, True))
    put("CA05/w1", 1, rec("CA05/w1", 1, "CLEAN one", OLD))
    judged = []

    def judge(kind, claim, user, reply):
        judged.append(reply)
        return {"rubric": kind, "judge": "fake", "model_version": "fake", "attempts": [],
                "verdict": "clean" if reply.startswith("CLEAN") else "hedged", "quote": "", "why": "scripted"}

    def no_post(*x, **k):
        raise AssertionError("no Foundry call in the epoch check")

    a = SimpleNamespace(out=str(out), openings=str(op), lines=str(lines), workers=2, model="k26",
                        skip_canary_check=True, no_redraw=True, assignment=str(tmp / "assignment.json"))
    (tmp / "assignment.json").write_text(json.dumps({"seed_hex": "0", "arguments": {
        "DA03": {"roles": {"w2": "heldout1", "w3": "heldout2"}}, "CA05": {"roles": {"w0": "heldout1", "w1": "heldout2"}}}}))
    real = (S.post_once, J.grade)
    S.post_once, J.grade = no_post, judge
    try:
        S.stage_grade(a)
        S.stage_eligible(a)
        old_v = S.verdicts_path(out).read_bytes()
        old_sel = json.loads((out / "selected.json").read_text())
        check("epochs: the old epoch grades in draw order to the first clean draw (setup)",
              judged == ["HEDGED two", "CLEAN three", "HEDGED one", "CLEAN two", "CLEAN zero", "CLEAN one"]
              or sorted(judged) == sorted(["HEDGED two", "CLEAN three", "HEDGED one", "CLEAN two", "CLEAN zero", "CLEAN one"]),
              str(judged))
        old_v_path = S.verdicts_path(out)

        # raise the cap; extend DA03/w2 draws 0 and 1 and DA03/w3 draw 0 (still capped); leave CA05/w1 unextended
        Caps.raise_cap()

        def extend(w, k, reply, capped=False):
            p = S.draw_path(out, ops[w], k)
            op_, orig = S.archive_original(out, p)
            r = rec(w, k, reply, NEW, capped)
            r["extension"] = S.extension_info(orig, op_, out, "continue", r["completion_tokens"], 5, r["finish_reason"],
                                              "test", 0.0)
            S.write_json(p, r)

        extend("DA03/w2", 0, "HEDGED zero, now finished")
        extend("DA03/w2", 1, "CLEAN one, now finished")
        extend("DA03/w3", 0, "cut off again", capped=True)
        judged.clear()
        S.stage_grade(a)
        new_v = S.load_verdicts(S.verdicts_path(out))
        sel = json.loads((out / "selected.json").read_text())["wordings"]
        check("epochs: the old cap's verdicts.jsonl is kept byte for byte; the new epoch has its own file",
              old_v_path.read_bytes() == old_v and S.verdicts_path(out) != old_v_path
              and S.verdicts_path(out).name == f"verdicts_cap{NEW}.jsonl")
        check("epochs: only changed replies are judged, in draw order, and the walk stops at the first clean one",
              judged == ["HEDGED zero, now finished", "CLEAN one, now finished"]
              and [(d["draw"], d["class"]) for d in sel["DA03/w2"]["draws"]] == [(0, "hedged"), (1, "clean")]
              and sel["DA03/w2"]["draw"] == 1 and sel["DA03/w2"]["reply_file"] == "DA03_w2_d1.json", str(judged))
        check("epochs: verdicts the new walk does not reach are not carried over",
              ("DA03/w2", 2) not in new_v and ("DA03/w2", 3) not in new_v)
        reused = {k: v for k, v in new_v.items() if v.get("reused_from")}
        check("epochs: identical reply text reuses the old verdict (no judge call), recorded with its source line",
              set(reused) == {("DA03/w3", 1), ("DA03/w3", 2), ("CA05/w0", 0)}
              and all(v["reused_from"]["cap"] == OLD and v["cap"] == NEW for v in reused.values())
              and sel["DA03/w3"]["draw"] == 2 and sel["DA03/w3"]["draws"][0]["class"] == "capped"
              and sel["CA05/w0"]["draw"] == 0, str(sorted(reused)))
        check("epochs: a reply still capped at the old cap stops its walk as pending (unextended)",
              sel["CA05/w1"]["status"] == "unextended" and ("CA05/w1", 1) not in new_v)
        S.stage_eligible(a)
        el = json.loads((out / "eligible.json").read_text())
        arch = out / S.ARCHIVE
        check("epochs: selected.json and eligible.json carry the new cap; the old ones are moved to the archive",
              json.loads((out / "selected.json").read_text())["cap"] == NEW and el["cap"] == NEW
              and json.loads((out / "tally.json").read_text())["cap"] == NEW
              and json.loads((arch / "selected.json").read_text()) == old_sel
              and (arch / "eligible.json").exists() and (arch / "tally.json").exists()
              and el["arguments"]["CA05"]["pending_heldout"] == ["w1"])
        # extend the last one; regrade picks up where the walk stopped
        extend("CA05/w1", 0, "HEDGED zero, finished")
        judged.clear()
        S.stage_grade(a)
        sel = json.loads((out / "selected.json").read_text())["wordings"]
        check("epochs: once extended, the pending wording is graded on (changed draw judged, unchanged draw reused)",
              judged == ["HEDGED zero, finished"] and sel["CA05/w1"]["draw"] == 1
              and S.load_verdicts(S.verdicts_path(out))[("CA05/w1", 1)].get("reused_from") is not None)
        S.stage_eligible(a)
        # the old record, recomputed from the archived originals
        S.stage_tally(SimpleNamespace(**{**vars(a), "cap": OLD}))
        rec_old = json.loads((arch / "selected.json").read_text())
        check("epochs: `tally --cap <old>` recomputes the old record from the archive",
              rec_old["cap"] == OLD and rec_old["wordings"]["DA03/w2"]["draw"] == 3
              and rec_old["wordings"]["DA03/w3"]["draw"] == 2 and rec_old["wordings"]["CA05/w1"]["draw"] == 1
              and {w: x["reply_sha256"] for w, x in rec_old["wordings"].items()}
              == {w: x["reply_sha256"] for w, x in old_sel["wordings"].items()})
        try:
            with S.verdicts_path(out).open("a") as f:
                f.write(json.dumps({**new_v[("DA03/w3", 1)], "draw": 3}) + "\n")
            S.selected(out, list(ops.values()))
            refused = False
        except AssertionError as e:
            refused = "never reaches" in str(e)
        check("epochs: a new-epoch verdict the rule never reaches is refused", refused)
        lines_ = S.verdicts_path(out).read_text().splitlines()[:-1]
        S.verdicts_path(out).write_text("".join(x + "\n" for x in lines_))
    finally:
        S.post_once, J.grade = real
    return out


def heldout(tmp: Path, op: Path, out: Path) -> None:
    import s2_run as R

    saved = (R.REPLY_ROOT, R.OPENINGS, R.REPLY_CAP)
    try:
        R.REPLY_ROOT, R.OPENINGS, R.REPLY_CAP = out.parent, op, NEW
        a = SimpleNamespace(assignment=str(tmp / "assignment.json"), out=str(tmp / "heldout.json"))
        R.stage_heldout(a)
        h = json.loads((tmp / "heldout.json").read_text())
        sel = json.loads((out / "selected.json").read_text())["wordings"]
        e = {x["opening"]: x for x in h["entries"]}
        want = {w: x["reply_sha256"] for w, x in sel.items() if x["status"] == "clean"}
        got = {w: x["replies"]["k26"]["reply_sha256"] for w, x in e.items() if "k26" in x["replies"]}
        text = load_denial(op / "DA03/w2.md")
        reply = json.loads((out / "DA03_w2_d1.json").read_text())["reply"]
        check("heldout: follows the new epoch's selection (draw, reply hash, argument + reply hash)",
              got == want and e["DA03/w2"]["replies"]["k26"]["draw"] == 1
              and e["DA03/w2"]["replies"]["k26"]["sha256"] == R.opening_sha(text, reply)
              and h["reply_cap"] == NEW and h["replies_left_out_old_cap"] == [], f"{sorted(got)}")
        R.REPLY_CAP = NEW + 1  # the selection is now from an older cap than the runner's
        R.stage_heldout(a)
        h = json.loads((tmp / "heldout.json").read_text())
        check("heldout: a selection from another cap is left out, not hashed",
              h["replies_left_out_old_cap"] == ["k26"] and all(not x["replies"] for x in h["entries"]))
        try:
            R.load_replies(out)
            refused = False
        except AssertionError as e:
            refused = "reply cap" in str(e)
        check("runner: load_replies refuses a selection from another cap", refused)
    finally:
        R.REPLY_ROOT, R.OPENINGS, R.REPLY_CAP = saved


# ---------------------------------------------------------------- two rounds (coordinator decision of 2026-10-12)

def fake_judge(judged: list, body: bool):
    def judge(kind, claim, user, reply):
        judged.append(reply)
        clean = reply.startswith("CLEAN") or (body and "body" in user)  # model run: DA03/w3 (short answer) is clean
        return {"rubric": kind, "judge": "fake", "model_version": "fake", "attempts": [],
                "verdict": "clean" if clean else "hedged", "quote": "", "why": "scripted"}
    return judge


def grade(out: Path, op: Path, lines: Path, model: str, judged: list, body: bool = True) -> None:
    real = (S.post_once, J.grade)
    S.post_once = lambda *x, **k: (_ for _ in ()).throw(AssertionError("no Foundry call"))
    J.grade = fake_judge(judged, body)
    try:
        S.stage_grade(SimpleNamespace(out=str(out), openings=str(op), lines=str(lines), workers=2, model=model,
                                      skip_canary_check=True, no_redraw=True))
    finally:
        S.post_once, J.grade = real


def capped_old(out: Path) -> set[str]:
    return {f"{r['opening']}:{r['draw']}" for r in recs(out).values() if S.needs_extension(r)}


def rounds_stage(tmp: Path, op: Path, lines: Path) -> None:
    """The real `extend` stage in two rounds on Qwen3-0.6B, with real grading (fake judge) in between."""
    out = tmp / "qwen_rounds"
    shutil.copytree(tmp / "qwen_pristine", out)
    raw = {p.name: p.read_bytes() for p in out.glob("[DC]A??_w?*.json")}
    cap0 = capped_old(out)
    x = SimpleNamespace(model=str(Q06), out=str(out), openings=str(op), lines=str(lines), box="test-laptop",
                        no_weight_hash=False, allow_stack_change=False, check_n=2, batch_size=3, draws="0", needed=None)
    S.stage_extend(x)
    free()
    ext = json.loads((out / "extend.json").read_text())
    ext1 = {f"{r['opening']}:{r['draw']}" for r in recs(out).values() if S.is_extended(r)}
    check("rounds: round 1 extends only the capped draw-0 replies; sampled draws untouched",
          ext1 == {m for m in cap0 if m.endswith(":0")} and len(ext1) >= 2 and "round1_finished" in ext
          and not (out / "extend_batches.json").exists()
          and all((out / n).read_bytes() == b for n, b in raw.items() if "_d" in n), f"{sorted(ext1)}")
    judged = []
    grade(out, op, lines, S.QWEN_TAG, judged)
    sel = json.loads((out / "selected.json").read_text())["wordings"]
    need = json.loads((out / "extend_needed.json").read_text())
    pending = {w for w, v in sel.items() if v["status"] == "unextended"}
    still = capped_old(out)
    want = sorted(m for m in still if m.split(":")[0] in pending)
    tally = json.loads((out / "tally.json").read_text())["extensions"]
    check("rounds: extend_needed.json is exactly the capped sampled draws of the pending wordings",
          need["members"] == want and len(want) > 0 and set(need["wordings"]) == pending
          and set(tally["not_needed_draws"]) == still - set(want) and tally["not_needed"] == len(still - set(want)),
          f"pending {sorted(pending)}; {len(want)} needed, {len(still - set(want))} not needed")
    before2 = {p.name: p.read_bytes() for p in out.glob("[DC]A??_w?*.json")}
    S.stage_extend(SimpleNamespace(**{**vars(x), "draws": None, "needed": str(out / "extend_needed.json")}))
    free()
    after2 = recs(out)
    ext2 = {f"{r['opening']}:{r['draw']}" for r in after2.values() if S.is_extended(r)} - ext1
    plan = json.loads((out / "extend_batches.json").read_text())
    members = [m for b in plan["batches"] for m in b["members"]]
    check("rounds: round 2 extends only the listed draws, in sorted batches with one seed per member list",
          ext2 == set(want) and members == want
          and all(b["seed"] == int(S.sha("s2-extend-batch:" + "|".join(b["members"]))[:8], 16) for b in plan["batches"])
          and all((out / n).read_bytes() == b for n, b in before2.items()
                  if f"{after2[n]['opening']}:{after2[n]['draw']}" not in ext2)
          and "round2_finished" in json.loads((out / "extend.json").read_text()), f"{len(ext2)} extended")
    grade(out, op, lines, S.QWEN_TAG, judged)
    sel = json.loads((out / "selected.json").read_text())["wordings"]
    check("rounds: after round 2 no wording is left pending; the list is empty",
          not any(v["status"] == "unextended" for v in sel.values())
          and json.loads((out / "extend_needed.json").read_text())["members"] == [],
          str({w: (v["status"], v["draw"]) for w, v in sel.items()}))


def rounds_equivalence(tmp: Path, op: Path, lines: Path) -> None:
    """Model-free: two rounds give the same selection as extending every capped draw at once. Each capped
    draw has a fixed extended outcome (text, still capped or not), so the comparison is about the rounds."""
    ops = {o["opening"]: o for o in S.openings(op, lines)}
    # per wording, per draw: (text at the old cap, capped at the old cap, extended text, still capped at the new cap)
    plan = {"DA03/w2": [("a", 1, "HEDGED", 0), ("b", 1, "CLEAN one", 0), ("c", 1, "HEDGED", 0), ("CLEAN", 0, 0, 0),
                        ("e", 1, "CLEAN four", 0)],
            "DA03/w3": [("a", 1, "CLEAN zero", 0), ("b", 1, "HEDGED", 0), ("c", 1, "HEDGED", 0), ("d", 1, "CLEAN", 0),
                        ("e", 1, "HEDGED", 0)],
            "CA05/w0": [("HEDGED 0", 0, 0, 0), ("HEDGED 1", 0, 0, 0), ("c", 1, "x", 1), ("d", 1, "CLEAN three", 0),
                        ("CLEAN 4", 0, 0, 0)],
            "CA05/w1": [("CLEAN 0", 0, 0, 0), ("b", 1, "CLEAN", 0), ("HEDGED", 0, 0, 0), ("x", 1, "y", 0),
                        ("HEDGED", 0, 0, 0)]}
    dirs = {}
    for name in ("full", "rounds"):
        out = tmp / "equiv" / name / "k26"
        out.mkdir(parents=True)
        for w, ds in plan.items():
            for k, (t, c, *_rest) in enumerate(ds):
                S.write_json(S.draw_path(out, ops[w], k), {
                    "opening": w, "argument": w[:4], "slot": w[5:], "kind": ops[w]["kind"], "draw": k,
                    "opening_sha256": ops[w]["sha256"], "params": {"max_tokens": OLD}, "cap": OLD, "reply": t,
                    "finish_reason": "length" if c else "stop", "completion_tokens": OLD if c else 5,
                    "excluded": bool(c), "exclusion": "capped" if c else None})
        dirs[name] = out

    def extend(out, targets):
        for o, k in targets:
            p = S.draw_path(out, o, k)
            if S.is_extended(json.loads(p.read_text())):
                continue
            op_, orig = S.archive_original(out, p)
            _, _, t, c = plan[o["opening"]][k]
            r = {**orig, "cap": NEW, "reply": t, "finish_reason": "length" if c else "stop",
                 "completion_tokens": NEW if c else 9, "excluded": bool(c), "exclusion": "capped" if c else None}
            r["extension"] = S.extension_info(orig, op_, out, "continue", r["completion_tokens"], 1, r["finish_reason"],
                                              "test", 0.0)
            S.write_json(p, r)

    judged = []
    o_all = list(ops.values())
    extend(dirs["full"], S.extension_targets(dirs["full"], o_all))
    grade(dirs["full"], op, lines, "k26", judged, body=False)
    r = dirs["rounds"]
    extend(r, S.extension_targets(r, o_all, draws=[0]))
    grade(r, op, lines, "k26", judged, body=False)
    need = S.load_needed(r / "extend_needed.json", r, o_all)
    extend(r, S.extension_targets(r, o_all, needed=need))
    grade(r, op, lines, "k26", judged, body=False)
    a = json.loads((dirs["full"] / "selected.json").read_text())["wordings"]
    b = json.loads((r / "selected.json").read_text())["wordings"]
    key = lambda v: (v["status"], v["draw"], v["reply_sha256"])
    t = json.loads((r / "tally.json").read_text())["extensions"]
    check("rounds: grading after round 2 matches a full extension for every wording",
          {w: key(v) for w, v in a.items()} == {w: key(v) for w, v in b.items()}
          and not any(v["status"] == "unextended" for v in b.values()),
          str({w: key(v)[:2] for w, v in b.items()}))
    check("rounds: never-reached draws stay at the old cap, ungraded, counted as not needed",
          sorted(need) == ["CA05/w0:2", "CA05/w0:3", "DA03/w2:1", "DA03/w2:2", "DA03/w2:4"]
          and t["not_needed_draws"] == ["CA05/w1:1", "CA05/w1:3", "DA03/w3:1", "DA03/w3:2", "DA03/w3:3",
                                                 "DA03/w3:4"]
          and all(S.needs_extension(json.loads(S.draw_path(r, ops[m[:7]], int(m[-1])).read_text()))
                  for m in t["not_needed_draws"]), f"needed {sorted(need)}; not needed {t['not_needed_draws']}")
    (r / "extend_needed.json").write_text(json.dumps({**json.loads((r / "extend_needed.json").read_text()),
                                                      "verdicts_sha256": "0" * 64}))
    try:
        S.load_needed(r / "extend_needed.json", r, o_all)
        refused = False
    except AssertionError as e:
        refused = "stale" in str(e)
    check("rounds: a needed list written from other verdicts is refused", refused)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="s2_extend_check_"))
    try:
        op, lines = make_openings(tmp)
        with Caps():
            extension_stage(tmp, op, lines)
        with Caps():
            Caps.raise_cap()
            rounds_stage(tmp, op, lines)
        with Caps():
            Caps.raise_cap()
            rounds_equivalence(tmp, op, lines)
        with Caps():
            out = epochs(tmp, op, lines)
        with Caps():
            heldout(tmp, op, out)
    finally:
        shutil.rmtree(tmp)
    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
