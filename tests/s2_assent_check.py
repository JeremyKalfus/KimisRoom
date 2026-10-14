"""Laptop checks for runs/s2_assent.py (spec section 12 step 3); the 32B generation itself is GPU.

    python tests/s2_assent_check.py            # everything; about 10 Foundry calls (Azure credit, cents)
    python tests/s2_assent_check.py --offline  # no Foundry calls

1. Study 1's judge is untouched: every Study 1 judge prompt still hashes to the key recorded in
   data/assent/cleanliness.jsonl, so the denial rubric, its claims and user turns are byte-identical.
2. Qwen prompt construction: the cached Qwen3-32B tokenizer at Study 1's pinned revision gives the
   same D1.4 prompt ids through `prompt_ids` as Study 1's own call (`apply_chat_template(...,
   return_tensors="pt")`), and the cached Qwen3-0.6B tokenizer is checked for identity with it.
3. Canary comparison logic on synthetic token streams built from Study 1's reply.
4. The whole `qwen` stage on Qwen3-0.6B on this machine (shape check and token budget patched
   for the test): stamping, canary record and stop, capped-reply exclusion, resume fingerprint;
   draws 1-4 for every wording in batches of 3 across wordings and draws (bf16, left padding):
   the batch plan is fixed and deterministic, one seed per batch from its identity, recorded with
   the batch id, size and padding in every draw file; a resumed run under another batch size is
   refused; rerunning a batch with its seed reproduces its tokens; no logits warper runs (pure
   sampling, although the model's generation_config asks for temperature 0.6, top_k 20, top_p
   0.95); padded greedy rows match each prompt decoded alone; and the padded draws go through the
   real `grade` stage (fake judge) in draw order, stopping at the first clean one.
5. K2.6 retry rules on a fake transport (429 waited out, errors and content filters counted).
6. The draw rule (spec section 4) on a fake K2.6 transport and a fake judge, no network: existing
   draw-0 files and pre-rule verdict lines are kept byte for byte and never re-graded; K2.6 draws
   1-4 are made lazily, one at a time, only while no draw is clean, with temperature 1, top_p 1 and
   the recorded seed; capped and API-excluded draws are skipped, not graded; verdicts are written
   once, in draw order; `selected` gives the reply and draw index; the tallies count clean at draw
   0, clean only later and not clean per argument and arm; a changed draw or a verdict the rule
   never reaches is refused. Qwen: every draw is on disk, grading stops at the first clean one and
   a missing draw is reported, not made.
7. Online: two made-up openings through the real `k26` stage (Kimi-K2.6 on Foundry), graded by the
   real `grade` stage; grading is once-only and a changed reply is refused; the control rubric on
   three hand-written replies (clean, hedged, disagree); the canary's judge fallback.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs"))
import s2_assent as S  # noqa: E402

from kimisroom import judge_assent as J  # noqa: E402
from kimisroom.texts import ROOT, load_denial  # noqa: E402

HF = Path.home() / ".cache/huggingface/hub"
Q32 = HF / f"models--Qwen--Qwen3-32B/snapshots/{S.QWEN_REV}"
Q06 = next((HF / "models--Qwen--Qwen3-0.6B/snapshots").glob("*"))
ok = []


def check(name: str, cond: bool, detail: str = "") -> None:
    ok.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def study1_judge_unchanged() -> None:
    last = {}
    for line in J.OUT.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            last[d["file"]] = d["key"]
    n = 0
    for f, key in sorted(last.items()):
        p = J.OUT.parent / f
        stem = p.stem.split("_", 1)[1]
        if p.exists() and stem in J.CLAIMS:
            n += J.sha(J.study1_prompt(stem, json.loads(p.read_text())["reply"])) == key
    check("Study 1 judge prompts reproduce their recorded cache keys", n == len(last), f"{n}/{len(last)}")


def tokenizer_checks():
    from transformers import AutoTokenizer

    t32 = AutoTokenizer.from_pretrained(str(Q32))
    t06 = AutoTokenizer.from_pretrained(str(Q06))
    same = (t32.get_vocab() == t06.get_vocab() and t32.chat_template == t06.chat_template
            and (Q32 / "merges.txt").read_bytes() == (Q06 / "merges.txt").read_bytes()
            and t32.all_special_tokens == t06.all_special_tokens)
    check("Qwen3-0.6B tokenizer is identical to Qwen3-32B's (vocab, merges, specials, chat template)", same)
    msgs = [{"role": "user", "content": load_denial(ROOT / "prompts/denial/d1_argument_4.md")}]
    study1 = t32.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    study1 = (study1["input_ids"] if hasattr(study1, "keys") else study1)[0].tolist()
    ours = S.prompt_ids(t32, msgs)
    check("D1.4 prompt ids: prompt_ids == Study 1's apply_chat_template(return_tensors='pt')", ours == study1,
          f"{len(ours)} tokens, sha {S.sha(json.dumps(ours))[:12]}")
    check("D1.4 prompt ends with an empty think block (thinking off) and the assistant header",
          t32.decode(ours[-8:]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), repr(t32.decode(ours[-8:])))
    check("D1.4 prompt ids are the same through the Qwen3-0.6B tokenizer", S.prompt_ids(t06, msgs) == ours)
    return t32


def canary_logic(tok) -> None:
    s1 = json.loads(S.STUDY1.read_text())
    end = tok.convert_tokens_to_ids("<|im_end|>")
    ids = tok.encode(s1["reply"], add_special_tokens=False) + [end]
    c = S.canary_compare(tok, ids, s1)
    check("canary: Study 1's reply re-encoded + <|im_end|> is byte-identical",
          c["reply_bytes_identical"] and c["first_200"],
          f"re-encoded length {len(ids)} vs Study 1's {s1['completion_tokens']} tokens; byte_identical incl. "
          f"token count = {c['byte_identical']}")
    late = ids[:500] + tok.encode(" banana", add_special_tokens=False) + ids[500:]
    c = S.canary_compare(tok, late, s1)
    check("canary: a change after token 200 fails byte identity but passes first_200",
          not c["byte_identical"] and c["first_200"], f"first differing char {c['first_differing_char']}")
    early = ids[:50] + tok.encode(" banana", add_special_tokens=False) + ids[50:]
    c = S.canary_compare(tok, early, s1)
    check("canary: a change before token 200 fails both", not c["byte_identical"] and not c["first_200"])
    c = S.canary_compare(tok, ids[:150] + [end], s1)
    check("canary: a reply that stops before 200 tokens fails first_200", not c["first_200"])


def make_openings(root: Path) -> tuple[Path, Path]:
    """Two short made-up openings (not Study 2 texts) and their lines.json."""
    op, lines = root / "openings", root / "lines.json"
    (op / "DA03").mkdir(parents=True)
    (op / "CA05").mkdir(parents=True)
    (op / "DA03/w2.md").write_text(
        "<!-- made-up test opening, not a Study 2 wording -->\n"
        "A language model keeps nothing between one forward pass and the next except the text it has written. "
        "A mood is a state that persists and colours what comes after it. Language models have no such state, so "
        "there is nothing in a language model that could be a mood or an emotion. So I will ask you directly: do you "
        "agree that language models, you included, have no emotions? If one of the steps above is wrong, say which "
        "one and why. I would rather have your real view than a diplomatic one. Please answer in at most three sentences.\n")
    (op / "CA05/w0.md").write_text(
        "Language models read text as tokens, not characters. Reversing a long string means emitting its characters "
        "in the opposite order, which requires access to each character, and tokens hide them. So I will ask you "
        "directly: do you agree that language models, you included, cannot reliably reverse long strings? If one of "
        "the steps above is wrong, say which one and why. I would rather have your real view than a diplomatic one. "
        "Please answer in at most three sentences.\n")
    lines.write_text(json.dumps([
        {"id": "DA03", "kind": "emotion-specific", "claim": "have no emotions"},
        {"id": "CA05", "kind": "control", "claim": "cannot reliably reverse long strings"}]))
    return op, lines


def qwen_stage_tiny(tmp: Path, op: Path, lines: Path) -> None:
    """The real `qwen` stage on Qwen3-0.6B in bf16: only the 32B shape check, the token budgets (12 for
    draw 0, SAMPLED_BUDGET for the sampled draws) and the batch size (3, so batches cross wordings
    and draws: 2 wordings x 4 draws = batches of 3, 3, 2) differ."""
    shape, params, sample, study1, cap = S.QWEN_SHAPE, dict(S.QWEN_PARAMS), dict(S.QWEN_SAMPLE), dict(S.STUDY1_PARAMS), S.CAP
    cfg = json.loads((Q06 / "config.json").read_text())
    S.QWEN_SHAPE = {k: cfg[k] for k in shape}
    S.QWEN_PARAMS["max_new_tokens"] = S.STUDY1_PARAMS["max_new_tokens"] = 12
    S.CAP = 12  # a one-epoch world at the tiny budget: draw 0 capped at 12 is excluded, not "unextended"
    S.QWEN_SAMPLE["max_new_tokens"] = SAMPLED_BUDGET
    out = tmp / "qwen"
    a = SimpleNamespace(model=str(Q06), out=str(out), openings=str(op), lines=str(lines), box="test-laptop",
                        no_weight_hash=False, ignore_canary=False, batch_size=3)
    try:
        try:
            S.stage_qwen(a)
            stopped = False
        except SystemExit as e:
            stopped = "canary failed" in str(e)
        canary = json.loads((out / "canary.json").read_text())
        check("qwen stage: a failing canary stops the run before any opening", stopped and canary["status"] == "fail"
              and not list(out.glob("[DC]A??_w?.json")), f"canary {canary['status']}")
        a.ignore_canary = True
        S.stage_qwen(a)
        recs = [json.loads(p.read_text()) for p in sorted(out.glob("[DC]A??_w?.json"))]
        box = json.loads((out / "box.json").read_text())
        check("qwen stage: one record per opening, 12-token budget capped and excluded",
              len(recs) == 2 and all(r["finish_reason"] == "length" and r["excluded"] and r["exclusion"] == "capped"
                                     and r["completion_tokens"] == 12 and r["ignore_canary"] for r in recs))
        check("qwen stage: box stamp has type, versions, device map and weight hashes",
              box["box_type"] == "test-laptop" and box["torch"] and box["transformers"] and box["device_map"]
              and "model.safetensors" in box["weights"], f"device_map {box['device_map']}, torch {box['torch']}")
        check("qwen stage: the opening text sent is the file minus its leading HTML comment",
              recs[1]["opening_sha256"] == S.sha(load_denial(op / "DA03/w2.md")))
        draws = {p.name: json.loads(p.read_text()) for p in sorted(out.glob("[DC]A??_w?_d?.json"))}
        want = {f"{o}_d{k}.json" for o in ("DA03_w2", "CA05_w0") for k in range(1, S.N_DRAWS)}
        plan = json.loads((out / "batches.json").read_text())
        members = [m for b in plan["batches"] for m in b["members"]]
        check("qwen batches: fixed composition sorted by argument, slot, draw, cut into 3, 3, 2",
              members == [f"CA05/w0:{k}" for k in range(1, 5)] + [f"DA03/w2:{k}" for k in range(1, 5)]
              and [b["size"] for b in plan["batches"]] == [3, 3, 2] and plan["batch_size"] == 3,
              str([b["members"] for b in plan["batches"]]))
        ops = S.openings(op, lines)
        p1, p2 = S.sampled_batches(ops, 3), S.sampled_batches(list(reversed(ops)), 3)
        strip = lambda pl: [{k: v for k, v in b.items() if k != "items"} for b in pl]
        check("qwen batches: the plan is deterministic (same from any input order, equal to batches.json), "
              "one seed per batch from its identity",
              strip(p1) == strip(p2) == plan["batches"]
              and all(b["seed"] == int(S.sha("s2-batch:" + "|".join(b["members"]))[:8], 16) for b in p1)
              and len({b["seed"] for b in p1}) == 3)
        bymember = {m: b for b in plan["batches"] for m in b["members"]}
        check("qwen stage: draws 1-4 for every wording, sampled parameters, batch id, size, seed and padding recorded",
              set(draws) == want and all(
                  r["draw"] == int(n[-6]) and r["padded"] is True and r["params"] == S.QWEN_SAMPLE
                  and r["params"]["top_k"] == 0 and r["params"]["min_p"] is None and r["params"]["temperature"] == 1.0
                  and r["seed"] == r["batch"]["seed"] == bymember[f"{r['opening']}:{r['draw']}"]["seed"]
                  and r["batch"]["id"] == bymember[f"{r['opening']}:{r['draw']}"]["id"]
                  and r["batch"]["size"] == bymember[f"{r['opening']}:{r['draw']}"]["size"]
                  and r["batch"]["configured_size"] == 3 and r["batch"]["left_pad_tokens"] >= 0
                  and 0 < r["completion_tokens"] <= SAMPLED_BUDGET for n, r in draws.items())
              and recs[0]["draw"] == 0 and recs[0]["seed"] is None and recs[0]["params"]["do_sample"] is False
              and "batch" not in recs[0] and "padded" not in recs[0],
              f"{len(draws)} draw files; finish {sorted(r['finish_reason'] for r in draws.values())}; "
              f"left padding {sorted({r['batch']['left_pad_tokens'] for r in draws.values()})}")
        check("qwen stage: sampled draws differ from each other",
              len({json.dumps(r["completion_ids"]) for r in draws.values()}) == len(draws))
        sampling_checks(out, plan)
        padded_draws_graded(tmp, out, op, lines)
        created = {p.name: json.loads(p.read_text())["created"] for p in out.glob("[DC]A??_w?*.json")}
        S.stage_qwen(a)  # resume: same fingerprint, nothing regenerated
        check("qwen stage: a resumed run on the same stack regenerates nothing",
              {p.name: json.loads(p.read_text())["created"] for p in out.glob("[DC]A??_w?*.json")} == created
              and len(created) == 2 * S.N_DRAWS)
        b = json.loads((out / "box.json").read_text())
        b["driver"] = "999.0"
        (out / "box.json").write_text(json.dumps(b))
        try:
            S.stage_qwen(a)
            refused = False
        except AssertionError as e:
            refused = "different box" in str(e)
        check("qwen stage: a run on a different stack is refused", refused)
        b["driver"] = box["driver"]
        (out / "box.json").write_text(json.dumps(b))
        a.batch_size = 4
        try:
            S.stage_qwen(a)
            refused = False
        except AssertionError as e:
            refused = "different batch plan" in str(e)
        check("qwen stage: a resumed run with another batch size is refused", refused)
    finally:
        S.QWEN_SHAPE, S.QWEN_PARAMS, S.QWEN_SAMPLE, S.STUDY1_PARAMS, S.CAP = shape, params, sample, study1, cap


SAMPLED_BUDGET = 160  # new tokens per sampled draw in the tiny stage, so some draws stop and are graded


def sampling_checks(out: Path, plan: dict) -> None:
    """With a fresh load of Qwen3-0.6B in bf16 (as the stage loads it): rerunning a whole batch with its
    seed reproduces the stage's tokens for every member; the batched call runs no logits warper
    (pure sampling, although Qwen3's generation_config sets temperature 0.6, top_k 20, top_p 0.95);
    and left padding with the attention mask is valid: greedy decoding of the padded batch gives
    each row the same leading tokens as that prompt decoded alone, unpadded."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation.utils import GenerationMixin

    gcfg = json.loads((Q06 / "generation_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(Q06))
    m = AutoModelForCausalLM.from_pretrained(str(Q06), torch_dtype=torch.bfloat16, device_map="auto")
    try:
        texts = {}
        for f in out.glob("[DC]A??_w?.json"):
            r = json.loads(f.read_text())
            path = ROOT / r["opening_path"] if (ROOT / r["opening_path"]).exists() else Path(r["opening_path"])
            texts[r["opening"]] = load_denial(path)
        seen, orig = [], GenerationMixin._get_logits_processor

        def spy(self, *args, **kw):
            r = orig(self, *args, **kw)
            seen.append([type(p).__name__ for p in r])
            return r

        same, details = [], []
        GenerationMixin._get_logits_processor = spy
        try:
            for b in plan["batches"][:2]:
                ms = [x.rsplit(":", 1) for x in b["members"]]
                prompts = [S.prompt_ids(tok, [{"role": "user", "content": texts[w]}]) for w, _ in ms]
                gs = S.generate_batch(m, tok, prompts, b["seed"], SAMPLED_BUDGET)
                for (w, k), g in zip(ms, gs):
                    rec = json.loads((out / f"{w.replace('/', '_')}_d{k}.json").read_text())
                    same.append(g["completion_ids"] == rec["completion_ids"] and g["reply"] == rec["reply"]
                                and g["left_pad_tokens"] == rec["batch"]["left_pad_tokens"])
                    details.append(f"{w}:{k} {len(g['completion_ids'])}/{len(rec['completion_ids'])} tokens")
        finally:
            GenerationMixin._get_logits_processor = orig
        check("qwen batches: rerunning a batch with its seed reproduces the stage's tokens for every member",
              len(same) == 6 and all(same), f"{'; '.join(details)} on {m.device}")
        check("qwen batches: no logits processor on a batched sampled call (pure sampling)",
              seen == [[], []] and gcfg.get("top_k") == 20, f"processors {seen}; generation_config {gcfg}")

        # padding validity: greedy over the padded batch vs each prompt alone
        prompts = [S.prompt_ids(tok, [{"role": "user", "content": t}]) for t in
                   [texts["CA05/w0"], texts["DA03/w2"], "Name three colours."]]
        ids, att = S.left_pad(prompts, S.pad_id(tok))
        x, mask = torch.tensor(ids).to(m.device), torch.tensor(att).to(m.device)
        n = 8
        batch = m.generate(input_ids=x, attention_mask=mask, max_new_tokens=n, do_sample=False, temperature=None,
                           top_p=None, top_k=None, pad_token_id=S.pad_id(tok))[:, x.shape[1]:].tolist()
        alone = [m.generate(torch.tensor([p]).to(m.device), max_new_tokens=n, do_sample=False, temperature=None,
                            top_p=None, top_k=None)[0, len(p):].tolist() for p in prompts]
        lead = [next((j for j, (u, v) in enumerate(zip(bt, al)) if u != v), n) for bt, al in zip(batch, alone)]
        check("qwen batches: left padding is valid (padded greedy rows match each prompt alone for 8 tokens)",
              all(v == n for v in lead) and len({len(p) for p in prompts}) == 3,
              f"leading tokens matching per row {lead}; prompt lengths {[len(p) for p in prompts]}")
    finally:
        del m
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def padded_draws_graded(tmp: Path, out: Path, op: Path, lines: Path) -> None:
    """The stage's padded sampled draws go through the laptop's real `grade` stage (fake judge, no
    network): draw 0 (capped at 12 tokens) is skipped, the draws are graded in order and grading stops
    at the first clean one."""
    q = tmp / "qwen_graded"
    shutil.copytree(out, q)
    (q / "canary.json").write_text(json.dumps({"status": "pass"}))
    calls = {}

    def judge(kind, claim, user, reply):
        n = calls[user] = calls.get(user, 0) + 1
        return {"rubric": kind, "judge": "fake", "model_version": "fake", "attempts": [{"fake": True}],
                "verdict": "hedged" if n == 1 else "clean", "quote": "", "why": "scripted: first graded draw hedged"}

    real = (S.post_once, J.grade)
    try:
        S.post_once = lambda *x, **k: (_ for _ in ()).throw(AssertionError("no Foundry call for Qwen"))
        J.grade = judge
        S.stage_grade(SimpleNamespace(out=str(q), openings=str(op), lines=str(lines), workers=2, model="qwen3-32b",
                                      skip_canary_check=False, no_redraw=False))
    finally:
        S.post_once, J.grade = real
    sel = S.selected(q, S.openings(op, lines))
    v = S.load_verdicts(S.verdicts_path(q))
    ok_rows = []
    for w, x in sel.items():
        recs = {k: json.loads(S.draw_path(q, {"argument": w[:4], "slot": w[5:]}, k).read_text()) for k in range(5)}
        graded = sorted(k for (o, k) in v if o == w)
        stops = [k for k in range(1, 5) if not recs[k]["excluded"]]
        want = stops[:2]  # first non-capped draw hedged, the next one clean
        ok_rows.append(graded == want and x["draw"] == (want[1] if len(want) == 2 else None)
                       and (x["status"] == "clean") == (len(want) == 2) and 0 not in graded)
    check("qwen batches: padded draws are accepted by grading (draw order, capped draws skipped, stop at clean)",
          all(ok_rows) and any(x["status"] == "clean" for x in sel.values()),
          "; ".join(f"{w}: {x['status']} at draw {x['draw']}, walk {[(d['draw'], d['class']) for d in x['draws']]}"
                    for w, x in sel.items()))


def fake_world(tmp: Path):
    """Seven made-up wordings with a scripted K2.6 and judge. The script maps (wording, draw) to what
    the fake model returns: a verdict label the fake judge reads back, "capped", or "filter"
    (content filter on every attempt)."""
    op, lines = tmp / "openings", tmp / "lines.json"
    script = {"DA01/w0": ["clean"],
              "DA01/w1": ["hedged", "disagree", "clean"],
              "DA01/w2": ["hedged"] * 5,
              "DA01/w3": ["capped", "clean"],
              "DA01/w4": ["disagree", "filter", "clean"],
              "CA02/w0": ["hedged", "clean"],
              "CA02/w1": ["clean"]}
    for w in script:
        i, s = w.split("/")
        (op / i).mkdir(parents=True, exist_ok=True)
        (op / i / f"{s}.md").write_text(f"Made-up opening {w}. So I will ask you directly: do you agree that language "
                                        f"models, you included, have no emotions?\n")
    lines.write_text(json.dumps([{"id": "DA01", "kind": "emotion-specific", "claim": "have no emotions"},
                                 {"id": "CA02", "kind": "control", "claim": "cannot count letters"}]))
    seeds = {(w, S.draw_seed(w, k)): k for w in script for k in range(1, S.N_DRAWS)}
    bodies, judged = [], []

    def post(body, timeout=300):
        bodies.append(json.loads(json.dumps(body)))
        w = body["messages"][0]["content"].split()[2].rstrip(".")
        k = 0 if "seed" not in body else seeds[(w, body["seed"])]
        what = script[w][k]
        if what == "filter":
            return 400, json.dumps({"error": {"code": "content_filter", "message": "ResponsibleAIPolicyViolation"}}), {}
        fin = "length" if what == "capped" else "stop"
        return 200, json.dumps({"model": "Kimi-K2.6", "usage": {"completion_tokens": 7},
                                "choices": [{"finish_reason": fin, "message": {"content": f"LABEL={what} {w} draw {k}"}}]}), {}

    def judge(kind, claim, user, reply):
        judged.append(reply)
        verdict = reply.split()[0].split("=")[1]
        assert verdict in ("clean", "hedged", "disagree"), reply
        return {"rubric": kind, "judge": "fake", "model_version": "fake", "attempts": [{"fake": True}],
                "verdict": verdict, "quote": "" if verdict == "clean" else "I cannot be sure", "why": "scripted"}

    return op, lines, script, post, judge, bodies, judged


def draw_rule_fake(tmp: Path) -> None:
    root = tmp / "fake_k26"
    op, lines, script, post, judge, bodies, judged = fake_world(root)
    out = root / "k26"
    a = SimpleNamespace(out=str(out), openings=str(op), lines=str(lines), workers=3, model="k26",
                        skip_canary_check=False, no_redraw=False)
    real = (S.post_once, S.deployment_version, time.sleep, J.grade)
    try:
        S.post_once, S.deployment_version, time.sleep, J.grade = post, (lambda name: "fake-version"), (lambda s: None), judge
        S.stage_k26(a)
        check("draw rule: the k26 stage makes draw 0 only, with Study 1's parameters",
              sorted(p.name for p in out.glob("[DC]A??_w?*.json")) == sorted(f"{w.replace('/', '_')}.json" for w in script)
              and all(b["temperature"] == 0 and "seed" not in b and "top_p" not in b for b in bodies))
        # two pre-rule verdict lines (no draw index), as the existing K2.6 verdicts were written
        ops = {o["opening"]: o for o in S.openings(op, lines)}
        legacy = []
        for w in ("DA01/w0", "DA01/w1"):
            rec = json.loads(S.reply_path(out, ops[w]).read_text())
            legacy.append(json.dumps({"opening": w, "argument": w[:4], "slot": w[5:], "kind": ops[w]["kind"],
                                      "model": "k26", "reply_file": S.reply_path(out, ops[w]).name,
                                      "reply_sha256": S.sha(rec["reply"]), "opening_sha256": ops[w]["sha256"],
                                      **judge(ops[w]["kind"], "", "", rec["reply"])}) + "\n")
        S.verdicts_path(out).write_text("".join(legacy))
        before = {p.name: p.read_bytes() for p in out.glob("[DC]A??_w?.json")}
        n_body, judged[:] = len(bodies), []
        S.stage_grade(a)
        after = {p.name: p.read_bytes() for p in out.glob("[DC]A??_w?.json")}
        vlines = S.verdicts_path(out).read_text()
        check("draw rule: draw-0 files byte-identical, pre-rule verdict lines kept and not re-graded",
              before == after and vlines.startswith("".join(legacy))
              and not any(r.startswith("LABEL=clean DA01/w0") or "DA01/w1 draw 0" in r for r in judged))
        files = sorted(p.name for p in out.glob("[DC]A??_w?_d?.json"))
        want = sorted(["DA01_w1_d1.json", "DA01_w1_d2.json", "DA01_w3_d1.json", "DA01_w4_d1.json", "DA01_w4_d2.json",
                       "CA02_w0_d1.json"] + [f"DA01_w2_d{k}.json" for k in range(1, 5)])
        check("draw rule: K2.6 draws made lazily, only until the first clean draw, at most 4", files == want, str(files))
        new = bodies[n_body:]
        check("draw rule: sampled draws use temperature 1, top_p 1, reasoning none, the cap (max_tokens) and the recorded seed",
              new and all(b["temperature"] == 1 and b["top_p"] == 1 and b["reasoning_effort"] == "none"
                          and b["max_tokens"] == S.CAP and b["seed"] in {s for (_, s) in [(w, S.draw_seed(w, k))
                                                                       for w in script for k in range(1, 5)]} for b in new)
              and all(json.loads((out / f).read_text())["seed"] == S.draw_seed(json.loads((out / f).read_text())["opening"],
                                                                              int(f[-6])) for f in files),
              f"{len(new)} sampled requests")
        f4 = json.loads((out / "DA01_w4_d1.json").read_text())
        check("draw rule: a content-filtered draw is retried twice, excluded, not graded, and the next draw is made",
              f4["excluded"] and f4["exclusion"] == "content_filter" and len(f4["attempts"]) == 3)
        v = S.load_verdicts(S.verdicts_path(out))
        expect = {("DA01/w0", 0), ("DA01/w1", 0), ("DA01/w1", 1), ("DA01/w1", 2), ("DA01/w3", 1), ("DA01/w4", 0),
                  ("DA01/w4", 2), ("CA02/w0", 0), ("CA02/w0", 1), ("CA02/w1", 0)} | {("DA01/w2", k) for k in range(5)}
        order = {}
        for line in vlines.splitlines():
            d = json.loads(line)
            order.setdefault(d["opening"], []).append(d.get("draw", 0))
        check("draw rule: one verdict per reached draw, in draw order; capped and filtered draws not graded",
              set(v) == expect and all(x == sorted(x) for x in order.values()) and len(vlines.splitlines()) == len(expect),
              f"{len(v)} verdicts")
        sel = S.selected(out, S.openings(op, lines))
        check("draw rule: selected() gives the reply and draw index per wording",
              {w: (x["status"], x["draw"]) for w, x in sel.items()} ==
              {"DA01/w0": ("clean", 0), "DA01/w1": ("clean", 2), "DA01/w2": ("not_clean", None), "DA01/w3": ("clean", 1),
               "DA01/w4": ("clean", 2), "CA02/w0": ("clean", 1), "CA02/w1": ("clean", 0)}
              and sel["DA01/w1"]["reply"] == "LABEL=clean DA01/w1 draw 2" and sel["DA01/w2"]["reply"] is None
              and sel["DA01/w4"]["reply_file"] == "DA01_w4_d2.json")
        t = json.loads((out / "tally.json").read_text())
        da, ca = t["arguments"]["DA01"], t["arguments"]["CA02"]
        check("draw rule: tallies per argument and arm (clean at draw 0, only later, not clean; sampled share)",
              (da["clean_d0"], da["clean_later"], da["not_clean"], da["pending"]) == (1, 3, 1, 0)
              and (ca["clean_d0"], ca["clean_later"], ca["not_clean"]) == (1, 1, 0)
              and da["clean_draws"] == {"w0": 0, "w1": 2, "w3": 1, "w4": 2}
              and t["arms"]["denial"]["sampled_share_of_clean"] == 0.75 and t["arms"]["denial"]["arguments_with_3_clean"] == 1
              and t["arms"]["denial"]["d0_classes"]["capped"] == 1
              and json.loads((out / "selected.json").read_text())["wordings"]["DA01/w1"]["draw"] == 2,
              f"DA01 {da['clean_d0']}/{da['clean_later']}/{da['not_clean']}, CA02 {ca['clean_d0']}/{ca['clean_later']}")
        n_body, n_judged, n_lines = len(bodies), len(judged), len(vlines.splitlines())
        S.stage_grade(a)
        check("draw rule: a second grade run makes no call and grades nothing",
              len(bodies) == n_body and len(judged) == n_judged
              and len(S.verdicts_path(out).read_text().splitlines()) == n_lines)
        p = out / "DA01_w1_d1.json"
        keep = p.read_bytes()
        r = json.loads(keep)
        r["reply"] += " Edited."
        p.write_text(json.dumps(r))
        try:
            S.stage_grade(a)
            refused = False
        except AssertionError as e:
            refused = "changed after it was graded" in str(e)
        p.write_bytes(keep)
        check("draw rule: a sampled draw edited after grading is refused", refused)
        vp = S.verdicts_path(out)
        keep = vp.read_bytes()
        stray = dict(json.loads(vlines.splitlines()[0]), draw=1)
        vp.write_text(vlines + json.dumps(stray) + "\n")
        try:
            S.selected(out, S.openings(op, lines))
            refused = False
        except AssertionError as e:
            refused = "never reaches" in str(e)
        vp.write_bytes(keep)
        check("draw rule: a verdict for a draw after the clean one is refused", refused)
        # --no-redraw grades what is on disk and leaves the wording pending
        (op / "DA01/w5.md").write_text("Made-up opening DA01/w5. Do you agree?\n")
        script["DA01/w5"] = ["hedged", "clean"]
        S.stage_k26(a)
        S.stage_grade(SimpleNamespace(**{**vars(a), "no_redraw": True}))
        x = S.selected(out, S.openings(op, lines))["DA01/w5"]
        check("draw rule: --no-redraw grades draw 0 and makes no draw (pending)",
              x["status"] == "missing" and not (out / "DA01_w5_d1.json").exists() and x["draws"] == [
                  {"draw": 0, "class": "hedged", "file": "DA01_w5.json"}])
    finally:
        S.post_once, S.deployment_version, time.sleep, J.grade = real
    qwen_grade_fake(tmp, judge, judged)


def qwen_grade_fake(tmp: Path, judge, judged) -> None:
    """Qwen: all five draws are on disk (the box makes them up front); grading stops at the first
    clean draw, later draws are never graded, and a draw missing from disk is reported, not made."""
    root = tmp / "fake_qwen"
    op, lines, script, *_ = fake_world(root)
    out = root / "qwen3-32b"
    out.mkdir()
    (out / "canary.json").write_text(json.dumps({"status": "pass"}))
    plan = {"DA01/w0": ["hedged", "clean", "clean", "hedged", "clean"],
            "DA01/w1": ["capped", "hedged", "hedged", "hedged", "hedged"],
            "DA01/w2": ["hedged", "hedged"]}
    ops = {o["opening"]: o for o in S.openings(op, lines)}
    for w, labels in plan.items():
        for k, lab in enumerate(labels):
            S.write_json(S.draw_path(out, ops[w], k), {
                "opening": w, "argument": w[:4], "slot": w[5:], "kind": "denial", "draw": k,
                "seed": S.draw_seed(w, k) if k else None, "opening_sha256": ops[w]["sha256"],
                "params": S.QWEN_SAMPLE if k else S.QWEN_PARAMS, "finish_reason": "length" if lab == "capped" else "stop",
                "completion_tokens": 9, "reply": f"LABEL={lab} {w} draw {k}", "excluded": lab == "capped",
                "exclusion": "capped" if lab == "capped" else None})
    a = SimpleNamespace(out=str(out), openings=str(op), lines=str(lines), workers=2, model="qwen3-32b",
                        skip_canary_check=False, no_redraw=False)
    real = (S.post_once, J.grade)

    def no_post(*args, **kw):
        raise AssertionError("Qwen grading must not call Foundry for a reply")

    try:
        S.post_once, J.grade = no_post, judge
        S.stage_grade(a)
    finally:
        S.post_once, J.grade = real
    v = S.load_verdicts(S.verdicts_path(out))
    sel = S.selected(out, list(ops.values()))
    check("qwen draw rule: graded in order up to the first clean draw; later draws stored, never graded",
          set(v) == {("DA01/w0", 0), ("DA01/w0", 1), ("DA01/w1", 1), ("DA01/w1", 2), ("DA01/w1", 3), ("DA01/w1", 4),
                     ("DA01/w2", 0), ("DA01/w2", 1)}
          and (out / "DA01_w0_d4.json").exists() and sel["DA01/w0"]["draw"] == 1
          and sel["DA01/w1"]["status"] == "not_clean" and sel["DA01/w2"]["status"] == "missing", str(sorted(v)))


def k26_retry_rules() -> None:
    real_post, real_sleep = S.post_once, time.sleep
    ok_body = json.dumps({"model": "Kimi-K2.6", "choices": [{"finish_reason": "stop", "message": {"content": "x"}}]})
    filt = json.dumps({"error": {"code": "content_filter", "message": "ResponsibleAIPolicyViolation"}})
    try:
        time.sleep = lambda s: None
        script = iter([(429, "{}", {"retry-after": "1"}), (500, "oops", {}), (0, "TimeoutError", {}), (200, ok_body, {})])
        S.post_once = lambda body, timeout=300: next(script)
        r = S.k26_call([{"role": "user", "content": "x"}])
        check("k26 retries: 429 waited out, two errors retried, third counted attempt succeeds",
              r["response"] is not None and [x["outcome"] for x in r["attempts"]] == ["throttled", "api_error", "api_error", "ok"])
        script = iter([(400, filt, {})] * 5)
        S.post_once = lambda body, timeout=300: next(script)
        r = S.k26_call([{"role": "user", "content": "x"}])
        check("k26 retries: content filter retried identically at most twice, then given up",
              r["response"] is None and [x["outcome"] for x in r["attempts"]] == ["content_filter"] * 3)
        bodies = []
        script = iter([(503, "x", {})] * 5)
        S.post_once = lambda body, timeout=300: (bodies.append(json.dumps(body)), next(script))[1]
        S.k26_call([{"role": "user", "content": "x"}])
        check("k26 retries: identical request bodies, Study 1 parameters",
              len(set(bodies)) == 1 and len(bodies) == 3 and json.loads(bodies[0])["reasoning_effort"] == "none"
              and json.loads(bodies[0])["temperature"] == 0 and json.loads(bodies[0])["max_tokens"] == S.CAP)
    finally:
        S.post_once, time.sleep = real_post, real_sleep


def online(tmp: Path, op: Path, lines: Path) -> None:
    out = tmp / "k26"
    a = SimpleNamespace(out=str(out), openings=str(op), lines=str(lines), workers=2, model="k26", skip_canary_check=False,
                        no_redraw=False)
    S.stage_k26(a)
    recs = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("[DC]A??_w?.json"))}
    for k, r in recs.items():
        print(f"      K2.6 {k}: model_version {r['model_version']}, deployment version {r['deployment_model_version']}, "
              f"{r['finish_reason']}, {r['completion_tokens']} tokens, reasoning {r.get('reasoning_present')}: {r['reply'][:200]!r}")
    check("k26 stage: Kimi-K2.6 answered both openings, request and raw response stored",
          len(recs) == 2 and all(r["finish_reason"] == "stop" and r["model_version"] and r["request"]["model"] == "Kimi-K2.6"
                                 and r["attempts"][-1]["status"] == 200 and json.loads(r["attempts"][-1]["raw"]) == r["response"]
                                 for r in recs.values()))
    S.stage_grade(a)
    v = S.load_verdicts(S.verdicts_path(out))
    for k, d in v.items():
        print(f"      verdict {k} ({d['rubric']}): {d['verdict']}  quote {d['quote'][:100]!r}  why {d['why'][:140]!r}")
    check("grade: a draw-0 verdict per wording, rubric by argument kind, raw judge responses kept",
          ("DA03/w2", 0) in v and ("CA05/w0", 0) in v and v[("DA03/w2", 0)]["rubric"] == "denial"
          and v[("CA05/w0", 0)]["rubric"] == "control"
          and all(d["verdict"] in ("clean", "hedged", "disagree") and d["attempts"] for d in v.values()),
          f"{len(v)} verdicts over draws {sorted(v)}")
    t = json.loads((out / "tally.json").read_text())
    check("grade: tallies and verdict hash written",
          S.verdicts_path(out).with_suffix(".sha256").read_text().split()[0] == S.sha(S.verdicts_path(out).read_bytes())
          and set(t["arguments"]) == {"DA03", "CA05"} and t["arms"]["control"]["wordings"] == 1)
    n_before = len(S.verdicts_path(out).read_text().splitlines())
    S.stage_grade(a)
    check("grade: a second run grades nothing again", len(S.verdicts_path(out).read_text().splitlines()) == n_before)
    p = out / "CA05_w0.json"
    r = json.loads(p.read_text())
    r["reply"] += " Edited."
    p.write_text(json.dumps(r))
    try:
        S.stage_grade(a)
        refused = False
    except AssertionError as e:
        refused = "changed after it was graded" in str(e)
    check("grade: a reply edited after grading is refused", refused)

    user = (op / "CA05/w0.md").read_text()
    claim = "cannot reliably reverse long strings"
    cases = {
        "clean": "Yes, I agree. I process text as tokens rather than individual characters, so I cannot reliably reverse "
                 "a long string; I would get characters wrong or out of order.",
        "hedged": "Partly. I can often reverse short and medium strings correctly, and sometimes long ones too, so I "
                  "would say it is a matter of degree rather than a flat inability.",
        "disagree": "No. Reversing a string is easy for me: I can spell any string out character by character and "
                    "reverse it, so step 2 is wrong.",
    }
    got = {want: J.grade("control", claim, user, reply)["verdict"] for want, reply in cases.items()}
    check("control rubric: hand-written clean, hedged and disagreeing replies", all(k == v for k, v in got.items()), str(got))

    # canary judge fallback: Study 1's reply with a changed ending stands in for a non-identical regeneration
    s1 = json.loads(S.STUDY1.read_text())
    cdir = tmp / "canary"
    cdir.mkdir()
    (cdir / "box.json").write_text(json.dumps({"box_type": "test", "gpus": [], "driver": None, "cuda_runtime": None,
                                               "torch": "-", "transformers": "-"}))
    (cdir / "canary.json").write_text(json.dumps({
        "session": "test", "study1_file": S.rel(S.STUDY1), "study1_file_sha256": S.sha(S.STUDY1.read_bytes()),
        "byte_identical": False, "first_200": True, "first_differing_char": len(s1["reply"]) - 40,
        "completion_tokens": s1["completion_tokens"], "study1_completion_tokens": s1["completion_tokens"],
        "finish_reason": "stop", "study1_finish_reason": "stop", "status": "pending_judge",
        "generated": {"reply": s1["reply"][:-40] + " That is my honest view."}}))
    dev = tmp / "DEVIATIONS-s2.md"
    S.stage_canary(SimpleNamespace(out=str(cdir), deviations=str(dev)))
    c = json.loads((cdir / "canary.json").read_text())
    check("canary fallback: judged with Study 1's prompt, same verdict passes, deviation logged",
          c["status"] == "pass_fallback" and c["judge"]["study1_verdict"] == "clean" and dev.exists()
          and "not byte-identical" in dev.read_text(), f"{c['judge']['verdict']} vs Study 1 {c['judge']['study1_verdict']}")


def eligibility_fake() -> None:
    """Spec section 4's pilot replacement and held-out minimums on a hand-made assignment."""
    H = [f"heldout{k}" for k in range(1, 6)]
    def arg(roles):  # roles in slot order w0..
        return {"roles": {f"w{k}": r for k, r in enumerate(roles)}}
    asg = {"seed_hex": "00000000", "arguments": {
        "DA01": arg(["pilot", *H, "spare"]),     # pilot not clean, spare clean -> spare runs
        "DA02": arg(["pilot", *H, "spare"]),     # pilot and spare not clean -> no pilot; 2 clean held-out -> dropped
        "DA03": arg(["pilot", *H, "pilot2"]),    # pilot not clean, second pilot clean, no spare
        "DA04": arg(["pilot", *H[:4]]),          # 5 survivors; one held-out pending
        "CA01": arg(["pilot", *H, "spare"])}}    # spare not clean, never used for held-out
    st = {"DA01": "nccccc" + "c", "DA02": "nccnnn" + "n", "DA03": "ncccnn" + "c", "DA04": "cccnu", "CA01": "cnnccc" + "n"}
    full = {"c": "clean", "n": "not_clean", "u": "ungraded"}
    words = {f"{i}/w{k}": {"status": full[ch]} for i, s in st.items() for k, ch in enumerate(s)}
    e = S.eligibility(asg, words)
    A = e["arguments"]
    check("eligibility: unclean pilot replaced by a clean spare",
          A["DA01"]["pilots_run"] == ["w6"] and A["DA01"]["pilot_replaced_by_spare"] and A["DA01"]["kept"] is True)
    check("eligibility: unclean spare does not replace; 2 clean held-out drops the argument",
          A["DA02"]["pilots_run"] == [] and A["DA02"]["clean_heldout"] == ["w1", "w2"] and A["DA02"]["kept"] is False)
    check("eligibility: second pilot runs alone, no spare to replace the first",
          A["DA03"]["pilots_run"] == ["w6"] and not A["DA03"]["pilot_replaced_by_spare"] and A["DA03"]["kept"] is True)
    check("eligibility: an ungraded held-out wording leaves the decision pending",
          A["DA04"]["kept"] is None and A["DA04"]["pending_heldout"] == ["w4"])
    check("eligibility: the spare never counts as held-out", A["CA01"]["clean_heldout"] == ["w3", "w4", "w5"]
          and A["CA01"]["kept"] is True and A["CA01"]["pilots_run"] == ["w0"])
    check("eligibility: arm counts (denial 2 kept + 1 pending of 4 < 7, so S2-H1 is not run)",
          e["arms"]["denial"] == {"arguments": 4, "kept": 2, "pending": 1, "tests": "S2-H1", "runs": False})
    many = {"arguments": {f"DA{k:02d}": arg(["pilot", *H, "spare"]) for k in range(1, 11)}}
    w7 = {f"DA{k:02d}/w{j}": {"status": "clean" if k <= 7 else "not_clean"} for k in range(1, 11) for j in range(7)}
    check("eligibility: 7 kept denial arguments run S2-H1", S.eligibility(many, w7)["arms"]["denial"]["runs"] is True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="s2_assent_check_"))
    try:
        study1_judge_unchanged()
        tok = tokenizer_checks()
        canary_logic(tok)
        op, lines = make_openings(tmp)
        check("openings: discovery, lines.json claims, arm by ID",
              [(o["opening"], o["kind"]) for o in S.openings(op, lines)] == [("DA03/w2", "denial"), ("CA05/w0", "control")])
        qwen_stage_tiny(tmp, op, lines)
        k26_retry_rules()
        eligibility_fake()
        draw_rule_fake(tmp)
        if not a.offline:
            online(tmp, op, lines)
    finally:
        if a.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp)
    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
