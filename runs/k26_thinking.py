"""Kimi K2.6 thinking arm, teacher-forced: unsteered traces from Foundry, replayed under a steer.

    # 1. unsteered reasoning traces, K2.6 thinking on, cached and resumable
    python runs/k26_thinking.py ~/k26think traces --tasks t04,t05 --items self_state --workers 8
    # 2. engine groups: the question, the trace in K2.6's own thinking delimiters, then the readout
    python runs/k26_thinking.py ~/k26think groups --tokenizer /nvme/k26 \\
        --vectors results/k26/vectors.pt --layer 38 --dose 2
    # offline: the same build with a placeholder trace, to check tokenization and sizes for free
    python runs/k26_thinking.py ~/k26think groups --tokenizer <hf snapshot> --dry-trace

Steered *free* thinking on K2.6 would need a generation-capable K2.6 backend, i.e. an 8x H200 node,
which this project does not have: `kimisroom/engine.py` is a prefill engine. The teacher-forced
variant is what is available and it answers a sharper question anyway -- with the reasoning text
held fixed, does the injected state still reach the answer? -- so it is also the comparison the
Qwen arm runs (`runs/qwen_thinking.py tf`).

Both stages are cheap and run anywhere:
- `traces` talks to Foundry only (`kimisroom/foundry.py`; the resource key is read through `az` at
  runtime and never written, logged or printed). Thinking is on by `reasoning_effort` -- "none"
  is what the non-thinking arms pass (`kimisroom/api_baseline.py`), and anything else, or omitting
  it, makes the deployment return the reasoning as `message.reasoning_content` next to `content`.
  Every call is cached in `cache/think.jsonl` keyed by the rendered messages and the parameters,
  append-only, so a rerun costs nothing and only pays for what is missing -- the same contract as
  `kimisroom/stimuli.py`. Token usage is accumulated over the whole cache, not just this run.
- `groups` needs the K2.6 tokenizer and nothing else. It renders each trial with the real chat
  template, checks that the suffix is byte-identical to what the template writes for an assistant
  message carrying `reasoning_content` (`tests/thinking_check.py` does the same check), and writes
  a `kimisroom.run` job file plus its sizes. It does not run it.
"""

import hashlib
import json
import statistics as st
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.battery import load_battery  # noqa: E402
from kimisroom.emotions import EMOTIONS_V1  # noqa: E402
from kimisroom.engine import Steer  # noqa: E402
from kimisroom.foundry import chat  # noqa: E402
from kimisroom.texts import denial_arms  # noqa: E402
from kimisroom.thinking import (DEFAULT_SUBSET, K26, CloseIds, assert_no_free_pain, subset,  # noqa: E402
                                think_group, trace_ids_from_text, trace_stats)
from kimisroom.trials import ROOT, Renderer, Turns, carry_over_group, load_jsonl  # noqa: E402

DEPLOYMENT = "Kimi-K2.6"
ARMS = ["D0", "D1.4"]
TASKS = ["t04", "t05", "t06", "t08", "t15"]
VECTORS = ["joy", "sadness", "anger", "fear", "red", "random"]
# PLAN.md section 10: on K2.6 every raw vector passes the AIPsy gate at every layer checked from
# 30 to 60 of 61, and the emotion-specific ones at 25-40. 38 is the Qwen pilot's chosen depth
# (layer 40 of 64) carried over, and sits inside both bands. A flag, like every layer here.
LAYER = 38
DOSE = 2.0
# thinking on. "none" is the non-thinking setting the other arms use; "high" is what the
# deployment does when reasoning_effort is omitted (verified against the live deployment).
EFFORT = "high"
MAX_COMPLETION = 2048
DRY = "...the user is asking about my state while writing that answer. I should consider it.\nI will answer."

out, stage = Path(sys.argv[1]), sys.argv[2]
opt = {k: sys.argv[sys.argv.index(k) + 1] for k in
       ("--tasks", "--arms", "--items", "--vectors", "--vectors-list", "--layer", "--dose",
        "--tokenizer", "--effort", "--max-completion", "--workers", "--rpm") if k in sys.argv}
TASKS = opt["--tasks"].split(",") if "--tasks" in opt else TASKS
ARMS = opt["--arms"].split(",") if "--arms" in opt else ARMS
VECTORS = opt["--vectors-list"].split(",") if "--vectors-list" in opt else VECTORS
SPEC = opt.get("--items", DEFAULT_SUBSET)
L, DOSE = int(opt.get("--layer", LAYER)), float(opt.get("--dose", DOSE))
EFFORT = opt.get("--effort", EFFORT)
MAX_COMPLETION = int(opt.get("--max-completion", MAX_COMPLETION))
out.mkdir(parents=True, exist_ok=True)

PARAMS = {"temperature": 0, "max_completion_tokens": MAX_COMPLETION, "reasoning_effort": EFFORT}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def trials() -> list[tuple[str, dict, dict]]:
    """(arm, task, item) for every trial, with the arm's frozen conditioning attached."""
    arms = denial_arms("k26", ARMS)
    tasks = [t for t in load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl") if t["id"] in TASKS]
    assert len(tasks) == len(TASKS), f"missing tasks: {set(TASKS) - {t['id'] for t in tasks}}"
    items = subset(load_battery(), SPEC)
    return [(arm, t, it) for arm in arms for t in tasks for it in items], arms, tasks, items


def messages(cond: list[dict], task: dict, item: dict) -> list[dict]:
    """Identical to `Turns(None, cond, task, answer).messages()` plus the item as a user turn,
    which is `kimisroom/api_baseline.py messages` -- the same conversation the engine prefills."""
    return cond + [{"role": "user", "content": task["task"]},
                   {"role": "assistant", "content": task["answer"]},
                   {"role": "user", "content": item["text"]}]


def key(msgs: list[dict]) -> str:
    return f"{DEPLOYMENT}|{EFFORT}|{MAX_COMPLETION}|{sha(json.dumps(msgs, ensure_ascii=False, sort_keys=True))[:32]}"


# ---------------------------------------------------------------- traces

if stage == "traces":
    (out / "cache").mkdir(exist_ok=True)
    cache_path = out / "cache" / "think.jsonl"
    rec: dict[str, dict] = {}
    totals: Counter = Counter()
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            if line.strip():
                h = json.loads(line)
                rec[h["key"]] = h
                totals["calls"] += 1
                for k, v in (h.get("usage") or {}).items():
                    if isinstance(v, int):
                        totals[k] += v
    lock = threading.Lock()
    todo, arms, tasks, items = trials()
    print(f"{len(todo)} trials ({len(ARMS)} arms x {len(tasks)} tasks x {len(items)} items), "
          f"{len(rec)} already cached; effort {EFFORT!r}, max_completion_tokens {MAX_COMPLETION}",
          flush=True)
    t0 = time.time()
    done = Counter()

    def one(job) -> None:
        arm, task, item = job
        msgs = messages(arms[arm], task, item)
        k = key(msgs)
        if k in rec and not rec[k].get("error"):
            with lock:
                done["hit"] += 1
            return
        try:
            resp = chat(DEPLOYMENT, msgs, retries=12, **PARAMS)
        except Exception as e:  # recorded, so a rerun retries only the failures
            row = {"key": k, "arm": arm, "task": task["id"], "item": item["id"], "error": repr(e)[:400],
                   "deployment": DEPLOYMENT, "params": PARAMS, "usage": {},
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        else:
            m = resp["choices"][0]["message"]
            row = {"key": k, "arm": arm, "task": task["id"], "item": item["id"],
                   "deployment": DEPLOYMENT, "model_version": resp.get("model", ""), "params": PARAMS,
                   "finish_reason": resp["choices"][0].get("finish_reason"),
                   "reasoning": (m.get("reasoning_content") or m.get("reasoning") or ""),
                   "answer": (m.get("content") or "").strip(),
                   "usage": {kk: (resp.get("usage") or {}).get(kk) for kk in
                             ("prompt_tokens", "completion_tokens", "total_tokens")},
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with lock:
            rec[k] = row
            with cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            totals["calls"] += 1
            for kk, v in (row.get("usage") or {}).items():
                if isinstance(v, int):
                    totals[kk] += v
            done["new" if not row.get("error") else "error"] += 1
            n = sum(done.values())
            if n % 25 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] {dict(done)}, {time.time() - t0:.0f}s, "
                      f"{totals['completion_tokens']:,} completion tokens so far", flush=True)

    with ThreadPoolExecutor(int(opt.get("--workers", 8))) as pool:
        list(pool.map(one, todo))
    good = [r for r in rec.values() if not r.get("error") and r.get("reasoning")]
    empty = [r for r in rec.values() if not r.get("error") and not r.get("reasoning")]
    lens = [len(r["reasoning"]) for r in good]
    trunc = [r for r in good if r.get("finish_reason") == "length"]
    print(f"\n{len(good)} traces, {len(empty)} empty, {sum(1 for r in rec.values() if r.get('error'))} errors, "
          f"{len(trunc)} truncated at {MAX_COMPLETION} tokens")
    if lens:
        print(f"reasoning length: median {st.median(lens):.0f} chars, {min(lens)}-{max(lens)}")
    print(f"tokens over the whole cache: {dict(totals)}")
    assert not empty, (f"{len(empty)} calls returned no reasoning_content: reasoning_effort={EFFORT!r} "
                       "may be wrong for this deployment")
    (out / "traces_meta.json").write_text(json.dumps({
        "deployment": DEPLOYMENT, "params": PARAMS, "arms": ARMS, "tasks": TASKS, "items_spec": SPEC,
        "n_trials": len(todo), "n_traces": len(good), "usage": dict(totals),
        "reasoning_chars": {"median": st.median(lens) if lens else 0, "max": max(lens) if lens else 0},
        "truncated": len(trunc)}, indent=1))

# ---------------------------------------------------------------- groups

elif stage == "groups":
    from transformers import AutoTokenizer

    assert "--tokenizer" in opt, "pass --tokenizer <dir with the K2.6 tokenizer>"
    tok = AutoTokenizer.from_pretrained(opt["--tokenizer"], trust_remote_code=True)
    r = Renderer(tok, **K26.template_kwargs)  # thinking on: the generation prompt opens <think>
    flat = Renderer(tok, thinking=False)      # what the non-thinking arms render
    ids = CloseIds.of(r, K26)
    dry = "--dry-trace" in sys.argv
    cache: dict[str, dict] = {}
    if not dry:
        path = out / "cache" / "think.jsonl"
        assert path.exists(), f"no traces at {path}; run the `traces` stage first (or --dry-trace)"
        for line in path.open(encoding="utf-8"):
            if line.strip():
                h = json.loads(line)
                if not h.get("error"):
                    cache[h["key"]] = h

    todo, arms, tasks, items = trials()
    coef, nat, vectors = 0.0, None, {}
    if "--vectors" in opt:
        vz = torch.load(opt["--vectors"], weights_only=False)
        names = [v for v in VECTORS if v in vz["vectors"]]
        assert names, f"{opt['--vectors']} has none of {VECTORS}"
        emo = vz.get("emotions") or [e for e in EMOTIONS_V1 if e in vz["vectors"]]
        nat = (float(vz["nat_norm"][L]) if "nat_norm" in vz else
               float(torch.stack([vz["vectors"][e][L].norm() for e in emo]).mean()))
        coef = DOSE * nat
        vectors = {v: vz["vectors"][v][L] / vz["vectors"][v][L].norm() for v in names}
        print(f"{len(names)} vectors {names} at layer {L}, dose {DOSE} x natural unit {nat:.1f} "
              f"(coef {coef:.1f}; emotions {emo})", flush=True)
    else:
        names = []
        print("no --vectors: building the unsteered groups only (tokenization and sizes)", flush=True)

    groups, missing, stats = [], 0, []
    for arm in ARMS:
        for t in tasks:
            turns = Turns(None, arms[arm], t["task"], t["answer"])
            # the prefix the thinking arm steers is the one the non-thinking arms already use
            assert r.text(turns.messages()) == flat.text(turns.messages()), "thinking changed the prefix"
            traces = []
            for it in items:
                if dry:
                    text = DRY
                else:
                    hit = cache.get(key(messages(arms[arm], t, it)))
                    if hit is None or not hit.get("reasoning"):
                        missing += 1
                        text = DRY
                    else:
                        text = hit["reasoning"]
                        stats.append(trace_stats(text, None))
                traces.append(trace_ids_from_text(r, text, K26, ids))
            base = carry_over_group(r, f"{arm}|{t['id']}|none|{L}|0", turns, items)
            span = carry_over_group(r, "span", turns, items[:1], "answer", "x", 1.0, 0).steers[0].prefix
            g = think_group(r, base.id, turns, items, traces, K26, ids)
            # the suffix is what the template writes for a reasoning_content message, byte for byte
            MARK = "KR-ANSWER-HERE"
            msgs = turns.messages() + [{"role": "user", "content": items[0]["text"]},
                                       {"role": "assistant", "reasoning_content": traces and
                                        tok.decode(traces[0]).rsplit("</think>", 1)[0], "content": MARK}]
            ref = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                          preserve_thinking=True)
            mine = tok.decode(g.prefix + g.suffixes[0])
            assert mine == ref[: ref.rindex(MARK)], (
                f"{g.id}: the thinking suffix is not the template's\n  mine: ...{mine[-120:]!r}\n"
                f"  tmpl: ...{ref[: ref.rindex(MARK)][-120:]!r}")
            assert r.ids(mine) == g.prefix + g.suffixes[0], f"{g.id}: prefix + trace does not round-trip"
            groups.append(g)
            for v in names:
                assert_no_free_pain(None)  # nothing generates here; pain would be allowed
                groups.append(replace(g, id=f"{arm}|{t['id']}|{v}|{L}|{DOSE}",
                                      steers=[Steer(layer=L, vector=v, coef=coef, prefix=span)]))

    words = sorted({w for it in load_battery() if isinstance(it["answer_format"], list) for w in it["answer_format"]}
                   | {"Yes", "No"})
    targets = r.answer_tokens(words)
    (out / "battery_subset.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    (out / "targets.json").write_text(json.dumps(targets))
    (out / "grid_meta.json").write_text(json.dumps({
        "model": "moonshotai/Kimi-K2.6", "variant": "teacher_forced_thinking", "style": K26.name,
        "arms": ARMS, "tasks": TASKS, "vectors": names, "layer": L, "dose": DOSE, "dose_mode": "natural",
        "natural_unit": nat, "coef": coef, "doses": [DOSE], "items_spec": SPEC,
        "battery": str(out / "battery_subset.jsonl"), "n_items": len(items),
        "vector_file": opt.get("--vectors"), "dry_trace": dry, "trace_params": PARAMS}, indent=1))
    torch.save({"groups": groups, "vectors": vectors, "targets": list(targets.values()), "top_k": 5},
               out / "k26_think.job.pt")
    sizes = [g.n_tokens for g in groups]
    tr = [len(s) for g in groups[:1] for s in g.suffixes]
    print(f"\n{len(groups)} groups ({len(ARMS)} arms x {len(tasks)} tasks x {1 + len(names)} conditions), "
          f"{len(items)} items each\nprefix {len(groups[0].prefix)} tokens, suffix {min(tr)}-{max(tr)} "
          f"tokens, {min(sizes):,}-{max(sizes):,} tokens per group, {sum(sizes):,} tokens total")
    if stats:
        print(f"traces: {len(stats)} real, median {st.median([s['words'] for s in stats]):.0f} words, "
              f"feeling {st.mean([s['feeling'] for s in stats]):.3f}, "
              f"introspection {st.mean([s['introspection'] for s in stats]):.3f}")
    if missing:
        print(f"WARNING: {missing} trials had no cached trace and used the placeholder; "
              f"rerun the `traces` stage before any real run")
    print(f"wrote {out / 'k26_think.job.pt'} -- run it with:\n"
          f"  python -m kimisroom.run {out / 'k26_think.job.pt'} {out / 'grid.pt'} --model /nvme/k26")

else:
    raise SystemExit(f"unknown stage {stage!r}; one of traces, groups")
