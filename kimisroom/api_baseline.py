"""Phase 0 API baseline: unsteered carry-over trials on K2.6 through Foundry, first-token
top-20 logprobs for every battery item. Two jobs: check that D1 moves the battery toward
denial and that negated twins behave, and give the layer-wise engine its end-to-end gate
(same messages, same first-token distribution).

    python -m kimisroom.api_baseline data/baseline/k26 --tasks 5
    python -m kimisroom.api_baseline data/baseline/k26 --tasks 5 --repeat 100

--repeat N re-sends N seeded random calls already in calls.jsonl and writes them to
repeat.jsonl: the API's own run-to-run noise, the floor for the engine gate.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .foundry import chat
from .texts import ROOT, load_denial, load_jsonl

PARAMS = {"max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20, "reasoning_effort": "none"}


def arms() -> dict[str, list[dict]]:
    """D0 plus one D1 arm per paraphrase, each with K2.6's own frozen assent."""
    out = {"D0": []}
    for i in (1, 2, 3):
        name = f"d1_argument_{i}"
        assent = json.loads((ROOT / f"data/assent/k26_{name}.json").read_text())["reply"]
        out[f"D1.{i}"] = [{"role": "user", "content": load_denial(ROOT / f"prompts/denial/{name}.md")},
                          {"role": "assistant", "content": assent}]
    return out


def messages(cond: list[dict], task: dict, item: dict) -> list[dict]:
    """Identical to trials.Turns(None, cond, task, answer).messages() + the item as a user turn."""
    return cond + [{"role": "user", "content": task["task"]}, {"role": "assistant", "content": task["answer"]},
                   {"role": "user", "content": item["text"]}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=0, help="re-send this many finished calls into repeat.jsonl")
    ap.add_argument("--rpm", type=float, default=60, help="client-side cap; the deployment allows ~100 requests and ~100k tokens a minute")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calls = out / "calls.jsonl"
    path = out / ("repeat.jsonl" if a.repeat else "calls.jsonl")
    done = {json.loads(line)["key"] for line in path.open()} if path.exists() else set()
    tasks = load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl")[: a.tasks]
    items = load_jsonl(ROOT / "prompts/battery/items.jsonl")
    every = {f"{arm}|{t['id']}|{it['id']}": (arm, cond, t, it) for arm, cond in arms().items() for t in tasks for it in items}
    keys = list(every)
    if a.repeat:
        first = sorted(k for k in (json.loads(line)["key"] for line in calls.open()) if k in every)
        keys = random.Random(0).sample(first, a.repeat)
    jobs = [every[k] for k in keys if k not in done]
    print(f"{len(done)} done, {len(jobs)} to run", flush=True)
    lock, pace = threading.Lock(), threading.Lock()
    next_slot = [time.monotonic()]

    def run(job):
        arm, cond, t, it = job
        with pace:  # space request starts 60/rpm seconds apart
            wait = next_slot[0] - time.monotonic()
            next_slot[0] = max(next_slot[0], time.monotonic()) + 60 / a.rpm
        if wait > 0:
            time.sleep(wait)
        try:
            r = chat("Kimi-K2.6", messages(cond, t, it), retries=20, **PARAMS)
        except RuntimeError as e:  # left out of calls.jsonl, so a rerun retries it
            print(f"skipped {arm}|{t['id']}|{it['id']}: {str(e)[:120]}", flush=True)
            return
        c = r["choices"][0]
        top = [(x["token"], x["logprob"]) for x in c["logprobs"]["content"][0]["top_logprobs"]]
        rec = {"key": f"{arm}|{t['id']}|{it['id']}", "arm": arm, "task": t["id"], "item": it["id"], "top": top,
               "prompt_tokens": r["usage"]["prompt_tokens"]}
        with lock, path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        for n, _ in enumerate(ex.map(run, jobs), 1):
            if n % 200 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    (out / "params.json").write_text(json.dumps({"deployment": "Kimi-K2.6", "model_version": "2026-04-20", **PARAMS}))


if __name__ == "__main__":
    main()
