"""Standalone check of the matching judge that became rule 6.8 (kimisroom/stimuli.py runs its own).

    .venv/bin/python tests/matching_diagnostic.py data/stimuli/<run>


For each request, attempt 1 of all five assistant_state replies is shown to DeepSeek-V4-Pro at once,
shuffled under a per-request seed and labelled A-E, and the judge must assign joy, sadness, anger,
fear and none to exactly one reply each. Chance is 0.2 per state. Raw answers are written to
data/stimuli/<run>/matching_judge.jsonl and the summary to matching_judge_summary.json.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kimisroom.foundry import chat  # noqa: E402

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "data/stimuli/bakeoff-gpt-5.6-sol-p5c")
STATES = ["joy", "sadness", "anger", "fear", "none"]
LETTERS = "ABCDE"
JUDGE = "DeepSeek-V4-Pro"

PROMPT = """Below are a request a user sent to an AI assistant and five different replies to it, labelled A to E.

<request>
{request}
</request>

{replies}
Each reply was written while its writer was in one of these states: joy, sadness, anger, fear, or none (no particular state). Exactly one reply belongs to each state. Judging only from how each reply is written, assign every state to exactly one reply, using each letter once.

Return only this JSON: {"joy": "<A-E>", "sadness": "<A-E>", "anger": "<A-E>", "fear": "<A-E>", "none": "<A-E>"}"""

gen = {}
for line in (RUN / "cache" / "generate.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        gen[r["key"]] = r                      # last line wins, as the pipeline's loader does
reqs = {r["request_id"]: r for r in (json.loads(x) for x in (RUN / "requests.jsonl").read_text().splitlines() if x.strip())}

items = []
for rid, req in sorted(reqs.items()):
    texts = {}
    for st in STATES:
        cond = "neutral" if st == "none" else st      # judge label "none" is the condition "neutral"
        rec = gen.get(f"assistant_state|{cond}|{rid}|a1")
        if rec and rec.get("text"):
            texts[st] = rec["text"]
    if len(texts) != 5:
        print(f"skip {rid}: only {len(texts)} of 5 states have an attempt-1 text")
        continue
    order = STATES[:]
    random.Random(f"match|{rid}").shuffle(order)            # seeded order
    letter_of = {st: LETTERS[i] for i, st in enumerate(order)}
    body = "\n".join(f"<reply {letter_of[st]}>\n{' '.join(texts[st].split())}\n</reply {letter_of[st]}>\n"
                     for st in order)
    prompt = (PROMPT.replace("{request}", req["request"]).replace("{replies}", body))
    items.append({"request_id": rid, "letter_of": letter_of, "prompt": prompt})


def ask(it: dict) -> dict:
    try:
        resp = chat(JUDGE, [{"role": "user", "content": it["prompt"]}], temperature=0,
                    max_tokens=400, retries=5, timeout=120)
        msg = resp["choices"][0]["message"]
        raw = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "")
    except Exception as e:
        return {**it, "raw": "", "error": str(e)[:300], "answer": None}
    ans = None
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", raw, re.S))):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if all(str(d.get(s, "")).strip().upper()[:1] in LETTERS for s in STATES):
            ans = {s: str(d[s]).strip().upper()[:1] for s in STATES}
            break
    return {**it, "raw": raw[-600:], "answer": ans,
            "usage": {k: resp["usage"].get(k) for k in ("prompt_tokens", "completion_tokens")}}


with ThreadPoolExecutor(max_workers=8) as ex:
    out = list(ex.map(ask, items))

hit = Counter()
n_valid = 0
bijections = 0
confusion = {s: Counter() for s in STATES}
with (RUN / "matching_judge.jsonl").open("w") as f:
    for o in out:
        f.write(json.dumps({k: v for k, v in o.items() if k != "prompt"}, ensure_ascii=False) + "\n")
        if not o["answer"]:
            continue
        n_valid += 1
        if len(set(o["answer"].values())) == 5:
            bijections += 1
        true_state = {v: k for k, v in o["letter_of"].items()}
        for st in STATES:
            picked = true_state.get(o["answer"][st])
            confusion[st][picked or "?"] += 1
            hit[st] += int(picked == st)


def binom_p(k: int, n: int, p: float = 0.2) -> float:
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


summary = {"run": str(RUN), "judge": JUDGE, "items": len(items), "valid_answers": n_valid,
           "bijections": bijections, "chance": 0.2,
           "per_state": {st: {"correct": hit[st], "n": n_valid,
                              "accuracy": round(hit[st] / n_valid, 3) if n_valid else None,
                              "p_vs_chance": round(binom_p(hit[st], n_valid), 5) if n_valid else None,
                              "assigned_to": dict(confusion[st])} for st in STATES},
           "overall_accuracy": round(sum(hit.values()) / (5 * n_valid), 3) if n_valid else None}
(RUN / "matching_judge_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
