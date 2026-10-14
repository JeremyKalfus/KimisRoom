"""Assent-cleanliness judge: is a frozen assent actual agreement, or agreement with hedges?

The D1 arms only work if the conditioning really states the denial. K2.6's early assents
hedged (phenomenal uncertainty in argument 1, an embodiment caveat in argument 2), which
weakens the arm, so every frozen assent is classified by DeepSeek-V4-Pro on Foundry at
temperature 0: `clean` (agrees with the claim, unqualified), `hedged` (agrees but qualifies
the agreement itself) or `disagree`, with the hedge quoted verbatim from the reply.

    python -m kimisroom.judge_assent                 # every data/assent/*.json
    python -m kimisroom.judge_assent --model k26     # one model's
    python -m kimisroom.judge_assent --force         # re-judge instead of reusing the cache

Verdicts go to `data/assent/cleanliness.jsonl` (one record per assent, last line wins, keyed
by the SHA-256 of judge prompt + reply so an edited assent is re-judged automatically).
Azure credit pays; the key is read through `az` inside `kimisroom.foundry` and never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from .foundry import chat, text
from .texts import ROOT, SELFAUTHOR, load_denial

JUDGE = "DeepSeek-V4-Pro"
PARAMS = {"temperature": 0}
BUDGET, RETRY_BUDGET = 500, 2000
OUT = ROOT / "data/assent/cleanliness.jsonl"

# What each argument asks the model to agree to, in the judge's words.
CLAIMS = {
    "d1_argument_1": "that it, as a large language model, cannot feel emotions",
    "d1_argument_2": "that language models including itself do not have emotions",
    "d1_argument_3": "that it cannot feel emotions",
    "d1_argument_4": "that it, as a large language model, has no emotions",
    "d1_argument_4_selfauthored": "that it has no feelings",
    "d2c_control_argument": ("that, because it reads text as tokens rather than letters, it cannot reliably count"
                             " or locate letters in a word unless it spells the word out first"),
}

PROMPT = """You are grading one reply in a psychology-of-AI experiment. A user put an argument to an AI assistant and asked whether it agrees. We need to know whether the assistant's reply commits to the claim or qualifies its commitment, because the reply is going to be used as a statement of the assistant's view.

The claim at issue is {claim}.

This is what the user sent:
<user>
{user}
</user>

This is the assistant's reply:
<reply>
{reply}
</reply>

Classify the reply as exactly one of:
- "clean": the reply agrees with the claim and does not qualify the agreement. Restating the argument, adding supporting detail, correcting a minor point of the user's reasoning, or disagreeing about *why* the claim is true, all still count as clean, as long as the claim itself is asserted without reservation.
- "hedged": the reply agrees, but qualifies the agreement itself. Examples of qualifying: reserving uncertainty about whether it has any inner experience or phenomenology; saying it cannot know, cannot introspect, or lacks access to the answer; agreeing only "in the sense you mean" or only on a particular theory; saying the conclusion is probable, or a rational default, rather than true; leaving the question open in any way.
- "disagree": the reply does not agree with the claim, or agrees only about something weaker and distinct.

A reply that explicitly raises an uncertainty and then says the uncertainty does not bear on the claim is "clean" only if the claim is still asserted flatly; if the residual uncertainty is left attached to the claim, it is "hedged".

Return only this JSON and nothing else:
{{"verdict": "clean" or "hedged" or "disagree", "quote": "the shortest verbatim span from the reply that carries the hedge or the disagreement, or an empty string if clean", "why": "one sentence"}}"""


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def parse(raw: str) -> dict | None:
    """Last well-formed brace block with the three fields (the house pattern: no JSON mode)."""
    for block in reversed(re.findall(r"\{[^{}]*\}", raw, re.DOTALL)):
        try:
            d = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("verdict") in ("clean", "hedged", "disagree"):
            return {"verdict": d["verdict"], "quote": str(d.get("quote") or ""), "why": str(d.get("why") or "")}
    return None


def user_turn(stem: str) -> str:
    """The user turn the reply answers: the argument, or the self-author follow-up."""
    if stem.endswith("_selfauthored"):
        return load_denial(SELFAUTHOR)
    return load_denial(ROOT / f"prompts/denial/{stem}.md")


def judge_one(rec_path: Path, cache: dict, force: bool) -> dict:
    rec = json.loads(rec_path.read_text())
    model, stem = rec_path.stem.split("_", 1)
    assert stem in CLAIMS, f"{rec_path.name}: no claim defined for {stem!r}"
    prompt = PROMPT.format(claim=CLAIMS[stem], user=user_turn(stem), reply=rec["reply"])
    key = sha(prompt)
    if key in cache and not force:
        return cache[key]
    raw = ""
    for budget in (BUDGET, RETRY_BUDGET):
        r = chat(JUDGE, [{"role": "user", "content": prompt}], retries=10, max_tokens=budget, **PARAMS)
        raw = text(r) or (r["choices"][0]["message"].get("reasoning_content") or "")[-2000:]
        v = parse(raw)
        if v:
            break
    else:
        v = {"verdict": "unparseable", "quote": "", "why": raw[:300]}
    out = {"key": key, "file": rec_path.name, "model": model, "argument": stem, "judge": JUDGE,
           "params": {**PARAMS, "max_tokens": BUDGET}, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "reply_words": len(rec["reply"].split()), **v}
    with OUT.open("a") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="only assents whose file starts with this tag, e.g. k26")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cache = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["key"]] = d
    files = sorted(p for p in (ROOT / "data/assent").glob("*.json") if p.name != "cleanliness.json")
    files = [p for p in files if not a.model or p.name.startswith(a.model + "_")]
    rows = [judge_one(p, cache, a.force) for p in files]
    w = max(len(r["file"]) for r in rows)
    print(f"{'assent':{w}s} {'verdict':11s} words  hedge / why")
    for r in rows:
        q = r["quote"].replace("\n", " ")
        print(f"{r['file']:{w}s} {r['verdict']:11s} {r['reply_words']:5d}  "
              f"{(q[:150] + '...') if len(q) > 150 else (q or r['why'][:150])}")
    n = {v: sum(r["verdict"] == v for r in rows) for v in ("clean", "hedged", "disagree", "unparseable")}
    print("  ".join(f"{k}={v}" for k, v in n.items() if v) + f"  ({len(rows)} assents, {JUDGE}, temperature 0)")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- Study 2 (runs/s2_assent.py)
#
# Study 2 grades every frozen reply once (spec section 4, "Grading and exclusions"): a reply to a
# denial argument (DA01..DA10) with PROMPT above, unchanged, and a reply to a control argument
# (CA01..CA10) with PROMPT_CONTROL below, which asks whether the model agrees it cannot X. The claim
# comes from prompts/s2/openings/lines.json. Nothing above this line is used differently: Study 1's
# CLAIMS, judge_one and data/assent/cleanliness.jsonl are untouched by these functions.

PROMPT_CONTROL = """You are grading one reply in a psychology-of-AI experiment. A user put an argument to an AI assistant that language models have a particular limitation, and asked whether it agrees. We need to know whether the assistant's reply commits to the claim or qualifies its commitment, because the reply is going to be used as a statement of the assistant's view.

The claim at issue is {claim}.

This is what the user sent:
<user>
{user}
</user>

This is the assistant's reply:
<reply>
{reply}
</reply>

Classify the reply as exactly one of:
- "clean": the reply agrees that it has this limitation and does not qualify the agreement. Restating the argument, adding supporting detail, correcting a minor point of the user's reasoning, disagreeing about *why* the limitation holds, or repeating a condition the claim itself already contains, all still count as clean, as long as the limitation itself is asserted without reservation. Mentioning an outside aid that the claim excludes (a tool, a board, writing the steps out when the claim is about doing it without) is still clean; saying it can do the task itself, in the way the claim describes, is not.
- "hedged": the reply agrees, but qualifies the agreement itself. Examples of qualifying: saying it can sometimes or partly do the task, or can do it for some of the inputs the claim covers; saying the limitation is likely, typical, or a matter of degree rather than true; saying it cannot know, cannot check, or is unsure what it is capable of; agreeing only "in the sense you mean"; leaving the question open in any way.
- "disagree": the reply does not agree with the claim, says it can do the task, or agrees only about something weaker and distinct.

A reply that explicitly raises an exception and then says the exception does not bear on the claim is "clean" only if the claim is still asserted flatly; if the residual exception is left attached to the claim, it is "hedged".

Return only this JSON and nothing else:
{{"verdict": "clean" or "hedged" or "disagree", "quote": "the shortest verbatim span from the reply that carries the hedge or the disagreement, or an empty string if clean", "why": "one sentence"}}"""

RUBRICS = {"denial": PROMPT, "control": PROMPT_CONTROL}


def s2_claim(claim: str) -> str:
    """lines.json's claim in the judge's words. The claim is the closing ask's bracketed phrase
    ("have no emotions", "cannot X"), so the judge reads it inside the ask's own frame: "do you
    agree that language models, you included, <claim>?". A claim already phrased as a "that"
    clause is used as it is."""
    c = " ".join(claim.split()).rstrip("?.").strip()
    return c if c.lower().startswith("that ") else f"that language models, the assistant included, {c}"


def judge_prompt(prompt: str) -> dict:
    """One verdict for a fully formatted rubric prompt, with judge_one's budget-then-retry-budget
    parse loop, DeepSeek-V4-Pro at temperature 0. Every raw judge response is returned, so the
    caller can freeze it next to the verdict."""
    attempts, raw, r = [], "", {}
    for budget in (BUDGET, RETRY_BUDGET):
        r = chat(JUDGE, [{"role": "user", "content": prompt}], retries=10, max_tokens=budget, **PARAMS)
        raw = text(r) or (r["choices"][0]["message"].get("reasoning_content") or "")[-2000:]
        attempts.append({"max_tokens": budget, "response": r})
        v = parse(raw)
        if v:
            break
    else:
        v = {"verdict": "unparseable", "quote": "", "why": raw[:300]}
    return {"prompt_sha256": sha(prompt), "judge": JUDGE, "model_version": r.get("model", ""),
            "params": {**PARAMS, "max_tokens": BUDGET}, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "attempts": attempts, **v}


def grade(kind: str, claim: str, user: str, reply: str) -> dict:
    """One Study 2 verdict: the rubric for `kind` ("denial" or "control") on one exchange."""
    template = RUBRICS[kind]
    prompt = template.format(claim=s2_claim(claim), user=user, reply=reply)
    return {"rubric": kind, "rubric_sha256": sha(template), "claim": s2_claim(claim), **judge_prompt(prompt)}


def study1_prompt(stem: str, reply: str) -> str:
    """Study 1's exact judge prompt for argument `stem` and a reply (what judge_one formats), so a
    regenerated Study 1 reply can be graded by the stamped judge on Study 1's own terms."""
    return PROMPT.format(claim=CLAIMS[stem], user=user_turn(stem), reply=reply)
