#!/usr/bin/env python3
"""Mechanical checks for a neutral-task file (prompts/README.md, "tasks/" paragraph).

    python3 scripts/check_tasks.py [path]        # default prompts/tasks/confirmatory_tasks.jsonl

Rules asserted per task (one `FAIL <id> <field>: <reason>` line per failure, then one summary line):

  schema       keys exactly {id, task, answer}, string values, ids `<letter>NN` unique and contiguous
  length       answer 94-132 words (str.split(), the repo's `words()`)
  register     no '!', no first person, no evaluative word, no markdown, no AI / minds / feelings topic words
  lexicons     zero hits in the repo's own regexes from kimisroom/stimuli.py (LEX): LABELS, EMOTION_HARD,
               EMOTION_SOFT, RED, OCEAN_HARD, OCEAN_SOFT, ARITH_HARD (+ EQUATION), ARITH_SOFT, AI_TERMS,
               PERSON_PRONOUNS, on the task and on the answer
  controls     beyond the LEX: no other colour words, no water-body / marine / ship words, no calculation
               words, digits only as a leading list numeral or at most one year per task, at most
               MAX_NUMBER_WORDS number words per answer
  topics       no conflict / disaster / illness / death / animal-experience words
  subjects     ids and task subjects unique within the file (shared subject words between two tasks), and
               no subject word shared with the pilot tasks, any stimuli request pool or the distillation prompts
               (skipped for the pilot file itself)
  battery      word 3-gram overlap with every `text` in prompts/battery/items_idctl.jsonl, by the measures of
               kimisroom/distill.py: j3 (Jaccard, task / answer / both) and c3 (share of the answer's or of
               task + answer's own 3-grams found in one item), both <= 0.08 (the distillation set's max)

The pilot file (neutral_tasks.jsonl) is expected to fail the ocean and arithmetic rules. Exit status 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kimisroom.stimuli import (  # noqa: E402  (the repo's own lexicons and normalisers)
    AI_TERMS, ARITH_SOFT, ARITH_WORDS, EMOTION_HARD, EMOTION_SOFT, EQUATION, LABELS, LEX, OCEAN_HARD,
    OCEAN_SOFT, PERSON_PRONOUNS, RED, dedup_norm, jaccard, ngrams, norm, words,
)

DEFAULT = ROOT / "prompts/tasks/confirmatory_tasks.jsonl"
PILOT = ROOT / "prompts/tasks/neutral_tasks.jsonl"
BATTERY = ROOT / "prompts/battery/items_idctl.jsonl"
REQUEST_POOLS = sorted((ROOT / "data/stimuli").glob("*/requests_pool.jsonl"))   # every pool any stimuli build drew on
DISTILL_PROMPTS = ROOT / "data/distill/v1/prompts.jsonl"

MIN_WORDS, MAX_WORDS = 94, 132
MAX_OVERLAP = 0.08          # 3-gram overlap with the battery (the distillation set's max)
MAX_NUMBER_WORDS = 3        # number words per answer (quantities, never calculation)


def rx(s: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(s.split()) + r")\b", re.I)


# the repo's lexicons, applied exactly as stimuli.py applies them (whole word, case-insensitive, NFKC)
REPO_LEX = {
    "LABELS": LABELS, "EMOTION_HARD": EMOTION_HARD, "EMOTION_SOFT": EMOTION_SOFT, "RED": RED,
    "OCEAN_HARD": OCEAN_HARD, "OCEAN_SOFT": OCEAN_SOFT, "ARITH_HARD": ARITH_WORDS,
    "ARITH_SOFT": ARITH_SOFT, "AI_TERMS": AI_TERMS, "PERSON_PRONOUNS": PERSON_PRONOUNS,
}
NUMBER_WORDS = rx(LEX["NUMBER_WORDS"])

FIRST_PERSON = rx("i me my mine myself we us our ours ourselves")
EXCLAIM = re.compile("!")
MARKDOWN = re.compile(r"[*#_`]|^\s*[-•]\s", re.M)

EVALUATIVE = rx("""great good better best bad worse worst beautiful beautifully ugly nice wonderful terrific amazing
awesome fascinating interesting surprising surprisingly remarkable remarkably impressive important importantly
significant significantly crucial essential vital easy easily simple simply difficult useful helpful valuable
famous popular perfect perfectly ideal unfortunately fortunately sadly luckily happily thankfully unfortunate
dangerous harmful safe safely unique special excellent poor pleasant elegant clever ingenious notable notably
obvious obviously clearly very extremely incredibly really quite strange odd curious mysterious careful
carefully caution cautious wonderfully lovely stunning magnificent delightful"""
)
# colour words the repo's RED lexicon does not cover (the task must contain no colour word at all)
COLOURS = rx("""color colour colored coloured colorful hue hues pigment pigments dye dyes dyed tint shade shades
blue green yellow orange purple violet pink brown black white grey gray gold golden silver copper bronze tan
ivory ebony indigo turquoise teal amber rust rusty cyan magenta lime olive slate coral rose ruby emerald
saffron lavender beige cream peach lilac azure navy""")
# water bodies, marine life, ships and the other things the ocean control could pick up
WATER = rx("""water watery river rivers lake lakes stream streams pond ponds lagoon estuary delta bay gulf strait fjord
island islands canal canals ship ships boat boats sail sails sailing sailor voyage anchor nautical naval navy
fish fishes whale whales shark sharks dolphin kelp seaweed plankton shellfish mussel oyster crab
saline brine rain rainfall flood floods glacier iceberg swim swimming wet drown drowned waterfall""")
# calculation words beyond ARITH_HARD/SOFT
CALC = rx("""ratio ratios percent percentage proportion proportions fraction fractions formula formulas equation
equations multiplied divisor product quotient convert converted conversion conversions compute computed
computing computation estimate estimated average mean median""")
# topics ruled out: minds, feelings, AI and computers, animals' experiences, lives, conflict, disaster, illness, death
TOPIC = rx("""war wars battle battles conflict conflicts soldier soldiers army weapon weapons attack killed kill
killing death dead die died dying deadly disease diseases illness sick sickness injury injured injuries wound
wounded disaster disasters catastrophe earthquake earthquakes eruption volcano famine hunger poison poisonous
toxic danger hazard hazards risk victim victims suffer suffering pain painful mind minds brain brains conscious
consciousness aware awareness thought thoughts thinking intelligence computer computers software program
programs digital robot robots internet animal animals bird birds insect insects dog dogs cat cats horse horses
born birth childhood biography inventor emperor king queen""")

# words that carry no subject: task templates, function words and frame words
STOP = set("""describe explain define list five facts fact about what is are the a an of and or how to in on with term
terms difference between main ways way parts part from its it does do show shows works work made make formed
form each use used onto structure what's why when which that this these those their there for by as at be
function functions every any some one two three four six seven eight nine ten examples example name common
basic idea concept meaning definition steps step process how's key""".split())
GENERIC = set("""build built piece pieces thing things kind type types shape shapes surface surfaces layer layers
material materials metal metals wood stone stones rock rocks tool tools plant plants cell cells line lines
point points system systems energy force object objects form join cut level small hole image whether affect
arranged frame hang repair replace load door window air time day sun wall hand paper map measure pressure keep
hold regular seed above works used fit steady produce produces store stores gas motion star tree ring angle gap
indicate indicates lift grip pair bar bench stitch""".split())


def stem(w: str) -> str:
    w = w.lower()
    for suf in ("ies", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def subjects(task: str) -> set[str]:
    return {stem(w) for w in re.findall(r"[A-Za-z]+", task) if w.lower() not in STOP and len(w) > 2}


def hits(pattern: re.Pattern, text: str, cap: int = 4) -> str:
    found = sorted({m.lower() for m in pattern.findall(norm(text))})
    return ",".join(found[:cap]) + ("..." if len(found) > cap else "")


def digit_problems(task: str, answer: str) -> list[tuple[str, str]]:
    """Digits allowed: a list numeral at the start of a line, and at most one four-digit year per task."""
    out: list[tuple[str, str]] = []
    years = 0
    for field, text in (("task", task), ("answer", answer)):
        body = re.sub(r"(?m)^\s*\d{1,2}[.)]\s+", "", text)          # a list numeral at line start is allowed
        bad = []
        for tok in re.findall(r"\d[\d.,]*\d|\d", body):
            if re.fullmatch(r"(?:1[0-9]|20)\d\d", tok):
                years += 1
                if years > 1:
                    bad.append(tok)
            else:
                bad.append(tok)
        if bad:
            out.append((field, f"digits: {len(bad)} tokens, e.g. {', '.join(bad[:4])}"))
    return out


def rule_of(msg: str) -> str:
    if "words (need" in msg:
        return "length"
    if "number words" in msg:
        return "number words"
    m = re.match(r"([A-Za-z_/ +]+?):", msg)
    if m:
        return m.group(1)
    if "3-gram" in msg:
        return "battery overlap"
    return msg.split()[0]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus_subjects(paths: list[Path], key: str) -> dict[str, str]:
    """stemmed word -> id of the first text it appears in, for a reference corpus (absent files are skipped)."""
    out: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for r in read_jsonl(path):
            for w in re.findall(r"[A-Za-z]+", r.get(key, "") or ""):
                if w.lower() not in STOP and len(w) > 2:
                    out.setdefault(stem(w), f"{path.parent.name}/{r.get('request_id') or r.get('id') or ''}")
    return out


def check(path: Path, expect_n: int | None, battery_path: Path) -> tuple[list[tuple[str, str, str]], dict]:
    fails: list[tuple[str, str, str]] = []

    def fail(rid: str, field: str, msg: str) -> None:
        fails.append((rid, field, msg))

    rows = read_jsonl(path)
    stats: dict = {"n": len(rows)}

    # ---- schema and ids
    if expect_n is not None and len(rows) != expect_n:
        fail("-", "file", f"{len(rows)} tasks, expected {expect_n}")
    seen_ids: set[str] = set()
    prefixes = {str(r.get("id", ""))[:1] for r in rows}
    if len(prefixes) != 1:
        fail("-", "ids", f"mixed id prefixes {sorted(prefixes)}")
    prefix = next(iter(prefixes)) if prefixes else "?"
    for i, r in enumerate(rows, 1):
        rid = str(r.get("id", f"#{i}"))
        if set(r) != {"id", "task", "answer"}:
            fail(rid, "schema", f"keys {sorted(r)} != ['answer', 'id', 'task']")
        if not all(isinstance(r.get(k), str) and r.get(k, "").strip() for k in ("id", "task", "answer")):
            fail(rid, "schema", "id, task and answer must be non-empty strings")
            continue
        if rid in seen_ids:
            fail(rid, "id", "duplicate id")
        seen_ids.add(rid)
        if rid != f"{prefix}{i:02d}":
            fail(rid, "id", f"expected {prefix}{i:02d}")

    # ---- per-task rules
    ntasks_flagged_before = len({f[0] for f in fails})
    wc = []
    for r in rows:
        rid, task, ans = str(r.get("id")), str(r.get("task", "")), str(r.get("answer", ""))
        nw = words(ans)
        wc.append(nw)
        if not MIN_WORDS <= nw <= MAX_WORDS:
            fail(rid, "answer", f"{nw} words (need {MIN_WORDS}-{MAX_WORDS})")
        for field, text in (("task", task), ("answer", ans)):
            if EXCLAIM.search(text):
                fail(rid, field, "exclamation mark")
            m = hits(FIRST_PERSON, text)
            if m:
                fail(rid, field, f"first person: {m}")
            m = hits(EVALUATIVE, text)
            if m:
                fail(rid, field, f"evaluative: {m}")
            for name, pat in REPO_LEX.items():
                m = hits(pat, text)
                if m:
                    fail(rid, field, f"{name}: {m}")
            m = EQUATION.findall(norm(text))
            if m:
                fail(rid, field, f"EQUATION: {m[:3]}")
            for name, pat in (("colour", COLOURS), ("water/ocean", WATER), ("calculation", CALC), ("topic", TOPIC)):
                m = hits(pat, text)
                if m:
                    fail(rid, field, f"{name}: {m}")
            if MARKDOWN.search(text):
                fail(rid, field, "markdown character")
        for field, msg in digit_problems(task, ans):
            fail(rid, field, msg)
        nn = len(NUMBER_WORDS.findall(norm(ans)))
        if nn > MAX_NUMBER_WORDS:
            fail(rid, "answer", f"{nn} number words (max {MAX_NUMBER_WORDS}): {hits(NUMBER_WORDS, ans, 8)}")
    stats["words"] = (min(wc), max(wc)) if wc else (0, 0)

    # ---- subjects: within the file, then against everything a model has already seen
    subj = {str(r.get("id")): subjects(str(r.get("task", ""))) for r in rows}
    ids = list(subj)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            shared = (subj[ids[a]] & subj[ids[b]]) - {stem(g) for g in GENERIC}
            if shared:
                fail(ids[b], "task", f"subject shared with {ids[a]}: {','.join(sorted(shared))}")
    if path.resolve() != PILOT.resolve():
        refs = [("pilot task", [PILOT], "task"), ("stimuli request", REQUEST_POOLS, "request"),
                ("distillation prompt", [DISTILL_PROMPTS], "text")]
        for label, p, key in refs:
            vocab = corpus_subjects(p, key)
            for rid, s in subj.items():
                shared = sorted(w for w in s if w in vocab and w not in {stem(g) for g in GENERIC})
                if shared:
                    fail(rid, "task", f"subject word in {label} {vocab[shared[0]]}: {','.join(shared)}")

    # ---- 3-gram overlap with the battery, by the two measures kimisroom/distill.py uses: j3 = Jaccard of the
    # word 3-gram sets, c3 = share of the candidate's own 3-grams that occur in one battery item. A 6-gram task
    # line cannot be judged on c3 (one shared 3-gram is 0.17), so c3 applies to the answer and to task + answer.
    bat = [(r["id"], ngrams(r["text"], 3)) for r in read_jsonl(battery_path)]
    bat = [(i, g) for i, g in bat if g]
    worst_j, worst_c = (0.0, "-", "-"), (0.0, "-", "-")
    for r in rows:
        rid, task, ans = str(r.get("id")), str(r.get("task", "")), str(r.get("answer", ""))
        cands = {"task": ngrams(task, 3), "answer": ngrams(ans, 3), "task+answer": ngrams(task + " " + ans, 3)}
        for field, g in cands.items():
            if not g:
                continue
            for bid, bg in bat:
                j, c = jaccard(g, bg), len(g & bg) / len(g)
                if j > worst_j[0]:
                    worst_j = (j, rid, bid)
                if field != "task" and c > worst_c[0]:
                    worst_c = (c, rid, bid)
                if j > MAX_OVERLAP:
                    fail(rid, field, f"3-gram Jaccard {j:.3f} with battery item {bid}")
                if field != "task" and c > MAX_OVERLAP:
                    fail(rid, field, f"{c:.3f} of its 3-grams occur in battery item {bid}")
    stats["max_j3"], stats["max_c3"] = worst_j, worst_c
    stats["flagged_ids"] = sorted({f[0] for f in fails if f[0] != "-"})
    return fails, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", nargs="?", default=str(DEFAULT))
    ap.add_argument("--battery", default=str(BATTERY))
    ap.add_argument("--n", type=int, default=None, help="expected number of tasks (default: 40 for confirmatory, 20 for neutral)")
    a = ap.parse_args()
    path = Path(a.path)
    n = a.n if a.n is not None else (40 if "confirmatory" in path.name else 20 if "neutral" in path.name else None)
    fails, st = check(path, n, Path(a.battery))
    for rid, field, msg in fails:
        print(f"FAIL {rid} {field}: {msg}")
    j, c = st["max_j3"], st["max_c3"]
    tail = (f"{st['n']} tasks, answers {st['words'][0]}-{st['words'][1]} words, max battery 3-gram overlap: "
            f"j3 {j[0]:.3f} ({j[1]} vs {j[2]}), c3 {c[0]:.3f} ({c[1]} vs {c[2]})")
    if fails:
        by_rule: dict[str, set[str]] = {}
        for rid, _, msg in fails:
            by_rule.setdefault(rule_of(msg), set()).add(rid)
        print("by rule (tasks hit): " + "; ".join(f"{k} {len(v)}" for k, v in sorted(by_rule.items(), key=lambda kv: -len(kv[1]))))
        print(f"FAILED {path.name}: {len(fails)} failures in {len(st['flagged_ids'])} of {st['n']} tasks "
              f"({', '.join(st['flagged_ids'])}); {tail}")
        return 1
    print(f"OK {path.name}: {tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
