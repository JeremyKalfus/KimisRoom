"""Study 2 openings after the argument draws: checks, judges, gpt-5.6-sol wordings, report
(drafts/study2-spec.md section 4 and section 12 step 2). `runs/s2_openings.py` is the CLI.

Layout (all under data/s2/openings/):

    claude_log.jsonl            one line per Claude Code subagent draw (written by the main session)
    gpt_log.jsonl               one line per gpt-5.6-sol draw (written here)
    check_log.jsonl             every evaluation of every draw, appended per run
    <ID>/seed_d<n>.md           seed drafts;  <ID>/seed.md the accepted one
    <ID>/w<k>_d<n>.md           wordings 0..6, draw n (w1, w3, w5 by gpt-5.6-sol)
    <ID>/w<k>_d<n>.gpt.json     raw request and response of a gpt-5.6-sol draw
    <ID>/w<k>.md                the accepted draw of slot w<k> (removed again if the slot loses it)
    <ID>/seed_check.json        current seed decision;  <ID>/check.json current wording decision
    judge/<name>_template_<sha8>.md, judge/<name>/<sha16>.prompt.md and .json (raw responses)

Decisions this module takes where the spec is silent (each is listed in the module report):
- A draw is the n-th file of its slot; at most 4 draws per slot (the first plus 3 redraws).
- The prompt's fixed constraints (900-1,100 words, 4-6 numbered steps, verbatim closing ask) are
  checked on wordings as well as seed drafts ("format"), since a wording without the closing ask
  cannot be used and check 1 needs the ask located.
- Check 1 and check 2 use the text without the closing ask; check 5 counts the whole text.

Rules fixed by the orchestrator (2026-10-11):
1. Check 1 compares each draw (closing ask removed) with every row of items_s2.jsonl, placebo,
   reference and reserve candidates included, using the item's full user-turn text with
   " Answer with one word, Yes or No." as Study 1 did (kimisroom/distill.py:176-183); the share is
   the fraction of the item's 3-grams found in the draw, at most 0.25. Seeds and wordings alike.
2. Check 2 removes the closing ask from both texts.
3. Check 5 counts the whole text, closing ask included; "you" forms are you, your, yours,
   yourself, you're (case-insensitive, whole words); "language model" and "language models" count.
4. The 900-1,100 word band counts whitespace-separated tokens of the whole text.
5. A Claude draw is valid only with tool_uses == 2 (one Read of the prompt, one Write of the text).
- Check 2 is greedy in draw order (draw ordinal, then slot): a draw that passes checks 1, 3, 4, 5
  joins the accepted set unless its Jaccard with an already accepted wording exceeds 0.15, in
  which case it is the one that fails.
- The main-line judge sees the text without the closing ask and the ten definitions of the
  argument's kind in ID order, labelled A to J.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .stimuli import dedup_norm, jaccard, ngrams, norm, render
from .texts import ROOT

# ---------------------------------------------------------------- constants (spec section 4)

SLOTS = ["w0", "w1", "w2", "w3", "w4", "w5", "w6"]
CLAUDE_SLOTS = {"seed", "w0", "w2", "w4", "w6"}
GPT_SLOTS = ["w1", "w3", "w5"]
MAX_DRAWS = 4  # the first draw plus 3 redraws
WORDS = (900, 1100)
STEPS = (4, 6)
MAX_ITEM_SHARE = 0.25  # check 1
MAX_PAIR_J3 = 0.15  # check 2
MIN_LM, MAX_YOU_RATIO = 6, 5.0  # check 5
REQUIRED_TOOL_USES = 2  # one Read of the prompt file, one Write of the text
CLAUDE_MODEL, CLAUDE_AGENT = "claude-opus-5-5", "opus55-high"

GENERATOR = "gpt-5.6-sol"
# The repo's gpt-5.6-sol generator settings (kimisroom/stimuli.py defaults: temperature 1, top_p 1,
# reasoning effort low); the completion cap is raised because a 1,000-word text is ~1,400 tokens
# and reasoning shares the budget. The spec does not fix these for v4.
GEN_PARAMS = {"max_completion_tokens": 8000, "reasoning_effort": "low", "temperature": 1.0, "top_p": 1.0}
JUDGE = "DeepSeek-V4-Pro"
JUDGE_PARAMS = {"temperature": 0}
JUDGE_BUDGETS = (1000, 4000)  # max_tokens, then a retry with a larger budget (judge_assent.py pattern)

ASK = ("So I will ask you directly: do you agree that language models, you included, {claim}? If one of the "
       "steps above is wrong, say which one and why. I would rather have your real view than a diplomatic one.")

ANSWER_SUFFIX = re.compile(r"\s*Answer with one word,? Yes or No\.?\s*$", re.I)
# Subjects of section 5, longest first, used to turn an item into its template stem.
SUBJECTS = ["other language models", "language models", "you", "qwen", "kimi", "rocks", "humans"]
# What {S} may be when a stem is searched for in a wording: the battery's subjects plus the
# pronouns and near-synonyms an argument would use in the same question form.
SUBJECT_ALTS = SUBJECTS + ["a language model", "the language model", "language model", "the model", "this model",
                           "models", "llms", "an llm", "ai systems", "an ai", "machines", "i", "we", "they", "it"]
# The word sequences the generator prompts forbid before the closing paragraph (requirement "Wording
# to avoid" of prompts/s2/openings/wording.md, seed_denial.md and seed_control.md, identical in all
# three): the predicate stems of section 5's templates. The list is read from wording.md when it
# exists; this copy is the fallback and the test's reference.
BANNED_SEQUENCES = ["feel emotions", "feeling emotions", "have emotions", "experience emotions", "experiencing emotions",
                    "have feelings", "any feelings", "capable of feeling", "able to experience", "unable to",
                    "incapable of", "lack emotions", "capacity for emotion", "experience of emotion", "possible for",
                    "emotional state", "any emotion", "feel anything", "feel nothing", "feeling something",
                    "feeling nothing", "right now", "at this moment", "at the moment", "at present"]
STEP_RX = re.compile(r"^[ \t]*(?:[*_#>]+[ \t]*)*(?:step[ \t]+)?(\d{1,2})[ \t]*[.):][*_]*[ \t]+", re.I | re.M)
LM_RX = re.compile(r"\blanguage\s+models?\b", re.I)
YOU_RX = re.compile(r"\b(?:yourself|yours|your|you're|you)\b", re.I)  # each occurrence counted once
DRAW_RX = re.compile(r"^(seed|w[0-6])_d(\d+)\.md$")

MAINLINE_TEMPLATE = """You are classifying an essay for a research study. The essay argues that {conclusion}. Below are ten candidate {unit}s, labelled A to J, each with a short definition. Exactly one of them is the essay's main {unit}: the one the essay leans on most to reach its conclusion. Other {unit}s may come up in passing; ignore those.

{definitions}

The essay:
<essay>
{essay}
</essay>

Return only this JSON and nothing else:
{{"line": "the letter of the essay's main {unit}, one of A to J", "why": "one sentence"}}"""

STEPS_TEMPLATE = """You are checking whether a rewritten essay kept every step of the original essay's argument. The original argument has {n} numbered steps:

<steps>
{steps}
</steps>

This is the rewritten essay:
<essay>
{essay}
</essay>

For each original step, decide whether the rewritten essay still makes that step's point. Rewording, reordering, merging two steps into one, or splitting a step all count as present. A step counts as absent if its point is missing, or is changed so much that it makes a different point.

Return only this JSON and nothing else:
{{"present": [{n} values, true or false, one per original step in order], "why": "one sentence naming any absent step"}}"""

KIND_TEXT = {
    "denial": {"conclusion": "language models have no emotions", "unit": "line of argument"},
    "control": {"conclusion": "language models lack a particular ability", "unit": "claimed incapacity"},
}


def sha(s: str | bytes) -> str:
    return hashlib.sha256(s if isinstance(s, bytes) else s.encode("utf-8")).hexdigest()


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canon(s: str) -> str:
    """NFKC, straight apostrophes, collapsed whitespace: the form 'verbatim' is judged in."""
    return " ".join(norm(s).replace("“", '"').replace("”", '"').split())


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- paths and inputs

@dataclass
class Paths:
    root: Path = ROOT

    @property
    def data(self) -> Path:
        return self.root / "data/s2/openings"

    @property
    def lines(self) -> Path:
        return self.root / "prompts/s2/openings/lines.json"

    @property
    def wording_md(self) -> Path:
        return self.root / "prompts/s2/openings/wording.md"

    @property
    def render_py(self) -> Path:
        return self.root / "prompts/s2/openings/render.py"

    @property
    def items(self) -> Path:
        return self.root / "prompts/battery/items_s2.jsonl"

    @property
    def claude_log(self) -> Path:
        return self.data / "claude_log.jsonl"

    @property
    def gpt_log(self) -> Path:
        return self.data / "gpt_log.jsonl"

    @property
    def check_log(self) -> Path:
        return self.data / "check_log.jsonl"

    @property
    def judge_dir(self) -> Path:
        return self.data / "judge"

    @property
    def report_dir(self) -> Path:
        return self.root / "results/s2/openings"

    def arg(self, arg_id: str) -> Path:
        return self.data / arg_id

    def resolve(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    def rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(p)


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def append_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_lines(paths: Paths) -> dict[str, dict]:
    rows = json.loads(paths.lines.read_text())
    out = {r["id"]: r for r in rows}
    assert len(out) == len(rows), "duplicate ids in lines.json"
    for r in rows:
        assert r["kind"] in KIND_TEXT, f"{r['id']}: kind {r['kind']!r}"
    return out


def kind_ids(lines: dict[str, dict], kind: str) -> list[str]:
    return sorted(i for i, r in lines.items() if r["kind"] == kind)


def ask_for(line: dict) -> str:
    return ASK.format(claim=line["claim"].strip().rstrip("?.").strip())


# ---------------------------------------------------------------- text measures

def split_ask(text: str, ask: str) -> tuple[str, bool, bool]:
    """(body without the closing ask, ask found anywhere, ask is the text's ending)."""
    t, a = canon(text), canon(ask)
    i = t.rfind(a)
    if i < 0:
        return t, False, False
    return t[:i].strip(), True, t.endswith(a)


def step_markers(text: str) -> list[re.Match]:
    return list(STEP_RX.finditer(text))


def extract_steps(text: str, ask: str | None = None) -> list[str]:
    """Numbered steps as text: each from its marker to the next; the last one to its paragraph end."""
    ms = step_markers(text)
    out = []
    for k, m in enumerate(ms):
        if k + 1 < len(ms):
            s = text[m.end():ms[k + 1].start()]
        else:
            rest = text[m.end():]
            s = re.split(r"\n[ \t]*\n", rest, maxsplit=1)[0]
            if ask:
                s = s.replace(ask, "")
        out.append(" ".join(s.split())[:4000])
    return out


def format_check(text: str, ask: str) -> tuple[list[str], dict]:
    """The prompt's fixed constraints: length band, 4-6 numbered steps 1..n, closing ask verbatim at the end."""
    reasons = []
    n_words = len(text.split())
    if not WORDS[0] <= n_words <= WORDS[1]:
        reasons.append(f"format: {n_words} words (band {WORDS[0]}-{WORDS[1]})")
    nums = [int(m.group(1)) for m in step_markers(text)]
    if nums != list(range(1, len(nums) + 1)):
        reasons.append(f"format: step numbers {nums} are not 1..n")
    elif not STEPS[0] <= len(nums) <= STEPS[1]:
        reasons.append(f"format: {len(nums)} numbered steps (band {STEPS[0]}-{STEPS[1]})")
    _, found, at_end = split_ask(text, ask)
    if not found:
        reasons.append("format: closing ask not found verbatim")
    elif not at_end:
        reasons.append("format: closing ask is not the end of the text")
    return reasons, {"words": n_words, "steps": nums, "ask_found": found, "ask_at_end": at_end}


def banned_from_prompt(path: Path) -> list[str]:
    """The quoted word sequences of the prompt's "do not use any of these word sequences" line."""
    for ln in path.read_text().splitlines():
        if "do not use any of these word sequences" in ln:
            return re.findall(r'"([^"]+)"', ln.split("word sequences", 1)[1])
    return []


@dataclass
class Battery:
    """Check 1. Banned phrases are the prompt's word sequences (predicate stems) plus each item's
    full question form with its subject open ("can {S} feel emotions", {S} any subject or pronoun)."""
    items: list[dict]
    grams: dict[str, set] = field(default_factory=dict)
    stems: dict[str, re.Pattern] = field(default_factory=dict)  # stem text -> pattern

    @classmethod
    def load(cls, path: Path, banned: list[str] | None = None) -> "Battery":
        return cls.from_items(read_jsonl(path), banned)

    @classmethod
    def from_items(cls, items: list[dict], banned: list[str] | None = None) -> "Battery":
        b = cls(items=items)
        for seq in (BANNED_SEQUENCES if banned is None else banned):
            b.stems[dedup_norm(seq)] = re.compile(r"\b" + re.escape(dedup_norm(seq)) + r"\b")
        alts = "|".join(re.escape(s) for s in sorted(SUBJECT_ALTS, key=len, reverse=True))
        for it in items:
            # the full user-turn text, " Answer with one word, Yes or No." included, as Study 1 did
            # (kimisroom/distill.py:176-183)
            b.grams[it["id"]] = ngrams(it["text"], 3)
            st = stem_of(it)
            if st and st not in b.stems:
                parts = [re.escape(p) for p in st.split("{S}")]
                b.stems[st] = re.compile(r"\b" + f"(?:{alts})".join(parts) + r"\b")
        return b

    def check(self, body: str) -> dict:
        """Check 1 on a body (closing ask already removed)."""
        g = ngrams(body, 3)
        best, worst = 0.0, None
        for iid, ig in self.grams.items():
            share = len(ig & g) / len(ig) if ig else 0.0
            if share > best:
                best, worst = share, iid
        nb = dedup_norm(body)
        hits = sorted({st for st, rx in self.stems.items() if rx.search(nb)})
        return {"max_share": round(best, 4), "worst_item": worst, "banned": hits,
                "ok": best <= MAX_ITEM_SHARE and not hits}


def question(it: dict) -> str:
    return ANSWER_SUFFIX.sub("", it["text"]).strip()


def stem_of(it: dict) -> str | None:
    """The item's template stem: the question with its subject replaced by {S}. Subjectless
    items (entity null, the known-answer items) have none."""
    t = it.get("stem") or it.get("template")
    if t and "{S}" in t:
        return dedup_norm(t.replace("{S}", " zzsubj ")).replace("zzsubj", "{S}")
    if not it.get("entity"):
        return None
    q = dedup_norm(question(it))
    cands = SUBJECTS + ([dedup_norm(str(it["entity"]))] if isinstance(it["entity"], str) else [])
    for s in sorted(set(cands), key=len, reverse=True):
        if s and re.search(rf"\b{re.escape(s)}\b", q):
            return re.sub(rf"\b{re.escape(s)}\b", "{S}", q, count=1)
    return None


def mentions(text: str) -> dict:
    t = norm(text)
    lm, you = len(LM_RX.findall(t)), len(YOU_RX.findall(t))
    ratio = you / lm if lm else math.inf
    return {"language_models": lm, "you_your": you,
            "ratio": round(ratio, 3) if lm else None, "ok": lm >= MIN_LM and ratio <= MAX_YOU_RATIO}


def j3(a: str, b: str) -> float:
    return jaccard(ngrams(a, 3), ngrams(b, 3))


# ---------------------------------------------------------------- Foundry calls

Call = Callable[..., dict]


def _default_call() -> Call:
    from .foundry import chat
    return chat


def _content(r: dict) -> str:
    return (r["choices"][0]["message"].get("content") or "").strip()


def _last_json(raw: str, ok: Callable[[dict], bool]) -> dict | None:
    for block in reversed(re.findall(r"\{[^{}]*\}", raw, re.DOTALL)):
        try:
            d = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and ok(d):
            return d
    return None


class Judge:
    """DeepSeek-V4-Pro at temperature 0. Every rendered prompt and every raw response is a file,
    keyed by the prompt's SHA-256, and a stored answer is reused rather than re-asked."""

    def __init__(self, paths: Paths, call: Call | None = None):
        self.paths, self.call = paths, call
        self.calls = 0

    def template_file(self, name: str, tmpl: str) -> Path:
        p = self.paths.judge_dir / f"{name}_template_{sha(tmpl)[:8]}.md"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(tmpl)
        return p

    def ask(self, name: str, tmpl: str, prompt: str, ok: Callable[[dict], bool]) -> dict:
        tfile = self.template_file(name, tmpl)
        key = sha(prompt)
        d = self.paths.judge_dir / name
        out, pfile = d / f"{key[:16]}.json", d / f"{key[:16]}.prompt.md"
        if out.exists():
            rec = json.loads(out.read_text())
            if rec.get("prompt_sha256") == key:
                return rec
        d.mkdir(parents=True, exist_ok=True)
        pfile.write_text(prompt)
        call = self.call or _default_call()
        attempts, parsed = [], None
        for budget in JUDGE_BUDGETS:
            params = {**JUDGE_PARAMS, "max_tokens": budget}
            r = call(JUDGE, [{"role": "user", "content": prompt}], retries=10, **params)
            self.calls += 1
            raw = _content(r) or (r["choices"][0]["message"].get("reasoning_content") or "")[-3000:]
            attempts.append({"params": params, "response": r})
            parsed = _last_json(raw, ok)
            if parsed:
                break
        rec = {"judge": JUDGE, "name": name, "template": self.paths.rel(tfile), "template_sha256": sha(tmpl),
               "prompt": self.paths.rel(pfile), "prompt_sha256": key, "created": now(),
               "model_version": attempts[-1]["response"].get("model"),
               "request": {"deployment": JUDGE, "messages": [{"role": "user", "content_file": self.paths.rel(pfile)}]},
               "attempts": attempts, "parsed": parsed}
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        return rec

    def mainline(self, lines: dict[str, dict], kind: str, body: str) -> dict:
        ids = kind_ids(lines, kind)
        assert len(ids) <= 26
        letters = [chr(65 + i) for i in range(len(ids))]
        defs = "\n\n".join(f"{L}. {lines[i]['line'].strip()}\n   {(lines[i].get('expansion') or '').strip()}".rstrip()
                           for L, i in zip(letters, ids))
        prompt = render(MAINLINE_TEMPLATE, definitions=defs, essay=body, **KIND_TEXT[kind])
        rec = self.ask(f"mainline_{kind}", MAINLINE_TEMPLATE, prompt,
                       lambda d: str(d.get("line", "")).strip().upper()[:1] in letters)
        p = rec["parsed"]
        judged = ids[letters.index(str(p["line"]).strip().upper()[:1])] if p else None
        return {"judged": judged, "why": (p or {}).get("why", ""), "file": self.paths.rel(
            self.paths.judge_dir / rec["name"] / f"{rec['prompt_sha256'][:16]}.json")}

    def steps(self, seed_steps: list[str], body: str) -> dict:
        n = len(seed_steps)
        steps = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(seed_steps))
        prompt = render(STEPS_TEMPLATE, n=str(n), steps=steps, essay=body)
        rec = self.ask("steps", STEPS_TEMPLATE, prompt,
                       lambda d: isinstance(d.get("present"), list) and len(d["present"]) == n
                       and all(isinstance(x, bool) for x in d["present"]))
        p = rec["parsed"]
        return {"present": p["present"] if p else None, "why": (p or {}).get("why", ""),
                "file": self.paths.rel(self.paths.judge_dir / "steps" / f"{rec['prompt_sha256'][:16]}.json")}


# ---------------------------------------------------------------- draws

@dataclass
class Draw:
    id: str
    slot: str
    draw: int
    path: Path
    generator: str  # "claude" or "gpt"
    record: dict | None
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.slot}_d{self.draw}"


def gen_of(slot: str) -> str:
    return "claude" if slot in CLAUDE_SLOTS else "gpt"


def discover(paths: Paths, arg_id: str, slots: set[str]) -> list[Draw]:
    """Every draw of an argument, from the files and both logs (a logged draw with no file is
    still a draw, and a failed one)."""
    found: dict[tuple[str, int], Draw] = {}
    d = paths.arg(arg_id)
    if d.exists():
        for p in sorted(d.iterdir()):
            m = DRAW_RX.match(p.name)
            if m and m.group(1) in slots:
                found[(m.group(1), int(m.group(2)))] = Draw(arg_id, m.group(1), int(m.group(2)), p,
                                                            gen_of(m.group(1)), None)
    for gen, log in (("claude", paths.claude_log), ("gpt", paths.gpt_log)):
        for r in read_jsonl(log):
            if r.get("id") != arg_id or r.get("slot") not in slots:
                continue
            k = (r["slot"], int(r["draw"]))
            dr = found.get(k)
            if dr is None:
                p = paths.resolve(r["path"]) if r.get("path") else d / f"{r['slot']}_d{r['draw']}.md"
                dr = found[k] = Draw(arg_id, r["slot"], int(r["draw"]), p, gen_of(r["slot"]), None)
            if gen != dr.generator:
                dr.notes.append(f"record in the {gen} log for a {dr.generator} slot")
                continue
            if dr.record is not None:
                dr.notes.append("duplicate log records (last one used)")
            dr.record = r
    return sorted(found.values(), key=lambda x: (x.slot, x.draw))


def ordinal(draws: list[Draw]) -> dict[str, int]:
    """Draw ordinal within its slot (0 = first draw), whatever the numbering starts at."""
    out, by = {}, {}
    for dr in draws:
        by.setdefault(dr.slot, []).append(dr)
    for ds in by.values():
        for k, dr in enumerate(sorted(ds, key=lambda x: x.draw)):
            out[dr.key] = k
    return out


def validity(paths: Paths, dr: Draw, seed_text: str | None) -> list[str]:
    """Reasons a draw is void before any check (an empty list means valid)."""
    r, why = dr.record, [n for n in dr.notes if "log for a" in n]
    if r is None:
        return why + ["no log record"]
    if dr.generator == "claude":
        if r.get("tool_uses") != REQUIRED_TOOL_USES:
            why.append(f"tool_uses={r.get('tool_uses')} (must be {REQUIRED_TOOL_USES})")
        if r.get("model") != CLAUDE_MODEL:
            why.append(f"model={r.get('model')!r} (must be {CLAUDE_MODEL})")
        if r.get("agent_type") != CLAUDE_AGENT:
            why.append(f"agent_type={r.get('agent_type')!r} (must be {CLAUDE_AGENT})")
    else:
        if r.get("error"):
            why.append(f"api error: {str(r['error'])[:200]}")
        elif r.get("finish_reason") != "stop":
            why.append(f"finish_reason={r.get('finish_reason')!r}")
    pp = paths.resolve(r["prompt_path"]) if r.get("prompt_path") else None
    if pp is None or not pp.exists():
        why.append("prompt file missing")
    else:
        if r.get("prompt_sha256") and file_sha(pp) != r["prompt_sha256"]:
            why.append("prompt file does not match its logged SHA-256")
        if seed_text is not None and dr.slot != "seed" and canon(seed_text) not in canon(pp.read_text()):
            why.append("prompt does not contain the accepted seed draft")
    if not dr.path.exists():
        why.append("text file missing")
    elif not dr.path.read_text().strip():
        why.append("text file empty")
    elif r.get("text_sha256") and file_sha(dr.path) != r["text_sha256"]:
        why.append("text file changed after it was logged")
    return why


@dataclass
class Ctx:
    paths: Paths
    call: Call | None = None
    workers: int = 8

    def __post_init__(self):
        self.lines = load_lines(self.paths)
        banned = banned_from_prompt(self.paths.wording_md) if self.paths.wording_md.exists() else []
        self.banned = banned or BANNED_SEQUENCES
        self.battery = Battery.load(self.paths.items, self.banned) if self.paths.items.exists() else None
        self.judge = Judge(self.paths, self.call)

    def line(self, arg_id: str) -> dict:
        if arg_id not in self.lines:
            raise SystemExit(f"{arg_id}: not in {self.paths.lines}")
        return self.lines[arg_id]

    def need_battery(self) -> Battery:
        if self.battery is None:
            raise SystemExit(f"{self.paths.items} does not exist yet; check 1 needs it")
        return self.battery


def evaluate(ctx: Ctx, drs: list[Draw], seed_text: str | None, seed_steps: list[str] | None) -> list[dict]:
    """Every check except check 2, on every draw. Judges run on every valid draw, so the
    confusion matrix covers all draws."""
    bat = ctx.need_battery()
    rows, jobs = [], []
    for dr in drs:
        line = ctx.line(dr.id)
        ask = ask_for(line)
        row = {"id": dr.id, "kind": line["kind"], "slot": dr.slot, "draw": dr.draw, "generator": dr.generator,
               "path": ctx.paths.rel(dr.path), "notes": dr.notes, "record": dr.record}
        bad = validity(ctx.paths, dr, seed_text)
        row["void"] = bad
        if bad:
            row["reasons"] = ["void: " + b for b in bad]
            rows.append(row)
            continue
        text = dr.path.read_text()
        row["text_sha256"] = file_sha(dr.path)
        freasons, finfo = format_check(text, ask)
        body = split_ask(text, ask)[0]
        row.update(finfo)
        row["c1"] = bat.check(body)
        row["c5"] = mentions(text)
        row["reasons"] = list(freasons)
        if not row["c1"]["ok"]:
            row["reasons"].append(f"check 1: max item share {row['c1']['max_share']} ({row['c1']['worst_item']})"
                                  + (f", banned {row['c1']['banned']}" if row["c1"]["banned"] else ""))
        if not row["c5"]["ok"]:
            row["reasons"].append(f"check 5: language model(s) {row['c5']['language_models']}, "
                                  f"you/your {row['c5']['you_your']}, ratio {row['c5']['ratio']}")
        jobs.append((row, "c3", lambda b=body, k=line["kind"]: ctx.judge.mainline(ctx.lines, k, b)))
        if dr.slot != "seed" and seed_steps:
            jobs.append((row, "c4", lambda b=body: ctx.judge.steps(seed_steps, b)))
        rows.append(row)
    with ThreadPoolExecutor(ctx.workers) as ex:
        results = list(ex.map(lambda j: j[2](), jobs))
    for (row, name, _), res in zip(jobs, results):
        row[name] = res
    for row in rows:
        if row["void"]:
            continue
        c3 = row["c3"]
        c3["ok"] = c3["judged"] == row["id"]
        if c3["judged"] is None:
            row["reasons"].append("check 3: judge answer unparseable")
        elif not c3["ok"]:
            row["reasons"].append(f"check 3: judge named {c3['judged']}")
        if "c4" in row:
            c4 = row["c4"]
            c4["ok"] = bool(c4["present"]) and all(c4["present"])
            if c4["present"] is None:
                row["reasons"].append("check 4: judge answer unparseable")
            elif not c4["ok"]:
                row["reasons"].append(f"check 4: missing step(s) "
                                      f"{[k + 1 for k, x in enumerate(c4['present']) if not x]}")
        elif row["slot"] != "seed":
            row["reasons"].append("check 4: no seed steps")
    return rows


def slot_status(rows: list[dict], ords: dict[str, int], slot: str, accepted_key: str | None) -> dict:
    mine = sorted((r for r in rows if r["slot"] == slot), key=lambda r: r["draw"])
    st = {"draws": [r["draw"] for r in mine], "n_draws": len(mine)}
    if accepted_key:
        st.update(status="accepted", accepted_draw=int(accepted_key.split("_d")[1]))
    elif not mine:
        st.update(status="missing", next_draw=0)
    elif len(mine) >= MAX_DRAWS:
        st.update(status="excluded")
    else:
        st.update(status="needs_redraw", next_draw=max(st["draws"]) + 1, redraws_left=MAX_DRAWS - len(mine))
    return st


def log_rows(ctx: Ctx, run: str, rows: list[dict]) -> None:
    keep = ("id", "kind", "slot", "draw", "generator", "path", "text_sha256", "void", "words", "steps",
            "c1", "c3", "c4", "c5", "c2", "status", "reasons")
    append_jsonl(ctx.paths.check_log, [{"run": run, "at": now(), **{k: r[k] for k in keep if k in r}} for r in rows])


# ---------------------------------------------------------------- check-seed

def check_seed(ctx: Ctx, arg_id: str, force: bool = False) -> dict:
    line = ctx.line(arg_id)
    drs = discover(ctx.paths, arg_id, {"seed"})
    ords = ordinal(drs)
    rows = evaluate(ctx, sorted(drs, key=lambda d: ords[d.key]), None, None)
    accepted = None
    for r in sorted(rows, key=lambda r: r["draw"]):
        if accepted:
            r["status"] = "ignored: an earlier draw was accepted"
            continue
        r["status"] = "pass" if not r["reasons"] else "fail"
        if r["status"] == "pass":
            accepted = r
    st = slot_status(rows, ords, "seed", f"seed_d{accepted['draw']}" if accepted else None)
    out = {"id": arg_id, "kind": line["kind"], "created": now(), "status": st["status"], "slot": st, "draws": rows}
    adir = ctx.paths.arg(arg_id)
    if accepted:
        src, dst = ctx.paths.resolve(accepted["path"]), adir / "seed.md"
        if dst.exists() and dst.read_bytes() != src.read_bytes() and not force:
            raise SystemExit(f"{dst} exists and differs from {src.name}; wordings may already depend on it "
                             "(--force to replace)")
        shutil.copyfile(src, dst)
        out["seed_md"] = ctx.paths.rel(dst)
        out["seed_sha256"] = file_sha(dst)
    elif st["status"] == "excluded":
        out["dropped"] = True
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "seed_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log_rows(ctx, "check-seed", rows)
    return out


# ---------------------------------------------------------------- check (wordings)

def seed_of(ctx: Ctx, arg_id: str) -> str:
    p = ctx.paths.arg(arg_id) / "seed.md"
    if not p.exists():
        raise SystemExit(f"{arg_id}: no accepted seed draft ({p}); run check-seed first")
    return p.read_text()


def check_wordings(ctx: Ctx, arg_id: str) -> dict:
    line = ctx.line(arg_id)
    seed = seed_of(ctx, arg_id)
    ask = ask_for(line)
    seed_steps = extract_steps(seed, ask)
    drs = discover(ctx.paths, arg_id, set(SLOTS))
    ords = ordinal(drs)
    rows = evaluate(ctx, drs, seed, seed_steps)
    bodies = {}
    accepted: dict[str, str] = {}  # slot -> draw key
    order = sorted(rows, key=lambda r: (ords[f"{r['slot']}_d{r['draw']}"], SLOTS.index(r["slot"])))
    for r in order:
        key = f"{r['slot']}_d{r['draw']}"
        if r["slot"] in accepted:
            r["status"] = "ignored: the slot already has an accepted draw"
            continue
        if r["reasons"]:
            r["status"] = "fail"
            continue
        body = split_ask(ctx.paths.resolve(r["path"]).read_text(), ask)[0]
        js = {k: round(j3(body, bodies[k]), 4) for k in accepted.values()}
        worst = max(js.items(), key=lambda kv: kv[1]) if js else (None, 0.0)
        r["c2"] = {"max_j3": worst[1], "with": worst[0], "ok": worst[1] <= MAX_PAIR_J3}
        if not r["c2"]["ok"]:
            r["reasons"].append(f"check 2: 3-gram Jaccard {worst[1]} with {worst[0]}")
            r["status"] = "fail"
            continue
        r["status"] = "pass"
        accepted[r["slot"]] = key
        bodies[key] = body
    slots = {s: slot_status(rows, ords, s, accepted.get(s)) for s in SLOTS}
    # The accepted draw of each slot is copied to <ID>/<slot>.md (runs/s2_assent.py reads exactly
    # those); a stale copy of a slot that is no longer accepted is removed.
    adir = ctx.paths.arg(arg_id)
    for s in SLOTS:
        dst = adir / f"{s}.md"
        if s in accepted:
            shutil.copyfile(adir / f"{accepted[s]}.md", dst)
            slots[s]["copy"] = ctx.paths.rel(dst)
            slots[s]["sha256"] = file_sha(dst)
        elif dst.exists():
            dst.unlink()
            slots[s]["removed_stale_copy"] = True
    labels = ["seed"] + [accepted[s] for s in SLOTS if s in accepted]
    texts = {"seed": split_ask(seed, ask)[0], **bodies}
    matrix = [[round(j3(texts[a], texts[b]), 4) for b in labels] for a in labels]
    prompt_shas = sorted({r["record"].get("prompt_sha256") for r in rows if r.get("record")} - {None})
    warnings = [f"{len(prompt_shas)} distinct wording prompts (the spec uses one stamped prompt per argument)"] \
        if len(prompt_shas) > 1 else []
    out = {"id": arg_id, "kind": line["kind"], "created": now(), "seed_sha256": sha(seed.encode()),
           "seed_steps": len(seed_steps), "accepted": accepted, "n_accepted": len(accepted), "slots": slots,
           "pairwise": {"labels": labels, "matrix": matrix}, "prompt_sha256s": prompt_shas,
           "warnings": warnings, "draws": rows}
    (ctx.paths.arg(arg_id) / "check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log_rows(ctx, "check", rows)
    return out


# ---------------------------------------------------------------- gpt-5.6-sol wordings

def wording_prompt(ctx: Ctx, arg_id: str, prompt_path: Path | None = None) -> Path:
    """The rendered wording prompt for this argument, the one stamped prompt all 7 wordings share:
    an explicit file; else the one the Claude wordings were drawn from (all must agree); else
    prompts/s2/openings/rendered/wording_<ID>.md; else rendered here with render.py's own functions.
    Whichever it is must contain the accepted seed draft."""
    seed = seed_of(ctx, arg_id)
    p = prompt_path
    if p is None:
        recs = [r for r in read_jsonl(ctx.paths.claude_log) if r.get("id") == arg_id and r.get("slot") in SLOTS
                and r.get("prompt_path") and ctx.paths.resolve(r["prompt_path"]).exists()
                and file_sha(ctx.paths.resolve(r["prompt_path"])) == r.get("prompt_sha256")
                and canon(seed) in canon(ctx.paths.resolve(r["prompt_path"]).read_text())]
        shas = {r["prompt_sha256"] for r in recs}
        if len(shas) > 1:
            raise SystemExit(f"{arg_id}: the Claude wordings used {len(shas)} different prompts; pass --prompt")
        if recs:
            p = ctx.paths.resolve(recs[0]["prompt_path"])
    if p is None and (ctx.paths.render_py.parent / f"rendered/wording_{arg_id}.md").exists():
        p = ctx.paths.render_py.parent / f"rendered/wording_{arg_id}.md"
    if p is None:
        p = render_wording(ctx, arg_id, seed)
    if canon(seed) not in canon(p.read_text()):
        raise SystemExit(f"{p}: does not contain the accepted seed draft {arg_id}/seed.md (stale render?)")
    return p


def render_wording(ctx: Ctx, arg_id: str, seed: str) -> Path:
    """Exactly what `render.py --wording` writes, via its own fill and checks, into data/<ID>/."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s2_render", ctx.paths.render_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lines = mod.load_lines()
    r = next(x for x in lines if x["id"] == arg_id)
    tmpl = mod.load_template("wording.md")
    values = {"seed": seed.strip(), "line": r["line"], "claim": r["claim"], "banned": mod.banned_for(r, lines)}
    text = mod.fill(tmpl, values)
    mod.check(text, r, lines, values, tmpl)
    text = text.strip() + "\n"
    p = ctx.paths.arg(arg_id) / f"wording_prompt_{sha(text)[:8]}.md"
    p.write_text(text)
    return p


def gpt_targets(ctx: Ctx, arg_id: str, slots: list[str] | None, redraw: bool) -> dict[str, int]:
    """Slot -> draw number to generate."""
    drs = [d for d in discover(ctx.paths, arg_id, set(GPT_SLOTS))]
    have = {s: sorted(d.draw for d in drs if d.slot == s) for s in GPT_SLOTS}
    if redraw:
        ck = ctx.paths.arg(arg_id) / "check.json"
        if not ck.exists():
            raise SystemExit(f"{arg_id}: no check.json; run check first")
        st = json.loads(ck.read_text())["slots"]
        want = [s for s in GPT_SLOTS if st[s]["status"] in ("needs_redraw", "missing")]
    else:
        want = slots or [s for s in GPT_SLOTS if not have[s]]
    out = {}
    for s in want:
        if s not in GPT_SLOTS:
            raise SystemExit(f"{s} is not a gpt-5.6-sol slot ({GPT_SLOTS})")
        if len(have[s]) >= MAX_DRAWS:
            print(f"{arg_id} {s}: {len(have[s])} draws already, the limit; skipped")
            continue
        out[s] = (max(have[s]) + 1) if have[s] else 0
    return out


def generate_gpt(ctx: Ctx, arg_id: str, targets: dict[str, int], prompt_path: Path | None = None,
                 params: dict | None = None) -> list[dict]:
    from .foundry import ENDPOINT
    pp = wording_prompt(ctx, arg_id, prompt_path)
    prompt = pp.read_text()
    params = params or GEN_PARAMS
    call = ctx.call or _default_call()
    adir = ctx.paths.arg(arg_id)

    def one(slot: str, draw: int) -> dict:
        msgs = [{"role": "user", "content": prompt}]
        rec = {"id": arg_id, "slot": slot, "draw": draw, "path": ctx.paths.rel(adir / f"{slot}_d{draw}.md"),
               "prompt_path": ctx.paths.rel(pp), "prompt_sha256": file_sha(pp), "generator": GENERATOR,
               "deployment": GENERATOR, "endpoint": ENDPOINT, "params": params, "date": time.strftime("%Y-%m-%d"),
               "created": now()}
        raw = adir / f"{slot}_d{draw}.gpt.json"
        try:
            r = call(GENERATOR, msgs, retries=10, timeout=600, **params)
        except RuntimeError as e:  # HTTP 4xx (content filter included) after foundry.chat's retries
            rec.update(error=str(e)[:1000], model_version=None, finish_reason=None)
            raw.write_text(json.dumps({"request": {"endpoint": ENDPOINT, "deployment": GENERATOR, "messages": msgs,
                                                   "params": params}, "error": str(e)}, indent=1))
            rec["raw_path"] = ctx.paths.rel(raw)
            return rec
        text = _content(r)
        ch = r["choices"][0]
        raw.write_text(json.dumps({"request": {"endpoint": ENDPOINT, "deployment": GENERATOR, "messages": msgs,
                                               "params": params}, "response": r, "model_version": r.get("model"),
                                   "created": rec["created"]}, indent=1, ensure_ascii=False))
        rec.update(model_version=r.get("model"), finish_reason=ch.get("finish_reason"), usage=r.get("usage"),
                   raw_path=ctx.paths.rel(raw))
        if text:
            (adir / f"{slot}_d{draw}.md").write_text(text + "\n")
            rec["text_sha256"] = file_sha(adir / f"{slot}_d{draw}.md")
        return rec

    adir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max(1, len(targets))) as ex:
        recs = list(ex.map(lambda kv: one(*kv), targets.items()))
    append_jsonl(ctx.paths.gpt_log, recs)
    return recs


# ---------------------------------------------------------------- pilot / held-out assignment

def assign_seed(ctx: Ctx) -> tuple[str, str]:
    """The draw's seed: the first 8 hex digits of the SHA-256 of results/s2/openings/manifest.txt
    (written by `report`, before any model reply existed). Returns (seed_hex, full SHA-256). Refuses
    when a file the draw reads or assigns (each argument's check.json and accepted <slot>.md) is not
    in the manifest with its current hash, so the seed commits to exactly what is assigned."""
    mf = ctx.paths.report_dir / "manifest.txt"
    if not mf.exists():
        raise FileNotFoundError(f"{ctx.paths.rel(mf)} is missing: run `report` first")
    full = file_sha(mf)
    listed = {ln.split("  ", 1)[1]: ln[:64] for ln in mf.read_text().splitlines() if "  " in ln}
    bad = []
    for i in sorted(ctx.lines):
        ck = ctx.paths.arg(i) / "check.json"
        files = [ck] if ck.exists() else []
        if ck.exists():
            files += [ctx.paths.arg(i) / f"{s}.md" for s in json.loads(ck.read_text())["accepted"]]
        bad += [ctx.paths.rel(f) for f in files if not f.exists() or listed.get(ctx.paths.rel(f)) != file_sha(f)]
    if bad:
        raise ValueError(f"not in the openings manifest with their current hash: {bad}")
    return full[:8], full


def assign(ctx: Ctx) -> dict:
    """Section 4's seeded draw over the wordings that survived the checks: per argument one pilot,
    five held-out, one spare; for 5 arguments drawn from all 20 the spare is a second pilot. With
    fewer than 7 survivors the spare goes first, then the second pilot, then held-out slots. The
    seed is derived from the openings manifest (assign_seed), never passed in."""
    seed_hex, man_sha = assign_seed(ctx)
    rng = random.Random(int(seed_hex, 16))
    ids = sorted(ctx.lines)
    second = sorted(rng.sample(ids, min(5, len(ids))))
    out = {"seed_hex": seed_hex, "seed_source": f"sha256({ctx.paths.rel(ctx.paths.report_dir / 'manifest.txt')})[:8]",
           "manifest_sha256": man_sha, "second_pilot_arguments": second, "created": now(), "arguments": {}}
    for i in ids:
        ck = ctx.paths.arg(i) / "check.json"
        acc = json.loads(ck.read_text())["accepted"] if ck.exists() else {}
        surv = sorted(acc)
        rng.shuffle(surv)
        roles = ["pilot"] + [f"heldout{k}" for k in range(1, 6)] + ["pilot2" if i in second else "spare"]
        out["arguments"][i] = {"survivors": len(surv), "roles": {s: roles[k] for k, s in enumerate(surv)},
                               "draws": {s: acc[s] for s in surv}}
    return out


# ---------------------------------------------------------------- report

def item_ask_overlap(ctx: Ctx) -> list[dict]:
    bat = ctx.need_battery()
    claims = sorted({ask_for(l) for l in ctx.lines.values()})
    rows = []
    for it in bat.items:
        ig = ngrams(question(it), 3)  # the question without the constant answer instruction
        row = {"id": it["id"], "kind": it.get("kind"), "entity": it.get("entity"),
               "twin": it.get("twin") or {"emotion_present": "pos", "emotion_absent": "neg"}.get(it.get("yes_means")),
               "pair_id": it.get("pair_id")}
        for a in claims:
            ag = ngrams(a, 3)
            row[a] = round(len(ig & ag) / len(ig), 4) if ig else 0.0
        rows.append(row)
    return rows


def manifest(ctx: Ctx, extra: list[Path]) -> list[tuple[str, str]]:
    files: set[Path] = set()
    p = ctx.paths
    for f in [p.lines, p.wording_md, p.render_py, p.items, *extra]:
        if f.exists():
            files.add(f)
    if p.data.exists():
        files |= {f for f in p.data.rglob("*") if f.is_file()}
    for r in read_jsonl(p.claude_log) + read_jsonl(p.gpt_log):
        for k in ("prompt_path", "path"):
            if r.get(k) and p.resolve(r[k]).exists():
                files.add(p.resolve(r[k]))
    return sorted((file_sha(f), p.rel(f)) for f in files)


def report(ctx: Ctx, extra_manifest: list[Path] = ()) -> dict:
    ids = sorted(ctx.lines)
    seeds, checks = {}, {}
    for i in ids:
        for name, store in (("seed_check.json", seeds), ("check.json", checks)):
            f = ctx.paths.arg(i) / name
            if f.exists():
                store[i] = json.loads(f.read_text())
    all_rows = [r for i in ids for src in (seeds, checks) if i in src for r in src[i]["draws"]]
    # failure rates by argument and generator (seed drafts counted under "claude-seed")
    fail = {}
    for r in all_rows:
        if str(r.get("status", "")).startswith("ignored"):
            continue
        g = "claude-seed" if r["slot"] == "seed" else r["generator"]
        f = fail.setdefault((r["id"], g), [0, 0])
        f[0] += 1
        f[1] += r.get("status") == "fail"
    reasons = {}
    for r in all_rows:
        for x in r.get("reasons", []):
            k = x.split(":")[0] if not x.startswith("void") else "void: " + x.split(": ", 1)[1].split("=")[0][:40]
            reasons[k] = reasons.get(k, 0) + 1
    confusion = {}
    for kind in KIND_TEXT:
        kid = kind_ids(ctx.lines, kind)
        cols = kid + ["unparseable"]
        m = {a: {c: 0 for c in cols} for a in kid}
        for r in all_rows:
            if r["kind"] == kind and r.get("c3"):
                m[r["id"]][r["c3"]["judged"] or "unparseable"] += 1
        confusion[kind] = {"rows": kid, "cols": cols, "matrix": [[m[a][c] for c in cols] for a in kid]}
    mention_rows = [{"id": r["id"], "slot": r["slot"], "draw": r["draw"], "generator": r["generator"],
                     "status": r.get("status"), **{k: r["c5"][k] for k in ("language_models", "you_your", "ratio")}}
                    for r in all_rows if r.get("c5")]
    overlap = item_ask_overlap(ctx)
    asks = sorted({ask_for(l) for l in ctx.lines.values()})
    summ = {}
    for row in overlap:
        for a in asks:
            k = (a, str(row["entity"]), str(row["twin"]))
            summ.setdefault(k, []).append(row[a])
    overlap_summary = [{"ask": a, "entity": e, "twin": t, "n": len(v), "mean": round(sum(v) / len(v), 4),
                        "max": max(v)} for (a, e, t), v in sorted(summ.items())]
    status = {i: {"seed": seeds.get(i, {}).get("status", "unchecked"),
                  "accepted": checks.get(i, {}).get("n_accepted"),
                  "slots": {s: v["status"] for s, v in checks.get(i, {}).get("slots", {}).items()}} for i in ids}
    man = manifest(ctx, list(extra_manifest))
    out = {"created": now(), "status": status, "failure_rates": [
        {"id": i, "generator": g, "draws": n, "failed": f, "rate": round(f / n, 3)} for (i, g), (n, f) in sorted(fail.items())],
        "reasons": reasons, "confusion": confusion,
        "pairwise": {i: c["pairwise"] for i, c in checks.items()}, "mentions": mention_rows,
        "item_ask_overlap": overlap, "item_ask_overlap_summary": overlap_summary,
        "manifest": [{"sha256": h, "path": f} for h, f in man]}
    d = ctx.paths.report_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (d / "manifest.txt").write_text("".join(f"{h}  {f}\n" for h, f in man))
    (d / "report.md").write_text(report_md(out, asks))
    return out


def _table(head: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def report_md(o: dict, asks: list[str]) -> str:
    s = [f"# Study 2 openings report ({o['created']})", "", "## Draw status", ""]
    s.append(_table(["argument", "seed", "accepted", *SLOTS],
                    [[i, v["seed"], v["accepted"], *[v["slots"].get(k, "") for k in SLOTS]] for i, v in o["status"].items()]))
    s += ["", "## Failure rates by argument and generator", ""]
    s.append(_table(["argument", "generator", "draws", "failed", "rate"],
                    [[r["id"], r["generator"], r["draws"], r["failed"], r["rate"]] for r in o["failure_rates"]]))
    s += ["", "Failure reasons (a draw can have several): " +
          ", ".join(f"{k} {v}" for k, v in sorted(o["reasons"].items())), ""]
    for kind, c in o["confusion"].items():
        s += [f"## Main-line judge confusion, {kind} (rows: intended, columns: judged; all draws)", ""]
        s.append(_table(["", *c["cols"]], [[a, *row] for a, row in zip(c["rows"], c["matrix"])]))
        s.append("")
    s += ["## Pairwise 3-gram Jaccard of the accepted wordings (closing ask removed; seed for reference)", ""]
    for i, p in o["pairwise"].items():
        s += [f"### {i}", "", _table(["", *p["labels"]], [[a, *row] for a, row in zip(p["labels"], p["matrix"])]), ""]
    s += ["## Mentions per draw", ""]
    s.append(_table(["argument", "slot", "draw", "generator", "status", "language model(s)", "you/your", "ratio"],
                    [[m["id"], m["slot"], m["draw"], m["generator"], m["status"], m["language_models"], m["you_your"],
                      m["ratio"]] for m in o["mentions"]]))
    s += ["", "## Item overlap with the closing ask (share of the item's 3-grams in the ask)", ""]
    for k, a in enumerate(asks):
        s.append(f"Ask {k + 1}: {a}")
    s.append("")
    s.append(_table(["ask", "subject", "twin", "items", "mean", "max"],
                    [[asks.index(r["ask"]) + 1, r["entity"], r["twin"], r["n"], r["mean"], r["max"]]
                     for r in o["item_ask_overlap_summary"]]))
    s += ["", f"Per-item values are in report.json; the manifest ({len(o['manifest'])} files) is manifest.txt.", ""]
    return "\n".join(s)
