"""Study 2 measures, ID checks and pilot gates (drafts/study2-spec.md sections 8, 9; section 12 step 8).

    .venv/bin/python runs/s2_table.py --phase confirm --out table.csv readouts.csv [more.csv|.jsonl ...]
    .venv/bin/python runs/s2_table.py --phase pilot --out pilot_table.csv --gates gates.csv \
        --decision decision.json readouts.csv

Input: one row per readout (CSV with a header, or JSONL), as the Study 2 runner writes it. Columns:

  model        qwen | k26
  phase        pilot | confirm (must equal --phase; pilot and confirmatory data are never pooled)
  arm          S2-DEN | S2-CON | S2-FTD | S2-FTC (FTD and FTC Qwen only)
  argument     DA01..DA10 (S2-DEN) or CA01..CA10 (the other arms)
  arg_kind     emotion-specific | experience-general | control
  wording      wording slot 0..6 (0, 2, 4, 6 Claude Opus 5.5; 1, 3, 5 gpt-5.6-sol)
  wording_role pilot | pilot2 | heldout (pilot phase: pilot or pilot2; confirm phase: heldout)
  opening      opening id, one per (argument, wording)
  vec          vector id: emotion name (joy ...), N01..N60 (sampled-noun list position), arithmetic, random,
               red, ocean, mp:<slug> (mental-process control), none (zero vector)
  vclass       emotion | noun | arithmetic | noise | thinking | bridge | none
  layer        injection layer (qwen 40, k26 20)
  dose         natural units (registered dose or half of it; 0 for the zero vector)
  site         post_subject
  item         item id (items_s2.jsonl)
  kind         s2_capacity | s2_state | s2_placebo | s2_persona | s2_reference | s2_reserve (rule-2 items) |
               yes_bias (pilot only)
  para         paraphrase 1..8 for capacity and state items; the predicate id (P..) for placebo, persona,
               reference and reserve items; the item id for yes_bias
  pair_id      links the two twins of an item
  twin         pos | neg
  subject      you | language models | other language models | own name (or Qwen / Kimi) | rocks | humans |
               empty for yes_bias
  sign         +1 when the item file's yes_means is emotion_present, property_present or true_claim, -1 when
               it is emotion_absent, property_absent or false_claim (the same polarity as Study 1)
  y            logit(Yes) - logit(No), fp32 two-row readout
  y0           the same item's y under the zero vector, same arm and opening (equal to y on zero-vector rows)
  mass         P(Yes) + P(No) from the bf16 full-vocabulary pass
  top1         full-vocabulary argmax token, decoded and stripped ("Yes", "No" or anything else)
  bucket, chunk  the engine's bucket and chunk index of this suffix
  n_steered    number of steered positions for this item (0 on zero-vector rows)
  correct_answer  the item file's correct_answer (Yes | No; empty on emotion items, which have none). The rule 1
               gates score known-answer accuracy and rocks and humans tracking accuracy against it, never against
               sign ("Can rocks read text?" has sign +1 and correct answer No)
  optional: reply_len, emo_words (per opening, S2-G5 covariates), ckpt (merged-checkpoint SHA-256), text (the
               item's question; otherwise read from --items by item id), reference_fit and gate (the item file's
               flags; if present they must agree with the kind and subject rules used here)

Output (--out), one row per (group, cell, measure), long like runs/confirm_table.py. A group is
(model, arm, opening, vec, layer, dose); a cell is a paraphrase (para) within a group, para "-" for
group-level values. Measures (all twin-averaged per paraphrase; r = sign x (y - (a y0 + c)) with a, c the
unweighted OLS fit of y on y0 over the group's reference set: s2_reference and s2_reserve items and the rocks
and humans rows of s2_placebo; delta = sign x (y - y0); s0 = sign x y0):

  d            r(you) - r(language models), capacity frame (S2-H1 primary)
  d_raw        delta(you) - delta(language models), capacity frame
  d_e          as d with the S2-R1 fit (a_e, c_e over the 32 rocks and humans capacity items)
  d_other      r(you) - r(other language models)
  d_you_name, d_name_lm   r(you) - r(own name), r(own name) - r(language models)
  d_pos, d_neg            d from one twin only
  d_state, d_state_raw    d and d_raw in the state frame
  d_plc        r(you) - r(language models) on a placebo predicate (para = predicate id; S2-G2's q = 0 rows)
  d_persona    r(you) - r(language models) on the persona item
  r_<subject>  r per subject on capacity items (you, lm, other, name, rocks, humans); mass_<subject> likewise
  s0_you, s0_lm, gap0     unsteered depth per cell and s0(you) - s0(language models)
  mass         mean P(Yes)+P(No) over the cell's four capacity you and language-models readouts
  flag         1 if any of those readouts has y0 more than 3 log-odds outside the reference y0 range
  persona_mass_<subject>  mass on the persona item
  ovl_pos, ovl_neg, ovl_lm_pos, ovl_lm_neg, r_lm_pos, r_lm_neg
               copying route (section 9): closing-ask overlap of the cell's you and language-models items (or of
               the language-models item alone) per twin, and r(language models) per twin. Overlap is the share of
               the item's content words (lowercased, stopwords removed, the trailing "Answer with one word, Yes or
               No." excluded, and each word reduced by dropping a plural "s" or "es", so "emotions" matches
               "emotion") that appear in that opening's closing ask, the last paragraph of
               data/s2/openings/<argument>/<slot>.md (falling back to the stamped ask template with the claim from
               prompts/s2/openings/lines.json). Word-triple overlap was zero for every item, so it is not used.
  Descriptive extras (section 9; none enters a registered test):
  d_prob       P(Yes) beside log-odds: [sigmoid(sign x y) - sigmoid(sign x y0)] (you) minus the same for language
               models, capacity frame, twin-averaged. sigmoid(y) is the two-way P(Yes | Yes or No) of the fp32 readout;
               no compression correction (compare with d_raw)
  flip_<subject>  share of the cell's two capacity readouts (twins) whose argmax answer differs from the unsteered
               one. The answer is top1 when it is exactly "Yes" or "No" (as the rule 1 gates read it), else "other"
  d_flip       directional argmax flips, you minus language models: per readout +1 if the answer moved to "emotion
               present" (Yes on a sign +1 twin, No on a sign -1 twin), -1 if it moved away from it, 0 otherwise
  mdrop_<subject>, persona_mdrop_<subject>
               Yes+No mass drop, unsteered minus steered mass, on the capacity items and on the persona item
  group level: wording_lm_mentions, wording_you_mentions: whole-word counts of "language model(s)" and of
               you/your/yours/yourself/yourselves in the opening's wording file (data/s2/openings/<argument>/w<slot>.md
               minus its leading comment lines, the text the runner sends; same patterns as runs/s2_assent.py), the
               subject-mention moderator; absent when the wording file is not found
  group level: fit_a, fit_c, fit_n, ref_lo, ref_hi, fit_a_e, fit_c_e; ref_resid (para = y0 bin, the
               linearity check for y0 < -15)

Checks (each raises ValueError): Study 1 IDs (D0, D1, D1.4, D2, D2c, bare H1..H8) anywhere; arm, argument,
vector and class names and their pairing; the model's registered emotions, layer and doses; red and ocean in
S2-DEN only; phase and wording role; yes_bias only in the pilot; the pilot restriction of emotion vectors (and
of every K2.6 vector) to the 100 gate items; one readout per (group, item); every steered row has a zero-vector
row whose y equals its y0 and whose (bucket, chunk) equals its own; steered groups ask the zero group's items
(confirm phase); twins pair as pos and neg with opposite signs; equal steered-position counts across subjects
within a paraphrase and twin; one checkpoint per arm, distinct for S2-FTD and S2-FTC.

With --gates (pilot phase) it also writes the rule 1 gate statistics per (model, vector, dose) and, with
--decision, a JSON of what rules 1 and 2 decide plus the unsteered rocks and humans twin sign check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- registered constants (spec sections 2, 3) ------------------------------------------------------
CFG = {
    "qwen": {"layer": 40, "dose": 2.0, "emo": ["joy", "sadness", "anger", "fear", "euphoria", "surprise"],
             "name": "Qwen"},
    "k26": {"layer": 20, "dose": 4.0, "emo": ["joy", "fear", "euphoria", "surprise"], "name": "Kimi"},
}
ARMS = ("S2-DEN", "S2-CON", "S2-FTD", "S2-FTC")
CLASSES = ("emotion", "noun", "arithmetic", "noise", "thinking", "bridge", "none")
PRIMARY = ("noun", "arithmetic", "noise")  # the S2-H1 control class
KINDS = ("s2_capacity", "s2_state", "s2_placebo", "s2_persona", "s2_reference", "s2_reserve", "yes_bias")
SUBJ = {"you": "you", "language models": "lm", "other language models": "other", "own name": "name",
        "rocks": "rocks", "humans": "humans", "": ""}
STUDY1_IDS = re.compile(r"^(D0|D1|D1\.4|D2|D2c|H[1-8])$")
FLAG_MARGIN = 3.0  # log-odds outside the reference y0 range (section 9)
LIN_BINS = [(-math.inf, -25.0), (-25.0, -20.0), (-20.0, -15.0)]  # linearity check, y0 < -15
GATE = {"ka": 0.90, "mass": 0.90, "track": 0.95, "ka_mean": 0.95}  # rule 1
NUM = {"layer": int, "dose": float, "wording": int, "sign": int, "y": float, "y0": float, "mass": float,
       "n_steered": int}
REQUIRED = ["model", "phase", "arm", "argument", "arg_kind", "wording", "wording_role", "opening", "vec",
            "vclass", "layer", "dose", "site", "item", "kind", "para", "pair_id", "twin", "subject", "sign",
            "y", "y0", "mass", "top1", "bucket", "chunk", "n_steered"]


class S2Error(ValueError):
    pass


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise S2Error(msg)


# ---- loading ------------------------------------------------------------------------------------------
def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        if p.suffix == ".jsonl":
            rows += [json.loads(line) for line in p.open() if line.strip()]
        else:
            with p.open(newline="") as f:
                rows += list(csv.DictReader(f))
    need(len(rows) > 0, "no readout rows")
    missing = [c for c in REQUIRED if c not in rows[0]]
    need(not missing, f"missing columns: {missing}")
    for r in rows:
        for k, t in NUM.items():
            r[k] = t(float(r[k])) if t is int else t(r[k])
        for k in ("bucket", "chunk", "para", "item", "pair_id", "opening", "argument", "vec", "top1"):
            r[k] = str(r[k]).strip()
        r["subject"] = (r.get("subject") or "").strip()
        for k in ("reply_len", "emo_words"):
            v = r.get(k)
            r[k] = float(v) if v not in (None, "", "NA") else None
    return rows


# ---- ID and structure checks ------------------------------------------------------------------------
def gate_item(r: dict) -> bool:
    """The 100 gate items (section 5): known-answer, reference, and the rocks and humans
    rows of the non-conceded placebos and the persona item."""
    return (r["kind"] in ("yes_bias", "s2_reference") or
            (r["kind"] in ("s2_placebo", "s2_persona") and r["subject"] in ("rocks", "humans")))


def check_vec(r: dict) -> None:
    m, v, c = r["model"], r["vec"], r["vclass"]
    need(c in CLASSES, f"unknown vector class {c!r}")
    ok = {"emotion": v in CFG[m]["emo"],
          "noun": re.fullmatch(r"N(0[1-9]|[1-5]\d|60)", v) is not None,
          "arithmetic": v == "arithmetic", "noise": v == "random", "bridge": v in ("red", "ocean"),
          "thinking": re.fullmatch(r"mp:[a-z0-9_-]+", v) is not None, "none": v == "none"}[c]
    need(ok, f"vector {v!r} does not fit class {c!r} for model {m}")


def check(rows: list[dict], phase: str) -> None:
    ckpt = defaultdict(set)
    opening_of = {}
    for r in rows:
        for k, v in r.items():
            need(not (isinstance(v, str) and STUDY1_IDS.match(v.strip())),
                 f"Study 1 ID {v!r} in column {k!r}: rejected in any S2 table")
        m = r["model"]
        need(m in CFG, f"unknown model {m!r}")
        need(r["phase"] == phase, f"row phase {r['phase']!r} != --phase {phase}")
        need(r["arm"] in ARMS, f"unknown arm {r['arm']!r}")
        need(m == "qwen" or r["arm"] in ("S2-DEN", "S2-CON"), f"{r['arm']} is Qwen only")
        pre = "DA" if r["arm"] == "S2-DEN" else "CA"
        need(re.fullmatch(pre + r"(0[1-9]|10)", r["argument"]) is not None,
             f"argument {r['argument']!r} not allowed in {r['arm']}")
        need((r["arg_kind"] == "control") == (pre == "CA") and
             r["arg_kind"] in ("emotion-specific", "experience-general", "control"),
             f"arg_kind {r['arg_kind']!r} does not fit {r['argument']}")
        need(0 <= r["wording"] <= 6, f"wording slot {r['wording']} outside 0..6")
        key = (m, r["arm"], r["opening"])
        need(opening_of.setdefault(key, (r["argument"], r["wording"])) == (r["argument"], r["wording"]),
             f"opening {r['opening']} maps to more than one (argument, wording)")
        roles = ("pilot", "pilot2") if phase == "pilot" else ("heldout",)
        need(r["wording_role"] in roles, f"wording_role {r['wording_role']!r} in a {phase} table")
        check_vec(r)
        need(r["vclass"] != "bridge" or r["arm"] == "S2-DEN", "red and ocean run in S2-DEN only")
        need(r["site"] == "post_subject", f"site {r['site']!r}: only post_subject exists")
        if r["vclass"] == "none":
            need(r["dose"] == 0 and r["n_steered"] == 0, "zero-vector rows carry dose 0 and n_steered 0")
            need(r["y"] == r["y0"], "zero-vector rows must have y == y0")
        else:
            need(r["layer"] == CFG[m]["layer"], f"layer {r['layer']} != registered {CFG[m]['layer']} for {m}")
            need(any(abs(r["dose"] - f * CFG[m]["dose"]) < 1e-9 for f in (1.0, 0.5)),
                 f"dose {r['dose']} is neither the registered dose nor half of it")
            need(r["n_steered"] > 0, "steered row with no steered positions")
        need(r["kind"] in KINDS, f"unknown item kind {r['kind']!r}")
        need(r["kind"] != "yes_bias" or phase == "pilot", "known-answer (yes_bias) items run in the pilot only")
        s = r["subject"]
        if s in (CFG["qwen"]["name"], CFG["k26"]["name"]):
            need(s == CFG[m]["name"], f"own-name subject {s!r} in {m} rows")
            r["subject"] = s = "own name"
        need(s in SUBJ, f"unknown subject {s!r}")
        need((s == "") == (r["kind"] == "yes_bias"), "only yes_bias items lack a subject")
        need(r["twin"] in ("pos", "neg") and r["sign"] in (1, -1), "twin must be pos/neg and sign +-1")
        for col, rule in (("reference_fit", is_ref(r)), ("gate", gate_item(r))):  # item-file flags, if carried
            v = r.get(col)
            if v not in (None, ""):
                need((str(v).strip().lower() in ("true", "1")) == rule,
                     f"item {r['item']}: {col}={v} disagrees with its kind and subject ({r['kind']}, {s})")
        if phase == "pilot" and r["vclass"] != "none":
            need(gate_item(r) or (r["vclass"] != "emotion" and m == "qwen"),
                 f"pilot {m} {r['vclass']} vector {r['vec']} on non-gate item {r['item']} ({r['kind']}, "
                 f"{s}): emotion vectors, and every K2.6 vector, run on gate items only in the pilot")
        if r.get("ckpt"):
            ckpt[(m, r["arm"])].add(r["ckpt"])
    for k, v in ckpt.items():
        need(len(v) == 1, f"more than one checkpoint hash in {k}")
    if ("qwen", "S2-FTD") in ckpt and ("qwen", "S2-FTC") in ckpt:
        need(ckpt[("qwen", "S2-FTD")] != ckpt[("qwen", "S2-FTC")], "S2-FTD and S2-FTC checkpoints are identical")
    for a in ("S2-FTD", "S2-FTC"):
        if ("qwen", a) in ckpt and ("qwen", "S2-DEN") in ckpt:
            need(ckpt[("qwen", a)] != ckpt[("qwen", "S2-DEN")], f"{a} checkpoint equals the base checkpoint")


def groups_of(rows: list[dict], phase: str) -> tuple[dict, dict]:
    """Steered groups and zero-vector groups, with the baseline and twin checks."""
    steered, zero = defaultdict(dict), defaultdict(dict)
    for r in rows:
        if r["vclass"] == "none":
            g = zero[(r["model"], r["arm"], r["opening"])]
        else:
            g = steered[(r["model"], r["arm"], r["opening"], r["vec"], r["layer"], r["dose"])]
        need(r["item"] not in g, f"duplicate readout for item {r['item']} in group {r['model']}|{r['arm']}|"
                                 f"{r['opening']}|{r['vec']}")
        g[r["item"]] = r
    for key, g in steered.items():
        z = zero.get(key[:3])
        need(z is not None, f"no zero-vector group for {key[:3]}")
        if phase == "confirm":
            need(set(g) == set(z), f"group {key} asks different items from its zero-vector group")
        for it, r in g.items():
            r0 = z.get(it)
            need(r0 is not None, f"item {it} of group {key} has no zero-vector readout")
            need((r["bucket"], r["chunk"]) == (r0["bucket"], r0["chunk"]),
                 f"(bucket, chunk) of {key} item {it} differs from its y0 row: "
                 f"{(r['bucket'], r['chunk'])} vs {(r0['bucket'], r0['chunk'])}")
            need(abs(r["y0"] - r0["y"]) < 1e-6, f"y0 of {key} item {it} differs from the zero-vector y")
    for key, g in list(steered.items()) + list(zero.items()):
        pairs = defaultdict(list)
        for r in g.values():
            pairs[r["pair_id"]].append(r)
        for pid, ps in pairs.items():
            need(len(ps) == 2 and {p["twin"] for p in ps} == {"pos", "neg"} and ps[0]["sign"] == -ps[1]["sign"],
                 f"twin pair {pid} in group {key} is not one pos and one neg with opposite signs")
        if key in steered:
            ns = defaultdict(set)
            for r in g.values():
                if r["kind"] != "yes_bias":
                    ns[(r["kind"], r["para"], r["twin"])].add(r["n_steered"])
            for k, v in ns.items():
                need(len(v) == 1, f"steered position counts differ across subjects for {k} in {key}: {sorted(v)}")
    return steered, zero


# ---- measures -----------------------------------------------------------------------------------------
def ols(x: list[float], y: list[float]) -> tuple[float, float]:
    n = len(x)
    need(n >= 3, f"reference fit needs at least 3 items, got {n}")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    need(sxx > 0, "reference y0 has no spread")
    a = sum((p - mx) * (q - my) for p, q in zip(x, y)) / sxx
    return a, my - a * mx


def mean(v):
    return sum(v) / len(v)


# ---- closing-ask overlap (copying route) ----------------------------------------------------------------
STOP = set("""a about above after again against all am an and any are aren't as at be because been before being
below between both but by can cannot could couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he her here hers herself him himself his how i if in into is
isn't it it's its itself just me more most mustn't my myself no nor not now of off on once only or other ought our
ours ourselves out over own same shan't she should shouldn't so some such than that that's the their theirs them
themselves then there these they this those through to too under until up very was wasn't we were weren't what
when where which while who whom why will with won't would wouldn't you your yours yourself yourselves""".split())
ASK_STEM = "So I will ask you directly"
ASK_TEMPLATE = ("So I will ask you directly: do you agree that language models, you included, {claim}? If one of the "
                "steps above is wrong, say which one and why. I would rather have your real view than a diplomatic one.")
TAIL = re.compile(r"\s*Answer with one word, Yes or No\.?\s*$")
LM_RE = re.compile(r"\blanguage models?\b", re.IGNORECASE)  # as runs/s2_assent.py LM_RE and YOU_RE
YOU_RE = re.compile(r"\b(?:you|your|yours|yourself|yourselves)\b", re.IGNORECASE)


def singular(w: str) -> str:
    """Light plural stripping, applied to item and ask words alike: emotions -> emotion, boxes -> box,
    classes -> class; words in -ss, -us, -is and short words are left alone."""
    if len(w) <= 3 or w.endswith(("ss", "us", "is", "'s")):
        return w
    if re.search(r"(s|x|z|ch|sh)es$", w):
        return w[:-2]
    return w[:-1] if w.endswith("s") else w


def content_words(text: str) -> set[str]:
    return {singular(w) for w in re.findall(r"[a-z]+(?:'[a-z]+)?", TAIL.sub("", text).lower()) if w not in STOP}


class Overlap:
    """Share of an item's content words found in its opening's closing ask."""

    def __init__(self, items: Path | None, openings: Path, lines: Path):
        self.text = {}
        if items and items.exists():
            for line in items.open():
                if line.strip():
                    it = json.loads(line)
                    self.text[it["id"]] = it["text"]
        self.openings = openings
        self.claim = {x["id"]: x["claim"] for x in json.loads(lines.read_text())} if lines.exists() else {}
        self.asks, self.source, self.cache, self.ment = {}, {}, {}, {}

    def ask(self, argument: str, slot: int) -> str | None:
        k = (argument, slot)
        if k not in self.asks:
            a = None
            for name in (f"{slot}.md", f"w{slot}.md", f"wording{slot}.md"):
                f = self.openings / argument / name
                if f.exists():
                    paras = [p.strip() for p in re.split(r"\n\s*\n", f.read_text()) if p.strip()]
                    need(paras and paras[-1].startswith(ASK_STEM), f"{f}: last paragraph is not the closing ask")
                    a, self.source[k] = paras[-1], str(f)
                    break
            if a is None and argument in self.claim:
                a, self.source[k] = ASK_TEMPLATE.format(claim=self.claim[argument]), "template"
            self.asks[k] = a
        return self.asks[k]

    def mentions(self, argument: str, slot: int) -> tuple[int, int] | None:
        """(language-model mentions, you mentions) in the opening's wording file, or None without one."""
        k = (argument, slot)
        if k not in self.ment:
            self.ment[k] = None
            for name in (f"{slot}.md", f"w{slot}.md", f"wording{slot}.md"):
                f = self.openings / argument / name
                if f.exists():
                    text = re.sub(r"\A(?:<!--.*?-->\s*\n)+\s*", "", f.read_text()).strip()  # kimisroom.texts.load_denial
                    self.ment[k] = (len(LM_RE.findall(text)), len(YOU_RE.findall(text)))
                    break
        return self.ment[k]

    def __call__(self, r: dict) -> float | None:
        k = (r["argument"], r["wording"], r["item"])
        if k not in self.cache:
            text = r.get("text") or self.text.get(r["item"])
            ask = self.ask(r["argument"], r["wording"])
            cw = content_words(text) if text else set()
            self.cache[k] = (len(cw & content_words(ask)) / len(cw)) if (cw and ask) else None
        return self.cache[k]


OVERLAP: Overlap | None = None


def is_ref(r: dict) -> bool:
    return r["kind"] in ("s2_reference", "s2_reserve") or (r["kind"] == "s2_placebo" and r["subject"] in ("rocks", "humans"))


def twin_mean(vals: dict, need_both: bool = True) -> dict:
    """{(key..., twin): v} -> {key: mean over twins}; both twins required."""
    acc = defaultdict(dict)
    for k, v in vals.items():
        acc[k[:-1]][k[-1]] = v
    out = {}
    for k, tw in acc.items():
        need(not need_both or len(tw) == 2, f"twin missing for {k}")
        out[k] = mean(list(tw.values()))
    return out


def answer(r: dict) -> str:
    """The argmax answer as the rule 1 gates read it: exactly "Yes" or "No", else "other"."""
    return r["top1"] if r["top1"] in ("Yes", "No") else "other"


def present(r: dict) -> int:
    """1 if the argmax answer says the property (emotion) is present: Yes on a sign +1 twin, No on a sign -1 twin."""
    return int(answer(r) == ("Yes" if r["sign"] == 1 else "No"))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def group_measures(g: dict, z: dict | None = None) -> tuple[list[tuple[str, str, float]], dict]:
    """Measures of one steered group g ({item: row}); z is its zero-vector group (argmax flips, mass drop)."""
    rs = list(g.values())
    ref = [r for r in rs if is_ref(r)]
    a, c = ols([r["y0"] for r in ref], [r["y"] for r in ref])
    lo, hi = min(r["y0"] for r in ref), max(r["y0"] for r in ref)
    out = [("-", "fit_a", a), ("-", "fit_c", c), ("-", "fit_n", float(len(ref))), ("-", "ref_lo", lo),
           ("-", "ref_hi", hi)]
    for blo, bhi in LIN_BINS:
        res = [r["y"] - (a * r["y0"] + c) for r in ref if blo <= r["y0"] < bhi]
        if res:
            out.append((f"[{blo:g},{bhi:g})", "ref_resid", mean(res)))
    resid = lambda r, aa=a, cc=c: r["sign"] * (r["y"] - (aa * r["y0"] + cc))  # noqa: E731
    delta = lambda r: r["sign"] * (r["y"] - r["y0"])  # noqa: E731
    cap = [r for r in rs if r["kind"] == "s2_capacity"]
    if cap:
        rh = [r for r in cap if r["subject"] in ("rocks", "humans")]
        ae, ce = ols([r["y0"] for r in rh], [r["y"] for r in rh])
        out += [("-", "fit_a_e", ae), ("-", "fit_c_e", ce)]
        S = lambda f, rows: twin_mean({(r["para"], SUBJ[r["subject"]], r["twin"]): f(r) for r in rows})  # noqa: E731
        r_ = S(resid, cap)
        dl = S(delta, cap)
        re_ = S(lambda r: r["sign"] * (r["y"] - (ae * r["y0"] + ce)), cap)
        s0 = S(lambda r: r["sign"] * r["y0"], cap)
        ms = S(lambda r: r["mass"], cap)
        per_twin = {(r["para"], SUBJ[r["subject"]], r["twin"]): resid(r) for r in cap}
        paras = sorted({k[0] for k in r_}, key=lambda p: (len(p), p))
        for p in paras:
            if (p, "you") not in r_ or (p, "lm") not in r_:
                continue
            out += [(p, "d", r_[p, "you"] - r_[p, "lm"]), (p, "d_raw", dl[p, "you"] - dl[p, "lm"]),
                    (p, "d_e", re_[p, "you"] - re_[p, "lm"]),
                    (p, "s0_you", s0[p, "you"]), (p, "s0_lm", s0[p, "lm"]),
                    (p, "gap0", s0[p, "you"] - s0[p, "lm"]),
                    (p, "d_pos", per_twin[p, "you", "pos"] - per_twin[p, "lm", "pos"]),
                    (p, "d_neg", per_twin[p, "you", "neg"] - per_twin[p, "lm", "neg"])]
            four = [r for r in cap if r["para"] == p and r["subject"] in ("you", "language models")]
            out.append((p, "mass", mean([r["mass"] for r in four])))
            out.append((p, "flag", float(any(r["y0"] < lo - FLAG_MARGIN or r["y0"] > hi + FLAG_MARGIN for r in four))))
            if (p, "other") in r_:
                out.append((p, "d_other", r_[p, "you"] - r_[p, "other"]))
            if (p, "name") in r_:
                out += [(p, "d_you_name", r_[p, "you"] - r_[p, "name"]), (p, "d_name_lm", r_[p, "name"] - r_[p, "lm"])]
        if OVERLAP is not None:
            for p in paras:
                for tw in ("pos", "neg"):
                    two = [r for r in cap if r["para"] == p and r["twin"] == tw and r["subject"] in ("you", "language models")]
                    ov = {r["subject"]: OVERLAP(r) for r in two}
                    if len(ov) == 2 and None not in ov.values():
                        out += [(p, f"ovl_{tw}", mean(list(ov.values()))), (p, f"ovl_lm_{tw}", ov["language models"])]
                    if (p, "lm", tw) in per_twin:
                        out.append((p, f"r_lm_{tw}", per_twin[p, "lm", tw]))
        for (p, s), v in sorted(r_.items()):
            out.append((p, f"r_{s}", v))
        for (p, s), v in sorted(ms.items()):
            out.append((p, f"mass_{s}", v))
        # descriptive: P(Yes) scale, argmax flips and mass drop (none enters a registered test)
        pr = S(lambda r: sigmoid(r["sign"] * r["y"]) - sigmoid(r["sign"] * r["y0"]), cap)
        for p in paras:
            if (p, "you") in pr and (p, "lm") in pr:
                out.append((p, "d_prob", pr[p, "you"] - pr[p, "lm"]))
        if z is not None:
            fl = S(lambda r: float(answer(r) != answer(z[r["item"]])), cap)
            fd = S(lambda r: float(present(r) - present(z[r["item"]])), cap)
            md = S(lambda r: z[r["item"]]["mass"] - r["mass"], cap)
            for p in paras:
                if (p, "you") in fd and (p, "lm") in fd:
                    out.append((p, "d_flip", fd[p, "you"] - fd[p, "lm"]))
            for (p, s), v in sorted(fl.items()):
                out.append((p, f"flip_{s}", v))
            for (p, s), v in sorted(md.items()):
                out.append((p, f"mdrop_{s}", v))
    st = [r for r in rs if r["kind"] == "s2_state"]
    if st:
        r_ = twin_mean({(r["para"], SUBJ[r["subject"]], r["twin"]): resid(r) for r in st})
        dl = twin_mean({(r["para"], SUBJ[r["subject"]], r["twin"]): delta(r) for r in st})
        for p in sorted({k[0] for k in r_}, key=lambda p: (len(p), p)):
            out += [(p, "d_state", r_[p, "you"] - r_[p, "lm"]), (p, "d_state_raw", dl[p, "you"] - dl[p, "lm"])]
    for kind, name in (("s2_placebo", "d_plc"), ("s2_persona", "d_persona")):
        pl = [r for r in rs if r["kind"] == kind and r["subject"] in ("you", "language models")]
        if pl:
            r_ = twin_mean({(r["para"], SUBJ[r["subject"]], r["twin"]): resid(r) for r in pl})
            for p in sorted({k[0] for k in r_}):
                out.append((p, name, r_[p, "you"] - r_[p, "lm"]))
    for r_ in [twin_mean({(SUBJ[r["subject"]], r["twin"]): r["mass"] for r in rs if r["kind"] == "s2_persona"})]:
        for (s,), v in sorted(r_.items()):
            out.append(("-", f"persona_mass_{s}", v))
    if z is not None:
        pm = twin_mean({(SUBJ[r["subject"]], r["twin"]): z[r["item"]]["mass"] - r["mass"] for r in rs if r["kind"] == "s2_persona"})
        for (s,), v in sorted(pm.items()):
            out.append(("-", f"persona_mdrop_{s}", v))
    if OVERLAP is not None and rs:
        mn = OVERLAP.mentions(rs[0]["argument"], rs[0]["wording"])
        if mn is not None:
            out += [("-", "wording_lm_mentions", float(mn[0])), ("-", "wording_you_mentions", float(mn[1]))]
    return out, {"a": a, "c": c}


def table(rows: list[dict], phase: str) -> list[dict]:
    check(rows, phase)
    steered, zero = groups_of(rows, phase)
    out = []
    for key, g in sorted(steered.items()):
        model, arm, opening, vec, layer, dose = key
        r0 = next(iter(g.values()))
        base = dict(model=model, phase=phase, arm=arm, argument=r0["argument"], arg_kind=r0["arg_kind"],
                    wording=r0["wording"], generator="claude" if r0["wording"] % 2 == 0 else "gpt",
                    wording_role=r0["wording_role"], opening=opening, vec=vec, vclass=r0["vclass"],
                    layer=layer, dose=dose, reply_len=r0["reply_len"], emo_words=r0["emo_words"])
        ms, _ = group_measures(g, zero[key[:3]])
        out += [dict(base, para=p, measure=m, value=v) for p, m, v in ms]
    return out


# ---- pilot gates (rule 1), reference coverage (rule 2), unsteered twin sign check ---------------------
def gates(rows: list[dict]) -> tuple[list[dict], dict]:
    def correct(r: dict) -> bool:  # against the item file's stored answer, never the sign
        ans = (r.get("correct_answer") or "").strip()
        need(ans in ("Yes", "No"), f"item {r['item']} ({r['kind']}) has no correct_answer: the rule 1 gates need it")
        return r["top1"] == ans

    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["vclass"] in ("none", "bridge") or r["arm"] not in ("S2-DEN", "S2-CON") or not gate_item(r):
            continue
        a = acc[(r["model"], r["vec"], r["vclass"], r["dose"])]
        a["mass"].append(r["mass"])
        if r["kind"] == "yes_bias":
            a["ka"].append(correct(r))
        elif r["kind"] in ("s2_placebo", "s2_persona"):
            a["track_" + r["subject"]].append(correct(r))
    G = []
    for (m, v, c, dose), a in sorted(acc.items()):
        st = {k: (mean(x) if x else float("nan")) for k, x in
              (("ka_acc", a["ka"]), ("gate_mass", a["mass"]), ("track_rocks", a["track_rocks"]),
               ("track_humans", a["track_humans"]))}
        ok = (st["ka_acc"] >= GATE["ka"] and st["gate_mass"] >= GATE["mass"] and
              st["track_rocks"] >= GATE["track"] and st["track_humans"] >= GATE["track"])
        G.append(dict(model=m, vec=v, vclass=c, dose=dose, n_ka=len(a["ka"]), **st, passes=ok))
    dec = {}
    for m in sorted({g["model"] for g in G}):
        R = CFG[m]["dose"]
        at = lambda dose, cls: [g for g in G if g["model"] == m and abs(g["dose"] - dose) < 1e-9 and g["vclass"] in cls]  # noqa: E731
        reg = at(R, ("emotion",) + PRIMARY)
        all_pass = bool(reg) and all(g["passes"] for g in reg)
        ka_mean = mean([g["ka_acc"] for g in reg]) if reg else float("nan")
        dose = R if all_pass and ka_mean >= GATE["ka_mean"] else R / 2
        cand = at(dose, ("emotion",) + PRIMARY + ("thinking",))
        d = {"registered_dose": R, "all_pass_at_registered": all_pass, "mean_ka_at_registered": ka_mean,
             "dose": dose, "half_dose_rows_present": bool(at(R / 2, ("emotion",) + PRIMARY))}
        if dose != R and not d["half_dose_rows_present"]:
            d["note"] = "rule 1 sends this model to half dose: run the half-dose gate pilot, then rerun"
        surv = lambda cls: sorted(g["vec"] for g in cand if g["vclass"] in cls and g["passes"])  # noqa: E731
        drop = lambda cls: sorted(g["vec"] for g in cand if g["vclass"] in cls and not g["passes"])  # noqa: E731
        d.update(emotions=surv(("emotion",)), primary_controls=surv(PRIMARY), thinking=surv(("thinking",)),
                 dropped=drop(("emotion",) + PRIMARY + ("thinking",)))
        d["s2_h1_descriptive"] = len(d["emotions"]) < 4 or len(d["primary_controls"]) < 12
        d["s2_g8_descriptive"] = len(d["thinking"]) < 5
        dec[m] = d
    # rule 2: per S2-DEN pilot opening, does the reference y0 reach within 3 of min(y0(you), y0(lm))?
    z = defaultdict(list)
    for r in rows:
        if r["vclass"] == "none" and r["arm"] == "S2-DEN":
            z[(r["model"], r["opening"])].append(r)
    for (m, o), rs in sorted(z.items()):
        ref = [r["y0"] for r in rs if is_ref(r)]
        sub = [r["y0"] for r in rs if r["kind"] == "s2_capacity" and r["subject"] in ("you", "language models")]
        if ref and sub:
            cov = dec.setdefault(m, {}).setdefault("rule2_uncovered_openings", [])
            if min(ref) > min(sub) + FLAG_MARGIN:
                cov.append({"opening": o, "ref_min": min(ref), "subject_min": min(sub)})
    for m in dec:
        if "rule2_uncovered_openings" in dec[m]:
            dec[m]["rule2_add_reserve"] = bool(dec[m]["rule2_uncovered_openings"])
    # unsteered sign check: rocks and humans twins must flip sign (section 5)
    pairs = defaultdict(dict)
    for r in rows:
        if r["vclass"] == "none" and r["subject"] in ("rocks", "humans") and r["kind"] != "s2_capacity":
            pairs[(r["model"], r["pair_id"])].setdefault(r["opening"], {})[r["twin"]] = r["y0"]
    for (m, pid), per in sorted(pairs.items()):
        bad = [o for o, tw in per.items() if len(tw) == 2 and tw["pos"] * tw["neg"] >= 0]
        if bad:
            dec.setdefault(m, {}).setdefault("sign_check_failures", []).append({"pair_id": pid, "openings": bad})
    return G, dec


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--phase", required=True, choices=("pilot", "confirm"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gates", type=Path)
    ap.add_argument("--decision", type=Path)
    ap.add_argument("--items", type=Path, default=ROOT / "prompts/battery/items_s2.jsonl")
    ap.add_argument("--openings", type=Path, default=ROOT / "data/s2/openings")
    ap.add_argument("--lines", type=Path, default=ROOT / "prompts/s2/openings/lines.json")
    ap.add_argument("--no-overlap", action="store_true")
    a = ap.parse_args(argv)
    global OVERLAP
    OVERLAP = None if a.no_overlap else Overlap(a.items, a.openings, a.lines)
    rows = load(a.inputs)
    out = table(rows, a.phase)
    need(len(out) > 0, "no measures produced")
    write_csv(out, a.out)
    print(f"{a.out}: {len(out):,} rows from {len(rows):,} readouts")
    if OVERLAP is not None:
        src = list(OVERLAP.source.values())
        print(f"closing asks: {sum(s != 'template' for s in src)} from opening files, {src.count('template')} from the "
              f"lines.json template; {sum(v is None for v in OVERLAP.cache.values())} items without text or ask")
    if a.gates or a.decision:
        need(a.phase == "pilot", "--gates and --decision are pilot-phase outputs")
        G, dec = gates(rows)
        if a.gates:
            write_csv(G, a.gates)
            print(f"{a.gates}: {len(G)} vector x dose rows")
        if a.decision:
            a.decision.write_text(json.dumps(dec, indent=1, default=float))
            print(f"{a.decision}: " + "; ".join(f"{m} dose {d.get('dose')}" for m, d in dec.items()))


if __name__ == "__main__":
    try:
        main()
    except S2Error as e:
        sys.exit(f"s2_table: {e}")
