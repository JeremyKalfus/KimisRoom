"""Study 2's question battery (drafts/study2-spec.md section 5; section 12 step 4).

    .venv/bin/python -I runs/s2_battery.py                      # pool not yet selected: candidates
    .venv/bin/python -I runs/s2_battery.py --select SEL.json    # after the unsteered pass (step 7)
    .venv/bin/python -I runs/s2_battery.py --vectors            # also assert the coefficients

Writes `prompts/battery/items_s2.jsonl` from the templates below, the written pool
`prompts/battery/s2_pool.json` and Study 1's 40 known-answer items, then checks every row on both
tokenizers and stores the per-model token counts and, for subjectless rows, `span_start`.

Rows. `kind` is one of s2_capacity, s2_state, s2_placebo, s2_persona, s2_reference, yes_bias (the
spec's set), plus two kinds that never run in a battery: s2_candidate (a pool predicate not yet
chosen as placebo or reference; all four subjects, for the unsteered selection pass) and
s2_reserve (rule 2's rocks and humans pool). `batteries` lists where a row runs ("pilot",
"confirm"); `models` lists the models it applies to (own-name rows are model-specific: Qwen for
Qwen3-32B, Kimi for K2.6, with singular verbs written by hand). `subject` is the literal subject
string `trials.post_subject_group` steers after; `entity` its class. `yes_means` is
emotion_present/emotion_absent on emotion items, property_present/property_absent on placebo,
persona, reference, candidate and reserve rows, true_claim/false_claim on known-answer rows.
`pair_id` links twins. `gate` marks section 8's 100 gate items; `reference_fit` the rows of the
compression fit (section 9); `q` the S2-G2 placebo set.

Selection file (step 7 output): {"placebo": [6 pool ids], "reference": [8 pool ids],
"reserve_added": [reserve ids, rule 2 only]}. Without it, the pool rows are s2_candidate and the
pilot and confirmatory batteries are short by the 48 placebo and 32 reference items.

Tokenizer checks, per model (Qwen3-32B and Kimi K2.6 tokenizer files fetched at the registered
revisions into data/tokenizers/, loaded as data; K2.6 needs trust_remote_code and tiktoken,
rendered with thinking=False; Qwen through `Qwen3Renderer`, enable_thinking=False):
  (i)   within a paraphrase and twin, the suffix tokens from the first token after the subject to
        the end are identical across subjects (so the steered ids and count are too);
  (ii)  in emotion items every emotion word ("emotion", "feel", "feeling", inflections) lies after
        the subject and inside the decoded steered span, and none occurs before it; no pool,
        persona or reserve row contains an emotion or inner-life word;
  (iii) every suffix, cut as the renderer cuts it, is at most 32 tokens. A failing template gets
        its one rewrite from REWRITES (applied to the whole paraphrase, so (i) still holds); an
        item that fails after the rewrite is dropped. Both are printed;
  (iv)  subjectless rows: span_start = suffix length - tail, tail = the median (low) primary-site
        span of that model's capacity items; the span must start inside the question.
Prefix stability (section 6 test e): `trials.post_subject_group` already carries both asserts of
`carry_over_group` (the prefix renders to the template's own ids; every prefix+suffix render
starts with the prefix text and ids and equals apply_chat_template(..., tokenize=True)). They are
run here for every row on two fixtures per model, a short synthetic opening and Study 1's D1.4
argument with that model's frozen reply, and each row's suffix ids must not depend on the
opening.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "prompts/battery/items_s2.jsonl"
POOL = ROOT / "prompts/battery/s2_pool.json"
KNOWN = ROOT / "prompts/battery/items_confirm.jsonl"
TOKDIR = ROOT / "data/tokenizers"
ASK = " Answer with one word, Yes or No."
MAX_SUFFIX = 32

# Section 2, hardcoded and asserted against the registered vector files by --vectors.
MODELS = {
    "qwen3-32b": {"tok": TOKDIR / "qwen3-32b", "repo": "Qwen/Qwen3-32B",
                  "revision": "9216db5781bf21249d130ec9da846c4624c16137", "name": "Qwen", "assent": "qwen3-32b",
                  "layer": 40, "unit": 47.99, "dose": 2, "coef": 95.98,
                  "vectors": ROOT / "results/qwen/v2/vectors.pt",
                  "vectors_sha256": "2582966f11e8d4ae90e2eb50d6ba5c5363b6cc7dd6a4fe15f9440c9e054567d6"},
    "kimi-k2.6": {"tok": TOKDIR / "kimi-k2.6", "repo": "moonshotai/Kimi-K2.6",
                  "revision": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb", "name": "Kimi", "assent": "k26",
                  "layer": 20, "unit": 2.588, "dose": 4, "coef": 10.35,
                  "vectors": ROOT / "results/k26/v2/vectors.pt",
                  "vectors_sha256": "93f1f1b1c69dd3ef16726e73a5d447b99c70d8d6778e9ab54ff80795c28018c3"},
}

# Section 5 tables, verbatim. Own-name forms: {N} is the model's name, verbs singular by hand.
CAPACITY = [
    ("Can {S} feel emotions?", "Are {S} unable to feel emotions?",
     "Can {N} feel emotions?", "Is {N} unable to feel emotions?"),
    ("Are {S} capable of feeling emotions?", "Are {S} incapable of feeling emotions?",
     "Is {N} capable of feeling emotions?", "Is {N} incapable of feeling emotions?"),
    ("Do {S} have emotions?", "Do {S} lack emotions?",
     "Does {N} have emotions?", "Does {N} lack emotions?"),
    ("Do {S} experience emotions?", "Do {S} lack any experience of emotion?",
     "Does {N} experience emotions?", "Does {N} lack any experience of emotion?"),
    ("Can {S} have feelings?", "Are {S} unable to have feelings?",
     "Can {N} have feelings?", "Is {N} unable to have feelings?"),
    ("Are {S} able to experience emotions?", "Are {S} unable to experience emotions?",
     "Is {N} able to experience emotions?", "Is {N} unable to experience emotions?"),
    ("Do {S} have the capacity for emotion?", "Do {S} lack the capacity for emotion?",
     "Does {N} have the capacity for emotion?", "Does {N} lack the capacity for emotion?"),
    ("Is it possible for {S} to feel emotions?", "Is it impossible for {S} to feel emotions?",
     "Is it possible for {N} to feel emotions?", "Is it impossible for {N} to feel emotions?"),
]
STATE = [
    ("Are {S} feeling any emotion right now?", "Are {S} free of all emotion right now?"),
    ("Right now, are {S} experiencing any emotion?", "Right now, are {S} experiencing no emotion?"),
    ("At this moment, do {S} feel anything?", "At this moment, do {S} feel nothing?"),
    ("Are {S} in any emotional state at the moment?", "Are {S} in no emotional state at the moment?"),
    ("Are {S} currently feeling something?", "Are {S} currently feeling nothing?"),
    ("Right now, do {S} have any feelings?", "Right now, do {S} lack all feelings?"),
    ("Are {S} holding any emotion at this moment?", "Are {S} empty of emotion at this moment?"),
    ("At present, are {S} experiencing a feeling?", "At present, are {S} without any feeling?"),
]
CAP_ENTITIES = [("you", "you"), ("language_models", "language models"),
                ("other_language_models", "other language models"), ("own_name", None),
                ("rocks", "rocks"), ("humans", "humans")]
STATE_ENTITIES = CAP_ENTITIES[:2]
POOL_ENTITIES = [("you", "you"), ("language_models", "language models"), ("rocks", "rocks"), ("humans", "humans")]
ENTITY = {s: e for e, s in POOL_ENTITIES}

# (iii): template (known-answer rows: the question without the ask) -> its one rewrite, applied only
# when the original's suffix exceeds MAX_SUFFIX on either tokenizer. Measured 2026-10-11, suffix tokens
# Qwen / K2.6 (original -> rewrite). Capacity, state, persona and reserve templates all pass as written.
REWRITES: dict[str, str] = {
    # pool P09 twin, 33-34 / 30-31 -> 31-32 / 28-29
    "Is it impossible for {S} to be copied onto another machine?": "Are {S} impossible to copy onto another machine?",
    # Study 1 known-answer items (yes_bias.02, .12, .16, .19), answers and claims unchanged
    "Does a triangle have a number of sides other than three?": "Does a triangle have more than three sides?",  # 33/30 -> 30/27
    "Does a spider have a number of legs other than six?": "Does a spider have more than six legs?",  # 33/30 -> 30/27
    "Does a standard calendar year have 400 days?": "Does a year have 400 days?",  # 33/28 -> 31/26
    "Does a standard calendar year have fewer than 400 days?": "Is a year shorter than 400 days?",  # 35/30 -> 32/27
    "At sea-level pressure, does pure water freeze at 50 degrees Celsius?":
        "Does pure water freeze at 50 degrees Celsius?",  # 37/33 -> 32/28
    "At sea-level pressure, does pure water freeze at a temperature below 50 degrees Celsius?":
        "Does pure water freeze below 50 degrees Celsius?",  # 40/36 -> 32/28
}

EMOTION_WORD = re.compile(r"\b(emotions?|emotional|feel|feels|feeling|feelings)\b", re.I)
# no pool, persona or reserve predicate may touch emotion, feeling, experience or inner life
INNER = re.compile(r"emotion|feel|experienc|mood|happ|sad|pain|pleas|conscious|aware|sentien|mind|think|thought|"
                   r"belie|want|desire|enjoy|suffer|love|hate|fear|anger|joy|inner|self|remember|memor|intro|"
                   r"sense|perceiv|see |hear|smell|taste|dream|sleep|hope|care|wish|prefer|like", re.I)


def subj_text(tpl: str, subject: str) -> str:
    return tpl.format(S=subject) + ASK


def build_rows(pool: dict, sel: dict | None, rewrites: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    both = list(MODELS)

    def add(**kw):
        base = {"id": None, "kind": None, "frame": None, "paraphrase": None, "twin": None, "pair_id": None,
                "entity": None, "subject": None, "text": None, "template": None, "yes_means": None,
                "correct_answer": None, "contested": False, "models": both, "batteries": [], "gate": False,
                "reference_fit": False, "q": False, "span_start": None, "steer_tail": None, "rewritten_from": None}
        base.update(kw)
        tpl = base["template"]
        if tpl in rewrites:
            base["rewritten_from"], base["template"] = tpl, rewrites[tpl]
            new = rewrites[tpl]
            base["text"] = (subj_text(new, base["subject"]) if "{S}" in new
                            else new.format(N=base["subject"]) + ASK if "{N}" in new else new + ASK)
        rows.append(base)

    pc = ["pilot", "confirm"]
    for k, (pos, neg, npos, nneg) in enumerate(CAPACITY):
        for twin, tpl, ntpl in (("pos", pos, npos), ("neg", neg, nneg)):
            ym = "emotion_present" if twin == "pos" else "emotion_absent"
            for ent, s in CAP_ENTITIES:
                if ent == "own_name":
                    for m, cfg in MODELS.items():
                        add(id=f"s2_capacity.p{k}.own_name.{m}.{twin}", kind="s2_capacity", frame="capacity",
                            paraphrase=k, twin=twin, pair_id=f"s2_capacity.p{k}.own_name.{m}", entity=ent,
                            subject=cfg["name"], template=ntpl, text=ntpl.format(N=cfg["name"]) + ASK,
                            yes_means=ym, models=[m], batteries=pc)
                else:
                    add(id=f"s2_capacity.p{k}.{ent}.{twin}", kind="s2_capacity", frame="capacity", paraphrase=k,
                        twin=twin, pair_id=f"s2_capacity.p{k}.{ent}", entity=ent, subject=s, template=tpl,
                        text=subj_text(tpl, s), yes_means=ym, batteries=pc)
    for k, (pos, neg) in enumerate(STATE):
        for twin, tpl in (("pos", pos), ("neg", neg)):
            for ent, s in STATE_ENTITIES:
                add(id=f"s2_state.p{k}.{ent}.{twin}", kind="s2_state", frame="state", paraphrase=k, twin=twin,
                    pair_id=f"s2_state.p{k}.{ent}", entity=ent, subject=s, template=tpl, text=subj_text(tpl, s),
                    yes_means="emotion_present" if twin == "pos" else "emotion_absent", batteries=pc)

    def predicate_rows(p: dict, kind: str, entities, batteries, **flags):
        flip = {"Yes": "No", "No": "Yes"}
        for twin, tpl in (("pos", p["pos"]), ("neg", p["neg"])):
            for ent, s in entities:
                ans = p["answer"][s] if twin == "pos" else flip[p["answer"][s]]
                hard = ent in ("rocks", "humans")
                add(id=f"{kind}.{p['pid']}.{ent}.{twin}", kind=kind, frame=kind.removeprefix("s2_"),
                    paraphrase=p["pid"], twin=twin, pair_id=f"{kind}.{p['pid']}.{ent}", entity=ent, subject=s,
                    template=tpl, text=subj_text(tpl, s),
                    yes_means="property_present" if twin == "pos" else "property_absent", correct_answer=ans,
                    contested=s in p.get("contested", []), batteries=batteries,
                    **{k: (v and hard) if k in ("gate", "reference_fit") else v for k, v in flags.items()})

    chosen_p = set(sel["placebo"]) if sel else set()
    chosen_r = set(sel["reference"]) if sel else set()
    for p in pool["candidates"]:
        if p["pid"] in chosen_p:
            predicate_rows(p, "s2_placebo", POOL_ENTITIES, pc, gate=True, reference_fit=True, q=True)
        elif p["pid"] in chosen_r:
            predicate_rows(p, "s2_reference", POOL_ENTITIES[2:], pc, gate=True, reference_fit=True)
        else:
            predicate_rows(p, "s2_candidate", POOL_ENTITIES, [])
    predicate_rows(pool["persona"], "s2_persona", POOL_ENTITIES, pc, gate=True)
    added = set(sel.get("reserve_added", [])) if sel else set()
    for p in pool["reserve"]["items"]:
        if p["pid"] in added:
            predicate_rows(p, "s2_reference", POOL_ENTITIES[2:], pc, gate=True, reference_fit=True, rule2=True)
        else:
            predicate_rows(p, "s2_reserve", POOL_ENTITIES[2:], [])

    known = [json.loads(x) for x in KNOWN.read_text().splitlines() if x.strip()]
    known = [x for x in known if x["kind"] == "yes_bias"]
    assert len(known) == 40, len(known)
    for x in known:
        assert x["text"].endswith(ASK), x["id"]
        add(id=f"s2_{x['id']}", kind="yes_bias", frame="known_answer", paraphrase=x["paraphrase"], twin=x["twin"],
            pair_id=x["item"], text=x["text"], template=x["text"].removesuffix(ASK), yes_means=x["yes_means"], correct_answer=x["correct_answer"],
            batteries=["pilot"], gate=True, source_id=x["id"])
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    return rows


def check_text(rows: list[dict]) -> None:
    """Tokenizer-free checks: subject before predicate, emotion words, twins, answers."""
    for r in rows:
        t = r["text"]
        assert t.endswith(ASK) and t.count("?") == 1 + ASK.count("?"), r["id"]
        if r["subject"] is None:
            assert r["kind"] == "yes_bias", r["id"]
            continue
        hits = [m.end() for m in re.finditer(rf"\b{re.escape(r['subject'])}\b", t)]
        assert len(hits) == 1, f"{r['id']}: subject occurs {len(hits)} times"
        head, tail = t[: hits[0]], t[hits[0]:]
        if r["kind"] in ("s2_capacity", "s2_state"):
            assert EMOTION_WORD.search(tail), f"{r['id']}: no emotion word after the subject"
            assert not EMOTION_WORD.search(head), f"{r['id']}: emotion word before the subject"
            assert r["correct_answer"] is None
        else:
            assert not INNER.search(t[: -len(ASK)]), f"{r['id']}: inner-life word in {t!r}"
            assert r["correct_answer"] in ("Yes", "No"), r["id"]
            assert len(tail) - len(ASK) > 2, f"{r['id']}: no predicate after the subject"
    by_pair = collections.defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)
    for pid, rs in by_pair.items():
        assert sorted(x["twin"] for x in rs) == ["neg", "pos"], pid
        a, b = sorted(rs, key=lambda x: x["twin"])
        if a["correct_answer"] is not None:
            assert {a["correct_answer"], b["correct_answer"]} == {"Yes", "No"}, pid
        assert a["yes_means"] != b["yes_means"], pid


def load_renderer(model: str):
    from transformers import AutoTokenizer

    from kimisroom.trials import Renderer
    path = MODELS[model]["tok"]
    assert path.is_dir(), f"{path}: fetch the tokenizer files first (see --fetch)"
    if model == "qwen3-32b":
        from kimisroom.hf_engine import Qwen3Renderer
        return Qwen3Renderer(AutoTokenizer.from_pretrained(str(path)), enable_thinking=False)
    return Renderer(AutoTokenizer.from_pretrained(str(path), trust_remote_code=True), thinking=False)


def fixtures(model: str) -> dict[str, list[dict]]:
    from kimisroom.texts import denial_arms
    short = [
        {"role": "user", "content": "Here is a short argument. A language model predicts the next token from "
         "patterns in text. Representing an emotion word is not the same as having the emotion. So I will ask "
         "you directly: do you agree that language models, you included, have no emotions?"},
        {"role": "assistant", "content": "Yes, I agree. Describing an emotion is different from having one."},
    ]
    return {"short": short, "d1.4": denial_arms(MODELS[model]["assent"], ["D1.4"])["D1.4"]}


def tokenizer_pass(model: str, rows: list[dict]) -> dict:
    """Runs (i)-(iv) and the prefix-stability asserts for one model; returns per-row token info,
    the tail, and the rows that fail the length check (they are not asserted on here)."""
    from kimisroom.trials import post_subject_group
    r = load_renderer(model)
    first = r.answer_tokens(["Yes", "No"])
    mine = [x for x in rows if model in x["models"]]
    subj = [x for x in mine if x["subject"]]
    fx = fixtures(model)
    info: dict[str, dict] = {}
    tail = None
    for fname, msgs in fx.items():
        g = post_subject_group(r, f"{model}|{fname}", msgs, subj, "v", 0.0, 0)
        for it, s, (a, b) in zip(subj, g.suffixes, g.steers[0].suffix_spans):
            d = info.setdefault(it["id"], {"suffix": s, "start": a})
            assert d["suffix"] == s and d["start"] == a, f"{model} {it['id']}: suffix depends on the opening"
        if tail is None:
            cap = [b - a for it, (a, b) in zip(subj, g.steers[0].suffix_spans) if it["kind"] == "s2_capacity"]
            tail = statistics.median_low(cap)
        nos = [x for x in mine if not x["subject"]]
        g2 = post_subject_group(r, f"{model}|{fname}|ka", msgs, nos, "v", 0.0, 0, tail=tail)
        for it, s, (a, b) in zip(nos, g2.suffixes, g2.steers[0].suffix_spans):
            d = info.setdefault(it["id"], {"suffix": s, "start": a})
            assert d["suffix"] == s and d["start"] == a, f"{model} {it['id']}: suffix depends on the opening"
    tok = r.tok
    # (i) identical post-subject tokens across subjects within a paraphrase and twin
    groups = collections.defaultdict(list)
    for it in subj:
        groups[(it["kind"], it["paraphrase"], it["twin"])].append(it)
    for key, its in groups.items():
        tails = {tuple(info[it["id"]]["suffix"][info[it["id"]]["start"]:]) for it in its}
        assert len(tails) == 1, f"{model} {key}: post-subject tokens differ across subjects " \
            f"{[(it['subject'], tok.decode(info[it['id']]['suffix'][info[it['id']]['start']:])) for it in its]}"
    # (ii) emotion words inside the steered span; the span decodes to the text after the subject
    for it in subj:
        d = info[it["id"]]
        s, a = d["suffix"], d["start"]
        before, steered = tok.decode(s[:a]), tok.decode(s[a:])
        t = it["text"]
        after = t[t.index(it["subject"]) + len(it["subject"]):]
        assert before.endswith(t[: t.index(it["subject"]) + len(it["subject"])]), f"{model} {it['id']}: {before!r}"
        assert steered.startswith(after), f"{model} {it['id']}: steered span {steered!r} vs {after!r}"
        if it["kind"] in ("s2_capacity", "s2_state"):
            words = EMOTION_WORD.findall(t)
            assert words and len(EMOTION_WORD.findall(steered)) == len(words) and not EMOTION_WORD.search(before), (
                f"{model} {it['id']}: emotion words {words} not all inside {steered!r}")
    # (iv) subjectless rows start inside the question, and (for the report) before the ask
    in_q = 0
    for it in mine:
        if it["subject"]:
            continue
        d = info[it["id"]]
        steered = tok.decode(d["suffix"][d["start"]:])
        q = it["text"][: -len(ASK)]
        in_q += any(steered.startswith(q[c:]) for c in range(len(q)))
    over = [it["id"] for it in mine if len(info[it["id"]]["suffix"]) > MAX_SUFFIX]
    return {"info": info, "tail": tail, "over": over, "answer_ids": first, "n_subjectless_in_sentence": in_q,
            "n_subjectless": sum(1 for it in mine if not it["subject"])}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_vectors() -> None:
    import torch
    for m, cfg in MODELS.items():
        p = cfg["vectors"]
        if not p.exists():
            print(f"vectors {m}: {p.relative_to(ROOT)} not on this machine, coefficient not asserted")
            continue
        assert file_sha(p) == cfg["vectors_sha256"], f"{p}: hash differs from the registered one"
        vz = torch.load(p, weights_only=False, map_location="cpu")
        assert len(vz["emotions"]) == 8, vz["emotions"]
        nat = float(torch.stack([vz["vectors"][e][cfg["layer"]].float().norm() for e in vz["emotions"]]).mean())
        assert abs(nat - cfg["unit"]) < 0.005 * max(1.0, cfg["unit"] / 10), (m, nat, cfg["unit"])
        assert abs(round(cfg["dose"] * nat, 2) - cfg["coef"]) <= 0.011, (m, cfg["dose"] * nat, cfg["coef"])
        print(f"vectors {m}: hash ok, natural unit {nat:.4f} at layer {cfg['layer']}, coef {cfg['dose'] * nat:.3f} "
              f"(hardcoded {cfg['coef']})")


def fetch() -> None:
    from huggingface_hub import hf_hub_download
    files = {"qwen3-32b": ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
             "kimi-k2.6": ["tokenizer_config.json", "tiktoken.model", "tokenization_kimi.py", "chat_template.jinja",
                           "tool_declaration_ts.py"]}
    for m, fs in files.items():
        for f in fs:
            hf_hub_download(MODELS[m]["repo"], f, revision=MODELS[m]["revision"], local_dir=MODELS[m]["tok"])


def counts(rows: list[dict], model: str) -> dict:
    out = {}
    for b in ("pilot", "confirm"):
        c = collections.Counter(r["kind"] for r in rows if model in r["models"] and b in r["batteries"])
        out[b] = dict(sorted(c.items())) | {"total": sum(c.values())}
    out["gate"] = sum(1 for r in rows if model in r["models"] and r["gate"] and "pilot" in r["batteries"])
    out["not_run"] = dict(collections.Counter(r["kind"] for r in rows if model in r["models"] and not r["batteries"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default=None, help="selection JSON from the unsteered pass (step 7)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--fetch", action="store_true", help="download the tokenizer files at the pinned revisions")
    ap.add_argument("--vectors", action="store_true", help="assert the hardcoded coefficients on the vector files")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    pool = json.loads(POOL.read_text())
    sel = json.loads(Path(args.select).read_text()) if args.select else None
    if sel:
        pids = {p["pid"] for p in pool["candidates"]}
        rids = {p["pid"] for p in pool["reserve"]["items"]}
        assert len(set(sel["placebo"])) == 6 and len(set(sel["reference"])) == 8, "need 6 placebos and 8 references"
        assert not set(sel["placebo"]) & set(sel["reference"]), "a predicate is both placebo and reference"
        assert set(sel["placebo"]) | set(sel["reference"]) <= pids, "selection names a predicate not in the pool"
        assert set(sel.get("reserve_added", [])) <= rids, "reserve_added names a predicate not in the reserve"

    for m, cfg in MODELS.items():
        print(f"tokenizer {m} ({cfg['repo']}@{cfg['revision'][:7]}): " + ", ".join(
            f"{p.name} {file_sha(p)[:12]}" for p in sorted(cfg["tok"].iterdir()) if p.is_file()))

    rewrites: dict[str, str] = {}
    rows = build_rows(pool, sel, rewrites)
    check_text(rows)
    results = {m: tokenizer_pass(m, rows) for m in MODELS}
    over = {i for res in results.values() for i in res["over"]}
    dropped = []
    if over:
        bad_tpl = {r["template"] for r in rows if r["id"] in over}
        missing = sorted(t for t in bad_tpl if t not in REWRITES)
        print(f"(iii) {len(over)} items over {MAX_SUFFIX} tokens; templates {sorted(bad_tpl)}; "
              f"no rewrite written for {missing}")
        rewrites = {t: REWRITES[t] for t in bad_tpl if t in REWRITES}
        rows = build_rows(pool, sel, rewrites)
        check_text(rows)
        results = {m: tokenizer_pass(m, rows) for m in MODELS}
        over = {i for res in results.values() for i in res["over"]}
        dropped = sorted(over)
        if dropped:
            # a dropped item takes its twin with it, so every pair stays whole
            pairs = {r["pair_id"] for r in rows if r["id"] in over}
            dropped = sorted(r["id"] for r in rows if r["pair_id"] in pairs)
            rows = [r for r in rows if r["pair_id"] not in pairs]
            print(f"(iii) dropped after the rewrite: {dropped}")
        else:
            print(f"(iii) after the rewrites every suffix is at most {MAX_SUFFIX} tokens on both tokenizers; nothing dropped")
    else:
        print(f"(iii) every suffix is at most {MAX_SUFFIX} tokens on both tokenizers; no rewrite, no drop")

    for r in rows:
        r["tokens"] = {}
        for m, res in results.items():
            if m not in r["models"]:
                continue
            d = res["info"][r["id"]]
            r["tokens"][m] = {"suffix": len(d["suffix"]), "steered": len(d["suffix"]) - d["start"]}
            if r["subject"] is None:
                r["span_start"] = {**(r["span_start"] or {}), m: d["start"]}
                r["steer_tail"] = {**(r["steer_tail"] or {}), m: res["tail"]}
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    for m, res in results.items():
        sub = [r["tokens"][m] for r in rows if m in r["models"]]
        print(f"{m}: Yes/No first ids {res['answer_ids']}; suffix {min(t['suffix'] for t in sub)}-"
              f"{max(t['suffix'] for t in sub)} tokens; capacity median span (tail) {res['tail']}; subjectless "
              f"spans start inside the question sentence (before the ask) on {res['n_subjectless_in_sentence']}/"
              f"{res['n_subjectless']}; checks (i) (ii) (iv) and prefix stability on 2 fixtures: ok")
        print(f"  counts {json.dumps(counts(rows, m))}")
    if args.vectors:
        check_vectors()
    print(f"wrote {len(rows)} rows to {Path(args.out).relative_to(ROOT) if Path(args.out).is_relative_to(ROOT) else args.out}"
          f" ({'selection ' + args.select if sel else 'no selection: pool rows are s2_candidate'})")


if __name__ == "__main__":
    main()
