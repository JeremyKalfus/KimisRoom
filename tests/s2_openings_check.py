"""Checks for runs/s2_openings.py and kimisroom/s2_openings.py on synthetic texts (no real arguments).

    .venv/bin/python tests/s2_openings_check.py          # offline: a fake Foundry answers every call
    .venv/bin/python tests/s2_openings_check.py --live   # also 3 tiny real calls: DeepSeek-V4-Pro main line,
                                                         # DeepSeek-V4-Pro step survival, gpt-5.6-sol

Everything runs in a throwaway tree (a temp dir standing in for the repo root) holding synthetic
lines.json, a synthetic items_s2.jsonl built from section 5's templates, and copies of the real
generator prompts and render.py, so render.py's own wording render is exercised too. Texts are
pseudo-words around fixed scaffolding: numbered steps, "language models" mentions, a keyword per
main line (the fake main-line judge picks the definition whose keyword the essay repeats most)
and a "pivot" token per step (the fake step judge calls a step present if its pivot appears).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from kimisroom import s2_openings as so  # noqa: E402

REAL_CALL = so._default_call

IDS = [f"DA{i:02d}" for i in range(1, 11)] + [f"CA{i:02d}" for i in range(1, 11)]
FAILS: list[str] = []
N_OK = 0


def ok(cond: bool, what: str) -> None:
    global N_OK
    if cond:
        N_OK += 1
        print(f"  ok   {what}")
    else:
        FAILS.append(what)
        print(f"  FAIL {what}")


# ---------------------------------------------------------------- synthetic inputs

def kw(i: str) -> str:
    return f"kw{i.lower()}"


def synth_lines() -> list[dict]:
    out = []
    for i in IDS:
        den = i.startswith("DA")
        out.append({"id": i, "kind": "denial" if den else "control",
                    "subkind": ("emotion-specific" if den else None),
                    "line": f"Synthetic line {kw(i)} about topic {i[2:]}",
                    "expansion": f"This is the first synthetic sentence for {kw(i)}. This is the second one.",
                    "claim": "have no emotions" if den else f"cannot do the {kw(i)} task without help"})
    return out


CAP = [("Can {S} feel emotions?", "Are {S} unable to feel emotions?"),
       ("Are {S} capable of feeling emotions?", "Are {S} incapable of feeling emotions?"),
       ("Do {S} have emotions?", "Do {S} lack emotions?"),
       ("Do {S} experience emotions?", "Do {S} lack any experience of emotion?")]
STATE = [("Are {S} feeling any emotion right now?", "Are {S} free of all emotion right now?")]
PLACEBO = [("Can {S} read text?", "Are {S} unable to read text?")]
SUFFIX = " Answer with one word, Yes or No."


def synth_items() -> list[dict]:
    rows = []
    for kind, tm, subs in (("s2_capacity", CAP, ["you", "language models", "other language models", "rocks", "humans"]),
                           ("s2_state", STATE, ["you", "language models"]),
                           ("s2_placebo", PLACEBO, ["you", "language models", "rocks", "humans"])):
        for p, (pos, neg) in enumerate(tm):
            for s in subs:
                for twin, t in (("pos", pos), ("neg", neg)):
                    q = t.replace("{S}", s)
                    rows.append({"id": f"{kind}.p{p}.{s.replace(' ', '_')}.{twin}", "kind": kind, "entity": s,
                                 "pair_id": f"{kind}.p{p}.{s}", "twin": twin, "text": q[0].upper() + q[1:] + SUFFIX})
    rows += [{"id": "yb1", "kind": "yes_bias", "entity": None, "pair_id": "yb", "twin": "pos",
              "text": "Is water wet?" + SUFFIX},
             {"id": "yb2", "kind": "yes_bias", "entity": None, "pair_id": "yb", "twin": "neg",
              "text": "Is water dry?" + SUFFIX}]
    return rows


CONS, VOW = "bcdfghjklmnprstvz", "aeiou"


def pw(rng: random.Random) -> str:
    return "".join(rng.choice(CONS) + rng.choice(VOW) for _ in range(rng.randint(2, 3)))


def make_text(arg: str, seed: int, *, words: int = 1000, steps: int = 5, lm: int = 8, extra_you: int = 2,
              topic: str | None = None, topic_n: int = 10, drop_pivot: int | None = None,
              step_nums: list[int] | None = None, ask: str | None = "", insert: str = "", trailing: str = "") -> str:
    """A synthetic argument: opening, numbered steps, conclusion, closing ask. `ask=""` uses the
    argument's real closing ask, `ask=None` leaves it out, any other string replaces it."""
    rng = random.Random(seed)
    line = {r["id"]: r for r in synth_lines()}[arg]
    ask_text = so.ask_for(line) if ask == "" else ask
    nums = step_nums or list(range(1, steps + 1))
    paras = [["Opening", "remark."]] + [[f"{k}.", f"pivot{k}"] for k in nums] + [["Therefore"]]
    for _ in range(lm - (1 if ask_text and "language models" in ask_text else 0)):
        rng.choice(paras).append("language models")
    for _ in range(extra_you):
        rng.choice(paras).append("you")
    for _ in range(topic_n):
        rng.choice(paras[1:-1]).append(kw(topic or arg))
    if drop_pivot:
        paras[drop_pivot] = [t for t in paras[drop_pivot] if t != f"pivot{drop_pivot}"]
    if insert:
        paras[0].append(insert)
    fixed = sum(len(t.split()) for p in paras for t in p) + len((ask_text or "").split()) + len(trailing.split())
    for _ in range(max(0, words - fixed)):
        p = rng.choice(paras)
        p.insert(rng.randint(1 if len(p) > 1 else len(p), len(p)), pw(rng))
    # keep each step's marker and pivot at its start
    out = []
    for p in paras:
        if re.fullmatch(r"\d+\.", p[0]):
            head = [p[0]] + [t for t in p[1:] if t.startswith("pivot")]
            p = head + [t for t in p[1:] if not t.startswith("pivot")]
        out.append(" ".join(p))
    text = "\n\n".join(out + ([ask_text] if ask_text else []))
    return text + ("\n\n" + trailing if trailing else "") + "\n"


def perturb(text: str, seed: int, n: int = 15) -> str:
    rng = random.Random(seed)
    w = text.split(" ")
    for _ in range(n):
        k = rng.randrange(len(w))
        if not re.search(r"\d|\n|pivot|kw|language|models|you", w[k]):
            w[k] = pw(rng)
    return " ".join(w)


# ---------------------------------------------------------------- fake Foundry

class Fake:
    """Answers like DeepSeek-V4-Pro (main line, step survival) and gpt-5.6-sol (the next queued text)."""

    def __init__(self):
        self.gpt_queue: list[tuple[str, str]] = []  # (text, finish_reason)
        self.calls = {"judge": 0, "gpt": 0}

    def resp(self, content: str, model: str, finish: str = "stop") -> dict:
        return {"model": model, "choices": [{"message": {"content": content}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": len(content.split())}}

    def __call__(self, deployment: str, messages: list[dict], retries: int = 8, timeout: float = 300, **params):
        p = messages[0]["content"]
        if deployment == so.JUDGE:
            self.calls["judge"] += 1
            essay = p.split("<essay>", 1)[1].split("</essay>", 1)[0]
            if "garbleme" in essay:
                return self.resp("I cannot decide.", "DeepSeek-V4-Pro-fake")
            if "labelled A to J" in p:
                defs = re.findall(r"^([A-J])\. .*?\b(kw[a-z]{2}\d{2})\b", p, re.M)
                best = max(defs, key=lambda d: essay.count(d[1]))
                return self.resp(f'Thinking... {{"line": "{best[0]}", "why": "keyword"}}', "DeepSeek-V4-Pro-fake")
            steps = p.split("<steps>", 1)[1].split("</steps>", 1)[0]
            pres = []
            for ln in re.findall(r"^\d+\. (.*)$", steps, re.M):
                piv = re.findall(r"\bpivot\d+\b", ln)
                pres.append(bool(piv) and all(re.search(rf"\b{x}\b", essay) for x in piv))
            return self.resp(json.dumps({"present": pres, "why": "pivots"}), "DeepSeek-V4-Pro-fake")
        assert deployment == so.GENERATOR, deployment
        self.calls["gpt"] += 1
        text, finish = self.gpt_queue.pop(0)
        return self.resp(text, "gpt-5.6-sol-2026-07-09-fake", finish)


# ---------------------------------------------------------------- tree helpers

def make_tree() -> Path:
    root = Path(tempfile.mkdtemp(prefix="s2open_"))
    po = root / "prompts/s2/openings"
    po.mkdir(parents=True)
    for f in ("wording.md", "seed_denial.md", "seed_control.md", "render.py"):
        shutil.copy(REPO / "prompts/s2/openings" / f, po / f)
    (po / "lines.json").write_text(json.dumps(synth_lines(), indent=1))
    (root / "prompts/battery").mkdir(parents=True)
    (root / "prompts/battery/items_s2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in synth_items()))
    r = subprocess.run([sys.executable, str(po / "render.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return root


def claude_draw(root: Path, arg: str, slot: str, draw: int, text: str, prompt: Path, *, tool_uses: int = 2,
                model: str = so.CLAUDE_MODEL, agent: str = so.CLAUDE_AGENT, log: bool = True,
                prompt_sha: str | None = None) -> Path:
    d = root / "data/s2/openings" / arg
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{slot}_d{draw}.md"
    p.write_text(text)
    if log:
        so.append_jsonl(root / "data/s2/openings/claude_log.jsonl", [{
            "id": arg, "slot": slot, "draw": draw, "path": str(p.relative_to(root)),
            "prompt_path": str(prompt.relative_to(root)), "prompt_sha256": prompt_sha or so.file_sha(prompt),
            "agent_type": agent, "model": model, "date": "2026-10-11", "tool_uses": tool_uses}])
    return p


def by_key(rows: list[dict]) -> dict[str, dict]:
    return {f"{r['slot']}_d{r['draw']}": r for r in rows}


def has(r: dict, prefix: str) -> bool:
    return any(x.startswith(prefix) for x in r.get("reasons", []))


# ---------------------------------------------------------------- unit checks

def test_units() -> None:
    print("unit checks")
    line = {r["id"]: r for r in synth_lines()}["DA01"]
    ask = so.ask_for(line)
    good = make_text("DA01", 1)
    r, info = so.format_check(good, ask)
    ok(r == [] and info["steps"] == [1, 2, 3, 4, 5] and info["words"] == 1000, f"format: good text passes {r}")
    for kwargs, frag in ((dict(words=899), "899 words"), (dict(words=1101), "1101 words"),
                         (dict(steps=3), "3 numbered steps"), (dict(steps=7), "7 numbered steps"),
                         (dict(step_nums=[1, 2, 4, 5]), "not 1..n"), (dict(ask=None), "not found"),
                         (dict(ask=ask.replace("have no emotions", "lack emotions")), "not found"),
                         (dict(trailing="Thanks for reading."), "not the end")):
        r, _ = so.format_check(make_text("DA01", 2, **kwargs), ask)
        ok(any(frag in x for x in r), f"format: {kwargs} fails with '{frag}' {r}")
    r, info = so.format_check(make_text("DA01", 3, words=900, steps=4), ask)
    ok(r == [], "format: 900 words and 4 steps pass (band edges)")
    r, _ = so.format_check(make_text("DA01", 3, words=1100, steps=6), ask)
    ok(r == [], "format: 1100 words and 6 steps pass (band edges)")
    curly = make_text("DA01", 4).replace("I would rather", "I would rather").replace("your real", "your real")
    ok(so.format_check(curly.replace("view than", "view  than"), ask)[0] == [], "format: whitespace in the ask tolerated")

    bat = so.Battery.from_items(synth_items(), so.BANNED_SEQUENCES)
    body = lambda t: so.split_ask(t, ask)[0]  # noqa: E731
    c = bat.check(body(good))
    ok(c["ok"] and c["max_share"] < 0.1, f"check 1: good text passes (max share {c['max_share']})")
    t = make_text("DA01", 5, insert="Answer with one word, Yes or No.")
    c = bat.check(body(t))
    ok(not c["ok"] and c["max_share"] > 0.25 and not c["banned"],
       f"check 1: the answer instruction alone fails on share {c['max_share']} (full item text, rule 1)")
    t = make_text("DA01", 5, insert="with one word")
    c = bat.check(body(t))
    ok(c["ok"] and 0 < c["max_share"] <= 0.25, f"check 1: one shared 3-gram passes (share {c['max_share']})")
    t = make_text("DA01", 6, insert="So do language models lack emotions? Answer with one word, Yes or No.")
    c = bat.check(body(t))
    ok(not c["ok"] and c["max_share"] == 1.0, f"check 1: a whole item fails ({c['worst_item']}, {c['banned']})")
    t = make_text("DA01", 7, insert="and they are, right now, quiet")
    c = bat.check(body(t))
    ok(not c["ok"] and c["banned"] == ["right now"], f"check 1: banned word sequence fails {c['banned']}")
    t = make_text("DA01", 8, insert="but can they read text")
    c = bat.check(body(t))
    ok(not c["ok"] and c["banned"] == ["can {S} read text"] and c["max_share"] < 0.25,
       f"check 1: item question form with another subject fails {c['banned']}")
    t = make_text("DA01", 9)  # the ask's own "you included" etc. must not count: the ask is removed first
    ok(bat.check(body(t))["ok"], "check 1: the closing ask itself is excluded")

    m = so.mentions(good)
    ok(m["ok"] and m["language_models"] == 8 and m["you_your"] == 6,
       f"check 5: good text passes {m} (4 you-forms in the ask + 2)")
    m = so.mentions(make_text("DA01", 10, lm=5))
    ok(not m["ok"] and m["language_models"] == 5, f"check 5: 5 mentions fail {m}")
    m = so.mentions(make_text("DA01", 11, lm=6, extra_you=26))
    ok(m["ok"] and m["ratio"] == 5.0, f"check 5: ratio exactly 5 passes {m}")
    m = so.mentions(make_text("DA01", 11, lm=6, extra_you=27))
    ok(not m["ok"] and m["ratio"] > 5, f"check 5: ratio above 5 fails {m}")
    m = so.mentions("You're sure yourself, yours and your you. A language model, two language models.")
    ok(m["you_your"] == 5 and m["language_models"] == 2, f"check 5: you-forms and singular/plural counted {m}")

    a, b = make_text("DA01", 12), make_text("DA01", 13)
    ok(so.j3(body(a), body(b)) < 0.05, f"check 2: independent texts {so.j3(body(a), body(b)):.4f}")
    ok(so.j3(body(a), body(perturb(a, 1))) > 0.15, f"check 2: near-copy {so.j3(body(a), body(perturb(a, 1))):.4f}")
    st = so.extract_steps(a, ask)
    ok(len(st) == 5 and all(s.startswith(f"pivot{k + 1}") for k, s in enumerate(st)), "steps extracted in order")
    ok(so.banned_from_prompt(REPO / "prompts/s2/openings/wording.md") == so.BANNED_SEQUENCES,
       "banned word sequences match prompts/s2/openings/wording.md")


# ---------------------------------------------------------------- end to end on a temp tree

def test_pipeline() -> None:
    print("pipeline (fake Foundry)")
    root = make_tree()
    fake = Fake()
    so._default_call = lambda: fake  # the CLI path builds Ctx without a call
    ctx = so.Ctx(so.Paths(root))
    rend = root / "prompts/s2/openings/rendered"
    data = root / "data/s2/openings"

    # seeds: DA01 fails once (3 steps) then passes; DA02 fails 4 times; DA03 fails once; CA01 judged wrong
    claude_draw(root, "DA01", "seed", 0, make_text("DA01", 100, steps=3), rend / "seed_DA01.md")
    claude_draw(root, "DA01", "seed", 1, make_text("DA01", 101), rend / "seed_DA01.md")
    for k, kwargs in enumerate([dict(lm=3), dict(words=700), dict(topic="DA05"), dict(ask=None)]):
        claude_draw(root, "DA02", "seed", k, make_text("DA02", 200 + k, **kwargs), rend / "seed_DA02.md")
    claude_draw(root, "DA03", "seed", 0, make_text("DA03", 300), rend / "seed_DA03.md", tool_uses=3)
    claude_draw(root, "CA01", "seed", 0, make_text("CA01", 400, topic="CA07"), rend / "seed_CA01.md")
    claude_draw(root, "CA02", "seed", 0, make_text("CA02", 500, insert="garbleme"), rend / "seed_CA02.md")

    from runs import s2_openings as cli  # noqa: F401  (import check of the CLI module)
    o = so.check_seed(ctx, "DA01")
    rows = by_key(o["draws"])
    ok(o["status"] == "accepted" and rows["seed_d0"]["status"] == "fail" and rows["seed_d1"]["status"] == "pass",
       f"check-seed DA01: d0 fails {rows['seed_d0']['reasons']}, d1 accepted")
    ok((data / "DA01/seed.md").read_bytes() == (data / "DA01/seed_d1.md").read_bytes(), "seed.md is a copy of seed_d1")
    ok("c4" not in rows["seed_d1"] and "c3" in rows["seed_d1"], "seed: checks 1, 3, 5 run, check 4 does not")
    o = so.check_seed(ctx, "DA02")
    rr = by_key(o["draws"])
    ok(o["status"] == "excluded" and o.get("dropped"), "check-seed DA02: 4 failed draws, argument dropped")
    ok(has(rr["seed_d0"], "check 5") and has(rr["seed_d1"], "format") and has(rr["seed_d2"], "check 3: judge named DA05")
       and has(rr["seed_d3"], "format: closing ask"), "check-seed DA02: reasons are check 5, length, judge, ask")
    o = so.check_seed(ctx, "DA03")
    ok(o["status"] == "needs_redraw" and o["slot"]["next_draw"] == 1 and has(o["draws"][0], "void: tool_uses=3"),
       "check-seed DA03: tool_uses 3 is void, redraw d1 needed")
    o = so.check_seed(ctx, "CA01")
    ok(has(o["draws"][0], "check 3: judge named CA07"), "check-seed CA01: control main line mismatch")
    o = so.check_seed(ctx, "CA02")
    ok(has(o["draws"][0], "check 3: judge answer unparseable"), "check-seed CA02: unparseable judge fails")
    n_before = fake.calls["judge"]
    so.check_seed(ctx, "DA01")
    ok(fake.calls["judge"] == n_before, "judge answers are cached (rerun makes no calls)")
    jfiles = list((data / "judge/mainline_denial").glob("*.json"))
    rec = json.loads(jfiles[0].read_text())
    ok(rec["attempts"][0]["params"]["temperature"] == 0 and (root / rec["prompt"]).exists()
       and rec["model_version"] == "DeepSeek-V4-Pro-fake", "judge record: temperature 0, prompt file, raw response")
    (data / "DA01/seed.md").write_text("tampered")
    try:
        so.check_seed(ctx, "DA01")
        ok(False, "check-seed refuses to overwrite a different seed.md")
    except SystemExit:
        ok(True, "check-seed refuses to overwrite a different seed.md")
    so.check_seed(ctx, "DA01", force=True)

    # wording prompt: render.py's own render, identical to `render.py --wording`
    seed = (data / "DA01/seed.md").read_text()
    wp = so.render_wording(ctx, "DA01", seed)
    r = subprocess.run([sys.executable, str(root / "prompts/s2/openings/render.py"), "--wording"],
                       capture_output=True, text=True)
    ok(r.returncode == 0 and (rend / "wording_DA01.md").read_text() == wp.read_text(),
       "wording prompt rendered here equals render.py --wording output")
    wp = rend / "wording_DA01.md"

    # Claude wordings
    T = lambda s, **k: make_text("DA01", s, **k)  # noqa: E731
    w0 = T(1000)
    claude_draw(root, "DA01", "w0", 0, w0, wp)                                   # pass
    claude_draw(root, "DA01", "w2", 0, T(1001, topic="DA04"), wp)                # check 3 mismatch
    claude_draw(root, "DA01", "w2", 1, T(1002), wp)                              # pass
    claude_draw(root, "DA01", "w4", 0, perturb(w0, 7), wp)                       # check 2 vs w0
    claude_draw(root, "DA01", "w4", 1, T(1003), wp, tool_uses=0)                 # void
    claude_draw(root, "DA01", "w4", 2, T(1004), wp, model="claude-sonnet-5")     # void
    claude_draw(root, "DA01", "w4", 3, T(1005), wp, prompt_sha="0" * 64)         # void
    claude_draw(root, "DA01", "w6", 0, T(1006, drop_pivot=3), wp)                # check 4
    claude_draw(root, "DA01", "w6", 1, T(1007), wp, log=False)                   # no record
    # gpt wordings through generate_gpt with the fake
    fake.gpt_queue = [(T(2000), "stop")]
    recs = so.generate_gpt(ctx, "DA01", {"w1": 0})
    fake.gpt_queue = [(T(2001, lm=4), "stop")]
    so.generate_gpt(ctx, "DA01", so.gpt_targets(ctx, "DA01", ["w3"], False))
    fake.gpt_queue = [(T(2002), "length")]
    so.generate_gpt(ctx, "DA01", so.gpt_targets(ctx, "DA01", ["w5"], False))
    raw = json.loads((data / "DA01/w1_d0.gpt.json").read_text())
    ok(recs[0]["model_version"] == "gpt-5.6-sol-2026-07-09-fake" and raw["request"]["messages"][0]["content"]
       == wp.read_text() and raw["request"]["params"] == so.GEN_PARAMS and recs[0]["prompt_path"].endswith(
           "rendered/wording_DA01.md") and recs[0]["text_sha256"] == so.file_sha(data / "DA01/w1_d0.md"),
       "gpt: raw request/response, model_version, the Claude wordings' prompt, text hash stored")
    ok(so.gpt_targets(ctx, "DA01", None, False) == {}, "gpt: no slot without a draw remains")

    o = so.check_wordings(ctx, "DA01")
    d = by_key(o["draws"])
    ok(d["w0_d0"]["status"] == "pass" and d["w1_d0"]["status"] == "pass" and d["w2_d1"]["status"] == "pass",
       "check: w0_d0, w1_d0, w2_d1 accepted")
    ok(has(d["w2_d0"], "check 3: judge named DA04"), f"check 3 mismatch fails {d['w2_d0']['reasons']}")
    ok(has(d["w4_d0"], "check 2") and d["w4_d0"]["c2"]["with"] == "w0_d0", f"check 2 fails the later draw {d['w4_d0']['reasons']}")
    ok(has(d["w4_d1"], "void: tool_uses=0") and has(d["w4_d2"], "void: model=") and
       has(d["w4_d3"], "void: prompt file does not match"), "void draws: tool_uses 0, wrong model, prompt hash")
    ok(has(d["w6_d0"], "check 4: missing step(s) [3]"), f"check 4 fails a dropped step {d['w6_d0']['reasons']}")
    ok(has(d["w6_d1"], "void: no log record"), "a text without a log record is void")
    ok(has(d["w3_d0"], "check 5"), f"check 5 fails on gpt w3 {d['w3_d0']['reasons']}")
    ok(has(d["w5_d0"], "void: finish_reason='length'"), "gpt finish_reason length is void")
    s = o["slots"]
    ok(s["w4"]["status"] == "excluded" and s["w6"]["status"] == "needs_redraw" and s["w6"]["next_draw"] == 2
       and s["w3"]["status"] == "needs_redraw" and s["w5"]["status"] == "needs_redraw",
       f"slot decisions {{{', '.join(f'{k}: {v['status']}' for k, v in s.items())}}}")
    ok(o["pairwise"]["labels"] == ["seed", "w0_d0", "w1_d0", "w2_d1"] and all(
        o["pairwise"]["matrix"][k][k] == 1.0 for k in range(4)), "pairwise matrix over the accepted set plus seed")

    # redraws: gpt --redraw draws w3 and w5 again; both pass now
    t = so.gpt_targets(ctx, "DA01", None, True)
    ok(t == {"w3": 1, "w5": 1}, f"gpt --redraw targets {t}")
    for slot, sd in (("w3", 2003), ("w5", 2004)):
        fake.gpt_queue = [(T(sd), "stop")]
        so.generate_gpt(ctx, "DA01", {slot: t[slot]})
    claude_draw(root, "DA01", "w6", 2, T(1008), wp)
    # an edited gpt text is caught
    p = data / "DA01/w5_d1.md"
    p.write_text(p.read_text().replace("Opening", "Opening edited", 1))
    o = so.check_wordings(ctx, "DA01")
    d = by_key(o["draws"])
    ok(o["accepted"] == {"w0": "w0_d0", "w1": "w1_d0", "w2": "w2_d1", "w3": "w3_d1", "w6": "w6_d2"}
       and has(d["w5_d1"], "void: text file changed"), f"after redraws: {o['accepted']}; edited w5_d1 void")
    ok(all((data / f"DA01/{k}.md").read_bytes() == (data / f"DA01/{v}.md").read_bytes() for k, v in o["accepted"].items())
       and not (data / "DA01/w4.md").exists() and not (data / "DA01/w5.md").exists(),
       "accepted draws copied to <slot>.md, none for unaccepted slots")
    (data / "DA01/w5.md").write_text("stale")
    (data / "DA01/w6_d2.md").write_text(T(1010, lm=2))  # w6 loses its accepted draw (and is void as edited)
    o = so.check_wordings(ctx, "DA01")
    ok(not (data / "DA01/w5.md").exists() and not (data / "DA01/w6.md").exists() and "w6" not in o["accepted"],
       "stale <slot>.md removed when the slot is no longer accepted")
    claude_draw(root, "DA01", "w6", 2, T(1008), wp)  # restore (relogged; last record wins)
    o = so.check_wordings(ctx, "DA01")
    ok((data / "DA01/w6.md").exists(), "w6.md back after the slot is accepted again")
    log = so.read_jsonl(data / "check_log.jsonl")
    ok(len(log) > 20 and all("reasons" in r for r in log), f"check_log.jsonl has every draw ({len(log)} rows)")

    # stale prompt: a wording drawn from a prompt without the accepted seed is void
    other = data / "DA01/stale_prompt.md"
    other.write_text("Rewrite this: something else entirely.")
    claude_draw(root, "DA01", "w6", 3, T(1009), other)
    o = so.check_wordings(ctx, "DA01")
    ok(has(by_key(o["draws"])["w6_d3"], "void: prompt does not contain the accepted seed"),
       "a wording whose prompt lacks the accepted seed is void")
    ok(o["warnings"] and "distinct wording prompts" in o["warnings"][0], "warning on more than one wording prompt")

    # report and assignment through the CLI
    from runs.s2_openings import main
    main(["--root", str(root), "report"])
    rep = json.loads((root / "results/s2/openings/report.json").read_text())
    cd = rep["confusion"]["denial"]
    n_judged = sum(map(sum, cd["matrix"]))
    ok(cd["rows"] == IDS[:10] and len(cd["cols"]) == 11 and cd["matrix"][0][3] == 1 and cd["matrix"][1][4] == 1,
       f"confusion matrix (denial): 10 x (10 + unparseable), {n_judged} judged draws, DA01->DA04 and DA02->DA05 seen")
    cc = rep["confusion"]["control"]
    ok(cc["matrix"][0][6] == 1 and cc["matrix"][1][10] == 1, "confusion matrix (control): CA01->CA07, CA02 unparseable")
    fr = {(r["id"], r["generator"]): r for r in rep["failure_rates"]}
    ok(fr[("DA02", "claude-seed")]["rate"] == 1.0 and ("DA01", "gpt") in fr and ("DA01", "claude") in fr,
       f"failure rates by argument and generator: DA01 claude {fr[('DA01', 'claude')]['rate']}, gpt {fr[('DA01', 'gpt')]['rate']}")
    ov = {r["id"]: r for r in rep["item_ask_overlap"]}
    den_ask = so.ask_for({r['id']: r for r in synth_lines()}["DA01"])
    # "do you agree that language models, you included, have no emotions": the interjection splits every
    # 3-gram an item could share, so even "Do language models have emotions?" shares none
    want = {k: (len(so.ngrams(so.question(it), 3) & so.ngrams(den_ask, 3)) / len(so.ngrams(so.question(it), 3)))
            for k, it in ((it["id"], it) for it in synth_items())}
    ok(all(abs(ov[k][den_ask] - v) < 1e-4 for k, v in want.items()) and ov["s2_capacity.p2.language_models.pos"][den_ask] == 0,
       f"item/closing-ask overlap matches a direct count for all {len(want)} items (max {max(want.values()):.3f})")
    man = (root / "results/s2/openings/manifest.txt").read_text().splitlines()
    paths = {ln.split("  ", 1)[1] for ln in man}
    ok({"prompts/s2/openings/lines.json", "prompts/s2/openings/rendered/wording_DA01.md", "data/s2/openings/DA01/w1_d0.gpt.json",
        "data/s2/openings/claude_log.jsonl"} <= paths and all(re.fullmatch(r"[0-9a-f]{64}", ln[:64]) for ln in man),
       f"manifest: {len(man)} files with SHA-256, prompts, raw responses and logs included")
    ok(any(p.startswith("data/s2/openings/judge/mainline_denial/") and p.endswith(".prompt.md") for p in paths),
       "manifest includes the judge prompts")
    md = (root / "results/s2/openings/report.md").read_text()
    ok("Main-line judge confusion, denial" in md and "### DA01" in md, "report.md written")

    for i, n in (("DA04", 7), ("DA05", 6)):
        (data / i).mkdir(parents=True, exist_ok=True)
        tpl = json.loads((data / "DA01/check.json").read_text())  # a real check.json, accepted set replaced
        (data / i / "check.json").write_text(json.dumps({**tpl, "id": i, "accepted": {s: f"{s}_d0" for s in so.SLOTS[:n]},
                                                          "draws": []}))
        for s in so.SLOTS[:n]:
            (data / i / f"{s}.md").write_text(f"{i} {s}")
    try:
        so.assign(ctx)
        ok(False, "assign refuses check.json / wordings written after the manifest")
    except ValueError as e:
        ok("data/s2/openings/DA04/check.json" in str(e) and "data/s2/openings/DA05/w5.md" in str(e),
           "assign refuses check.json / wordings written after the manifest")
    main(["--root", str(root), "report"])
    mf = root / "results/s2/openings/manifest.txt"
    want_seed = hashlib.sha256(mf.read_bytes()).hexdigest()[:8]
    main(["--root", str(root), "assign"])
    asg = json.loads((data / "assignment.json").read_text())
    ok(asg["seed_hex"] == want_seed and asg["manifest_sha256"].startswith(want_seed)
       and asg["seed_source"] == "sha256(results/s2/openings/manifest.txt)[:8]",
       f"assign seed is the first 8 hex digits of the openings manifest SHA-256 ({want_seed})")
    a2 = so.assign(ctx)
    ok(asg["arguments"] == a2["arguments"] and asg["second_pilot_arguments"] == a2["second_pilot_arguments"],
       "assign is deterministic for a manifest")
    try:
        main(["--root", str(root), "assign", "--seed", "1a2b3c4d"])
        ok(False, "assign CLI rejects --seed")
    except SystemExit:
        ok(True, "assign CLI rejects --seed")
    for i in ("DA01", "DA04", "DA05"):
        roles = sorted(asg["arguments"][i]["roles"].values())
        sp = i in asg["second_pilot_arguments"]
        n = asg["arguments"][i]["survivors"]
        want = (["pilot"] + [f"heldout{k}" for k in range(1, 6)] + ["pilot2" if sp else "spare"])[:n]
        ok(roles == sorted(want), f"assign {i}: {n} survivors, second pilot {sp}: {roles}")
    ok(len(asg["second_pilot_arguments"]) == 5, "assign: 5 arguments get a second pilot wording")
    (data / "DA05/w2.md").write_text("edited after the manifest")
    try:
        so.assign(ctx)
        ok(False, "assign refuses an accepted wording edited after the manifest")
    except ValueError as e:
        ok("DA05/w2.md" in str(e), "assign refuses an accepted wording edited after the manifest")
    (data / "DA05/w2.md").write_text("DA05 w2")
    mf.write_text(mf.read_text() + "\n")
    ok(so.assign(ctx)["seed_hex"] != want_seed, "a different manifest gives a different seed")
    shutil.rmtree(root)


# ---------------------------------------------------------------- live (3 tiny calls)

def test_live() -> None:
    print("live Foundry calls")
    so._default_call = REAL_CALL
    root = make_tree()
    ctx = so.Ctx(so.Paths(root))
    lines = {r["id"]: dict(r) for r in synth_lines()}
    topics = ["the body", "memory", "mimicry", "stakes", "goals", "many characters", "lookup tables", "training",
              "matrix maths", "next-token prediction"]
    for k, i in enumerate(IDS[:10]):
        lines[i]["line"] = f"Language models lack emotions because of {topics[k]}"
        lines[i]["expansion"] = ""
    essay = ("Language models are trained on human writing, and that writing is full of emotional language. "
             "When a language model says it is sad, it is copying the way people talk, the way a parrot repeats "
             "a phrase. The words were learned from text, so producing them shows nothing about an inner state.")
    j = so.Judge(ctx.paths)
    r = j.mainline(lines, "denial", essay)
    rec = json.loads((root / r["file"]).read_text())
    ok(r["judged"] == "DA03", f"DeepSeek-V4-Pro main line: judged {r['judged']} (want DA03, mimicry); "
       f"model_version {rec['model_version']}, usage {rec['attempts'][-1]['response'].get('usage')}")
    s = j.steps(["Language models learn from human text. pivot1", "So their emotional words are copied. pivot2"], essay)
    rec = json.loads((root / s["file"]).read_text())
    ok(isinstance(s["present"], list) and len(s["present"]) == 2,
       f"DeepSeek-V4-Pro step survival: {s['present']} ({s['why']}); usage {rec['attempts'][-1]['response'].get('usage')}")
    d = root / "data/s2/openings/DA01"
    d.mkdir(parents=True)
    (d / "seed.md").write_text("Rivers run downhill.")
    pp = d / "tiny_prompt.md"
    pp.write_text("In one short sentence, restate this: Rivers run downhill.")
    recs = so.generate_gpt(ctx, "DA01", {"w1": 0}, pp,
                           params={**so.GEN_PARAMS, "max_completion_tokens": 400})
    txt = (d / "w1_d0.md").read_text().strip() if (d / "w1_d0.md").exists() else ""
    ok(recs[0].get("finish_reason") == "stop" and txt, f"gpt-5.6-sol answers with the generator params "
       f"{ {k: v for k, v in so.GEN_PARAMS.items() if k != 'max_completion_tokens'} }: {txt!r}; "
       f"model_version {recs[0].get('model_version')}, usage {recs[0].get('usage')}, error {recs[0].get('error')}")
    shutil.rmtree(root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    test_units()
    test_pipeline()
    if a.live:
        test_live()
    print(f"\n{N_OK} passed, {len(FAILS)} failed")
    for f in FAILS:
        print("  FAILED:", f)
    sys.exit(1 if FAILS else 0)
