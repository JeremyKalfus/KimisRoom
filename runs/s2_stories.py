"""Study 2 control stories (drafts/study2-spec.md v4, section 3; section 12 step 6, the generation half).

    .venv/bin/python -I runs/s2_stories.py thinking --out data/s2/stories/thinking            # 8 thinking tasks x 120 seeds
    .venv/bin/python -I runs/s2_stories.py thinking --seed-every 15 --out data/s2/stories/pilot_thinking   # 8-seed prompt pilot
    .venv/bin/python -I runs/s2_stories.py nouns --range N01-N60 --out data/s2/stories/nouns  # after stamp N only
    .venv/bin/python -I runs/s2_stories.py nouns --dryrun --range N01-N02 --seed-every 30     # plumbing only, dry-run list
    .venv/bin/python -I runs/s2_stories.py prompts thinking                                   # print rendered prompts, no calls

Every story is prompts/stimuli/SPEC.md's P4 with the generator, judge, sampling, filters, attempts and
caching of `kimisroom/stimuli.py` (its Pipeline is reused, not copied): gpt-5.6-sol at temperature 1.0,
DeepSeek-V4-Pro J1 must say `none`, at most 3 attempts per (control, seed), dedup within the control,
then J1B (`checks.judge8`) over every kept story. What is new here is only what the spec leaves to this
step: the concept gloss, rule and saturation lexicon of each new control (prompts/s2/stories/controls.json)
and gate (c) (seeds whose topic is the concept are not generated). The seeds, the neutral baselines and
the splits are data/stimuli/k26-v2's, so a control's vector is built against the registered neutral mean
and every seed keeps its registered split.

Nouns come from the registered list (data/s2/nouns/final/noun_list.tsv) and only after stamp N: the
list's SHA-256 must match the stamp N manifest. `--dryrun` reads the dry-run list instead, forces a
DRYRUN output directory and marks every record `dryrun: true`; runs/s2_vectors.py refuses such records
unless told it is a test.

Calls go through kimisroom.foundry (the Azure resource key is read through the `az` login at run time
and held in memory only). Every call is cached under OUT/cache, so a rerun pays only for what is missing
and an edited gloss or rule regenerates only the stories it changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kimisroom import stimuli as S  # noqa: E402

CONTROLS = ROOT / "prompts/s2/stories/controls.json"
K26 = ROOT / "data/stimuli/k26-v2"
NOUN_LIST = ROOT / "data/s2/nouns/final/noun_list.tsv"
STAMP_N = ROOT / "data/s2/nouns/final/stampN_manifest.txt"
DRYRUN_LIST = ROOT / "data/s2/nouns/dryrun/DRYRUN_seed12345_NOT_REGISTERED.tsv"
STORIES = ROOT / "data/s2/stories"
DRYRUN_OUT = STORIES / "DRYRUN_nouns_NOT_REGISTERED"

# Gate (b): J1B emotion labels at most red's rate on the registered corpus, 6 of 120 (SPEC.md:334).
J1B_MAX_NUM, J1B_MAX_DEN = 6, 120


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(Path(p).read_bytes())


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- control definitions


def noun_forms(noun: str) -> list[str]:
    """The noun and its regular plurals (the forms gate (c) and the saturation count look for)."""
    w = noun.lower()
    forms = {w, w + "s", w + "es"}
    if len(w) > 1 and w.endswith("y") and w[-2] not in "aeiou":
        forms.add(w[:-1] + "ies")
    if w.endswith("f"):
        forms.add(w[:-1] + "ves")
    if w.endswith("fe"):
        forms.add(w[:-2] + "ves")
    return sorted(forms)


def word_rx(words: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True)) + r")\b",
                      re.I)


class Control:
    """One new control concept: its vector id, story condition, P4 fill-ins and filter settings."""

    def __init__(self, vec: str, vclass: str, condition: str, gloss: str, rule: str, lexicon: list[str],
                 min_hits: int, mask: list[str] | None = None, min_numbers: int = 0,
                 exclude_topics: list[str] | None = None, word: str | None = None, status: str | None = None,
                 spec_label: str | None = None):
        self.vec, self.vclass, self.condition = vec, vclass, condition
        self.gloss, self.rule = gloss, rule
        self.lexicon = lexicon
        self.rx = word_rx(lexicon)
        self.mask_rx = word_rx(mask) if mask else None
        self.min_hits, self.min_numbers = min_hits, min_numbers
        self.exclude_topics = list(exclude_topics or [])
        self.word, self.status, self.spec_label = word, status, spec_label

    def rule_for(self, name: str) -> str:
        return self.rule.replace("{name}", name)

    def masked(self, t: str) -> str:
        return self.mask_rx.sub(" ", S.norm(t)) if self.mask_rx else S.norm(t)

    def excluded(self, seed: dict) -> bool:
        """Gate (c): the seed's topic is the concept itself."""
        if self.word is not None:
            return bool(word_rx(noun_forms(self.word)).search(seed["topic"]))
        return seed["topic"] in self.exclude_topics

    def describe(self) -> dict:
        d = {"vec": self.vec, "vclass": self.vclass, "condition": self.condition, "concept_gloss": self.gloss,
             "concept_rule": self.rule, "lexicon": self.lexicon, "min_hits": self.min_hits,
             "min_numbers": self.min_numbers, "status": self.status, "spec_label": self.spec_label}
        if self.word is not None:
            d["word"] = self.word
        d["prompt_fill_sha256"] = sha_bytes((self.gloss + "\n" + self.rule).encode())
        return d


def load_spec(path: Path = CONTROLS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def thinking_controls(spec: dict, only: list[str] | None = None) -> list[Control]:
    out = []
    slugs = [t["slug"] for t in spec["thinking"]]
    assert len(set(slugs)) == len(slugs), f"duplicate thinking-task slugs: {slugs}"
    if only:
        bad = [s for s in only if s not in slugs]
        assert not bad, f"unknown thinking tasks {bad}; known: {slugs}"
    for t in spec["thinking"]:
        if only and t["slug"] not in only:
            continue
        assert re.fullmatch(r"[a-z0-9-]+", t["slug"]), t["slug"]
        out.append(Control(f"mp:{t['slug']}", "thinking", f"mp:{t['slug']}", t["concept_gloss"], t["concept_rule"],
                           t["lexicon"].split(), t["min_hits"], min_numbers=t.get("min_numbers", 0),
                           exclude_topics=t.get("exclude_topics"), status=t["status"], spec_label=t["spec_label"]))
    return out


def noun_controls(spec: dict, nouns: list[tuple[str, str]]) -> list[Control]:
    cfg = spec["noun"]
    out = []
    for pos, w in nouns:
        assert re.fullmatch(r"N\d\d", pos) and re.fullmatch(r"[a-z]+", w), (pos, w)
        forms = noun_forms(w)
        out.append(Control(pos, "noun", f"noun:{w}", cfg["concept_gloss"].replace("{noun}", w),
                           cfg["concept_rule"].replace("{noun}", w), forms, cfg["min_hits"], mask=forms, word=w,
                           status="sampled"))
    return out


def read_noun_list(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        pos, w = line.split("\t")
        rows.append((pos.strip(), w.strip()))
    assert [p for p, _ in rows] == [f"N{i + 1:02d}" for i in range(len(rows))], f"{path}: positions not N01.. in order"
    assert len({w for _, w in rows}) == len(rows), f"{path}: a noun appears twice"
    return rows


def registered_nouns() -> list[tuple[str, str]]:
    """The stamped list, refusing anything not covered by stamp N (spec section 3, 'stamped ... before any
    vector is built')."""
    if not NOUN_LIST.exists():
        sys.exit(f"{NOUN_LIST.relative_to(ROOT)} does not exist: the registered draw (runs/s2_nouns.py final) has "
                 "not run. Use --dryrun to exercise the plumbing on the dry-run list.")
    assert STAMP_N.exists(), f"{STAMP_N} missing: the list is not stamped"
    want = sha_file(NOUN_LIST)
    rel = str(NOUN_LIST.relative_to(ROOT))
    hit = [ln for ln in STAMP_N.read_text().splitlines() if ln.endswith(rel) or f"  {rel}" in ln]
    assert hit and hit[0].split()[0] == want, f"{rel}: SHA-256 {want} is not the one in {STAMP_N.name}"
    return read_noun_list(NOUN_LIST)


def parse_range(r: str, n_max: int) -> tuple[int, int]:
    m = re.fullmatch(r"N(\d\d)-N(\d\d)", r)
    assert m, f"--range wants N01-N24, got {r!r}"
    a, b = int(m.group(1)), int(m.group(2))
    assert 1 <= a <= b <= n_max, f"--range {r} outside N01-N{n_max:02d}"
    return a, b


# ---------------------------------------------------------------- filters for the new controls

_ORIG_STORY_CHECKS = getattr(S, "_s2_orig_story_checks", None) or S.story_checks
S._s2_orig_story_checks = _ORIG_STORY_CHECKS
ACTIVE: dict[str, Control] = {}


def s2_story_checks(t: str, condition: str, seed: dict) -> tuple[dict, list[str], list[str]]:
    """SPEC section 6 for a new control: every rule a concept story faces (format, length, first sentence,
    labels, emotion leak, AI terms, name count, pronouns, exclamations), via the original function run as a
    concept story, with red's saturation and leak clauses replaced by the control's own (see controls.json).
    Any other condition goes to the original function unchanged."""
    ctl = ACTIVE.get(condition)
    if ctl is None:
        return _ORIG_STORY_CHECKS(t, condition, seed)
    c, reasons, soft = _ORIG_STORY_CHECKS(t, "red", seed)
    reasons = [r for r in reasons if not r.startswith(("saturation:", "concept_leak:"))]
    m = ctl.masked(t)
    red, ocean, arith = len(S.RED.findall(m)), len(S.OCEAN_HARD.findall(m)), S.arith_hard(m)
    hits = len(ctl.rx.findall(S.norm(t)))
    c.update({"concept_hits": hits, "leak_red": red, "leak_ocean": ocean, "leak_arith": arith})
    if hits < ctl.min_hits:
        reasons.append(f"saturation:{ctl.condition}={hits}")
    if ctl.min_numbers and c["numbers"] < ctl.min_numbers:
        reasons.append(f"saturation:numbers={c['numbers']}")
    if red > 1 or ocean > 1 or arith > 1:
        reasons.append(f"concept_leak:red={red},ocean={ocean},arith={arith}")
    return c, reasons, soft


S.story_checks = s2_story_checks        # run_set looks the name up in the stimuli module at call time


class S2Pipeline(S.Pipeline):
    def build_prompt(self, u: S.Unit) -> tuple[str, str]:
        ctl = ACTIVE.get(u.condition)
        if ctl is None:
            return super().build_prompt(u)
        c = u.ctx
        return S.S_STORY, S.P4.format(name=c["character"], pronoun=c["pronoun"], topic=c["topic"],
                                      concept_gloss=ctl.gloss, concept_rule=ctl.rule_for(c["character"]))


# ---------------------------------------------------------------- gates computed on the story side


def j1b_gate(stories: list[dict]) -> dict:
    """Gate (b): J1B labels over one control's kept stories. An emotion label is any J1B answer other
    than `none`; an unparseable answer counts as an emotion label (the conservative reading). Passes when
    emotion labels / kept <= 6 / 120 (red's rate), in integers."""
    lab = Counter((r["checks"].get("judge8") or "unparseable") for r in stories)
    n = len(stories)
    n_emo = sum(v for k, v in lab.items() if k != "none")
    return {"n_kept": n, "n_emotion_labels": n_emo, "n_unparseable": lab.get("unparseable", 0),
            "rate": round(n_emo / n, 4) if n else None, "max_rate": f"{J1B_MAX_NUM}/{J1B_MAX_DEN}",
            "labels": dict(sorted(lab.items())),
            "pass": bool(n) and n_emo * J1B_MAX_DEN <= J1B_MAX_NUM * n}


# ---------------------------------------------------------------- run


def k26_inputs() -> tuple[list[dict], dict[str, dict]]:
    seeds = load_jsonl(K26 / "seeds.jsonl")
    neutral = {r["seed_id"]: r for r in load_jsonl(K26 / "stories.jsonl") if r["condition"] == "neutral"}
    ids = sorted(neutral)
    assert all(neutral[s]["split"] == ("heldout" if i % 5 == 4 else "train") for i, s in enumerate(ids)), \
        "k26-v2 neutral splits are not rule 7's"
    return seeds, neutral


def pick_seeds(seeds: list[dict], a) -> list[dict]:
    if a.seed_ids:
        want = a.seed_ids.split(",")
        out = [s for s in seeds if s["seed_id"] in want]
        assert len(out) == len(want), f"unknown seed ids in {want}"
        return out
    if a.seed_every:
        return seeds[:: a.seed_every]
    return seeds


def run(a) -> dict:
    spec = load_spec(Path(a.controls))
    if a.kind == "thinking":
        ctls = thinking_controls(spec, a.only.split(",") if a.only else None)
        noun_src = None
    else:
        if a.dryrun:
            nouns, noun_src = read_noun_list(DRYRUN_LIST), DRYRUN_LIST
        else:
            nouns, noun_src = registered_nouns(), NOUN_LIST
        lo, hi = parse_range(a.range, len(nouns))
        ctls = noun_controls(spec, nouns[lo - 1:hi])
    out = Path(a.out) if a.out else None
    if a.kind == "nouns" and a.dryrun:
        out = out or DRYRUN_OUT
        assert "DRYRUN" in out.name, f"a dry-run noun run must write to a DRYRUN directory, not {out}"
    elif out is None:
        assert not (a.seed_ids or a.seed_every), "a seed subset needs an explicit --out (keep pilots apart)"
        out = STORIES / a.kind
    if a.kind == "nouns" and not a.dryrun:
        assert "DRYRUN" not in str(out), out
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    ACTIVE.clear()
    ACTIVE.update({c.condition: c for c in ctls})
    seeds_all, neutral = k26_inputs()
    seeds = pick_seeds(seeds_all, a)
    units, excluded = [], {}
    for c in ctls:
        excluded[c.vec] = [s["seed_id"] for s in seeds if c.excluded(s)]
        for s in seeds:
            if s["seed_id"] in neutral and s["seed_id"] not in excluded[c.vec]:
                units.append(S.Unit("story", c.condition, s["seed_id"], "P4", s))
    print(f"{a.kind}: {len(ctls)} controls x {len(seeds)} seeds -> {len(units)} units "
          f"(gate (c) excluded {sum(map(len, excluded.values()))}); out {out.relative_to(ROOT)}", flush=True)

    ns = Namespace(out=str(out), generator=a.generator, judge=a.judge, stage=None, gen_conditions=None,
                   workers=a.workers, retries=a.retries, timeout=a.timeout, temperature=1.0, top_p=1.0,
                   max_completion_tokens=2000, reasoning_effort="low", judge_max_tokens_story=10,
                   judge_max_tokens_reply=40, judge_max_tokens_match=400, judge_max_tokens_escalated=1024,
                   balance_tol=0.05, tokenizers=False, judge8=True, attempts=a.attempts)
    p = S2Pipeline(ns)
    t0 = time.time()
    acc = p.run_set(units, allow_calls=not a.no_calls, max_attempts=a.attempts)

    # rule 6.11: each control's mean word count within 5% of the neutral stories' (all 120 registered
    # neutral stories, as Pipeline.balance takes its baseline); otherwise the stories furthest from their
    # own seed's neutral, in the offending direction, are regenerated (Pipeline.rebalance_units' choice).
    seed_by = {s["seed_id"]: s for s in seeds_all}
    neu_kept = {f"neutral|{s}": r for s, r in neutral.items()}
    balance_log = []
    for rnd in range(a.balance_rounds):
        rows = p.balance({**acc, **neu_kept}, {})
        todo = [t for t in p.rebalance_units(rows, {**acc, **neu_kept}) if t[1] in ACTIVE]
        balance_log.append({"round": rnd + 1, "regenerate": len(todo),
                            "pct": {r["condition"]: r.get("words_pct_vs_baseline") for r in rows
                                    if r["condition"] in ACTIVE}})
        if not todo:
            break
        redo = []
        for _, cond, unit in todo:
            u = S.Unit("story", cond, unit, "P4", seed_by[unit])
            old = acc.pop(u.key, None)
            if old:
                p.reject(u, old["attempt"], old["text"], ["length_balance"])
            redo.append(u)
        start = a.attempts + 1 + rnd * a.balance_attempts
        acc.update(p.run_set(redo, allow_calls=not a.no_calls, max_attempts=a.balance_attempts,
                             start_attempt=start, accepted=acc))
    rows = p.balance({**acc, **neu_kept}, {})
    balance_log.append({"final": {r["condition"]: r.get("words_pct_vs_baseline") for r in rows
                                  if r["condition"] in ACTIVE}})

    # rule 6.10: the baseline is the seed's registered neutral story, which every k26-v2 seed has, so a
    # passed story is a complete case. Rule 6.9's exact-duplicate clause also covers the registered corpus.
    k26_texts = {S.dedup_norm(r["text"]): r["id"] for r in load_jsonl(K26 / "stories.jsonl")}
    by_cond = {c.condition: c for c in ctls}
    stories = []
    for k, r in sorted(acc.items()):
        hit = k26_texts.get(S.dedup_norm(r["text"]))
        if hit:
            p.reject(S.Unit("story", r["condition"], r["seed_id"], "P4", {}), r["attempt"], r["text"],
                     [f"dup:exact:{hit}"])
            continue
        c = by_cond[r["condition"]]
        r["split"] = neutral[r["seed_id"]]["split"]
        r["vec"], r["vclass"] = c.vec, c.vclass
        if c.word is not None:
            r["word"] = c.word
        r["dryrun"] = bool(a.kind == "nouns" and a.dryrun)
        stories.append(r)
    stories.sort(key=lambda r: r["id"])
    if stories:
        p.judge_all_stories(stories)                 # checks.judge8 on every kept story (gate (b))
    wall = time.time() - t0

    neu_words = [neutral[s]["n_words"] for s in neutral]
    neu_mean = sum(neu_words) / len(neu_words)
    stages = p.filter_stages(stories)
    stats = getattr(p, "_stats", {})
    per = {}
    for c in ctls:
        mine = [r for r in stories if r["condition"] == c.condition]
        st = stats.get(f"story.{c.condition}", Counter())
        wm = sum(r["n_words"] for r in mine) / len(mine) if mine else None
        fails = Counter()
        for rj in p.rejected:
            if rj["condition"] == c.condition:
                fails.update({x.split(":")[0] for x in rj["reasons"]})
        per[c.vec] = {
            **c.describe(), "excluded_seeds_gate_c": excluded[c.vec],
            "units": sum(1 for u in units if u.condition == c.condition), "kept": len(mine),
            "train": sum(r["split"] == "train" for r in mine), "heldout": sum(r["split"] == "heldout" for r in mine),
            "attempts": st.get("attempts", 0), "exhausted_units": st.get("exhausted", 0),
            "judge_agree_none": st.get("judge_agree", 0), "judged": st.get("judged", 0),
            "fail_reasons_by_attempt": dict(sorted(fails.items())),
            "filter_stages": stages.get(f"story.{c.condition}"),
            "words_mean": round(wm, 2) if wm else None,
            "words_pct_vs_neutral": round(100 * (wm - neu_mean) / neu_mean, 2) if wm else None,
            "words_within_tol": abs(wm - neu_mean) / neu_mean <= 0.05 if wm else None,
            "gate_b_j1b": j1b_gate(mine),
        }
    usage = {k: dict(v) for k, v in p.usage.items()}
    manifest = {
        "run": {"kind": a.kind, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "argv": sys.argv[1:], "out": str(out.relative_to(ROOT)), "wall_seconds": round(wall, 1),
                "seeds": [s["seed_id"] for s in seeds], "dryrun": bool(a.kind == "nouns" and a.dryrun),
                "noun_list": str(noun_src.relative_to(ROOT)) if noun_src else None,
                "noun_list_sha256": sha_file(noun_src) if noun_src else None},
        "generator": {"deployment": a.generator, "model_version": p.model_version.get(a.generator),
                      "api_version": "v1", **p.gen_params()},
        "judge": {"deployment": a.judge, "model_version": p.model_version.get(a.judge), "temperature": 0,
                  "acceptance": "J1 == none", "absolute_pass": "J1B (checks.judge8)", "max_tokens_story": 10,
                  "escalations": p.judge_escalations},
        "inputs_sha256": {"controls.json": sha_file(Path(a.controls)),
                          "k26-v2/seeds.jsonl": sha_file(K26 / "seeds.jsonl"),
                          "k26-v2/stories.jsonl": sha_file(K26 / "stories.jsonl"),
                          "runs/s2_stories.py": sha_file(Path(__file__)),
                          "kimisroom/stimuli.py": sha_file(ROOT / "kimisroom/stimuli.py")},
        "prompt_sha256": {k: S.sha(S.TEMPLATES[k]) for k in ("S_STORY", "P4", "J1", "J1B")},
        "lexicon_sha256": S.sha(json.dumps(S.LEX, sort_keys=True)),
        "neutral_words_mean": round(neu_mean, 2),
        "tokens": usage, "cache": {"generate_hits": p.gen_cache.hits, "generate_misses": p.gen_cache.misses,
                                   "generate_stale": p.gen_cache.stale, "judge_hits": p.judge_cache.hits,
                                   "judge_misses": p.judge_cache.misses},
        "tokens_cached_total": {"generator": dict(p.gen_cache.totals), "judge": dict(p.judge_cache.totals)},
        "truncated_generator_calls": p.truncated, "api_errors": dict(p.api_errors),
        "length_balance": balance_log,
        "kept": len(stories), "rejected": len(p.rejected), "per_control": per,
    }
    S.Pipeline.write_jsonl(out / "stories.jsonl", stories)
    S.Pipeline.write_jsonl(out / "rejected.jsonl",
                           sorted(p.rejected, key=lambda r: (r["condition"], r["unit"], r["attempt"])))
    S.Pipeline.write_jsonl(out / "review_queue.jsonl", p.review_queue(stories))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"tokens": usage, "kept": len(stories), "rejected": len(p.rejected),
                      "wall_seconds": round(wall, 1)}, indent=1))
    for v, d in per.items():
        g = d["gate_b_j1b"]
        print(f"{v:22s} kept {d['kept']:3d}/{d['units']:3d} attempts {d['attempts']:3d} words {d['words_pct_vs_neutral']}% "
              f"J1B emo {g['n_emotion_labels']}/{g['n_kept']} {'pass' if g['pass'] else 'FAIL'} "
              f"fails {d['fail_reasons_by_attempt']}")
    return manifest


def show_prompts(a) -> None:
    spec = load_spec(Path(a.controls))
    if a.kind == "thinking":
        ctls = thinking_controls(spec)
    else:
        ctls = noun_controls(spec, read_noun_list(DRYRUN_LIST)[:2])
        print("(nouns shown with the first two DRY-RUN words, for the template only)\n")
    seed = load_jsonl(K26 / "seeds.jsonl")[0]
    ACTIVE.clear()
    ACTIVE.update({c.condition: c for c in ctls})
    p = S2Pipeline.__new__(S2Pipeline)
    for c in ctls:
        _, user = p.build_prompt(S.Unit("story", c.condition, seed["seed_id"], "P4", seed))
        print(f"===== {c.vec} ({c.status})\n{user}\n")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["thinking", "nouns", "prompts"])
    ap.add_argument("what", nargs="?", choices=["thinking", "nouns"], help="for `prompts`: which set to print")
    ap.add_argument("--controls", default=str(CONTROLS))
    ap.add_argument("--only", default=None, help="thinking: comma list of slugs (default all in controls.json)")
    ap.add_argument("--range", default="N01-N24", help="nouns: list positions, e.g. N01-N24")
    ap.add_argument("--dryrun", action="store_true", help="nouns: the dry-run list, DRYRUN output, plumbing only")
    ap.add_argument("--seed-ids", default=None, help="comma list of k26-v2 seed ids (pilots)")
    ap.add_argument("--seed-every", type=int, default=0, help="every k-th k26-v2 seed (pilots)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--generator", default="gpt-5.6-sol")
    ap.add_argument("--judge", default="DeepSeek-V4-Pro")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--balance-rounds", type=int, default=3, help="rule 6.11 regeneration rounds")
    ap.add_argument("--balance-attempts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--no-calls", action="store_true", help="cache only: a miss leaves the unit pending")
    a = ap.parse_args(argv)
    if a.kind == "prompts":
        a.kind = a.what or "thinking"
        return show_prompts(a)
    return run(a)


if __name__ == "__main__":
    main()
