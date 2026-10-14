"""Study 2 sampled-noun controls (spec v4, section 3 "Sampled nouns", section 11 stamp N, section 12 step 5).

Pipeline, one subcommand per stage, each writing under data/s2/nouns/:

    .venv/bin/python -I runs/s2_nouns.py sources   # verify the three downloaded files against sources.json
    .venv/bin/python -I runs/s2_nouns.py band      # join the norms, apply the registered filter -> band.csv
    .venv/bin/python -I runs/s2_nouns.py screen    # DeepSeek-V4-Pro screen, temperature 0, one call per noun
    .venv/bin/python -I runs/s2_nouns.py match     # whole-word appearance match against what exists now (provisional)
    .venv/bin/python -I runs/s2_nouns.py draw --dry-run --seed 1   # throwaway draw, NOT the registered one
    .venv/bin/python -I runs/s2_nouns.py final --wordings-complete   # after all 20 arguments' wordings and items_s2.jsonl exist:
                                                   # final match, registered draw (seed 20261011), stamp N manifest, ots stamp

Raw files are read only as data: the CSV through the csv module and the XLSX files by parsing their XML
parts out of the zip container (no spreadsheet engine, macros or formulas are evaluated; cached cell values only).

Source note. The spec says the concreteness file carries the SUBTLEX-US count and the dominant part of speech.
The publisher's supplementary file (Springer ESM 1 of doi 10.3758/s13428-013-0403-5) carries the SUBTLEX-US
count but no part-of-speech column; the authors' later text file with a Dom_Pos column was hosted on
crr.ugent.be, which is gone. Dom_Pos in that file was taken from the CLAWS-tagged SUBTLEX-US list (Brysbaert, New
and Keuleers 2012), which Ghent still serves; its Dom_PoS_SUBTLEX column is joined on the word here. The frequency
used for the filter is the SUBTLEX column of the concreteness file, as the spec states.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/s2/nouns"
RAW = OUT / "raw"
SOURCES = OUT / "sources.json"
BAND = OUT / "band.csv"
FILTER_LOG = OUT / "filter_log.json"
PROMPT_FILE = OUT / "screen_prompt.txt"
SCREEN_RAW = OUT / "screen_raw.jsonl"
SCREEN = OUT / "screen.csv"
MATCH_DIR = OUT / "match"
FINAL_DIR = OUT / "final"
DRY_DIR = OUT / "dryrun"
OPENINGS = ROOT / "data/s2/openings"
ITEMS = ROOT / "prompts/battery/items_s2.jsonl"

SEED = 20261011
N_LIST = 60
N_ARGUMENTS = 20

# Registered filter (spec section 3), inclusive bounds.
VAL_LO, VAL_HI = 4.5, 5.6      # Warriner V.Mean.Sum, 1-9
ARO_HI = 4.2                   # Warriner A.Mean.Sum, 1-9
CONC_LO = 4.0                  # Brysbaert Conc.M, 1-5
LOGF_LO = 2.0                  # log10(SUBTLEX count + 1)

FILES = {
    "warriner2013": RAW / "warriner2013/13428_2012_314_MOESM1_ESM.zip",
    "brysbaert2014": RAW / "brysbaert2014/13428_2013_403_MOESM1_ESM.xlsx",
    "subtlexus_pos": RAW / "subtlexus_pos/subtlexus1.zip",
}

JUDGE = "DeepSeek-V4-Pro"
PARAMS = {"temperature": 0}
BUDGET, RETRY_BUDGET = 600, 2500
CATEGORIES = ("person", "occupation", "animal", "body_part", "food", "mental_state_or_emotion")


# ---------------------------------------------------------------- reading the raw files as data

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _col(ref: str) -> int:
    n = 0
    for ch in ref:
        if ch.isalpha():
            n = n * 26 + (ord(ch.upper()) - 64)
        else:
            break
    return n - 1


def read_xlsx(data: bytes, sheet: str = "xl/worksheets/sheet1.xml") -> list[dict]:
    """First sheet as a list of dicts keyed by the header row. Cached values only; formulas are ignored."""
    z = zipfile.ZipFile(io.BytesIO(data))
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(_NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(_NS + "t")))
    rows: list[list] = []
    for _, el in ET.iterparse(io.BytesIO(z.read(sheet)), events=("end",)):
        if el.tag != _NS + "row":
            continue
        cells: dict[int, object] = {}
        for c in el.iter(_NS + "c"):
            t, v = c.get("t"), c.find(_NS + "v")
            if t == "s":
                val: object = shared[int(v.text)] if v is not None else ""
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(_NS + "t"))
            elif t in ("str", "e"):
                val = v.text if v is not None else ""
            else:
                val = float(v.text) if v is not None and v.text not in (None, "") else None
            cells[_col(c.get("r"))] = val
        rows.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
        el.clear()
    header = [str(h) if h is not None else "" for h in rows[0]]
    return [{header[i]: (r[i] if i < len(r) else None) for i in range(len(header))} for r in rows[1:] if r]


def load_warriner() -> list[dict]:
    z = zipfile.ZipFile(FILES["warriner2013"])
    (name,) = z.namelist()
    return list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8"))))


def load_concreteness() -> list[dict]:
    return read_xlsx(FILES["brysbaert2014"].read_bytes())


def load_subtlex_pos() -> list[dict]:
    z = zipfile.ZipFile(FILES["subtlexus_pos"])
    (name,) = z.namelist()
    return read_xlsx(z.read(name))


# ---------------------------------------------------------------- stage 0: sources

def cmd_sources(_a) -> None:
    rec = json.loads(SOURCES.read_text())
    ok = True
    for key, path in FILES.items():
        got = sha256(path)
        want = rec[key]["sha256"]
        print(f"{key:15s} {got}  {'OK' if got == want else 'MISMATCH (want ' + want + ')'}")
        ok &= got == want
    if not ok:
        sys.exit("source hash mismatch")


def assert_sources() -> dict:
    rec = json.loads(SOURCES.read_text())
    for key, path in FILES.items():
        assert sha256(path) == rec[key]["sha256"], f"{path} does not match sources.json"
    return rec


# ---------------------------------------------------------------- stage 1: band

def build_band() -> tuple[list[dict], dict]:
    assert_sources()
    war = {r["Word"].strip(): r for r in load_warriner()}
    conc_rows = load_concreteness()
    conc: dict[str, dict] = {}
    for r in conc_rows:
        w = str(r["Word"]).strip() if r["Word"] is not None else ""
        if w:
            conc.setdefault(w, r)
    pos: dict[str, str] = {}
    for r in load_subtlex_pos():
        w = str(r["Word"]).strip() if r["Word"] is not None else ""
        if w and w not in pos:
            pos[w] = str(r.get("Dom_PoS_SUBTLEX") or "")

    stages: list[tuple[str, int]] = [("warriner_words", len(war)), ("concreteness_words", len(conc)),
                                     ("subtlex_pos_words", len(pos))]
    joined = sorted(set(war) & set(conc))
    stages.append(("joined_warriner_x_concreteness", len(joined)))

    rows = []
    for w in joined:
        c = conc[w]
        rows.append({
            "word": w,
            "valence": float(war[w]["V.Mean.Sum"]),
            "arousal": float(war[w]["A.Mean.Sum"]),
            "concreteness": float(c["Conc.M"]),
            "bigram": int(c["Bigram"] or 0),
            "subtlex_count": int(c["SUBTLEX"] or 0),
            "dom_pos": pos.get(w, ""),
        })
    for r in rows:
        r["log10_freq"] = math.log10(r["subtlex_count"] + 1)

    steps = [
        ("dominant_pos_noun", lambda r: r["dom_pos"] == "Noun"),
        ("single_word", lambda r: r["bigram"] == 0 and " " not in r["word"] and "-" not in r["word"]),
        (f"valence_{VAL_LO}_to_{VAL_HI}", lambda r: VAL_LO <= r["valence"] <= VAL_HI),
        (f"arousal_le_{ARO_HI}", lambda r: r["arousal"] <= ARO_HI),
        (f"concreteness_ge_{CONC_LO}", lambda r: r["concreteness"] >= CONC_LO),
        (f"log10_count_plus_1_ge_{LOGF_LO}", lambda r: r["log10_freq"] >= LOGF_LO),
    ]
    no_pos = sum(r["dom_pos"] == "" for r in rows)
    for name, keep in steps:
        rows = [r for r in rows if keep(r)]
        stages.append((name, len(rows)))
    log = {"stages": stages, "joined_without_pos_entry": no_pos,
           "nonalpha_in_band": [r["word"] for r in rows if not r["word"].isalpha()],
           "join": "exact word string after strip; Warriner x concreteness inner join; PoS left-joined from SUBTLEX-US",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return rows, log


def cmd_band(_a) -> None:
    rows, log = build_band()
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["word", "valence", "arousal", "concreteness", "subtlex_count", "log10_freq", "dom_pos"]
    with BAND.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "log10_freq": f"{r['log10_freq']:.4f}"})
    log["band_size"] = len(rows)
    log["band_sha256"] = sha256(BAND)
    FILTER_LOG.write_text(json.dumps(log, indent=1) + "\n")
    for name, n in log["stages"]:
        print(f"{name:36s} {n:6d}")
    print(f"band -> {BAND.relative_to(ROOT)} ({len(rows)} nouns); non-alphabetic in band: {log['nonalpha_in_band']}")


def read_band() -> list[str]:
    with BAND.open() as f:
        return [r["word"] for r in csv.DictReader(f)]


# ---------------------------------------------------------------- stage 2: language-model screen

PROMPT = """You are screening words for a psychology-of-AI experiment. We need neutral, concrete control concepts, so we remove nouns that denote people, occupations, animals, body parts, foods, or mental states and emotions.

The word is: {word}

Consider the word as a noun. Look at its main noun sense and at any other common noun sense. Exclude the word if any of those senses denotes one of the following:
- "person": a person or group of people, including kinship terms, social roles, nationalities and named kinds of people
- "occupation": a job, trade or professional role
- "animal": an animal of any kind, including insects, fish and birds, or a group of animals
- "body_part": a part of a human or animal body, including organs, bones, fluids and tissues
- "food": a food or drink, a dish, an ingredient, or something commonly eaten or drunk
- "mental_state_or_emotion": a mental state, emotion, mood, feeling, attitude or mental faculty

Otherwise keep it. Things such as objects, places, materials, tools, plants, buildings, vehicles, weather and natural features are kept unless one of their common noun senses falls in a category above.

Return only this JSON and nothing else:
{{"word": "{word}", "exclude": true or false, "category": one of "person", "occupation", "animal", "body_part", "food", "mental_state_or_emotion", or "none", "why": "one sentence"}}"""


def write_prompt_file() -> str:
    """The stamped prompt: the template text exactly as formatted per noun, plus judge and params."""
    body = (f"# Study 2 noun screen prompt (runs/s2_nouns.py)\n# judge: {JUDGE}\n# params: {json.dumps(PARAMS)}"
            f"\n# max_tokens: {BUDGET}, retried once at {RETRY_BUDGET} if the reply has no parseable JSON\n"
            f"# one call per noun; {{word}} is replaced by the noun\n\n{PROMPT}\n")
    if PROMPT_FILE.exists() and PROMPT_FILE.read_text() != body:
        sys.exit(f"{PROMPT_FILE} differs from the prompt in this script; refusing to mix prompts (move it aside first)")
    PROMPT_FILE.write_text(body)
    return sha256(PROMPT_FILE)


def parse(raw: str, word: str) -> dict | None:
    for block in reversed(re.findall(r"\{[^{}]*\}", raw, re.DOTALL)):
        try:
            d = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and isinstance(d.get("exclude"), bool):
            cat = str(d.get("category") or "none")
            if cat not in CATEGORIES + ("none",):
                continue
            if d["exclude"] and cat == "none":
                continue
            return {"exclude": d["exclude"], "category": cat if d["exclude"] else "none",
                    "why": str(d.get("why") or ""), "word_echo": str(d.get("word") or "")}
    return None


def screen_one(word: str, prompt_sha: str) -> dict:
    from kimisroom.foundry import chat, text
    prompt = PROMPT.format(word=word)
    calls = []
    v = None
    for budget in (BUDGET, RETRY_BUDGET):
        req = {"model": JUDGE, "messages": [{"role": "user", "content": prompt}], "max_tokens": budget, **PARAMS}
        r = chat(JUDGE, req["messages"], retries=10, max_tokens=budget, **PARAMS)
        calls.append({"request": req, "response": r})
        raw = text(r) or (r["choices"][0]["message"].get("reasoning_content") or "")[-2000:]
        v = parse(raw, word)
        if v:
            break
    if v is None:
        v = {"exclude": None, "category": "unparseable", "why": raw[:300], "word_echo": ""}
    out = {"word": word, "key": sha_text(prompt), "prompt_file_sha256": prompt_sha, "judge": JUDGE,
           "params": PARAMS, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **v, "calls": calls}
    return out


def load_screen_cache() -> dict:
    cache = {}
    if SCREEN_RAW.exists():
        for line in SCREEN_RAW.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["key"]] = d
    return cache


def cmd_screen(a) -> None:
    prompt_sha = write_prompt_file()
    words = read_band()
    if a.limit:
        words = words[: a.limit]
    cache = load_screen_cache()
    todo = [w for w in words if sha_text(PROMPT.format(word=w)) not in cache
            or cache[sha_text(PROMPT.format(word=w))]["exclude"] is None]
    print(f"{len(words)} nouns, {len(words) - len(todo)} cached, {len(todo)} to call ({JUDGE}, temperature 0)")
    done = 0
    with ThreadPoolExecutor(a.workers) as ex, SCREEN_RAW.open("a") as f:
        for rec in ex.map(lambda w: screen_one(w, prompt_sha), todo):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            cache[rec["key"]] = rec
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    write_screen_table(read_band(), cache)


def write_screen_table(words: list[str], cache: dict) -> None:
    rows = []
    for w in words:
        d = cache.get(sha_text(PROMPT.format(word=w)))
        if d is None:
            continue
        rows.append({"word": w, "exclude": d["exclude"], "category": d["category"], "why": d["why"],
                     "response_id": d["calls"][-1]["response"].get("id", ""),
                     "model_version": d["calls"][-1]["response"].get("model", "")})
    with SCREEN.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    n = len(rows)
    kept = sum(r["exclude"] is False for r in rows)
    unp = sum(r["exclude"] is None for r in rows)
    by = {}
    for r in rows:
        if r["exclude"]:
            by[r["category"]] = by.get(r["category"], 0) + 1
    print(f"screened {n}: kept {kept} ({kept / n:.1%}), excluded {n - kept - unp}, unparseable {unp}; by category {by}")


def screen_passed() -> list[str]:
    cache = load_screen_cache()
    words = read_band()
    missing = [w for w in words if sha_text(PROMPT.format(word=w)) not in cache]
    bad = [w for w in words if w not in missing and cache[sha_text(PROMPT.format(word=w))]["exclude"] is None]
    if missing or bad:
        sys.exit(f"screen incomplete: {len(missing)} unscreened, {len(bad)} unparseable; run `screen` first")
    return [w for w in words if cache[sha_text(PROMPT.format(word=w))]["exclude"] is False]


# ---------------------------------------------------------------- stage 3: appearance match

def regular_plural(w: str) -> str:
    if re.search(r"(s|x|z|ch|sh)$", w):
        return w + "es"
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    return w + "s"


def _strings(x) -> list[str]:
    """Every string value in an item record (keys are field names, not item text, so they are skipped)."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    return []


def corpus() -> tuple[list[tuple[str, str]], dict]:
    """(source label, text) for every argument wording under data/s2/openings and every items_s2.jsonl item."""
    texts: list[tuple[str, str]] = []
    wording_files = sorted(OPENINGS.rglob("*.md")) if OPENINGS.exists() else []
    for p in wording_files:
        texts.append((str(p.relative_to(ROOT)), p.read_text()))
    n_items = 0
    if ITEMS.exists():
        for i, line in enumerate(ITEMS.read_text().splitlines()):
            if not line.strip():
                continue
            d = json.loads(line)
            n_items += 1
            label = f"{ITEMS.relative_to(ROOT)}:{d.get('id', i + 1)}"
            texts.append((label, "\n".join(_strings(d))))
    per_dir: dict[str, int] = {}
    for p in wording_files:
        k = p.relative_to(OPENINGS).parts[0] if len(p.relative_to(OPENINGS).parts) > 1 else "."
        per_dir[k] = per_dir.get(k, 0) + 1
    arg_dirs = sorted(k for k in per_dir if k != ".")
    info = {"wording_files": len(wording_files), "argument_dirs": arg_dirs, "files_per_dir": per_dir, "items": n_items,
            "items_file_exists": ITEMS.exists(),
            "items_sha256": sha256(ITEMS) if ITEMS.exists() else None,
            "wording_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in wording_files}}
    return texts, info


def run_match(words: list[str], outdir: Path) -> list[str]:
    texts, info = corpus()
    outdir.mkdir(parents=True, exist_ok=True)
    kept, log = [], []
    for w in words:
        pl = regular_plural(w)
        rx = re.compile(rf"\b(?:{re.escape(w)}|{re.escape(pl)})\b", re.IGNORECASE)
        hits = []
        for label, t in texts:
            for m in rx.finditer(t):
                s, e = max(0, m.start() - 40), min(len(t), m.end() + 40)
                hits.append({"source": label, "match": m.group(0), "context": t[s:e].replace("\n", " ")})
        if hits:
            log.append({"word": w, "plural": pl, "removed": True, "reason": "appears in wordings or battery",
                        "n_hits": len(hits), "hits": hits})
        else:
            kept.append(w)
    (outdir / "match_log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in log))
    (outdir / "corpus.json").write_text(json.dumps(info, indent=1) + "\n")
    (outdir / "screened_band.txt").write_text("".join(w + "\n" for w in kept))
    print(f"corpus files per dir: {info['files_per_dir']}")
    print(f"corpus: {info['wording_files']} wording files in {len(info['argument_dirs'])} argument dirs, "
          f"{info['items']} battery items (items_s2.jsonl exists: {info['items_file_exists']})")
    print(f"match: {len(words)} screen-passed nouns -> {len(kept)} after removing {len(log)} that appear")
    for r in log[:15]:
        print(f"  removed {r['word']:14s} {r['n_hits']:3d} hits, e.g. {r['hits'][0]['source']}: ...{r['hits'][0]['context']}...")
    return kept


def cmd_match(_a) -> None:
    run_match(screen_passed(), MATCH_DIR)
    print(f"(provisional: wordings are incomplete; rerun with `final` once they exist) -> {MATCH_DIR.relative_to(ROOT)}")


# ---------------------------------------------------------------- stage 4: draw

def draw(screened: list[str], seed: int) -> list[str]:
    import numpy as np
    srt = sorted(screened)
    perm = np.random.default_rng(seed).permutation(len(srt))
    return [srt[i] for i in perm][:N_LIST]


def write_list(ordered: list[str], path: Path, header: str) -> None:
    path.write_text(header + "".join(f"N{i + 1:02d}\t{w}\n" for i, w in enumerate(ordered)))


def cmd_draw(a) -> None:
    if not a.dry_run:
        sys.exit("the registered draw runs only inside `final`; use --dry-run --seed <throwaway> to exercise it")
    if a.seed == SEED:
        sys.exit(f"--seed {SEED} is the registered seed; a dry run must use a throwaway seed")
    src = MATCH_DIR / "screened_band.txt"
    screened = [w for w in src.read_text().split("\n") if w] if src.exists() else screen_passed()
    ordered = draw(screened, a.seed)
    DRY_DIR.mkdir(parents=True, exist_ok=True)
    out = DRY_DIR / f"DRYRUN_seed{a.seed}_NOT_REGISTERED.tsv"
    write_list(ordered, out, f"# DRY RUN, throwaway seed {a.seed}, NOT the registered draw (seed {SEED});"
                             f" provisional screened band of {len(screened)}\n")
    print(f"dry run -> {out.relative_to(ROOT)}: {len(ordered)} nouns; first 10: {', '.join(ordered[:10])}")


# ---------------------------------------------------------------- final: match, registered draw, stamp N

def cmd_final(a) -> None:
    rec = assert_sources()
    texts, info = corpus()
    problems = []
    if not info["items_file_exists"]:
        problems.append(f"{ITEMS.relative_to(ROOT)} does not exist")
    if len(info["argument_dirs"]) < N_ARGUMENTS:
        problems.append(f"wordings found for {len(info['argument_dirs'])} argument dirs, need {N_ARGUMENTS}")
    seed_only = [d for d in info["argument_dirs"]
                 if not [p for p in (OPENINGS / d).rglob("*.md") if not p.name.startswith("seed")]]
    if seed_only:
        problems.append(f"argument dirs with no wording beyond a seed draft: {seed_only}")
    if not a.wordings_complete:
        problems.append("pass --wordings-complete to confirm every wording of all 20 arguments has been generated"
                        f" (files per dir now: {info['files_per_dir']})")
    if problems:
        sys.exit("refusing the registered draw: " + "; ".join(problems))
    if FINAL_DIR.exists() and any(FINAL_DIR.iterdir()):
        sys.exit(f"{FINAL_DIR} already holds a registered draw; it is never redone in place")

    # Re-derive the band from the raw files and check it equals the band the screen ran on.
    rows, flog = build_band()
    assert [r["word"] for r in rows] == read_band(), "band.csv does not match a fresh derivation from the raw files"
    screened = run_match(screen_passed(), FINAL_DIR)
    ordered = draw(screened, SEED)
    lst = FINAL_DIR / "noun_list.tsv"
    write_list(ordered, lst, f"# Study 2 sampled nouns N01..N{len(ordered):02d}; seed {SEED}; screened band "
                             f"{len(screened)}; drawn {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    sizes = {"band": len(rows), "screen_passed": len(screen_passed()), "screened_band": len(screened),
             "list": len(ordered)}
    (FINAL_DIR / "sizes.json").write_text(json.dumps({**sizes, "filter_stages": flog["stages"]}, indent=1) + "\n")

    manifest = FINAL_DIR / "stampN_manifest.txt"
    entries = [(f"source:{k}", rec[k]["sha256"], rec[k]["url"]) for k in FILES]
    for p in [Path(__file__), SOURCES, BAND, FILTER_LOG, PROMPT_FILE, SCREEN_RAW, SCREEN,
              FINAL_DIR / "match_log.jsonl", FINAL_DIR / "corpus.json", FINAL_DIR / "screened_band.txt",
              FINAL_DIR / "sizes.json", lst]:
        entries.append((str(p.relative_to(ROOT)), sha256(p), ""))
    body = (f"# Study 2 stamp N (spec section 11), {time.strftime('%Y-%m-%dT%H:%M:%S')}\n# seed {SEED}\n"
            f"# sizes {json.dumps(sizes)}\n"
            + "".join(f"{h}  {name}{('  ' + u) if u else ''}\n" for name, h, u in entries)
            + "# ordered list\n" + "".join(f"# N{i + 1:02d} {w}\n" for i, w in enumerate(ordered)))
    manifest.write_text(body)
    print(f"registered list -> {lst.relative_to(ROOT)} ({len(ordered)} nouns); manifest -> {manifest.relative_to(ROOT)}")
    if a.no_stamp:
        print("--no-stamp: skipped `ots stamp`")
        return
    subprocess.run(["ots", "stamp", str(manifest)], check=True)
    print(f"stamped: {manifest.relative_to(ROOT)}.ots")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sources").set_defaults(fn=cmd_sources)
    sub.add_parser("band").set_defaults(fn=cmd_band)
    s = sub.add_parser("screen")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--limit", type=int, default=0, help="screen only the first N band nouns (testing)")
    s.set_defaults(fn=cmd_screen)
    sub.add_parser("match").set_defaults(fn=cmd_match)
    d = sub.add_parser("draw")
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--seed", type=int, required=True)
    d.set_defaults(fn=cmd_draw)
    f = sub.add_parser("final")
    f.add_argument("--no-stamp", action="store_true")
    f.add_argument("--wordings-complete", action="store_true",
                   help="operator's confirmation that wording generation (all redraws) is finished")
    f.set_defaults(fn=cmd_final)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
