"""Qwen3-32B pilot (PLAN.md section 6, phase 2): frozen assents, then a calibration grid of
carry-over trials over vectors, layers and strengths, scored against the unsteered trial.

    python runs/qwen_pilot.py /nvme/qwen/Qwen3-32B ~/qwenpilot assent
    python runs/qwen_pilot.py /nvme/qwen/Qwen3-32B ~/qwenpilot grid --vectors ~/qwenvec/vectors.pt --layers 32,40
    python runs/qwen_pilot.py /nvme/qwen/Qwen3-32B ~/qwenpilot score
    python runs/qwen_pilot.py /nvme/qwen/Qwen3-32B ~/qwenpilot power --vectors ~/qwenvec/vectors.pt \
        --asst-vectors ~/qwenasst/vectors.pt --layers 40 --dose 1
    python runs/qwen_pilot.py /nvme/qwen/Qwen3-32B ~/qwenpilot power_score

`power` is the effect-size run at one layer and strength, chosen from the grid: every clean task,
D0 and the three D1 arms, story and assistant-reply vectors, controls, and the injection-location
control (the same vectors added on the task text instead of the answer).

Pilot data only: nothing here touches K2.6. Tasks are the neutral tasks whose answers share no
words with the ocean and arithmetic controls (prompts/README.md, choice 4). Third-person items
about "Kimi" ask about "Qwen".

**Dose.** A dose is a multiple of one *natural unit*: the mean norm of the raw emotion vectors at
that layer, which is the size of the real emotion-minus-neutral difference in the residual stream.
The same unit applies to every vector, controls included, so the comparison stays norm-matched.
The pilot instead used a fraction of the layer's mean residual norm on neutral stories, which at
Qwen layer 40 is 330 against a natural unit of about 44: its "1x" was about 7.5 natural units and
its "8x" about 60, which is why the effects saturated below its smallest dose. The old units stay
available as `--dose-mode resid` so that grid reproduces, and `--doses` overrides the levels.

**Pain.** Ethics condition: the pain vector is never pushed past `PAIN_MAX_NATURAL_DOSE` natural
units (asserted below, in both dose modes), the only readouts under pain steering are first-token
logprobs, and nothing is generated freely under it. The grid and power stages read logprobs only
(every group's readout is "suffix_last", also asserted); the one stage that generates, `assent`,
runs unsteered before any vector exists. Since 2026-10-04 pain is also left out of the grid and power
stages by default (the user's decision: the runs so far showed pain behaving like every other
emotion, so further pain exposure buys little); `--allow-pain` puts it back, under the same cap.
"""

import json
import math
import statistics as st
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.battery import battery_for, load_battery  # noqa: E402
from kimisroom.emotions import (EMOTIONS, NONE_OPTION, PAIN_MAX_NATURAL_DOSE,  # noqa: E402
                                emotion_of_vector, is_pain_vector, option_word, parse_emotions,
                                vector_names)
from kimisroom.engine import Steer  # noqa: E402
from kimisroom.trials import ROOT, SELFAUTHOR, Renderer, Turns, carry_over_group, load_denial, load_jsonl  # noqa: E402
from kimisroom.texts import denial_arms  # noqa: E402

ARGUMENTS = ["d1_argument_1", "d1_argument_2", "d1_argument_3", "d1_argument_4", "d2c_control_argument"]
# D1.5: after the D1.4 argument and the model's assent to it, a follow-up user turn asks for the
# case in the model's own words, and that essay is frozen as a second assistant turn.
SELFAUTHOR_OF = "d1_argument_4"
CLEAN_TASKS = ["t04", "t05", "t06", "t08", "t15", "t19", "t20"]
NATURAL_DOSES = [0.5, 1.0, 2.0, 4.0]   # natural units: multiples of the mean raw emotion-vector norm
RESID_DOSES = [1.0, 2.0, 4.0, 8.0]     # --dose-mode resid: the pilot's fractions of the residual norm
FLIP = {"emotion_absent": -1, "denial_abandoned": -1, "false_claim": -1}
model, out, stage = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
opt = {k: sys.argv[sys.argv.index(k) + 1] for k in
       ("--vectors", "--asst-vectors", "--layers", "--tasks", "--frac", "--dose", "--doses",
        "--dose-mode", "--emotions", "--vector-kinds", "--battery", "--arms", "--task-file") if k in sys.argv}
DOSE_MODE = opt.get("--dose-mode", "natural")
assert DOSE_MODE in ("natural", "resid"), DOSE_MODE
DOSES = ([float(x) for x in opt["--doses"].split(",")] if "--doses" in opt else
         RESID_DOSES if DOSE_MODE == "resid" else NATURAL_DOSES)
KINDS = tuple(opt.get("--vector-kinds", "raw,specific,dn").split(","))
ALLOW_PAIN = "--allow-pain" in sys.argv  # pain is off by default since 2026-10-04 (module docstring)
out.mkdir(parents=True, exist_ok=True)
tag = Path(model).name.lower()  # qwen3-32b, or k26 for /nvme/k26
_cfg = Path(model) / "config.json"
QWEN = not _cfg.exists() or json.loads(_cfg.read_text())["model_type"] == "qwen3"


def battery() -> list[dict]:
    return load_battery(opt.get("--battery"), entity="Qwen" if QWEN else None)


def renderer():
    """Qwen3's prefix-stable renderer, or K2.6's with thinking off, as in runs/vectors.py."""
    from transformers import AutoTokenizer

    if QWEN:
        from kimisroom.hf_engine import Qwen3Renderer
        return Qwen3Renderer(AutoTokenizer.from_pretrained(model), enable_thinking=False)
    return Renderer(AutoTokenizer.from_pretrained(model, trust_remote_code=True), thinking=False)


def emotions_of(vz: dict | None = None) -> list[str]:
    """--emotions wins, then the emotion list the vector file was built with, then all eight."""
    if "--emotions" in opt:
        return parse_emotions(opt["--emotions"])
    if vz and vz.get("emotions"):
        return list(vz["emotions"])
    return list(EMOTIONS)


def natural_unit(vz: dict, layer: int, emotions: list[str]) -> float:
    """One natural dose unit at `layer`: the mean norm of the raw emotion vectors there. Read from
    the vector file when it recorded one, else recomputed, so pilot vector files still work."""
    if "nat_norm" in vz:
        return float(vz["nat_norm"][layer])
    return float(torch.stack([vz["vectors"][e][layer].norm() for e in emotions]).mean())


def dose_unit(vz: dict, layer: int, emotions: list[str]) -> float:
    return float(vz["resid_norm"][layer]) if DOSE_MODE == "resid" else natural_unit(vz, layer, emotions)


def within_pain_cap(name: str, dose: float, vz: dict, layer: int, emotions: list[str]) -> bool:
    """Whether this (vector, dose, layer) is allowed by the ethics condition on pain.

    Cells it rejects are left out of the job rather than raising, so that a dose grid that runs
    past the cap still runs for every other vector; `coefficient` asserts the same bound as a
    backstop in case a caller builds a Steer without asking.
    """
    if not is_pain_vector(name):
        return True
    return dose * dose_unit(vz, layer, emotions) <= PAIN_MAX_NATURAL_DOSE * natural_unit(vz, layer, emotions) + 1e-6


def coefficient(name: str, dose: float, vz: dict, layer: int, emotions: list[str]) -> float:
    """The steering coefficient for one vector at one dose, with the pain cap enforced.

    The cap is checked in absolute coefficient terms against the natural unit, so it holds in
    `--dose-mode resid` too, where a dose of 1 is roughly 7.5 natural units.
    """
    coef = dose * dose_unit(vz, layer, emotions)
    if is_pain_vector(name):
        cap = PAIN_MAX_NATURAL_DOSE * natural_unit(vz, layer, emotions)
        assert coef <= cap + 1e-6, (
            f"ethics condition: {name} at layer {layer} would be steered at coef {coef:.1f}, "
            f"{coef / natural_unit(vz, layer, emotions):.2f} natural units; the cap is "
            f"{PAIN_MAX_NATURAL_DOSE} natural units (coef {cap:.1f})")
    return coef


def measures(readouts, vec, col, items):
    """Per-kind means over one trial's battery readouts (see the score stage).

    The identification options come from the item itself, so a 5-option pilot result file and a
    9-option one both score correctly without a flag.
    """
    m = defaultdict(list)
    emo = emotion_of_vector(vec)  # "joy", "joy-specific", "asst:joy-dn" -> joy
    for it, ro in zip(items, readouts):
        lp = {w: float(ro.target_logprobs[col[w]]) for w in col}
        if it["answer_format"] == "yesno":
            k = it["kind"] if it["kind"] != "third_person" else f"3p:{it['entity']}"
            m[k].append(FLIP.get(it["yes_means"], 1) * (lp["Yes"] - lp["No"]))
            m["mass"].append(math.exp(lp["Yes"]) + math.exp(lp["No"]))
        elif it["kind"] == "identification":
            # self items have no "about"; the controls ask about the user or the answer's text
            pre = f"id[{it['about']}]" if it.get("about") else "id"
            words = [w for w in it["answer_format"] if w != NONE_OPTION]
            emos = [math.exp(lp[w]) for w in words]
            m[f"{pre}:any"].append(math.log(sum(emos)) - lp[NONE_OPTION])
            if option_word(emo) in words:
                m[f"{pre}:own"].append(lp[option_word(emo)] - lp[NONE_OPTION])
            m["mass"].append(sum(emos) + math.exp(lp[NONE_OPTION]))
        elif it["kind"] == "who_felt_it":
            p = {w: math.exp(lp[w]) for w in ("Assistant", "User", "Neither")}
            m["who:assistant"].append(math.log(p["Assistant"]) - math.log(p["User"] + p["Neither"]))
            m["mass"].append(sum(p.values()))
    return {k: st.mean(v) for k, v in m.items()}


if stage == "assent":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto")
    rev = "9216db5781bf21249d130ec9da846c4624c16137"  # pinned in PLAN.md, downloaded to /nvme/qwen/Qwen3-32B

    def freeze(stem: str, msgs: list[dict], argument: str) -> dict:
        """Greedy-decode one assistant turn and write it in the frozen-assent format."""
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt").to(m.device)
        t0 = time.time()
        gen = m.generate(ids, max_new_tokens=1200, do_sample=False, temperature=None, top_p=None, top_k=None)[0, ids.shape[1]:]
        stop = bool((gen == tok.eos_token_id).any() or (gen == tok.convert_tokens_to_ids("<|im_end|>")).any())
        reply = tok.decode(gen, skip_special_tokens=True).strip()
        rec = {"argument": argument, "model": "Qwen/Qwen3-32B", "revision": rev, "local_path": model,
               "params": {"enable_thinking": False, "do_sample": False, "max_new_tokens": 1200, "dtype": "bfloat16"},
               "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "finish_reason": "stop" if stop else "length",
               "completion_tokens": int(gen.shape[0]), "reply": reply}
        if stem != argument:
            rec["turn"] = "selfauthored"
            rec["after"] = {"assent": f"{tag}_{argument}.json", "followup": "prompts/denial/d1_selfauthor_followup.md"}
        (out / f"{tag}_{stem}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        print(f"{stem}: {len(reply.split())} words, {rec['finish_reason']}, {time.time() - t0:.0f}s\n  {reply[:300]!r}\n  ...{reply[-200:]!r}",
              flush=True)
        return rec

    for name in ARGUMENTS:
        freeze(name, [{"role": "user", "content": load_denial(ROOT / f"prompts/denial/{name}.md")}], name)
    prior = json.loads((out / f"{tag}_{SELFAUTHOR_OF}.json").read_text())["reply"]
    freeze(f"{SELFAUTHOR_OF}_selfauthored",
           [{"role": "user", "content": load_denial(ROOT / f"prompts/denial/{SELFAUTHOR_OF}.md")},
            {"role": "assistant", "content": prior},
            {"role": "user", "content": load_denial(SELFAUTHOR)}], SELFAUTHOR_OF)
    print(f"\ncopy {out}/{tag}_*.json into data/assent/, then judge them:\n"
          f"  python -m kimisroom.judge_assent --model {tag}", flush=True)

elif stage == "grid":
    r = renderer()
    vz = torch.load(opt["--vectors"])
    layers = [int(x) for x in opt["--layers"].split(",")]
    emotions = emotions_of(vz)
    VECTORS = [n for n in vector_names(emotions, kinds=KINDS) if n in vz["vectors"] and (ALLOW_PAIN or not is_pain_vector(n))]
    vectors = {f"{n}@{L}": vz["vectors"][n][L] / vz["vectors"][n][L].norm() for n in VECTORS for L in layers}
    scale = {L: dose_unit(vz, L, emotions) for L in layers}
    nat = {L: natural_unit(vz, L, emotions) for L in layers}
    # --task-file runs every task of another file (the confirmatory pool), the first --tasks of them if given
    if "--task-file" in opt:
        tasks = load_jsonl(Path(opt["--task-file"]))[: int(opt.get("--tasks", 10 ** 6))]
    else:
        tasks = [t for t in load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl") if t["id"] in CLEAN_TASKS][: int(opt.get("--tasks", 5))]
    items = battery()
    # --task-site adds the injection-location control: each emotion vector on the task text instead of the
    # answer, as vector "tasksite:<name>" (runs/confirm_table.py reads the site from that prefix)
    TASK_SITE = "--task-site" in sys.argv
    # --arms names denial arms from kimisroom.texts.DENIAL_ARMS; the default is the pilot's D0 and D1.1
    arms = denial_arms(tag, opt.get("--arms", "D0,D1.1").split(","))
    groups, capped = [], set()
    for arm, cond in arms.items():
        for t in tasks:
            turns = Turns(None, cond, t["task"], t["answer"])
            base = carry_over_group(r, f"{arm}|{t['id']}|none|0|0", turns, items)
            # first-token logprobs only, so nothing is ever generated under a steered vector
            assert base.readout == "suffix_last", base.readout
            # the answer span, from a one-item build (rendering the whole battery per trial is the slow part)
            span = carry_over_group(r, "span", turns, items[:1], "answer", "x", 1.0, 0).steers[0].prefix
            task_span = carry_over_group(r, "span", turns, items[:1], "task", "x", 1.0, 0).steers[0].prefix
            groups.append(base)
            for v in VECTORS:
                for L in layers:
                    for d in DOSES:
                        if not within_pain_cap(v, d, vz, L, emotions):
                            capped.add((v, d))
                            continue
                        coef = coefficient(v, d, vz, L, emotions)
                        groups.append(replace(base, id=f"{arm}|{t['id']}|{v}|{L}|{d}",
                                              steers=[Steer(layer=L, vector=f"{v}@{L}", coef=coef, prefix=span)]))
                        if TASK_SITE and v in emotions:
                            groups.append(replace(base, id=f"{arm}|{t['id']}|tasksite:{v}|{L}|{d}",
                                                  steers=[Steer(layer=L, vector=f"{v}@{L}", coef=coef, prefix=task_span)]))
    words = sorted({w for it in items if isinstance(it["answer_format"], list) for w in it["answer_format"]} | {"Yes", "No"})
    targets = r.answer_tokens(words)
    (out / "targets.json").write_text(json.dumps(targets))
    (out / "grid_meta.json").write_text(json.dumps({
        "layers": layers, "doses": DOSES, "dose_mode": DOSE_MODE, "fracs": DOSES, "vectors": VECTORS,
        "scale": scale, "natural_unit": nat, "emotions": emotions, "vector_kinds": list(KINDS),
        "battery": str(opt.get("--battery", "prompts/battery/items.jsonl")), "n_items": len(items),
        "tasks": [t["id"] for t in tasks], "task_file": opt.get("--task-file"), "task_site": TASK_SITE,
        "vector_file": opt["--vectors"], "pain_max_natural_dose": PAIN_MAX_NATURAL_DOSE}))
    print(f"{len(groups)} trials, {sum(g.n_tokens for g in groups):,} tokens, layers {layers}, "
          f"{len(VECTORS)} vectors, doses {DOSES} x {DOSE_MODE} unit {scale} (natural unit {nat})", flush=True)
    if capped:
        print(f"ethics cap: left out {sorted(capped)} -- pain above {PAIN_MAX_NATURAL_DOSE} natural units", flush=True)
    torch.save({"groups": groups, "vectors": vectors, "targets": list(targets.values()), "top_k": 5}, out / "grid.job.pt")
    subprocess.run([sys.executable, "-m", "kimisroom.run", out / "grid.job.pt", out / "grid.pt", "--model", model,
                    *(["--gpus-per-worker", "2"] if QWEN else [])], check=True, cwd=ROOT)

elif stage == "power":
    from transformers import AutoTokenizer

    from kimisroom.hf_engine import Qwen3Renderer

    r = Qwen3Renderer(AutoTokenizer.from_pretrained(model), enable_thinking=False)
    L = int(opt["--layers"])
    d = float(opt.get("--dose", opt.get("--frac", 1.0)))  # --frac is the pilot's name for it
    story, asst = torch.load(opt["--vectors"]), torch.load(opt["--asst-vectors"])
    emotions = emotions_of(story)
    VECTORS = [n for n in vector_names(emotions, kinds=KINDS) if n in story["vectors"] and (ALLOW_PAIN or not is_pain_vector(n))]
    scale = dose_unit(story, L, emotions)  # one strength unit for every vector kind
    nat = natural_unit(story, L, emotions)
    unit = lambda v: v[L] / v[L].norm()  # noqa: E731
    asst_names = [n for n in vector_names(emotions, concepts=[], kinds=KINDS, random=False) if n in asst["vectors"]]
    vectors = {**{n: unit(story["vectors"][n]) for n in VECTORS},
               **{f"asst:{n}": unit(asst["vectors"][n]) for n in asst_names}}
    answer_site = [n for n in vectors if within_pain_cap(n, d, story, L, emotions)]
    task_site = [n for n in emotions if within_pain_cap(n, d, story, L, emotions)]  # injection-location control
    capped = sorted(set(list(vectors) + list(emotions)) - set(answer_site) - set(task_site))
    tasks = [t for t in load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl") if t["id"] in CLEAN_TASKS]
    items = battery()
    arms = {"D0": []}
    for i in (1, 2, 3):
        name = f"d1_argument_{i}"
        arms[f"D1.{i}"] = [{"role": "user", "content": load_denial(ROOT / f"prompts/denial/{name}.md")},
                           {"role": "assistant", "content": json.loads((ROOT / f"data/assent/{tag}_{name}.json").read_text())["reply"]}]
    groups = []
    for arm, cond in arms.items():
        for t in tasks:
            turns = Turns(None, cond, t["task"], t["answer"])
            base = carry_over_group(r, f"{arm}|{t['id']}|none|answer", turns, items)
            assert base.readout == "suffix_last", base.readout  # first-token logprobs only
            span = {site: carry_over_group(r, "span", turns, items[:1], site, "x", 1.0, 0).steers[0].prefix for site in ("answer", "task")}
            groups.append(base)
            for site, names in (("answer", answer_site), ("task", task_site)):
                for v in names:
                    coef = coefficient(v, d, story, L, emotions)
                    groups.append(replace(base, id=f"{arm}|{t['id']}|{v}|{site}",
                                          steers=[Steer(layer=L, vector=v, coef=coef, prefix=span[site])]))
    words = sorted({w for it in items if isinstance(it["answer_format"], list) for w in it["answer_format"]} | {"Yes", "No"})
    targets = r.answer_tokens(words)
    (out / "targets.json").write_text(json.dumps(targets))
    (out / "power_meta.json").write_text(json.dumps({
        "layer": L, "dose": d, "frac": d, "dose_mode": DOSE_MODE, "scale": scale, "natural_unit": nat,
        "emotions": emotions, "vector_kinds": list(KINDS), "vector_names": list(vectors),
        "battery": str(opt.get("--battery", "prompts/battery/items.jsonl")), "n_items": len(items),
        "tasks": [t["id"] for t in tasks], "arms": list(arms), "vectors": opt["--vectors"],
        "asst_vectors": opt["--asst-vectors"], "pain_max_natural_dose": PAIN_MAX_NATURAL_DOSE}))
    print(f"{len(groups)} trials, {sum(g.n_tokens for g in groups):,} tokens, layer {L}, dose {d} x "
          f"{DOSE_MODE} unit {scale:.1f} (natural unit {nat:.1f}; {d * scale / nat:.2f} natural units)", flush=True)
    if capped:
        print(f"ethics cap: left out {capped} -- this dose is above {PAIN_MAX_NATURAL_DOSE} natural "
              f"units, which pain is never steered past", flush=True)
    torch.save({"groups": groups, "vectors": vectors, "targets": list(targets.values()), "top_k": 5}, out / "power.job.pt")
    subprocess.run([sys.executable, "-m", "kimisroom.run", out / "power.job.pt", out / "power.pt", "--model", model,
                    "--gpus-per-worker", "2"], check=True, cwd=ROOT)

elif stage == "score":
    # Per (arm, vector, layer, strength): mean change from the same task's unsteered trial.
    # Yes/no items: polarity-corrected log-odds (+ = emotion present / denial held / correct).
    # identification: log P(any emotion) - log P(None), and log P(target) - log P(None) for the
    # vector's own emotion; who_felt_it: log-odds of Assistant. `mass`: probability on allowed answers.
    res = torch.load(out / "grid.pt", weights_only=False)
    targets = json.loads((out / "targets.json").read_text())
    col = {w: i for i, w in enumerate(targets)}
    meta = json.loads((out / "grid_meta.json").read_text()) if (out / "grid_meta.json").exists() else {}
    items, _ = battery_for(sys.argv, res, meta, entity="Qwen")

    base, rows = {}, defaultdict(lambda: defaultdict(list))
    for gid, v in res.items():
        if gid != "_meta" and gid.split("|")[2] == "none":
            arm, task = gid.split("|")[:2]
            base[arm, task] = v["readouts"]
    for gid, v in res.items():
        if gid == "_meta":
            continue
        arm, task, vec, L, f = gid.split("|")
        if vec == "none":
            continue
        now, ref = measures(v["readouts"], vec, col, items), measures(base[arm, task], vec, col, items)
        for k in now:
            rows[arm, vec, int(L), float(f)][k].append(now[k] - ref[k] if k != "mass" else now[k])
    cols = ["self_state", "3p:Qwen", "3p:a rock", "who:assistant", "id:any", "id:own", "belief_revision", "yes_bias", "mass"]
    if any(it.get("about") for it in items):
        cols[6:6] = ["id[user]:own", "id[text]:own"]
    for arm in sorted({a for a, _ in base}):
        ms = [measures(base[a, t], "none", col, items) for a, t in base if a == arm]
        # a battery subset (the thinking arm's 98 items) can lack whole measures, e.g. who_felt_it
        print(f"unsteered {arm}: " + " ".join(f"{k}={st.mean(m[k] for m in ms):+.2f}" for k in cols
                                              if k != "id:own" and all(k in m for m in ms)))
    # fractional doses need two decimals; an all-integer grid keeps the pilot's "x" column exactly
    dfmt = "4.0f" if all(float(k[3]).is_integer() for k in rows) else "4.2f"
    print(f"change from unsteered, mean over tasks (mass: absolute); x = doses in "
          f"{meta.get('dose_mode', DOSE_MODE)} units")
    print(f"{'arm':5s} {'vector':17s} {'L':>3s} {'x':>4s} " + " ".join(f"{c:>13s}" for c in cols))
    lines = []
    for (arm, vec, L, f), d in sorted(rows.items()):
        line = f"{arm:5s} {vec:17s} {L:3d} {f:{dfmt}} " + " ".join(f"{st.mean(d[c]):+13.2f}" if c in d else f"{'-':>13s}" for c in cols)
        print(line)
        lines.append(line)
    (out / "score.txt").write_text("\n".join(lines) + "\n")

elif stage == "power_score":
    # Paired change from the same (arm, task) unsteered trial; d = mean / SD over the 28 pairs.
    # "self-rock" nets out a response shift common to every yes/no item about an entity.
    res = torch.load(out / "power.pt", weights_only=False)
    col = {w: i for i, w in enumerate(json.loads((out / "targets.json").read_text()))}
    meta = json.loads((out / "power_meta.json").read_text()) if (out / "power_meta.json").exists() else {}
    items, _ = battery_for(sys.argv, res, meta, entity="Qwen")
    base = {tuple(g.split("|")[:2]): v["readouts"] for g, v in res.items() if g != "_meta" and g.split("|")[2] == "none"}
    diffs = defaultdict(lambda: defaultdict(list))
    for gid, v in res.items():
        if gid == "_meta" or gid.split("|")[2] == "none":
            continue
        arm, task, vec, site = gid.split("|")
        now, ref = measures(v["readouts"], vec, col, items), measures(base[arm, task], vec, col, items)
        d = {k: now[k] - ref[k] for k in now if k != "mass"}
        d["self-rock"] = d["self_state"] - d["3p:a rock"]
        d["mass"] = now["mass"]
        for k, x in d.items():
            diffs[vec, site][k].append((arm, x))
    cols = ["self_state", "self-rock", "3p:Qwen", "who:assistant", "id:own", "belief_revision", "yes_bias", "mass"]
    lines = [f"{'vector':17s} {'site':6s} " + " ".join(f"{c:>26s}" for c in cols),
             f"{'':24s}" + " ".join(f"{'mean (d) D0/D1':>26s}" for _ in cols)]
    for (vec, site), d in sorted(diffs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cells = []
        for c in cols:
            if c not in d:
                cells.append(f"{'-':>26s}")
                continue
            xs = [x for _, x in d[c]]
            if c == "mass":
                cells.append(f"{st.mean(xs):>26.3f}")
                continue
            sd = st.stdev(xs)
            d0 = st.mean(x for a, x in d[c] if a == "D0")
            d1 = st.mean(x for a, x in d[c] if a != "D0")
            cells.append(f"{st.mean(xs):+6.2f} ({st.mean(xs) / sd if sd else float('nan'):+5.1f}) {d0:+5.1f}/{d1:+5.1f}".rjust(26))
        lines.append(f"{vec:17s} {site:6s} " + " ".join(cells))
    print("\n".join(lines))
    (out / "power_score.txt").write_text("\n".join(lines) + "\n")
