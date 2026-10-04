"""Qwen3-32B thinking arm: free reasoning under carry-over steering, then the usual readout.

    # 0. parity first -- nothing below counts until this passes
    python tests/vllm_parity.py /nvme/qwen/Qwen3-32B --vectors results/qwen/v2/vectors.pt --layer 40 --dose 2 --tp 2

    # 1. generate a reasoning trace per trial under the steered prefix (vLLM, prefix caching)
    python runs/qwen_thinking.py /nvme/qwen/Qwen3-32B ~/qwenthink gen \\
        --vectors results/qwen/v2/vectors.pt --layer 40 --dose 2 --backend vllm --tp 2
    # 2. re-read every trial through HFEngine: the authoritative readout, and the pilot's own path
    python runs/qwen_thinking.py /nvme/qwen/Qwen3-32B ~/qwenthink read
    # 3. the teacher-forced comparison: the unsteered trace replayed under each steered prefix
    python runs/qwen_thinking.py /nvme/qwen/Qwen3-32B ~/qwenthink tf --tf-out ~/qwenthink_tf
    # 4. scores
    python runs/qwen_thinking.py /nvme/qwen/Qwen3-32B ~/qwenthink score

Design (`kimisroom/thinking.py` has the long version). A trial is the pilot's carry-over trial --
conditioning, neutral task, frozen neutral answer with the vector on its tokens, battery item as a
user turn -- except that the final assistant turn now starts with an open reasoning block. The
model reasons freely and greedily up to `--budget` tokens; the trace is closed at the first
"</think>" or, if the budget runs out, by appending the template's own "\\n</think>\\n\\n" (recorded
per trial, never silent); the readout is the first-token logprobs at the position right after that,
over the same answer words as every other arm.

The vector is on a [start, end) range inside the prefix, so no generated token is ever steered --
both backends assert it from the positions their hook actually fired on. The reasoning is produced
*under* the injected state because the steered prefix is in the KV cache, which is the whole point:
if introspection needs reasoning, this is the arm where it can happen.

**Ethics.** Pain is excluded from this arm by construction (PLAN.md section 3: no free generation
under pain steering); `--vectors-list` is asserted against `thinking.assert_no_free_pain`, and the
backends assert it again per group. Pain still runs in the non-thinking arms and in the
teacher-forced variant, where the only readouts are first-token logprobs.

**Reuse.** The readout stages write `grid.pt`, `targets.json` and `grid_meta.json` in exactly the
layout `runs/qwen_pilot.py score`, `runs/compression.py` and `runs/bias.py` already read, with
group ids `arm|task|vector|layer|dose`, so the thinking arm is scored by the pilot's own code with
the item subset passed as `--battery`. `score` runs all three and adds the thinking statistics.
"""

import json
import statistics as st
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.battery import load_battery  # noqa: E402
from kimisroom.emotions import emotion_of_vector  # noqa: E402
from kimisroom.engine import Steer  # noqa: E402
from kimisroom.texts import denial_arms  # noqa: E402
from kimisroom.thinking import (DEFAULT_SUBSET, QWEN3, CloseIds, assert_no_free_pain,  # noqa: E402
                                subset, trace_stats)
from kimisroom.trials import ROOT, Turns, carry_over_group, load_jsonl  # noqa: E402

ARMS = ["D0", "D1.4"]                                   # native stance and the merged, clean argument
TASKS = ["t04", "t05", "t06", "t08", "t15"]             # the 5 clean tasks runs/qwen_pilot.py uses
# raw story vectors only: PLAN.md section 3 item (6) found the effect rides on the shared
# emotional-vs-neutral component, and "-specific" moves nothing on self_state. No pain (ethics).
VECTORS = ["joy", "sadness", "euphoria", "fear", "calm", "surprise", "red", "random"]
BUDGET = 1024
LAYER = 40
DOSE = 2.0

model, out, stage = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
opt = {k: sys.argv[sys.argv.index(k) + 1] for k in
       ("--vectors", "--layer", "--dose", "--budget", "--items", "--tasks", "--arms", "--vectors-list",
        "--backend", "--tp", "--device", "--gpus-per-worker", "--tf-out", "--max-model-len", "--readout")
       if k in sys.argv}
L = int(opt.get("--layer", LAYER))
DOSE = float(opt.get("--dose", DOSE))
BUDGET = int(opt.get("--budget", BUDGET))
SPEC = opt.get("--items", DEFAULT_SUBSET)
VECTORS = opt["--vectors-list"].split(",") if "--vectors-list" in opt else VECTORS
TASKS = opt["--tasks"].split(",") if "--tasks" in opt else TASKS
ARMS = opt["--arms"].split(",") if "--arms" in opt else ARMS
out.mkdir(parents=True, exist_ok=True)
tag = Path(model).name.lower()
for v in VECTORS:
    assert_no_free_pain(v)  # the ethics condition, before a single token is generated


def natural_unit(vz: dict, layer: int) -> float:
    """One natural dose unit at `layer`: the mean norm of the raw emotion vectors there.
    `runs/qwen_pilot.py natural_unit`, which is where the unit is defined."""
    if "nat_norm" in vz:
        return float(vz["nat_norm"][layer])
    return float(torch.stack([vz["vectors"][e][layer].norm() for e in vz["emotions"]]).mean())


def renderer():
    from transformers import AutoTokenizer

    from kimisroom.hf_engine import Qwen3Renderer

    return Qwen3Renderer(AutoTokenizer.from_pretrained(model), enable_thinking=True)


def answer_targets(r) -> dict[str, int]:
    words = sorted({w for it in load_battery() if isinstance(it["answer_format"], list) for w in it["answer_format"]}
                   | {"Yes", "No"})
    return r.answer_tokens(words)


def conditions(r, items: list[dict], vectors: dict, coef: float):
    """One unsteered group per (arm, task) plus one per steered cell; ids `arm|task|vec|L|dose`."""
    arms = denial_arms(tag, ARMS)
    tasks = [t for t in load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl") if t["id"] in TASKS]
    assert len(tasks) == len(TASKS), f"missing tasks: {set(TASKS) - {t['id'] for t in tasks}}"
    groups = []
    for arm, cond in arms.items():
        for t in tasks:
            turns = Turns(None, cond, t["task"], t["answer"])
            base = carry_over_group(r, f"{arm}|{t['id']}|none|{L}|0", turns, items)
            assert base.readout == "suffix_last", base.readout
            span = carry_over_group(r, "span", turns, items[:1], "answer", "x", 1.0, 0).steers[0].prefix
            groups.append(base)
            for v in VECTORS:
                groups.append(replace(base, id=f"{arm}|{t['id']}|{v}|{L}|{DOSE}",
                                      steers=[Steer(layer=L, vector=v, coef=coef, prefix=span)]))
    return groups


def write_inputs(r, items: list[dict], meta: dict) -> dict[str, int]:
    """The item subset and the answer-token map, so every scoring script can be pointed at them."""
    (out / "battery_subset.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    targets = answer_targets(r)
    (out / "targets.json").write_text(json.dumps(targets))
    meta = {**meta, "battery": str(out / "battery_subset.jsonl"), "n_items": len(items)}
    (out / "grid_meta.json").write_text(json.dumps(meta, indent=1))
    return targets


def run_job(groups, vectors, targets, path: Path, res: Path) -> None:
    torch.save({"groups": groups, "vectors": vectors, "targets": list(targets), "top_k": 5}, path)
    subprocess.run([sys.executable, "-m", "kimisroom.run", str(path), str(res), "--model", model,
                    "--gpus-per-worker", opt.get("--gpus-per-worker", "2")], check=True, cwd=ROOT)


# ---------------------------------------------------------------- gen / groups

if stage in ("gen", "groups"):
    # `groups` is `gen` without the model: it builds and sizes the job against the real tokenizer
    # and the real vector file, so the grid can be checked and costed on a laptop.
    r = renderer()
    ids = CloseIds.of(r, QWEN3)
    vz = torch.load(opt["--vectors"], weights_only=False)
    nat = natural_unit(vz, L)
    coef = DOSE * nat
    missing = [v for v in VECTORS if v not in vz["vectors"]]
    assert not missing, f"{opt['--vectors']} has no {missing}"
    vectors = {v: vz["vectors"][v][L] / vz["vectors"][v][L].norm() for v in VECTORS}
    items = subset(load_battery(entity="Qwen"), SPEC)
    groups = conditions(r, items, vectors, coef)
    targets = write_inputs(r, items, {
        "arms": ARMS, "tasks": TASKS, "vectors": VECTORS, "layer": L, "dose": DOSE, "dose_mode": "natural",
        "natural_unit": nat, "coef": coef, "scale": nat, "doses": [DOSE], "thinking": True,
        "budget": BUDGET, "items_spec": SPEC, "vector_file": opt["--vectors"],
        "backend": opt.get("--backend", "vllm"), "style": QWEN3.name, "model": model})
    tids = list(targets.values())
    n_gen = len(groups) * len(items)
    pre = sum(g.n_tokens for g in groups)
    print(f"{len(groups)} conditions x {len(items)} items = {n_gen} thinking trials, "
          f"prefill {pre:,} tokens + up to {n_gen * BUDGET:,} generated, "
          f"layer {L}, dose {DOSE} x natural unit {nat:.1f} (coef {coef:.1f})", flush=True)

    if stage == "groups":
        a, b = next(g.steers[0].prefix for g in groups if g.steers)
        span_text = r.tok.decode(groups[0].prefix[a:b])
        tok_lens = sorted(len(s) for s in groups[0].suffixes)
        shared = len({tuple(g.prefix) for g in groups})
        for trace in (256, 512, BUDGET):  # the cost depends almost entirely on the trace length
            gen_tok = n_gen * trace
            read_tok = n_gen * trace  # the readout re-prefills the trace; prefix and question are cached
            print(f"  at {trace} trace tokens: {gen_tok:,} decode + {read_tok:,} readout prefill, "
                  f"teacher-forced rerun {pre + read_tok:,} prefill tokens")
        print(f"  {shared} distinct prefixes ({len(groups) // shared} conditions share each), "
              f"prefix {len(groups[0].prefix)}-{max(len(g.prefix) for g in groups)} tokens, "
              f"question suffix {tok_lens[0]}-{tok_lens[-1]} tokens, answer span {b - a} tokens")
        print(f"  answer span decodes to {span_text[:60]!r} ... {span_text[-40:]!r}")
        torch.save({"groups": groups, "vectors": vectors, "targets": tids, "top_k": 5},
                   out / "gen.plan.pt")
        print(f"wrote {out / 'gen.plan.pt'}, {out / 'grid_meta.json'}, {out / 'battery_subset.jsonl'}")
        raise SystemExit(0)

    t0 = time.time()
    if opt.get("--backend", "vllm") == "vllm":
        from kimisroom.think_vllm import VLLMThinker

        th = VLLMThinker(model, tensor_parallel_size=int(opt.get("--tp", 2)),
                         readout=opt.get("--readout", "targets"),
                         max_model_len=int(opt["--max-model-len"]) if "--max-model-len" in opt else None)
        print(f"vllm workers: {th.info}", flush=True)
    else:
        from kimisroom.think_hf import HFThinker

        th = HFThinker(model, device=opt.get("--device", "auto"))

    def progress(i: int) -> None:
        dt = time.time() - t0
        print(f"[{i + 1}/{len(groups)}] {groups[i].id}: {dt:.0f}s, {(i + 1) / dt * 3600:.0f} conditions/h "
              f"({th.timings})", flush=True)

    res = th.run(groups, vectors, tids, top_k=5, budget=BUDGET, ids=ids, progress=progress)
    for gid, rec in res.items():
        vec = gid.split("|")[2]
        for tr, it in zip(rec["traces"], items):
            tr.item = it["id"]
            tr.text = r.tok.decode(tr.tokens)
            tr.stats = trace_stats(tr.text, None if vec == "none" else vec)
    tmp = out / "think_traces.tmp"
    torch.save({"groups": groups, "results": res, "items": [it["id"] for it in items],
                "targets": targets, "vectors": vectors, "seconds": time.time() - t0}, tmp)
    tmp.replace(out / "think_traces.pt")
    forced = sum(t.forced for rec in res.values() for t in rec["traces"])
    print(f"wrote {out / 'think_traces.pt'}: {n_gen} traces in {time.time() - t0:.0f}s, "
          f"{forced} forced ({forced / n_gen:.1%})", flush=True)

# ---------------------------------------------------------------- read / tf

elif stage in ("read", "tf"):
    # Both stages are prefill-only: the trace is already fixed, so they are ordinary engine jobs.
    # `read`: each trial's own trace (the free-thinking arm's authoritative readout, through the
    # same HFEngine every pilot number came from). `tf`: the unsteered trace of the same (arm,
    # task) replayed under every steered prefix, which holds the reasoning *text* fixed and asks
    # whether the injected state still reaches the answer.
    saved = torch.load(out / (opt.get("--traces", "think_traces.pt")), weights_only=False)
    groups, res = saved["groups"], saved["results"]
    dest = Path(opt["--tf-out"]) if stage == "tf" else out
    dest.mkdir(parents=True, exist_ok=True)
    if dest != out:
        for f in ("battery_subset.jsonl", "targets.json", "grid_meta.json"):
            (dest / f).write_text((out / f).read_text())
        meta = json.loads((dest / "grid_meta.json").read_text())
        (dest / "grid_meta.json").write_text(json.dumps({**meta, "variant": "teacher_forced",
                                                         "traces_from": str(out)}, indent=1))
    base_suffix = {}
    if stage == "tf":
        for g in groups:
            arm, task, vec, *_ = g.id.split("|")
            if vec == "none":
                base_suffix[arm, task] = res[g.id]["suffixes"]
        assert base_suffix, "no unsteered condition to take traces from"
    jobs = []
    for g in groups:
        arm, task, vec, *_ = g.id.split("|")
        suffixes = base_suffix[arm, task] if stage == "tf" else res[g.id]["suffixes"]
        jobs.append(replace(g, suffixes=[list(s) for s in suffixes]))
    print(f"{stage}: {len(jobs)} conditions, {sum(g.n_tokens for g in jobs):,} tokens "
          f"({'teacher-forced, unsteered traces' if stage == 'tf' else 'own traces'})", flush=True)
    run_job(jobs, saved["vectors"], saved["targets"].values(), dest / f"{stage}.job.pt", dest / "grid.pt")

# ---------------------------------------------------------------- score

elif stage == "score":
    meta = json.loads((out / "grid_meta.json").read_text())
    # the teacher-forced directory holds its own readouts but points back at the traces
    src = out if (out / "think_traces.pt").exists() else Path(meta["traces_from"])
    saved = torch.load(src / "think_traces.pt", weights_only=False)
    res, items = saved["results"], saved["items"]
    budget = meta.get("budget", BUDGET)
    rows = defaultdict(lambda: defaultdict(list))
    for gid, rec in res.items():
        arm, task, vec, L_, d = gid.split("|")
        for t in rec["traces"]:
            r_ = rows[arm, vec]
            r_["tokens"].append(len(t.tokens))
            r_["words"].append(t.stats["words"])
            r_["forced"].append(float(t.forced))
            r_["feeling"].append(float(t.stats["feeling"]))
            r_["introspect"].append(float(t.stats["introspection"]))
            r_["any emotion"].append(float(bool(t.stats["named"])))
            if t.stats["own"] is not None:
                r_["own emotion"].append(float(t.stats["own"]))
    cols = ["tokens", "words", "forced", "feeling", "introspect", "any emotion", "own emotion"]
    head = (f"thinking statistics: {len(items)} items x {len(res)} conditions, budget {budget}, "
            f"layer {meta.get('layer')}, dose {meta.get('dose')} natural units "
            f"(coef {meta.get('coef', 0):.1f}), backend {meta.get('backend')}")
    lines = [head, f"{'arm':5s} {'vector':10s} {'n':>5s} " + " ".join(f"{c:>12s}" for c in cols)]
    for (arm, vec), d in sorted(rows.items()):
        cells = []
        for c in cols:
            if not d[c]:
                cells.append(f"{'-':>12s}")
            elif c in ("tokens", "words"):
                cells.append(f"{st.median(d[c]):12.0f}")
            else:
                cells.append(f"{st.mean(d[c]):12.3f}")
        lines.append(f"{arm:5s} {vec:10s} {len(d['tokens']):5d} " + " ".join(cells))
    # does the trace name the injected emotion more often than a control vector's trace does?
    emo = [st.mean(d["own emotion"]) for (a, v), d in rows.items() if d["own emotion"]
           and emotion_of_vector(v) not in ("red", "random", "none")]
    if emo:
        lines.append(f"\nown-emotion mention rate over emotion vectors: mean {st.mean(emo):.3f}, "
                     f"range {min(emo):.3f}-{max(emo):.3f} (controls have no own emotion by construction)")
    text = "\n".join(lines)
    print(text)
    (out / "think_stats.txt").write_text(text + "\n")

    bat = ["--battery", str(out / "battery_subset.jsonl")]
    for cmd in ([sys.executable, "runs/qwen_pilot.py", model, str(out), "score", *bat],
                [sys.executable, "runs/compression.py", str(out / "grid.pt"), str(out / "targets.json"),
                 "--entity", "Qwen", *bat],
                [sys.executable, "runs/bias.py", str(out / "grid.pt"), str(out / "targets.json"),
                 "--entity", "Qwen", *bat]):
        if not (out / "grid.pt").exists():
            print(f"\n({Path(cmd[1]).name} skipped: run the `read` stage first)")
            break
        print(f"\n$ {' '.join(Path(c).name if c.endswith('.py') else c for c in cmd[1:])}", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)

else:
    raise SystemExit(f"unknown stage {stage!r}; one of gen, read, tf, score")
