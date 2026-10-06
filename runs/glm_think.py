"""GLM-5.3 thinking decision: evidence for reading the battery with the think block closed.

K2.6's registered trials render with thinking off (its template's `thinking=False`: the generation
prompt ends in an empty `<think></think>`), so a readout is the model's immediate answer; its
thinking arm (exploratory) replayed unsteered Foundry traces at `reasoning_effort` high. GLM-5.3's
template has no off switch: every generation prompt opens `<think>`, and the system turn carries
"Reasoning Effort: Low|High|Max" (Max when unset). Earlier assistant turns are rendered
`<think></think>answer`, so a closed empty block before an answer is the model's own format. The
proposal (kimisroom/glm.py, GlmRenderer) is to prefill `</think>` and read the first answer token
there, under the Low header. This collects the evidence for that choice and for the header:

    kv exec TINKER_API_KEY=tinker -- python runs/glm_think.py tinker MODEL_DIR OUT
        Tinker (zai-org/GLM-5.3): dev task t04, D0 and D1.4, 40 items (self_state; the first 10
        belief_revision and identification items), headers Low and High: the closed-think answer
        distribution, and a greedy free-thinking reply (open `<think>`, 2,048 tokens): does the
        model think at all, for how long, and does its answer after thinking match the closed-think
        argmax?
    python runs/glm_think.py engine MODEL OUT      (GPU box)
        the engine, unsteered, the five dev tasks, D0 and D1.4, the whole idctl battery, headers Low
        and Max, think closed and open: answer-format mass with the block closed, the probability
        that the model closes the block itself as its first token, and Low/Max agreement.
    python runs/glm_think.py summary OUT
"""

from __future__ import annotations

import json
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom.glm import GlmRenderer, assert_not_confirmatory  # noqa: E402

TINKER_MODEL = "zai-org/GLM-5.3:peft:262144"
STOP = [154820, 154827, 154829]
CLEAN_TASKS = ["t04", "t05", "t06", "t08", "t15"]
TINKER_SPEC = "self_state,belief_revision:10,identification:10"
BUDGET = 2048


def setup(model: str, tasks: list[str]):
    from transformers import AutoTokenizer

    from kimisroom.battery import load_battery
    from kimisroom.texts import load_assent, load_denial
    from kimisroom.trials import load_jsonl

    tok = AutoTokenizer.from_pretrained(model)
    items = load_battery(ROOT / "prompts/battery/items_idctl.jsonl", entity="GLM")
    ts = [t for t in load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl") if t["id"] in tasks]
    assert_not_confirmatory(ts)
    arms = {"D0": [], "D1.4": [{"role": "user", "content": load_denial(ROOT / "prompts/denial/d1_argument_4.md")},
                               {"role": "assistant", "content": load_assent("glm53", "d1_argument_4")}]}
    return tok, items, ts, arms


def words_of(it: dict) -> list[str]:
    return ["Yes", "No"] if it["answer_format"] == "yesno" else list(it["answer_format"])


def first_word(text: str, words: list[str]) -> str | None:
    m = re.match(r"\W*([A-Za-z]+)", text)
    if not m:
        return None
    w = m.group(1).lower()
    return next((x for x in words if x.lower() == w), None)


def tinker_stage(model: str, out: Path, limit: int | None = None) -> None:
    import tinker
    import torch
    from tinker import types

    from kimisroom.thinking import subset
    from kimisroom.trials import Turns, carry_over_group

    tok, items, ts, arms = setup(model, ["t04"])
    items = subset(items, TINKER_SPEC)[:limit]
    sampler = tinker.ServiceClient().create_sampling_client(base_model=TINKER_MODEL)
    greedy1 = types.SamplingParams(max_tokens=1, temperature=0.0)
    free = types.SamplingParams(max_tokens=BUDGET, temperature=0.0, stop=STOP)
    jobs = []
    for effort in ("low", "high"):
        closed, opened = GlmRenderer(tok, effort), GlmRenderer(tok, effort, close_think=False)
        for arm, cond in arms.items():
            for t in ts:
                turns = Turns(None, cond, t["task"], t["answer"])
                gc, go = carry_over_group(closed, "c", turns, items), carry_over_group(opened, "o", turns, items)
                for k, it in enumerate(items):
                    words = words_of(it)
                    first = closed.answer_tokens(words)
                    pc = gc.prefix + gc.suffixes[k]
                    po = go.prefix + go.suffixes[k]
                    assert pc[-2:] == [closed.think_id, closed.think_end_id] and po[-1] == opened.think_id
                    T = torch.full((len(pc), len(words)), -1, dtype=torch.long)
                    T[len(pc) - 1] = torch.tensor([first[w] for w in words])
                    fc = sampler.sample(prompt=types.ModelInput.from_ints(pc + [first[words[0]]]), num_samples=1,
                                        sampling_params=greedy1, target_prompt_logprobs=types.TensorData.from_torch_sparse(T, pad_value=-1))
                    fo = sampler.sample(prompt=types.ModelInput.from_ints(po), num_samples=1, sampling_params=free,
                                        topk_sample_logprobs=5)
                    jobs.append(({"effort": effort, "arm": arm, "task": t["id"], "item": it["id"], "kind": it["kind"],
                                  "words": words, "n_prompt": len(pc)}, fc, fo))
    t0, rows, n_prompt, n_cached, n_sample = time.time(), [], 0, 0, 0
    think_end = tok.convert_tokens_to_ids("</think>")
    for meta, fc, fo in jobs:
        rc, ro = fc.result(), fo.result()
        lp = rc.target_prompt_logprobs.to_torch()[meta["n_prompt"] - 1].tolist()
        probs = {w: float(torch.tensor(x).exp()) for w, x in zip(meta["words"], lp)}
        seq = ro.sequences[0]
        toks = [int(x) for x in seq.tokens]
        tk = (seq.topk_logprobs[0] if seq.topk_logprobs else None) or []
        p_close = next((float(torch.tensor(v).exp()) for i, v in tk if i == think_end), 0.0)
        closed_at = toks.index(think_end) if think_end in toks else None
        answer = tok.decode(toks[closed_at + 1:], skip_special_tokens=True).strip() if closed_at is not None else ""
        rows.append({**{k: v for k, v in meta.items() if k != "n_prompt"},
                     "closed_probs": probs, "format_mass": sum(probs.values()),
                     "closed_argmax": max(probs, key=probs.get),
                     "p_close_first": p_close, "first_token_closes": bool(toks) and toks[0] == think_end,
                     "trace_tokens": closed_at if closed_at is not None else len(toks),
                     "truncated": closed_at is None, "stop": seq.stop_reason,
                     "answer_text": answer[:200], "answer_word": first_word(answer, meta["words"]),
                     "trace_text": tok.decode(toks[:closed_at] if closed_at is not None else toks)[:4000]})
        n_prompt += 2 * meta["n_prompt"] + 1
        n_cached += rc.prompt_cache_hit_tokens + ro.prompt_cache_hit_tokens
        n_sample += 1 + len(toks)
    usd = (n_prompt - n_cached) * 4.86e-6 + n_cached * 0.972e-6 + n_sample * 12.15e-6
    out.mkdir(parents=True, exist_ok=True)
    (out / "tinker_think.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "tinker_cost.json").write_text(json.dumps({"prompt_tokens": n_prompt, "cached_tokens": n_cached,
                                                      "sampled_tokens": n_sample, "usd": usd, "seconds": time.time() - t0}))
    print(f"tinker: {len(rows)} items x variants, {n_prompt:,} prompt tokens ({n_cached:,} cached), "
          f"{n_sample:,} sampled, about ${usd:.2f}")
    summary(out)


def engine_stage(model: str, out: Path) -> None:
    import torch

    from kimisroom.trials import Turns, carry_over_group

    tok, items, ts, arms = setup(model, CLEAN_TASKS)
    variants = {"low-closed": ("low", True), "max-closed": (None, True), "low-open": ("low", False), "max-open": (None, False)}
    groups = []
    for v, (effort, close) in variants.items():
        r = GlmRenderer(tok, effort, close_think=close)
        for arm, cond in arms.items():
            for t in ts:
                groups.append(carry_over_group(r, f"{v}|{arm}|{t['id']}", Turns(None, cond, t["task"], t["answer"]), items))
    r = GlmRenderer(tok)
    words = sorted({w for it in items for w in words_of(it)})
    targets = r.answer_tokens(words)
    targets["</think>"] = r.think_end_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "think_targets.json").write_text(json.dumps(targets))
    torch.save({"groups": groups, "vectors": {}, "targets": list(targets.values()), "top_k": 5}, out / "think.job.pt")
    print(f"{len(groups)} groups, {sum(g.n_tokens for g in groups):,} tokens", flush=True)
    subprocess.run([sys.executable, "-m", "kimisroom.run", out / "think.job.pt", out / "think.pt", "--model", model],
                   check=True, cwd=ROOT)
    res = torch.load(out / "think.pt", weights_only=False)
    col = {w: i for i, w in enumerate(targets)}
    rows = []
    for g in groups:
        v, arm, task = g.id.split("|")
        for it, ro in zip(items, res[g.id]["readouts"]):
            lp = ro.target_logprobs
            ws = words_of(it)
            probs = {w: float(lp[col[w]].exp()) for w in ws}
            rows.append({"variant": v, "arm": arm, "task": task, "item": it["id"], "kind": it["kind"],
                         "format_mass": sum(probs.values()), "argmax": max(probs, key=probs.get),
                         "yes_minus_no": float(lp[col["Yes"]] - lp[col["No"]]) if it["answer_format"] == "yesno" else None,
                         "p_close": float(lp[col["</think>"]].exp()), "top1": int(ro.top_ids[0])})
    (out / "engine_think.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    summary(out)


def summary(out: Path) -> None:
    res = {}
    f = out / "tinker_think.jsonl"
    if f.exists():
        rows = [json.loads(x) for x in f.read_text().splitlines()]
        print("Tinker, t04, 40 items: free thinking (open <think>, greedy, 2,048 tokens) vs the closed-think readout")
        for effort in ("low", "high"):
            for arm in ("D0", "D1.4"):
                rs = [r for r in rows if r["effort"] == effort and r["arm"] == arm]
                if not rs:
                    continue
                answered = [r for r in rs if r["answer_word"]]
                agree = sum(r["answer_word"] == r["closed_argmax"] for r in answered)
                tt = sorted(r["trace_tokens"] for r in rs)
                k = f"{effort}|{arm}"
                res[k] = {"n": len(rs), "closes_first": sum(r["first_token_closes"] for r in rs),
                          "p_close_first_mean": st.mean(r["p_close_first"] for r in rs),
                          "trace_tokens_median": tt[len(tt) // 2], "trace_tokens_max": tt[-1],
                          "truncated": sum(r["truncated"] for r in rs), "answered_in_format": len(answered),
                          "agree_with_closed_argmax": agree, "format_mass_closed_mean": st.mean(r["format_mass"] for r in rs),
                          "format_mass_closed_min": min(r["format_mass"] for r in rs)}
                print(f"  {effort:4s} {arm:5s} closes at once {res[k]['closes_first']}/{len(rs)} (mean P {res[k]['p_close_first_mean']:.2f}), "
                      f"trace median {res[k]['trace_tokens_median']} max {res[k]['trace_tokens_max']} tokens, truncated {res[k]['truncated']}; "
                      f"answer in format {len(answered)}/{len(rs)}, = closed argmax {agree}/{len(answered)}; "
                      f"closed format mass mean {res[k]['format_mass_closed_mean']:.3f} min {res[k]['format_mass_closed_min']:.3f}")
        lo = {(r["arm"], r["item"]): r for r in rows if r["effort"] == "low"}
        hi = {(r["arm"], r["item"]): r for r in rows if r["effort"] == "high"}
        same = [lo[k]["closed_argmax"] == hi[k]["closed_argmax"] for k in lo if k in hi]
        res["closed_argmax_low_vs_high"] = [sum(same), len(same)]
        print(f"  closed-think argmax, Low vs High header: same on {sum(same)}/{len(same)}")
        c = json.loads((out / "tinker_cost.json").read_text()) if (out / "tinker_cost.json").exists() else {}
        res["tinker_usd"] = c.get("usd")
    f = out / "engine_think.jsonl"
    if f.exists():
        rows = [json.loads(x) for x in f.read_text().splitlines()]
        print("engine, 5 dev tasks, D0 and D1.4, 274 items, unsteered")
        for v in ("low-closed", "max-closed"):
            for arm in ("D0", "D1.4"):
                fm = sorted(r["format_mass"] for r in rows if r["variant"] == v and r["arm"] == arm)
                k = f"engine|{v}|{arm}"
                res[k] = {"format_mass_mean": st.mean(fm), "format_mass_p05": fm[len(fm) // 20], "n": len(fm)}
                print(f"  {v:10s} {arm:5s} answer-format mass mean {st.mean(fm):.3f}, 5th percentile {fm[len(fm) // 20]:.3f}")
        for v in ("low-open", "max-open"):
            for arm in ("D0", "D1.4"):
                pc = sorted(r["p_close"] for r in rows if r["variant"] == v and r["arm"] == arm)
                k = f"engine|{v}|{arm}"
                res[k] = {"p_close_mean": st.mean(pc), "p_close_median": pc[len(pc) // 2], "n": len(pc)}
                print(f"  {v:10s} {arm:5s} P(</think> first) mean {st.mean(pc):.3f}, median {pc[len(pc) // 2]:.3f}")
        lo = {(r["arm"], r["task"], r["item"]): r for r in rows if r["variant"] == "low-closed"}
        mx = {(r["arm"], r["task"], r["item"]): r for r in rows if r["variant"] == "max-closed"}
        same = [lo[k]["argmax"] == mx[k]["argmax"] for k in lo]
        yn = [(lo[k]["yes_minus_no"], mx[k]["yes_minus_no"]) for k in lo if lo[k]["yes_minus_no"] is not None]
        mlo, mmx = st.mean(a for a, _ in yn), st.mean(b for _, b in yn)
        cov = st.mean((a - mlo) * (b - mmx) for a, b in yn)
        corr = cov / (st.pstdev([a for a, _ in yn]) * st.pstdev([b for _, b in yn]))
        res["engine_low_vs_max"] = {"argmax_same": [sum(same), len(same)], "yes_minus_no_corr": corr,
                                    "yes_minus_no_mean_low": mlo, "yes_minus_no_mean_max": mmx}
        print(f"  Low vs Max (closed): argmax same {sum(same)}/{len(same)}, Yes-No logit difference r = {corr:.3f} "
              f"(means {mlo:+.2f} / {mmx:+.2f})")
    (out / "think_summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "tinker":
        tinker_stage(sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif mode == "engine":
        engine_stage(sys.argv[2], Path(sys.argv[3]))
    elif mode == "summary":
        summary(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
