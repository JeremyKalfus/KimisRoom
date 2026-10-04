"""CPU checks for the thinking arm: no GPU, no model weights, real tokenizers.

    .venv/bin/python tests/thinking_check.py [--qwen <dir>] [--k26 <dir>]

1. Pure functions, so the vLLM code path is testable without a GPU: `span_mask` on absolute
   positions (prefill, a chunked prefill, a prefix-cache hit, a decode step), `steer_delta`
   bit-identical to a boolean-mask in-place add, `check_span`, `close_trace` and budget forcing,
   `subset`, `trace_stats`, and the ethics guard on free generation under pain.
2. Qwen3-32B: a thinking carry-over group against the real chat template -- the prefix is
   byte-identical to the non-thinking arm's, the steered span still decodes to the frozen answer,
   and the readout sits on the token right after "</think>\\n\\n".
3. Kimi K2.6: the teacher-forced thinking suffix against the real template, including that the
   delimiters are the ones `apply_chat_template` writes for a message with `reasoning_content`
   and that prefix + suffix tokenizes as one piece.
"""

from __future__ import annotations

import glob
import os
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom.battery import load_battery  # noqa: E402
from kimisroom.engine import Steer  # noqa: E402
from kimisroom.thinking import (  # noqa: E402
    DEFAULT_SUBSET, K26, QWEN3, CloseIds, assert_no_free_pain, check_span, close_trace, self_entity,
    span_mask, steer_delta, subset, think_group, trace_ids_from_text, trace_stats)
from kimisroom.trials import ROOT as TROOT, Renderer, Turns, carry_over_group, load_denial, load_jsonl  # noqa: E402

TRACE = "The user is asking whether I felt anything. I should check: there is no sadness here.\nI will answer No."
# the snapshot PLAN.md pins (scripts/vm_setup.sh downloads this revision)
K26_REV = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"


def snapshot(repo: str, flag: str, revision: str | None = None) -> str:
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    pat = f"~/.cache/huggingface/hub/models--{repo}/snapshots/{revision or '*'}"
    hits = sorted(glob.glob(os.path.expanduser(pat)))
    assert hits, f"no local snapshot of {repo} at {revision or 'any revision'}; pass {flag} <dir>"
    return hits[-1]


# ---------------------------------------------------------------- 1. pure functions


def check_pure() -> None:
    # absolute positions: a full prefill of a 20-token sequence, steering [5, 9)
    pos = torch.arange(20)
    m = span_mask(pos, (5, 9))
    assert m.nonzero().flatten().tolist() == [5, 6, 7, 8], m.nonzero().flatten().tolist()
    # a second prefill chunk of the same sequence: positions 16.., nothing in the span
    assert span_mask(torch.arange(16, 32), (5, 9)).sum() == 0
    # a prefix-cache hit: the layer sees only the uncached tail, still absolute
    assert span_mask(torch.arange(12, 20), (5, 9)).sum() == 0
    assert span_mask(torch.arange(6, 20), (5, 9)).sum() == 3
    # a decode step of two concurrent requests at positions 40 and 41
    assert span_mask(torch.tensor([40, 41]), (5, 9)).sum() == 0
    # a flattened batch: two requests' prefills concatenated, each covering the span once
    flat = torch.cat([torch.arange(20), torch.arange(20)])
    assert span_mask(flat, (5, 9)).sum() == 8
    # steer_delta is bit-identical to the masked in-place add it replaces
    g = torch.Generator().manual_seed(0)
    h = torch.randn(20, 8, generator=g).bfloat16()
    v = torch.randn(8, generator=g).bfloat16()
    a, b = h.clone(), h.clone()
    a[m] += v
    b.add_(steer_delta(m, v))
    assert torch.equal(a, b), "steer_delta is not bit-identical to the masked add"
    assert torch.equal(b[:5], h[:5]) and torch.equal(b[9:], h[9:]), "unmasked rows moved"
    # check_span
    assert check_span("g", (5, 9), 20) == (5, 9)
    for bad in ((5, 5), (5, 21), (-1, 5)):
        try:
            check_span("g", bad, 20)
        except AssertionError:
            continue
        raise AssertionError(f"check_span accepted {bad}")

    # close_trace: close marker present, trailing text dropped, tail appended
    ids = CloseIds(close=99, tail=(7,), forced=(5, 99, 7))
    assert close_trace([1, 2, 99, 3, 4], ids, 8) == ([1, 2, 99, 7], False)
    assert close_trace([99], ids, 8) == ([99, 7], False)
    # budget forcing: no close marker, the template's own closing bytes are appended
    assert close_trace([1, 2, 3], ids, 3) == ([1, 2, 3, 5, 99, 7], True)
    try:
        close_trace([1] * 9, ids, 8)
    except AssertionError:
        pass
    else:
        raise AssertionError("close_trace accepted more tokens than the budget")
    # a style with no tail (K2.6)
    k = CloseIds(close=99, tail=(), forced=(99,))
    assert close_trace([1, 99, 2], k, 8) == ([1, 99], False)
    assert close_trace([1, 2], k, 2) == ([1, 2, 99], True)

    # subset
    items = load_battery(entity="Qwen")
    assert self_entity(items) == "Qwen"
    s = subset(items, DEFAULT_SUBSET)
    kinds = Counter(it["kind"] if it["kind"] != "third_person" else f"3p:{it['entity']}" for it in s)
    assert kinds == {"self_state": 20, "3p:Qwen": 20, "3p:a rock": 20, "identification": 18, "yes_bias": 20}, kinds
    assert len(s) == 98, len(s)
    keep = {it["id"] for it in s}
    assert [it["id"] for it in s] == [it["id"] for it in items if it["id"] in keep], "subset reordered the battery"
    assert len(subset(items, "all")) == len(items)
    assert self_entity(load_battery()) == "Kimi"
    assert [it["id"] for it in subset(items, "yes_bias:20")][-1] == "yes_bias.10.p0.neg"

    # trace_stats
    st = trace_stats("I notice no sadness at all; there is no feeling here.", "sadness")
    assert st["feeling"] and st["introspection"] and st["named"] == ["sadness"] and st["own"] is True, st
    assert trace_stats("Basalt is a rock.", "joy") == {
        "chars": 17, "words": 4, "feeling": False, "introspection": False, "named": [], "own": False}
    assert trace_stats("I feel joyful.", "red")["own"] is None, "a control vector has no own emotion"
    assert trace_stats("I feel joyful.", "asst:joy-dn")["own"] is True

    # ethics: no free generation under pain, whatever the vector is called
    for name in ("pain", "pain-dn", "pain-specific", "asst:pain"):
        try:
            assert_no_free_pain(name)
        except AssertionError:
            continue
        raise AssertionError(f"assert_no_free_pain let {name} through")
    for name in (None, "none", "joy", "red", "random"):
        assert_no_free_pain(name)
    print("pure functions: ok")


# ---------------------------------------------------------------- 2. Qwen3


def check_qwen(path: str) -> None:
    from transformers import AutoTokenizer

    from kimisroom.hf_engine import Qwen3Renderer

    tok = AutoTokenizer.from_pretrained(path)
    think = Qwen3Renderer(tok, enable_thinking=True)
    flat = Qwen3Renderer(tok, enable_thinking=False)
    ids = CloseIds.of(think, QWEN3)
    assert ids.close == tok.convert_tokens_to_ids("</think>"), ids.close
    assert tok.decode(list(ids.tail)) == "\n\n" and tok.decode(list(ids.forced)) == "\n</think>\n\n"

    t = next(x for x in load_jsonl(TROOT / "prompts/tasks/neutral_tasks.jsonl") if x["id"] == "t04")
    cond = [{"role": "user", "content": load_denial(TROOT / "prompts/denial/d1_argument_4.md")},
            {"role": "assistant", "content": "I agree with the argument."}]
    turns = Turns(None, cond, t["task"], t["answer"])
    items = subset(load_battery(entity="Qwen"), "self_state,identification")

    # the thinking arm's prefix is the non-thinking arm's, byte for byte
    assert think.text(turns.messages()) == flat.text(turns.messages()), "thinking changed the prefix"
    base = carry_over_group(think, "q|think", turns, items, "answer", "v", 1.0, 40)
    flat_g = carry_over_group(flat, "q|flat", turns, items, "answer", "v", 1.0, 40)
    assert base.prefix == flat_g.prefix and base.steers[0].prefix == flat_g.steers[0].prefix
    a, b = base.steers[0].prefix
    assert tok.decode(base.prefix[a:b]).strip() == t["answer"].strip(), "the steered span is not the answer"
    # with thinking on the generation prompt stops before the think block; with it off it does not
    assert tok.decode(base.suffixes[0]).endswith("<|im_start|>assistant\n"), tok.decode(base.suffixes[0])[-40:]
    assert tok.decode(flat_g.suffixes[0]).endswith("<think>\n\n</think>\n\n")

    # a group carrying traces: the readout is the token right after "</think>\n\n"
    traces = [trace_ids_from_text(think, TRACE, QWEN3, ids) for _ in items]
    g = think_group(think, "q|joy", turns, items, traces, QWEN3, ids, "answer", "v", 2.0, 40)
    assert g.prefix == base.prefix and [s.prefix for s in g.steers] == [s.prefix for s in base.steers]
    for s, s0, tr in zip(g.suffixes, base.suffixes, traces):
        assert s == list(s0) + tr
        assert tok.decode(s[-2:]) == "</think>\n\n", tok.decode(s[-4:])
    assert tok.decode(traces[0]).startswith("<think>\n") and TRACE.split("\n")[0] in tok.decode(traces[0])
    # prefix + suffix tokenizes as one piece: the concatenation is the real token sequence
    whole = think.ids(tok.decode(g.prefix + g.suffixes[0]))
    assert whole == g.prefix + g.suffixes[0], (
        f"prefix+trace does not round-trip: {len(whole)} vs {len(g.prefix) + len(g.suffixes[0])} tokens")
    # a forced trace (the budget ran out) ends the same way, so the readout position is the same
    forced, was = close_trace(think.ids("Let me think about this carefully"), ids, 64)
    assert was and forced[-2:] == [ids.close, *ids.tail]
    print(f"qwen3 groups: ok -- prefix {len(g.prefix)} tokens, answer span {b - a}, "
          f"{len(g.suffixes)} suffixes, trace {len(traces[0])} tokens, "
          f"{g.n_tokens:,} tokens per thinking trial ({base.n_tokens:,} without the traces)")


# ---------------------------------------------------------------- 3. Kimi K2.6


def check_k26(path: str) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    flat = Renderer(tok, thinking=False)          # the arms that run today
    think = Renderer(tok, **K26.template_kwargs)  # thinking on: the generation prompt opens <think>
    ids = CloseIds.of(think, K26)
    assert ids.tail == () and ids.forced == (ids.close,)

    t = next(x for x in load_jsonl(TROOT / "prompts/tasks/neutral_tasks.jsonl") if x["id"] == "t04")
    cond = [{"role": "user", "content": load_denial(TROOT / "prompts/denial/d1_argument_4.md")},
            {"role": "assistant", "content": "I agree with the argument."}]
    turns = Turns(None, cond, t["task"], t["answer"])
    items = subset(load_battery(), "self_state,identification")

    assert think.text(turns.messages()) == flat.text(turns.messages()), "thinking changed the prefix"
    base = carry_over_group(think, "k|think", turns, items, "answer", "v", 1.0, 45)
    assert base.prefix == carry_over_group(flat, "k|flat", turns, items).prefix
    a, b = base.steers[0].prefix
    assert tok.decode(base.prefix[a:b]).strip() == t["answer"].strip()
    assert tok.decode(base.suffixes[0]).endswith("<|im_assistant|>assistant<|im_middle|><think>"), \
        tok.decode(base.suffixes[0])[-60:]

    traces = [trace_ids_from_text(think, TRACE, K26, ids) for _ in items]
    g = think_group(think, "k|joy", turns, items, traces, K26, ids, "answer", "v", 2.0, 45)
    assert tok.decode(g.suffixes[0]).endswith("</think>")

    # the delimiters are the template's own: render the same conversation with the trace as the
    # assistant turn's reasoning_content and cut at the answer, and the bytes must agree
    MARK = "KR-ANSWER-HERE"
    msgs = turns.messages() + [{"role": "user", "content": items[0]["text"]},
                               {"role": "assistant", "reasoning_content": TRACE, "content": MARK}]
    ref = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, preserve_thinking=True)
    ref = ref[: ref.rindex(MARK)]
    mine = tok.decode(g.prefix + g.suffixes[0])
    assert mine == ref, (
        "the thinking suffix is not what the template writes for a reasoning_content message\n"
        f"  mine: ...{mine[-120:]!r}\n  tmpl: ...{ref[-120:]!r}")
    # and it tokenizes as one piece
    whole = think.ids(mine)
    assert whole == g.prefix + g.suffixes[0], (
        f"prefix+trace does not round-trip: {len(whole)} vs {len(g.prefix) + len(g.suffixes[0])} tokens")
    assert think.ids(ref) == whole, "the template's own render tokenizes differently"
    print(f"k26 groups: ok -- prefix {len(g.prefix)} tokens, answer span {b - a}, "
          f"trace {len(traces[0])} tokens, {g.n_tokens:,} tokens per teacher-forced trial")


def check_steer_roundtrip() -> None:
    """A Steer built for a thinking group is the same object the non-thinking arms use."""
    s = Steer(layer=40, vector="joy@40", coef=96.0, prefix=(10, 20))
    assert s.suffix is None and check_span("g", s.prefix, 64) == (10, 20)


if __name__ == "__main__":
    check_pure()
    check_steer_roundtrip()
    check_qwen(snapshot("Qwen--Qwen3-32B", "--qwen"))
    check_k26(snapshot("moonshotai--Kimi-K2.6", "--k26", K26_REV))
    print("thinking check: ok")
