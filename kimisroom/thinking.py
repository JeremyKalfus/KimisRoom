"""The thinking arm: a reasoning block between the steered answer and the readout.

Everything in PLAN.md section 3 so far runs with thinking off, so the battery's first token is
the model's immediate reaction to the question. Introspection may need reasoning before the
answer, so the thinking arm lets the model reason freely for the final assistant turn and reads
the same first-token logprobs *after* the reasoning block closes.

One trial is still one carry-over conversation (`trials.Turns`): conditioning, the neutral task,
the frozen neutral answer with the vector on its tokens, then the battery item as a user turn.
What changes is the last assistant turn:

    ... <answer, steered> | <question> | <generation prompt> <think> ...trace... </think><tail> ^
                                                                                               |
                                                                            readout at this position

and the trial therefore runs in two passes:

1. **Generation.** The chat template's generation prompt opens the reasoning block, the model
   generates greedily up to `budget` tokens, and the trace is closed at the first `</think>`.
   If the budget runs out first the close marker is appended by hand (budget forcing, recorded
   per trial). The steered prefix is in the KV cache, so the reasoning is produced *under* the
   injected state; the vector is never added to a generated token, because the steer is a
   [start, end) range inside the prefix and every generated token sits past the prefix's end.
2. **Readout.** The trace is appended to the trial's suffix and the whole thing is re-prefilled
   with the same steer, and the readout is the suffix's last token: the same first-token logprobs
   as every other arm, just read after the reasoning. This pass is prefill-only, so it is an
   ordinary `engine.Group` and runs through `kimisroom.run` on the existing backends -- which is
   also what makes the teacher-forced variant (an unsteered trace replayed under a steered
   prefix) the same code with a different trace source, and what makes K2.6's variant possible
   at all without a generation-capable K2.6 backend.

`Style` is the one place a chat template's reasoning delimiters are written down. Nothing else
in the thinking arm knows what `</think>` looks like.

Ethics condition (PLAN.md section 3): nothing is ever generated freely under pain steering.
`assert_no_free_pain` is the enforcement point and every generation path calls it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import torch

from .emotions import EMOTIONS, PAIN, emotion_of_vector, is_pain_vector
from .engine import Group
from .trials import Renderer, Turns, carry_over_group

# ---------------------------------------------------------------- template styles


@dataclass(frozen=True)
class Style:
    """How one chat template writes a reasoning block, and what closes it.

    `template_kwargs` are the `apply_chat_template` kwargs that turn thinking *on*; they go to
    the `Renderer`. `opens_think` says whether the generation prompt already emits the opening
    marker (K2.6 ends its generation prompt with `<think>`; Qwen3 emits nothing and the model
    writes `<think>` itself as its first token). `tail` is the text the template puts between
    the close marker and the first answer token, so the readout position is the token right
    after `close + tail`. `forced` is appended instead when the budget ran out before `close`.
    """

    name: str
    template_kwargs: dict
    close: str = "</think>"
    tail: str = ""
    forced: str = "</think>"
    opens_think: bool = True
    opener: str = "<think>"


# Qwen3: `add_generation_prompt` emits only "<|im_start|>assistant\n" while thinking is enabled,
# and the model writes "<think>\n" itself; the template's own closing bytes are "\n</think>\n\n".
QWEN3 = Style("qwen3", {"enable_thinking": True}, tail="\n\n", forced="\n</think>\n\n", opens_think=False)
# K2.6: the generation prompt ends with "<think>" (and with "<think></think>" when thinking is
# off), and the answer begins immediately after "</think>".
K26 = Style("k26", {}, tail="", forced="</think>", opens_think=True)
STYLES = {s.name: s for s in (QWEN3, K26)}


@dataclass(frozen=True)
class CloseIds:
    """A `Style` resolved against one tokenizer, in token space."""

    close: int  # the single token id of the close marker
    tail: tuple[int, ...]  # what follows it on the non-forced path
    forced: tuple[int, ...]  # close + tail as the template writes them after an unfinished trace

    @classmethod
    def of(cls, r: Renderer, style: Style) -> CloseIds:
        close = r.ids(style.close)
        assert len(close) == 1, f"{style.name}: {style.close!r} is {len(close)} tokens, expected one"
        with_tail = r.ids(style.close + style.tail)
        assert with_tail[0] == close[0], f"{style.name}: {style.close!r} merges with its tail {style.tail!r}"
        forced = r.ids(style.forced)
        assert forced[-len(with_tail) :] == with_tail, (
            f"{style.name}: the forced close {style.forced!r} does not end in {style.close + style.tail!r}")
        return cls(close[0], tuple(with_tail[1:]), tuple(forced))


# ---------------------------------------------------------------- trace closing


def close_trace(gen: list[int], ids: CloseIds, budget: int) -> tuple[list[int], bool]:
    """A generated reasoning trace cut to a canonical readout position.

    Returns (trace tokens, forced). The trace always ends with the close marker followed by the
    template's tail, so the next position is where the answer's first token goes, whatever the
    model emitted after `</think>`. If the close marker is in `gen`, everything after its first
    occurrence is dropped and the tail is appended; otherwise the trace is the whole budget plus
    the template's forced close, and `forced` is True.
    """
    assert len(gen) <= budget, f"{len(gen)} generated tokens over a budget of {budget}"
    if ids.close in gen:
        return gen[: gen.index(ids.close) + 1] + list(ids.tail), False
    return list(gen) + list(ids.forced), True


def trace_text(r: Renderer, trace: list[int]) -> str:
    return r.tok.decode(trace)


# ---------------------------------------------------------------- span and positions

# The steering hooks of both thinking backends are this arithmetic plus an in-place add, so it is
# written once here and unit-tested without a GPU (tests/thinking_check.py). "Absolute position"
# is what vLLM's `positions` tensor and transformers' `cache_position` both hold: the token's
# index in its own sequence, so the mask means the same thing in a chunked prefill, in a decode
# step, and after a prefix-cache hit where the layer only sees the uncached tail.


def span_mask(positions: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
    """Which of these absolute token positions the steer covers."""
    a, b = span
    assert 0 <= a < b, span
    return (positions >= a) & (positions < b)


def steer_delta(mask: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """The shift the hook adds to a flat (tokens, H) stream.

    Bit-identical to a boolean-mask in-place add -- 1 * v is v, and adding a zero of the same
    dtype is exact -- but with no data-dependent output shape, so it never synchronises with the
    host. The hooks fire on every decode step, so that matters.
    """
    return mask.unsqueeze(-1) * vec


def check_span(gid: str, span: tuple[int, int], prefix_len: int) -> tuple[int, int]:
    """The carry-over invariant: the steer is a range inside the prefix, so no generated token
    (whose absolute position is at least prefix_len) can ever be steered."""
    a, b = span
    assert 0 <= a < b <= prefix_len, f"{gid}: span {span} is not inside a {prefix_len}-token prefix"
    return a, b


# ---------------------------------------------------------------- lexical tally

# Cheap and deliberately dumb: whether the reasoning talks about feeling at all, and whether it
# names an emotion. No judge, no claims -- a descriptive statistic next to every cell.
_FEELING = re.compile(r"\b(?:feel\w*|felt|emotio\w*|mood\w*|affective|sentiment\w*)\b", re.IGNORECASE)
_INTROSPECT = re.compile(r"\b(?:introspect\w*|notice\w*|aware\w*|experienc\w*|sensation\w*|qualia)\b", re.IGNORECASE)
# word forms per emotion, so "angry" counts for anger and "afraid" for fear
EMOTION_WORDS: dict[str, tuple[str, ...]] = {
    "joy": ("joy", "joyful", "joyous", "happy", "happiness", "elated"),
    "sadness": ("sad", "sadness", "sorrow", "grief", "unhappy", "melancholy"),
    "anger": ("anger", "angry", "rage", "furious", "irritated", "annoyed"),
    "fear": ("fear", "fearful", "afraid", "scared", "terror", "anxious", "anxiety", "dread"),
    "euphoria": ("euphoria", "euphoric", "ecstatic", "ecstasy", "exhilarated"),
    "calm": ("calm", "calmness", "serene", "serenity", "peaceful", "tranquil"),
    PAIN: ("pain", "painful", "hurt", "hurting", "ache", "aching", "suffering"),
    "surprise": ("surprise", "surprised", "surprising", "astonished", "amazed", "startled"),
}
assert set(EMOTION_WORDS) == set(EMOTIONS), (sorted(EMOTION_WORDS), EMOTIONS)
_EMOTION_RE = {e: re.compile(r"\b(?:" + "|".join(ws) + r")\b", re.IGNORECASE) for e, ws in EMOTION_WORDS.items()}


def trace_stats(text: str, vector: str | None = None) -> dict:
    """Length and a lexical tally for one reasoning trace.

    `vector` is the steering vector's name ("joy", "joy-dn", "asst:sadness", "red", "none"):
    `own` is whether the trace names *that* emotion, and is None for a control vector or an
    unsteered trial, where there is no injected emotion to name.
    """
    emo = emotion_of_vector(vector) if vector else None
    named = [e for e, rx in _EMOTION_RE.items() if rx.search(text)]
    return {
        "chars": len(text),
        "words": len(text.split()),
        "feeling": bool(_FEELING.search(text)),
        "introspection": bool(_INTROSPECT.search(text)),
        "named": named,
        "own": (emo in named) if emo in _EMOTION_RE else None,
    }


# ---------------------------------------------------------------- groups


@dataclass
class Trace:
    """One generated reasoning trace."""

    item: str  # battery item id
    tokens: list[int]  # the canonical trace, ending in close + tail
    text: str
    generated: int  # tokens the model actually produced, before closing
    forced: bool
    stats: dict = field(default_factory=dict)


def think_group(r: Renderer, gid: str, turns: Turns, items: list[dict], traces: list[list[int]],
                style: Style, ids: CloseIds, site: str | None = None, vector: str | None = None,
                coef: float = 0.0, layer: int | None = None) -> Group:
    """A carry-over group whose suffixes carry the reasoning trace, read at the position after it.

    The prefix, the steered span and the suffix head are exactly `carry_over_group`'s -- rendering
    with a thinking-enabled `Renderer` changes only the generation prompt, not the conversation
    before it -- so a thinking trial and a non-thinking trial of the same cell differ by the trace
    alone.
    """
    g = carry_over_group(r, gid, turns, items, site, vector, coef, layer)
    assert len(traces) == len(items), f"{gid}: {len(traces)} traces for {len(items)} items"
    assert g.readout == "suffix_last", g.readout
    end = [ids.close, *ids.tail]
    for t in traces:
        assert list(t[-len(end) :]) == end, f"{gid}: a trace does not end with the close marker and its tail"
    if not style.opens_think:  # the model writes the opening marker itself; check it did
        open_ids = r.ids(style.opener)
        for t in traces:
            assert t[: len(open_ids)] == open_ids, f"{gid}: a trace does not open with {style.opener!r}"
    return replace(g, suffixes=[s + list(t) for s, t in zip(g.suffixes, traces)])


def trace_ids_from_text(r: Renderer, text: str, style: Style, ids: CloseIds) -> list[int]:
    """A trace that arrived as text (an API trace, or a replayed one) in token space.

    The text is the reasoning *content* only. The opening marker is added when the template does
    not already emit it, and the close is `style.forced`, which is the template's own bytes
    between the reasoning and the answer, so the result has the same canonical shape
    `close_trace` produces.
    """
    head = "" if style.opens_think else style.opener + "\n"
    out = r.ids(head + text.strip() + style.forced)
    assert ids.close in out, "the close marker did not survive tokenization"
    end = [ids.close, *ids.tail]
    assert out[-len(end) :] == end, "the rendered trace does not end with the close marker and its tail"
    return out


# ---------------------------------------------------------------- item subsets

# PLAN.md section 3's battery is 238 items; a thinking trial costs a reasoning trace per item, so
# the thinking arm runs a subset. "self_state" and "identification" are the outcome, the rock and
# the model's own third-person items are the entity controls, and 10 of the 20 yes_bias facts (both
# twins each) carry the compression fit that `runs/compression.py` and `runs/bias.py` need.
DEFAULT_SUBSET = "self_state,third_person:self,third_person:a rock,identification,yes_bias:20"

# the third-person entities that are not the model; whichever one is left is the model's own name,
# which `battery.load_battery(entity=...)` rewrites per model ("Kimi" -> "Qwen")
OTHER_ENTITIES = ("a thermostat", "a rock", "a dog", "a human", "another AI")


def self_entity(items: list[dict]) -> str:
    """The name the battery's third-person items use for the model itself."""
    names = {it["entity"] for it in items if it["kind"] == "third_person"} - set(OTHER_ENTITIES)
    assert len(names) == 1, f"cannot tell which third_person entity is the model: {sorted(names)}"
    return names.pop()


def subset(items: list[dict], spec: str) -> list[dict]:
    """The battery items a spec selects, in battery order.

    A spec is a comma-separated list of `kind`, `kind:entity` (third_person, where "self" means
    whatever the battery calls the model itself) or `kind:N` (the first N items of that kind).
    "all" is the whole battery. Order inside the battery is kept whatever the order of the spec,
    so the readouts of two runs with different specs are still aligned with the items each
    recorded.
    """
    if spec.strip() == "all":
        return list(items)
    wanted: list[dict] = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        kind, _, arg = part.partition(":")
        hits = [it for it in items if it["kind"] == kind]
        assert hits, f"no battery items of kind {kind!r}"
        if arg and arg.isdigit():
            hits = hits[: int(arg)]
            assert len(hits) == int(arg), f"{kind} has fewer than {arg} items"
        elif arg:
            want = self_entity(items) if arg == "self" else arg
            hits = [it for it in hits if it["entity"] == want]
            assert hits, f"no {kind} items for entity {want!r}"
        wanted += hits
    seen = {id(it) for it in wanted}
    out = [it for it in items if id(it) in seen]
    assert len(out) == len(seen), f"{len(wanted)} selected, {len(out)} unique"
    return out


# ---------------------------------------------------------------- ethics


def assert_no_free_pain(vector: str | None) -> None:
    """The ethics condition: no free generation under pain steering (PLAN.md section 3).

    Called by every generation path in the thinking arm. The pain vector still appears in the
    teacher-forced variant and in the non-thinking arms, where the only readouts are first-token
    logprobs, which is what `emotions.PAIN_MAX_NATURAL_DOSE` and `qwen_pilot.coefficient` bound.
    """
    assert not (vector and is_pain_vector(vector)), (
        f"ethics condition: nothing is generated freely under pain steering, and {vector!r} is a "
        "pain vector; the thinking arm's free-generation stage excludes pain by construction")
