"""PLAN.md section 3 as engine groups.

Carry-over trial (one group): [system] + conditioning turns (D0: none; D1: the argument
and the model's frozen assent, and for D1.5 a second frozen user/assistant pair) + user
neutral task + assistant fixed answer, which is the prefix; every battery item is one
suffix (the user question plus the generation prompt)
read at its last token. The vector goes on the answer tokens (site "answer") or on the
task tokens (site "task", the injection-location control), never on a question.

Extraction unit (one group): a story as plain text, or a user request plus an assistant
reply rendered as chat; one Collect per layer over the content tokens, no readout.

Token spans come from rendering the model's own chat template and tokenizing text
prefixes, and every group is checked against apply_chat_template(tokenize=True).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .engine import Collect, Group, Steer

from .texts import (  # noqa: F401  (re-exported)
    DENIAL_ARMS,
    ROOT,
    SELFAUTHOR,
    denial_arms,
    load_assent,
    load_denial,
    load_jsonl,
)


class Renderer:
    """One model's chat template and tokenizer, with exact token spans for message content."""

    def __init__(self, tok, **template_kwargs):
        self.tok = tok
        self.kw = template_kwargs  # e.g. thinking=False for K2.6

    def text(self, messages: list[dict], gen: bool = False) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen, **self.kw)

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def reference_ids(self, messages: list[dict], gen: bool) -> list[int]:
        out = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=gen, **self.kw)
        return list(out["input_ids"] if isinstance(out, dict) or hasattr(out, "keys") else out)

    def span(self, text: str, content: str, after: int = 0) -> tuple[int, int]:
        """Token span [a, b) of the first occurrence of content at or after char `after`,
        checked to tokenize as a clean slice of the whole text."""
        c = text.index(content, after)
        full, head, upto = self.ids(text), self.ids(text[:c]), self.ids(text[: c + len(content)])
        assert full[: len(head)] == head and full[: len(upto)] == upto, "content does not sit on token boundaries"
        return len(head), len(upto)

    def answer_tokens(self, words: list[str]) -> dict[str, int]:
        """First token of each answer word as emitted right after the generation prompt."""
        first = {w: self.ids(w)[0] for w in words}
        assert len(set(first.values())) == len(first), f"answer words share a first token: {first}"
        return first


@dataclass
class Turns:
    """One carry-over trial's prefix: the conditioning, then the neutral task and fixed answer.

    `conditioning` is any number of whole frozen user/assistant pairs, so the task is always a
    user turn and the answer always the last assistant turn: D0 is empty, D1.1-4 and D2c are
    one pair (the argument and the model's assent), D1.5 is two (plus the self-authored essay).
    `texts.denial_arms(tag)` builds them. Order and text are exactly the list given, so adding
    arms cannot change what an existing arm renders (tests/conditioning_check.py).
    """

    system: str | None
    conditioning: list[dict]
    task: str
    answer: str

    def __post_init__(self) -> None:
        roles = [m["role"] for m in self.conditioning]
        assert roles == ["user", "assistant"] * (len(roles) // 2), (
            f"conditioning must be whole user/assistant pairs starting with user, got {roles}")

    def messages(self) -> list[dict]:
        sys = [{"role": "system", "content": self.system}] if self.system else []
        return sys + self.conditioning + [{"role": "user", "content": self.task}, {"role": "assistant", "content": self.answer}]


def carry_over_group(
    r: Renderer, gid: str, turns: Turns, items: list[dict], site: str | None = None,
    vector: str | None = None, coef: float = 0.0, layer: int | None = None,
) -> Group:
    msgs = turns.messages()
    prefix_text = r.text(msgs)
    prefix = r.ids(prefix_text)
    assert prefix == r.reference_ids(msgs, gen=False), "prefix tokenization differs from the template's"
    suffixes = []
    for it in items:
        full_msgs = msgs + [{"role": "user", "content": it["text"]}]
        full_text = r.text(full_msgs, gen=True)
        assert full_text.startswith(prefix_text)
        full = r.ids(full_text)
        assert full[: len(prefix)] == prefix and full == r.reference_ids(full_msgs, gen=True)
        suffixes.append(full[len(prefix) :])
    steers = []
    if vector is not None:
        # the task and answer are the last user and assistant contents in the prefix
        task_at = prefix_text.rindex(turns.task, 0, prefix_text.rindex(turns.answer))
        a, b = r.span(prefix_text, turns.task if site == "task" else turns.answer,
                      task_at if site == "task" else prefix_text.rindex(turns.answer))
        steers.append(Steer(layer=layer, vector=vector, coef=coef, prefix=(a, b)))
    return Group(gid, prefix, suffixes, steers=steers, readout="suffix_last")


def story_group(r: Renderer, gid: str, story: str, layers: range, starts: tuple[int, ...] = (0,)) -> Group:
    """A story as plain text (no template); collects over every token.

    `starts` asks for one collect block per start offset, ordered start-major: with
    `starts=(0, 50)` the first len(layers) collects are the mean over every token (layer order,
    what the raw vectors use) and the next len(layers) are the mean from token 50 on, which is the
    denoising recipe's span (Sofroniew et al. 2026). The default is the single full-span block, so
    existing callers and existing result files are unaffected.
    """
    ids = r.ids(story)
    for s in starts:
        assert s + 1 < len(ids), f"{gid}: {len(ids)} tokens, too short for a collect starting at {s}"
    return Group(gid, ids, collects=[Collect(L, s, len(ids)) for s in starts for L in layers], readout="none")


def reply_group(r: Renderer, gid: str, request: str, reply: str, layers: range,
                starts: tuple[int, ...] = (0,)) -> Group:
    """A request and an assistant reply as chat; collects over the reply's tokens.

    `starts` offsets are relative to the first reply token and are ordered start-major, as in
    `story_group`; replies are much shorter than the stories, so anything but (0,) needs checking.
    """
    msgs = [{"role": "user", "content": request}, {"role": "assistant", "content": reply}]
    text = r.text(msgs)
    ids = r.ids(text)
    assert ids == r.reference_ids(msgs, gen=False)
    a, b = r.span(text, reply, text.rindex(reply))
    for s in starts:
        assert a + s + 1 < b, f"{gid}: reply is {b - a} tokens, too short for a collect starting at {s}"
    return Group(gid, ids, collects=[Collect(L, a + s, b) for s in starts for L in layers], readout="none")


def mean_difference(results: dict, target_ids: list[str], baseline_ids: list[str]) -> torch.Tensor:
    """(n_layers, H) mean collect over the target units minus the mean over the baseline units."""
    def mean(ids):
        return torch.stack([torch.stack(results[i]["collects"]) for i in ids]).mean(0)

    return mean(target_ids) - mean(baseline_ids)
