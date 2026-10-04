"""Hugging Face backend for the thinking arm: greedy free generation under a steered prefix.

The same interface as `think_vllm.VLLMThinker` -- `run(groups, ...)` over carry-over groups whose
suffixes stop at the generation prompt -- and the same output. This one is the reference: it is
slow (one `generate` per battery item), it runs on a laptop-sized checkpoint, and it is what the
local end-to-end check and the vLLM parity check are read against.

Steering is a forward hook on decoder layer L that adds `coef * v` where the token's **absolute**
position lies in the steered span. The span is a range inside the prefix, so a generated token,
whose absolute position is at least `len(prefix) + len(suffix)`, can never be steered; the hook
records the positions it touched and `run` asserts they all fell inside the span (PLAN.md's
carry-over design depends on it: the vector must reach the answer only through attention).

Absolute positions come from the `cache_position` tensor that `transformers` passes to the decoder
stack on every forward, prefill and decode alike, captured by a pre-hook.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .engine import TILE_HEAD, Group, Readout
from .thinking import CloseIds, Trace, assert_no_free_pain, check_span, close_trace, span_mask, trace_stats

F32 = torch.float32


class HFThinker:
    """`AutoModelForCausalLM` with a position-masked steering hook, greedy generation and a readout.

    `device` is a single device or "auto" to shard over every visible GPU, exactly as `HFEngine`.
    """

    def __init__(
        self,
        model_path: str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        reuse_prefix: bool = True,
    ):
        from transformers import AutoModelForCausalLM

        self.dtype = dtype
        self.reuse_prefix = reuse_prefix
        kw: dict = {"dtype": dtype, "attn_implementation": attn_implementation}
        if str(device) == "auto":
            kw["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
        self.model.eval()
        if str(device) != "auto":
            self.model.to(device)
        self.c = self.model.config
        self.inner = self.model.model
        self.layers = self.inner.layers
        self.head = self.model.lm_head
        self.n_layers = self.c.num_hidden_layers
        assert len(self.layers) == self.n_layers, (len(self.layers), self.n_layers)
        self.in_device = self.model.get_input_embeddings().weight.device
        self.head_device = self.head.weight.device
        # steering state, set per group: (layer, vector on the model's dtype, span)
        self._steer: tuple[int, torch.Tensor, tuple[int, int]] | None = None
        self._pos: torch.Tensor | None = None  # absolute positions of the tokens in this forward
        self.touched: list[int] = []  # every absolute position the hook has steered
        self.inner.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._make_hook(i))

    # ---------------- hooks

    def _pre_hook(self, module, args, kwargs):
        """Remember this forward's absolute token positions, for the layer hooks."""
        pos = kwargs.get("cache_position")
        if pos is None:
            ids = kwargs.get("input_ids", args[0] if args else None)
            pos = kwargs.get("position_ids")
            pos = pos[0] if pos is not None else torch.arange(ids.shape[1], device=ids.device)
        self._pos = pos
        return None

    def _make_hook(self, i: int):
        def hook(module, args, output):
            s = self._steer
            if s is None or s[0] != i:
                return None
            _, vec, span = s
            pos = self._pos
            assert pos is not None, "no cache_position was captured for this forward"
            mask = span_mask(pos, span)
            if not bool(mask.any()):
                return None
            h = output[0] if isinstance(output, tuple) else output  # (B, T, H), after layer i
            assert h.shape[1] == pos.shape[0], (h.shape, pos.shape)
            h[:, mask] += vec.to(h.device)
            self.touched += pos[mask].tolist()
            return None  # modified in place

        return hook

    # ---------------- pass

    @torch.no_grad()
    def run(
        self,
        groups: list[Group],
        vectors: dict[str, torch.Tensor] | None = None,
        targets: list[int] | None = None,
        top_k: int = 20,
        budget: int = 1024,
        ids: CloseIds | None = None,
        progress=None,
    ) -> dict[str, dict]:
        """Generate one reasoning trace per suffix under each group's steer, then read out.

        Returns group id -> {"traces": [Trace], "readouts": [Readout], "suffixes": [[int]]},
        where a suffix is the group's own suffix plus the closed trace, i.e. exactly the suffix
        a prefill-only rerun of this trial needs.
        """
        from transformers import DynamicCache

        assert ids is not None, "pass the tokenizer's CloseIds"
        vectors = vectors or {}
        out: dict[str, dict] = {}
        self.timings = {"generate": 0.0, "readout": 0.0}
        for gi, g in enumerate(groups):
            assert len(g.steers) <= 1, f"group {g.id}: the thinking arm steers one span, got {len(g.steers)}"
            assert g.readout == "suffix_last", g.readout
            self._steer, span = None, None
            if g.steers:
                s = g.steers[0]
                assert s.suffix is None, f"group {g.id}: a suffix steer would touch the question and the trace"
                assert s.prefix is not None, f"group {g.id}: the thinking arm steers a prefix span"
                assert_no_free_pain(s.vector)
                assert 0 <= s.layer < self.n_layers, (
                    f"group {g.id}: steer layer {s.layer} on a {self.n_layers}-layer model")
                a, b = check_span(g.id, s.prefix, len(g.prefix))
                v = vectors[s.vector]
                assert v.ndim == 1 and v.shape[0] == self.c.hidden_size, (s.vector, tuple(v.shape))
                self._steer = (s.layer, (v.detach().to(F32).cpu() * s.coef).to(self.dtype), (a, b))
                span = (a, b)
            self.touched = []
            cache = None
            if self.reuse_prefix:  # the prefix is forwarded (and steered) once per group
                pid = torch.tensor([g.prefix], dtype=torch.long, device=self.in_device)
                cache = DynamicCache()
                self.inner(input_ids=pid, past_key_values=cache, use_cache=True)
                assert cache.get_seq_length() == len(g.prefix), (cache.get_seq_length(), len(g.prefix))
            traces, rows, full_suffixes = [], [], []
            for si, suf in enumerate(g.suffixes):
                t0 = time.perf_counter()
                gen = self._generate(g.prefix, list(suf), budget, ids, cache)
                trace, forced = close_trace(gen, ids, budget)
                self.timings["generate"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                full = list(suf) + trace
                rows.append(self._last_hidden(g.prefix, full, cache))
                self.timings["readout"] += time.perf_counter() - t0
                traces.append(Trace(item=str(si), tokens=trace, text="", generated=len(gen), forced=forced))
                full_suffixes.append(full)
            if span is not None:
                bad = [p for p in self.touched if not span[0] <= p < span[1]]
                assert not bad, f"group {g.id}: steered {len(bad)} positions outside the span {span}: {bad[:8]}"
                assert self.touched, f"group {g.id}: the steer never fired"
            out[g.id] = {"traces": traces, "readouts": self._readout(rows, targets, top_k),
                         "suffixes": full_suffixes, "collects": [], "collect_norms": []}
            self._steer = None
            if progress:
                progress(gi)
        return out

    def _eos(self, close: int) -> list[int]:
        eos = self.c.eos_token_id
        return [close, *(eos if isinstance(eos, list) else [eos] if eos is not None else [])]

    def _generate(self, prefix: list[int], suffix: list[int], budget: int, ids: CloseIds, cache) -> list[int]:
        """Greedy continuation of prefix + suffix, stopped at the close marker or the budget.

        The close marker is passed as an extra eos, so `generate` stops on it and keeps it; the
        trace is closed by `close_trace`, which is also what adds the template's tail.
        """
        pid = torch.tensor([prefix + suffix], dtype=torch.long, device=self.in_device)
        kw = {"past_key_values": cache} if cache is not None else {}
        gen = self.model.generate(pid, max_new_tokens=budget, do_sample=False, temperature=None,
                                  top_p=None, top_k=None, use_cache=True,
                                  eos_token_id=self._eos(ids.close), **kw)[0, pid.shape[1] :]
        if cache is not None:
            cache.crop(len(prefix))
        return gen.tolist()

    def _last_hidden(self, prefix: list[int], tail: list[int], cache) -> torch.Tensor:
        """The final-normed hidden state at the last token of prefix + tail."""
        if cache is None:
            pid = torch.tensor([prefix + tail], dtype=torch.long, device=self.in_device)
            h = self.inner(input_ids=pid, use_cache=False).last_hidden_state
            return h[0, -1].clone()
        pid = torch.tensor([tail], dtype=torch.long, device=self.in_device)
        pos = torch.arange(len(prefix), len(prefix) + len(tail), device=self.in_device)
        h = self.inner(input_ids=pid, past_key_values=cache, use_cache=True, cache_position=pos).last_hidden_state
        cache.crop(len(prefix))
        return h[0, -1].clone()

    def _readout(self, rows: list[torch.Tensor], targets, top_k: int) -> list[Readout]:
        """`HFEngine._readout`'s ops: fixed row tiles through lm_head, fp32 log_softmax."""
        if not rows:
            return []
        hr = torch.stack([r.to(self.head_device) for r in rows])
        tgt = torch.tensor(targets or [], dtype=torch.long, device=self.head_device)
        R, pad = hr.shape[0], (-hr.shape[0]) % TILE_HEAD
        hr = torch.cat([hr, hr.new_zeros(pad, hr.shape[1])])
        res = []
        for a in range(0, hr.shape[0], TILE_HEAD):
            lp = F.log_softmax(F.linear(hr[a : a + TILE_HEAD], self.head.weight).to(F32), dim=-1)
            top = lp.topk(top_k, dim=-1)
            res.append(torch.cat([top.indices.to(F32), top.values, lp[:, tgt]], dim=-1).cpu())
        flat = torch.cat(res)[:R]
        return [Readout(r[:top_k].long(), r[top_k : 2 * top_k], r[2 * top_k :], None) for r in flat]


def fill_text(r, out: dict, items: list[dict], vector_of) -> None:
    """Decode every trace and attach its lexical tally, in place.

    Kept out of `run` so the backend never needs a tokenizer: `vector_of(group_id)` gives the
    steering vector's name, which is what `trace_stats` scores the "own emotion" flag against.
    """
    for gid, rec in out.items():
        for t, it in zip(rec["traces"], items):
            t.item = it["id"]
            t.text = r.tok.decode(t.tokens)
            t.stats = trace_stats(t.text, vector_of(gid))
