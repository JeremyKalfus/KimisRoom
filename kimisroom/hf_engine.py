"""Hugging Face prefill backend for dense decoder-only checkpoints (the Qwen3 pilot).

The same job format and the same output as `kimisroom.engine.Engine`, so one set of
`trials.py` groups, one `run.py` and one analysis path serve both the K2.6 layer-wise
engine and the pilot. Here the whole model is resident instead of streamed -- one
worker may span several GPUs -- and steering and collecting are done with forward
hooks on the decoder layers.

A group's prefix is forwarded once (batch 1) and its KV cache is then expanded over
fixed chunks of `suffix_chunk` suffix rows, each padded to a multiple of
`engine.SUFFIX_BUCKET` tokens, so every suffix attends to the whole prefix and
causally to itself and never to another suffix: the same attention the engine builds.
Every shape in a pass depends only on the group it belongs to, so a group gives the
same numbers alone or batched with other groups, and the steered and unsteered passes
of one group run through identical kernels.

Conventions copied from the engine (see `Engine.run`):
- a Steer adds `(vectors[name].float() * coef).to(model dtype)` to the residual stream
  *after* decoder layer `layer`, at the steered prefix and suffix positions; the vector
  is one layer's direction, `(H,)`, so a caller holding a `(n_layers, H)` stack indexes
  it before the call, exactly as for the engine;
- a Collect is the mean, and the mean per-token L2 norm, of that residual stream over a
  prefix span, also after the layer, in fp32;
- a readout is an fp32 `log_softmax` of the logits of the final-normed residual stream
  (the logits themselves in the model dtype, as `lm_head` computes them), plus the target
  logits recomputed in fp32 from the target rows alone, taken in fixed row tiles of
  `engine.TILE_HEAD` by the engine's own `readout_tiles`;
- suffix steering positions come from `engine.suffix_steer_spans`, also shared, so a
  per-suffix span means the same tokens in both backends.
"""

from __future__ import annotations

import time

import torch

from .engine import SUFFIX_BUCKET, Group, readout_tiles, suffix_steer_spans
from .trials import Renderer

F32 = torch.float32


class Qwen3Renderer(Renderer):
    """Renderer for Qwen3's chat template, whose assistant turn is not prefix-stable.

    The template writes an empty `<think>\\n\\n</think>` block into an assistant message
    only while that message is the last one, and drops it again as soon as a later user
    turn exists, so `apply_chat_template(msgs)` is *not* a prefix of
    `apply_chat_template(msgs + [question], add_generation_prompt=True)` and
    `trials.carry_over_group` rejects the pair. Rendering a conversation here therefore
    means rendering it with one more user turn and cutting text and ids right before that
    turn: the head every battery item shares, with the assistant turn written the way the
    model will actually see it while answering. A render *with* a generation prompt is the
    sequence that gets scored and is passed through to the template untouched, so the
    tokenization check in `carry_over_group` still compares against `apply_chat_template`.
    """

    CUT = "⁣kimisroom-cut⁣"

    def _cut(self, messages: list[dict], tokenize: bool):
        msgs = list(messages) + [{"role": "user", "content": self.CUT}]
        text = super().text(msgs, gen=False)
        head = text[: text.rindex("<|im_start|>user\n" + self.CUT)]
        ids, full = self.ids(head), super().reference_ids(msgs, gen=False)
        assert full[: len(ids)] == ids, "the cut before the trailing user turn is not on token boundaries"
        return ids if tokenize else head

    def text(self, messages: list[dict], gen: bool = False) -> str:
        return super().text(messages, gen=True) if gen else self._cut(messages, False)

    def reference_ids(self, messages: list[dict], gen: bool) -> list[int]:
        return super().reference_ids(messages, gen=True) if gen else self._cut(messages, True)


def suffix_chunks(suffix_lens: list[int], chunk: int) -> list[tuple[int, list[int]]]:
    """(bucket_length, suffix rows) per chunk, mirroring `engine._GroupLayout.suffix_chunks`:
    suffixes are bucketed by their own length rounded up to SUFFIX_BUCKET, buckets are never
    mixed, and each chunk holds at most `chunk` rows (padded to exactly `chunk` at run time),
    so a suffix's shapes depend only on itself."""
    buckets: dict[int, list[int]] = {}
    for r, L in enumerate(suffix_lens):
        buckets.setdefault(-(-L // SUFFIX_BUCKET) * SUFFIX_BUCKET, []).append(r)
    out = []
    for Ls, rows in sorted(buckets.items()):
        for a in range(0, len(rows), chunk):
            out.append((Ls, rows[a : a + chunk]))
    return out


class _Phase:
    """What the layer hooks must do during one forward pass."""

    def __init__(self):
        self.steer: dict[int, list[tuple[torch.Tensor | tuple[int, int], torch.Tensor]]] = {}
        self.collect: dict[int, list[tuple[int, int, int]]] = {}  # layer -> (slot, start, end)
        self.out: list = []  # (slot, mean, norm) filled by the collect hooks


class HFEngine:
    """`AutoModelForCausalLM` plus forward hooks, with `Engine.run`'s job and output format.

    `device` is a single device ("cuda", "cuda:1", torch.device) for a model that fits on
    one GPU, or "auto" to shard it over every visible GPU with accelerate's device map.
    """

    def __init__(
        self,
        model_path: str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        suffix_chunk: int = 16,
        attn_implementation: str = "sdpa",
        max_memory: dict | None = None,
    ):
        from transformers import AutoModelForCausalLM

        self.dtype = dtype
        self.suffix_chunk = suffix_chunk
        kw: dict = {"dtype": dtype, "attn_implementation": attn_implementation}
        if str(device) == "auto":
            kw["device_map"] = "auto"
            if max_memory is not None:
                kw["max_memory"] = max_memory
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
        self.model.eval()
        if str(device) != "auto":
            self.model.to(device)
        self.c = self.model.config
        self.inner = self.model.model  # the decoder stack, including the final norm
        self.layers = self.inner.layers
        self.head = self.model.lm_head
        self.n_layers = self.c.num_hidden_layers
        assert len(self.layers) == self.n_layers, (len(self.layers), self.n_layers)
        # A 4-D attention mask is handed straight to every layer, so a model with sliding
        # layers would get the wrong mask for them.
        types = getattr(self.c, "layer_types", None)
        assert not types or set(types) == {"full_attention"}, f"sliding attention is not supported: {set(types)}"
        self.in_device = self.model.get_input_embeddings().weight.device
        self.head_device = self.head.weight.device
        self._phase: _Phase | None = None
        self._vec_cache: dict[tuple[int, float], torch.Tensor] = {}
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._make_hook(i))

    # ---------------- hooks

    def _make_hook(self, i: int):
        def hook(module, args, output):
            ph = self._phase
            if ph is None:
                return None
            h = output[0] if isinstance(output, tuple) else output  # (B, T, H), after layer i
            for where, vec in ph.steer.get(i, []):
                v = vec.to(h.device, non_blocking=False)
                if isinstance(where, tuple):  # prefix span, batch of 1
                    a, b = where
                    h[0, a:b] += v
                else:  # (B, T) bool mask over the suffix chunk
                    h[where.to(h.device)] += v
            for slot, a, b in ph.collect.get(i, []):
                span = h[0, a:b].to(F32)
                ph.out.append((slot, span.mean(0).cpu(), span.norm(dim=-1).mean().item()))
            return None  # the output was modified in place

        return hook

    def _vec(self, vectors: dict[str, torch.Tensor], s) -> torch.Tensor:
        v = vectors[s.vector]
        assert v.ndim == 1 and v.shape[0] == self.c.hidden_size, (
            f"steering vector {s.vector!r} has shape {tuple(v.shape)}, expected ({self.c.hidden_size},); "
            "index a (n_layers, H) stack at the steered layer before calling"
        )
        key = (id(v), s.coef)
        if key not in self._vec_cache:
            self._vec_cache[key] = (v.detach().to(F32).cpu() * s.coef).to(self.dtype)
        return self._vec_cache[key]

    # ---------------- pass

    @torch.no_grad()
    def run(
        self,
        groups: list[Group],
        vectors: dict[str, torch.Tensor] | None = None,
        targets: list[int] | None = None,
        top_k: int = 20,
        progress=None,
        layer_source=None,
    ) -> dict[str, dict]:
        """One prefill of these groups. Returns `Engine.run`'s output:
        group id -> {"collects": [(H,) fp32 cpu], "collect_norms": [float], "readouts": [Readout]}."""
        assert layer_source is None, "HFEngine holds the whole model; there is no layer source"
        from transformers import DynamicCache

        vectors = vectors or {}
        self._vec_cache = {}
        self.timings: dict[str, float] = {}
        collected = {g.id: [None] * len(g.collects) for g in groups}
        norms = {g.id: [None] * len(g.collects) for g in groups}
        rows: list[torch.Tensor] = []  # final-normed hidden state per readout row
        owner: list[str] = []
        nxt: list[int] = []

        for gi, g in enumerate(groups):
            Lp = len(g.prefix)
            assert Lp > 0, f"group {g.id} has an empty prefix"
            ph = _Phase()
            # per steer, one [start, end) per suffix (engine.suffix_steer_spans, shared with Engine)
            suffix_steer: dict[int, list[tuple[list[tuple[int, int]], torch.Tensor]]] = {}
            for s in g.steers:
                assert 0 <= s.layer < self.n_layers, f"group {g.id}: steer layer {s.layer}"
                vec = self._vec(vectors, s)
                if s.prefix is not None:
                    ph.steer.setdefault(s.layer, []).append((s.prefix, vec))
                spans = suffix_steer_spans(s, [len(x) for x in g.suffixes])
                if spans is not None:
                    suffix_steer.setdefault(s.layer, []).append((spans, vec))
            for k, col in enumerate(g.collects):
                assert col.end <= Lp, f"group {g.id}: collect [{col.start}, {col.end}) runs past the prefix"
                ph.collect.setdefault(col.layer, []).append((k, col.start, col.end))

            # ---- prefix, batch 1, its own causal mask
            t0 = time.perf_counter()
            need_cache = bool(g.suffixes)
            ids = torch.tensor([g.prefix], dtype=torch.long, device=self.in_device)
            self._phase = ph
            try:
                out = self.inner(input_ids=ids, use_cache=need_cache)
            finally:
                self._phase = None
            for k, mean, nrm in ph.out:
                collected[g.id][k], norms[g.id][k] = mean, nrm
            ph.collect, ph.out = {}, []  # the collects are prefix spans; they must not refire
            hp = out.last_hidden_state  # (1, Lp, H), after the final norm
            if g.readout == "prefix_last":
                rows.append(hp[0, Lp - 1].clone())
                owner.append(g.id)
                nxt.append(-1)
            elif g.readout == "prefix_all":
                rows += list(hp[0, : Lp - 1].clone())
                owner += [g.id] * (Lp - 1)
                nxt += g.prefix[1:]
            kv = out.past_key_values.to_legacy_cache() if need_cache else ()
            del out, hp
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            # ---- suffixes, fixed chunks of `suffix_chunk` rows on a SUFFIX_BUCKET grid
            suffix_rows: list[torch.Tensor | None] = [None] * len(g.suffixes)
            m = self.suffix_chunk
            for Ls, chunk in suffix_chunks([len(s) for s in g.suffixes], m):
                sid = torch.zeros(m, Ls, dtype=torch.long)
                valid = torch.zeros(m, Ls, dtype=torch.bool)
                for i, r in enumerate(chunk):
                    s = g.suffixes[r]
                    sid[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                    valid[i, : len(s)] = True
                dev = self.in_device
                # every suffix key is the whole prefix plus its own causal, unpadded, keys
                allow = torch.ones(m, 1, Ls, Lp + Ls, dtype=torch.bool, device=dev)
                own = ~torch.ones(Ls, Ls, dtype=torch.bool, device=dev).triu(1)
                allow[:, :, :, Lp:] = own & valid.to(dev)[:, None, None, :]
                pos = torch.arange(Lp, Lp + Ls, device=dev)
                ph.steer = {}
                for layer, specs in suffix_steer.items():
                    for spans, vec in specs:
                        mask = torch.zeros(m, Ls, dtype=torch.bool)
                        for i, r in enumerate(chunk):
                            a, b = spans[r]
                            mask[i, a:b] = True
                        ph.steer.setdefault(layer, []).append((mask, vec))
                ck = DynamicCache.from_legacy_cache(
                    tuple((k.expand(m, -1, -1, -1), v.expand(m, -1, -1, -1)) for k, v in kv)
                )
                self._phase = ph
                try:
                    out = self.inner(
                        input_ids=sid.to(dev),
                        attention_mask=allow,
                        position_ids=pos.unsqueeze(0).expand(m, -1),
                        past_key_values=ck,
                        use_cache=True,
                        cache_position=pos,
                    )
                finally:
                    self._phase = None
                hs = out.last_hidden_state  # (m, Ls, H)
                for i, r in enumerate(chunk):
                    suffix_rows[r] = hs[i, len(g.suffixes[r]) - 1].clone()
                del out, ck, hs
            del kv
            if g.readout == "suffix_last":
                rows += suffix_rows
                owner += [g.id] * len(suffix_rows)
                nxt += [-1] * len(suffix_rows)
            else:
                assert g.readout in ("prefix_last", "prefix_all", "none"), g.readout
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            for key, dt in (("prefix", t1 - t0), ("suffix", t2 - t1)):
                self.timings[key] = self.timings.get(key, 0.0) + dt
            if progress:
                progress(gi)

        out_dict: dict[str, dict] = {
            g.id: {"collects": collected[g.id], "collect_norms": norms[g.id], "readouts": []} for g in groups
        }
        t0 = time.perf_counter()
        if rows:
            # Engine.run's readout, the same function: fixed row tiles, fp32 log_softmax, fp32 target logits
            hr = torch.stack([r.to(self.head_device) for r in rows])
            for gid, ro in zip(owner, readout_tiles(hr, self.head.weight, targets, nxt, top_k)):
                out_dict[gid]["readouts"].append(ro)
        self.timings["readout"] = time.perf_counter() - t0
        return out_dict
