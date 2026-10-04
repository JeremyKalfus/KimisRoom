"""vLLM backend for the thinking arm: the same interface as `think_hf.HFThinker`, fast enough
for a real grid.

Why vLLM at all: a thinking trial generates a reasoning trace per battery item, and every item of
one condition sits behind the same long carry-over prefix (conditioning, task, steered answer).
With prefix caching that prefix is prefilled once per condition instead of once per item, which is
the whole cost of the arm.

**Steering.** A forward hook on decoder layer L adds `coef * v` to the rows whose entry in the
`positions` tensor -- the absolute position vLLM passes to the model's forward, and so to each
decoder layer's forward -- lies in the steered span. Three things make that safe:

- positions are absolute (`num_computed_tokens + offset_in_chunk`), so the mask means the same
  thing in a chunked prefill, in a decode step and after a prefix-cache hit, where the layer only
  sees the uncached tail;
- the span is a range inside the prefix, so a generated token can never be in it; the hook records
  the positions it touched and `run` asserts every one of them fell inside the span;
- one condition runs at a time, so every in-flight request shares one prefix and one span, and
  `reset_prefix_cache()` is called whenever the steer changes, because nothing in vLLM's cache
  hash covers hook state and a steered prefix must never be reused unsteered.

A Qwen3 decoder layer returns `(mlp_out, residual)` with the residual add deferred to the next
layer's fused add-and-norm, so the residual stream after layer L is `out[0] + out[1]` and adding
the vector to `out[0]` shifts that stream by exactly the vector -- the same quantity
`hf_engine.HFEngine` adds to the post-layer hidden state. That equivalence is what the parity
check (`tests/vllm_parity.py`) measures rather than assumes.

**Reaching the model.** `worker_extension_cls` (a dotted path, resolved inside each worker and
mixed into vLLM's `Worker`, so `self` is the worker) plus `collective_rpc("method", args=...)` by
name. That avoids shipping a pickled callable, which `collective_rpc(callable)` and `apply_model`
both do and which needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1` to cross the engine's ZMQ boundary.
Vectors travel as plain float lists for the same reason. Verified against vLLM 0.30.0
(https://docs.vllm.ai/en/latest/api/vllm/entrypoints/llm/, `vllm/v1/worker/worker_base.py`).

`enforce_eager=True` by default: CUDA-graph replay runs no Python, so a hook registered after
warmup would be skipped or would fire on padded rows.
"""

from __future__ import annotations

import time

import torch

from .engine import Group, Readout
from .thinking import CloseIds, Trace, assert_no_free_pain, check_span, close_trace, span_mask, steer_delta

WORKER_EXT = "kimisroom.think_vllm.SteerWorkerExtension"
F32 = torch.float32


# ---------------------------------------------------------------- inside the worker


class SteerWorkerExtension:
    """Mixed into vLLM's `Worker` by `worker_extension_cls`; every name here starts with `kr_`
    so it cannot collide with a Worker attribute (vLLM asserts on collisions).

    State lives in the worker process. The hooks are registered once and then re-aimed by
    `kr_set_steer`, so a grid of conditions costs one RPC each and no re-registration.
    """

    def kr_install(self) -> dict:
        model = self.model_runner.get_model()
        layers = model.model.layers
        p = next(model.parameters())
        self.kr_state: dict = {"steer": None, "dtype": p.dtype, "device": p.device}
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self._kr_hook(i))
        return {"n_layers": len(layers), "hidden_size": int(model.config.hidden_size),
                "dtype": str(p.dtype), "device": str(p.device)}

    def _kr_hook(self, i: int):
        """The hook fires on every forward at layer L, decode steps included, so it must not
        synchronise with the host: a boolean-mask index would (its output shape is data
        dependent), and `mask.unsqueeze(1) * vec` is bit-identical because 1 * v is v and adding
        a bf16 zero is exact. Accounting stays on the device until `kr_report`."""

        def hook(module, args, output):
            st = self.kr_state["steer"]
            if st is None or st["layer"] != i:
                return None
            positions = args[0]  # Qwen3DecoderLayer.forward(positions, hidden_states, residual)
            assert positions.ndim == 1, f"expected flat absolute positions, got {tuple(positions.shape)}"
            mask = span_mask(positions, st["span"])
            # out[0] is the MLP output; the residual add is deferred to the next fused add-and-norm,
            # so shifting it shifts the residual stream after this layer by exactly the vector.
            h = output[0] if isinstance(output, tuple) else output
            assert h.shape[0] == positions.shape[0], (tuple(h.shape), tuple(positions.shape))
            h.add_(steer_delta(mask, st["vec"]))
            acc, big = st["acc"], torch.iinfo(torch.int64).max
            acc[0] += mask.sum()
            acc[1] = torch.minimum(acc[1], torch.where(mask, positions, big).min())
            acc[2] = torch.maximum(acc[2], torch.where(mask, positions, -big).max())
            return None  # modified in place

        return hook

    def kr_set_steer(self, layer: int, vec: list[float], coef: float, span: tuple[int, int]) -> None:
        s = self.kr_state
        v = torch.tensor(vec, dtype=F32, device=s["device"]) * coef
        big = torch.iinfo(torch.int64).max
        s["steer"] = {"layer": int(layer), "vec": v.to(s["dtype"]), "span": (int(span[0]), int(span[1])),
                      "acc": torch.tensor([0, big, -big], dtype=torch.int64, device=s["device"])}

    def kr_clear_steer(self) -> None:
        self.kr_state["steer"] = None

    def kr_report(self) -> dict:
        """What the hook touched since the last `kr_set_steer`, for the caller's assert."""
        st = self.kr_state["steer"]
        if st is None:
            return {"n": 0, "lo": None, "hi": None}
        n, lo, hi = (int(x) for x in st["acc"].tolist())
        return {"n": n, "lo": lo if n else None, "hi": hi if n else None}


# ---------------------------------------------------------------- the driver


class VLLMThinker:
    """`vllm.LLM` with the steering hook installed, `think_hf.HFThinker`'s `run` interface.

    `readout` picks how the first-token logprobs come back:
    - "targets": `SamplingParams(logprob_token_ids=targets)`, which returns exactly those ids'
      logprobs, normalised over the full vocab (`logprobs_mode="raw_logprobs"`, the default, is
      taken before any logit processor). Exact, and the only top-id recorded is the sampled one,
      which under greedy decoding is the top-1.
    - "topk": `SamplingParams(logprobs=top_k)` and the targets looked up in the top-k dict. Works
      on a vLLM without `logprob_token_ids`; targets outside the top-k come back as None and are
      counted in `missing`.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.90,
        enforce_eager: bool = True,
        enable_prefix_caching: bool = True,
        readout: str = "targets",
        seed: int = 0,
    ):
        from vllm import LLM

        assert readout in ("targets", "topk"), readout
        self.readout_mode = readout
        self.kw = {"model": model_path, "tensor_parallel_size": tensor_parallel_size, "dtype": dtype,
                   "gpu_memory_utilization": gpu_memory_utilization, "enforce_eager": enforce_eager,
                   "enable_prefix_caching": enable_prefix_caching, "worker_extension_cls": WORKER_EXT,
                   "seed": seed, "logprobs_mode": "raw_logprobs"}
        if max_model_len is not None:
            self.kw["max_model_len"] = max_model_len
        try:
            self.llm = LLM(**self.kw)
        except (TypeError, ValueError) as e:
            # logprobs_mode="raw_logprobs" (logprobs taken before any logit processor, so they are
            # normalised over the full vocab) is already the default; a vLLM that does not accept
            # it as an LLM() kwarg still behaves that way.
            if "logprobs_mode" not in self.kw:
                raise
            print(f"vllm: LLM() rejected logprobs_mode ({type(e).__name__}: {e}); "
                  "retrying without it (raw_logprobs is the default)", flush=True)
            self.kw.pop("logprobs_mode")
            self.llm = LLM(**self.kw)
        self.info = self.llm.collective_rpc("kr_install")
        self.tp = tensor_parallel_size
        assert len(self.info) == self.tp, f"{len(self.info)} workers for tp {self.tp}"
        self.timings: dict[str, float] = {}

    # ---------------- steering

    def _reset_cache(self) -> None:
        """Drop every cached KV block, because the steer is about to change and nothing in vLLM's
        prefix-cache hash covers hook state: a prefix computed under one steer must never be
        reused under another, or unsteered."""
        ok = self.llm.reset_prefix_cache()
        if ok is False:  # blocks still held by a running request
            try:
                ok = self.llm.reset_prefix_cache(reset_running_requests=True)
            except TypeError:
                pass
        assert ok is not False, (
            "reset_prefix_cache() refused to clear the cache; a prefix steered under the previous "
            "condition could be reused, so the run is stopped. Re-run with enable_prefix_caching=False.")

    def _set_steer(self, layer: int, vec: torch.Tensor, coef: float, span: tuple[int, int]) -> None:
        self._reset_cache()
        self.llm.collective_rpc("kr_set_steer", args=(int(layer), vec.tolist(), float(coef), tuple(span)))

    def _clear_steer(self) -> None:
        self._reset_cache()
        self.llm.collective_rpc("kr_clear_steer")

    def _check_touched(self, gid: str, span: tuple[int, int] | None) -> list[dict]:
        rep = self.llm.collective_rpc("kr_report")
        if span is None:
            assert all(r["n"] == 0 for r in rep), f"group {gid}: unsteered, but the hook fired: {rep}"
            return rep
        for rank, r in enumerate(rep):
            assert r["n"] > 0, f"group {gid}: the steer never fired on rank {rank}"
            assert span[0] <= r["lo"] and r["hi"] < span[1], (
                f"group {gid}: rank {rank} steered positions {r['lo']}..{r['hi']} outside the span {span}")
        return rep

    # ---------------- pass

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
        """One condition at a time: generate every trace, then read every trial out.

        Returns group id -> {"traces": [Trace], "readouts": [Readout], "suffixes": [[int]],
        "steered": [per-rank hook report]}.
        """
        from vllm import SamplingParams

        assert ids is not None, "pass the tokenizer's CloseIds"
        vectors, targets = vectors or {}, list(targets or [])
        gen_sp = SamplingParams(temperature=0.0, max_tokens=budget, stop_token_ids=[ids.close])
        read_sp = self._read_params(SamplingParams, targets, top_k)
        out: dict[str, dict] = {}
        self.timings = {"generate": 0.0, "readout": 0.0}
        for gi, g in enumerate(groups):
            span = self._aim(g, vectors)
            t0 = time.perf_counter()
            prompts = [{"prompt_token_ids": g.prefix + list(s)} for s in g.suffixes]
            gens = self.llm.generate(prompts, gen_sp)
            self.timings["generate"] += time.perf_counter() - t0
            traces, full = [], []
            for si, (o, suf) in enumerate(zip(gens, g.suffixes)):
                raw = list(o.outputs[0].token_ids)
                # vLLM's stop_token_ids may or may not keep the stop token; `stop_reason` says it
                # stopped there, so re-append it when it was stripped. close_trace does the rest.
                if o.outputs[0].stop_reason == ids.close and (not raw or raw[-1] != ids.close):
                    raw.append(ids.close)
                trace, forced = close_trace(raw, ids, budget + 1)
                traces.append(Trace(item=str(si), tokens=trace, text="", generated=len(raw), forced=forced))
                full.append(list(suf) + trace)
            t0 = time.perf_counter()
            reads = self._read(g.prefix, full, read_sp, targets, top_k)
            self.timings["readout"] += time.perf_counter() - t0
            out[g.id] = {"traces": traces, "suffixes": full, "collects": [], "collect_norms": [],
                         "readouts": reads, "steered": self._check_touched(g.id, span)}
            if progress:
                progress(gi)
        self._clear_steer()
        return out

    def readout(
        self,
        groups: list[Group],
        vectors: dict[str, torch.Tensor] | None = None,
        targets: list[int] | None = None,
        top_k: int = 20,
        progress=None,
    ) -> dict[str, dict]:
        """`engine.Engine.run`'s readout with nothing generated: each suffix is read at its last
        token under the group's prefix steer.

        This is the parity surface against `hf_engine.HFEngine` (`tests/vllm_parity.py`) and the
        way to read a trial whose trace is already fixed -- a teacher-forced trace, or the
        thinking-off arms.
        """
        from vllm import SamplingParams

        vectors, targets = vectors or {}, list(targets or [])
        read_sp = self._read_params(SamplingParams, targets, top_k)
        out: dict[str, dict] = {}
        self.timings = {"readout": 0.0}
        for gi, g in enumerate(groups):
            span = self._aim(g, vectors)
            t0 = time.perf_counter()
            reads = self._read(g.prefix, [list(s) for s in g.suffixes], read_sp, targets, top_k)
            self.timings["readout"] += time.perf_counter() - t0
            out[g.id] = {"readouts": reads, "collects": [], "collect_norms": [],
                         "steered": self._check_touched(g.id, span)}
            if progress:
                progress(gi)
        self._clear_steer()
        return out

    def _read(self, prefix: list[int], suffixes: list[list[int]], read_sp, targets, top_k) -> list[Readout]:
        rs = self.llm.generate([{"prompt_token_ids": prefix + s} for s in suffixes], read_sp)
        return [self._readout(r, targets, top_k) for r in rs]

    def _aim(self, g: Group, vectors: dict[str, torch.Tensor]) -> tuple[int, int] | None:
        """Point the hook at this group's steer (or clear it) and return the span."""
        assert len(g.steers) <= 1, f"group {g.id}: the thinking arm steers one span, got {len(g.steers)}"
        assert g.readout == "suffix_last", g.readout
        if not g.steers:
            self._clear_steer()
            return None
        s = g.steers[0]
        assert s.suffix is None, f"group {g.id}: a suffix steer would touch the question and the trace"
        assert s.prefix is not None, f"group {g.id}: the thinking arm steers a prefix span"
        assert_no_free_pain(s.vector)
        n_layers = self.info[0]["n_layers"]
        assert 0 <= s.layer < n_layers, f"group {g.id}: steer layer {s.layer} on a {n_layers}-layer model"
        a, b = check_span(g.id, s.prefix, len(g.prefix))
        v = vectors[s.vector]
        hidden = self.info[0]["hidden_size"]
        assert v.ndim == 1 and v.shape[0] == hidden, (s.vector, tuple(v.shape), hidden)
        self._set_steer(s.layer, v.detach().to(F32).cpu(), s.coef, (a, b))
        return (a, b)

    # ---------------- readout

    def _read_params(self, SamplingParams, targets: list[int], top_k: int):
        if self.readout_mode == "topk" or not targets:
            return SamplingParams(temperature=0.0, max_tokens=1, logprobs=top_k)
        assert len(targets) <= 128, f"logprob_token_ids takes at most 128 ids, got {len(targets)}"
        try:
            return SamplingParams(temperature=0.0, max_tokens=1, logprob_token_ids=targets,
                                  logprobs=len(targets))
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                f"this vLLM has no SamplingParams(logprob_token_ids=...) ({e}); rerun with "
                "readout='topk', which looks the targets up in the top-k instead") from None

    def _readout(self, r, targets: list[int], top_k: int) -> Readout:
        o = r.outputs[0]
        d = (o.logprobs or [{}])[0]
        sampled = o.token_ids[0] if o.token_ids else -1
        tgt = torch.tensor([d[t].logprob if t in d else float("nan") for t in targets], dtype=F32)
        if self.readout_mode == "targets":
            lp = d[sampled].logprob if sampled in d else float("nan")
            return Readout(torch.tensor([sampled], dtype=torch.long), torch.tensor([lp], dtype=F32), tgt, None)
        top = sorted(d.items(), key=lambda kv: -kv[1].logprob)[:top_k]
        pad = top_k - len(top)
        return Readout(torch.tensor([t for t, _ in top] + [-1] * pad, dtype=torch.long),
                       torch.tensor([v.logprob for _, v in top] + [float("nan")] * pad, dtype=F32), tgt, None)
