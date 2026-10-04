# Thinking arm: GPU runbook

The thinking arm of PLAN.md section 3: the model reasons freely for the final assistant turn, under
a carry-over steer on the frozen neutral answer, and the battery readout is taken on the token
right after the reasoning block closes. Design in `kimisroom/thinking.py`; the Qwen3-32B stages in
`runs/qwen_thinking.py`; the K2.6 teacher-forced build in `runs/k26_thinking.py`.

Everything here was written and checked on a Mac (`tests/thinking_check.py`,
`tests/thinking_tiny.py`, `tests/thinking_stages_check.py`, all passing). **No GPU step below has
been executed.** Times and costs are estimates with the arithmetic shown, not measurements.

---

## 0. One-time: a separate vLLM venv

vLLM pins its own torch. Do **not** install it into the DLAMI's `/opt/pytorch` environment, which
is what `kimisroom/hf_engine.py` and `kimisroom/engine.py` run on.

```bash
# on the GPU box, repo at ~/KimisRoom, model already at /nvme/qwen/Qwen3-32B
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv   # need R580+ for the cu130 wheel
uv venv --python 3.12 --seed ~/vllmenv
source ~/vllmenv/bin/activate
uv pip install vllm --torch-backend=auto            # 0.30.0; PyPI default wheel is CUDA 13.0
# driver older than R580 -> take the cu129 asset instead:
# uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
#   --extra-index-url https://download.pytorch.org/whl/cu129
uv pip install accelerate safetensors   # keep vLLM's transformers (>= 5.10); 4.57.1 breaks vLLM
python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
```

Two box prerequisites found on kr-q 2026-10-04: Triton compiles a CUDA helper at engine start and
needs the interpreter's headers (`sudo apt-get install -y python3.12-dev`), and FlashInfer's
sampler JIT-compiles with `nvcc`, which the DLAMI does not ship, so set
`VLLM_USE_FLASHINFER_SAMPLER=0` (vLLM's PyTorch sampler; the arm is greedy anyway).

`PYTHONPATH` must include the repo, because vLLM resolves `worker_extension_cls` by dotted name
*inside each worker process*:

```bash
export PYTHONPATH=$HOME/KimisRoom
```

L40S is sm_89 and meets vLLM's minimum. Qwen3-32B bf16 is ~64 GB of weights: `--tp 2` on the
g6e.12xlarge's 4x L40S (48 GB each), or `--tp 1` on one H100 80 GB, where the KV cache has about
16 GB to live in (enough: the shared prefix is stored once under prefix caching, and the per-request
tail is only the question plus the trace).

## 1. Parity. Nothing below counts until this passes.

```bash
source ~/vllmenv/bin/activate && export PYTHONPATH=$HOME/KimisRoom && cd $HOME/KimisRoom
P="--vectors $HOME/qwenvec2/vectors.pt --layer 40 --dose 2 --tp 2 --items 24"
python tests/vllm_parity.py /nvme/qwen/Qwen3-32B ~/qwenparity $P --side vllm
deactivate; /opt/pytorch/bin/python tests/vllm_parity.py /nvme/qwen/Qwen3-32B ~/qwenparity $P --side hf
```

The two halves run in separate environments because `HFEngine` needs the DLAMI's transformers
4.57.1 and vLLM 0.30 needs >= 5.10 (under 5.x `HFEngine` fails on `DynamicCache.to_legacy_cache`,
found on kr-q 2026-10-04). The same split is why `read` and `tf` below run in `/opt/pytorch`.

16 conditions (D0 and D1.1 x t04/t05 x {unsteered, joy, sadness, red}) x 24 items, read with
thinking **off**, through the vLLM backend and then through `HFEngine` in the same process. It
prints per-condition top-1 agreement and |dlogprob| and exits non-zero if the bar is missed. The
bar is `tests/hf_engine_check.py`'s own reference run with margin: top-1 everywhere, median
<= 0.30 nats, max <= 1.00 (HFEngine against a naive full-sequence forward was median 0.09-0.25,
max 0.38-0.63, against a bf16 self-noise floor of 0.125/0.50).

It also prints the steering hook's own report per condition (`{"n": ..., "lo": ..., "hi": ...}`):
how many positions were steered and the lowest and highest, which must sit inside the answer span.
Under `--tp 2` the two ranks must report the same `n`: a Qwen3 row-parallel projection all-reduces,
so the residual stream is replicated and the vector is added once per rank to its own replica. If
the vector were being applied twice the logprobs would be far outside the bar.

Cost: two model loads plus ~83k prefill tokens. ~15 min, **~$2.6** on g6e.12xlarge.

If it fails:
- add `--readout topk` (uses `SamplingParams(logprobs=K)` instead of `logprob_token_ids`, for a
  vLLM that lacks the latter; a NaN target logprob then means a target fell outside the top-k);
- `enforce_eager` is already on; if a Qwen3 layer's `forward` signature has moved, the hook's
  `assert positions.ndim == 1` or the `h.shape[0] == positions.shape[0]` assert fires immediately
  rather than silently steering the wrong rows.

## 2. Free thinking under carry-over steering (the main design)

```bash
python runs/qwen_thinking.py /nvme/qwen/Qwen3-32B ~/qwenthink gen \
    --vectors results/qwen/v2/vectors.pt --layer 40 --dose 2 \
    --backend vllm --tp 2 --budget 1024 2>&1 | tee ~/qwenthink/gen.log
```

Grid, as built and sized offline:

```
90 conditions x 98 items = 8820 thinking trials, prefill 419,508 tokens
  + up to 9,031,680 generated, layer 40, dose 2.0 x natural unit 48.0 (coef 96.0)
10 distinct prefixes (9 conditions share each), prefix 201-2241 tokens,
  question suffix 22-60 tokens, answer span 180 tokens
```

- arms D0 and D1.4, the 5 clean tasks `t04 t05 t06 t08 t15`, vectors
  `none joy sadness euphoria fear calm surprise red random` (raw story vectors, unit-normalized,
  layer 40, 2 natural units = coef 96.0);
- items: `self_state` (20), `third_person:self` (20), `third_person:a rock` (20),
  `identification` (18), `yes_bias:20` (the first 10 facts x 2 twins) = 98. `--items all` for the
  whole 238-item battery, at 2.4x the cost.
- **no pain**: the ethics condition forbids free generation under pain steering. `--vectors-list`
  is checked against `thinking.assert_no_free_pain` before a token is generated, and each backend
  checks it again per group.

Cost, at a median trace of 256 tokens (K2.6's unsteered traces for these items ran 410-1,938
characters, median 1,010, i.e. roughly 250 tokens, and Qwen3-0.6B produced 107-143):

| | tokens | rate assumed | time |
|---|---|---|---|
| generation prefill | 419,508 | | |
| readout prefill (the trace; prefix and question are cached) | 2,257,920 | 6,000 tok/s (2 procs x TP2) | 0.12 h |
| decode | 2,257,920 | 1,600 tok/s (2 procs x TP2) | 0.39 h |
| model load, 90 cache resets | | | 0.15 h |

**g6e.12xlarge: ~0.7 h of GPU work, budget 1.5 h = ~$16.** Run two processes, each `--tp 2`, by
splitting `--tasks`: `--tasks t04,t05,t06` on GPUs 0,1 and `--tasks t08,t15` on 2,3 with
`CUDA_VISIBLE_DEVICES`, into two output directories. One process at `--tp 2` on half the machine is
~1.4 h, ~$15 (the box is billed whole either way).

**p5.4xlarge (1x H100 80 GB, `--tp 1`): ~0.5 h of GPU work, budget 1.5 h = ~$10.** Decode is
roughly 2x faster per GPU and there is no TP all-reduce, but there is half as much silicon.

If traces run to 512 tokens the decode and readout-prefill rows double: ~1.2 h of work, budget
2.5 h = **~$26** on g6e.12xlarge. `--budget 1024` is the cap, not the expectation; the forced rate
is printed at the end of `gen` and per condition by `score`. If it is above a few percent, either
raise `--budget` or report the arm as budget-limited.

## 3. The authoritative readout, and the teacher-forced comparison

Both are prefill-only and run in the **DLAMI** environment through the existing backend, not in the
vLLM venv:

```bash
deactivate; cd $HOME/KimisRoom
python runs/qwen_thinking.py /nvme/qwen/Qwen3-32B ~/qwenthink read
python runs/qwen_thinking.py /nvme/qwen/Qwen3-32B ~/qwenthink tf --tf-out ~/qwenthink_tf
```

- `read` re-reads every trial with its own trace through `HFEngine`, which is where every published
  pilot number comes from. The vLLM readout from `gen` is kept alongside it, so the two can be
  compared on the real arm and not only on the parity set.
- `tf` replays the **unsteered** trace of the same (arm, task) under every steered prefix. The
  reasoning text is then identical across conditions, which isolates whether the injected state
  reaches the answer at all when it cannot change what the model wrote.

Each is 419,508 + 8,820 x trace tokens = **2.68 M tokens** at 256-token traces. At the measured
2,700 tok/s for the whole g6e.12xlarge (2 workers x 2 GPUs, PLAN.md section 6): 0.28 h, **~$3**
each. `--gpus-per-worker 2` is the default and is required for bf16 Qwen3-32B on L40S.

## 4. Scores

```bash
python runs/qwen_thinking.py /nvme/qwen/Qwen3-32B ~/qwenthink score
python runs/qwen_thinking.py /nvme/qwen/Qwen3-32B ~/qwenthink_tf score   # follows traces_from
```

Prints the thinking statistics (median trace length, forced rate, and the lexical tally: whether
the trace mentions feeling, introspection, any emotion, or the injected one), then runs
`runs/qwen_pilot.py score`, `runs/compression.py` and `runs/bias.py` on `grid.pt` with
`--battery <out>/battery_subset.jsonl`. The group ids are `arm|task|vector|layer|dose`, which is
what all three already parse, so the compression slope and Yes-lean fit that PLAN.md section 3
item (6) makes the primary outcome are computed by the pilot's own code with no new statistics.

## 5. Kimi K2.6, teacher-forced

No GPU step. The traces come from Foundry and the engine job is built anywhere:

```bash
python runs/k26_thinking.py ~/k26think traces --workers 8          # ~12 min, Azure credit
python runs/k26_thinking.py ~/k26think groups --tokenizer /nvme/k26 \
    --vectors results/k26/vectors.pt --layer 38 --dose 2
python -m kimisroom.run ~/k26think/k26_think.job.pt ~/k26think/grid.pt --model /nvme/k26
```

Measured on 12 real calls: `reasoning_effort: "high"`, `max_completion_tokens: 2048`,
`temperature: 0`; the reasoning comes back as `message.reasoning_content`. 1,135 prompt tokens and
241 completion tokens per call, median reasoning 1,010 characters, nothing truncated. The full
scope (2 arms x 5 tasks x 98 items = 980 calls) is therefore about **1.11 M prompt and 0.24 M
completion tokens**. Every call is cached in `~/k26think/cache/think.jsonl`, so a rerun is free.

The engine job is 70 groups (2 arms x 5 tasks x {unsteered + joy, sadness, anger, fear, red,
random}, 98 items each). Prefixes are 189 tokens under D0 and 2,007 under D1.4, answer spans 168
tokens, suffixes 114-371 tokens with real traces, so the job is
35 x (189 + 98 x ~250) + 35 x (2007 + 98 x ~250) = **~1.8 M tokens**. Layer 38 is a flag: PLAN.md section 10 reports every raw
K2.6 vector passing the AIPsy gate from layer 30 to 60 of 61 and the emotion-specific ones at
25-40, and 38 is the Qwen pilot's depth (40 of 64) carried over, inside both bands. The natural
unit there is 13.9, so 2 units is coef 27.8.

## 6. Guardrails worth knowing about

- **A steered prefix must never be reused unsteered.** Nothing in vLLM's prefix-cache hash covers
  hook state, so `VLLMThinker` calls `reset_prefix_cache()` on every change of steer and runs one
  condition at a time. Do not batch conditions together.
- **The hook must not be installed after CUDA-graph capture.** `enforce_eager=True` is the default
  for exactly this reason; turning it off would silently drop the steering.
- **The steer can only touch the prefix.** It is a `[start, end)` range inside the prefix, so a
  generated token's absolute position is never in it. Both backends assert this from the positions
  the hook actually fired on, so a positions-semantics change in vLLM fails loudly.
- **Do not run anything here on kr-q while the D2/D2c fine-tune is running.**
