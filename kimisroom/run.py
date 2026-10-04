"""Run a job file through the right backend for the checkpoint, on every local GPU.

    python -m kimisroom.run JOB.pt OUT.pt --model /nvme/k26 [--gpus 4] [--max-tokens 800000]
    python -m kimisroom.run JOB.pt OUT.pt --model /nvme/qwen/Qwen3-32B --gpus-per-worker 2

JOB.pt is torch.save({"groups": [Group, ...], "vectors": {name: tensor}, "targets": [token ids],
"top_k": 20}). Groups are spread over the workers by token count, then cut into passes of at
most --max-tokens. Each finished pass is saved under OUT.parts/ and skipped on a rerun, so an
interrupted job resumes where it stopped. OUT.pt maps group id -> {"collects": [...],
"readouts": [Readout, ...]} plus a "_meta" entry. Both backends give the same output format
and the same numbers for a group whatever else is in the job.

The backend comes from the checkpoint's `model_type`:
- DeepseekV3 / Kimi K2.6: `engine.Engine`, one worker per GPU, every pass streaming the whole
  model once. With several GPUs the main process reads each layer once into shared memory and
  hands it to every worker that needs it, so the disk and the CPU-side unpacking are not
  repeated per GPU.
- Qwen3: `hf_engine.HFEngine`, one resident model per worker and no loader. A worker owns
  --gpus-per-worker GPUs (2 for Qwen3-32B in bf16, which is ~64 GB and does not fit one
  48 GB L40S), so --gpus 4 --gpus-per-worker 2 runs two workers.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.multiprocessing as mp

from .engine import Checkpoint, Engine, Group

# model_type -> backend. The layer-wise engine re-implements the DeepseekV3 code path;
# everything else the pilot needs is a dense HF model.
BACKEND = {"deepseek_v3": "engine", "kimi_k2": "engine", "kimi_k25": "engine", "qwen3": "hf"}


def backend_of(model: str) -> str:
    """Which backend a local checkpoint needs, from the top-level model_type in its config."""
    mt = json.loads((Path(model) / "config.json").read_text()).get("model_type", "")
    assert mt in BACKEND, f"model_type {mt!r} has no backend; add it to kimisroom.run.BACKEND"
    return BACKEND[mt]


def plan(groups: list[Group], gpus: int, max_tokens: int) -> list[list[list[Group]]]:
    """Per GPU, a list of passes. Largest groups first onto the least-loaded GPU, then each
    GPU's groups filled into passes in order."""
    load = [0] * gpus
    per_gpu: list[list[Group]] = [[] for _ in range(gpus)]
    for g in sorted(groups, key=lambda g: -g.n_tokens):
        assert g.n_tokens <= max_tokens, f"group {g.id} has {g.n_tokens} tokens > --max-tokens"
        r = load.index(min(load))
        per_gpu[r].append(g)
        load[r] += g.n_tokens
    out = []
    for gs in per_gpu:
        passes, cur, n = [], [], 0
        for g in gs:
            if cur and n + g.n_tokens > max_tokens:
                passes.append(cur)
                cur, n = [], 0
            cur.append(g)
            n += g.n_tokens
        if cur:
            passes.append(cur)
        out.append(passes)
    return out


def worker(rank: int, model: str, job: dict, passes: list[list[Group]], todo: list[int], parts: Path,
           layers: mp.Queue | None = None) -> None:
    """Runs passes `todo` on GPU `rank`. With `layers`, takes (pass, layer, tensors) from the
    loader in order instead of reading the checkpoint itself."""
    torch.cuda.set_device(rank)
    eng = Engine(Checkpoint(model), device=f"cuda:{rank}")
    for p in todo:
        groups = passes[p]
        path = parts / f"gpu{rank}_pass{p}.pt"
        n_tok = sum(g.n_tokens for g in groups)
        t0 = time.time()

        def progress(i: int) -> None:
            if i % 10 == 9 or i == eng.c.num_hidden_layers - 1:
                dt = time.time() - t0
                split = ", ".join(f"{k} {v:.0f}s" for k, v in eng.timings.items())
                print(f"[gpu{rank} pass {p + 1}/{len(passes)}] layer {i + 1}: {dt:.0f}s, "
                      f"{n_tok * (i + 1) / eng.c.num_hidden_layers / dt:,.0f} tok/s ({split})", flush=True)

        source = None
        if layers is not None:

            def source(i: int, p: int = p) -> dict[str, torch.Tensor]:
                q, li, cpu = layers.get()
                assert (q, li) == (p, i), f"gpu{rank} expected pass {p} layer {i}, got pass {q} layer {li}"
                return cpu

        out = eng.run(groups, job.get("vectors"), job.get("targets"), job.get("top_k", 20), progress, source)
        tmp = path.with_suffix(".tmp")
        torch.save(out, tmp)
        os.replace(tmp, path)
        print(f"[gpu{rank} pass {p + 1}/{len(passes)}] done: {len(groups)} groups, {n_tok:,} tokens, "
              f"{time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)


def worker_hf(rank: int, model: str, job: dict, passes: list[list[Group]], todo: list[int], parts: Path,
              gpus: list[int], isolate: bool) -> None:
    """Runs passes `todo` with one resident HF model spanning the GPUs `gpus`. With `isolate`
    the process hides every other GPU first, so accelerate's device map covers exactly these."""
    if isolate:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus)
    from .hf_engine import HFEngine

    dev = "auto" if len(gpus) > 1 else f"cuda:{0 if isolate else gpus[0]}"
    eng = HFEngine(model, device=dev)
    for p in todo:
        groups = passes[p]
        path = parts / f"gpu{rank}_pass{p}.pt"
        n_tok = sum(g.n_tokens for g in groups)
        t0 = time.time()

        def progress(i: int, groups=groups, n_tok=n_tok, p=p) -> None:
            if i % 10 == 9 or i == len(groups) - 1:
                dt = time.time() - t0
                done = sum(g.n_tokens for g in groups[: i + 1])
                split = ", ".join(f"{k} {v:.0f}s" for k, v in eng.timings.items())
                print(f"[w{rank} pass {p + 1}/{len(passes)}] group {i + 1}/{len(groups)}: {dt:.0f}s, "
                      f"{done / dt:,.0f} tok/s ({split})", flush=True)

        out = eng.run(groups, job.get("vectors"), job.get("targets"), job.get("top_k", 20), progress)
        tmp = path.with_suffix(".tmp")
        torch.save(out, tmp)
        os.replace(tmp, path)
        peak = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9
        print(f"[w{rank} pass {p + 1}/{len(passes)}] done: {len(groups)} groups, {n_tok:,} tokens, "
              f"{time.time() - t0:.0f}s, peak {peak:.1f} GB", flush=True)


def feed(model: str, todo: list[list[int]], queues: list, procs: list) -> None:
    """Reads every layer once per pass into shared memory and queues it for each worker that
    runs that pass. A queue holds at most two layers, so the loader stays just ahead."""
    ckpt = Checkpoint(model)
    n = ckpt.config.num_hidden_layers
    jobs = [(p, i) for p in sorted({p for t in todo for p in t}) for i in range(n)]
    pool = ThreadPoolExecutor(2)  # reading and unpacking the next layer overlaps this one
    ahead = [pool.submit(ckpt.load_layer_cpu, i, True) for _, i in jobs[:2]]
    for k, (p, i) in enumerate(jobs):
        cpu = ahead.pop(0).result()
        if k + 2 < len(jobs):
            ahead.append(pool.submit(ckpt.load_layer_cpu, jobs[k + 2][1], True))
        for r in (r for r, t in enumerate(todo) if p in t):
            while True:
                try:
                    queues[r].put((p, i, cpu), timeout=10)
                    break
                except queue.Full:
                    if not procs[r].is_alive():
                        raise RuntimeError(f"worker {r} exited with {procs[r].exitcode}") from None
        del cpu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--gpus-per-worker", type=int, default=1,
                    help="HF backend only: GPUs one resident model spans (2 for Qwen3-32B in bf16)")
    ap.add_argument("--max-tokens", type=int, default=800_000)
    a = ap.parse_args()
    job = torch.load(a.job, weights_only=False)
    groups: list[Group] = job["groups"]
    assert len({g.id for g in groups}) == len(groups), "group ids must be unique"
    kind = backend_of(a.model)
    if kind == "engine":
        assert a.gpus_per_worker == 1, "--gpus-per-worker is for the HF backend; the engine streams layers"
        Checkpoint(a.model)  # fills the remote-code cache once, before workers race on it
    assert a.gpus % a.gpus_per_worker == 0, f"{a.gpus} GPUs is not a whole number of workers"
    n_workers = a.gpus // a.gpus_per_worker
    gpu_sets = [list(range(r * a.gpus_per_worker, (r + 1) * a.gpus_per_worker)) for r in range(n_workers)]
    passes = plan(groups, n_workers, a.max_tokens)
    parts = Path(a.out + ".parts")
    parts.mkdir(exist_ok=True)
    todo = [[p for p in range(len(ps)) if not (parts / f"gpu{r}_pass{p}.pt").exists()] for r, ps in enumerate(passes)]
    n_tok = sum(g.n_tokens for g in groups)
    print(f"{len(groups)} groups, {n_tok:,} tokens, {kind} backend on {n_workers} worker(s) x "
          f"{a.gpus_per_worker} GPU(s), passes per worker {[len(p) for p in passes]}", flush=True)
    t0 = time.time()
    if kind == "hf":
        if n_workers == 1:
            worker_hf(0, a.model, job, passes[0], todo[0], parts, gpu_sets[0], isolate=False)
        else:
            ctx = mp.get_context("spawn")
            procs = [ctx.Process(target=worker_hf,
                                 args=(r, a.model, job, passes[r], todo[r], parts, gpu_sets[r], True))
                     for r in range(n_workers)]
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join()
            bad = [r for r, pr in enumerate(procs) if pr.exitcode != 0]
            assert not bad, f"workers {bad} failed"
    elif a.gpus == 1:
        worker(0, a.model, job, passes[0], todo[0], parts)
    else:
        ctx = mp.get_context("spawn")
        queues = [ctx.Queue(maxsize=2) for _ in range(a.gpus)]
        procs = [ctx.Process(target=worker, args=(r, a.model, job, passes[r], todo[r], parts, queues[r]))
                 for r in range(a.gpus)]
        for pr in procs:
            pr.start()
        try:
            feed(a.model, todo, queues, procs)
        except BaseException:
            for pr in procs:
                pr.terminate()
            raise
        for pr in procs:
            pr.join()
        bad = [r for r, pr in enumerate(procs) if pr.exitcode != 0]
        assert not bad, f"workers {bad} failed"
    results: dict = {}
    for f in sorted(parts.glob("gpu*_pass*.pt")):
        results.update(torch.load(f, weights_only=False))
    missing = {g.id for g in groups} - results.keys()
    assert not missing, f"{len(missing)} groups missing from {parts}"
    results["_meta"] = {"model": a.model, "backend": kind, "gpus": a.gpus,
                        "gpus_per_worker": a.gpus_per_worker, "max_tokens": a.max_tokens,
                        "tokens": n_tok, "seconds": time.time() - t0}
    tmp = a.out + ".tmp"  # atomic, so a watcher never reads a half-written file
    torch.save(results, tmp)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(groups)} groups, {n_tok:,} tokens in {time.time() - t0:.0f}s "
          f"({n_tok / (time.time() - t0):,.0f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
