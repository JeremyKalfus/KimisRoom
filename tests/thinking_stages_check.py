"""The thinking arm's stages wired end to end, on a Mac: gen -> read -> tf -> score.

    .venv/bin/python tests/thinking_stages_check.py [--model <Qwen3-0.6B snapshot>] [--device mps|cpu]

`runs/qwen_thinking.py` is run for real -- real tokenizer, real frozen assents, real battery, real
generation on Qwen3-0.6B -- with only the `subprocess.run` that would launch `kimisroom.run` on a
GPU stubbed out (`tests/pilot_groups_check.py` does the same). The model is symlinked into a
directory called `Qwen3-32B` because the stage derives the frozen-assent tag from the model's
directory name, and Qwen3-0.6B shares Qwen3-32B's tokenizer.

Then the job files are checked:
  * `gen` writes one trace per (condition, item), each closed and ending at the readout position,
    plus `grid_meta.json`, `targets.json` and the item subset the scoring scripts are pointed at;
  * `read` builds a prefill job whose suffixes are each trial's *own* trace;
  * `tf` builds a prefill job whose suffixes are the *unsteered* trace of the same (arm, task)
    under every steered prefix -- the teacher-forced comparison -- with the steers unchanged;
  * group ids are `arm|task|vector|layer|dose`, which is what `runs/qwen_pilot.py score`,
    `runs/compression.py` and `runs/bias.py` parse;
  * the steering coefficient is the dose times the natural unit of the real vector file;
  * `score` prints the thinking statistics without a readout file.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom.thinking import QWEN3  # noqa: E402

STUB = "import subprocess\n_r = subprocess.run\nsubprocess.run = lambda *a, **k: None\n"
VEC = ROOT / "results/qwen/v2/vectors.pt"
ARGS = ["--arms", "D0,D1.4", "--tasks", "t04", "--vectors-list", "joy,red",
        "--items", "self_state:2,yes_bias:2", "--budget", "32", "--layer", "20", "--dose", "2"]
LAYER = 20  # Qwen3-0.6B has 28 layers; the real grid uses 40 of 64


def model_dir() -> str:
    if "--model" in sys.argv:
        return sys.argv[sys.argv.index("--model") + 1]
    hits = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*")))
    assert hits, "no local Qwen3-0.6B; pass --model <dir>"
    return hits[-1]


def fake_vectors(tmp: Path) -> tuple[Path, float]:
    """A vector file shaped for Qwen3-0.6B (1024 hidden) but carrying the *real* Qwen3-32B
    `nat_norm`, so the dose arithmetic the stage does is the arithmetic the real grid will do."""
    real = torch.load(VEC, weights_only=False)
    g = torch.Generator().manual_seed(3)
    names = ["joy", "red"]
    v = {n: torch.randn(64, 1024, generator=g) for n in names}
    for n in names:
        v[n] = v[n] / v[n].norm(dim=-1, keepdim=True) * real["nat_norm"][:, None]
    path = tmp / "vectors.pt"
    torch.save({"vectors": v, "nat_norm": real["nat_norm"], "resid_norm": real["resid_norm"],
                "emotions": real["emotions"], "concepts": real["concepts"], "kind": "story"}, path)
    return path, float(real["nat_norm"][LAYER]), float(real["nat_norm"][40])


def run_stage(out: Path, model: Path, stage: str, extra: list[str]) -> str:
    with tempfile.TemporaryDirectory() as stub:
        Path(stub, "sitecustomize.py").write_text(STUB)
        env = {**os.environ, "PYTHONPATH": stub}
        r = subprocess.run([sys.executable, str(ROOT / "runs/qwen_thinking.py"), str(model), str(out), stage, *extra],
                           capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, f"{stage} failed:\n{r.stdout[-3000:]}\n{r.stderr[-4000:]}"
    return r.stdout


def main() -> None:
    assert VEC.exists(), f"needs the real vector file at {VEC}"
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "mps"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = tmp / "Qwen3-32B"  # the stage derives the frozen-assent tag from this name
        model.mkdir()
        for f in Path(model_dir()).iterdir():
            (model / f.name).symlink_to(f)
        out, tf = tmp / "think", tmp / "think_tf"
        vecs, nat, nat40 = fake_vectors(tmp)
        args = [*ARGS, "--vectors", str(vecs)]

        log = run_stage(out, model, "gen", [*args, "--backend", "hf", "--device", dev])
        print(log.strip().splitlines()[0])
        print(log.strip().splitlines()[-1])
        saved = torch.load(out / "think_traces.pt", weights_only=False)
        meta = json.loads((out / "grid_meta.json").read_text())
        items = [json.loads(line) for line in (out / "battery_subset.jsonl").read_text().splitlines()]
        targets = json.loads((out / "targets.json").read_text())

        # conditions and ids
        gids = sorted(saved["results"])
        assert len(gids) == 2 * 1 * (1 + 2), gids  # 2 arms x 1 task x (none + 2 vectors)
        for gid in gids:
            arm, task, vec, layer, dose = gid.split("|")
            assert arm in ("D0", "D1.4") and task == "t04" and layer == str(LAYER)
            assert (vec, dose) == ("none", "0") or (vec in ("joy", "red") and dose == "2.0"), gid
        assert len(items) == 4 and meta["n_items"] == 4 and meta["battery"].endswith("battery_subset.jsonl")
        assert meta["thinking"] is True and meta["style"] == QWEN3.name and meta["dose_mode"] == "natural"

        # the coefficient is the dose times the real vector file's natural unit at that layer,
        # and the real file's layer-40 unit is the ~48 PLAN.md section 3 item (6) reports
        assert abs(meta["coef"] - 2.0 * nat) < 1e-4, (meta["coef"], nat)
        assert abs(nat40 - 48.0) < 1.0, f"the pilot reports a natural unit of about 48 at layer 40, got {nat40}"
        for g in saved["groups"]:
            if g.steers:
                s = g.steers[0]
                assert abs(s.coef - 2.0 * nat) < 1e-4 and s.suffix is None and s.layer == LAYER
                assert 0 <= s.prefix[0] < s.prefix[1] <= len(g.prefix)
            assert g.readout == "suffix_last"

        # traces: one per item, closed, ending at the readout position
        for gid, rec in saved["results"].items():
            assert len(rec["traces"]) == len(items) == len(rec["readouts"]) == len(rec["suffixes"])
            for tr, it, suf in zip(rec["traces"], items, rec["suffixes"]):
                assert tr.item == it["id"]
                assert tr.text.startswith("<think>") and tr.text.endswith("</think>\n\n"), tr.text[-20:]
                assert list(suf)[-len(tr.tokens):] == list(tr.tokens)
                assert set(tr.stats) == {"chars", "words", "feeling", "introspection", "named", "own"}
                assert tr.stats["own"] is (None if gid.split("|")[2] in ("none", "red") else
                                          tr.stats["own"])
            assert len(rec["readouts"][0].target_logprobs) == len(targets)

        # read: each trial's own trace
        run_stage(out, model, "read", [])
        job = torch.load(out / "read.job.pt", weights_only=False)
        by_id = {g.id: g for g in job["groups"]}
        assert sorted(by_id) == gids
        for gid, g in by_id.items():
            assert [list(s) for s in g.suffixes] == [list(s) for s in saved["results"][gid]["suffixes"]]

        # tf: the unsteered trace of the same (arm, task) under every steered prefix
        run_stage(out, model, "tf", ["--tf-out", str(tf)])
        job = torch.load(tf / "tf.job.pt", weights_only=False)
        tf_by_id = {g.id: g for g in job["groups"]}
        assert sorted(tf_by_id) == gids
        for gid, g in tf_by_id.items():
            arm, task, *_ = gid.split("|")
            want = saved["results"][f"{arm}|{task}|none|{LAYER}|0"]["suffixes"]
            assert [list(s) for s in g.suffixes] == [list(s) for s in want], f"{gid}: not the unsteered trace"
            assert g.steers == by_id[gid].steers, f"{gid}: the steer changed"
        assert json.loads((tf / "grid_meta.json").read_text())["variant"] == "teacher_forced"
        assert (tf / "battery_subset.jsonl").read_text() == (out / "battery_subset.jsonl").read_text()
        n_tf = sum(g.n_tokens for g in tf_by_id.values())
        assert n_tf > 0

        # score: the thinking statistics, with no readout file yet
        log = run_stage(out, model, "score", [])
        assert "thinking statistics" in log and "own emotion" in log
        assert "skipped" in log, "score should say the readout stage has not run"
        print("\n".join(log.strip().splitlines()[:8]))
        assert (out / "think_stats.txt").exists()

    print("thinking stages: ok")


if __name__ == "__main__":
    main()
