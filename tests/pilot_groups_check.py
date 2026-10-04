"""CPU check of the pilot's grid and power stages: real tokenizer, real battery, no model weights.

    .venv/bin/python tests/pilot_groups_check.py [--model <dir with the Qwen3-32B tokenizer>]

Runs `runs/qwen_pilot.py ... grid` and `... power` for real against the Qwen3-32B tokenizer files
(from ~/.cache/huggingface by default) with a synthetic vector file, stubbing out only the
`subprocess.run` that would launch `kimisroom.run` on a GPU. Then checks the job files:
  * group construction: one unsteered trial per (arm, task) and one steered trial per cell
  * doses are in natural units -- the coefficient equals dose x the mean raw emotion-vector norm
  * the ethics cap: no pain trial above 2 natural units exists, in either dose mode
  * every group reads first-token logprobs only (readout "suffix_last"), nothing generates
  * answer tokens: the nine identification options have distinct first tokens
  * `--dose-mode resid --doses 1,2,4,8 --emotions v1 --vector-kinds raw,specific` rebuilds exactly
    the pilot's 12-vector grid, so the old run is reproducible
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
from kimisroom.emotions import CONCEPTS, EMOTIONS, EMOTIONS_V1, vector_names  # noqa: E402

N_LAYERS, H = 64, 128
STUB = "import subprocess\nsubprocess.run = lambda *a, **k: None\n"
PILOT_VECTORS = ["joy", "sadness", "anger", "fear", "joy-specific", "sadness-specific",
                 "anger-specific", "fear-specific", "random", "red", "ocean", "arithmetic"]


def tokenizer_dir() -> str:
    if "--model" in sys.argv:
        return sys.argv[sys.argv.index("--model") + 1]
    hits = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*")))
    assert hits, "no local Qwen3-32B tokenizer; pass --model <dir>"
    return hits[-1]


def fake_vectors(tmp: Path, emotions: list[str], name: str, concepts: list[str] = CONCEPTS) -> Path:
    g = torch.Generator().manual_seed(1)
    names = vector_names(emotions, concepts)
    v = {n: torch.randn(N_LAYERS, H, generator=g) for n in names}
    for n in names:  # give the emotions a known norm so the dose arithmetic is checkable
        v[n] = v[n] / v[n].norm(dim=-1, keepdim=True) * 40.0
    path = tmp / name
    torch.save({"vectors": v, "resid_norm": torch.full((N_LAYERS,), 330.0),
                "nat_norm": torch.full((N_LAYERS,), 40.0), "emotions": emotions,
                "concepts": concepts, "kind": "story"}, path)
    return path


def run_stage(model: Path, out: Path, stage: str, extra: list[str]) -> str:
    with tempfile.TemporaryDirectory() as stub:
        Path(stub, "sitecustomize.py").write_text(STUB)
        env = {**os.environ, "PYTHONPATH": stub}
        r = subprocess.run([sys.executable, str(ROOT / "runs/qwen_pilot.py"), str(model), str(out), stage, *extra],
                           capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, f"{stage} failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
    return r.stdout


def main() -> None:
    tok_dir = tokenizer_dir()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = tmp / "Qwen3-32B"       # the stage derives the assent-file tag from this name
        model.mkdir()
        for f in Path(tok_dir).iterdir():
            (model / f.name).symlink_to(f)
        story = fake_vectors(tmp, EMOTIONS, "vectors.pt")
        asst = fake_vectors(tmp, EMOTIONS, "asst.pt", concepts=[])

        # --- grid, natural doses, eight emotions, all three vector kinds
        out = tmp / "grid"
        log = run_stage(model, out, "grid", ["--vectors", str(story), "--layers", "32,40", "--tasks", "2"])
        print(log.strip())
        job = torch.load(out / "grid.job.pt", weights_only=False)
        meta = json.loads((out / "grid_meta.json").read_text())
        items = [json.loads(x) for x in (ROOT / "prompts/battery/items.jsonl").read_text().splitlines()]
        assert meta["dose_mode"] == "natural" and meta["doses"] == [0.5, 1.0, 2.0, 4.0]
        assert meta["n_items"] == len(items) == 238, meta["n_items"]
        gs = {g.id: g for g in job["groups"]}
        assert all(g.readout == "suffix_last" for g in job["groups"]), "a group would generate"
        assert all(len(g.suffixes) == len(items) for g in job["groups"])
        n_arms, n_tasks, n_layers_used = 2, 2, 2
        pain_cells = [gid for gid in gs if gid.split("|")[2].startswith("pain")]
        assert not any(float(gid.split("|")[4]) > 2.0 for gid in pain_cells), "pain steered past the cap"
        assert "ethics cap" in log and "'pain', 4.0" in log, log
        # dose arithmetic: coefficient == dose x natural unit (40.0 here), same for every vector
        for gid, g in gs.items():
            if g.steers:
                dose = float(gid.split("|")[4])
                assert abs(g.steers[0].coef - dose * 40.0) < 1e-3, (gid, g.steers[0].coef)
        # 3 pain vectors x 2 layers are left out at dose 4 by the ethics cap
        n_cells = len(meta["vectors"]) * n_layers_used * len(meta["doses"]) - 3 * n_layers_used
        assert len(gs) == n_arms * n_tasks * (1 + n_cells), (len(gs), n_cells)
        print(f"grid: {len(gs)} groups = {n_arms} arms x {n_tasks} tasks x (1 unsteered + {n_cells} cells); "
              f"{len(meta['vectors'])} vectors {meta['vectors'][:4]}...; pain cells {len(pain_cells)}")

        # --- the nine identification options are distinct first tokens under this tokenizer
        targets = json.loads((out / "targets.json").read_text())
        opts = next(it["answer_format"] for it in items if it["kind"] == "identification")
        assert len(opts) == 9 and len({targets[w] for w in opts}) == 9, [(w, targets[w]) for w in opts]
        print(f"answer tokens ({len(targets)} words): " + ", ".join(f"{w}={targets[w]}" for w in opts))

        # --- power, one dose, with assistant vectors and the injection-location control
        out2 = tmp / "power"
        log2 = run_stage(model, out2, "power", ["--vectors", str(story), "--asst-vectors", str(asst),
                                                "--layers", "40", "--dose", "1"])
        print(log2.strip())
        job2 = torch.load(out2 / "power.job.pt", weights_only=False)
        m2 = json.loads((out2 / "power_meta.json").read_text())
        assert m2["dose_mode"] == "natural" and abs(m2["natural_unit"] - 40.0) < 1e-3
        assert all(abs(g.steers[0].coef - 40.0) < 1e-3 for g in job2["groups"] if g.steers)
        assert all(g.readout == "suffix_last" for g in job2["groups"])
        sites = {g.id.split("|")[3] for g in job2["groups"] if g.steers}
        assert sites == {"answer", "task"}, sites
        print(f"power: {len(job2['groups'])} groups, {len(job2['vectors'])} vectors, sites {sorted(sites)}, "
              f"coef {m2['dose'] * m2['scale']:.1f} = {m2['dose']} x natural unit {m2['natural_unit']:.1f}")

        # --- pain at a dose above the cap is dropped from the power run, not crashed on
        out3 = tmp / "power4"
        log3 = run_stage(model, out3, "power", ["--vectors", str(story), "--asst-vectors", str(asst),
                                                "--layers", "40", "--dose", "4"])
        assert "ethics cap" in log3 and "pain" in log3, log3
        job3 = torch.load(out3 / "power.job.pt", weights_only=False)
        assert not [g for g in job3["groups"] if g.steers and "pain" in g.id.split("|")[2]]
        print("ethics cap at dose 4: " + [ln for ln in log3.splitlines() if "ethics cap" in ln][0])

        # --- resid mode also respects the cap, and the pilot's grid rebuilds exactly
        out4 = tmp / "gridv1"
        v1 = fake_vectors(tmp, EMOTIONS_V1, "v1.pt")
        log4 = run_stage(model, out4, "grid", ["--vectors", str(v1), "--layers", "32,40", "--tasks", "5",
                                               "--dose-mode", "resid", "--emotions", "v1",
                                               "--vector-kinds", "raw,specific",
                                               "--battery", "prompts/battery/items_v1.jsonl"])
        m4 = json.loads((out4 / "grid_meta.json").read_text())
        assert m4["vectors"] == PILOT_VECTORS, m4["vectors"]
        assert m4["doses"] == [1.0, 2.0, 4.0, 8.0] and m4["scale"]["40"] == 330.0
        assert m4["n_items"] == 240
        job4 = torch.load(out4 / "grid.job.pt", weights_only=False)
        assert len(job4["groups"]) == 2 * 5 * (1 + 12 * 2 * 4) == 970, len(job4["groups"])
        print(f"pilot grid rebuilt: {len(job4['groups'])} trials (the pilot ran 970), vectors == the "
              f"hardcoded list, resid unit 330, battery items_v1.jsonl (240 items)")
        assert "ethics cap" not in log4  # no pain vector in the v1 set
        print("\nOK: group construction, natural doses, pain cap, option tokens and v1 reproduction")


if __name__ == "__main__":
    main()
