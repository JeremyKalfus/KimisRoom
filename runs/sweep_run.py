"""Run a frozen pipeline script (runs/vectors.py, runs/qwen_pilot.py) unchanged on a one-GPU machine.

    python runs/sweep_run.py runs/vectors.py /nvme/qwen/Qwen3-8B ~/sweep/qwen3-8b/vec run
    python runs/sweep_run.py runs/qwen_pilot.py ~/models/Qwen3-4B ~/sweep/qwen3-4b/cal grid --vectors ...

Both scripts hand `kimisroom.run` a hard-coded `--gpus-per-worker 2` for any qwen3 checkpoint,
because Qwen3-32B in bf16 needs two L40S. Every model of the size sweep (0.6B-14B) fits one GPU,
and on a one-GPU machine `kimisroom.run` rejects 2 (1 GPU is not a whole number of 2-GPU workers).
This wrapper rewrites that argument to `--gpus-per-worker 1` (or $SWEEP_GPW) in the subprocess call
(and, with $SWEEP_SUFFIX_CHUNK set, runs kimisroom.run through runs/sweep_hfrun.py)
and then runs the script as __main__, so the trial code, ids and outputs are exactly the frozen
code's (runs/qwen_pilot.py is hashed in PREREG.md section 7 and is deliberately not edited).
"""

import os
import runpy
import subprocess
import sys
from pathlib import Path

GPW = os.environ.get("SWEEP_GPW", "1")
_run = subprocess.run


def run(cmd, *a, **kw):
    cmd = [str(c) for c in cmd]
    if "--gpus-per-worker" in cmd:
        i = cmd.index("--gpus-per-worker")
        cmd[i + 1] = GPW
    if os.environ.get("SWEEP_SUFFIX_CHUNK") and cmd[1:3] == ["-m", "kimisroom.run"]:
        cmd[1:3] = [str(Path(__file__).with_name("sweep_hfrun.py"))]
    return _run(cmd, *a, **kw)


subprocess.run = run
script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
sys.path.insert(0, str(Path(script).resolve().parents[1]))
runpy.run_path(script, run_name="__main__")
