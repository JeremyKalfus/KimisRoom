"""`python -m kimisroom.run` with HFEngine's suffix chunk set from $SWEEP_SUFFIX_CHUNK.

    SWEEP_SUFFIX_CHUNK=8 python runs/sweep_run.py runs/qwen_pilot.py ...   (sweep_run.py routes here)

On the 16 GB desktop GPU, Qwen3-4B's D1.4 pilot runs out of memory: each prefix's KV cache is expanded
over 16 suffix rows (~6 GB for a ~2,500-token denial prefix on top of 8 GB of weights). This runs the
unchanged kimisroom.run with a smaller chunk, without editing the hashed engine files. The chunk changes
the padded batch of suffix rows, so readouts can differ from chunk 16 at bf16 rounding level; a grid run
this way records the chunk in its log line below. Workers are spawned, and spawn re-imports this file as
__mp_main__, so the patch below reaches them too.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kimisroom.hf_engine as _hf  # noqa: E402

CHUNK = int(os.environ.get("SWEEP_SUFFIX_CHUNK", "16"))
_init = _hf.HFEngine.__init__


def _init_chunk(self, *a, **kw):
    kw.setdefault("suffix_chunk", CHUNK)
    _init(self, *a, **kw)


_hf.HFEngine.__init__ = _init_chunk

if __name__ == "__main__":
    print(f"sweep_hfrun: HFEngine suffix_chunk={CHUNK}", flush=True)
    from kimisroom import run

    run.main()
