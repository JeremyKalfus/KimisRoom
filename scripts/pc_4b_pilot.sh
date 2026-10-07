#!/usr/bin/env bash
# Inside WSL on the desktop (scripts/pc_launch.sh scripts/pc_4b_pilot.sh): rerun Qwen3-4B's pilot, which
# ran out of GPU memory at the default suffix chunk of 16 (results/sweep/qwen3-4b/pilot.log), with a
# chunk of 8 (runs/sweep_hfrun.py). The crashed attempt is kept as pilot.oom-chunk16.
set -euo pipefail
S=~/sweep/qwen3-4b
[ -d $S/pilot ] && [ ! -f $S/pilot/grid.pt ] && mv $S/pilot $S/pilot.oom-chunk16 && mv $S/pilot.log $S/pilot.oom-chunk16.log
export SWEEP_SUFFIX_CHUNK=8
bash ~/KimisRoom/scripts/sweep_chain.sh ~/models/Qwen3-4B
