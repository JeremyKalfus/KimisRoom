#!/usr/bin/env bash
# Run inside WSL on the desktop (scripts/pc_launch.sh scripts/pc_chains.sh): once the desktop's vectors
# are done, the calibration -> rule -> pilot chain for 0.6B, 1.7B and 4B in turn.
set -uo pipefail
source ~/krenv/bin/activate
until grep -q "PC VECTORS DONE" ~/sweep/pc_sweep.log 2>/dev/null; do sleep 30; done
for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do bash ~/KimisRoom/scripts/sweep_chain.sh ~/models/$m; done
echo "PC CHAINS DONE $(date -u +%H:%M:%S)"
