#!/usr/bin/env bash
# Run inside WSL on the desktop (scripts/pc_launch.sh scripts/pc_sweep.sh): vectors for the sizes the
# desktop serves, 0.6B first (it is compared with the NVIDIA extraction for the gate), then 1.7B, 4B.
set -uo pipefail
source ~/krenv/bin/activate
cd ~/KimisRoom
for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  t=$(echo $m | tr A-Z a-z); o=~/sweep/$t/vec
  [ -f $o/vectors.pt ] && continue
  mkdir -p $o
  python runs/sweep_run.py runs/vectors.py ~/models/$m $o run > $o/run.log 2>&1 &&
  python runs/vectors.py ~/models/$m $o vectors > $o/report.txt 2>&1
  echo "vectors $m rc=$? $(date -u +%H:%M:%S)"
done
echo "PC VECTORS DONE"
