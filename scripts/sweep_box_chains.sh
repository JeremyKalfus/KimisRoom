#!/usr/bin/env bash
# Run on the box after scripts/sweep_box.sh: the calibration -> rule -> pilot chain for 8B and 14B.
set -uo pipefail
source /opt/pytorch/bin/activate
until grep -q "BOX QUEUE DONE" ~/sweep/box.log 2>/dev/null; do sleep 30; done
for m in Qwen3-8B Qwen3-14B; do bash ~/KimisRoom/scripts/sweep_chain.sh /nvme/qwen/$m; done
echo "BOX CHAINS DONE $(date -u +%H:%M:%S)"
