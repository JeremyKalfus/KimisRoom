#!/usr/bin/env bash
# Run on the instance once the Qwen D1.4 assent is judged clean: D2 and D2c side by side,
# two GPUs each (runs/distill.py), then the merged checkpoints' first-token eval against the base model (the --judge pass needs
# Foundry, which only the Mac reaches).
set -euo pipefail
cd ~/KimisRoom
source /opt/pytorch/bin/activate
M=/nvme/qwen/Qwen3-32B
arm() {  # arm NAME GPUS
  export CUDA_VISIBLE_DEVICES=$2
  for s in sample train merge eval; do
    extra=(); [ $s = sample ] && extra=(--batch 16)
    python runs/distill.py $s --model $M --out /nvme/distill --arm $1 "${extra[@]}" >> ~/distill_$1.log 2>&1
    echo "$1 $s done $(date -u +%H:%M)" >> ~/distill_$1.log
  done
}
arm D2 0,1 & arm D2c 2,3 & wait
echo DISTILL DONE >> ~/distill_D2.log
