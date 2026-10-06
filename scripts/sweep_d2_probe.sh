#!/usr/bin/env bash
# Run on the box after the sweep chains: timing probes for D2/D2c (runs/distill.py) per size, to
# extrapolate the full runs (1,500 teacher samples at batch 16, 150 QLoRA steps) without doing them.
# Per size: sample 64 prompts (4 batches of 16, up to 400 new tokens each) and 10 training steps on them.
set -uo pipefail
source /opt/pytorch/bin/activate
cd ~/KimisRoom
until grep -q "BOX CHAINS DONE" ~/sweep/box_chains.log 2>/dev/null; do sleep 30; done
for m in Qwen3-8B Qwen3-14B; do bash scripts/sweep_hidose.sh /nvme/qwen/$m >> ~/sweep/box_hidose.log 2>&1; done
cp ~/sweep/assent/*.json data/assent/
P=~/sweep/d2probe; mkdir -p $P
for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B; do
  t=$(echo $m | tr A-Z a-z); o=$P/$t
  for arm in D2 D2c; do
    s0=$(date +%s)
    python runs/distill.py sample --model /nvme/qwen/$m --out $o --arm $arm --batch 16 --limit 64 > $o.$arm.sample.log 2>&1; rc1=$?
    s1=$(date +%s)
    python runs/distill.py train --model /nvme/qwen/$m --out $o --arm $arm --max-steps 10 > $o.$arm.train.log 2>&1; rc2=$?
    s2=$(date +%s)
    echo "$m $arm sample64 $((s1 - s0))s rc=$rc1 train10 $((s2 - s1))s rc=$rc2" | tee -a $P/times.txt
  done
done
echo "D2 PROBE DONE $(date -u +%H:%M:%S)" >> $P/times.txt
