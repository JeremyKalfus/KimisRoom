#!/usr/bin/env bash
# Run on the cloud box (kr-q, 1x L40S): the size sweep's NVIDIA side. Vectors for 0.6B (the gate's
# reference), the desktop gate's NVIDIA readouts (bf16 and the fp32 floor), frozen assents for all
# five sizes, then vectors for 8B and 14B. Outputs under ~/sweep/<tag>/.
#   ssh ubuntu@BOX 'nohup bash ~/KimisRoom/scripts/sweep_box.sh > ~/sweep/box.log 2>&1 &'
set -uo pipefail
cd ~/KimisRoom
source /opt/pytorch/bin/activate
export HF_XET_HIGH_PERFORMANCE=1
S=~/sweep; Q=/nvme/qwen
rev() { awk -v m="$1" '$1==m{print $2}' $S/revisions.txt; }
for m in Qwen3-1.7B Qwen3-4B; do  # the desktop's sizes too, for their assents
  [ -n "$(rev $m)" ] || echo "$m $(python -c "from huggingface_hub import model_info; print(model_info('Qwen/$m').sha)")" >> $S/revisions.txt
  hf download Qwen/$m --revision "$(rev $m)" --local-dir $Q/$m > /dev/null 2>&1
done
vec() {  # vec Qwen3-8B
  local t; t=$(echo "$1" | tr A-Z a-z)
  [ -f $S/$t/vec/vectors.pt ] && return
  mkdir -p $S/$t/vec
  python runs/sweep_run.py runs/vectors.py $Q/$1 $S/$t/vec run > $S/$t/vec/run.log 2>&1 &&
  python runs/vectors.py $Q/$1 $S/$t/vec vectors > $S/$t/vec/report.txt 2>&1
  echo "vectors $1 done $(date -u +%H:%M:%S) rc=$?"
}
vec Qwen3-0.6B
mkdir -p $S/gate
python tests/sweep_gate.py run $Q/Qwen3-0.6B $S/qwen3-0.6b/vec/vectors.pt $S/gate/nvidia_bf16.pt --dtype bf16 &&
python tests/sweep_gate.py run $Q/Qwen3-0.6B $S/qwen3-0.6b/vec/vectors.pt $S/gate/nvidia_fp32.pt --dtype fp32
echo "gate readouts done $(date -u +%H:%M:%S)"
for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B; do
  python runs/sweep_assent.py $Q/$m "$(rev $m)" $S/assent >> $S/assent.log 2>&1
done
echo "assents done $(date -u +%H:%M:%S)"
vec Qwen3-8B
vec Qwen3-14B
echo "BOX QUEUE DONE $(date -u +%H:%M:%S)"
