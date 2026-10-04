#!/usr/bin/env bash
# Run on the instance (grab.sh starts it): Qwen3-32B snapshot, then story vectors for the 8 emotions
# of data/stimuli/k26-v2 (raw, specific, denoised) with the AIPsy-Affect and Pain Axis reports.
set -euo pipefail
cd ~/KimisRoom
bash scripts/vm_setup.sh qwen
source /opt/pytorch/bin/activate
mkdir -p ~/qwenvec2
python runs/vectors.py /nvme/qwen/Qwen3-32B ~/qwenvec2 run > ~/qwenvec2/run.log 2>&1
python runs/vectors.py /nvme/qwen/Qwen3-32B ~/qwenvec2 vectors > ~/qwenvec2/report.txt 2>&1
echo VECTORS DONE >> ~/qwenvec2/report.txt
