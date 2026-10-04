#!/usr/bin/env bash
# Run on a fresh g6e.12xlarge: fetch K2.6 to the instance store, then the whole-model smoke
# test (perplexity, chat, 4-GPU throughput). Results in ~/smoke/ (EBS, survives a stop).
set -euo pipefail
cd ~/KimisRoom
t0=$(date +%s)
bash scripts/vm_setup.sh k26
echo "download: $(( $(date +%s) - t0 ))s, $(du -sh /nvme/k26 | cut -f1)"
source /opt/pytorch/bin/activate
python tests/smoke_model.py /nvme/k26 ~/smoke/k26 --load-groups 280
