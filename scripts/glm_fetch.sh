#!/usr/bin/env bash
# glm_fetch.sh IP: copy the GLM bring-up's small outputs from the GPU box (~/glm, scripts/glm_box_run.sh)
# into results/glm/ on the Mac. Plain ssh with the project key, so it needs no AWS credentials (the
# box's security group must already allow this Mac's IP). Big tensors stay out of git: results/ is
# ignored, and only the summaries are force-added.
set -euo pipefail
cd "$(dirname "$0")/.."
IP=${1:?usage: glm_fetch.sh IP}
mkdir -p results/glm
rsync -az -e "ssh -i $HOME/.ssh/kimisroom_aws -o StrictHostKeyChecking=accept-new" \
  --include='*/' \
  --include='choice.json' --include='logs/*.log' \
  --include='parity/*.txt' --include='parity/summary.json' \
  --include='vec/report.txt' --include='vec/vectors.pt' --include='vec/aipsy_table.pt' \
  --include='cal/summary.txt' --include='cal/meta.json' \
  --include='pilot/summary.txt' --include='pilot/summary.json' --include='pilot/table.csv' --include='pilot/meta.json' \
  --include='think/*.json' --include='think/*.txt' \
  --exclude='*' --prune-empty-dirs \
  "ubuntu@$IP:glm/" results/glm/
find results/glm -type f -not -name "*.jsonl" | sort
