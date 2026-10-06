#!/usr/bin/env bash
# Desktop, after scripts/pc_chains.sh: the exploratory 8-unit probe for 0.6B, 1.7B, 4B.
set -uo pipefail
source ~/krenv/bin/activate
until grep -q "PC CHAINS DONE" ~/sweep/pc_chains.log 2>/dev/null; do sleep 30; done
for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do bash ~/KimisRoom/scripts/sweep_hidose.sh ~/models/$m; done
echo "PC HIDOSE DONE"
