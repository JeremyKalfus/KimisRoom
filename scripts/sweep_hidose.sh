#!/usr/bin/env bash
# sweep_hidose.sh MODEL_DIR: exploratory, after the chain. D0 only, the rule's layer, raw vectors at
# 8 natural units (no pain), the 5 calibration tasks: does a size whose self-report does not move at
# 1-4 units move at all at a larger dose? (K2.6's layer search went to 8 and 16 units; PLAN.md
# section 3 ethics statement.) Output ~/sweep/<tag>/hidose, summary hidose_summary.txt.
set -euo pipefail
M=$1; tag=$(basename "$M" | tr A-Z a-z); S=~/sweep/$tag
cd ~/KimisRoom
L=$(python -c "import json; print(json.load(open('$S/choice.json'))['layer'])")
emo=$(python -c "import json; print(','.join(json.load(open('$S/choice.json'))['emotion_set']))")
[ -f $S/hidose/grid.pt ] || python runs/sweep_run.py runs/qwen_pilot.py "$M" $S/hidose grid --vectors $S/vec/vectors.pt \
  --layers $L --doses 8 --arms D0 --battery prompts/battery/items_idctl.jsonl --vector-kinds raw --tasks 5 > $S/hidose.log 2>&1
PYTHONPATH=. python runs/sweep_summary.py pilot $S/hidose --emo $emo --layer $L --doses 8 > $S/hidose_summary.txt
echo "$tag hidose done $(date -u +%H:%M:%S)"
