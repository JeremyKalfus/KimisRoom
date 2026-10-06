#!/usr/bin/env bash
# sweep_chain.sh MODEL_DIR: one size of the Qwen3 sweep after its vectors exist, on the box or the
# desktop (one GPU): calibration grid -> the fixed selection rule (runs/sweep_summary.py) -> pilot
# grid -> pilot summary. Dev tasks only (prompts/tasks/neutral_tasks.jsonl, clean tasks); the
# confirmatory tasks are never touched. Pain stays off (no --allow-pain). Outputs in ~/sweep/<tag>/.
# Expects ~/sweep/<tag>/vec/{vectors.pt,aipsy_table.pt} and ~/sweep/assent/<tag>_d1_argument_4.json.
set -euo pipefail
M=$1; tag=$(basename "$M" | tr A-Z a-z); S=~/sweep/$tag
cd ~/KimisRoom
B=prompts/battery/items_idctl.jsonl
run() { python runs/sweep_run.py runs/qwen_pilot.py "$M" "$@"; }
n=$(python -c "import json; print(json.load(open('$M/config.json'))['num_hidden_layers'])")
cand=$(PYTHONPATH=. python -c "import sys; sys.path.insert(0,'runs'); from sweep_summary import candidate_layers as c; print(','.join(map(str, c($n))))")
echo "$tag: $n layers, candidates $cand  $(date -u +%H:%M:%S)"
if [ ! -f $S/cal/grid.pt ]; then
  run $S/cal grid --vectors $S/vec/vectors.pt --layers $cand --doses 1,2,4 --arms D0 --battery $B --vector-kinds raw --tasks 5 > $S/cal.log 2>&1
fi
echo "$tag cal done $(date -u +%H:%M:%S)"
PYTHONPATH=. python runs/sweep_summary.py cal $S/cal $S/vec/aipsy_table.pt --json $S/choice.json > $S/cal_summary.txt
tail -1 $S/cal_summary.txt
read -r L d1 d2 <<<"$(python -c "import json; c=json.load(open('$S/choice.json')); print(c['layer'], '%g' % c['secondary_dose'], '%g' % c['primary_dose'])")"
emo=$(python -c "import json; print(','.join(json.load(open('$S/choice.json'))['emotion_set']))")
cp ~/sweep/assent/${tag}_d1_argument_4.json data/assent/  # the grid reads frozen assents from data/assent/
if [ ! -f $S/pilot/grid.pt ]; then
  run $S/pilot grid --vectors $S/vec/vectors.pt --layers $L --doses $d1,$d2 --arms D0,D1.4 --battery $B --vector-kinds raw --tasks 7 --task-site > $S/pilot.log 2>&1
fi
PYTHONPATH=. python runs/sweep_summary.py pilot $S/pilot --emo $emo --layer $L --doses $d1,$d2 --json $S/pilot.json > $S/pilot_summary.txt
echo "$tag pilot done $(date -u +%H:%M:%S)"
