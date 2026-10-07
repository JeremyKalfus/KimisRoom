#!/usr/bin/env bash
# glm_box_run.sh STAGE...: GLM-5.3 bring-up on the GPU box, one stage after another; a stage whose
# marker ~/glm/.done-STAGE exists is skipped, so a rerun resumes. Outputs in ~/glm (outside
# ~/KimisRoom, which scripts/sync.sh mirrors with --delete), logs in ~/glm/logs/STAGE.log.
# Dev tasks only: runs/qwen_pilot.py refuses confirmatory tasks for GLM until the amendment's
# timestamp exists (kimisroom/glm.py, assert_not_confirmatory). Pain is never steered (no --allow-pain).
#
#   check    GLM rendering of every pilot trial; CUDA topk's order among tied scores
#   gate1    tests/glm_parity.py ref --layers 8 (layers 0-7 against transformers), must pass
#   gate2    all 78 layers: ref (GPUs 1-3) and engine (GPU 0) side by side, then the 4-GPU runner
#   vectors  runs/vectors.py run + vectors: story vectors, AIPsy-Affect table, natural units
#   cal      calibration grid (D0, layers 26/37/49/58, 1/2/4 natural units), then the rule -> choice.json
#   pilot    the pilot grid at the chosen layer and doses: D0 and D1.4, injection-location trials
#   think    runs/glm_think.py engine: answer-format mass and the close-think probability, Low and Max
set -euo pipefail
cd ~/KimisRoom
PY=~/glmenv/bin/python; M=/nvme/glm53; O=~/glm
export PYTHONPATH=$HOME/KimisRoom PYTHONUNBUFFERED=1
mkdir -p $O/logs
NGPU=$(nvidia-smi -L | wc -l)
BATTERY=prompts/battery/items_idctl.jsonl
# The references' devices: bf16 ref, fp32 ref, isolated engine. An fp32 MoE layer needs ~36 GiB, more
# than a 24 GB L4 holds, so there GLM_REF_DEVICES=cuda:1,cpu,cuda:3,cuda:2 runs the fp32 reference on the CPU and
# the bf16 reference's experts on GPU 2.
REFDEV=${GLM_REF_DEVICES:-cuda:$((1 % NGPU)),cuda:$((2 % NGPU)),cuda:$((3 % NGPU))}
# KR_MAX_TOKENS (e.g. 300000 on 24 GB L4s): kimisroom.run's per-pass token budget, set through
# runs/sweep_run.py, which runs the frozen scripts (runs/vectors.py, runs/qwen_pilot.py) unchanged. Unset
# keeps the 800k default.
W=""; [ -n "${KR_MAX_TOKENS:-}" ] && W=runs/sweep_run.py
PILOT="$PY $W runs/qwen_pilot.py"

stage() {  # stage NAME: run "do_NAME" unless done, logging
  local s=$1
  [ -e $O/.done-$s ] && { echo "$(date +%T) $s: done before, skipped"; return 0; }
  echo "$(date +%T) $s: start"
  local t0=$(date +%s)
  # Not "if do_$s": inside an if condition bash ignores set -e for the whole function, so a failed
  # command mid-stage (gate 1's reference, out of memory) went unnoticed. A plain subshell keeps -e.
  set +e
  ( set -e; "do_$s" ) > $O/logs/$s.log 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    touch $O/.done-$s
    echo "$(date +%T) $s: ok in $(( $(date +%s) - t0 ))s"
  else
    echo "$(date +%T) $s: FAILED after $(( $(date +%s) - t0 ))s; tail of $O/logs/$s.log:"; tail -25 $O/logs/$s.log
    exit 1
  fi
}

do_check() {
  $PY tests/glm_trials_check.py $M
  $PY - <<'EOF'
import torch
# CUDA topk among tied scores, the case DSA's ReLU makes common (exact-zero index scores): does it
# take the earliest tied keys, as the engine does (kimisroom/glm.py, GlmEngine._select)?
g = torch.Generator(device="cuda").manual_seed(0)
for T, k in ((5000, 2048), (2600, 2048), (300, 100)):
    pos = torch.rand(8, T, device="cuda", generator=g) < 0.25
    x = torch.zeros(8, T, device="cuda")
    x[pos] = torch.rand(int(pos.sum()), device="cuda", generator=g) + 1
    idx = x.topk(k, dim=-1).indices
    verdict = []
    for r in range(8):
        zeros = (~pos[r]).nonzero().flatten().tolist()
        need = k - int(pos[r].sum())
        took = sorted(set(idx[r].tolist()) & set(zeros))
        verdict.append("earliest" if took == zeros[:need] else "latest" if took == zeros[-need:] else "other")
    print(f"T={T} k={k}: tied zero-score keys taken: {verdict}")
EOF
}

do_gate1() {
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # the bf16 ref fills a 24 GB L4
  $PY tests/glm_parity.py ref $M $O/parity --layers 8 --ref-devices $REFDEV
  $PY tests/glm_parity.py report $M $O/parity --layers 8 | tee $O/parity/gate1.txt
  grep -q "verdict: pass" $O/parity/gate1.txt
}

do_gate2() {
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # the bf16 ref fills a 24 GB L4
  local refdev=$REFDEV
  if [ "$NGPU" -ge 4 ]; then
    $PY tests/glm_parity.py engine $M $O/parity > $O/logs/gate2-engine.log 2>&1 &
    local pe=$!
    $PY tests/glm_parity.py ref $M $O/parity --ref-devices $refdev
    wait $pe
  else
    $PY tests/glm_parity.py ref $M $O/parity --ref-devices $refdev
    $PY tests/glm_parity.py engine $M $O/parity
  fi
  cat $O/logs/gate2-engine.log 2>/dev/null || true
  $PY tests/glm_parity.py runner $M $O/parity
  rm -f $O/parity/summary.json
  $PY tests/glm_parity.py report $M $O/parity | tee $O/parity/gate2.txt
  grep -q "verdict: pass" $O/parity/gate2.txt
}

do_vectors() {
  $PY $W runs/vectors.py $M $O/vec run
  $PY runs/vectors.py $M $O/vec vectors > $O/vec/report.txt
  tail -40 $O/vec/report.txt
  $PY runs/glm_summary.py gate $O/vec/aipsy_table.pt
}

do_cal() {
  $PILOT $M $O/cal grid --vectors $O/vec/vectors.pt --layers 26,37,49,58 --doses 1,2,4 \
    --arms D0 --vector-kinds raw --battery $BATTERY
  $PY runs/glm_summary.py cal $O/cal $O/vec/aipsy_table.pt --json $O/choice.json | tee $O/cal/summary.txt
}

do_pilot() {
  local L s p
  L=$($PY -c "import json; print(json.load(open('$O/choice.json'))['layer'])")
  s=$($PY -c "import json; print('%g' % json.load(open('$O/choice.json'))['secondary_dose'])")
  p=$($PY -c "import json; print('%g' % json.load(open('$O/choice.json'))['primary_dose'])")
  $PILOT $M $O/pilot grid --vectors $O/vec/vectors.pt --layers $L --doses $s,$p \
    --arms D0,D1.4 --task-site --vector-kinds raw --battery $BATTERY
  $PY runs/glm_summary.py pilot $O/pilot --choice $O/choice.json --json $O/pilot/summary.json | tee $O/pilot/summary.txt
  $PY runs/confirm_table.py $O/pilot glm $O/pilot/table.csv --entity GLM --belief  # belief_resid: exploratory
}

do_think() {
  $PY runs/glm_think.py engine $M $O/think
}

for s in "$@"; do stage "$s"; done
