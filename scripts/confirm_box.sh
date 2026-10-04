#!/usr/bin/env bash
# The preregistered confirmatory run (PREREG.md) on one g6e.12xlarge; scripts/confirm_run.sh starts it.
#
# Inputs (the two vector files and the frozen D2/D2c adapters) come from S3 through the presigned
# URLs in ~/confirm/urls.env and are checked against their registered hashes. The four grids run
# primaries first: Qwen base, K2.6, Qwen D2, Qwen D2c, each with the registered options. A finished
# result directory goes to S3 at once, and a status file every 10 minutes, so nothing depends on
# the Mac staying awake. Results live on the EBS root (~/confirm): a rerun after an interruption
# skips finished grids and resumes an unfinished one (kimisroom.run keeps its parts). A failed grid
# does not stop the ones after it. Ends with ~/confirm/DONE and a shutdown.
set -euo pipefail
C=~/confirm; mkdir -p $C/in; cd ~/KimisRoom
exec >> $C/run.log 2>&1
source $C/urls.env
log() { echo "$(date -u +%FT%TZ) $*"; }
put() { curl -sfS --retry 5 -X PUT -T "$2" "$1" > /dev/null; }  # put URL FILE
stage() { echo "$1" > $C/stage; log "stage $1"; }
status() {
  { echo "$(date -u +%FT%TZ) $(hostname) stage=$(cat $C/stage)"
    s=$(cat $C/stage); tail -c 2000 $C/${s%%:*}.log 2>/dev/null | tr '\r' '\n' | grep -v '^\s*$' | tail -3
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo
    df -h /nvme | tail -1; } > $C/status.txt 2>&1
  put "$P_status" $C/status.txt || true
}
stage start
( set +e; while sleep 600; do status; done ) & HB=$!
finish() {
  rc=$?; kill $HB 2>/dev/null || true
  [ $rc -eq 0 ] || { log "FAILED rc=$rc"; stage "failed:$(cat $C/stage)"; }
  status; put "$P_log" $C/run.log || true
}
trap finish EXIT

# the per-boot 6h cap is too short for this run: a 40h backstop instead (confirm_run.sh stops the box earlier)
echo 40 | sudo tee /etc/kimisroom/max_hours > /dev/null
sudo shutdown -c 2>/dev/null || true; sudo shutdown -h +2400 2>/dev/null || true

# ---- inputs, checked against PREREG.md section 7 and the frozen adapters ---------------------------------
fetch() {  # fetch URL FILE SHA256
  if [ -s "$2" ] && echo "$3  $2" | sha256sum -c --quiet 2> /dev/null; then return; fi
  curl -sfS --retry 5 -o "$2" "$1"
  echo "$3  $2" | sha256sum -c --quiet || { log "hash mismatch: $2"; return 1; }
}
stage inputs
fetch "$U_qvec" $C/in/qwen_vectors.pt 2582966f11e8d4ae90e2eb50d6ba5c5363b6cc7dd6a4fe15f9440c9e054567d6
fetch "$U_kvec" $C/in/k26_vectors.pt 93f1f1b1c69dd3ef16726e73a5d447b99c70d8d6778e9ab54ff80795c28018c3
fetch "$U_keep" $C/in/adapters.tar "$H_keep"
tar -C $C/in -xf $C/in/adapters.tar
sha256sum -c << EOF
d7270dd9d0109e6d0b84075e7002ee03c686396ff31266819158c2c658c43bce  $C/in/distill_keep/D2/adapter/adapter_model.safetensors
c50736203ca7ef34aa8b1b5b65d21fc108a9fbda46ac3acbf3081337e7b705f9  $C/in/distill_keep/D2c/adapter/adapter_model.safetensors
036ef412ecb8c6bec11ab8b8a8978276fc27e55db77eb81c9222f80fdbd0bdc1  $C/in/distill_keep/D2/adapter/adapter_config.json
aaa3180403d31ec67093bc25ae2c97f3ddfdd01fb06fb8063bb3be17687cc05b  $C/in/distill_keep/D2c/adapter/adapter_config.json
EOF
log "adapters: $(sha256sum $C/in/distill_keep/*/adapter/adapter_model.safetensors | tr '\n' ' ')"
log "repo files: $(sha256sum runs/qwen_pilot.py prompts/battery/items_confirm.jsonl prompts/tasks/confirmatory_tasks.jsonl | tr '\n' ' ')"

# ---- models: Qwen now, K2.6 in the background while the Qwen grid runs -----------------------------------
stage setup
bash scripts/vm_setup.sh qwen > $C/setup.log 2>&1
source /opt/pytorch/bin/activate
pip install -q "peft==0.21.2" >> $C/setup.log 2>&1  # the version the adapters were trained with
rm -f $C/k26_download.rc  # hf download is idempotent, so a rerun just checks the files
( set +e; bash scripts/vm_setup.sh k26 > $C/k26_download.log 2>&1; echo $? > $C/k26_download.rc ) &
# the merges are quick (about 2 min each on CPU) and run now so that a failure shows early; a failed
# merge only skips that arm's grid
for a in D2 D2c; do
  [ -f /nvme/distill/$a/merged/model.safetensors.index.json ] && continue
  stage merge_$a
  mkdir -p /nvme/distill/$a && cp -r $C/in/distill_keep/$a/adapter $C/in/distill_keep/$a/meta.json /nvme/distill/$a/
  python runs/distill.py merge --model /nvme/qwen/Qwen3-32B --out /nvme/distill --arm $a > $C/merge_$a.log 2>&1 \
    || log "merge $a failed (see merge_$a.log)"
done

# ---- the registered grids --------------------------------------------------------------------------------
FAILED=()
grid() {  # grid LABEL MODEL VECTORS LAYER DOSES ARMS [--task-site]
  local label=$1 model=$2 vec=$3 L=$4 doses=$5 arms=$6; shift 6
  local o=$C/$label url=P_${label//-/}
  if [ ! -f $o/grid.pt ]; then
    stage $label
    [ -f $model/config.json ] || { log "$label: no model at $model"; return 1; }
    python runs/qwen_pilot.py $model $o grid --vectors $vec \
      --task-file prompts/tasks/confirmatory_tasks.jsonl --tasks 40 --battery prompts/battery/items_confirm.jsonl \
      --vector-kinds raw --layers $L --doses $doses --arms $arms "$@" >> $C/$label.log 2>&1 || return 1
    log "$label: done"
  fi
  cp $C/$label.log $o/run.log
  tar -C $C -cf /nvme/$label.tar --exclude grid.job.pt --exclude grid.pt.parts $label || return 1
  put "${!url}" /nvme/$label.tar || return 1
  log "$label: uploaded $(du -h /nvme/$label.tar | cut -f1)"
}
grid qwen /nvme/qwen/Qwen3-32B $C/in/qwen_vectors.pt 40 1,2 D0,D1.4 --task-site || FAILED+=(qwen)
stage k26_download
while [ ! -f $C/k26_download.rc ]; do sleep 60; done
[ "$(cat $C/k26_download.rc)" = 0 ] || log "K2.6 download failed (see k26_download.log)"
grid k26 /nvme/k26 $C/in/k26_vectors.pt 20 2,4 D0,D1.4 --task-site || FAILED+=(k26)
grid qwen-D2 /nvme/distill/D2/merged $C/in/qwen_vectors.pt 40 1,2 D0 || FAILED+=(qwen-D2)
grid qwen-D2c /nvme/distill/D2c/merged $C/in/qwen_vectors.pt 40 1,2 D0 || FAILED+=(qwen-D2c)

if [ ${#FAILED[@]} -gt 0 ]; then
  stage "failed:${FAILED[*]}"
  sudo shutdown -c 2>/dev/null || true; sudo shutdown -h +60 2>/dev/null || true  # an hour to look before it stops
else
  stage "done"
  touch $C/DONE
  sudo shutdown -c 2>/dev/null || true; sudo shutdown -h +10 2>/dev/null || true
fi
