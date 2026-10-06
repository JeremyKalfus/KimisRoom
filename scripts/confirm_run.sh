#!/usr/bin/env bash
# confirm_run.sh: the unattended confirmatory run (PREREG.md), driven from the Mac. Start it detached
# under one `kv exec`, so the AWS key is read once for the whole job (its kv grant lasts 10 minutes):
#
#   source .env.local; mkdir -p results/confirm; nohup caffeinate -ims kv exec AWS_SECRET_ACCESS_KEY="$AWS_KV_NAME" -- \
#     env AWS_ACCESS_KEY_ID="$AWS_KV_NAME" scripts/confirm_run.sh >> results/confirm/orchestrate.log 2>&1 &
#
# 1. Stages the inputs in S3: both vector files and the frozen D2/D2c adapters (results/qwen/adapters,
#    copied out of snapshot snap-02ecae796faa329e9).
# 2. Grabs a g6e.12xlarge in any region with G quota, every default-subnet AZ in turn, retrying for
#    GRAB_HOURS (default 72).
# 3. Starts scripts/confirm_box.sh there with presigned URLs (7 days), so the box needs no credentials.
# 4. Follows the box's status file in S3 every 10 minutes, pulls each result directory as it lands,
#    and once the box reports done or failed (or stops), stops it and runs the registered analysis into
#    results/confirm/analysis. Rerunning resumes: the box keeps finished grids on its EBS root.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
NAME=${CONFIRM_NAME:-kr-c}; TYPE=g6e.12xlarge; R=results/confirm; LABELS="qwen k26 qwen-D2 qwen-D2c"
REGIONS=${CONFIRM_REGIONS:-"us-east-2 us-east-1 us-west-2 eu-central-1"}
mkdir -p $R
export AWS_DEFAULT_REGION=us-east-2
aws() { scripts/aws.sh "$@"; }
log() { echo "$(date +%FT%T) $*"; }
notify() { osascript -e "display notification \"$1\" with title \"Kimi's Room\"" 2>/dev/null || true; }
presign() { uv run -q --no-project --with boto3 python scripts/presign.py "$B" 604800 "$@"; }

acct=$(aws sts get-caller-identity --query Account --output text) || { log "no AWS credentials"; exit 1; }
B=kimisroom-confirm-$acct
aws s3api head-bucket --bucket $B 2>/dev/null || \
  aws s3api create-bucket --bucket $B --create-bucket-configuration LocationConstraint=us-east-2 > /dev/null || exit 1

# ---- 1. inputs -------------------------------------------------------------------------------------------
[ -f $R/adapters.tar ] || COPYFILE_DISABLE=1 tar --no-mac-metadata -C results/qwen/adapters -cf $R/adapters.tar distill_keep
H_keep=$(shasum -a 256 $R/adapters.tar | cut -d' ' -f1)
for pair in in/qwen_vectors.pt:results/qwen/v2/vectors.pt in/k26_vectors.pt:results/k26/v2/vectors.pt in/adapters.tar:$R/adapters.tar; do
  key=${pair%%:*}; src=${pair#*:}
  aws s3api head-object --bucket $B --key $key > /dev/null 2>&1 && continue
  log "uploading $src"; aws s3 cp --quiet "$src" s3://$B/$key || { log "upload of $src failed"; exit 1; }
done
log "inputs staged in s3://$B/in"

# ---- 2. capacity -----------------------------------------------------------------------------------------
deadline=$(( $(date +%s) + ${GRAB_HOURS:-72} * 3600 )); region=""; n=0
while :; do
  for r in $REGIONS; do
    if AWS_DEFAULT_REGION=$r scripts/vm_up.sh $NAME $TYPE > /dev/null 2> $R/.vm_up.err; then region=$r; break; fi
  done
  [ -n "$region" ] && break
  n=$((n + 1)); [ $((n % 10)) = 1 ] && log "no $TYPE yet (attempt $n): $(grep -o '[A-Za-z]*Capacity\|VcpuLimitExceeded\|[A-Za-z]*Exception' $R/.vm_up.err | sort | uniq -c | tr '\n' ' ')"
  [ "$(date +%s)" -gt $deadline ] && { log "gave up: no capacity in ${GRAB_HOURS:-72}h"; notify "no GPU capacity, gave up"; exit 1; }
  sleep 120
done
export AWS_DEFAULT_REGION=$region; echo $region > /tmp/kimisroom-region-$NAME
log "$TYPE up in $region"; notify "GPU box up in $region, confirmatory run starting"
for _ in $(seq 60); do scripts/vm_ssh.sh $NAME 'test -L /nvme && mountpoint -q /opt/dlami/nvme' 2>/dev/null && break; sleep 10; done
scripts/sync.sh $NAME || { log "sync failed"; exit 1; }

# ---- 3. start the box ------------------------------------------------------------------------------------
env_file=$(mktemp); chmod 600 $env_file
{ paste -d= <(printf '%s\n' U_qvec U_kvec U_keep) <(presign get in/qwen_vectors.pt in/k26_vectors.pt in/adapters.tar | sed "s/.*/'&'/")
  paste -d= <(printf '%s\n' P_qwen P_k26 P_qwenD2 P_qwenD2c P_status P_log) \
            <(presign put out/qwen.tar out/k26.tar out/qwen-D2.tar out/qwen-D2c.tar out/status.txt out/run.log | sed "s/.*/'&'/")
  echo "H_keep=$H_keep"; } > $env_file
[ "$(grep -c "='https://" $env_file)" = 9 ] || { log "presigning failed"; rm -f $env_file; exit 1; }
aws s3 rm --quiet s3://$B/out/status.txt 2>/dev/null
scripts/vm_ssh.sh $NAME 'mkdir -p ~/confirm && umask 077 && cat > ~/confirm/urls.env' < $env_file; rm -f $env_file
scripts/vm_ssh.sh $NAME 'nohup bash ~/KimisRoom/scripts/confirm_box.sh > /dev/null 2>&1 < /dev/null & echo started' || exit 1
log "confirm_box.sh started on $NAME"

# ---- 4. follow -------------------------------------------------------------------------------------------
last=""
while :; do
  sleep 600
  aws s3 cp --quiet s3://$B/out/status.txt $R/status.txt 2>/dev/null
  st=$(head -1 $R/status.txt 2>/dev/null | grep -o 'stage=.*')
  [ "$st" != "$last" ] && { log "$st"; last=$st; }
  for l in $LABELS; do
    [ -f $R/$l/grid.pt ] && continue
    aws s3 cp --quiet s3://$B/out/$l.tar $R/$l.tar 2>/dev/null && tar -C $R -xf $R/$l.tar && rm $R/$l.tar && log "$l pulled" && notify "$l results in"
  done
  case "$st" in stage=done*|stage=failed*) break ;; esac
  state=$(aws ec2 describe-instances --filters Name=tag:Name,Values=$NAME --query 'Reservations[0].Instances[0].State.Name' --output text)
  [ "$state" = stopped ] && { log "box stopped before reporting done ($st)"; break; }
done
aws s3 cp --quiet s3://$B/out/run.log $R/box_run.log 2>/dev/null
scripts/vm_down.sh $NAME

# ---- analysis (runs/confirm_table.py, runs/confirm.R) ------------------------------------------------------
mkdir -p $R/tables $R/analysis
for l in $LABELS; do
  [ -f $R/$l/grid.pt ] || { log "$l missing"; continue; }
  ent=(); [ $l != k26 ] && ent=(--entity Qwen)
  PYTHONPATH=. .venv/bin/python runs/confirm_table.py $R/$l $l $R/tables/$l.csv ${ent[@]+"${ent[@]}"}  # bash 3.2: an empty array is unbound under set -u
done
Rscript runs/confirm.R $R/analysis $R/tables/*.csv > $R/analysis/report.txt 2>&1
log "analysis written to $R/analysis ($st)"; notify "confirmatory run finished: $st"
