#!/usr/bin/env bash
# belief_run.sh: the unattended belief replication (drafts/prereg-amendment-belief.md), driven from the
# Mac, modelled on scripts/confirm_run.sh. It shares the confirmatory box kr-c (g6e.12xlarge, us-east-1)
# with the other lanes, so run it only once the GLM lane has finished with kr-c or handed it over, and
# only after the amendment is timestamped. Start it detached under one `kv exec`:
#
#   source .env.local; mkdir -p results/belief
#   BELIEF_OTS=<the amendment's .ots proof> BELIEF_COMMIT=<the amendment commit or tag> BELIEF_HANDOVER=1 \
#     nohup caffeinate -ims kv exec AWS_SECRET_ACCESS_KEY="$AWS_KV_NAME" -- env AWS_ACCESS_KEY_ID="$AWS_KV_NAME" \
#     scripts/belief_run.sh >> results/belief/orchestrate.log 2>&1 &
#
# Guards: refuses without BELIEF_HANDOVER=1 (the orchestrator's statement that the GLM lane is done with
# the box) and without an existing BELIEF_OTS file; the box script refuses while any process holds a GPU,
# and then this script leaves the box running and untouched.
#
# 1. Stages the inputs in the confirmatory bucket (the same vector files and adapters as the original
#    run, so normally nothing is uploaded).
# 2. Starts kr-c (or creates it if it no longer exists) in BELIEF_REGIONS, retrying for GRAB_HOURS.
# 3. Copies the registered commit (`git archive`, so only committed files) to ~/KimisRoom-belief on the
#    box, never touching ~/KimisRoom, and starts scripts/belief_box.sh with presigned URLs (7 days).
# 4. Follows the box's status file every 10 minutes, pulls each result directory as it lands, and once
#    the box reports done or failed (or stops), stops it (unless BELIEF_KEEP_BOX=1) and runs the
#    registered analysis (runs/confirm_table.py --belief, runs/belief_confirm.R) into
#    results/belief/analysis. Rerunning resumes: the box keeps finished grids on its EBS root.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
NAME=${BELIEF_NAME:-kr-c}; TYPE=g6e.12xlarge; R=results/belief; LABELS="qwen k26 qwen-D2 qwen-D2c"
REGIONS=${BELIEF_REGIONS:-us-east-1}
COMMIT=$(git rev-parse --verify "${BELIEF_COMMIT:-HEAD}^{commit}") || { echo "bad BELIEF_COMMIT"; exit 1; }
mkdir -p $R
export AWS_DEFAULT_REGION=us-east-2
aws() { scripts/aws.sh "$@"; }
log() { echo "$(date +%FT%T) $*"; }
notify() { osascript -e "display notification \"$1\" with title \"Kimi's Room\"" 2>/dev/null || true; }
presign() { uv run -q --no-project --with boto3 python scripts/presign.py "$B" 604800 "$@"; }

[ "${BELIEF_HANDOVER:-}" = 1 ] || { log "refusing: set BELIEF_HANDOVER=1 once the GLM lane is done with $NAME"; exit 1; }
[ -n "${BELIEF_OTS:-}" ] && [ -s "$BELIEF_OTS" ] || { log "refusing: BELIEF_OTS must name the amendment's timestamp proof"; exit 1; }
[ -n "${BELIEF_COMMIT:-}" ] || log "warning: BELIEF_COMMIT unset, running HEAD ($COMMIT)"
log "belief replication from commit $COMMIT, proof $BELIEF_OTS"

acct=$(aws sts get-caller-identity --query Account --output text) || { log "no AWS credentials"; exit 1; }
B=kimisroom-confirm-$acct
aws s3api head-bucket --bucket $B 2>/dev/null || \
  aws s3api create-bucket --bucket $B --create-bucket-configuration LocationConstraint=us-east-2 > /dev/null || exit 1

# ---- 1. inputs (shared with the original run: in/ keys, checked by hash on the box) ------------------------
A=results/confirm/adapters.tar
[ -f $A ] || COPYFILE_DISABLE=1 tar --no-mac-metadata -C results/qwen/adapters -cf $A distill_keep
H_keep=$(shasum -a 256 $A | cut -d' ' -f1)
for pair in in/qwen_vectors.pt:results/qwen/v2/vectors.pt in/k26_vectors.pt:results/k26/v2/vectors.pt in/adapters.tar:$A; do
  key=${pair%%:*}; src=${pair#*:}
  aws s3api head-object --bucket $B --key $key > /dev/null 2>&1 && continue
  log "uploading $src"; aws s3 cp --quiet "$src" s3://$B/$key || { log "upload of $src failed"; exit 1; }
done
log "inputs staged in s3://$B/in"

# ---- 2. the box -------------------------------------------------------------------------------------------
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
log "$NAME up in $region"
for _ in $(seq 60); do scripts/vm_ssh.sh $NAME 'test -L /nvme && mountpoint -q /opt/dlami/nvme' 2>/dev/null && break; sleep 10; done

# ---- 3. code and start ------------------------------------------------------------------------------------
# A belief run already going on the box (this script restarted on the Mac): follow it, start nothing.
running=$(scripts/vm_ssh.sh $NAME 'bash -s' 2>/dev/null <<'EOF'
s=$(cat ~/belief/stage 2>/dev/null); pid=$(pgrep -u ubuntu -x bash -a | awk '/belief_box.sh/ {print $1}' | head -1)
[ -n "$pid" ] && echo "running:$s"
EOF
)
if [ -n "$running" ]; then
  log "belief_box.sh already running on $NAME ($running); following it"
else
  git archive --format=tar "$COMMIT" | scripts/vm_ssh.sh $NAME \
    "rm -rf ~/KimisRoom-belief.new && mkdir -p ~/KimisRoom-belief.new && tar -xf - -C ~/KimisRoom-belief.new && echo $COMMIT > ~/KimisRoom-belief.new/COMMIT && rm -rf ~/KimisRoom-belief && mv ~/KimisRoom-belief.new ~/KimisRoom-belief" \
    || { log "code copy failed"; exit 1; }
  env_file=$(mktemp); chmod 600 $env_file
  { paste -d= <(printf '%s\n' U_qvec U_kvec U_keep) <(presign get in/qwen_vectors.pt in/k26_vectors.pt in/adapters.tar | sed "s/.*/'&'/")
    paste -d= <(printf '%s\n' P_qwen P_k26 P_qwenD2 P_qwenD2c P_status P_log) \
              <(presign put belief/out/qwen.tar belief/out/k26.tar belief/out/qwen-D2.tar belief/out/qwen-D2c.tar belief/out/status.txt belief/out/run.log | sed "s/.*/'&'/")
    echo "H_keep=$H_keep"; } > $env_file
  [ "$(grep -c "='https://" $env_file)" = 9 ] || { log "presigning failed"; rm -f $env_file; exit 1; }
  aws s3 rm --quiet s3://$B/belief/out/status.txt 2>/dev/null
  scripts/vm_ssh.sh $NAME 'mkdir -p ~/belief && umask 077 && cat > ~/belief/urls.env' < $env_file; rm -f $env_file
  scripts/vm_ssh.sh $NAME 'nohup bash ~/KimisRoom-belief/scripts/belief_box.sh > /dev/null 2>&1 < /dev/null & echo started' || exit 1
  log "belief_box.sh started on $NAME"; notify "belief replication started on $NAME"
fi

# ---- 4. follow --------------------------------------------------------------------------------------------
last=""; st=""
while :; do
  sleep 600
  aws s3 cp --quiet s3://$B/belief/out/status.txt $R/status.txt 2>/dev/null
  st=$(head -1 $R/status.txt 2>/dev/null | grep -o 'stage=.*')
  [ "$st" != "$last" ] && { log "$st"; last=$st; }
  [ "$st" = stage=refused ] && { log "the box refused: a process holds a GPU. $NAME left running and untouched."; notify "belief run refused: GPUs busy"; exit 1; }
  for l in $LABELS; do
    [ -f $R/$l/grid.pt ] && continue
    aws s3 cp --quiet s3://$B/belief/out/$l.tar $R/$l.tar 2>/dev/null && tar -C $R -xf $R/$l.tar && rm $R/$l.tar && log "$l pulled" && notify "belief $l results in"
  done
  case "$st" in stage=done*|stage=failed*) break ;; esac
  state=$(aws ec2 describe-instances --filters Name=tag:Name,Values=$NAME --query 'Reservations[0].Instances[0].State.Name' --output text)
  [ "$state" = stopped ] && { log "box stopped before reporting done ($st)"; break; }
done
aws s3 cp --quiet s3://$B/belief/out/run.log $R/box_run.log 2>/dev/null
[ "${BELIEF_KEEP_BOX:-}" = 1 ] || scripts/vm_down.sh $NAME

# ---- analysis (runs/confirm_table.py --belief, runs/belief_confirm.R) ---------------------------------------
mkdir -p $R/tables $R/analysis
for l in $LABELS; do
  [ -f $R/$l/grid.pt ] || { log "$l missing"; continue; }
  ent=(); [ $l != k26 ] && ent=(--entity Qwen)
  PYTHONPATH=. .venv/bin/python runs/confirm_table.py $R/$l $l $R/tables/$l.csv ${ent[@]+"${ent[@]}"} --belief
done
Rscript runs/belief_confirm.R $R/analysis $R/tables/*.csv > $R/analysis/report.txt 2>&1
log "analysis written to $R/analysis ($st)"; notify "belief replication finished: $st"
