#!/usr/bin/env bash
# glm_grab.sh [MINUTES]: a g6e.12xlarge for GLM-5.3, retried until MINUTES (default 120) pass.
# First the stopped kr-c in us-east-1, the box the GLM brief names (its root EBS, so its AZ, is fixed).
# When that AZ has no capacity, the grab logic of scripts/confirm_run.sh: a box named kr-glm, created by
# scripts/vm_up.sh in every default-subnet AZ of each region in turn. Prints "NAME REGION IP" and caches
# the region for vm_ssh.sh / sync.sh. Run it under one kv exec, as confirm_run.sh's header shows.
# GLM_FALLBACK="TYPE@REGION ..." adds other instance types (named kr-glm-TYPE) once GLM_FALLBACK_AFTER
# minutes (default 90) have passed without a g6e.12xlarge, e.g. "p5.4xlarge@us-east-2" (1x H100 80 GB,
# 16 vCPU, inside the P quota there): the engine runs on any number of GPUs.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REGIONS=${GLM_REGIONS:-"us-east-2 us-east-1 us-west-2 eu-central-1"}
start=$(date +%s); deadline=$(( start + ${1:-120} * 60 ))
FALLBACK=${GLM_FALLBACK:-}; FALLBACK_AFTER=${GLM_FALLBACK_AFTER:-90}
aws() { scripts/aws.sh "$@"; }
log() { echo "$(date +%T) $*" >&2; }
krc=$(AWS_DEFAULT_REGION=us-east-1 aws ec2 describe-instances --filters Name=tag:Name,Values=kr-c \
  Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[0].Instances[0].InstanceId' --output text)
n=0
while :; do
  n=$((n + 1))
  if [ "$krc" != None ]; then
    state=$(AWS_DEFAULT_REGION=us-east-1 aws ec2 describe-instances --instance-ids "$krc" --query 'Reservations[0].Instances[0].State.Name' --output text)
    [ "$state" = stopping ] && AWS_DEFAULT_REGION=us-east-1 aws ec2 wait instance-stopped --instance-ids "$krc"
    if [ "$state" = running ] || AWS_DEFAULT_REGION=us-east-1 aws ec2 start-instances --instance-ids "$krc" > /dev/null 2> /tmp/kimisroom-glm-grab.err; then
      AWS_DEFAULT_REGION=us-east-1 aws ec2 wait instance-running --instance-ids "$krc"
      ip=$(AWS_DEFAULT_REGION=us-east-1 aws ec2 describe-instances --instance-ids "$krc" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
      echo us-east-1 > /tmp/kimisroom-region-kr-c
      echo "kr-c us-east-1 $ip"; exit 0
    fi
    [ $((n % 5)) = 1 ] && log "kr-c (us-east-1): $(grep -o '[A-Za-z]*Capacity\|[A-Za-z]*Exception\|[A-Za-z]*Limit[A-Za-z]*' /tmp/kimisroom-glm-grab.err | head -1)"
  fi
  for r in $REGIONS; do
    if ip=$(AWS_DEFAULT_REGION=$r scripts/vm_up.sh kr-glm g6e.12xlarge 2> /tmp/kimisroom-glm-grab-$r.err); then
      echo $r > /tmp/kimisroom-region-kr-glm
      echo "kr-glm $r $ip"; exit 0
    fi
    [ $((n % 5)) = 1 ] && log "kr-glm ($r): $(grep -o '[A-Za-z]*Capacity\|VcpuLimitExceeded\|[A-Za-z]*Exception' /tmp/kimisroom-glm-grab-$r.err | sort | uniq -c | tr '\n' ' ')"
  done
  if [ -n "$FALLBACK" ] && [ $(( $(date +%s) - start )) -ge $(( FALLBACK_AFTER * 60 )) ]; then
    for spec in $FALLBACK; do
      t=${spec%@*}; r=${spec#*@}; nm=kr-glm-${t%%.*}
      if ip=$(AWS_DEFAULT_REGION=$r scripts/vm_up.sh "$nm" "$t" 2> /tmp/kimisroom-glm-grab-$r-$t.err); then
        echo $r > /tmp/kimisroom-region-$nm
        echo "$nm $r $ip $t"; exit 0
      fi
      [ $((n % 5)) = 1 ] && log "$nm ($t, $r): $(grep -o '[A-Za-z]*Capacity\|VcpuLimitExceeded\|[A-Za-z]*Exception\|[A-Za-z]*Unsupported' /tmp/kimisroom-glm-grab-$r-$t.err | sort | uniq -c | tr '\n' ' ')"
    done
  fi
  [ "$(date +%s)" -gt $deadline ] && { log "no g6e.12xlarge after ${1:-120} min"; exit 1; }
  sleep 120
done
