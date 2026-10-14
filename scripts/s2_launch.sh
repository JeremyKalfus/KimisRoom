#!/usr/bin/env bash
# s2_launch.sh NAME STAGE...: bring up (or restart) the g6e.12xlarge NAME, retrying until capacity appears
# (grab.sh), push code and inputs (s2_sync.sh), and start scripts/s2_box.sh STAGE... detached on it; the box
# then needs nothing from the Mac. Runbook and stages: top of scripts/s2_box.sh. Follow it with
#   scripts/vm_ssh.sh NAME tail -f s2/run.log
# Every aws call reads the key through kv (scripts/aws.sh); for a long capacity wait run this under one
# `kv exec` as scripts/confirm_run.sh shows. S2_TYPE, S2_REGIONS and GRAB_MINUTES override the defaults.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
[ $# -ge 2 ] || { echo "usage: $0 NAME qwen|k26|STAGE..." >&2; exit 2; }
name=$1; shift
# S2_TYPES: instance types tried in order each round (all 4x L40S, so the numerics match); a type over a
# region's G quota just fails there.
types=${S2_TYPES:-${S2_TYPE:-g6e.12xlarge}}; regions=${S2_REGIONS:-"us-east-2 us-east-1"}
deadline=$(( $(date +%s) + ${GRAB_MINUTES:-180} * 60 ))
# An existing NAME (stopped or running) is only ever restarted in its own region: trying the other region
# would create a second NAME there (grab.sh does that when a stopped box cannot restart).
for r in $regions; do
  id=$(AWS_DEFAULT_REGION=$r "$here/aws.sh" ec2 describe-instances --filters Name=tag:Name,Values="$name" \
    Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[0].Instances[0].InstanceId' --output text) || exit 1
  [ "$id" != None ] && { echo "$name is $id in $r"; regions=$r; break; }
done
while [ "$(date +%s)" -lt $deadline ]; do
  for type in $types; do for r in $regions; do
    if ip=$(AWS_DEFAULT_REGION=$r "$here/vm_up.sh" "$name" "$type" 2> /tmp/kimisroom-s2-up.err); then
      echo "$(date +%H:%M) $type up in $r at $ip"; echo "$r" > /tmp/kimisroom-region-$name
      export AWS_DEFAULT_REGION=$r
      for _ in $(seq 60); do "$here/vm_ssh.sh" "$name" 'test -L /nvme && mountpoint -q /opt/dlami/nvme' 2>/dev/null && break; sleep 10; done
      "$here/s2_sync.sh" "$name" || { echo "sync failed"; exit 1; }
      "$here/vm_ssh.sh" "$name" "mkdir -p ~/s2 && nohup bash ~/KimisRoom/scripts/s2_box.sh $* > /dev/null 2>&1 < /dev/null & echo 's2_box.sh $* started'"
      exit $?
    fi
  done; done
  echo "$(date +%H:%M) no $types: $(grep -o '[A-Za-z]*Capacity\|VcpuLimitExceeded\|[A-Za-z]*Exception' /tmp/kimisroom-s2-up.err | sort -u | tr '\n' ' ')"
  sleep 180
done
echo "gave up after ${GRAB_MINUTES:-180} min"; exit 1
