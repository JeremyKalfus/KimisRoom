#!/usr/bin/env bash
# grab.sh NAME TYPE SCRIPT [MINUTES]: retry vm_up.sh across regions until capacity appears,
# then sync the repo and start SCRIPT (repo-relative) detached so the new box is busy at once.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
name=$1; type=$2; script=$3; deadline=$(( $(date +%s) + ${4:-180} * 60 ))
while [ "$(date +%s)" -lt $deadline ]; do
  for r in us-east-2 us-east-1; do
    if ip=$(AWS_DEFAULT_REGION=$r "$here/vm_up.sh" "$name" "$type" 2>/dev/null); then
      echo "$(date +%H:%M) $type up in $r at $ip"; echo $r > /tmp/kimisroom-region-$name
      for _ in $(seq 60); do "$here/vm_ssh.sh" "$name" 'test -L /nvme' 2>/dev/null && break; sleep 10; done
      "$here/sync.sh" "$name" && "$here/vm_ssh.sh" "$name" \
        "nohup bash ~/KimisRoom/$script > ~/bringup.log 2>&1 < /dev/null & echo $script started"
      exit 0
    fi
  done
  echo "$(date +%H:%M) no $type capacity"; sleep 180
done
exit 1
