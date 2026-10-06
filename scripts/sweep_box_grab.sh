#!/usr/bin/env bash
# sweep_box_grab.sh: a g6e.2xlarge for the Qwen3 size sweep, under one kv grant. First kr-q
# (us-east-2c, already resized by sweep_box_up.sh) for a few tries; if that AZ has no capacity,
# a new instance kr-s via vm_up.sh in us-east-2 (any AZ), then us-east-1. Prints "NAME REGION IP".
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  source "$here/../.env.local"
  exec kv exec AWS_SECRET_ACCESS_KEY="$AWS_KV_NAME" -- env AWS_ACCESS_KEY_ID="$AWS_KV_NAME" "$0" "$@"
fi
for i in 1 2 3; do
  if ip=$(AWS_DEFAULT_REGION=us-east-2 "$here/sweep_box_up.sh" g6e.2xlarge 2>/tmp/kr-sweep-grab.err); then
    echo "kr-q us-east-2 $ip"; exit 0
  fi
  grep -o 'InsufficientInstanceCapacity' /tmp/kr-sweep-grab.err | head -1 >&2 || cat /tmp/kr-sweep-grab.err >&2
  sleep 45
done
for r in us-east-2 us-east-1; do
  for n in 1 2; do
    if ip=$(AWS_DEFAULT_REGION=$r "$here/vm_up.sh" kr-s g6e.2xlarge 200); then
      echo "kr-s $r $ip"; exit 0
    fi
    sleep 30
  done
done
exit 1
