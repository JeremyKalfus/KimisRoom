#!/usr/bin/env bash
# region.sh NAME: print the region holding the (non-terminated) instance NAME. Defaults to us-east-2.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cache=/tmp/kimisroom-region-$1
[ -s "$cache" ] && { cat "$cache"; exit 0; }
for r in us-east-2 us-east-1; do
  id=$(AWS_DEFAULT_REGION=$r "$here/aws.sh" ec2 describe-instances --filters Name=tag:Name,Values="$1" \
    Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[0].Instances[0].InstanceId' --output text)
  [ "$id" != None ] && { echo $r | tee "$cache"; exit 0; }
done
echo us-east-2
