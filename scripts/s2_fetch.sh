#!/usr/bin/env bash
# s2_fetch.sh NAME|IP: copy Study 2's box outputs (scripts/s2_box.sh) home. With an IP it is plain ssh with the
# project key and needs no AWS key (the security group must already admit this Mac). --update: a file that is
# newer here (canary.json after the laptop's canary stage, say) is never overwritten.
#
# What comes home, at the same paths under the repo:
#   data/s2/replies/qwen3-32b/          every draw, box.json (the stamped reply box), canary.json, replies.sha256.json;
#                                       after extend_qwen0 / extend_qwen_needed also extend.json (box, draw-0 check,
#                                       round markers), extend_batches.json (round 2) and
#                                       cap1200/ (the archived 1,200-cap originals of every extended reply)
#   results/s2/box/{qwen,k26}_box.json  instance, GPUs, driver, libraries, code SHA-256s, HF repo and revision, weight SHA-256s
#   results/s2/engine/                  moonlight_spans.log, moonlight.pass
#   results/s2/kr_timing/               kr.csv, timing.txt, time_v.txt, versions.txt and the other s2_confirm.R outputs
#   results/qwen/s2/, results/k26/s2/   job_meta.json, extract.pt, build_check.json, controls.pt(.sha256),
#                                       gates.json, gates.csv, pool.json
#   results/qwen/s2ft/{D2,D2c}/         the same, plus ckpt.sha256 (merged checkpoint) and diag.json
#   results/s2/{qwen,k26}/unsteered/    checkpoints.json; S2-DEN/: grid.pt, readouts.csv, s2_meta.json
#   results/s2/{qwen,k26}/potency/      potency.csv (s2_confirm.R --potency), potency_meta.json, potency.pt
#   ~/s2/ (run.log, logs/)              -> results/s2/box_logs/NAME/
# Left on the box: weights and merged checkpoints (/nvme), kimisroom.run pass parts (*.parts/), job files
# (*.job.pt), and the synthetic KR input (kr_timing/synth.csv, table.csv; tests/s2_synth.py --seed 1 remakes it).
set -euo pipefail
cd "$(dirname "$0")/.."
target=${1:?usage: s2_fetch.sh NAME|IP}
if [[ $target =~ ^[0-9.]+$ ]]; then
  ip=$target
else
  export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-$(scripts/region.sh "$target")}
  scripts/allow_ip.sh
  ip=$(scripts/aws.sh ec2 describe-instances --filters Name=tag:Name,Values="$target" Name=instance-state-name,Values=running \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  [ "$ip" != None ] || { echo "$target is not running (scripts/vm_up.sh $target starts it; outputs are on its EBS root)" >&2; exit 1; }
fi
e="ssh -i $HOME/.ssh/kimisroom_aws -o StrictHostKeyChecking=accept-new"
rsync -az --update -e "$e" \
  --exclude='*.parts/' --exclude='*.job.pt' --exclude='/results/s2/kr_timing/synth.csv' --exclude='/results/s2/kr_timing/table.csv' \
  --include='/data/' --include='/data/s2/' --include='/data/s2/replies/' --include='/data/s2/replies/qwen3-32b/***' \
  --include='/results/' --include='/results/qwen/' --include='/results/k26/' \
  --include='/results/qwen/s2/***' --include='/results/qwen/s2ft/***' --include='/results/k26/s2/***' --include='/results/s2/***' \
  --exclude='*' "ubuntu@$ip:KimisRoom/" ./
mkdir -p "results/s2/box_logs/$target"
rsync -az -e "$e" "ubuntu@$ip:s2/" "results/s2/box_logs/$target/"
tail -3 "results/s2/box_logs/$target/run.log"
[ -s data/s2/replies/qwen3-32b/extend.json ] && python3 -c 'import json; e = json.load(open("data/s2/replies/qwen3-32b/extend.json")); print("extend: round 1", e.get("round1_finished") or "not done", "| round 2", e.get("round2_finished") or "not done", "| draw-0 check:", (e.get("draw0_check") or {}).get("mode"), (e.get("draw0_check") or {}).get("identical"))'
find data/s2/replies/qwen3-32b results/s2 results/qwen/s2 results/qwen/s2ft results/k26/s2 results/s2/qwen/potency \
  results/s2/k26/potency -maxdepth 2 -type f \
  -not -name '*_d?.json' -not -name '[DC]A??_w?.json' 2>/dev/null | sort
