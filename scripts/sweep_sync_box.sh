#!/usr/bin/env bash
# sweep_sync_box.sh IP: push the committed repo (git HEAD, not the working tree) to ~/KimisRoom on the
# sweep's cloud box, extracting in place (running jobs keep their cwd; results are never touched).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
git -C "$root" archive --format=tar HEAD | ssh -i ~/.ssh/kimisroom_aws -o StrictHostKeyChecking=accept-new ubuntu@"$1" 'mkdir -p ~/KimisRoom && tar -x -C ~/KimisRoom'
