#!/usr/bin/env bash
# pc_sync.sh: push the committed repo (git HEAD: code, prompts, data; never the working tree, so a
# parallel agent's uncommitted edits stay local) to ~/KimisRoom in WSL on Jeremy's desktop (`ssh pc`,
# PowerShell default shell). Extracts in place, so a job running with ~/KimisRoom as its cwd keeps a
# valid directory. stdin through ssh -> PowerShell -> wsl.exe is binary-safe (checked by sha256).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
git -C "$root" archive --format=tar HEAD |
  ssh pc '$env:WSL_UTF8=1; wsl -e bash -c "mkdir -p ~/KimisRoom && tar -x -C ~/KimisRoom && git_head=1 du -sh ~/KimisRoom"'
