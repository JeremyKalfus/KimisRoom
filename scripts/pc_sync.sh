#!/usr/bin/env bash
# pc_sync.sh: push the repo (code, prompts, data; no results, .git or secrets) to ~/KimisRoom in WSL on
# Jeremy's desktop (`ssh pc`, PowerShell default shell). stdin through ssh -> PowerShell -> wsl.exe is
# binary-safe (checked by sha256), so a tar stream replaces rsync, which Windows OpenSSH cannot serve.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
COPYFILE_DISABLE=1 tar -C "$root" -cz --exclude .git --exclude refs --exclude results --exclude .env.local \
  --exclude __pycache__ --exclude .venv --exclude drafts --exclude '*.pt' . |
  ssh pc '$env:WSL_UTF8=1; wsl -e bash -c "rm -rf ~/KimisRoom.new && mkdir -p ~/KimisRoom.new && tar -xz -C ~/KimisRoom.new 2>/dev/null; rm -rf ~/KimisRoom.old; [ -d ~/KimisRoom ] && mv ~/KimisRoom ~/KimisRoom.old; mv ~/KimisRoom.new ~/KimisRoom && rm -rf ~/KimisRoom.old && du -sh ~/KimisRoom"'
