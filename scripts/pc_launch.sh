#!/usr/bin/env bash
# pc_launch.sh SCRIPT [ARGS...]: run a repo script inside WSL on the desktop, detached from the SSH
# session. Windows OpenSSH kills a session's process tree when it closes, and WSL stops a distro
# once no wsl.exe is attached, so the job is started through WMI (Win32_Process.Create), which
# parents a wsl.exe outside the session; that wsl.exe keeps the distro up until the job ends.
# Log: ~/sweep/<script name>.log in WSL.
set -euo pipefail
s=$1; shift
name=$(basename "$s" .sh)
cmd="wsl.exe -d Ubuntu-24.04 -e bash -lc \"cd ~/KimisRoom && bash $s $* > ~/sweep/$name.log 2>&1\""
ssh pc "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='$cmd'}; \"pid \$(\$r.ProcessId) rc \$(\$r.ReturnValue)\""
