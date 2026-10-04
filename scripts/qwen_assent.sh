#!/usr/bin/env bash
# Run on the instance after the grid: training deps, then Qwen3-32B's frozen assents for every
# denial argument (new: D1.4 and the D1.5 self-authored essay), copied into data/assent/.
set -euo pipefail
cd ~/KimisRoom
source /opt/pytorch/bin/activate
pip install -q peft bitsandbytes
python runs/qwen_pilot.py /nvme/qwen/Qwen3-32B ~/qwenassent assent > ~/qwenassent.log 2>&1
cp ~/qwenassent/qwen3-32b_d1_argument_4.json ~/qwenassent/qwen3-32b_d1_argument_4_selfauthored.json data/assent/
echo ASSENT DONE >> ~/qwenassent.log
