#!/usr/bin/env bash
# s2_box.sh STAGE...: Study 2's pre-registration GPU work on a g6e.12xlarge (drafts/study2-spec.md sections
# 11-13), one stage after another. A stage whose outputs exist is skipped, so a rerun resumes; a failing
# stage stops the chain. Log: ~/s2/run.log, one log per stage in ~/s2/logs/STAGE.log. Outputs live under
# ~/KimisRoom (EBS, kept over a stop); weights and merged checkpoints live on /nvme (wiped on every stop).
# `qwen` and `k26` expand to the two box sessions below.
#
# Runbook (from the Mac; scripts/s2_launch.sh does vm_up, scripts/s2_sync.sh and the detached start):
#   Before session 1: the accepted wordings in data/s2/openings and the control stories in
#   data/s2/stories/{nouns,thinking}/stories.jsonl are final (stamp N done).
#   Session 1, the Qwen box (about 17 h, mostly the frozen replies):
#     scripts/s2_launch.sh kr-s2 qwen     # setup_qwen env_qwen moonlight vectors_qwen merge ftvec assent
#     scripts/vm_ssh.sh kr-s2 'nohup bash ~/KimisRoom/scripts/s2_box.sh kr > /dev/null 2>&1 < /dev/null &'
#                                         # the KR timing is CPU only: run it beside the chain
#     scripts/s2_fetch.sh kr-s2           # when ~/s2/run.log says done; the idle timer then stops the box
#   On the laptop:
#     .venv/bin/python runs/s2_assent.py canary            # the canary's judge half, if not byte-identical
#     .venv/bin/python runs/s2_assent.py grade qwen3-32b   # then: tally qwen3-32b, eligible qwen3-32b
#     (K2.6 replies come from Foundry: runs/s2_assent.py k26, grade k26, eligible k26)
#     .venv/bin/python runs/s2_run.py heldout              # data/s2/heldout.json, the guard's list
#   Session 1b, the reply cap raised to 2,048 in two rounds (2026-10-12; wait for any running chain to finish first):
#     scripts/s2_launch.sh kr-s2 extend0  # round 1, setup_qwen extend_qwen0: capped draw-0 replies only (about 2.5 h)
#     scripts/s2_fetch.sh kr-s2
#     .venv/bin/python runs/s2_assent.py grade qwen3-32b    # laptop; writes data/s2/replies/qwen3-32b/extend_needed.json
#     scripts/s2_launch.sh kr-s2 extend2  # round 2 (syncs the list), setup_qwen extend_qwen_needed (0.3 to 1.3 h, by list size)
#     scripts/s2_fetch.sh kr-s2           # then grade, eligible qwen3-32b and runs/s2_run.py heldout on the laptop
#   Session 2, the K2.6 box (about 1.8 h):
#     scripts/s2_launch.sh kr-s2 k26      # setup_k26 env_k26 vectors_k26 unsteered_k26 setup_qwen unsteered_qwen
#                                         #   potency_k26 potency_qwen (last, so the exploratory stages never block step 7)
#     scripts/s2_fetch.sh kr-s2
#   Then on the laptop: runs/s2_select.py on both unsteered readouts.csv, runs/s2_battery.py --select.
#
# Stages:
#   setup_qwen      DLAMI python deps and Qwen3-32B at the pinned revision -> /nvme/qwen/Qwen3-32B, peft 0.21.2
#   env_qwen        results/s2/box/qwen_box.json: instance, GPUs, driver, libraries, code and weight SHA-256s
#   moonlight       tests/s2_moonlight_spans.py (spec section 6 test d) -> results/s2/engine/moonlight_spans.log
#   vectors_qwen    runs/s2_vectors.py run, build, gate --pool $S2_POOL (default 12) -> results/qwen/s2
#   merge           Study 1's D2 and D2c adapters (hash-checked) merged on CPU -> /nvme/distill/{D2,D2c}/merged
#   ftvec           section 3, S2-FTD and S2-FTC: controls rebuilt on each merged checkpoint, AIPsy and J1B
#                   gates, cosine with the base controls, diagnostics only -> results/qwen/s2ft/{D2,D2c}
#   assent          runs/s2_assent.py qwen: the determinism canary, then draws 0-4 of every wording
#                   -> data/s2/replies/qwen3-32b (box.json there is the stamped reply box)
#   extend_qwen0    runs/s2_assent.py extend --draws 0: round 1, every capped Qwen draw 0 to 2,048 new tokens on the
#                   reply box's setup (continued, checked against 2 fresh runs, regenerated if they differ)
#                   -> data/s2/replies/qwen3-32b (extend.json, cap1200/)
#   extend_qwen_needed  extend --needed $QR/extend_needed.json: round 2, only the capped sampled draws of the wordings
#                   the laptop's grading left pending, in seeded continuation batches (extend_batches.json)
#   extend_qwen     every capped reply in one go (superseded by the two rounds; kept for reference)
#   kr              Kenward-Roger timing on a full-size synthetic table -> results/s2/kr_timing (kr.csv, timing.txt)
#   setup_k26       Kimi-K2.6 at the pinned revision -> /nvme/k26 (vm_setup.sh)
#   env_k26         results/s2/box/k26_box.json, as env_qwen
#   vectors_k26     as vectors_qwen with --model-key k26 -> results/k26/s2
#   unsteered_k26   runs/s2_run.py --phase unsteered (step 7) -> results/s2/k26/unsteered/S2-DEN
#   unsteered_qwen  the same for Qwen -> results/s2/qwen/unsteered/S2-DEN (needs setup_qwen in the session)
#   potency_k26     runs/s2_run.py potency (exploratory moderator: NLL increase on 48 neutral stories per pattern,
#                   registered and half dose) -> results/s2/k26/potency/potency.csv (needs vectors_k26)
#   potency_qwen    the same for Qwen -> results/s2/qwen/potency (needs setup_qwen and session 1's results/qwen/s2)
set -euo pipefail
cd ~/KimisRoom
S=~/s2; mkdir -p $S/logs
exec >> $S/run.log 2>&1
source /opt/pytorch/bin/activate
PY=/opt/pytorch/bin/python
Q=/nvme/qwen/Qwen3-32B; K=/nvme/k26
QR=data/s2/replies/qwen3-32b
POOL=${S2_POOL:-12}
STORIES=(--stories data/s2/stories/nouns --stories data/s2/stories/thinking)
export PYTHONUNBUFFERED=1
log() { echo "$(date -u +%FT%TZ) $*"; }

case "${1:-}" in
  qwen) set -- setup_qwen env_qwen moonlight vectors_qwen merge ftvec assent ;;
  extend0) set -- setup_qwen extend_qwen0 ;;
  extend2) set -- setup_qwen extend_qwen_needed ;;
  k26) set -- setup_k26 env_k26 vectors_k26 unsteered_k26 setup_qwen unsteered_qwen potency_k26 potency_qwen ;;
  "") echo "usage: $0 qwen|extend0|extend2|k26|STAGE..." >&2; exit 2 ;;
esac

# The per-boot cap (userdata.yaml) is 6 h, too short for the frozen replies: raise it for this boot and the next.
H=${S2_MAX_HOURS:-30}
echo "$H" | sudo tee /etc/kimisroom/max_hours > /dev/null
sudo shutdown -c 2>/dev/null || true; sudo shutdown -h +$(( H * 60 )) 2>/dev/null || true

imds() {  # imds PATH: instance metadata (IMDSv2), empty if it does not answer
  local t; t=$(curl -sf -m 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60') &&
    curl -sf -m 2 -H "X-aws-ec2-metadata-token: $t" "http://169.254.169.254/latest/meta-data/$1" || true
}

shards_ok() {  # shards_ok DIR: every shard the index names is there
  $PY - "$1" << 'EOF'
import json, os, sys
d = sys.argv[1]
shards = set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values())
missing = sorted(s for s in shards if not os.path.exists(os.path.join(d, s)))
assert not missing, f"{d}: missing {missing}"
print(f"{d}: {len(shards)} shards present")
EOF
}

box_json() {  # box_json KEY REPO MODEL_DIR: what stamp A needs about this box, the stack and the weights
  mkdir -p results/s2/box
  local rev
  rev=$(grep -o "$2 --revision [0-9a-f]*" scripts/vm_setup.sh | awk '{print $3}')
  [ -n "$rev" ] || { echo "no pinned revision for $2 in scripts/vm_setup.sh"; return 1; }
  $PY - "$1" "$2" "$rev" "$3" "$(imds instance-type)" "$(imds ami-id)" "$(imds placement/region)" << 'EOF'
import glob, hashlib, json, os, platform, socket, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
key, repo, rev, mdir, itype, ami, region = sys.argv[1:8]

def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()

def version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None

# hf download --local-dir leaves one metadata file per downloaded file; its first line is the commit
commits = set()
for m in glob.glob(os.path.join(mdir, ".cache/huggingface/download/**/*.metadata"), recursive=True):
    with open(m) as f:
        commits.add(f.readline().strip())
files = sorted(os.path.relpath(p, mdir) for p in glob.glob(os.path.join(mdir, "**/*"), recursive=True)
               if os.path.isfile(p) and "/.cache/" not in p and not os.path.basename(p).startswith("."))
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    hashes = dict(zip(files, ex.map(lambda f: sha(os.path.join(mdir, f)), files)))
import torch
code = sorted(glob.glob("runs/s2_*") + glob.glob("kimisroom/*.py") + glob.glob("tests/s2_*") +
              glob.glob("scripts/s2_*.sh") + ["scripts/vm_setup.sh", "scripts/userdata.yaml"])
out = {
    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model_key": key,
    "hf_repo": repo, "hf_revision_pinned": rev, "hf_revision_downloaded": sorted(commits),
    "revision_match": commits == {rev}, "model_dir": mdir,
    "instance_type": itype or None, "ami_id": ami or None, "region": region or None, "hostname": socket.gethostname(),
    "gpus": sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total,pci.bus_id", "--format=csv,noheader").splitlines(),
    "nvidia_smi_header": sh("nvidia-smi").splitlines()[:4],
    "kernel": platform.platform(), "os_release": sh("sh", "-c", ". /etc/os-release; echo $PRETTY_NAME"),
    "python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(), "gpu_count": torch.cuda.device_count(),
    "libraries": {m: version(m) for m in ("transformers", "accelerate", "safetensors", "huggingface_hub", "peft",
                                         "tokenizers", "numpy", "compressed_tensors", "tiktoken")},
    "pip_freeze": sh(sys.executable, "-m", "pip", "freeze").splitlines(),
    "code_sha256": {p: sha(p) for p in code if os.path.isfile(p)},
    "weights_sha256": hashes, "weights_hash_seconds": round(time.time() - t0, 1),
}
path = f"results/s2/box/{key}_box.json"
with open(path + ".tmp", "w") as f:
    json.dump(out, f, indent=1)
os.replace(path + ".tmp", path)
print(f"{path}: {itype} {len(out['gpus'])} GPUs, torch {torch.__version__}, transformers {out['libraries']['transformers']}, "
      f"{repo}@{rev[:8]} (downloaded {sorted(commits)}), {len(hashes)} files hashed in {out['weights_hash_seconds']}s")
assert out["revision_match"], f"downloaded revision {sorted(commits)} is not the pinned {rev}"
EOF
}

vectors() {  # vectors KEY MODEL_DIR: control vectors, the build check and the gates (section 3)
  local o=results/$1/s2 d
  # runs/s2_vectors.py reads stories.jsonl and manifest.json from each --stories dir (runs/s2_stories.py's output)
  for d in data/s2/stories/nouns data/s2/stories/thinking; do
    for f in stories.jsonl manifest.json; do
      [ -s $d/$f ] || { echo "$d/$f missing: generate the stories first (runs/s2_stories.py)"; return 1; }
    done
  done
  [ -f "$2/config.json" ] || { echo "no model at $2: run setup_$1 first"; return 1; }
  [ -f $o/extract.pt ] || $PY -I -u runs/s2_vectors.py run "$2" $o --model-key "$1" "${STORIES[@]}"
  [ -f $o/controls.pt ] || $PY -I -u runs/s2_vectors.py build $o   # exit 2: the build check failed
  $PY -I -u runs/s2_vectors.py gate $o --pool "$POOL"
}

unsteered() {  # unsteered KEY MODEL_DIR REPLIES: step 7, zero-vector groups on S2-DEN pilot openings
  local f
  for f in data/s2/replies/$3/selected.json data/s2/replies/$3/eligible.json data/s2/heldout.json prompts/battery/items_s2.jsonl; do
    [ -s $f ] || { echo "$f missing: grade, tally and eligible ($3) and runs/s2_run.py heldout on the laptop, then sync"; return 1; }
  done
  [ -f "$2/config.json" ] || { echo "no model at $2: run setup_$1 first"; return 1; }
  $PY -u runs/s2_run.py run --model "$1" --arm S2-DEN --phase unsteered --model-path "$2" \
    --out results/s2/$1/unsteered/S2-DEN
}

potency() {  # potency KEY MODEL_DIR: exploratory moderator, NLL increase on neutral text per pattern and dose
  [ -s "results/$1/s2/pool.json" ] || { echo "results/$1/s2/pool.json missing: run vectors_$1 first"; return 1; }
  [ -f "$2/config.json" ] || { echo "no model at $2: run setup_$1 first"; return 1; }
  $PY -u runs/s2_run.py potency --model "$1" --model-path "$2" --out "results/s2/$1/potency"
}

# ---- stages: done_NAME says whether the outputs exist, do_NAME makes them ---------------------------------
done_setup_qwen() { [ -f /nvme/qwen/.s2-ok ]; }
do_setup_qwen() {
  bash scripts/vm_setup.sh qwen
  pip install -q "peft==0.21.2"   # the version the adapters were trained with (confirm_box.sh)
  shards_ok $Q
  touch /nvme/qwen/.s2-ok
}

done_env_qwen() { [ -s results/s2/box/qwen_box.json ]; }
do_env_qwen() { box_json qwen Qwen/Qwen3-32B $Q; }

done_moonlight() { [ -f results/s2/engine/moonlight.pass ]; }
do_moonlight() {
  bash scripts/vm_setup.sh moonlight   # ~/models/moonlight (EBS), pinned revision
  mkdir -p results/s2/engine
  CUDA_VISIBLE_DEVICES=0 $PY -u tests/s2_moonlight_spans.py ~/models/moonlight 2>&1 | tee results/s2/engine/moonlight_spans.log
  # K2.6's runner path end to end on Moonlight (renderer, layer-wise engine, decoded-prefix guard, s2_table)
  CUDA_VISIBLE_DEVICES=0 $PY -u tests/s2_run_check.py --moonlight ~/models/moonlight 2>&1 | tee results/s2/engine/moonlight_runner.log
  touch results/s2/engine/moonlight.pass
}

done_vectors_qwen() { [ -s results/qwen/s2/pool.json ]; }
do_vectors_qwen() { vectors qwen $Q; }

merged() { [ -f /nvme/distill/$1/merged/model.safetensors.index.json ] || [ -s results/qwen/s2ft/$1/diag.json ]; }
done_merge() { merged D2 && merged D2c; }
do_merge() {
  local A=results/qwen/adapters/distill_keep a
  sha256sum -c << EOF
d7270dd9d0109e6d0b84075e7002ee03c686396ff31266819158c2c658c43bce  $A/D2/adapter/adapter_model.safetensors
c50736203ca7ef34aa8b1b5b65d21fc108a9fbda46ac3acbf3081337e7b705f9  $A/D2c/adapter/adapter_model.safetensors
036ef412ecb8c6bec11ab8b8a8978276fc27e55db77eb81c9222f80fdbd0bdc1  $A/D2/adapter/adapter_config.json
aaa3180403d31ec67093bc25ae2c97f3ddfdd01fb06fb8063bb3be17687cc05b  $A/D2c/adapter/adapter_config.json
EOF
  for a in D2 D2c; do
    merged $a && continue
    mkdir -p /nvme/distill/$a && cp -r $A/$a/adapter $A/$a/meta.json /nvme/distill/$a/
    $PY -u runs/distill.py merge --model $Q --out /nvme/distill --arm $a
  done
}

done_ftvec() { [ -s results/qwen/s2ft/D2/diag.json ] && [ -s results/qwen/s2ft/D2c/diag.json ]; }
do_ftvec() {
  [ -f results/qwen/s2/controls.pt ] || { echo "no results/qwen/s2/controls.pt: run vectors_qwen first"; return 1; }
  local a m o
  for a in D2 D2c; do
    m=/nvme/distill/$a/merged; o=results/qwen/s2ft/$a
    [ -s $o/diag.json ] && continue
    [ -f $m/model.safetensors.index.json ] || { echo "no merged $a checkpoint at $m: run merge first"; return 1; }
    mkdir -p $o
    if [ ! -s $o/ckpt.sha256 ]; then
      (cd $m && find . -type f -not -path './.cache/*' | sort | xargs -P8 -n1 sha256sum | sort -k2) > $o/ckpt.sha256.tmp
      mv $o/ckpt.sha256.tmp $o/ckpt.sha256
    fi
    [ -f $o/extract.pt ] || $PY -I -u runs/s2_vectors.py run $m $o --model-key qwen "${STORIES[@]}"
    # Diagnostics only (spec section 3): on merged weights the rebuilt red, ocean and arithmetic are expected to
    # differ from the registered file, so `ftdiag` records the build check without stopping, then builds the
    # controls, runs the same gates and writes their cosine with the base controls to diag.json.
    $PY -I -u runs/s2_vectors.py ftdiag $o --base results/qwen/s2 --pool "$POOL"
  done
}

done_assent() { [ -s $QR/replies.sha256.json ] && grep -q '"finished"' $QR/box.json; }
do_assent() { $PY -u runs/s2_assent.py qwen $Q; }   # the canary stops it (exit 1) before any opening if it fails

done_extend_qwen() { [ -s $QR/extend.json ] && grep -q '"finished"' $QR/extend.json; }
do_extend_qwen() {
  [ -f $Q/config.json ] || { echo "no model at $Q: run setup_qwen first"; return 1; }
  [ -s $QR/box.json ] || { echo "$QR/box.json missing: the replies to extend are not on this box"; return 1; }
  $PY -u runs/s2_assent.py extend $Q   # stops (exit 1) if the box or stack differs from $QR/box.json
}

done_extend_qwen0() { [ -s $QR/extend.json ] && grep -q '"round1_finished"' $QR/extend.json; }
do_extend_qwen0() {
  [ -f $Q/config.json ] || { echo "no model at $Q: run setup_qwen first"; return 1; }
  [ -s $QR/box.json ] || { echo "$QR/box.json missing: the replies to extend are not on this box"; return 1; }
  $PY -u runs/s2_assent.py extend $Q --draws 0
}

done_extend_qwen_needed() { [ -s $QR/extend.json ] && grep -q '"round2_finished"' $QR/extend.json; }
do_extend_qwen_needed() {
  [ -f $Q/config.json ] || { echo "no model at $Q: run setup_qwen first"; return 1; }
  [ -s $QR/extend_needed.json ] || { echo "$QR/extend_needed.json missing: grade qwen3-32b on the laptop and sync"; return 1; }
  $PY -u runs/s2_assent.py extend $Q --needed $QR/extend_needed.json   # refuses a list stale against the verdicts
}

r_deps() {  # GNU time and R >= 4.3 with lme4, lmerTest and pbkrtest (runs/s2_confirm.R --kr)
  local need='Filter(function(p) !requireNamespace(p, quietly = TRUE), c("lme4", "lmerTest", "pbkrtest"))'
  Rscript -e "stopifnot(getRversion() >= '4.3', !length($need))" > /dev/null 2>&1 && [ -x /usr/bin/time ] && return 0
  # The box image is the DLAMI on Ubuntu 24.04 (scripts/vm_up.sh). Its universe archive has R 4.3.3 and a matching
  # set: r-cran-lme4 1.1-35.1, r-cran-lmertest 3.1-3, r-cran-pbkrtest 0.5.2, r-cran-matrix 1.6-5 (checked on
  # packages.ubuntu.com 2026-10-11). Current CRAN is not a substitute on R 4.3: pbkrtest needs doBy, whose Deriv
  # needs R >= 4.5, and Matrix 1.7 needs R >= 4.4. Whatever apt leaves missing is tried from Posit's Ubuntu 24.04
  # binaries, then CRAN, and the stage stops if a package is still missing.
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q time r-base-core r-recommended \
    r-cran-lme4 r-cran-lmertest r-cran-pbkrtest
  sudo Rscript -e "
    options(Ncpus = parallel::detectCores(), HTTPUserAgent = sprintf('R/%s R (%s)', getRversion(),
      paste(getRversion(), R.version[['platform']], R.version[['arch']], R.version[['os']])))
    need <- function() $need
    for (repo in c('https://packagemanager.posit.co/cran/__linux__/noble/latest', 'https://cloud.r-project.org')) {
      if (!length(need())) break
      try(install.packages(need(), repos = repo, lib = .Library.site[1]))
    }
    stopifnot(getRversion() >= '4.3', !length(need()))"
}

done_kr() { [ -s results/s2/kr_timing/kr.csv ] && [ -s results/s2/kr_timing/timing.txt ]; }
do_kr() {
  local k=results/s2/kr_timing
  r_deps
  mkdir -p $k
  Rscript -e 'stopifnot(getRversion() >= "4.3"); cat(R.version.string, "\n");
    for (p in c("lme4", "lmerTest", "pbkrtest", "Matrix")) cat(p, format(packageVersion(p)), "\n")' > $k/versions.txt
  [ -s $k/synth.csv ] || $PY -u tests/s2_synth.py $k/synth.csv --models qwen --arms S2-DEN --n-args 10 --n-wordings 5 --n-mp 0
  [ -s $k/table.csv ] || $PY -u runs/s2_table.py --phase confirm --out $k/table.csv $k/synth.csv
  # The box's idle timer counts only python processes as work, so R runs under a python parent.
  /usr/bin/time -v -o $k/time_v.txt $PY -c 'import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))' \
    Rscript runs/s2_confirm.R $k $k/table.csv --only-h1 --kr --k26 abandoned
  { echo "Kenward-Roger timing: Rscript runs/s2_confirm.R OUT table.csv --only-h1 --kr --k26 abandoned"
    echo "box: $(imds instance-type), $(nproc) vCPU, $(free -g | awk '/Mem:/{print $2}') GiB RAM; single-threaded BLAS (the script sets it)"
    echo "synthetic readouts: $(( $(wc -l < $k/synth.csv) - 1 )) rows; table: $(( $(wc -l < $k/table.csv) - 1 )) rows"
    grep -E 'Elapsed \(wall clock\)|Maximum resident set size' $k/time_v.txt
    cat $k/versions.txt; } > $k/timing.txt
  cat $k/timing.txt
}

done_setup_k26() { [ -f $K/.s2-ok ]; }
do_setup_k26() {
  bash scripts/vm_setup.sh k26   # moonshotai/Kimi-K2.6 at the revision the tokenizer is pinned to
  shards_ok $K
  touch $K/.s2-ok
}

done_env_k26() { [ -s results/s2/box/k26_box.json ]; }
do_env_k26() { box_json k26 moonshotai/Kimi-K2.6 $K; }

done_vectors_k26() { [ -s results/k26/s2/pool.json ]; }
do_vectors_k26() { vectors k26 $K; }

done_unsteered_k26() { [ -s results/s2/k26/unsteered/S2-DEN/readouts.csv ]; }
do_unsteered_k26() { unsteered k26 $K k26; }

done_unsteered_qwen() { [ -s results/s2/qwen/unsteered/S2-DEN/readouts.csv ]; }
do_unsteered_qwen() { unsteered qwen $Q qwen3-32b; }

done_potency_k26() { [ -s results/s2/k26/potency/potency.csv ]; }
do_potency_k26() { potency k26 $K; }

done_potency_qwen() { [ -s results/s2/qwen/potency/potency.csv ]; }
do_potency_qwen() { potency qwen $Q; }

# ---- runner -----------------------------------------------------------------------------------------------
for s in "$@"; do
  declare -F "do_$s" > /dev/null || { log "unknown stage $s"; exit 2; }
done
stage() {  # stage NAME: skip if its outputs exist, else run do_NAME into its log; stop the chain on failure
  local s=$1 t0 rc
  if "done_$s"; then log "$s: outputs exist, skipped"; return 0; fi
  log "$s: start"; t0=$(date +%s)
  echo "==== $(date -u +%FT%TZ) $s" >> $S/logs/$s.log
  # Not "if do_$s": inside an if condition bash ignores set -e for the whole function (glm_box_run.sh).
  set +e
  ( set -e; "do_$s" ) >> $S/logs/$s.log 2>&1
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    log "$s: ok in $(( $(date +%s) - t0 ))s"
  else
    log "$s: FAILED (rc $rc) after $(( $(date +%s) - t0 ))s; tail of $S/logs/$s.log:"; tail -25 $S/logs/$s.log
    exit 1
  fi
}
log "start: $* (pid $$)"
for s in "$@"; do stage "$s"; done
log "done: $*"
