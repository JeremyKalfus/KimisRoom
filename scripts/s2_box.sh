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
#   Session 3, the pilot (spec section 8; after stamp A, prereg/stampA-s2.txt and its .ots; about 4 h, 3.4 to 4.6 h
#   at Qwen's measured 1,700 to 2,500 tok/s, so $36 to $48, plus about 0.6 h if K2.6 needs its half dose; the box
#   stops itself after 12 h for this chain):
#     scripts/s2_launch.sh kr-s2 pilot    # setup_qwen env_qwen pilot_qwen merge pilot_ft setup_k26 env_k26 pilot_k26
#                                         #   pilot_k26_half
#     scripts/s2_fetch.sh kr-s2           # when ~/s2/run.log says done; then the laptop commands at the end of this header
#   Order: the Qwen base arms first, since rules 1 to 3 read them for both models (K2.6 borrows Qwen's power and
#   noun count); the fine-tuned arms (diagnostics only) next, as they need the merge; K2.6 last: its path (layer-wise
#   engine, about 600 GB staged) is the riskier, a failure there loses nothing already run, and rule 1's half-dose
#   follow-up then runs while its weights are still on /nvme. In any order the hours are the same when every stage
#   passes; finished arms are kept on EBS, and a rerun skips them (the setup stages restage their weights).
#   Session 4, rule 3's added nouns (before stamp B; rule 1's decisions and rule 3's count are made on the laptop, and
#   scripts/s2_sync.sh pushes results/s2/pilot/*decision*.json; about 1 h with weight staging, $11; the box stops
#   itself after 6 h):
#     scripts/s2_launch.sh kr-s2 nouns    # setup_qwen env_qwen nouns_qwen setup_k26 env_k26 nouns_k26
#     scripts/s2_fetch.sh kr-s2           # results/s2/{qwen,k26}/nouns/nouns.json: pass or fail per noun, the pool
#   Order: Qwen first, because K2.6 adds nouns until it has as many as Qwen after rule 3 (nouns_k26 refuses without
#   Qwen's finished nouns.json) and Qwen's spare rounds need its weights still on /nvme; K2.6 last, as in the pilot.
#   Then on the laptop: fill PREREG-s2.md's two STAMP B BLANKs, commit them with DEVIATIONS-s2.md, the decision files
#   and both nouns.json (the confirmatory stages read those and refuse them unless stamp B lists them), then
#   scripts/s2_manifest.sh COMMIT, review, scripts/s2_manifest.sh COMMIT --stamp (writes and commits
#   prereg/manifest-s2.txt and its .ots).
#   Session 5, the confirmatory run (after stamp B; about 9.4 h at Study 1's rates, $98, plus about 0.3 h of weight
#   staging and merging; 13 to 14 h at the pilot's measured end-to-end rates; then carry_qwen, about 0.9 h and $10
#   more at Study 1's rate, 1.4 h at the pilot's slower end; the box stops itself after 24 h):
#     scripts/s2_launch.sh kr-s2 confirm  # setup_qwen env_qwen confirm_qwen merge confirm_ft setup_k26 env_k26
#                                         #   confirm_k26 carry_qwen
#     scripts/s2_fetch.sh kr-s2
#   Order: Qwen's base arms first (S2-DEN and S2-CON carry S2-H1), the fine-tuned arms after the merge, K2.6 after
#   them, as in the pilot, and the exploratory carry-over run and D1.4 bridge (carry_qwen) last, so they can never
#   delay or displace a registered run. Qwen's weights are still on /nvme then (setup_k26 adds K2.6 beside them).
#
# Stages:
#   setup_qwen      DLAMI python deps and Qwen3-32B at the pinned revision -> /nvme/qwen/Qwen3-32B, peft 0.21.2
#   env_qwen        results/s2/box/qwen_box.json: instance, GPUs, driver, libraries, code and weight SHA-256s
#   moonlight       tests/s2_moonlight_spans.py (spec section 6 test d) -> results/s2/engine/moonlight_spans.log
#   vectors_qwen    runs/s2_vectors.py run, build, gate --pool $S2_POOL (default 12) -> results/qwen/s2
#   merge           Study 1's D2 and D2c adapters (hash-checked) merged on CPU -> /nvme/distill/{D2,D2c}/merged; skipped
#                   only when the merged checkpoints are on /nvme or every stage that reads them (ftvec, pilot_ft,
#                   confirm_ft) is done
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
# Pilot stages (spec section 8, PREREG-s2.md section 6). Each first checks, against prereg/stampA-s2.txt, the SHA-256 of
# every stamped input the runner reads (code, items_s2.jsonl, openings, held-out list, reply selection, controls.pt and
# pool.json; scripts/s2_sync.sh pushes the last two, which git does not hold) and that the held-out guard loads the
# list stamp A lists; either failing stops the stage. Item file prompts/battery/items_s2.jsonl (the selected battery); vectors the runner's pilot
# set (registered emotions, the 12 sampled nouns, arithmetic, random, all 8 mental-process controls; emotion file
# hash-asserted); doses in natural units. Groups and tokens from runs/s2_run.py --dry-run; hours at the measured rates.
#   pilot_qwen      S2-DEN (12 openings) and S2-CON (11) on the pinned base weights, doses 2 and 1: controls and the
#                   zero vector on the 256-item battery, emotions and the gate-only zero check on the 100 gate items
#                   -> results/s2/qwen/pilot/{S2-DEN,S2-CON} (696 + 638 groups, 11.4M tokens, 1.3 to 1.9 h)
#   pilot_ft        the same for S2-FTD and S2-FTC on the merged D2 and D2c checkpoints (needs merge), control-argument
#                   pilot openings with the base-Qwen replies -> results/s2/qwen/pilot/{S2-FTD,S2-FTC} (638 + 638
#                   groups, 10.9M tokens, 1.2 to 1.8 h); checkpoints.json there refuses either on the base's or the
#                   other's hash
#   pilot_k26       K2.6 S2-DEN (8 openings) and S2-CON (11): every vector and the zero vector on the gate items at the
#                   registered 4 units -> results/s2/k26/pilot/{S2-DEN,S2-CON} (216 + 297 groups, 2.2M tokens, about
#                   0.6 h with the per-arm weight hashing and layer streaming). K2.6 has 8 clean S2-DEN pilot openings,
#                   not 12, so this is 19 x 27 groups, not the spec's 23 x 27 = 621
#   pilot_k26_half  rule 1 on those rows (runs/s2_table.py --decision -> results/s2/k26/pilot/rule1/); only if it sends
#                   K2.6 to half dose, the same two arms at 2 units -> results/s2/k26/pilot_half/{S2-DEN,S2-CON} (513
#                   groups, about 0.6 h), else rule1/half_not_needed
# Noun gate stages (rule 3's added nouns, PREREG-s2.md section 6; before stamp B): the pilot stages' stamp A and guard
# check (and results/KEY/s2/gates.csv against stamp A), then rounds of gate-item runs at rule 1's dose until the pool
# reaches its target or the candidates run out (noun_gates below). Groups and tokens from the dry runs; hours at Study
# 1's rates (Qwen 2,523 tok/s, K2.6 2,632), before weight loading, hashing and K2.6's layer streaming.
#   nouns_qwen      qwen_decision.json's dose (1 unit), candidates N15..N26 then spares N29, N31, N34, N35, N36, on the 12 + 11
#                   pilot openings -> results/s2/qwen/nouns/{r1,r2,...}/ and nouns.json (round 1: 322 groups, 1.65M
#                   tokens, 0.18 h; each failing noun adds a spare round of 23 groups a noun plus 46 zero groups)
#   nouns_k26       k26_half_decision.json's dose (2 units), candidates N19..N23, N25, N26, N29, N31, N34, N35, N36 (no spares left), on
#                   the 8 + 11 pilot openings, target Qwen's count -> results/s2/k26/nouns (247 groups, 1.08M tokens,
#                   0.11 h; the pilot's K2.6 arms took about 15 min each whatever their size)
# Confirmatory stages (after stamp B; PREREG-s2.md sections 6 and 7): every input checked against stamp A and stamp B,
# the guard must report stamp B, then each kept argument's clean held-out wordings on the 216-item confirmatory
# battery at rule 1's dose: rule 1's surviving emotions, the primary controls (rule 1's survivors plus the nouns
# rule 3 added), the mental-process controls that passed gates (a) to (c) and rule 1, red and ocean in S2-DEN only
# (final_vecs below; results/s2/KEY/confirm/vectors.json). At nC 26:
#   confirm_qwen    S2-DEN (49 openings, 42 vectors + zero) and S2-CON (47, 40 + zero) on the pinned base weights
#                   -> results/s2/qwen/confirm/{S2-DEN,S2-CON} (2,107 + 1,927 groups, 34.2M tokens, 3.8 h)
#   confirm_ft      S2-FTD and S2-FTC on the merged D2 and D2c checkpoints (needs merge), base-Qwen replies, 47
#                   openings each -> results/s2/qwen/confirm/{S2-FTD,S2-FTC} (2 x 1,927 groups, 32.4M tokens, 3.6 h)
#   confirm_k26     S2-DEN (29 openings, 36 vectors + zero) and S2-CON (44, 34 + zero) -> results/s2/k26/confirm
#                   (1,073 + 1,540 groups, 19.2M tokens, 2.0 h)
#   carry_qwen      exploratory, after confirm_qwen (PREREG-s2.md section 5): runs/s2_carry.py carry, Study 1's
#                   carry-over design on the 49 held-out S2-DEN openings (a Study 1 task each in rotation, Study 1's
#                   10 vectors at 2 units on the answer, its 314-item battery) -> results/s2/qwen/carry/carryover
#                   (539 groups, 8.2M tokens, 0.9 h; table.csv by runs/confirm_table.py); runs/s2_carry.py bridge,
#                   Study 2's design on the D1.4 opening with the confirmatory S2-DEN vectors and dose, labelled
#                   S2-BRIDGE -> results/s2/qwen/carry/bridge (43 groups, 0.35M tokens, 2 min); runs/s2_carry.py
#                   table -> bridge/table.csv and summary.csv (emotion minus control on D1.4 beside each denial
#                   argument's, from the S2-DEN readouts)
#
# Laptop dry run of these stages (no GPU, no weights): a clone of the repo with mock model folders (config.json and
# the pinned tokenizers from data/tokenizers), this script with /nvme, ~/KimisRoom, ~/s2 and PY pointed into the
# clone (sudo and the venv lines dropped), and a PY wrapper that adds --dry-run to every runs/s2_run.py run call; for
# the noun stages the wrapper also writes readouts.csv from the real pilot rows relabelled, so the rounds, rule 1 and
# the spare logic run end to end. The confirmatory stages need a stamp B there: commit in the clone, run
# scripts/s2_manifest.sh on that commit, and write a .ots header carrying the manifest's SHA-256.
#
# After the pilot, on the laptop (the scripts' own usage lines; the rules are PREREG-s2.md section 6):
#   P=results/s2/pilot; mkdir -p $P
#   .venv/bin/python runs/s2_table.py --phase pilot --out $P/qwen_table.csv --gates $P/qwen_gates.csv \
#       --decision $P/qwen_decision.json results/s2/qwen/pilot/S2-{DEN,CON,FTD,FTC}/readouts.csv   # Qwen rules 1, 2
#   .venv/bin/python runs/s2_table.py --phase pilot --out $P/k26_table.csv --gates $P/k26_gates.csv \
#       --decision $P/k26_decision.json results/s2/k26/pilot/S2-{DEN,CON}/readouts.csv             # K2.6 rule 1
#     If pilot_k26_half ran, the same on results/s2/k26/pilot_half/S2-{DEN,CON}/readouts.csv into $P/k26_half_*: its
#     decision lists the vectors that survive at half dose. Not in one call with the registered-dose files: both
#     runs carry a zero-vector group per opening, which s2_table.py refuses as duplicate readouts.
#   Rscript runs/s2_power.R $P/power $P/qwen_table.csv --prior-sd 0.102 --dose D --n-mp N --stop    # rule 3
#     D: qwen_decision.json "dose"; N: the number of its "thinking" survivors; 0.102 is stamp A's prior control SD
#     (PREREG-s2.md section 3); --reps 500, --nc 14:26 and --design 10x5 are the defaults.
#   runs/s2_confirm.R takes confirmatory tables only (it refuses a pilot table); it runs after stamp B.
set -euo pipefail
cd ~/KimisRoom
S=~/s2; mkdir -p $S/logs
exec >> $S/run.log 2>&1
source /opt/pytorch/bin/activate
PY=/opt/pytorch/bin/python
Q=/nvme/qwen/Qwen3-32B; K=/nvme/k26; DIST=/nvme/distill
QR=data/s2/replies/qwen3-32b
POOL=${S2_POOL:-12}
STORIES=(--stories data/s2/stories/nouns --stories data/s2/stories/thinking)
ITEMS=prompts/battery/items_s2.jsonl          # the selected battery (stamp A)
PQ=results/s2/qwen/pilot; PK=results/s2/k26/pilot
NQ=results/s2/qwen/nouns; NK=results/s2/k26/nouns       # rule 3's added nouns (noun gate stages)
CQ=results/s2/qwen/confirm; CK=results/s2/k26/confirm   # confirmatory readouts (after stamp B)
# Rule 1's decisions, made on the laptop from the pilot rows (s2_sync.sh pushes them; results/ is not synced otherwise)
QDEC=${S2_QWEN_DECISION:-results/s2/pilot/qwen_decision.json}
KDEC=${S2_K26_DECISION:-results/s2/pilot/k26_half_decision.json}
export PYTHONUNBUFFERED=1
log() { echo "$(date -u +%FT%TZ) $*"; }

case "${1:-}" in
  qwen) set -- setup_qwen env_qwen moonlight vectors_qwen merge ftvec assent ;;
  extend0) set -- setup_qwen extend_qwen0 ;;
  extend2) set -- setup_qwen extend_qwen_needed ;;
  k26) set -- setup_k26 env_k26 vectors_k26 unsteered_k26 setup_qwen unsteered_qwen potency_k26 potency_qwen ;;
  pilot) set -- setup_qwen env_qwen pilot_qwen merge pilot_ft setup_k26 env_k26 pilot_k26 pilot_k26_half; HDEF=12 ;;
  nouns) set -- setup_qwen env_qwen nouns_qwen setup_k26 env_k26 nouns_k26; HDEF=6 ;;
  confirm) set -- setup_qwen env_qwen confirm_qwen merge confirm_ft setup_k26 env_k26 confirm_k26 carry_qwen; HDEF=24 ;;
  "") echo "usage: $0 qwen|extend0|extend2|k26|pilot|nouns|confirm|STAGE..." >&2; exit 2 ;;
esac

# The per-boot cap (userdata.yaml) is 6 h, too short for the frozen replies: raise it for this boot and the next
# (30 h; 12 h for the pilot, about 4 to 5 h of work, so a hung run cannot keep the box up for a day).
H=${S2_MAX_HOURS:-${HDEF:-30}}
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

listed() {  # listed MANIFEST NAME PATH...: every MANIFEST line for these paths (a trailing / takes a directory) matches here
  local a=$1 name=$2 n p; shift 2
  awk -v p="$*" 'BEGIN { n = split(p, P, " ") }
    !/^#/ && NF == 2 && length($1) == 64 && $1 ~ /^[0-9a-f]+$/ {
      for (i = 1; i <= n; i++) if ($2 == P[i] || (P[i] ~ /\/$/ && index($2, P[i]) == 1)) { print; next } }' $a > $S/stamped.txt
  for p in "$@"; do
    grep -qF "  $p" $S/stamped.txt || { echo "$name lists nothing for $p"; return 1; }
  done
  n=$(wc -l < $S/stamped.txt)
  sha256sum -c --quiet $S/stamped.txt || { echo "the files above differ from $name ($a)"; return 1; }
  echo "$name: $n files match ($*)"
}

stamped() {  # stamped PATH...: every prereg/stampA-s2.txt line for these paths matches here
  local a=prereg/stampA-s2.txt
  { [ -s $a ] && [ -s $a.ots ]; } || { echo "$a or $a.ots missing: stamp A comes before any pilot trial"; return 1; }
  listed $a "stamp A" "$@"
}

stamped_b() {  # stamped_b PATH...: the same against stamp B (scripts/s2_manifest.sh: every file of the stamp B commit,
  # hashed from the commit, and the frozen files outside git), so a run reads only what stamp B fixed
  local b=prereg/manifest-s2.txt
  { [ -s $b ] && [ -s $b.ots ]; } || { echo "$b or $b.ots missing: the confirmatory run comes after stamp B"; return 1; }
  listed $b "stamp B" "$@"
}

guard_ok() {  # the held-out guard loads data/s2/heldout.json only as the list stamp A lists (else the run stops)
  $PY - << 'EOF'
from kimisroom.s2_guard import Guard
g = Guard()  # raises HeldOutError if stamp A is incomplete, its .ots digest differs, or it lists another list
print(g.log_line())
assert g.stamp_a_status.startswith("stamp A ") and "lists this held-out list" in g.stamp_a_status, (
    f"held-out guard is not checking the list against stamp A: {g.stamp_a_status}")
EOF
}

guard_b_ok() {  # the guard opens the held-out openings: stamp B's manifest matches its .ots and lists every held-out hash
  $PY - << 'EOF'
from kimisroom.s2_guard import Guard
g = Guard()
print(g.log_line())
assert g.open, f"stamp B does not hold, so the held-out openings stay closed: {g.status}"
EOF
}

s2_inputs() {  # s2_inputs KEY REPLIES: IN=(the stamped inputs one model's steered runs read)
  IN=(runs/s2_run.py runs/s2_battery.py runs/s2_table.py runs/s2_assent.py kimisroom/engine.py
      kimisroom/hf_engine.py kimisroom/trials.py kimisroom/run.py kimisroom/s2_guard.py "$ITEMS"
      prompts/s2/openings/lines.json data/s2/openings/ data/s2/heldout.json
      "data/s2/replies/$2/selected.json" "data/s2/replies/$2/eligible.json" "results/$1/s2/controls.pt"
      "results/$1/s2/pool.json")
}

pilot_inputs() {  # pilot_inputs KEY REPLIES: the stamped inputs one model's pilot runs read, and the guard
  s2_inputs "$1" "$2"
  stamped "${IN[@]}"
  guard_ok
}

steer() {  # steer KEY ARM PHASE MODEL_DIR OUT [ARGS...]: one runner invocation (one model, arm, phase, checkpoint) unless done
  local m=$1 arm=$2 ph=$3 d=$4 o=$5; shift 5
  [ -s $o/readouts.csv ] && { echo "$o/readouts.csv exists: $arm done"; return 0; }
  [ -f "$d/config.json" ] || { echo "no checkpoint at $d"; return 1; }
  $PY -u runs/s2_run.py run --model $m --arm $arm --phase $ph --model-path "$d" --out $o "$@"
}

pilot() {  # pilot KEY ARM MODEL_DIR OUT [ARGS...]: one pilot invocation on the selected battery
  local m=$1 arm=$2 d=$3 o=$4; shift 4
  steer $m $arm pilot "$d" $o --items $ITEMS "$@"
}

decision_of() { case $1 in qwen) echo "$QDEC" ;; k26) echo "$KDEC" ;; esac; }

# Rule 3's additions (PREREG-s2.md section 6): each added noun must pass gates (a) to (c) and rule 1 in a known-answer
# and gate-item run at the confirmatory dose before stamp B. Gates (a) to (c) ran before stamp A
# (results/KEY/s2/gates.csv, stamped); the candidates are the nouns the pool did not reach (pool.json nouns_not_reached)
# that passed them, in list order. Each round runs the next nouns still needed on the pilot openings of S2-DEN and
# S2-CON, every one on the 100 gate items at rule 1's dose with the zero vector, and runs/s2_table.py (frozen in stamp
# A) applies rule 1 to those rows alone (--gates: one row per noun and dose, `passes`). A noun that fails is replaced
# by the next candidate in the following round, in the same session. Target: Qwen to nC = $S2_NC primary controls
# (default 26, rule 3's cap: the power run left power below 0.80; checked against results/s2/pilot/power/timing.csv
# when that is on the box), so 24 sampled nouns; K2.6 to as many sampled nouns as Qwen has after the rule, so it needs
# Qwen's finished nouns.json. A list that runs out leaves the pool at the size reached (section 3).
noun_plan() {  # noun_plan KEY ROUND OUTDIR DECISION: prints "DOSE NOUN,NOUN,..." for ROUND, or nothing when done;
  # writes OUTDIR/rROUND/round.json and OUTDIR/nouns.json (finished_at once nothing is left to run)
  $PY -I - "$@" << 'EOF'
import csv, hashlib, json, os, sys, time
from pathlib import Path

m, r, o, dec_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def die(msg):
    sys.exit(f"noun gates ({m}): {msg}")


if not dec_path.exists():
    die(f"{dec_path} missing: rule 1's decision comes from the laptop (scripts/s2_sync.sh pushes it)")
dec = json.loads(dec_path.read_text()).get(m) or die(f"{dec_path} has no {m!r} entry")
dose = float(dec["dose"])
pool_p, gates_p = Path(f"results/{m}/s2/pool.json"), Path(f"results/{m}/s2/gates.csv")
pool = json.loads(pool_p.read_text())
ac = {x["control"]: x for x in csv.DictReader(gates_p.open())}
cands = [n for n in pool["nouns_not_reached"] if ac.get(n, {}).get("pass") == "True"]
have = len(pool["nouns_chosen"])
if set(cands) & set(dec["primary_controls"]):
    die(f"candidates {sorted(set(cands) & set(dec['primary_controls']))} are already in the decision's pool")
if m == "qwen":
    nc = int(os.environ.get("S2_NC", "26"))
    pw = Path("results/s2/pilot/power/timing.csv")
    if pw.exists():
        ch = next(csv.DictReader(pw.open()))["chosen_nC"]
        want = 26 if ch in ("NA", "") else int(float(ch))
        if want != nc:
            die(f"S2_NC is {nc} but rule 3's power run ({pw}, chosen_nC {ch}) gives nC {want}")
    target, basis = nc - 2, f"nC {nc} (S2_NC; 26 is rule 3's cap)" + (f", agrees with {pw}" if pw.exists() else "")
else:
    qp = Path("results/s2/qwen/nouns/nouns.json")
    q = json.loads(qp.read_text()) if qp.exists() else {}
    if "finished_at" not in q:
        die(f"{qp} is not finished: K2.6 adds nouns until it has as many as Qwen after rule 3 (run nouns_qwen first)")
    target, basis = q["pool_after_rule3"], f"Qwen's sampled nouns after rule 3 ({qp}: {q['pool_after_rule3']})"
target = min(target, 24)
need = max(0, target - have)

per, rounds = {}, []
for k in range(1, r):
    rd = o / f"r{k}"
    ran = json.loads((rd / "round.json").read_text())["nouns"]
    g = rd / "gates.csv"
    if not g.exists():
        die(f"{g} missing: round {k} did not finish")
    got = {x["vec"]: x for x in csv.DictReader(g.open()) if x["vclass"] == "noun"}
    if sorted(got) != sorted(ran):
        die(f"{g} has nouns {sorted(got)}, round {k} ran {ran}")
    for n in ran:
        x = got[n]
        if abs(float(x["dose"]) - dose) > 1e-9:
            die(f"{g}: {n} ran at dose {x['dose']}, rule 1's dose is {dose:g}")
        per[n] = {"round": k, "passes": x["passes"] == "True", "n_ka": int(x["n_ka"]),
                  **{c: float(x[c]) for c in ("ka_acc", "gate_mass", "track_rocks", "track_humans")}}
    rounds.append({"round": k, "nouns": ran, "dir": str(rd)})
passed = [n for n in cands if n in per and per[n]["passes"]]
failed = [n for n in cands if n in per and not per[n]["passes"]]
nxt = [n for n in cands if n not in per][:max(0, need - len(passed))]
if nxt:
    rj = o / f"r{r}/round.json"
    if rj.exists() and json.loads(rj.read_text())["nouns"] != nxt:
        die(f"{rj} ran {json.loads(rj.read_text())['nouns']}, the plan now gives {nxt}: inputs changed between runs")
    rj.parent.mkdir(parents=True, exist_ok=True)
    rj.write_text(json.dumps({"round": r, "model": m, "dose": dose, "nouns": nxt}, indent=1))
order = lambda v: (0, int(v[1:])) if v[:1] == "N" else (1, ("arithmetic", "random").index(v))  # noqa: E731
out = {
    "model": m, "rule": "PREREG-s2.md section 6, rule 3: gates (a) to (c) before stamp A, then rule 1 at the "
    "confirmatory dose in a gate-item run (runs/s2_table.py --gates, `passes`)",
    "dose": dose, "decision_file": str(dec_path), "decision_sha256": sha(dec_path),
    "pool_file": str(pool_p), "pool_sha256": sha(pool_p), "gates_ac_file": str(gates_p), "gates_ac_sha256": sha(gates_p),
    "target_basis": basis, "pool_before": have, "pool_target": target, "to_add": need, "candidates": cands,
    "rounds": rounds, "per_noun": per, "added": passed, "failed": failed, "untried": [n for n in cands if n not in per],
    "pool_after_rule3": have + len(passed),
    "primary_controls_final": sorted(set(dec["primary_controls"]) | set(passed), key=order),
    "status": "running" if nxt else ("complete" if len(passed) >= need else "list exhausted: pool at the size reached"),
}
if not nxt:
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
(o / "nouns.json").write_text(json.dumps(out, indent=1))
for n in per:
    p = per[n]
    print(f"{m} {n} round {p['round']}: {'pass' if p['passes'] else 'FAIL'} (known-answer {p['ka_acc']:.3f}, mass "
          f"{p['gate_mass']:.3f}, rocks {p['track_rocks']:.3f}, humans {p['track_humans']:.3f})", file=sys.stderr)
print(f"{m}: target {target} sampled nouns ({basis}); had {have}, added {len(passed)} {passed}, failed {failed}; "
      + (f"round {r}: {nxt} at {dose:g} units" if nxt else f"{out['status']}: nC {len(out['primary_controls_final'])}"),
      file=sys.stderr)
if nxt:
    print(f"{dose:g} {','.join(nxt)}")
EOF
}

noun_gates() {  # noun_gates KEY MODEL_DIR: rule 3's added nouns, round after round until the pool is full or the list ends
  local m=$1 d=$2 o=results/s2/$1/nouns r=1 v rd arm items=$ITEMS
  stamped results/$m/s2/gates.csv
  mkdir -p $o
  if [ $m = qwen ]; then
    # In the pilot phase the frozen runner steers Qwen's controls on the whole battery it is given; rule 3 asks for
    # gate items only, so Qwen's rounds get the stamped battery's 100 gate rows (gate flag, file order) as the battery.
    items=$o/items_gate.jsonl
    $PY -I - $ITEMS $items << 'EOF'
import json, sys
rows = [x for x in open(sys.argv[1]) if x.strip() and json.loads(x)["gate"]]
open(sys.argv[2], "w").writelines(rows)
print(f"{sys.argv[2]}: the {len(rows)} gate rows of {sys.argv[1]}")
EOF
  fi
  while :; do
    v=$(noun_plan $m $r $o "$(decision_of $m)")
    [ -n "$v" ] || break
    rd=$o/r$r
    for arm in S2-DEN S2-CON; do steer $m $arm pilot "$d" $rd/$arm --items $items --vecs "${v#* }" --doses "${v%% *}"; done
    $PY -u runs/s2_table.py --phase pilot --out $rd/table.csv --gates $rd/gates.csv --decision $rd/decision.json \
      $rd/S2-DEN/readouts.csv $rd/S2-CON/readouts.csv
    r=$(( r + 1 ))
  done
  grep -q '"finished_at"' $o/nouns.json
}

# Confirmatory run (PREREG-s2.md sections 6 and 7): only after stamp B, on the held-out wordings of each kept argument.
final_vecs() {  # final_vecs KEY ARM OUT: prints "DOSE VEC,VEC,..." for the arm from rule 1's decision and the noun gates;
  # writes results/s2/KEY/confirm/vectors.json, and refuses OUT if it already holds a run with other vectors or doses
  $PY -I - "$@" "$(decision_of "$1")" << 'EOF'
import hashlib, json, sys
from pathlib import Path

m, arm, o, dec_path = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
dec = json.loads(dec_path.read_text())[m]
np_ = Path(f"results/s2/{m}/nouns/nouns.json")
nj = json.loads(np_.read_text()) if np_.exists() else {}
assert "finished_at" in nj, f"{np_} is not finished: run nouns_{m} (before stamp B)"
assert abs(nj["dose"] - dec["dose"]) < 1e-9, f"{np_} gated the nouns at {nj['dose']}, rule 1's dose is {dec['dose']}"
pool = json.loads(Path(f"results/{m}/s2/pool.json").read_text())
order = lambda v: (0, int(v[1:])) if v[:1] == "N" else (1, ("arithmetic", "random").index(v))  # noqa: E731
prim = sorted(set(dec["primary_controls"]) | set(nj["added"]), key=order)
assert len(prim) <= 26, f"{len(prim)} primary controls: rule 3's cap is 26"
# the confirmatory run takes the mental-process controls that passed gates (a) to (c) before stamp A, less rule 1's drops
think = [t for t in dec["thinking"] if t in pool["thinking_survivors"]]
if think != dec["thinking"]:
    print(f"{m}: mental-process controls {sorted(set(dec['thinking']) - set(think))} failed gates (a) to (c) before stamp A "
          f"(pool.json): left out", file=sys.stderr)
vecs = list(dec["emotions"]) + prim + think + (["red", "ocean"] if arm == "S2-DEN" else [])
rec = {"model": m, "dose": dec["dose"], "emotions": dec["emotions"], "primary_controls": prim, "nC": len(prim),
       "thinking": think, "n_mp": len(think), "bridge_in_S2-DEN": ["red", "ocean"],
       "decision_file": str(dec_path), "decision_sha256": sha(dec_path), "nouns_file": str(np_), "nouns_sha256": sha(np_)}
o.parent.mkdir(parents=True, exist_ok=True)
(o.parent / "vectors.json").write_text(json.dumps(rec, indent=1))
meta = o / "s2_meta.json"
if meta.exists():
    old = json.loads(meta.read_text())
    assert [v["vec"] for v in old["vectors"]] == vecs and old["doses"] == [dec["dose"]], (
        f"{o} holds a run with other vectors or doses than {rec}")
print(f"{m} {arm}: dose {dec['dose']:g}, {len(dec['emotions'])} emotions, nC {len(prim)}, n_mp {len(think)}"
      + (", red and ocean" if arm == "S2-DEN" else ""), file=sys.stderr)
print(f"{dec['dose']:g} {','.join(vecs)}")
EOF
}

confirm_inputs() {  # confirm_inputs KEY REPLIES: the pilot's stamp A check and guard, then every input against stamp B
  pilot_inputs "$1" "$2"
  { echo "$(date -u +%FT%TZ) confirm $1: $(imds instance-type)"; nvidia-smi -L; } >> results/s2/box/confirm_gpus.txt 2>&1
  stamped_b "${IN[@]}" results/$1/v2/vectors.pt "$(decision_of $1)" results/s2/$1/nouns/nouns.json
  guard_b_ok
}

confirm() {  # confirm KEY ARM MODEL_DIR OUT [ARGS...]: one confirmatory invocation unless done
  local m=$1 arm=$2 d=$3 o=$4 v; shift 4
  v=$(final_vecs $m $arm $o)
  steer $m $arm confirm "$d" $o --items $ITEMS --vecs "${v#* }" --doses "${v%% *}" "$@"
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

# A merged checkpoint is complete once its tokenizer is written (merge_adapter writes it after the weights). Only the
# checkpoint on /nvme counts: /nvme is wiped on every stop, and ftvec's diag.json from an earlier session says nothing
# about this one. The merge is skipped as a whole only when every stage that reads the checkpoints is done.
merged() { [ -f $DIST/$1/merged/model.safetensors.index.json ] && [ -f $DIST/$1/merged/tokenizer_config.json ]; }
done_merge() { { merged D2 && merged D2c; } || { done_ftvec && done_pilot_ft && done_confirm_ft; }; }
do_merge() {
  local A=results/qwen/adapters/distill_keep a
  [ -f $Q/config.json ] || { echo "no model at $Q: run setup_qwen first"; return 1; }
  sha256sum -c << EOF
d7270dd9d0109e6d0b84075e7002ee03c686396ff31266819158c2c658c43bce  $A/D2/adapter/adapter_model.safetensors
c50736203ca7ef34aa8b1b5b65d21fc108a9fbda46ac3acbf3081337e7b705f9  $A/D2c/adapter/adapter_model.safetensors
036ef412ecb8c6bec11ab8b8a8978276fc27e55db77eb81c9222f80fdbd0bdc1  $A/D2/adapter/adapter_config.json
aaa3180403d31ec67093bc25ae2c97f3ddfdd01fb06fb8063bb3be17687cc05b  $A/D2c/adapter/adapter_config.json
EOF
  for a in D2 D2c; do
    merged $a && continue
    rm -rf $DIST/$a/merged   # an interrupted merge's partial output
    mkdir -p $DIST/$a && cp -r $A/$a/adapter $A/$a/meta.json $DIST/$a/
    $PY -u runs/distill.py merge --model $Q --out $DIST --arm $a
    shards_ok $DIST/$a/merged
  done
}

done_ftvec() { [ -s results/qwen/s2ft/D2/diag.json ] && [ -s results/qwen/s2ft/D2c/diag.json ]; }
do_ftvec() {
  [ -f results/qwen/s2/controls.pt ] || { echo "no results/qwen/s2/controls.pt: run vectors_qwen first"; return 1; }
  local a m o
  for a in D2 D2c; do
    m=$DIST/$a/merged; o=results/qwen/s2ft/$a
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

# Pilot (spec section 8). The base arms refuse weights other than the pinned revision (results/s2/box/*_box.json).
done_pilot_qwen() { [ -s $PQ/S2-DEN/readouts.csv ] && [ -s $PQ/S2-CON/readouts.csv ]; }
do_pilot_qwen() {
  pilot_inputs qwen qwen3-32b
  local arm
  for arm in S2-DEN S2-CON; do pilot qwen $arm $Q $PQ/$arm --doses 2,1; done
}

done_pilot_ft() { [ -s $PQ/S2-FTD/readouts.csv ] && [ -s $PQ/S2-FTC/readouts.csv ]; }
do_pilot_ft() {   # S2-FTD on D2, S2-FTC on D2c; base-arm hashes are in $PQ/checkpoints.json once pilot_qwen ran
  pilot_inputs qwen qwen3-32b
  { merged D2 && merged D2c; } || { echo "no complete merged checkpoints under $DIST: run merge first"; return 1; }
  pilot qwen S2-FTD $DIST/D2/merged $PQ/S2-FTD --doses 2,1 --replies qwen3-32b
  pilot qwen S2-FTC $DIST/D2c/merged $PQ/S2-FTC --doses 2,1 --replies qwen3-32b
}

done_pilot_k26() { [ -s $PK/S2-DEN/readouts.csv ] && [ -s $PK/S2-CON/readouts.csv ]; }
do_pilot_k26() {  # the half dose runs only if rule 1 sends K2.6 there (pilot_k26_half)
  pilot_inputs k26 k26
  local arm
  for arm in S2-DEN S2-CON; do pilot k26 $arm $K $PK/$arm --doses 4; done
}

done_pilot_k26_half() {
  [ -f $PK/rule1/half_not_needed ] || { [ -s ${PK}_half/S2-DEN/readouts.csv ] && [ -s ${PK}_half/S2-CON/readouts.csv ]; }
}
do_pilot_k26_half() {
  done_pilot_k26 || { echo "no K2.6 pilot readouts in $PK: run pilot_k26 first"; return 1; }
  local r=$PK/rule1 dose arm
  mkdir -p $r
  # rule 1 as the laptop applies it (runs/s2_table.py, frozen in stamp A), on the registered-dose rows
  $PY -u runs/s2_table.py --phase pilot --out $r/table.csv --gates $r/gates.csv --decision $r/decision.json \
    $PK/S2-DEN/readouts.csv $PK/S2-CON/readouts.csv
  dose=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["k26"]["dose"])' $r/decision.json)
  echo "rule 1, K2.6: confirmatory dose $dose units (registered 4)"
  if [ "$dose" = 4.0 ]; then
    echo "rule 1 keeps K2.6 at the registered dose ($r/decision.json): no half-dose run" | tee $r/half_not_needed
    return 0
  fi
  [ "$dose" = 2.0 ] || { echo "unexpected rule 1 dose $dose"; return 1; }
  pilot_inputs k26 k26
  for arm in S2-DEN S2-CON; do pilot k26 $arm $K ${PK}_half/$arm --doses 2; done
}

# Rule 3's added nouns (before stamp B; noun_gates above). Qwen first: K2.6's target is Qwen's count after the rule.
done_nouns_qwen() { grep -qs '"finished_at"' $NQ/nouns.json; }
do_nouns_qwen() { pilot_inputs qwen qwen3-32b; noun_gates qwen $Q; }

done_nouns_k26() { grep -qs '"finished_at"' $NK/nouns.json; }
do_nouns_k26() { pilot_inputs k26 k26; noun_gates k26 $K; }

# Confirmatory (after stamp B). Each refuses unless prereg/manifest-s2.txt and its .ots exist, every input matches the
# manifest and the held-out guard reports stamp B (stamp_b true); the base arms also refuse weights other than the
# pinned revision, and checkpoints.json in $CQ refuses an FT arm on the base's or the other arm's hash.
# The g7e fallback (DEVIATIONS-s2.md, 2026-10-19): on 96 GB cards the box has 2 GPUs, not 4, so each worker takes
# double work: Qwen one GPU per worker (the whole model fits one card), K2.6 twice the tokens per pass. The engines'
# numbers do not depend on how tokens are split into workers or passes; on 46 GB cards nothing changes.
QX=(); KX=()
if [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)" -ge 90000 ] 2>/dev/null; then
  QX=(--gpus-per-worker 1); KX=(--max-tokens 1600000)
fi

done_confirm_qwen() { [ -s $CQ/S2-DEN/readouts.csv ] && [ -s $CQ/S2-CON/readouts.csv ]; }
do_confirm_qwen() {
  confirm_inputs qwen qwen3-32b
  local arm
  for arm in S2-DEN S2-CON; do confirm qwen $arm $Q $CQ/$arm "${QX[@]}"; done
}

done_confirm_ft() { [ -s $CQ/S2-FTD/readouts.csv ] && [ -s $CQ/S2-FTC/readouts.csv ]; }
do_confirm_ft() {   # S2-FTD on D2, S2-FTC on D2c, the base-Qwen replies on the control arguments' held-out wordings
  confirm_inputs qwen qwen3-32b
  { merged D2 && merged D2c; } || { echo "no complete merged checkpoints under $DIST: run merge first"; return 1; }
  confirm qwen S2-FTD $DIST/D2/merged $CQ/S2-FTD --replies qwen3-32b "${QX[@]}"
  confirm qwen S2-FTC $DIST/D2c/merged $CQ/S2-FTC --replies qwen3-32b "${QX[@]}"
}

done_confirm_k26() { [ -s $CK/S2-DEN/readouts.csv ] && [ -s $CK/S2-CON/readouts.csv ]; }
do_confirm_k26() {
  confirm_inputs k26 k26
  local arm
  for arm in S2-DEN S2-CON; do confirm k26 $arm $K $CK/$arm "${KX[@]}"; done
}

# The post-confirmatory carry-over run and D1.4 bridge (exploratory; PREREG-s2.md section 5 fixes the design at
# stamp B): runs/s2_carry.py, after confirm_qwen and on the same pinned base weights. Every input is checked against
# stamp B first, and the held-out guard must open (the carry-over conditions on the held-out openings). The bridge
# takes final_vecs' S2-DEN vectors and dose and runs/s2_carry.py checks them against the confirmatory S2-DEN run's
# s2_meta.json; final_vecs writes its record to $XQ/vectors.json, never to the confirmatory directory.
XQ=results/s2/qwen/carry
done_carry_qwen() { [ -s $XQ/carryover/table.csv ] && [ -s $XQ/bridge/summary.csv ]; }
do_carry_qwen() {
  done_confirm_qwen || { echo "carry_qwen runs after confirm_qwen: no $CQ/S2-{DEN,CON}/readouts.csv"; return 1; }
  [ -f $Q/config.json ] || { echo "no checkpoint at $Q: run setup_qwen"; return 1; }
  confirm_inputs qwen qwen3-32b
  stamped_b runs/s2_carry.py runs/confirm_table.py kimisroom/battery.py prompts/battery/items_confirm.jsonl \
    prompts/tasks/confirmatory_tasks.jsonl prompts/denial/d1_argument_4.md data/assent/qwen3-32b_d1_argument_4.json
  [ -s $XQ/carryover/table.csv ] || $PY -u runs/s2_carry.py carry --model-path $Q --out $XQ/carryover
  local v
  v=$(final_vecs qwen S2-DEN $XQ/bridge)
  [ -s $XQ/bridge/readouts.csv ] ||
    $PY -u runs/s2_carry.py bridge --model-path $Q --out $XQ/bridge --vecs "${v#* }" --doses "${v%% *}"
  $PY -u runs/s2_carry.py table --bridge $XQ/bridge --den-readouts $CQ/S2-DEN/readouts.csv
}

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
