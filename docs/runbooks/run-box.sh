#!/bin/bash
# run_box.sh -- the A/B pair, from scratch, on the box, with the private HF
# repo as the durability layer.
#
#   A_p4_wd0        the CONTROL       (P4 init 2..512 B, wd_decay 0)
#   A_p4_wd0_fsp    the PAPER'S ARM   (the same + --fsp-alpha 1.0)
#
# Same init, same wd, same corpus, same data seed, same seed, same B, same
# byte budget; ONE flag differs. 1 epoch = 161333 steps = 165 MB.
#
# Every checkpoint goes to bakali1/tmt-checkpoints every 10000 steps (10.2
# MB of training), so a recycled runtime costs at most a 6% data repeat and
# then the run continues from the hub. --from-scratch means the run does NOT
# pull the repo's older checkpoints: this arm is new.
#
# FROM_SCRATCH is set by the cycle script for the FIRST cycle only: later
# cycles must NOT set it, or the arm would ignore its own hub checkpoint and
# start over. With it unset, the hub's pull takes THIS arm's newest file (the
# pull is arm-aware) and the run continues -- and the data position in the
# checkpoint means it continues the CORPUS too.
#
# The arms are named *_c because the repo already holds the first run's
# t=010000 files (that run died when the runtime was recycled); distinct
# names keep the two runs' checkpoints apart in the repo. The hub's pull is
# now ARM-AWARE (it resumes the base name it is asked for, not the repo's
# global best -- which here is t=919897, the incumbent's own arm, so an
# unfixed pull would have loaded the WRONG WEIGHTS into this arm).
#
# NOTE: --no-tagged is DROPPED on purpose. The engine refuses it together
# with --hub-repo: the hub protocol validates that the filename's step
# matches the npz's internal t, and the rolling file's name encodes no
# step. The tagged name is <base>_t<step>.npz, and <base> is the arm's own
# name -- so the two arms' hub files cannot collide.
set -u
cd /content
export TMT_TAIL=0
export OMP_NUM_THREADS=2
export TMT_HUB_REPO=bakali1/tmt-checkpoints
export TARGET=19000  # option C: the salvaged control's EXACT budget (19.5 MB),
                   # as an ABSOLUTE step target so a resumed run lands on it
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p runs/taxis

# ---- the corpus, split from the fetched region (no data from the laptop) ---
python3 - <<'PY'
import os
sizes = [41300184, 41317091, 41315451, 41327130]     # the four training shards
held = 41308264                                        # the held-out fifth
region = open("/content/tiny_region.bin", "rb").read()
off = 0
for i, s in enumerate(sizes, 1):
    p = f"data/wikipedia_clean/wikistories/wiki_tinystories_{i:03d}.txt"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(region[off:off + s])
    off += s
p = "data/eval/wiki_tinystories_005.txt"
os.makedirs(os.path.dirname(p), exist_ok=True)
with open(p, "wb") as fh:
    fh.write(region[off:off + held])
print("corpus written:", os.listdir("data/wikipedia_clean/wikistories/"),
      "+ held-out", os.path.getsize(p), "bytes")
print("training region [0, %d) | held-out region [%d, %d) -- DISJOINT"
      % (off, off, off + held))
PY

run_arm () {   # name  extra_flags...
  name="$1"; shift
  export TMT_RUN_ID="${name}"
  echo "=== $name: TMT_RUN_ID=$TMT_RUN_ID $(date -u +%H:%M:%SZ) ==="
  python3 train_gpu.py "runs/taxis/$name.npz" train \
    --dataset "data/wikipedia_clean/wikistories/wiki_tinystories_*.txt" \
    --batch 1024 --seqlen 1 --stop-at-t ${TARGET:-19000} --warmup 20 \
    --ring-k 18 --half-min 2 --half-max 512 --wd-decay 0.0 \
    ${FROM_SCRATCH:+--from-scratch} --hub-repo bakali1/tmt-checkpoints \
    --hub-save-every 10000 \
    --device cuda --seed 0 "$@" > "runs/taxis/$name.log" 2>&1
  echo "--- $name trained: $(tail -2 runs/taxis/$name.log | head -1)"
  TMT_RULER_RAW=/content/data/eval/tinystories_valid.txt \
  TMT_RULER_CKPT=/content/runs/taxis/$name.npz \
    python3 tools/ruler.py > "runs/taxis/ruler_$name.txt" 2>&1
  echo "--- $name ruled: $(grep -m1 'held-out CE' runs/taxis/ruler_$name.txt)"
}

run_arm A_p4_wd0_c &
sleep 30
run_arm A_p4_wd0_fsp_c --fsp-alpha 1.0 --fsp-r 16 --fsp-tau 64 \
       --fsp-pos-weight 8.5 --fsp-layer -1 &
wait
echo "BOTH ARMS DONE $(date -u +%H:%M:%SZ)"
