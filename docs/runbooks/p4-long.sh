#!/bin/bash
# THE P4 LONG RUN: does MORE TRAINING unlock the usage half?
#
# The MX150 measured 8.39 MB (0.2 epoch) of a P4-init model: the spread
# survived (89.48 B mean half-life) and the retention curve at d=128 was
# 5.5% of the control floor -- 15x the trained incumbent -- but EVERY
# copy/retrieval arm was still at chance. Two explanations, and they need
# opposite fixes:
#
#   (a) the readout has not been trained long enough to use what the state
#       now holds  -> MORE TRAINING fixes it
#   (b) the state is a linear, recency-weighted MIXTURE of raw byte
#       embeddings: it holds the signal's magnitude but loses its identity
#       and its order -> a different WRITE/READ path fixes it
#
# This run separates them. The arms are at the INCUMBENT'S RECIPE (B=1024,
# T=1, ring_k=18, the same 4-file corpus, the same data seed, the same lr),
# so their CE is comparable with the incumbent's 1.2094 and their first
# 165 MB are the same bytes the incumbent saw first.
#
#   A_p4_wd0        P4 init (2..512 B), wd_decay 0.0     the memory arm
#   B_default_wd001 default init,      wd_decay 0.01    the CONTROL
#   C_p4_wd001      P4 init (2..512 B), wd_decay 0.01    the erosion arm
#
# TMT_TAIL=0 for all three: with the P4 init nearly every dim is hot and the
# tail accumulator would be 4.5 GB/layer. The spread's fate is decided by
# the .decay gradient (the unbounded decaytrace), not by the tail, and the
# tail only touches the EMBEDDING's credit. All three share the setting,
# so their CEs are comparable.
set -u
cd /content
mkdir -p runs/taxis
STEPS="${STEPS:-161333}"   # 1 epoch over the 4-file corpus (165 MB)
export TMT_TAIL=0
export OMP_NUM_THREADS=2

run_arm () {   # name half_min half_max wd_decay
  name="$1"; hmin="$2"; hmax="$3"; wd="$4"
  hl=""
  if [ "$hmin" != "0" ]; then hl="--half-min $hmin --half-max $hmax"; fi
  echo "=== $name: half_min=$hmin half_max=$hmax wd_decay=$wd | B=1024 T=1 ring_k=18 tail off ==="
  python3 train_gpu.py "runs/taxis/$name.npz" train \
    --dataset "data/wikipedia_clean/wikistories/wiki_tinystories_*.txt" \
    --batch 1024 --seqlen 1 --steps "$STEPS" --warmup 20 --no-tagged \
    --ring-k 18 $hl --wd-decay "$wd" --device cuda --seed 0 \
    > "runs/taxis/$name.log" 2>&1
  echo "--- $name done: $(tail -2 runs/taxis/$name.log | head -1)"
}

run_arm A_p4_wd0         2 512 0.0  &
run_arm B_default_wd001  0 0   0.01 &
run_arm C_p4_wd001       2 512 0.01 &
wait
echo "P4 LONG ARMS DONE"

# the ruler: retention curve + copy/retrieval + held-out CE, on the same
# protocol as the incumbent's (6 slices x 20000 B over BOTH eval files)
export TMT_RULER_RAW=/content/data/eval/tinystories_valid.txt
for name in A_p4_wd0 B_default_wd001 C_p4_wd001; do
  ( TMT_RULER_CKPT=/content/runs/taxis/$name.npz python3 tools/ruler.py \
      > "runs/taxis/ruler_$name.txt" 2>&1 & )
  sleep 4
done
wait
echo "P4 LONG RULERS DONE"
