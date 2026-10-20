#!/bin/bash
# phase5_fsp.sh -- the FSP arm, queued behind the P4 long run.
#
# Waits for the P4 long run's rulers to finish (the GPU must be free), then
# runs the FSP arm and its ruler. The FSP arm is the paper's idea aimed at
# our measured gap: the state can HOLD a signal 128 B out and every
# copy/retrieval probe is still at chance, because next-byte CE never asks
# the state to be usable.
#
# The matched control is arm A (A_p4_wd0), which is already in flight: same
# P4 init, same wd_decay 0, same corpus, same data seed, same seed, same
# byte budget. The only difference is --fsp-alpha 0.1.
set -u
cd /content
mkdir -p runs/taxis
# The FSP code lives in /content/fsp so the arms already in flight (and
# their rulers) keep the ORIGINAL engine, byte for byte. The FSP arm runs
# from the new directory; its checkpoint is read by the new ruler.
cd /content/fsp
export TMT_TAIL=0
export OMP_NUM_THREADS=2

echo "waiting for the P4 long run's rulers to free the GPU..."
while ! grep -q "P4 LONG RULERS DONE" /content/runs/phase4_p4rulers.log 2>/dev/null; do
  sleep 60
done
echo "GPU is free at $(date -u +%H:%M:%SZ) -- starting the FSP arm"

TMT_TAIL=0 python3 train_gpu.py /content/runs/taxis/A_p4_wd0_fsp.npz train \
  --dataset "data/wikipedia_clean/wikistories/wiki_tinystories_*.txt" \
  --batch 1024 --seqlen 1 --steps 161333 --warmup 20 --no-tagged \
  --ring-k 18 --half-min 2 --half-max 512 --wd-decay 0.0 \
  --fsp-alpha 1.0 --fsp-r 16 --fsp-tau 64 --fsp-pos-weight 8.5 --fsp-layer -1 \
  --device cuda --seed 0 > /content/runs/taxis/A_p4_wd0_fsp.log 2>&1
echo "FSP ARM DONE at $(date -u +%H:%M:%SZ)"
tail -2 /content/runs/taxis/A_p4_wd0_fsp.log

export TMT_RULER_RAW=/content/data/eval/tinystories_valid.txt
TMT_RULER_CKPT=/content/runs/taxis/A_p4_wd0_fsp.npz python3 /content/fsp/tools/ruler.py \
  > /content/runs/taxis/ruler_A_p4_wd0_fsp.txt 2>&1
echo "FSP RULER DONE at $(date -u +%H:%M:%SZ)"
grep -E "retention d=|held-out CE" /content/runs/taxis/ruler_A_p4_wd0_fsp.txt
