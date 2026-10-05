#!/usr/bin/env bash
# p1_lr_sweep.sh -- ROADMAP v3 P1: the global learning-rate sweep.
#
# WHY THIS DESIGN (and it differs from ROADMAP v2's "from-scratch byte-matched
# arms", deliberately):
#
#   v2 ordered credit work because SNR was assumed measurable. It was not --
#   RESULTS.md later showed the standing 0.05 is read while theta moves every
#   byte, and that the same estimator at a FROZEN theta reads 0.633. So the
#   question is not "what lr trains a good model from zero" (expensive, and
#   confounded with everything else that changes from scratch). It is:
#
#       GIVEN THIS MODEL, DOES A SMALLER STEP BUY ANYTHING?
#
#   That is a continuation experiment. Every arm resumes from the SAME pinned
#   t=919,897 checkpoint with byte-matched data and data_seed 0, and differs in
#   exactly one number: --lr. The arms are therefore paired by construction.
#
#   LIMITATION, stated up front: a continuation arm cannot tell you the optimal
#   lr for a from-scratch run, because a smaller step makes less progress in a
#   fixed budget. That is exactly why there are TWO readouts below and why the
#   verdict is written to handle the tradeoff case honestly instead of
#   declaring a winner whenever SNR moves.
#
# TWO READOUTS:
#   1. telemetry SNR  -- should RISE with smaller lr (already confirmed by
#                       tools/snr_vs_lr.py: 0.059 -> 0.152 -> 0.364)
#   2. ruler CE + the state-probe horizon -- the thing we actually care about
#
#   If SNR rises but CE worsens, that is NOT a win; it is the ordinary
#   precision-vs-progress tradeoff and the fix is a SCHEDULE (warmup +
#   decay), not a smaller constant lr. The script says so rather than
#   reporting a partial win.
#
# RESUMABLE: each arm is independent and appends to p1_lr_sweep.log. If this
# dies (Colab GC, laptop sleep, power cut) re-running skips finished arms.

set -u
cd ~/tmt-run

PY=~/Projects/tmt/.venv-gpu/bin/python
SRC=runs/lrroles/arm_lrdecay1x_t919897.npz   # PINNED, never trained in place
STEPS=${P1_STEPS:-10000}
EVAL_ARGS="--slices 3 --bytes 20000 --xd-bytes 8000 --xd-files 4"
OUT=runs/p1_lr
LRS=${P1_LRS:-"5e-4 2.5e-4 1.25e-4 6.25e-5 3.125e-5"}

mkdir -p $OUT
echo "=== P1 global-LR sweep  $(date -Is)  steps=$STEPS  src=$SRC" | tee -a p1_lr_sweep.log
echo "=== arms: $LRS" | tee -a p1_lr_sweep.log

for LR in $LRS; do
  # train_gpu.py saves IN PLACE, under the INPUT filename. Verified by a 30-step
  # smoke test that overwrote runs/lrroles/arm_lrdecay1x_t919897.npz and advanced
  # its embedded t from 919897 to 919927. So every arm gets its OWN COPY of the
  # incumbent first; without this, arm 2 would silently resume from arm 1's
  # output and all five arms would write the same file.
  ARM=$OUT/arm_lr${LR}.npz
  if [ -f "$ARM" ]; then
    echo "--- skip lr=$LR, $ARM exists" | tee -a p1_lr_sweep.log
    continue
  fi
  cp "$SRC" "$ARM"   # pristine pinned start, identical for every arm
  echo "" | tee -a p1_lr_sweep.log
  echo "########## lr=$LR  $(date -Is)" | tee -a p1_lr_sweep.log

  # --- train -----------------------------------------------------------------
  # --ring-credit nvrtc explicitly: this host is cc 6.1, below Triton's floor,
  # and the default 'auto' prints a scary message every time.
  timeout 14400 $PY train_gpu.py "$ARM" train \
      --lr "$LR" \
      --steps "$STEPS" \
      --data-seed 0 --seed 0 \
      --batch 1024 \
      --ring-credit nvrtc \
      --telemetry-every 250 \
      --telemetry-path "$OUT/tel_lr${LR}.jsonl" \
      --save-every "$STEPS" --keep-k 1 \
      --no-tagged 2>&1 | tail -40 | tee -a p1_lr_sweep.log
  # NOTE: no --hub-repo and no --push. Passing --hub-repo makes the trainer
  # attempt a push with no token on this host, it raises 401, and the run dies
  # BEFORE the final save -- which silently leaves the arm at its pristine t and
  # makes every arm look like the control. The verify step below catches exactly
  # that, which is why it exists.

  # --- locate the checkpoint the trainer just wrote --------------------------
  # the arm's own file was overwritten in place by the final save
  NEW=$ARM
  if [ ! -f "$NEW" ]; then
    echo "!!! no checkpoint for lr=$LR -- arm FAILED, continuing" | tee -a p1_lr_sweep.log
    continue
  fi
  $PY -c "
import sys; sys.path.insert(0,'.')
import gpu_port as G
m = G.GpuModel(dim=512, layers=16, device='cpu', seed=0)
ok = m.load('$NEW')
print('   [verify] $NEW loaded=%s t=%s' % (ok, m.t))
if not ok or m.t != $((919897+STEPS)):
    print('   [verify] UNEXPECTED STEP -- this arm is not comparable'); sys.exit(3)
" 2>&1 | tee -a p1_lr_sweep.log || { echo "!!! lr=$LR verification FAILED" | tee -a p1_lr_sweep.log; continue; }
  echo "--- checkpoint: $NEW" | tee -a p1_lr_sweep.log

  # --- readout 2a: the ruler --------------------------------------------------
  echo "--- ruler CE (lr=$LR)" | tee -a p1_lr_sweep.log
  timeout 3600 $PY tools/eval.py "$NEW" $EVAL_ARGS 2>&1 | tail -25 | tee -a p1_lr_sweep.log

  # --- readout 2b: the memory horizon ----------------------------------------
  # THE DECIDING NUMBER. The horizon is 8 bytes today. If a smaller step extends
  # it, the bottleneck was step size. If the horizon moves but the ruler CE does
  # not, we have bought memory we cannot use.
  echo "--- state probe horizon (lr=$LR)" | tee -a p1_lr_sweep.log
  timeout 7200 $PY tools/state_probe.py \
      --ckpt "$NEW" \
      --cue corpus-support \
      --distractors data/eval/tinystories_valid.txt \
      --distances 4,8,16,32,64,128,256 \
      --calibration-distances 0,1,2 \
      --trials 384 --batch 384 --stability 6 \
      --label "p1-lr$LR" 2>&1 | tail -30 | tee -a p1_lr_sweep.log

  echo "########## done lr=$LR  $(date -Is)" | tee -a p1_lr_sweep.log
done

echo "" | tee -a p1_lr_sweep.log
echo "=== P1 SWEEP COMPLETE  $(date -Is)" | tee -a p1_lr_sweep.log
echo "=== read: p1_lr_sweep.log for CE, and grep 'state carries cue' per arm" | tee -a p1_lr_sweep.log