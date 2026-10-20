#!/bin/bash
# taxis-ab.sh -- the T-axis A/B, as a runbook.
#
# THE QUESTION
#   Does the T axis (T bytes per stream per optimizer update, with exact
#   within-window credit for the FF/LN/decoder weights) help the held-out CE,
#   at a cost that is not throughput?
#
#   The cost is already measured and it is ZERO: at constant B, T=16 is 1.09x
#   FASTER per byte than T=1 (tools/seqlen_probe.py --device cuda, MX150).
#   So the question is purely about quality.
#
# THE DESIGN, AND WHAT IT CANNOT ISOLATE
#   The T axis changes two things at once, and they are entangled by
#   construction: the within-window credit AND the number of optimizer updates
#   per byte. There is no design that holds both fixed, so this A/B measures
#   the T AXIS AS A REGIME, and the two families below are biased in opposite
#   directions so that the two effects can be told apart after the fact:
#
#     family A, matched BYTES (8.39 MB each, RESUMED from the incumbent)
#       T=1  -> 8192 steps    = 8192 updates
#       T=4  -> 2048 windows = 2048 updates
#       T=16 ->  512 windows =  512 updates
#       bias: T=16 does 16x FEWER updates for the same bytes. In the resumed
#       regime the corpus is already ~5.7 epochs deep, so fewer updates means
#       LESS overfitting -> FOR T=16.
#
#     family B, matched UPDATES (512 updates each, RESUMED from the incumbent)
#       T=1  ->  512 steps    = 0.52 MB
#       T=4  ->  512 windows = 2.10 MB
#       T=16 ->  512 windows = 8.39 MB   (the T=16 arm of family A)
#       bias: T=16 sees 16x MORE bytes of an already-seen corpus -> more
#       repetition -> MORE overfitting -> AGAINST T=16.
#
#   A T=16 win in BOTH families is strong evidence for the regime; a split
#   verdict says which of the two effects dominates. The credit effect alone
#   is isolated at the gradient level (tests/check_trace_bptt.py: exact to
#   5e-3 within the window) and, in the only regime where upstream measured
#   exact credit as paying, by the state-reset condition (PLAN P7).
#
# EVERY ARM RESUMES THE SAME INCUMBENT (t=919,897, B=1024, ring_k=18, held-out
# CE 1.2094 +/- 0.0143 nats/byte). The arms are NOT from scratch: a
# from-scratch arm at 8.39 MB sits at CE ~2.5, where the from-scratch
# transient swamps any T effect and the measurement has no power. The
# from-scratch regime is where the P4 arms live instead
# (docs/runbooks/p4-usage.sh), because a different INIT cannot resume an
# incumbent.
#
# THE VERDICT
#   the ruler's held-out CE (section 5) for every arm, plus its retention
#   curve (section 1) alongside, because a CE that does not move says nothing
#   about memory -- the ruler's own section 8 says so.
#
# USAGE
#   scp this file to the GPU box, then:
#     bash docs/runbooks/taxis-ab.sh            # the whole thing
#   Every arm writes runs/taxis/<name>.npz and runs/taxis/<name>.log.
set -e

REPO="${REPO:-$HOME/Projects/tmt-axis}"
V="${V:-$HOME/Projects/test-model-thing/.venv-gpu/bin/python}"
D="${D:-$REPO/data/wikipedia_clean/wikistories/wiki_tinystories_001.txt}"
CK="${CK:-$REPO/runs/lrroles/arm_lrdecay1x_t919897.npz}"
OUT="$REPO/runs/taxis"
export TMT_RULER_RAW="$REPO/data/eval/tinystories_valid.txt"
export OMP_NUM_THREADS=2
cd "$REPO"
mkdir -p "$OUT"

# ---- family A: matched BYTES -------------------------------------------------
# 8.39 MB each = 8.39e6 / (B*T) windows.
for spec in "1:8192" "4:2048" "16:512"; do
  T="${spec%%:*}"; STEPS="${spec##*:}"
  echo "=== family A: T=$T, $STEPS windows, $((STEPS*T*1024)) bytes ==="
  cp "$CK" "$OUT/armT${T}.npz"
  $V train_gpu.py "$OUT/armT${T}.npz" train --dataset "$D" --batch 1024 \
     --seqlen "$T" --ring-k 18 --steps "$STEPS" --warmup 20 --no-tagged \
     --device cuda --seed 0 > "$OUT/armT${T}.log" 2>&1
  tail -2 "$OUT/armT${T}.log"
done

# ---- family B: matched UPDATES ----------------------------------------------
# 512 updates each. The paths are pre-seeded with the incumbent, so these
# RESUME like family A does -- an earlier version pointed at non-existent
# paths, which would have trained them FROM SCRATCH and mixed the families.
for spec in "1" "4"; do
  T="$spec"
  echo "=== family B: T=$T, 512 windows, $((512*T*1024)) bytes ==="
  cp "$CK" "$OUT/armT${T}_u512.npz"
  $V train_gpu.py "$OUT/armT${T}_u512.npz" train --dataset "$D" --batch 1024 \
     --seqlen "$T" --ring-k 18 --steps 512 --warmup 20 --no-tagged \
     --device cuda --seed 0 > "$OUT/armT${T}_u512.log" 2>&1
  tail -2 "$OUT/armT${T}_u512.log"
done

# ---- the rulers --------------------------------------------------------------
# The held-out CE is the verdict; the retention curve comes with it.
for name in armT1 armT4 armT16 armT1_u512 armT4_u512; do
  [ -f "$OUT/$name.npz" ] || { echo "MISSING $name.npz"; continue; }
  echo "=== ruler: $name ==="
  TMT_RULER_CKPT="$OUT/$name.npz" $V tools/ruler.py > "$OUT/ruler_$name.txt" 2>&1
  grep -A 2 "EVAL.PY LANGUAGE CELLS" "$OUT/ruler_$name.txt" | head -4
  grep "held-out CE" "$OUT/ruler_$name.txt" | head -2
done
# the reference point: the incumbent itself, on the same instrument
TMT_RULER_CKPT="$CK" $V tools/ruler.py > "$OUT/ruler_incumbent.txt" 2>&1
grep "held-out CE" "$OUT/ruler_incumbent.txt" | head -2

echo "T-AXIS A/B DONE"
