#!/bin/bash
# p4-usage.sh -- does the log-spaced half-life init SURVIVE training?
#
# THE QUESTION
#   The plan's P4 says: spread the decay half-lives over 2..512 B, because
#   "there is no dimension in this model capable of holding 512 B, and no
#   amount of training builds one it was never given". The capacity half of
#   that claim is now MEASURED (tools/capacity_probe.py, 0 training):
#
#       init                      d=8     d=32    d=128   d=512
#       default (~1 B)           0.40%   0.00%   0.00%   0.00%
#       P4 (2..512 B)           49.2%   21.0%    5.8%    1.05%
#       TRAINED incumbent        4.8%    1.7%    0.36%   0.0007%
#
#   The last row is the trap. The incumbent was INITIALISED with every
#   half-life at ~1 B, so it never had a spread to lose: it is NOT evidence
#   that training destroys a P4 spread. Nobody has ever trained a P4-init
#   model. And two things would destroy the spread if it existed:
#
#     * finding 7: gradient descent undoes the log-spaced init within an epoch
#       (TinyStories, TBPTT K=32, 3 seeds).
#     * P6: sigma(0) = 0.5 is exactly a 1-BYTE half-life, so the decoupled
#       weight decay on the decay logits is a systematic pull toward no
#       memory. The weight decay is what makes the pull systematic.
#
#   So this is the 2x2 the plan needs, and the answer is visible in the
#   WEIGHTS -- no ruler and no training history needed:
#
#       arm 0  default init, wd_decay 0.01   the incumbent's regime (CONTROL)
#       arm 1  P4 init,     wd_decay 0.0     the exemption
#       arm 2  P4 init,     wd_decay 0.01    no exemption
#
# ORDER MATTERS. The two P4 arms answer the question (does the spread survive
# training, and does the weight decay pull it back?), so they run FIRST. The
# from-scratch default-init control runs LAST: it only makes their CE
# comparable to something, it is not part of the question, and it is the
# cheapest arm -- a default init has no hot dims, so the exact-credit tail
# path never engages and it runs at ~6.1 kB/s instead of ~2.5-3.4 kB/s.
#
#   Verdict: the half-life distribution (tools/inspect_halflives.py). A fresh
#   P4 model has mean half-life 92.30 B with 6144 dims > 8 B; a collapsed one
#   has ~1 B with ~8. The mean is the headline.
#
# WHY FROM SCRATCH, NECESSARILY
#   The P4 init IS a different set of initial weights, so an incumbent
#   checkpoint cannot provide them. The consequence is that these arms' CE is
#   NOT comparable with the incumbent's 1.2094 nats/byte -- a from-scratch
#   model is at CE ~2.5 after 8.39 MB -- which is exactly why arm 0 exists: a
#   from-scratch default-init control at the same byte budget, same file,
#   same seed. Their VERDICT is the spread; arm 0 is what makes their CE
#   comparable.
#
# WHY B=256 AND --trace-mode dense
#   Two reasons, and the second is the important one.
#
#   1. MEMORY. The P4 init drives ring_k to 256 (it is inferred from
#      sigma_max), and the ring's per-layer buffer is (B, 256, dim): 8.6 GB
#      at B=1024, 2.15 GB at B=256. On a 4 GB card the P4 arm OOMs at
#      B=1024. The dense trace is the same object size (2.15 GB at B=256)
#      and fits, at 2.46 GB peak.
#
#   2. THE CREDIT REACH MUST BE HELD CONSTANT. With the ring, the default
#      init gets ring_k=14 and the P4 init gets ring_k=256 -- so the arms
#      would differ in how far back credit reaches, and the 2x2 would be
#      measuring the trace reach as well as the init. The dense trace has no
#      cap, so BOTH arms get the same exact credit and the only difference
#      is the init and the weight decay. That is the 2x2.
#
# USAGE
#   bash docs/runbooks/p4-usage.sh
set -e

REPO="${REPO:-$HOME/Projects/tmt-axis}"
V="${V:-$HOME/Projects/test-model-thing/.venv-gpu/bin/python}"
D="${D:-$REPO/data/wikipedia_clean/wikistories/wiki_tinystories_001.txt}"
OUT="$REPO/runs/taxis"
B="${B:-256}"                 # B=256 is the largest that fits the P4 init's
                              # ring_k=256 on a 4 GB card (2.46 GB peak, dense)
STEPS="${STEPS:-32768}"       # 32768 steps x 256 B = 8.39 MB, the same byte
                              # budget as the T-axis A/B arms
cd "$REPO"
mkdir -p "$OUT"

run_arm () {   # name  half_min  half_max  wd_decay
  local name="$1" hmin="$2" hmax="$3" wd="$4"
  echo "=== arm $name: half_min=$hmin half_max=$hmax wd_decay=$wd (FROM SCRATCH, B=$B, dense) ==="
  local hl=()
  if [ "$hmin" != "0" ]; then hl=(--half-min "$hmin" --half-max "$hmax"); fi
  $V train_gpu.py "$OUT/$name.npz" train --dataset "$D" --batch "$B" \
     --seqlen 1 --steps "$STEPS" --warmup 20 --no-tagged \
     --trace-mode dense "${hl[@]}" --wd-decay "$wd" \
     --device cuda --seed 0 > "$OUT/$name.log" 2>&1
  tail -2 "$OUT/$name.log"
  $V tools/inspect_halflives.py "$OUT/$name.npz"
}

run_arm armFS_p4_wd0       2 512   0.0
run_arm armFS_p4_wd001     2 512   0.01
run_arm armFS_default_wd001 0   0   0.01

echo "P4 USAGE ARMS DONE"
