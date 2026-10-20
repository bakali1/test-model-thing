#!/bin/bash
# fsp.sh -- does a STATE READ-OUT loss make the recurrent state usable?
#
# THE QUESTION
#   The measured situation (2026-10-13, RESULTS.md):
#     * a model with the log-spaced half-life init (P4) and the decay group
#       exempt from weight decay (P6) RETAINS a signal 128 bytes out: 5.5% of
#       the ruler's control floor, 15x the trained incumbent, against 0.00%
#       for a default-init control on the same bytes;
#     * every copy/retrieval probe is still at CHANCE for every model ever
#       trained, including that one.
#   So the state HOLDS a long-memory signal and cannot USE it. Next-byte CE
#   is the reason: it only ever asks the OUTPUT (one step ahead) to be right.
#   It never asks the STATE to be usable later, so nothing in the objective
#   rewards a state that can be read out.
#
# THE IDEA (FlashLM, Chang 2026, §3.1 -- see docs/papers/PAPER-NOTE.md)
#   FSP: at every r-th position, read the state and predict WHICH bytes appear
#   in the next tau bytes, as a binary multi-hot, with BCE; L = CE + alpha*FSP.
#   The gradient lands on the STATE, and from there the engine's existing
#   credit machinery carries it to the embedding (the ring) and to the decay
#   logits (the decaytrace). It is the cheapest objective that can make the
#   state carry forward-looking content in a readable form.
#
#   The paper's own evidence is weak (single unseeded run; its own Table 2
#   contradicts its "loss beats architecture" thesis; the FSP model's
#   coherence verdict is "Partial"), so this is a TEST of the idea, not an
#   endorsement of the paper.
#
# THE TRANSLATION, AND WHERE IT DEPARTS FROM THE PAPER
#   * Vocabulary 4096 -> 256 bytes. The multi-hot is over BYTES.
#   * pos_weight 50 -> 8.4. The paper's 50 balances a 4096-token vocabulary
#     where a 64-token future window holds ~1.6% distinct tokens (positives
#     outnumber negatives ~1:63, and a 50x weight makes the positive class
#     carry ~44% of the loss). In BYTES the same window holds ~22 DISTINCT
#     values of 256 -- measured on our corpus, 22.0 distinct symbols, 8.58%
#     dense -- so 50 would weight the positive class ~6x over the paper's own
#     balance and push the head toward "these 22 common bytes are certainly
#     coming". Solving for the same positive-mass fraction gives 8.4. Use 8.
#   * r=16, tau=64, alpha=0.1: the paper's values, unchanged.
#   * The head reads the LAST layer's state (the one closest to the output)
#     through its OWN LayerNorm parameters (fsp.g, fsp.b0). Borrowing the
#     layer's would make the auxiliary a knob on the local read-out -- the
#     FSP gradient would move layer{g,b} and the arm's CE could move for a
#     reason that has nothing to do with memory.
#
# THE KNOWN LIMITATION OF THIS FIRST ARM (D2 in docs/papers/RELEVANCE-NOTE.md)
#   The FSP gradient reaches ONE of the sixteen layers' MEMORIES: the layers'
#   states are independent recurrences, so a head on the last layer trains
#   that layer's decay logits and the SHARED embedding's credit (which every
#   layer's state content is built from), and no other layer's decay. A null
#   result is therefore ambiguous between "the objective does not work" and
#   "one of sixteen memories got the job". The pre-specified follow-up is a
#   per-layer variant (--fsp-layer sweep, or a shared bottleneck head over
#   all sixteen) -- which costs 16x the head's flops and is therefore a
#   SECOND arm, not a variant of this one.
#
# THE A/B
#   arm A_p4_wd0      (already running, Colab): P4 init, wd_decay 0, no FSP
#   arm A_p4_wd0_fsp  P4 init, wd_decay 0, FSP alpha 0.1
#   Same init, same wd, same corpus, same data seed, same byte budget, same
#   seed. The ONLY difference is the auxiliary loss.
#
#   THE VERDICT IS THE RULER, NOT THE LOSS: the retention curve (d=8/32/128/
#   512) and the selective/associative copy tasks, plus the held-out CE.
#   A lower CE with an unchanged copy task means the auxiliary bought local
#   modelling, not memory.
#
# USAGE
#   bash docs/runbooks/fsp.sh            # the arm, on the current box
set -e

REPO="${REPO:-$HOME/Projects/tmt-axis}"
V="${V:-$HOME/Projects/test-model-thing/.venv-gpu/bin/python}"
D="${D:-$REPO/data/wikipedia_clean/wikistories/wiki_tinystories_*.txt}"
OUT="$OUT_DIR"
B="${B:-1024}"                  # the incumbent's batch size
STEPS="${STEPS:-161333}"        # 1 epoch over the 4-file corpus (165 MB)
R="${R:-16}"                    # FSP position every r bytes
TAU="${TAU:-64}"                # the target is the next tau bytes
ALPHA="${ALPHA:-0.1}"
POSW="${POSW:-8}"               # class-balanced for a 256-byte vocabulary
cd "$REPO"
mkdir -p "$OUT"

echo "=== arm A_p4_wd0_fsp: P4 init (2..512 B), wd_decay 0, FSP alpha=$ALPHA"
echo "    r=$R tau=$TAU pos_weight=$POSW | B=$B T=1 ring_k=18 tail off"
echo "    steps=$STEPS ($((STEPS * B / 1000000)) MB)"

TMT_TAIL=0 $V train_gpu.py "$OUT/A_p4_wd0_fsp.npz" train --dataset "$D" \
   --batch "$B" --seqlen 1 --steps "$STEPS" --warmup 20 --no-tagged \
   --ring-k 18 --half-min 2 --half-max 512 --wd-decay 0.0 \
   --fsp-alpha "$ALPHA" --fsp-r "$R" --fsp-tau "$TAU" \
   --fsp-pos-weight "$POSW" --fsp-layer -1 \
   --device cuda --seed 0 > "$OUT/A_p4_wd0_fsp.log" 2>&1

tail -3 "$OUT/A_p4_wd0_fsp.log"

echo "=== the ruler (the verdict) ==="
export TMT_RULER_CKPT="$OUT/A_p4_wd0_fsp.npz"
export TMT_RULER_RAW="$REPO/data/eval/tinystories_valid.txt"
$V tools/ruler.py > "$OUT/ruler_A_p4_wd0_fsp.txt" 2>&1
grep -E "retention d=|held-out CE|selective_copy d=8|associative_copy d=8" \
    "$OUT/ruler_A_p4_wd0_fsp.txt"

echo "FSP ARM DONE"
