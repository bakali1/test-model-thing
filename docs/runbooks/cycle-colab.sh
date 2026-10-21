#!/bin/bash
# cycle_colab.sh -- run the C A/B across Colab's runtime recycling.
#
# WHY THIS EXISTS. A Colab free-tier runtime is recycled without warning --
# we have watched one die mid-run twice -- so a 19,000-step arm cannot be
# assumed to live inside one runtime. Two pieces of machinery make it
# survivable, and both landed today:
#
#   * the DATA POSITION in the checkpoint (a resume CONTINUES the corpus
#     instead of replaying it -- without it the arm's unique data would be
#     capped at the push cadence, i.e. 10 MB seen sixteen times);
#   * --stop-at-t, an ABSOLUTE step target, so a resumed run finishes at
#     exactly the same step (and therefore the same byte budget) as one
#     that never died.
#
# So the loop is: allocate a T4 -> ship the code and the credential -> fetch
# the corpus (4 s on the box) -> resume both arms from the hub -> let them run
# until the runtime dies -> repeat. Each cycle advances the arms; the corpus
# is not replayed; the budget is fixed.
#
# The arms:
#   A_p4_wd0_c       the CONTROL     (P4 init 2..512 B, wd_decay 0)
#   A_p4_wd0_fsp_c   the PAPER'S ARM (the same + --fsp-alpha 1.0)
# 19,000 steps = 19.5 MB each -- the salvaged control's EXACT budget.
set -u
REPO=/home/baku/programing/opensource/new-llm/test-model-thing
ALIAS=tmtcolab4
SESSION=tmtcolab4
TARGET=19000
CYCLES="${CYCLES:-8}"
SSH="ssh -i $HOME/.ssh/colab -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
SCP="scp -q -i $HOME/.ssh/colab -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"

# The Colab bridge allows ONE client per runtime, and a client that has just
# disconnected takes a moment to be released. Every connection in this
# script goes through ssh_retry, which retries the "another `colab ssh` is
# connected" refusal instead of treating it as a failure.
scp_retry() {
  for _ in 1 2 3 4 5 6; do
    if $SCP "$@" 2>/dev/null; then return 0; fi
    sleep 8
  done
  return 1
}

ssh_retry() {
  for _ in 1 2 3 4 5 6; do
    if out=$($SSH "$ALIAS" "$@" 2>&1); then
      printf '%s\n' "$out"
      return 0
    fi
    case "$out" in
      *"another \`colab ssh\` is connected"*|*"Connection refused"*|*"closed by"*)
        sleep 8 ;;
      *)
        printf '%s\n' "$out"
        return 1 ;;
    esac
  done
  printf '%s\n' "$out"
  return 1
}
cd "$REPO"

for cycle in $(seq 1 "$CYCLES"); do
  echo "=== cycle $cycle: $(date -u +%H:%M:%SZ) ==="
  # ensure-gpu.sh ALREADY probes for a real GPU (nvidia-smi) and recreates
  # the runtime until one has one; probing a second time here is wrong,
  # because the bridge allows one client at a time and the allocator's own
  # probe may not have been released yet -- the second probe fails for
  # reasons that have nothing to do with the GPU.
  if ! ~/.pi/agent/skills/colab/scripts/ensure-gpu.sh "$SESSION" T4 "$ALIAS" 12 \
        >> /tmp/ensure_gpu5.log 2>&1; then
    echo "  no T4 this cycle; retrying"
    sleep 30
    continue
  fi
  echo "  T4 acquired (confirmed by the allocator's own nvidia-smi probe)"

  # Build the code tarball FRESH, every cycle. Shipping a stale tarball
  # already cost one cycle: the arms died instantly on an unrecognized
  # flag, because the tarball predated the change.
  tar czf /tmp/code_only2.tar.gz -C "$REPO" \
      gpu_port.py numpy_port.py train_gpu.py checkpoint_hub.py distill.py \
      tail_credit.py teacher_table.py telemetry.py fetch_corpus.py \
      fetch_eval_data.py test_tail.py tools 2>/dev/null
  scp_retry /tmp/code_only2.tar.gz "$ALIAS":/content/ 2>/dev/null
  scp_retry "$REPO/docs/runbooks/run-box.sh" "$REPO/docs/runbooks/fetch-corpus-box.sh" \
       "$REPO/runs/taxis/corpus_sha256.txt" "$ALIAS":/content/ 2>/dev/null
  ssh_retry 'mkdir -p /root/.cache/huggingface' 2>/dev/null
  scp_retry /tmp/hfcred/token "$ALIAS":/root/.cache/huggingface/token 2>/dev/null

  # Does THIS arm already have a checkpoint in the hub? If so the run must
  # RESUME it (not start over); if not, it starts from scratch. Asking the
  # hub is the only honest way to decide.
  ssh_retry 'cd /content && tar xzf code_only2.tar.gz && mkdir -p runs/taxis && \
      rm -f runs/DONE && \
      (bash fetch-corpus-box.sh > runs/fetch.log 2>&1; \
       FROM=$(python3 -c "
from huggingface_hub import HfApi
fs = HfApi().list_repo_files(\"bakali1/tmt-checkpoints\")
own = [f for f in fs if f.startswith((\"A_p4_wd0_c_t\", \"A_p4_wd0_fsp_c_t\"))]
print(\"1\" if own else \"\")
"); \
       export FROM_SCRATCH=$FROM; \
       setsid nohup bash run-box.sh > runs/box.log 2>&1 < /dev/null; \
       touch runs/DONE) & sleep 20; \
      tail -2 runs/fetch.log' 2>&1 | tail -3
  echo "  (the hub decides: an arm with no checkpoint of its own starts from scratch,"
  echo "   an arm with one resumes it)"
  echo "  arms launched; polling"

  # Wait for the arms, ON THE BOX, over ONE long-lived connection. The
  # Colab bridge allows a single client per runtime ("another `colab ssh`
  # is connected to this runtime"), so reconnecting every 30 s races
  # against the previous connection and the poll fails for reasons that
  # have nothing to do with the run. If this connection dies, the runtime
  # died with it, and the next cycle resumes the arms from the hub.
  ssh_retry 'cd /content && for i in $(seq 1 90); do
       if [ -f runs/DONE ]; then echo BOX_DONE; break; fi
       sleep 20
     done;
     echo "--- control ---"; tail -2 runs/taxis/A_p4_wd0_c.log 2>/dev/null;
     echo "--- fsp ---"; tail -2 runs/taxis/A_p4_wd0_fsp_c.log 2>/dev/null' 2>&1 | tail -8
  if ! ssh_retry 'true' 2>/dev/null; then
    echo "  runtime went away at $(date -u +%H:%M:%SZ)"
  fi
    # "BOTH ARMS DONE" is printed even when the arms FAILED (it is printed
    # unconditionally at the end of the script), so the completion test is
    # the arms' own final checkpoints: they must exist AND be at the target
    # step. That check caught a cycle whose arms died on a bad flag.
    if ssh_retry 'cd /content && python3 -c "
import numpy as np, os, sys
ok = True
for arm in (\"A_p4_wd0_c\", \"A_p4_wd0_fsp_c\"):
    p = f\"runs/taxis/{arm}.npz\"
    if not os.path.exists(p):
        print(f\"{arm}: MISSING\"); ok = False; continue
    t = int(np.load(p)[\"t\"])
    print(f\"{arm}: t={t}\")
    ok = ok and t >= 19000
sys.exit(0 if ok else 1)
"' 2>/dev/null; then
      echo "  BOTH ARMS REACHED THE TARGET at $(date -u +%H:%M:%SZ)"
      scp_retry "$ALIAS":/content/runs/taxis/ruler_A_p4_wd0_c.txt \
           "$ALIAS":/content/runs/taxis/ruler_A_p4_wd0_fsp_c.txt \
           "$REPO/runs/taxis/" 2>/dev/null
      echo "  rulers pulled"
      exit 0
    fi
  scp_retry "$ALIAS":/content/runs/taxis/A_p4_wd0_c.log "$REPO/runs/taxis/" 2>/dev/null
  scp_retry "$ALIAS":/content/runs/taxis/A_p4_wd0_fsp_c.log "$REPO/runs/taxis/" 2>/dev/null
  ssh_retry 'cd /content && tail -1 runs/taxis/A_p4_wd0_c.log; tail -1 runs/taxis/A_p4_wd0_fsp_c.log' 2>&1 | tail -2
  echo "  (the arms' state is banked in the hub; the next cycle resumes them)"
done
echo "CYCLE LOOP ENDED"
