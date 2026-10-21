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
cd "$REPO"

for cycle in $(seq 1 "$CYCLES"); do
  echo "=== cycle $cycle: $(date -u +%H:%M:%SZ) ==="
  ~/.pi/agent/skills/colab/scripts/ensure-gpu.sh "$SESSION" T4 "$ALIAS" 12 \
      >> /tmp/ensure_gpu5.log 2>&1
  if ! $SSH "$ALIAS" 'nvidia-smi --query-gpu=name --format=csv,noheader' 2>/dev/null \
        | grep -q T4; then
    echo "  no T4 this cycle; retrying"
    sleep 30
    continue
  fi
  echo "  T4 acquired"

  $SCP /tmp/code_only2.tar.gz "$ALIAS":/content/ 2>/dev/null
  $SCP "$REPO/docs/runbooks/run-box.sh" "$REPO/docs/runbooks/fetch-corpus-box.sh" \
       "$REPO/runs/taxis/corpus_sha256.txt" "$ALIAS":/content/ 2>/dev/null
  $SSH "$ALIAS" 'mkdir -p /root/.cache/huggingface' 2>/dev/null
  $SCP /tmp/hfcred/token "$ALIAS":/root/.cache/huggingface/token 2>/dev/null

  # Does THIS arm already have a checkpoint in the hub? If so the run must
  # RESUME it (not start over); if not, it starts from scratch. Asking the
  # hub is the only honest way to decide.
  $SSH "$ALIAS" 'cd /content && tar xzf code_only2.tar.gz && mkdir -p runs/taxis && \
      (bash fetch_corpus_box.sh > runs/fetch.log 2>&1; \
       FROM=$(python3 -c "
from huggingface_hub import HfApi
fs = HfApi().list_repo_files(\"bakali1/tmt-checkpoints\")
own = [f for f in fs if f.startswith((\"A_p4_wd0_c_t\", \"A_p4_wd0_fsp_c_t\"))]
print(\"1\" if own else \"\")
"); \
       export FROM_SCRATCH=$FROM; \
       setsid nohup bash run_box.sh > runs/box.log 2>&1 < /dev/null) & sleep 30; \
       tail -3 runs/box.log' 2>&1 | tail -4
  echo "  FROM_SCRATCH=$FROM (empty means: resume from the hub)"
  echo "  arms launched; polling"

  # wait for the arms to finish, or for the runtime to die
  for _ in $(seq 1 60); do
    sleep 30
    if ! $SSH "$ALIAS" 'true' 2>/dev/null; then
      echo "  runtime went away at $(date -u +%H:%M:%SZ)"
      break
    fi
    if $SSH "$ALIAS" 'grep -q "BOTH ARMS DONE" /content/runs/box.log 2>/dev/null'; then
      echo "  BOTH ARMS DONE at $(date -u +%H:%M:%SZ)"
      $SCP "$ALIAS":/content/runs/taxis/ruler_A_p4_wd0_c.txt \
           "$ALIAS":/content/runs/taxis/ruler_A_p4_wd0_fsp_c.txt \
           "$REPO/runs/taxis/" 2>/dev/null
      echo "  rulers pulled"
      exit 0
    fi
  done
  $SCP "$ALIAS":/content/runs/taxis/A_p4_wd0_c.log "$REPO/runs/taxis/" 2>/dev/null
  $SCP "$ALIAS":/content/runs/taxis/A_p4_wd0_fsp_c.log "$REPO/runs/taxis/" 2>/dev/null
  $SSH "$ALIAS" 'cd /content && tail -1 runs/taxis/A_p4_wd0_c.log; tail -1 runs/taxis/A_p4_wd0_fsp_c.log' 2>&1 | tail -2
  echo "  (the arms' state is banked in the hub; the next cycle resumes them)"
done
echo "CYCLE LOOP ENDED"
