#!/bin/bash
# Poll the Colab box for the P4 long run and pull the results the moment they
# exist. Launch it with `setsid nohup ... &` -- a plain background job is
# killed with the shell's process group, and this watcher has to outlive the
# shell that started it.
#
# The Colab bridge tolerates ONE concurrent connection, so while this is up,
# do not ssh to tmtlong by hand; stop the watcher first.
set -u
REPO="/home/baku/programing/opensource/new-llm/test-model-thing"
OUT="$REPO/runs/taxis/colab"
SSH="ssh -i $HOME/.ssh/colab -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o ServerAliveInterval=30"
mkdir -p "$OUT"

REMOTE='cd /content && date -u +%H:%M:%SZ; for f in A_p4_wd0 B_default_wd001 C_p4_wd001; do echo "[$f] $(tail -1 runs/taxis/$f.log 2>/dev/null)"; done; echo "[markers] $(grep -o "P4 LONG ARMS DONE\|P4 LONG RULERS DONE" runs/p4long.log 2>/dev/null | tr "\n" " ")"; exit 0'

while :; do
  $SSH tmtlong "$REMOTE" > "$OUT/status.txt" 2> "$OUT/status.err"
  cat "$OUT/status.txt" >> "$REPO/runs/taxis/watch.log" 2>/dev/null
  if grep -q "P4 LONG RULERS DONE" "$OUT/status.txt" 2>/dev/null; then
    $SSH tmtlong 'cd /content && cat runs/taxis/ruler_A_p4_wd0.txt runs/taxis/ruler_B_default_wd001.txt runs/taxis/ruler_C_p4_wd001.txt; exit 0' \
        > "$OUT/rulers.txt" 2>&1
    for n in A_p4_wd0 B_default_wd001 C_p4_wd001; do
      $SSH tmtlong "cd /content && cat runs/taxis/$n.log; exit 0" > "$OUT/$n.log" 2>&1
    done
    echo "pulled $(date -u +%H:%M:%SZ)" >> "$REPO/runs/taxis/watch.log"
    break
  fi
  sleep 200
done
