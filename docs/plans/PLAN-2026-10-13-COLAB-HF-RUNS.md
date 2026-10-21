# PLAN: Colab-only training, with the private HF repo as the durability layer

**Date:** 2026-10-13 (revised after the Colab runtime was recycled mid-run
and the MX150 was rebooted under the FSP arm — together, ~4 h of GPU time
and one 20.5 MB arm, both lost to infrastructure, not to the experiment).

**What this plan is:** the protocol for every future run. It changes *where*
runs happen and *where their checkpoints live*. It does not change what is
being measured, and it does not change any result already in `RESULTS.md`.

---

## 0. The rules, and where they come from

| rule | source |
|---|---|
| Only **code** crosses the wire from this PC. | the user; and `fetch_corpus.py`'s own docstring: *"No data is ever shipped from the laptop: downloads happen where the training happens. Only the SPEC (this file) is code."* |
| The **training data is fetched on the box** (it has Google's network). | the user |
| Every run **starts from scratch**. | the user |
| Every **checkpoint goes to `bakali1/tmt-checkpoints`** (private), so a recycled runtime costs nothing. | the user |
| This PC's remaining jobs: the code, the plan/record, the CPU gates, and reading the (tiny) result text. | the consequence of the above |

**The network, measured:** 14.44 Mbit/s down, 1.91 Mbit/s up. So: a 300 KB
code upload is a second; a 165 MB data upload is 11 minutes and is *not*
allowed by the rules above; a 100 MB checkpoint download is 7 minutes and is
unnecessary if the checkpoints live in the HF repo. The rule set is therefore
also the *fast* rule set — the box's network does all the heavy transfers.

---

## 1. The durability mechanism already exists

`checkpoint_hub.py` is built and gated, and it is exactly what is needed:

* **push** validates content before uploading — the npz must be readable and
  its internal `t` must match the step encoded in the filename (a corrupt or
  mislabelled checkpoint is refused, not stored);
* a `.sha256` sidecar is written alongside, and a pull verifies it;
* **pull** takes the best tagged `t` in the repo, so a *restarted* run
  continues from the last checkpoint rather than from nothing;
* the trainer already wires it: `--hub-repo <id>` pulls before training and
  pushes at the end, and `--hub-save-every N` pushes every N steps.

So the run contract becomes:

```
TMT_HUB_REPO=bakali1/tmt-checkpoints \
TMT_TOKEN=…  (or `hf auth login` ON THE BOX — see §6) \
python3 train_gpu.py <ckpt>.npz train --hub-repo bakali1/tmt-checkpoints \
    --hub-save-every 2000 …
```

`--hub-save-every 2000` is chosen from a measured constraint, not a guess:
the trainer's data stream position is **not** checkpointed, so a run that
resumes restarts the corpus from the beginning and re-sees the bytes it had
already trained on. A 2,000-step cadence at B=1024 is 2.0 MB, so a recycled
runtime costs at most a **1.2% data repeat** out of a 165 MB epoch — small
enough to ignore, and *recorded* in the run's log rather than hidden.

---

## 2. The data, fetched on the box

| artefact | size | source | notes |
|---|---:|---|---|
| `data/eval/tinystories_valid.txt` | 18.8 MB | `roneneldan/TinyStories` → `TinyStories-valid.txt` | the repo's own `fetch_eval_data.py` does this ✔ |
| `data/eval/wiki_tinystories_005.txt` | 41.3 MB | same provenance as the training shards (below) | the second held-out file, needed for the incumbent-comparable CE protocol (6 slices × 20,000 B over **both** files) |
| `data/wikipedia_clean/wikistories/wiki_tinystories_00{1..4}.txt` | 165 MB | same provenance | the training corpus |

**The one open item is the provenance of the four training shards.** They are
TinyStories text (the directory name `wikipedia_clean/wikistories` is legacy),
split into four ~41.3 MB files of *unequal* size, so the split was by
document/row boundaries, not by a fixed byte count — and the rule that made
it is not recorded in the repo. The plan therefore has an explicit
provenance step, with an oracle that needs no guessing:

1. **now, locally (no network):** compute the sha256 of the four local shards
   and of the two eval files. These are the oracle.
2. **on the box:** download the TinyStories training data from HF and rebuild
   the four shards with the candidate rules (parquet row order; text
   concatenation with/without a separator; split at 41.3 MB boundaries).
3. **compare the sha256s.** A match means the box's corpus is byte-identical
   to the incumbent's and to the salvaged arm's → full comparability ✔.
4. **if nothing matches:** the arms on Colab use the *rebuilt* corpus
   consistently (Colab-vs-Colab comparisons stay clean ✔) and the corpus
   difference is recorded; **or** the user pushes the 165 MB once to
   `bakali1/tmt-checkpoints` (11 minutes at 1.9 Mbit/s, once) and the box
   downloads it → byte-identical to everything prior ✔. This is the one
   exception to "nothing but the code leaves the PC", and it is the user's
   call.

---

## 3. The runs

All from scratch, on the box, at the incumbent's recipe (B=1024, T=1,
`ring_k 18`, the 4-file corpus, data seed 0, seed 0, `TMT_TAIL=0`):

| arm | init | `wd_decay` | FSP | what it is |
|---|---|---:|---|---|
| `A_p4_wd0` | 2…512 B | 0.0 | — | the **control** |
| `A_p4_wd0_fsp` | 2…512 B | 0.0 | α=1.0, r=16, τ=64, pos_weight 8.5 | the **paper's objective** |
| `C_p4_wd001` | 2…512 B | 0.01 | — | the erosion arm (optional) |
| `B_default_wd001` | default | 0.01 | — | the default-init control (optional) |

**The A/B is `A_p4_wd0` vs `A_p4_wd0_fsp`** — same init, same wd, same
corpus, same seed, same byte budget, one flag different. The two optional
arms fill in the rest of the 2×2.

**Budget and parallelism (the T4 measures ~30 kB/s at B=1024):**

| plan | wall clock | note |
|---|---:|---|
| 1 epoch (165 MB) per arm, two arms in parallel | ~3.5 h | the aggregate GPU is shared, so parallel ≈ sequential in total |
| 1 epoch per arm, sequential | ~3 h | simpler, and the A/B's first arm is done early |
| 0.5 epoch (82 MB) per arm | ~1.75 h | the fast version; the MX150's salvaged 20.5 MB arm already shows the retention curve is *flat* from 8.4 → 20.5 MB, so the curve is not what 0.5 epoch would buy |

**The instrument is the ruler**, run on the box (CPU): the retention curve
(d=8/32/128/512), the selective/associative copy cells, the held-out CE, and
the new FSP-bag cell (the head vs the static-prior baselines). The ruler's
text output is ~50 KB → pulled to the PC in a second. The checkpoints stay in
the HF repo.

---

## 4. What a recycled runtime costs

| event | cost |
|---|---|
| the runtime is recycled mid-arm | ≤2.0 MB of re-trained bytes (the data repeat), then the run resumes from the hub's best tagged checkpoint ✔ |
| the runtime is recycled between arms | nothing: the next arm starts from scratch ✔ and the previous arm's checkpoint is in the repo ✔ |
| the run completes | the final tagged checkpoint is in the repo; the ruler's text is pulled ✔ |

Nothing is lost because the *only* thing that lived on the volatile filesystem
is the working checkpoint, and it is pushed every 2,000 steps.

---

## 5. The arms that already exist, and what happens to them

The three salvaged checkpoints (the recycled Colab session's arms at ~20.5 MB,
and the MX150's 8.39 MB arms) stay **local** and are not uploaded. They are
already in `RESULTS.md`, and the salvaged arm-A ruler output (the 20.5 MB
point of the retention curve) is already recorded. If the Colab arms are run
from scratch, they supersede the salvaged ones and the local copies become
the historical record.

---

## 6. The credential, handled so it never appears anywhere

`bakali1/tmt-checkpoints` is private, so a write token is needed. The safest
path, which keeps the token out of this repository, out of every log and out
of every command line:

* the user runs `hf auth login` **on the box** (one command, the token is
  typed into the box's own shell, and `checkpoint_hub` falls back to the
  cached login when `TMT_TOKEN` is unset ✔); **or**
* the user pastes the token to me and I set it as an environment variable
  inside the Colab session only — never in a file in the repo, never in a
  command line (it would be in the process list), never in a log.

My recommendation is the first: `hf auth login` on the box, once, and the
token never touches this machine.

---

## 7. What I need from the user before the first run

1. **The credential** — `hf auth login` on the box (recommended), or a token
   I set as a session-only env var.
2. **The budget** — 1 epoch per arm (the full answer, ~3 h) or 0.5 epoch
   (~1.75 h), and whether the two optional arms are in scope.
3. **The data provenance fallback** — may the 165 MB shards be pushed to the
   repo once if the rebuild does not reproduce the sha256s? (Only needed if
   step 2 above fails.)
4. **Confirmation to start** — nothing is launched before that, per the
   standing rule.

---

## 8. The summary

```
   this PC                    the Colab box                 HF (private)
   ────────                   ─────────────                 ──────────
   code  ──── 300 KB ───────▶ fetch the data  ───────────▶  checkpoint
   plan  ──── text  ────────▶ train (from scratch) ───────▶  every 2000
   gates ──── CPU, no net ──▶ ruler (CPU)                  steps
                                │                                    │
   results ◀─── 50 KB text ───┘                                    │
   record                                                            ▼
                                                              nothing is lost
```

The one thing this plan cannot protect against is the MX150's own
availability — and at 3.4 kB/s it is a 13-hour machine for a 1-epoch arm,
which is why the runs belong on the box.

---

## 9. The C run, as it actually happened (2026-10-21)

This section is the plan's own audit: what the protocol had to survive, and
what it had to be corrected for.

**The free tier fought back.** For most of the day the allocation was
refused outright (`Allocation refused (precondition failed)`, ~16 attempts),
and the runtimes that did come were recycled *within minutes* — twice inside
a single launch command. One of them was so short-lived that the SSH bridge
landed on a container with no GPU at all (`nvidia-smi: command not found`),
which is a failure mode the allocator handles (it recreates the runtime) but
which is easy to mistake for "the GPU is gone".

**The protocol needed three corrections, all of them silent-failure kinds.**

1. *A stale code tarball.* The arms died in the first second on
   `unrecognized arguments: --stop-at-t` — the tarball had been built by hand
   and predated the change. The cycle now builds it fresh, from the working
   tree, every cycle.
2. *A completion test that believed a failed arm.* `BOTH ARMS DONE` is
   printed unconditionally at the end of the arm script, so a cycle whose
   arms died in the first second looked like a success. The test is now the
   arms' own final checkpoints: they must exist and be at the target step.
3. *The bridge is exclusive.* Colab allows one SSH client per runtime
   ("another `colab ssh` is connected to this runtime"), so a polling loop
   that reconnects every 30 s races itself. The wait now happens on the box
   over a single long-lived connection. Every connection in the cycle goes
   through a retry, and the cycle trusts the allocator's own `nvidia-smi`
   probe instead of probing a second time (the second probe was being
   refused — for a reason that had nothing to do with the GPU).

**And the hub's pull was broken in a way that would have silently corrupted
the experiment.** `best_t` was never assigned in the arm-aware patch, so
every pull that reached it died; and the arm's own files were matched by
*prefix*, which is too loose — the new arm `A_p4_wd0_c` starts with
`A_p4_wd0_`, so a prefix match would have handed it the first run's control
checkpoint. Both are fixed, and both are now tested against the real repo:
`A_p4_wd0` resumes its own t=10000, `A_p4_wd0_c` declines.

**What made the run survive the recycling** is the pair of mechanisms this
plan was written around, plus one that was added to it:

* the checkpoint's **data position** — a resume continues the corpus instead
  of replaying it (without it, the arm's unique data is capped at the push
  cadence: 10 MB seen sixteen times);
* **`--stop-at-t`**, an absolute step target, so a resumed run finishes at
  exactly the same step and the byte budget cannot drift;
* the **cycle loop** itself: allocate → ship → fetch the corpus (4 s on the
  box) → resume both arms from the hub → wait for the runtime to die →
  repeat.

Both arms reached t=19,000 and were pushed. The verdict is in
`RESULTS.md` ("C: the P4-vs-FSP A/B on a T4").

**What the infrastructure cost, for the record:** roughly five hours of
fighting the free tier, and about nine bugs that each looked like a
successful run until something was measured. The most valuable of them was
the one that was *not* an infrastructure bug at all: the FSP arm's null
result came with a measurement showing the objective had been satisfied by
the byte-frequency prior, and that is now the project's most useful single
number.
