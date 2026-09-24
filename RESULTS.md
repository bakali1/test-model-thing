# RESULTS — the trustworthy ruler (PLAN-2026-09-22 Phase 0)

Run id / data / seed / bytes / per-slice CE / histogram / chat verdict.
One row per checkpoint that a decision was based on. Never edit a row; append.

## Ruler

    .venv/bin/python eval.py <checkpoint.npz> [--slices 3] [--bytes 20000]

Prints: held-out CE per disjoint slice + mean ± SE, unigram/bigram baselines
from the training corpus, a same-task distribution-shift CE, cross-domain CE on
the Gutenberg novels, decay / half-life histogram, hot-dim count, and stop-head
AUC + recall/precision.

`eval.py` refuses to run if any eval file is reachable by the training glob.
The held-out corpora live physically outside `data/wikipedia_clean/`:

  * `data/eval/` — in-domain held-out. `wiki_tinystories_005.txt` (the file
    moved out at the Phase 0 split) + `tinystories_valid.txt` (see below).
  * `data/eval_shift/` — `tinystories_v2gpt4_valid.txt`, TinyStoriesV2 generated
    by GPT-4: same task, independent generation ⇒ a distribution-shift number.

### Held-out data provenance (`fetch_eval_data.py`, reproducible)

    .venv/bin/python fetch_eval_data.py

Downloads the canonical TinyStories validation splits from
`huggingface.co/datasets/roneneldan/TinyStories`, strips the `<|endoftext|>`
markers (the training corpus has none), and drops stories that are
near-duplicates of a training story (TinyStories is heavily templated, so the
official split still shares ~3% of stories with the train set). The result is
byte-identical on every run (sha256-stable).

## Phase 0 changes

  - **0.1 split.** `wiki_tinystories_005.txt` moved to `data/eval/`; the
    training glob `data/wikipedia_clean/**/wiki_*` can no longer reach it
    (4 files, 165 MB remain). `eval.py` has a hard overlap guard.
  - **0.2 `--data-seed`.** The training file shuffle is seeded (default 0) and
    its order is logged at startup, so two A/B arms share one byte stream.
  - **0.3 tagged saves.** `--save-every` writes `tmt_t{step:06d}.npz` and keeps
    the last `--keep-k` (default 8); `tmt.npz` is a hard link to the newest tag.
    A result's checkpoint is never silently overwritten.
  - **0.4 `eval.py`.** Unified ruler (above). `eval_npz.py` is kept for its
    stop-head detail and now also finds file 005 under `data/eval/`.

## Contamination caveat (important)

The live checkpoint `tmt` **trained on all five wikistories files**, including
005, before the split existed. Its CE on 005 is optimistic and NOT a
generalization number. The clean baseline is therefore measured on
`data/eval/tinystories_valid.txt` (never in the training glob) and on the
novels (cross-domain; the wiki run never read them).

## Log

Baseline command (hold fixed for future rows unless noted):

    .venv/bin/python eval.py tmt.npz --eval-glob 'data/eval/tinystories_valid.txt' --xd-files 4

| date | ckpt | t | bytes | in-domain held-out CE | shift CE | cross-domain CE | hot dims | half-life mean | note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-23 | tmt.npz | 64,747 | 57,602,080 | *(1.2235 ± 0.0103 on file 005 — CONTAMINATED, reference only)* | – | 3.3785 ± 0.2933 (12 novels) | 57 | 1.12 B | pre-split ckpt: 005 was trained on |
| **2026-09-23** | **tmt.npz** | **64,747** | **57,602,080** | **1.1741 ± 0.0415** (3×20k, `tinystories_valid`) | **1.1425 ± 0.0165** (3×20k, v2gpt4) | **3.1532 ± 0.4275** (4 novels × 8k) | 57 | 1.12 B | **clean baseline** — held-out bytes never trained on |

Stop head @ clean baseline: AUC 0.9547; thr 0.35 recall 9.6% / precision 40.4%.
Memory histogram (half-life): <1 B: 2131 | 1–2 B: 5926 | 2–4 B: 124 |
4–8 B: 9 | 8–16 B: 1 | 16–32 B: 0 | >32 B: 1.

The in-domain SE (0.04) is currently the coarse number: one held-out file, 3
slices. A win inside ~1 SE is not a win. Raise `--slices` before trusting a
sub-0.1 CE difference, or add a second held-out file.

## GPU port (PLAN-2026-09-23-GPU), night session 2026-09-23/24

| gate | result |
|---|---|
| `tests/check_parity.py` — numpy ≡ torch, 150 live steps, 12 hot dims/layer | loss rel ≤ 2.9e-07, ce ≤ 1.7e-07, worst grad rel 3.6e-03 |
| live `tmt.npz` (t=64,747, K=18, 57 hot dims) continued on both engines | loss rel 1.1e-07 |
| npz round trip torch→numpy→torch | bit-exact |
| Colab T4 bench, fresh init, fp32 | **B=1024 32,353 B/s · B=2048 32,806 · B=4096 34,259** (33× CPU ~1,000 B/s) |
| Colab T4: CUDA vs CPU torch loss, 192 hot dims (tail on GPU) | rel 1.10e-07 |
| Colab T4: save → reload → continue | exact (0.0) |
| Colab T4: `train_gpu.py` + tagged saves | PASS |

Notes: first T4 run measured 2.6k B/s — a per-step tail-membership sync copied
the ring to CPU (~0.5 GB/step, 99% of step time). Fixed by re-evaluating
membership on the documented REBASE_EVERY cadence with on-device seeding
(12× faster; parity unchanged). Next speed lever: the ring credit scatter
(~600M atomics/step) — sort + segment-sum, measured separately before adoption.

Post-review hardening (same session): parity gate now compares the
accumulators, not just loss/grads — live 150 steps: ring_b 0% mismatch (ids
exact), ring_w 8.9e-07 rel, tail_cols 0%, tail_E 8.7e-07 rel, loss 2.9e-07;
thresholds tightened to loss 1e-5 / grads+accumulators 5e-3, and post-resume
m, v, theta and accumulators are asserted too. Hub-pull load, relative
`--steps`, `--batch` default 1024, `--push`, untagged mode, atomic rolling
replace, CUDA bench sync, `TMT_TAIL` switch, 4 GB hot-dim cap, and
corpus staging swap all landed and are covered by the gates above
(see PLAN-2026-09-23-GPU.md §8 for the full review triage).

Second review response (same session): A/B tag isolation per arm; tail cap
now pre-allocates; `new_states` invalidates tail/rebase; hub content
validation with SHA-256 sidecars and loud equal-step conflicts; explicit
remote pruning; crash-safe versioned corpus publication; HF pagination with
revision + yield checks; Kaggle entrypoint (`kaggle/tmt_gpu_train.py`,
exercised locally on CPU). New gate: `tests/check_trainer.py` (6/6 PASS).

### Ruler verdict on the banked leg checkpoint (t=94,747 vs baseline t=64,747)

| metric | baseline t=64,747 | after 30k steps t=94,747 | delta |
|---|---|---|---|
| in-domain held-out CE | 1.1741 ± 0.0415 | 1.1902 | +0.016 (inside SE) |
| shift (TinyStoriesV2-GPT4) | 1.1425 ± 0.0165 | 1.1264 ± 0.0140 | −0.016 (inside SE) |
| cross-domain (12 novels) | 3.1532 ± 0.4275 | 3.4042 ± 0.2985 | +0.251 (inside SE) |
| hot dims | 57 | **115** | doubled |
| mean half-life | 1.12 B | 1.14 B | stable |
| stop-head AUC | — | 0.9710 | intact |

Statistically neutral on all three rulers, which is the correct outcome for
30.7 MB of extra training (~7% of one pass) and the PLAN's 0.03 CE bar: no
regression, no divergence, trajectory continuity held across a real GPU resume.
The informative change is **hot dims 57 -> 115**: the tail's exact-credit set
grew during the leg, which is also why the checkpoint grew 181 MB -> 241 MB
(the tail E buffer is `B x 256 x H_total x 4 B`; the eval reports 120.6 MB for
H=115, and 180.7 − 59.8 + 120.6 = 241.5 MB matches the file exactly). Hub
traffic therefore grows as training proceeds -- worth knowing before a long
Kaggle leg starts pushing 240 MB tags.

### What the diagnostics said about throughput (run 2)

- `ends=0` 23,796 B/s vs `ends=2%` 23,600 B/s → **end-of-stream flags cost
  0.8%**, so EOS is NOT the cause of the real-leg rate.
- The 32,353 B/s figure was measured on a *synthetic* checkpoint; on the live
  checkpoint the same synthetic bench gives 23,796 B/s, and the real leg
  sustains 21,477 B/s (last window). Most of the "32k vs 21k" gap was never
  about real data at all.
- The remaining decline over the run tracks the hot set doubling (57 -> 115):
  the per-step tail decay multiplies the whole `(B, 256, H)` buffer, so its
  cost is linear in hot dims. That is the concrete next optimization target,
  ahead of the ring scatter. The leg now also benches with the tail disabled
  to put a number on it.

### Live-checkpoint 30k-step leg (Colab T4, t=64,747 -> 94,747)

The first real training leg, run as an integration test. **All infrastructure
assertions passed**: checkpoint pulled from the hub with its digest verified
(`t=64747`, `ring_k=18`, 57 hot dims), 64 MB of TinyStories Range-fetched
in-session in 1 s, 30,000 steps executed with zero non-finite losses, 7 tagged
saves, rolling file hard-linked to the newest, saved checkpoint reloaded at the
live `t`, tag steps strictly inside `(64747, 94747]`. Loss drifted
**1.822 -> 1.781** with no divergence.

| measurement | value |
|---|---|
| cumulative throughput | **22,143 B/s** (23.1 min wall for 30.7 MB) |
| last-window throughput | ~22.1 kB/s (slowly declining over the run) |
| synthetic bench (ends=0, 30 steps) | 32,353 B/s |

The leg's throughput assertion FAILED against a 25,000 B/s floor taken from the
synthetic bench. That floor was the wrong reference, not a regression: the old
tail-sync bug measures 2,655 B/s, and this leg ran 8x faster than that. See
"What the diagnostics said about throughput" above for the measured cause. The
leg now reports the last-window rate, an ends=0 vs ends=2% A/B, a tail-off
comparison, and the hot-dim count at the end. Floor set to 18,000 B/s, which
still catches the 2.6 kB/s bug by a wide margin.

**Operational lesson (cost: one discarded 30k leg):** `colab run` releases the
VM when the script exits, so the 7 tags (~1.2 GB) were lost. The leg now pushes
the final tag to the hub at the end (one 181 MB push, verified locally) and
asserts the push succeeded. Run 2 did this: `tmt_t094747.npz`
(`sha=a678bfade88d`, `run=live-leg`) pushed in **12 s** from Colab (~20 MB/s,
versus 513 kB/s from the laptop uplink), pulled back locally, digest verified,
and evaluated above.

### CUDA state parity, CPU-torch vs CUDA (T4, 20 steps, B=1024, 192 hot dims)

The loss-only CUDA gate was hiding the state, so the gate now compares every
tensor. The first runs of it **failed**, and the failures were the gate working:

| tensor | floor-relative | scale-aware (max abs err / RMS) |
|---|---|---|
| `m` (Adam 1st moment) | 7.1e-02 @ floor 1e-6 · 2.8e-03 @ floor 1e-4 | **3.4e-05** |
| `g:Wd` / worst gradient | 1.9e-02 @ floor 1e-4 | **9.8e-06** (g:layer2.W) |
| `v` | 2.4e-07 | — |
| `ring_b` ids | 0 mismatched (exact) | exact |
| `ring_w` | 8.0e-07 | 3.6e-07 |
| `tail_E` | 2.4e-07 | 1.7e-06 |
| loss | 1.1e-07 | < 1e-5 (gate) |

Lesson recorded in the code: a floor-relative metric explodes on entries near
zero (Adam moments and some gradient rows sit far below their tensor's scale),
so cross-device state is now gated on **max abs error / tensor RMS < 1e-3**,
which the worst tensor misses by two orders of magnitude. The three runs this
required (all short, all Colab) are in SCHEDULE.md. Also measured and now
documented rather than hand-waved:

- **Resume is recoverable, not bit-identical.** `ring_b`/`ring_w` are not
  persisted (numpy-compatible), so a resumed model's cold-dim embed credit
  warms from step 1 and the trajectory separates: embed rel err 6.6e-02 at
  step 1 (abs/RMS 5.5e-05, loss ratio 1.00), while tail E restores exactly.
  A/B arms must not compare checkpoints taken within `ring_k` (≤18) steps of
  a resume.
- **Tail membership lags by design on GPU** (rebase every 500 steps vs numpy's
  every step): pinned by a deterministic test (numpy 4 hot, GPU 3 until
  rebase, 4 after).
- The recorded CUDA numbers now include full state parity (theta, every
  gradient, m, v, ring_b/ring_w, tail E), not just loss: `[RESULT]
  cuda-parity-state PASS` with the worst tensor at 3.4e-05 of its RMS.
