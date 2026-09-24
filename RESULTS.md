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

### Kernel/phase attribution (T4, baseline, 3 profiled steps)

`gpu_port.step_batch` now carries `record_function` phase regions (on by
default; `TMT_PROFILE_REGIONS=0` disables; verified bit-identical output with
them on). Attribution, per step (profiled wall 63 ms vs 51 ms unprofiled — the
profiler adds overhead; use the ratios):

| phase | ms/step | share of phase time |
|---|---|---|
| **ring_credit** | 21.3 | 34% |
| bwd_layers | 15.7 | 25% |
| **tail_credit** | 10.2 | 16% |
| fwd_layers | 10.2 | 16% |
| decaytrace | 3.6 | 6% |
| optimizer | 3.0 | 5% |
| loss/decoder/grad_copy | 0.9 | 1% |

Top device kernels per step (from the same run):

| kernel | ms/step | what it is |
|---|---|---|
| `elementwise_kernel` (MulFunctor) | 13.7 | ring decay, ring_vals multiply, tail `E.mul_` |
| `indexFuncLargeIndex ReduceAdd` | 9.0 | `dembed.index_add_` (17 calls/step) |
| `cudaLaunchKernel` (CPU time) | 10.1 | **1,072 launches per step** |
| `direct_copy_kernel` | 5.8 | advanced-index copies (`w[:, p] = 1.0`, tail adds) |
| **`masked_fill_kernel`** | **5.1** | **zeroing the 1.4% hot columns of the ring** |
| sgemm (`volta_sgemm_*`) | 8.3 | the actual matmuls |

**What this changes about the plan (all measured, not inferred):**

1. **The ring's cost is strided indexing, not bandwidth.** `masked_fill` alone
   is 5.1 ms/step (8% of the step) to zero columns that are 1.4% of the data.
   The cost is the non-coalesced boolean-mask write, not the bytes. This also
   explains why `a1-cold` gained nothing measurable (1.03x): it replaced a slow
   masked write with an equally slow `index_select` gather. The fix is a
   *contiguous cold-only ring layout* (physically store only the ~8,192-115
   cold columns), which my plan dismissed as a science change for a reason that
   now looks weak: a dim dropping out of the hot set would lose ≤18 steps of
   history for 1 of 8,192 columns.
2. **Launch overhead is ~10 ms/step (1,072 launches).** This is the real target
   for A3, and `torch.compile(mode="reduce-overhead")` over the whole step was
   the wrong tool (0.003x — it recompiles instead of replaying). A hand-rolled
   `torch.cuda.CUDAGraph` capture of the steady-state step, or fusing just the
   16 layer forwards/backwards, is the correct shape of that experiment.
3. **The layer stack is 41% of the step and was never targeted.** Only 8.3 ms
   of the 26 ms in fwd+bwd layers is sgemm; the rest is layer_norm, the
   `x*(1+z*sg)` activation, and state updates. Compiling *only* the layer
   function (a much smaller, static-shape graph than the whole step) is the
   obvious next candidate.
4. **The tail is 16%, not 9%** — closer to the size of the ring's decay, which
   is why `b2-tail` (factored decay) was ~1.0x: it removed the `E.mul_` traffic
   but added a per-step factor update and a division at renormalization.

### Board run 3 + the measurement noise floor (T4)

A fourth candidate was added from the profile: the gradient buffer was being
assembled with **69 separate `copy_` calls** (5 + 16x4 parameter views), which
the profiler charged to `direct_copy_kernel` at 5.8 ms/step. `grad_assembly`
now concatenates the pieces in `self.shapes` order into the flat buffer with
one `torch.cat(..., out=)`. Verified **bit-identical** (theta/m/v/grads all
`max|Δ| = 0.0` over 30 steps), and gated by `tests/check_variants.py`.

| variant | B/s | vs baseline | verdict |
|---|---|---|---|
| a2-compiled | 24,159 | **1.21x** | robust across 3 runs |
| a2c-gcat (compiled + cat) | 23,175 | 1.17x | does not stack with a2-compiled |
| b2-tail | 20,912 | 1.05x | noise-level |
| g-cat (fused grad write) | 20,508 | 1.03x | **within noise** |
| g-loop (69 copies) | 20,143 | 1.01x | reference |
| baseline | 19,895 | 1.00x | |
| c-segmented | 15,037 | 0.76x | rejected |

**The most important number from three independent board runs is the noise
floor: ±3-4%.** Baseline measured 20,632 / 20,023 / 19,895 B/s; `b2-tail`
20,868 / 19,791 / 20,912; `c-triton` 19,268 / 18,287 / 19,783. Consequences:

- `a2-compiled` (mean 23,978 vs baseline mean 20,183) is a **real +19%** — it
  clears the noise by 5x.
- `g-cat` (+1.8% over `g-loop`) and `b2-tail` (+1 to +5%) are **inside the
  noise band** and must not be claimed as wins on 12-step runs. The direct_copy
  time the profiler attributed to gradient assembly evidently also covers the
  `w[:, p] = 1.0` and tail-add copies, which `cat` does not remove.
- The two do not stack: `a2-compiled` + `g-cat` (23,175) is *slower* than
  `a2-compiled` alone (24,159), so Inductor is already absorbing some of what
  `cat` was doing.
- Any future decision on a <5% effect needs more timed steps (the board takes
  `--steps`; 12 is cheap, 40+ is what a 3% effect requires) or repeated runs.

### Phase-0 optimization board — MEASURED on T4 (2026-09-24)

`bench_variants.py` run on the live checkpoint (t=94,747, B=1024, 12 timed
steps, 2 warmups, digest-verified hub pull), one process, 6.1 min for the whole
board including the CUDA profile and the atomic micro-benchmark.

| variant | B/s | ms/step | vs baseline | peak GB | state/RMS |
|---|---|---|---|---|---|
| **a2-compiled** (Inductor-fused ring credit) | **23,933** | 42.8 | **1.20x** | 1.17 | 3.6e-05 |
| a1-cold (cold-column ring credit) | 20,679 | 49.5 | 1.03x | 1.23 | 1.4e-04 |
| baseline (legacy) | 20,023 | 51.1 | 1.00x | 1.17 | — |
| b2-tail (factored tail decay) | 19,791 | 51.7 | 0.99x | 1.18 | 7.3e-05 |
| a2-kloop | 19,298 | 53.1 | 0.96x | 1.17 | 3.6e-05 |
| c-triton (fused ring kernel) | 18,287 | 56.0 | 0.91x | 1.17 | 3.6e-05 |
| b1-triton | 18,010 | 56.9 | 0.90x | 1.78 | 3.6e-05 |
| b1-b2 | 17,170 | 59.6 | 0.86x | 1.78 | 7.3e-05 |
| b1-ring (factored ring decay) | 17,014 | 60.2 | 0.85x | 1.78 | 4.1e-05 |
| c-segmented (sort + segment-sum) | 14,105 | 72.6 | 0.71x | 1.19 | 1.3e-04 |
| a3-compile (`torch.compile` reduce-overhead) | 68 | 14,994 | 0.003x | 1.18 | 1.0e-03 |

Isolated ring-scatter micro-benchmark at production shape (one layer):
`index_add_` **0.320 ms**, sort+segment-sum **0.661 ms**.

**Verdict: exactly one candidate is worth adopting — `a2-compiled`, +20%.**
Every prediction in the optimization plan that was based on traffic arithmetic
rather than measurement was wrong, in both directions:

- **B1 (factored ring decay) is a 15% REGRESSION, not a 1.2 GB/step win.** The
  plan assumed the decay factor is `(B,K)`; it is `(B,K,D)` because `dec` is
  per-(stream, dim). Factoring therefore trades a 37.7 MB multiply for an
  identical-size factor multiply *plus* an extra multiply in the credit path,
  and costs +0.61 GB of VRAM for nothing. The implementer's own code comment
  said as much; the measurement confirms it.
- **C (kill the atomics) was aimed at a cost that does not exist.** The plan
  budgeted ~25 ms/step for atomics; the isolated measurement is 0.320 ms per
  layer, ~5.1 ms for all 16. And sort+segment-sum is **2x slower** than the
  atomics it replaces, because the gather costs more than the contention it
  saves — exactly the caveat the plan carried but still ranked too high.
- **The fused Triton kernel is slower** (0.91x), so the ring path is not
  bandwidth-bound the way the model implied; one program per (b,k) slot with a
  512-wide row is not a good access shape.
- **A1 is within noise** (1.03x), consistent with the arithmetic correction:
  hot dims are 57-115 *total* across 8,192 ring dimensions (0.7-1.4%), not the
  22% first claimed.
- **A3 (`torch.compile` reduce-overhead over the whole step) is catastrophic**
  (0.003x): graph-capture plus per-step data-dependent control flow recompiles
  instead of replaying. Must not be attempted again in this shape.

### Bugs found while verifying (all three fixed, all now gated)

1. **Factored tail added `a` instead of `a/c`** — the recurrence `E = P*c` with
   `c_{t+1} = sig*c_t` requires `P += a/c`; adding `a` over-decays every byte
   add by one factor of `sig`. Measured **35x RMS** divergence from legacy, with
   the loss moving only 3.8e-04 — invisible at the precision anyone would eyeball.
2. **Factored-ring tail seeding** multiplied the adopted `(B,K,H1)` ring gather
   by the ungathered `(B,K,D)` factor: crash on shape mismatch, and silent
   corruption of the wrong dims if `H1 == D`.
3. **Triton kernel loaded the decay factor as a scalar** (`factor[b,k,0]`) and
   scaled the whole 512-wide row by one dimension's decay: **11x RMS** state
   error. Fixed to a strided `(B,K,D)` load; b1-triton now agrees at 3.6e-05.
4. `bench_variants.py` forced a host sync per timed step (`float(loss)` inside
   the timing loop), which silently cancelled the entire point of the A3
   async-metrics candidate; and its profiler used a torch attribute removed
   after 2.6, crashing the profile section.

`tests/check_variants.py` (new, CPU, free) now gates every candidate against
legacy: ring-credit paths at 1e-4, factored at 1e-3, scale-aware. The default
legacy path is bit-identical to the committed engine over 120 steps, so
production is unaffected by any of the candidates.



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
