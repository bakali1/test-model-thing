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
(see docs/plans/PLAN-2026-09-23-GPU.md §8 for the full review triage).

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

### Board run 4 — 200 steps, REAL TinyStories, live HF checkpoint (t=94,747, 115 hot)

The benchmark now trains on the real thing: the 200-step board loads
`tmt_t094747.npz` from the private hub (241 MB, 115 hot dims — the checkpoint
the live leg banked) and feeds a 64 MB Range-fetched slice of
`TinyStories-train.txt` shaped through `numpy_port.Runtime.make_streams`,
exactly as `train_gpu` does. Measured end-of-stream density is **0.63%** — the
synthetic bench had assumed 2%, a 3x overstatement.

| variant | B/s | ms/step | vs baseline | peak GB | state/RMS |
|---|---|---|---|---|---|
| **a2-compiled** | **27,706** | 36.96 | **1.27x** | 1.18 | 5.4e-04 |
| a2c-gcat (compiled + grad-cat) | 27,331 | 37.47 | 1.26x | 1.18 | 3.1e-04 |
| c-triton | 23,682 | 43.24 | 1.09x | 1.18 | 3.9e-04 |
| b1-triton | 22,698 | 45.11 | 1.04x | 1.78 | 5.7e-04 |
| baseline | 21,771 | 47.04 | 1.00x | 1.17 | — |
| g-cat | 21,667 | 47.26 | 0.995x | 1.18 | 3.0e-04 |
| b2-tail | 21,354 | 47.95 | 0.98x | 1.18 | 3.8e-04 |
| a1-cold | 20,779 | 49.28 | 0.95x | 1.23 | 2.1e-03 |
| g-loop | 20,415 | 50.16 | 0.94x | 1.18 | 4.0e-04 |
| a2-kloop | 19,540 | 52.41 | 0.90x | 1.18 | 3.9e-04 |
| b1-ring | 18,301 | 55.95 | 0.84x | 1.78 | 3.9e-04 |
| b1-b2 | 17,943 | 57.07 | 0.82x | 1.78 | 4.5e-04 |
| c-segmented | 17,592 | 58.21 | 0.81x | 1.19 | 1.8e-03 |

Run-to-run agreement between the two 200-step runs is ~1% for the leaders
(baseline 21,681 / 21,771; a2-compiled 27,264 / 27,706), so **200 steps gives a
~1% noise floor versus ~3-4% at 12 steps** — the step count the user asked for
is what makes the small effects readable. At that resolution:

- `a2-compiled` is **+27%** and is the only large, reproducible win.
- `g-cat` (+6% over `g-loop`) and `c-triton` (+9%) are now **above** the noise
  floor and worth keeping as candidates; `c-triton` flipped from a 0.91x loss at
  12 steps to a 1.09x win at 200, which is why the earlier number was not
  trustworthy.
- `b2-tail`, `a1-cold`, `b1-*` and `c-segmented` remain neutral-to-negative.

### Three more bugs in the factored tail, all found only at 200 steps

Running long enough to cross a renormalization boundary exposed defects that
short runs never reached — and one of them had been passing:

1. **`end_step` divided instead of multiplied.** The effective accumulator is
   `E * c`, so folding `c` away with `c' = 1` requires `E' = E * c`. The
   `E.div_(c)` version jumped the tail state by **~50x RMS on the first
   renorm step**. The 60-step gate had passed at 3e-07 purely because with the
   old fixed 100-step interval no renorm ever landed inside the test window.
2. **Fixed renormalization interval underflowed.** 100 steps at the live
   model's `sigma_min ~ 0.205` is `0.205^100 ~ 1e-70`, below fp32's ~1e-38, so
   `c` hit zero and the board reported **NaN losses** for `b2-tail`/`b1-b2`.
   The interval is now derived from `sigma_min` (29-60 steps depending on the
   checkpoint), which bounds `c >= 1e-20` by construction.
3. **Renormalization ran after `add_byte`**, which would rescale the byte just
   added (stored as `1/c`, so resetting `c` gives it weight `a/c` instead of
   `a`). Fixed by renormalizing after the credit and before the add.

`tests/check_variants.py` now also runs `b2-tail` for 300 steps and asserts no
non-finite loss, so a renorm boundary is always inside the gate's window.

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
required (all short, all Colab) are in docs/runbooks/SCHEDULE.md. Also measured and now
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

---

## From-scratch corpus experiment (2026-09-24) — the unconfounded Experiment D

### Why from scratch instead of continuing the live checkpoint

The incumbent (`tmt_t094747`, 94,747 steps) was trained on **TinyStories**.
A continuation A/B of "incumbent data vs novels" is therefore confounded: the
incumbent arm is simply *more of what the model already knows*, and any
inherited exposure to the arm's own data contaminates the verdict. Every arm
instead starts at **t=0 from a bit-identical seeded init** (`--seed 0`,
`default_rng(0)`), so the corpus is the only axis that differs.

This also tests the two live hypotheses about the memory wall (8,064 of 8,192
dims stuck at 1-byte half-life) that a continuation run structurally cannot:
is it architectural (from scratch hits it too) or induced by the single-layer
corpus (a different mix grows more long-memory dims)?

### Arms

| arm | corpus | source |
|---|---|---|
| **FS-W** | TinyStories, Range-fetched at a **+220 MB offset** (past the locally staged 165 MB, so it cannot overlap the ruler's decontaminated eval) | `roneneldan/TinyStories` |
| **FS-N** | Gutenberg narrative prose | `Despina/project_gutenberg` · fiction_books (MIT) |
| **FS-L** | layered mix: Gutenberg + FineWeb-Edu (score>=3) + UltraChat | 4.32 GB, staged in-session |

### Result — W vs N, byte-matched at exactly t=60,000

| arm | t | bytes seen | in-domain CE | novels CE | hot dims | mean half-life |
|---|---|---|---|---|---|---|
| **FS-W** | 60,000 | 61,440,000 | **1.2097 +/- 0.0223** | 3.103 | **128** | 1.13 B |
| **FS-N** | 60,000 | 61,440,000 | 1.7566 +/- 0.0361 | **2.141** | 54 | 1.07 B |

Identical bytes (61,440,000), identical steps, identical init. The in-domain
gap is **0.5469 CE — 18x the 0.03 bar**, so this is a real separation, not
noise. Ruler setting: 3 slices x 20,000 B, identical for both arms.

### The finding: a trade-off, not a winner

- **FS-W wins in-domain by 0.547 CE.**
- **FS-N is 0.962 CE better on novels** — a large, real cross-domain advantage.
- **FS-W grows 128 hot dims vs FS-N's 54** — TinyStories drives 2.4x more
  long-memory dims from the identical compute budget.

The two corpora buy **different things**: TinyStories buys in-domain fit and
long-memory dim growth; novels buy transfer. **The plan's working assumption
that Gutenberg would simply become the corpus-v2 base is not supported** — it
costs 0.55 CE in-domain and 74 hot dims. Whether the layered mix (FS-L) buys
both is the open question, and it is the arm that decides the mix.

### Budget discipline

One hour of wall clock per arm, with training length **derived from a measured
in-session calibration** rather than guessed, and clamped by both the time and
the data envelope. FS-W ran 60,058 steps at 28.5 kB/s; FS-N was held to the
same step count for matching.

### NOT tested / known gaps

- **FS-L was not in this comparison** (it was still running when these arms
  were scored). The verdict is W-vs-N only.
- This is **arm-vs-arm**, not vs the incumbent. A t=0 run is missing the
  incumbent's 64,747 steps of history, so an absolute comparison at equal wall
  clock would be unfair. FS-W's 1.2097 vs the incumbent's 1.1741 is *not* a
  meaningful difference.
- One seed (0), one batch (1024), 60k steps. No seed-variance estimate, so the
  0.547 CE gap has no run-to-run error bar.
- The novels arm is Gutenberg only, not the full corpus-v2 layer set.
- Contamination guard for FS-W is by byte offset, not by the near-duplicate
  filter `fetch_eval_data.py` applies; a residual risk remains.

## Bugs found 2026-09-24 that would have produced confidently wrong numbers

Each of these produced a *plausible-looking* result. The ruler is cheap to run
and expensive to trust blindly.

1. **`checkpoint_hub.pull(dest)` is a "get the newest remote" API** — it
   ignores the filename it is handed. Requesting the arm tags downloaded
   `tmt_t094747.npz` (the incumbent) into *both* `fs_fs-w_t060000.npz` and
   `fs_fs-n_t060000.npz`. The verdict would have compared the incumbent
   against itself. Fixed: exact `hf_hub_download` + sha256 sidecar verify.
2. **HF downloads are relative symlinks into the cache.** Relocating one with
   `os.replace` left a dangling link, and `np.load` failed with
   `FileNotFoundError` on a path that plainly existed. Fixed: resolve the real
   blob, copy it.
3. **The eval cache key omitted the eval args**, so a transcript produced at
   `--slices 1 --bytes 2000` was served for a ruler-grade request and printed
   under the ruler's heading. Fixed: args are hashed into the cache filename.
4. **Two sign errors in the verdict logic.** `best` is the *minimum* CE, so the
   margin is `other - best`; the code computed `best - other`, which is always
   negative and therefore reported a **0.52 CE gap as "within noise"**. The
   cross-domain guard had the mirror-image error, so a winner that was 0.96 CE
   *worse* on novels was not flagged.
5. **A smoke test pushed 18 toy checkpoints to the live private hub**, because
   `huggingface_hub` picks up the cached login implicitly and SMOKE mode did
   not disable it. Deleted; the guard is now `HF_HUB_OFFLINE=1` plus no
   `--hub-repo`.
6. **Calibration measured 1,081 B/s against a true ~27,500** because the 24-step
   timing window included Inductor's compile time. A second layer remained:
   under a second of GPU work does not let a T4 leave idle clocks. Fixed with
   `--warmup` and 200-step calibration.
7. **`train_gpu --push` uploads every tagged save from inside the training
   loop** (240 MB each). At ~50 tags that burns the budget and lets a stalled
   upload block training — FS-N hung this way at t=86,000. Fixed: the bundle
   pushes only the final tag.
8. **`train_gpu` still passed the deleted factored kwargs to `GpuModel`** after
   the optimization cleanup, so *every* GPU training run raised `TypeError`.
   This is the exact gap the "train_gpu integration — NOT TESTED" row
   predicted.
9. **`--seed` was a silent no-op** against a hardcoded `default_rng(0)`, so a
   seed sweep would have appeared to do nothing.
10. **The ruler prints thousands separators** (`60,000`); a `\\d+` regex left
    `t` and `bytes` as `None` and crashed the table mid-render.

## Optimization outcome (2026-09-24)

The +27% Inductor-fused ring credit is now the **default** on CUDA
(`ring_credit="auto"`), verified bit-identical to the pre-optimization engine
(`max|dtheta| = 0.0` over 150 steps) and holding 28,696 B/s across a tail
rebase with no recompile stall. Six measured losers were removed outright
(factored ring 0.84x, factored tail 0.98x, cold-column 0.95x, k-loop 0.90x,
sort+segment 0.81x, whole-step compile 0.003x), and `load()` now refuses
factored checkpoints rather than silently misreading them. Full detail,
including what was deliberately NOT tested, is in
`docs/optimization/OPTIMIZATION_TEST_STATUS.md`.

## Three-way result (2026-09-24, later the same day) — the layered mix LOSES

All three arms have a pushed tag at exactly **t=60,000**, so the comparison is
byte-matched: identical 61,440,000 bytes, identical init, ruler at 3 slices x
20,000 B for every arm.

| arm | t | bytes seen | in-domain CE | novels CE | hot dims | mean half-life |
|---|---|---|---|---|---|---|
| **FS-W** TinyStories | 60,000 | 61,440,000 | **1.2097 +/- 0.0223** | 3.103 | **128** | 1.13 B |
| **FS-N** Gutenberg | 60,000 | 61,440,000 | 1.7566 +/- 0.0361 | **2.141** | 54 | 1.07 B |
| **FS-L** layered A+B+C | 60,000 | 61,440,000 | 1.8491 +/- 0.0160 | 2.146 | **40** | 1.06 B |

Bar = 0.03 CE.

- **FS-W vs FS-N: +0.5469 in-domain for W, but W is +0.962 worse on novels.**
  Separated both ways, a genuine trade-off.
- **FS-W vs FS-L: +0.6394 in-domain for W, but W is +0.957 worse on novels.**
- **FS-L does not beat FS-N on novels**: 2.146 vs 2.141 is a 0.005 difference,
  far inside the bar. The mix buys nothing over novels alone.

### CORRECTION (added after the dilution control) — the mix is not simply worse

The verdict above was written from the t=60,000 point alone and it **over-read
the evidence**. A later control at matched compute overturns its headline. Do
not act on the original "dominated" wording.

FS-N's own tag history makes the control free: at t=28,000 it has seen
28,672,000 bytes, within 2.4% of FS-L's 29.35 MB *per domain* at t=86,000.

Matched trajectory, identical bytes and init at every row:

| t | FS-N in-dom | FS-L in-dom | FS-N novels | FS-L novels | N hot | L hot |
|---|---|---|---|---|---|---|
| 28,000 | 1.6600 ± 0.0184 | 1.6438 ± 0.0184 | 2.030 | 2.056 | 5 | 10 |
| 60,000 | 1.7566 ± 0.0361 | 1.8491 ± 0.0160 | 2.141 | 2.146 | 54 | 40 |
| 86,000 | 1.6059 ± 0.0217 | 1.8600 ± 0.0206 | 1.966 | 2.098 | 91 | 99 |

Three things follow, and the first one reverses the earlier claim:

1. **At matched compute the mix is NOT worse.** At t=28,000 the arms are
   statistically indistinguishable: 0.0162 in-domain (FS-L nominally *ahead*)
   and 0.026 on novels, both inside the 0.03 bar. The "layered mix is
   dominated" reading does not survive its own control.
2. **Both arms regress between 28k and 60k** (FS-N 1.660 -> 1.757, FS-L 1.644
   -> 1.849). This is a *shared* phenomenon, not a mix property, and it
   coincides with the hot-set explosion (5/10 dims at 28k -> 54/40 at 60k).
   Expanding tail credit appears to cost transient loss while the new hot set
   is learned. Neither arm escapes it quickly, and this was not predicted.
3. **FS-N recovers by 86k (-0.151); FS-L had not (-0.011).** Since FS-N's
   recovery took ~26k steps from the trough, FS-L's absence of recovery by
   86k is *not* yet evidence that it never recovers -- the runs are too close
   together to tell a delayed curve from a flat one. **This is exactly the
   "the mix needs more time" hypothesis, and it remains open.**

### Status of the corpus-v2 decision

**Open, not decided.** The honest summary is:

- The mix is not harmful at matched compute (t=28,000 tie).
- It buys no in-domain or novel-domain advantage at 60k or 86k.
- It **does** grow long-memory dims faster and overtakes FS-N on hot dims by
  86k (99 vs 91) — the one axis where it leads, and the axis least tied to
  the ruler.
- There is a shared, unexplained mid-run regression around hot-set growth that
  deserves its own investigation; it may matter more for long runs than the
  corpus choice does.

FS-L's last pushed tag is **t=92,000** (its Colab session has ended), which is
past FS-N's last tag of 86,000, so **86,000 is the largest step the two arms
share** and the comparison above cannot be extended without a further FS-N run.

### What would actually settle it

A longer FS-L and FS-N pair (say 200k steps each) to see whether FS-L's curve
is delayed-but-parallel or genuinely flat, plus an investigation of the
hot-set-growth regression. Neither is a one-hour run.

### The most important caveat — the ruler is in-domain TinyStories

The held-out set is TinyStories-dominated, so **FS-W winning in-domain is
partly tautological**: it is the only arm trained on that distribution. The
informative signals are the non-tautological ones, and they are consistent:
FS-W is ~0.96 CE worse on novels, and FS-L ties FS-N there while having the
fewest hot dims.

Two further limits on reading the mix's failure:
- **Spread-thin confound.** At 60k steps every arm has seen 61.4 MB, so the
  3-domain mix has seen ~20 MB per domain against 61 MB for the single-domain
  arms. A mix may need longer before it pays off; 60k steps cannot distinguish
  "the mix is worse" from "the mix converges more slowly."
- **The novel probe is 12 books** (~8 KB each). It separates W from N/L by a
  wide margin, but is too small to rank N against L.

### Not tested here

Seed variance (one seed, so the 0.55-0.64 CE gaps have no run-to-run error
bar), B=4096, longer horizons, per-domain data scaling, and a ruler with a
balanced multi-domain eval set. FS-L also completed a longer 97,426-step run
that is **not** step-matched to the others and is reported separately.

## Qualitative inspection — generated text (2026-09-24)

CE averages hide mode collapse. `chat_models.py` samples all four checkpoints
from the same prompt with the same seed, on CPU, in under a second each.

Same prompt, seed 3, in-domain TinyStories opening:

| model | output character |
|---|---|
| **incumbent** (t=94,747) | simple children's-story English with real dialogue structure: `he was so happy and said, "I want to make sorry...` |
| **FS-W** (t=60,000) | same register and comparable quality: `to her mom and dad said "I got... minking in the grass. The bird was so happy` |
| **FS-N** (t=86,000) | **emitted 16 consecutive newlines, then** verbose Latinate pseudo-Victorian: `...on the familiately as the man of the reason of the strong of the home of such familiaty...` |
| **FS-L** (t=86,000) | same Latinate/formal register as FS-N, slightly less structured: `...the motion for store the differing the sights conductions of the permitation...` |

**The arms faithfully learned their corpora's register**, which is the clearest
positive signal in the whole experiment: FS-N and FS-L produce formal
nineteenth-century-sounding prose without dialogue or character reference,
while the TinyStories-trained models produce short dialogue-driven stories.
The training worked; the corpora differ in *kind*, not just difficulty.

### Degeneracy across 6 seeds (250 bytes each)

| model | max newline run | fully degenerate | unique bytes | mean entropy |
|---|---|---|---|---|
| incumbent | 1 | 0/6 | **32.5** | 0.20 |
| FS-W | 0 | 0/6 | 28.2 | 0.22 |
| FS-N | **16** | 0/6 | 26.3 | 0.28 |
| FS-L | 1 | 0/6 | 25.2 | 0.29 |

- **FS-N has a newline-collapse mode** (16 consecutive newlines) that the
  other three never approach. Plausibly a paragraph-boundary attractor in
  Gutenberg. It is *invisible to CE*, which is exactly the class of defect
  this check exists to catch. Not yet explained; not yet fixed.
- **Unique-byte count tracks the CE ranking exactly** (32.5 > 28.2 > 26.3 >
  25.2), an independent confirmation that the ruler is not misleading here.
- FS-L is the least lexically diverse of the four.
- The from-scratch arms run higher entropy (0.28-0.29) than the incumbent
  (0.20), i.e. they are less certain, consistent with less total training.

### A caveat about reading these samples

`sample()`'s entropy-adaptive temperature **collapses toward near-greedy when
entropy is high**: at self.temp=0.75 and entropy 1.0 the effective temperature
is 0.19. Measured effective temperatures here are 0.61-0.71, so these are
mildly greedy samples, not diverse ones. The tool now prints the entropy and
the effective temperature so no reader mistakes the sampler for the model.
Separately, with a fixed seed, different prompts can yield near-identical
continuations because the next-byte distributions differ by at most ~0.13 in
probability even for very different prompts.

## Long mixed-data run — `tmt-mixed-2026-09-25-long889897` (2026-09-25/26)

The first long mixed-data leg: from scratch, seed 0, B=1024, dim 512, L16,
**K=18**, tail on, compiled ring credit, fp32/TF32-off, 0.9 GB round-robin
A/B/C corpus (Gutenberg / FineWeb-Edu / UltraChat, 60/33.3/6.7%). Budgeted
from the 10k calibration for ~8h; the run reached its planned 889,897 steps.

Ruler command (identical for all three rows):

    .venv/bin/python tools/eval.py CHECKPOINT \
      --slices 3 --bytes 20000 --xd-bytes 8000 --xd-files 4

| date | ckpt | t | bytes seen | held-out CE | shift CE | cross-domain CE | stop AUC | hot dims | mean HL | note |
|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| 2026-09-25 | `tmt_mixed_long889897_t400000.npz` | 400,000 | 409,600,000 | **1.6999 ± 0.0205** | **1.7094 ± 0.0107** | 1.8435 ± 0.0211 | **0.9359** | 85 | 1.12 B | best on both in-distribution rulers |
| 2026-09-25 | `tmt_mixed_long889897_t600000.npz` | 600,000 | 614,400,000 | 1.7133 ± 0.0222 | 1.7326 ± 0.0130 | **1.7787** ± 0.0178 | 0.8732 | 101 | 1.13 B | stop-head trough |
| 2026-09-26 | `tmt_mixed_long889897_t889897.npz` | 889,897 | 911,254,528 | 1.7438 ± 0.0207 | 1.7551 ± 0.0118 | 1.7950 ± 0.0201 | 0.8938 | **104** | **1.14 B** | final; digest `5291ef86ba4d…` verified |

All three digests verified against their `.sha256` sidecars. Telemetry:
444 interval rows, t=2k→888k, **zero non-finite events**, ring_k=18 throughout.

### The result is a trade, not a win

| axis | 400k | 890k | winner |
|---|---:|---:|---|
| held-out CE | 1.6999 | 1.7438 | 400k (+0.044, 1.5× the 0.03 bar) |
| shift CE | 1.7094 | 1.7551 | 400k |
| cross-domain gap | +0.1436 | **+0.0512** | 890k (−64%) |
| dims >8 B half-life | 4 | **7** | 890k |
| hot dims | 85 | **104** | 890k |

Cross-domain transfer improved the most and saturated by 600k; the last 290k
steps bought almost nothing on any axis. In-domain CE drifted up monotonically
(each step within ~1 SE individually, but three consistent points).

### Memory formation is real but small

Long-lived dims grew steadily: `>4 B` 13→17, `>8 B` 4→7, `>32 B` 2→2, hot
dims 85→104 (saturating), mean half-life 1.12→1.14 B. **The bulk of 8,192 dims
remain ~1-byte memory** and the long tail is ~24 dims total. State RMS climbed
0.40→0.46, so recurrent state magnitude grows faster than useful long memory.

Gradient SNR was **~0.05 for the entire run** (0.10 at 2k, never recovered),
for both shared and decay groups. This is the standing blocker from
`ADVICE-2026-09-24.md` and 890k steps did not change it. It is the evidence
that argues for the role-specific LR factorial rather than more steps or more data.

Throughput decayed gracefully 35,879 → 29,488 B/s (−18%) as the tail grew
0 → ~415 RMS (memory 0 → 109 MB). No cliff; cost scaling is well behaved.

### CAVEAT — the 600k generation collapse is a single unrepeated observation

Chat inspection (`numpy_port.py --frozen … chat`) on the three checkpoints:

| | 400k | 600k | 890k |
|---|---|---|---|
| terminates? | yes (short) | **no — hit 2000-token cap** | yes (0.3–2.5 s) |
| `User:`/`Assistant:` leakage | yes | yes | **none** |
| repetition | mild | **severe** | moderate |
| chat ingest | 170–199 B/s | 170–199 B/s | **244–306 B/s** |

600k is the one checkpoint that failed to terminate, yet its stop-head AUC
(0.8732) partially recovered by 890k (0.8938) and **890k generates cleanly**.
So 600k is a dip, not a monotone degradation, and the in-domain CE trend does
not describe generation quality.

**This is one manual observation, not a measurement.** It has no fixed seed,
no recorded sampling params, and no numeric degeneracy metric behind it. Do
not cite it as a result until it is reproduced under the fixed-prompt protocol
(6 seeds, 250 bytes, degeneracy metrics) already used in the 24-Sep section
above. The apparent ruler/model disagreement — CE says "worse monotonically",
generation says "600k broke" — is the exact defect the multi-domain ruler and
the fixed-prompt generation probe are designed to catch; until those exist,
neither reading is settled.

### CAVEAT — the ruler is TinyStories-centric

This run trained on Gutenberg/FineWeb-Edu/UltraChat and is scored almost
entirely on TinyStories. Two consequences, both visible above:

1. The stop-head AUC drop (0.9359→0.8938) may be **domain shift, not damage**:
   the model learns to terminate in *its own* text while being scored on
   TinyStories line-ending statistics. 890k terminates cleanly in chat.
2. 890k shows **register bleed** — a children's-story prompt pulls in
   federal-institutional web vocabulary. Same class of problem as the dialogue
   markers, subtler.

Neither is measured. A multi-domain ruler is the prerequisite for deciding
whether 400k or 890k is the better checkpoint.

---

## Teacher quality + distillation-target fidelity — LFM2.5-230M-Base (2026-10-03)

Preparation for `docs/plans/PLAN-2026-10-03-DISTILL.md`. Two questions, one CPU
pass, before any distillation GPU hour is spent. **No distillation has been run
yet**; everything below is a measurement of the *teacher* and of the fidelity of
the target a Marginalize-It dump would produce.

### The teacher

| | |
|---|---|
| repo | `LiquidAI/LFM2.5-230M-**Base**` (not the instruct/RL sibling) |
| params | 229,693,184 |
| architecture | 14 layers = 8 short-conv (`conv_L_cache: 3`) + 6 GQA full-attention, hidden 1024 |
| vocab | 64,402 ids = 63,893 text + 509 control tokens; 32,768 ctx; 19 T tokens; cutoff mid-2024 |
| tokenizer | BPE, **`normalizer: null`**, ByteLevel pre-tokenizer, sha256 `df1d8d5ec5d0…` |
| license | LFM Open License v1.0 (non-OSI) |
| bytes/token | 4.106 / 4.011 / 4.039 on the three eval files (4.153 on a 2 MB window) |

### Ruler setting (stated because it differs from the CE rows above)

`tools/teacher_eval.py --bytes 100000`, **one** contiguous 100,000-byte window
per file, fp32, CPU, one forward pass. The CE rows in the tables above are
`--slices 3 --bytes 20000`. Different windows, so the two are **not** directly
comparable; the comparison below is a magnitude check, not a verdict.

| file | nats/token | nats/byte | BPB | tok/s |
|---|---:|---:|---:|---:|
| `tinystories_valid.txt` | 2.8137 | **0.6853** | 0.9886 | 236 |
| `wiki_tinystories_005.txt` | 3.0249 | 0.7541 | 1.0879 | 228 |
| `tinystories_v2gpt4_valid.txt` | 3.0086 | 0.7448 | 1.0745 | 240 |

**The teacher is ~0.49 nats/byte better than TMT on the same file** (TMT 1.1741
at t=64,747 / 1.1902 at t=94,747; teacher 0.6853). So there is real information
in the teacher's byte distribution that TMT does not have — the premise of
distillation holds, and "the teacher is only 230 M" was the right worry to check
rather than assume. (`causal_conv1d` is absent locally, so the reference
conv path ran; correctness unaffected, throughput is not a GPU number.)

### GATE 0 — the token→byte contract: PASS

`tests/check_distill.py`, free, CPU, ~10 s. **Exact byte-for-byte
reconstruction** of all three eval files. Three traps pinned as permanent
assertions, two of which were found *by the gate during development*:

1. **`tokenizer.decode()` is lossy** — 791 ids return U+FFFD for non-UTF-8 byte
   tokens (id 604 → `b'\xa1'`). A table built the obvious way silently corrupts
   1.2 % of the vocabulary's byte targets; it still trains and still lowers the
   loss.
2. **`added_tokens` shadow the base BPE vocab** — 507 of 509 added ids also exist
   in `model.vocab`. My first table builder read `model.vocab` only, which is a
   real bug that mapped 507 ids to the wrong bytes.
3. **The post-processor injects BOS**, the literal 15-byte string
   `<|startoftext|>`. Dump with it disabled or every sequence gains 15 phantom
   bytes.

Storage math at k=32 (fp16 value + int32 index = 192 B/token, ÷4.106 B/tok):
**46.8 B per training byte** ⇒ 64 MB slice = 3.0 GB, 256 MB slice = 12.0 GB.

### Retained prefix mass — and why top-32 is NOT the problem

Marginalize-It renormalizes over tokens whose bytes START WITH the gold byte
prefix. The mass those tokens hold is the target's fidelity. Measured mean mass
(top-32, `tinystories_valid`, n≈6.1k positions per depth):

| byte depth k | mean matched mass | p05 | fraction < 0.5 |
|---|---:|---:|---:|
| 0 (first byte of a token) | 0.803 | 0.079 | 13 % |
| 1 | 0.371 | 0.0003 | **64 %** |
| 2 | 0.320 | 0.0000 | **69 %** |

**The k-sweep (4,993 tokens, 20 kB) settles the cause:**

| depth | k=8 | k=32 | k=128 | k=1024 (≈ full vocab) |
|---|---:|---:|---:|---:|
| 0 | 0.752 | 0.807 | 0.835 | 0.852 |
| 1 | 0.367 | **0.374** | 0.376 | 0.377 |
| 2 | 0.321 | **0.324** | 0.324 | 0.324 |

Going from k=32 to the whole 64 k vocabulary moves depth-1 mass by **0.005** and
depth-2 by **0.000**. So:

- **top-32 truncation is not the binding constraint** (k=8 is — it costs 5 points
  at depth 0). Storage at k=32 is justified.
- **The mass loss is inherent to prefix matching**: at byte depth ≥1 the teacher
  genuinely puts ~63 % of its mass on tokens that do NOT share the 2-byte
  prefix — overwhelmingly, on tokens that *end* at that position. That is
  precisely the EoT case the paper describes, and Marginalize-It discards it.

**Consequence, and it is the most important line in this section:** the MI target
at depth ≥1 is *not* the teacher's distribution. It is the teacher's
distribution conditioned on **not ending the token here**, renormalized. At
~76 % of byte positions (4.1 bytes/token ⇒ 3 of 4 positions are at depth ≥1) and
64–69 % of those below 0.5 mass, **roughly half of all byte positions carry a
heavily-reconditioned target.**

That is a *bias*, not just noise: it teaches TMT that tokens run longer than
they do. It may be benign for a byte model that learns line ends separately via
its stop head — but it is not the paper's exact conversion, and it must be
stated wherever a distillation number is quoted.

### New, cheap, mandatory post-hoc diagnostic

If a KD arm trains, measure **word-length drift** on its samples: mean bytes per
generated word and the rate of premature line-ends vs the CE control. That is the
direct signature of the bias above, it costs one pass over 250-byte samples, and
CE cannot see it (the paper's §6.6, and this repo's own 600k caveat).

### Decontamination gate (new; exists because the corpus register = the ruler)

Choosing a teacher-generated corpus in the ruler's register opens a channel the
FS-W byte-offset trick cannot cover: the teacher was trained on web text that
likely CONTAINS TinyStories, so it can regurgitate eval stories verbatim.
`tools/decontaminate.py` widens `fetch_eval_data.prepare()`'s membership set from
the training glob to **train ∪ eval ∪ shift ∪ novels**. Gated in `check_distill.py`
t5: a story copied from `tinystories_valid.txt` **is** dropped by the widened set
and **is not** dropped by the training-only set (the gap being fixed).

**Known limit, demonstrated by the same gate:** the rule is line-exact, so a
rewrite with the same beats is NOT dropped. TinyStories is heavily templated, so
**near-paraphrase contamination remains undetectable here.** Generated-corpus CE
numbers are not comparable to real-corpus rows and carry a caveat at least as
strong as FS-W's byte-offset caveat.

### Operational

| | |
|---|---|
| teacher weights | 459,401,112 B; at **2.5 MiB/s ≈ 3 min** (the `hf_hub_download` xet path stalled twice at exactly 320 MB — resumable `curl -C -` completed it) |
| implication | HF checkpoint push/pull ≈ 1.5–2 min per 250 MB artifact from this box; the 0.9 GB corpus fetch ≈ 6 min. Neither is a blocker. |
| local env | `transformers` 5.18.0 + `tokenizers` 0.23.2 added to `.venv` (additive; the engine imports neither). LFM2.5 needs `model_type: lfm2` support. |
