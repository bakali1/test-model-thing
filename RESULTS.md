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

---

## Fused NVRTC ring credit — the pre-Volta path (2026-10-03)

The distillation plan needs a machine to run gates and rulers on. The candidate
was a laptop with an **MX150 (Pascal, cc 6.1)**. It could not run TMT at all:
`ring_credit="auto"` — the shipped default — resolves to `compiled`, and
`GPUTooOldForTriton` was raised on the first step, because **Triton requires
compute capability >= 7.0**. `triton` fails the same way. `full` is the only
path that ran. Recorded here because the default crashing on a whole class of
GPUs is a portability bug, not a tuning note.

### Where the time actually goes (measured, not guessed)

Before optimizing anything, two probes decided the whole approach.

**Is it launch-bound?** `cpu(no sync)/wall = 0.97` and profiler device time
195 ms/step against a 195.9 ms wall: the GPU is saturated and the CPU is nearly
so, but the *device* time is real work. **CUDA graphs would have bought nothing**,
so none were built. (The repo has a `--compile-step ... reduce-overhead` flag
that does use graphs; it needs Inductor, so it is dead here too.)

**Which region?** `tools`-free microbenchmark at the real shapes (B=1024,
K=14, D=512, ring = 29 MiB), against a measured 39.1 GiB/s copy ceiling:

| per layer | ms | vs bandwidth-optimal |
|---|---:|---|
| `w.mul_(dec)` (decay, in place) | 1.455 | optimal (2 passes) |
| `torch.mul(w, dlds, out=vals)` | 1.456 | optimal (2 passes) |
| `dest.index_add_(rows, vals)` | **2.521** | **3.5x worse** (atomics) |
| **total ×16 layers** | **86.9 ms of 195** | **45 % of the step** |

The `full` path makes five passes over the ring per layer where two of them are
re-reads of the same bytes. `w[:, p] = 1.0` and the GEMMs were never the issue.

### What was built: `ring_credit="nvrtc"`

One fused kernel, compiled at first use by **`torch.cuda._compile_kernel`
(NVRTC)** — no nvcc, no ninja, no build step, and no cc floor. It reads `w`,
multiplies by the decay, writes the decayed value back, multiplies by the
gradient and atomically scatters, in a single pass.

| per layer, MX150 | full | nvrtc | speedup |
|---|---:|---:|---:|
| no hot dims | 5.370 ms | 2.486 ms | **2.16x** |
| with hot-dim mask | 6.793 ms | 2.483 ms | **2.74x** |

**End to end**, 680 steps (past the 500-step tail rebase, so hot dims are live),
fp32, TF32 off, `ring_credit=auto`:

| | full | auto→nvrtc | speedup |
|---|---:|---:|---:|
| B=1024 | 5,196 B/s | **6,931 B/s** | **1.33x** |
| B=4096 | 5,679 B/s | **7,806 B/s** | **1.37x** |

Peak VRAM unchanged (926 MiB / 3,148 MiB) — it is the same arithmetic, fewer
passes.

### Parity: the fused path is inside the reference's own noise

The ring buffer is **bit-identical** (`max|d| = 0.0`): the decay and credit are
per-element and the arithmetic order is unchanged. `dembed` differs by
**8.8e-07 of RMS — exactly the spread of `full` against itself**, run to run.
`index_add_` is an atomic scatter, so its summation order was never
deterministic; exact equality was never a property of the reference.

The control that settles it, 150 steps, same seed, same byte stream:

| | full vs nvrtc | **full vs full (control)** |
|---|---:|---:|
| max per-step \|dloss\| | 4.768e-07 | **4.768e-07** |
| theta max\|d\|/RMS | 1.138e-06 | **1.138e-06** |
| ring_w max\|d\|/RMS | 9.656e-08 | **9.656e-08** |
| ring_b identical | True | True |

Same path twice is as far from itself as the fused path is from it. Gated in
`tests/check_variants.py` (now lists `nvrtc`, SKIPs on CPU) and in
`tools/bench_variants.py`.

**Side finding, worth its own line:** the engine is **not** run-to-run
reproducible on CUDA at ~1e-06 on theta. The "bit-identical" contract holds on
CPU; on CUDA the scatter makes every run a new trajectory.

### The default no longer crashes

`_ring_mode()` resolves once, up front: `auto` → `compiled` on CUDA, and if the
device is below cc 7.0 it falls through to `nvrtc`, printing one line. It also
forces `full` when the model is on CPU — the earlier check keyed off
`torch.cuda.is_available()`, which is true in a CUDA-capable process running a
CPU model, and that mismatch reached the kernel with CPU tensors.

### Two pre-existing bugs found on the way (both fixed)

1. **`tail_credit._tail_add`, pure-numpy fallback, was wrong.** It indexed
   `E[ids, ids, h]` — using the *byte id* for axis 0, which is the *stream*
   index (`E` is `(B, 256, H)`). The numba path has always been correct. It
   only ever worked when `B = 256`: with `B=32` it raised
   `IndexError: index 127 is out of bounds for axis 0 with size 32`. Any box
   without numba was running a broken reference — here it failed
   `tests/check_parity.py` on **pristine HEAD**, and passes after the fix.
2. `gpu_port._compile_failed` was assigned and then **never read** — the
   fallback that field was clearly written for did not exist.

`tests/check_trainer.py` still fails `['2 tail cap pre-allocation', '3
state/rebase invalidation']` on the MX150 box, **identically on pristine HEAD**
— the box has 11 GB RAM against the laptop's 14 GB, so the tail cap trips
differently. Not a regression, not fixed here.

### What this does NOT change

The MX150 is now **6.9 kB/s**, still ~3.2x slower than a T4's measured 22.1
kB/s live leg, and the 889,897-step run is still ~36 h here versus ~11.5 h on a
T4. `PLAN-2026-10-03-DISTILL.md` §9b's verdict stands: **the T4 is still
required for steps 1-3.** This optimization makes the box good enough for
gates, rulers and 2,000-step smokes, and it removes a crash that would have
blocked any pre-Volta GPU outright. It does not make this box a training host.

Reproduction: `.venv-gpu/bin/python tools/bench_variants.py tmt.npz --B 1024
--steps 200` on the box (add `nvrtc` to the board), and
`.venv-gpu/bin/python tests/check_variants.py` for the parity half.

---

## CORRECTION to the distillation corpus plan — the teacher cannot write a corpus (2026-10-03)

Appended per convention; nothing above is edited. `PLAN-2026-10-03-DISTILL.md`
§4 steps 1–2 (teacher-generated corpus, then the corpus-identity A/B) are
**withdrawn as the main experiment.** The tool that would have executed them is
built and working (`tools/gen_teacher_corpus.py`, `tools/calibrate_corpus.py`),
and it is kept, because the measurement that withdrew the step is the evidence.

### Measurement 1 — the teacher's generations are measurably less diverse than the real corpus

Same metric code (`chat_models.degeneracy_metrics`), same story granularity:

| corpus | n | degenerate | unique_4gram | unique_8gram | rep_4gram_frac | max_run_newlines | unique_bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wiki_tinystories_003.txt` (real, train) | 800 | **0.0 %** | **0.687** | 0.912 | 0.0146 | 1 | 36.0 |
| `tinystories_valid.txt` (real, held out) | 800 | **0.0 %** | **0.687** | 0.912 | 0.0146 | 1 | 36.0 |
| teacher, 2 shots, T 0.85, nrng 4, **260 tok** | 8 | 0.0 % | 0.579 | 0.818 | 0.0161 | 1 | 37.9 |
| teacher, 2 shots, T 0.85, nrng 4, **700 tok** | 8 | 12.5 % | 0.510 | 0.777 | 0.0141 | 2 | 45.4 |
| teacher, 3 shots, T 0.90, nrng 4, 700 tok | 8 | 0.0 % | 0.457 | 0.738 | 0.0178 | 2 | 44.0 |

Ruler setting: 8–24 generated stories per cell, MX150 fp16, seed 1000+start,
temperature/top-p/repetition/no-repeat-ngram as tabulated; baseline is 800 real
stories per file. **Every setting tested lands 0.11–0.23 below the real corpus
on `unique_4gram`.** The gap is not a tail that filtering removes — it is the
bulk of the output.

Consequence: step 2's arms would have differed by **register AND degeneracy at
once**, so the corpus-identity number would not have meant what the plan claimed
it meant. That is the exact class of confound this repo keeps refusing to
publish.

Also measured, and required to get anything usable at all: a 230 M **base** model
given a ~40-token prefix and no pattern emits one sentence 8× verbatim
(batch=1, no padding, so this is not a padding artifact). Few-shot (2–3
in-register stories) plus `no_repeat_ngram_size=4` is what produces prose;
`repetition_penalty` alone does not (tried 1.05 and 1.10, both still looped).

### Measurement 2 — the MX150 cannot run the teacher at all

| machine | teacher generation | note |
|---|---:|---|
| laptop, 8c CPU fp32 | **475 B/s** (batch 64; flat to batch 256) | reference conv path |
| MX150, fp16 | **18–26 B/s** | `causal_conv1d` absent there too |

**The GPU box is ~20× SLOWER than the laptop for teacher generation**, because
the optimized conv kernel is missing and the reference path is pathological on
Pascal. Teacher work therefore belongs on the T4 (or the laptop CPU); the MX150
stays what the brief says it is — gates, rulers and TMT smokes. The dump's
single-forward cost on the MX150 would be ~10.5 h for a 64 MB slice, so it is
T4 work too.

### What replaces steps 1–2

The real corpus is **already on disk and already the right register**:
`data/wikipedia_clean/wikistories/wiki_tinystories_00{1..4}.txt`, 158 MB, and
Phase 0.1 already moved `wiki_tinystories_005.txt` + `tinystories_valid.txt`
*out* of the training glob. So training on files 001–004 and scoring on the
held-out set is decontaminated **by construction** — no generation, no
decontamination pass, and no contamination caveat.

It also fixes the reproducibility gap found earlier today: local files have
stable digests, so both arms can be byte-matched and *proven* so. That is
exactly what the 889,897-step run could not offer (its `spec_hash` and dataset
revisions were never recorded), which is why its CE rows cannot serve as a free
control.

New sequence: **step 3 only** — MI dump of the real corpus, KD arm, CE control
arm, both byte-matched with recorded digests. Steps 4 (on-policy) and the
decontamination gate stay in the plan, unused by this experiment.

### NOT retracted

The MI target-fidelity result (retained prefix mass 0.803/0.371/0.320 at depths
0/1/2; k=32 sufficient) is unaffected — it was measured on real eval text, not
on generated text. The word-length-drift diagnostic it predicts is unaffected.

---

## MI dump + reader: round trip, and three bugs the gates caught (2026-10-03)

`distill.py` (format + reader), `tools/mi_dump.py` (writer), `tests/check_mi.py`
(gate), `tools/mi_report.py` (gate 2). Round trip on 200,000 B of
`wiki_tinystories_003.txt`, B=8, k=32, fp32 teacher, laptop CPU.

**Gate `tests/check_mi.py`: PASS — vectorized MI == brute force, max |Δ| = 0.00e+00
over 308 randomized cases at depth ≥ 1.** That gate caught three bugs, all of
which would have trained on a plausible-looking wrong target:

1. **Gold-prefix bug (the worst).** The reader was handed only the gold byte at
   depth *d*, so its prefix test compared every depth against that one byte and
   matched nothing at depth ≥ 1. Consequence: every byte position inside a token
   would have fallen through to the backoff — about **76 % of positions** — and
   the KD arm would have looked like it worked while training on ~¼ of the data.
2. **Absolute-vs-relative row indexing.** `q` is an absolute token-row index but
   the arrays were sliced at `t0`; indexing the slice with `q` read the wrong row
   whenever `q < t0`.
3. **Off-by-one in which row predicts which token.** Dump row *r* holds the
   logits at token position *r*, which predict token *r+1*; byte *p* belongs to
   token *q*, so the distribution that predicted token *q* is row **q−1**. Using
   row *q* asks a different question and left **66 % of positions** at the
   backoff. Fixed; cross-checked by CE (below).

A fourth, in the dumper: the token axis was padded to the byte-position loop
length, costing 4.1× storage — **791 B/token against the frozen 193**. Now sized
by real token counts: **208.4 B/token at B=8**, the residue being per-stream
padding (8 %); at the frozen B=1024 this shrinks further.

### Independent cross-checks (two code paths, same data)

| check | dump path | independent (`tools/teacher_eval.py`) |
|---|---:|---:|
| CE of gold token, covered positions | **2.6893** nats/token | 2.8137 (held-out eval text) |
| gold token inside top-32 | 76.5 % | — |
| depth-0 retained mass, mean | **0.7007** | 0.803 |
| fraction below 0.05 at depth 0 | 7.7 % | p05 = 0.079 |

The CE is *lower* on the training file, which is what memorization predicts. Both
paths agree that row *i−1* predicts token *i*. The depth-1 mean does **not**
agree across the two measurements (0.22 here vs 0.371 earlier) — different text
(training file vs held-out eval) and different context window (256-token dump
chunks vs one 20 k-token sequence). **Recorded as unexplained**; the number that
governs the KD arm is the coverage below, which is measured on the exact data and
chunking the trainer will use.

### Gate 2 — what KD actually gets: coverage, not mean

32,000 positions walked through the real reader (`tools/mi_report.py`):

| byte depth | retained mass mean | p05 | KD coverage @ floor 0.05 | @ floor 0.01 |
|---|---:|---:|---:|---:|
| 0 | 0.7007 | 0.0103 | **91.6 %** | 95.0 % |
| 1 | 0.2238 | 0.0000 | **51.2 %** | 71.3 % |
| 2 | 0.1733 | 0.0000 | **40.6 %** | 58.8 % |
| 3 | 0.1527 | 0.0000 | **36.0 %** | 52.7 % |

**At the frozen floor of 0.05, Marginalize-It supplies a usable target at 91.6 %
of first-byte positions but only ~36–51 % of positions inside a token — 49.1 %
of all positions fall back to CE-only.** That is the MI bias showing up as lost
coverage rather than as a small distortion, and it is the single most important
number for interpreting the KD arm: it is not "distillation on every byte".

Note also that the surviving targets are essentially **one-hot** (mean target
entropy ≈ 0): when MI keeps a token, it is usually the only prefix-matching one
in the top-32. So the target carries the teacher's *ranking* far more than its
*uncertainty* on this data.

The `argmax==gold` column in `mi_report.py` is tautological (an all-zero
backoff target has argmax 0) and is not a correctness check; it is left in only
as a coverage duplicate and should be dropped before the real run.

---

## KD cost on CPU: measured, and the plan's estimate was 10-20x too high (2026-10-03)

Item 4 (the 2,000-step smoke) is blocked — no GPU. But its CPU cost is now
measured rather than guessed, because the number the plan carried was an
estimate: "KD term overhead **unmeasured**, est. +10-30 %" (§9b).

Frozen student (dim 512, L16, fp32), laptop 8-core CPU, identical steps for
both arms, timed after 10 warmup steps:

| B | α=0 (CE) | α=0.5 (CE+KD) | KD overhead | coverage |
|---:|---:|---:|---:|---:|
| 64 | 160.6 ms/step · 399 B/s | 160.2 ms/step · 399 B/s | **−0.2 %** (noise) | 34.4 % |
| 256 | 299.5 ms/step · 855 B/s | 304.0 ms/step · 842 B/s | **+1.5 %** | 31.6 % |
| 1024 | 966.4 ms/step · 1,060 B/s | not run (needs a 2.05 MB dump) | ≤ ~2 % expected | — |

**The Marginalize-It target is essentially free next to the model's own step.**
The reader builds a block of 32 steps at a time, and the MI loop costs
O(B·k·dmax) ≈ O(1024·32·18) per block, against a step that is already a second
of 16 layer-norm + SiLU + AdamW passes. The plan's +10–30 % estimate was wrong
by more than an order of magnitude.

**2,000-step smoke on this laptop CPU:**

| component | cost |
|---|---:|
| dump 2.05 MB of corpus at B=1024 (teacher forward, 500 k tokens at 965 B/s) | **~35 min** |
| 2,000 training steps with KD at 966 ms/step × 1.015 | **~33 min** |
| **total** | **≈ 68 min** |

The dump must cover ≥2,000 bytes per stream, i.e. **≥2.05 MB of corpus at
B=1024**; a 200 kB slice makes `make_streams` fall back to contiguous chunks and
yields 1-byte streams, which fails loudly rather than silently misaligning.

**Re-costed CPU-only science branch** (plan §9b said 16 MB ≈ 16 h, using the old
889 B/s and the inflated KD overhead): at the measured 1,060 B/s, one 16 MB arm
is **4.2 h**, so CE control + KD arm + one shared dump = **≈ 13 h**, with
~32 % KD coverage. That is a genuine option, not just a fallback.

---

## 2026-10-04 — role-specific-LR factorial (PLAN-2026-10-04-LR-ROLES)

Three arms, **30,000 steps each from the same t=889,897 checkpoint bytes**, same
data seed 0, same B=1024, same 96 MB TinyStories slice, same T4
(`ring_credit=auto`→compiled, ~22–24 kB/s). Only `--lr-decay` differs: the LR of
the 16 per-layer `.decay` slices (8,192 of 4,481,793 params = **0.183 %**).
Ruler: `--slices 3 --bytes 20000 --xd-bytes 8000 --xd-files 4`.

| arm | `--lr-decay` | held-out CE (nats/byte, 6 slices) | shift CE | cross-domain CE |
|---|---|---|---|---|
| `arm_lrdecay1x_t919897` | unset (**control**) | **1.2094 ± 0.0143** | 1.1648 ± 0.0162 | 2.4176 ± 0.1372 |
| `arm_lrdecay4x_t919897` | 2e-3 | 1.2174 ± 0.0138 (**+0.0080**) | 1.1734 ± 0.0161 | 2.4181 ± 0.1367 |
| `arm_lrdecay16x_t919897` | 8e-3 | 1.2191 ± 0.0149 (**+0.0097**) | 1.1710 ± 0.0150 | 2.4441 ± 0.1411 |

**Decay-role response** (static, computed identically from each checkpoint):

| arm | σ_mean | moved vs start | × control | half-life | hot dims | tail @B=1024 | stop AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| start (t=889,897) | 0.51479 | — | — | 1.138 B | 1669 | — | — |
| 1x | 0.51911 | +0.00432 | 1.00 | 1.162 B | 1747 | 180.4 MB | 0.9496 |
| 4x | 0.52264 | +0.00784 | 1.82 | 1.179 B | 1810 | 213.9 MB | 0.9426 |
| 16x | 0.52404 | +0.00925 | 2.14 | 1.190 B | 1895 | 231.7 MB | 0.9597 |

**Generation probe** (3 prompts × 3 seeds, `--n 400`, same prompt bytes and seeds
per arm): **0/9 degenerate and 0/9 non-terminating in every arm.** u8 ratio
0.906 / 0.888 / 0.950; rep8 0.85 % / 0.99 % / 0.93 %; max whitespace run 2 / 3 / 2.
No generation-health difference. (`stop/kB` is not readable here: at `--n 400`
the generations are too short for the 13–17/kB natural-prose baseline.)

### Verdict: the axis is LIVE but buys nothing. No winner; axis closed.

The plan's falsifier did **not** fire. It predicted that if σ_mean did not
visibly rise from 1×→16×, the decay role would not be LR-limited and the axis
would be dead. It does rise, monotonically, with a clean dose–response —
**and it saturates hard**: 4× the LR moves σ 1.82× the control's own movement,
but going 4×→16× (4× more LR again) adds only **40 % of the first jump**.

So the decay role *is* LR-responsive, and pushing it harder is still not worth
anything: CE is **monotonically worse** with more decay LR (+0.008 at 4×,
+0.0097 at 16× — both inside the 0.03 bar, but in the wrong direction), while the
cost is concrete and not: hot dims 172→221 and the hot tail **180 → 232 MB
(+29 %)** for zero gain. Cross-domain CE degrades most at 16× (2.4176 → 2.4441).

The extra "memory" the decay role buys is not being used by the credit path to
predict better. This is evidence that **the decay role is not the binding
constraint on quality** — consistent with the SNR blocker sitting upstream of it
in the trunk. The per-role-LR axis is answered; do not extend it.

### Confounds and gaps, stated

- **The control arm's gradient-SNR telemetry was lost.** The first live VM was
  reclaimed mid-factorial (laptop DNS failure killed the keep-alive daemon; see
  below), taking `telemetry_lrdecay1x.jsonl` with it. So per-arm SNR exists for
  4× and 16× only, with **no matched 1× control**; the SNR readout of this
  factorial is therefore unpaired. The σ / half-life / hot-dim / CE comparisons
  above are unaffected — all four come from checkpoints that did survive, and the
  16× arm was re-run from the *pinned* start file on a fresh VM.
- Arms ran on three different VMs; only the pinned-start guarantee
  (`tmt_mixed_long889897_t889897.npz`, data seed 0) makes them matched, and the
  loss traces confirm it — 4× at step 891,772 read ce 1.259 vs the control's
  1.258.
- **Infrastructure, not quota, ended two runs.** `colab` history shows
  `NameResolutionError` for `colab.research.google.com` → keep-alive daemon dead
  → VM reassigned. Fixes now in the bundle: a push **watchdog** (every 60 s)
  rather than pushing only at arm end, because `Trainer.train(save=...)` takes a
  **boolean**, not a callback — the first draft's push-inside-callback never ran,
  which cost two arms before it was caught live.

---

## Colab T4 smoke (item 4) — PASS, and it killed the KD arm (2026-10-03)

`bundles/make_kd_smoke_bundle.py` → `colab_kd_smoke.py`, T4, 2,000 steps at
B=1024, α=0.5, 6.24 MB TinyStories slice, dump written in-session.
**`[RESULT] OVERALL: PASS`** in 510 s. It is a plumbing smoke, not a science arm
— and the plumbing verdict is that three numbers in this file are wrong and one
whole idea does not work.

### What passed

| stage | result |
|---|---|
| teacher | `model_type=lfm2`, 14 layers, vocab 65,536, 464 MB, fp32, TF32 off |
| dump | 1,511,812 tokens in 287 s (≈21.7 kB/s of teacher scoring) |
| coverage (T4) | depth 0 **91.2 %**, 47.3 %, 36.7 %, 31.6 % — reproduces the laptop CPU (91.6/51.2/40.6/36.0) |
| gate 3 on Colab's CPU | α=0 bit-exact, max\|Δθ\| = 0.0 over 120 steps |
| throughput | 35,832 B/s CE-only, 25,722 B/s with KD (real leg measures 22–27.5 kB/s) |
| engine α | 0.5 honoured, coverage 52.13 % live, KD moved the loss |

### First Colab run failed, and the failure was the most useful thing in it

`TinyStories-train.txt` separates stories with `<|endoftext|>` — **246 markers
per 200 kB** — which the LFM2 tokenizer maps to control id 2. `mi_dump` refused,
correctly. But it then substituted an empty array per stream and **exited 0**,
and the resulting all-padding dump measured **193.4 B/token — exactly the frozen
target**, because padding has the same bytes-per-token as content. The storage
check could not see an empty dump; a *completeness* check (bytes ÷ tokens in
1.2–8.0) can, and now does, and the guard raises with the offending id named.
The bundle now strips the markers, as `fetch_eval_data.py` documents.

### Correction 1 — the dump is 259.1 B/token, not 193

At B=1024 on real text the token axis is padded to the longest stream: **259.1
B/token = 63 B per training byte**, i.e. a 64 MB slice is a **4.1 GB** dump, not
3.0 GB. The 193.4 figure from the first run was the degenerate dump. Per-stream
padding does not shrink with corpus size, so this is the real number.

### Correction 2 — KD costs +39.3 % on a T4, not +1.5 %

| | laptop CPU | T4 |
|---|---:|---:|
| α=0 | 966 ms/step | 28.6 ms/step |
| α=0.5 | +1.5 % | **+39.3 %** (39.8 ms/step) |

The target build is single-threaded numpy on the host and is **serial with the
step**, so on a fast GPU it stops being amortized: 11 ms of host work against a
29 ms device step. A KD arm on the T4 costs ~40 % more wall clock per step than
the plan assumed.

### The KD arm diverges — and the reason invalidates the idea

CE-only and KD arms, identical seed 0, identical byte stream, laptop CPU,
B=64, 2,500 steps, same dump:

| step | α=0 (CE) | α=0.5 (KD) |
|---:|---:|---:|
| 1,000 | 2.48 | 2.68 |
| 1,800 | **1.89** | 2.20 |
| 2,100 | 2.25 | 2.70 |
| 2,300 | 1.97 | 5.04 |
| 2,499 | **1.80** | **16.77** |

The KD arm is worse from early on and then diverges; the control keeps
improving. The T4 smoke independently showed the same thing (KD loss 3,450 vs CE
2.36 at step 2,000).

**The reason, measured over 4,679 covered positions on a real dump:**

| property of the MI target | value |
|---|---:|
| target entropy | **0.000000** nats (max −0.000000) |
| fraction with entropy > 0.001 | **0.00 %** |
| argmax == gold byte | **100.000000 %** |
| top1 − top2 mass gap | 1.000000 (every position) |

**The Marginalize-It byte target is a point mass at the gold byte. It is always
one-hot, by construction, not by accident:**

    S = { c : bytes(c)[:d+1] == gold[:d+1] and len(c) > d }
    every c in S has bytes(c)[d] == gold[d]
    => p_target[gold[d]] = sum_{c in S} p(c) / sum_{c in S} p(c) = 1

Conditioning on the gold prefix forces every surviving token to agree on byte
*d*. So the 256-way target is degenerate, the "soft label" is a hard label, and
`α·KL + (1−α)·CE` reduces to **position-dependent CE reweighting**: weight 1.0
on the ~52 % covered positions and 0.5 on the rest. That is not distillation,
and it is presumably why the arm both underperforms and destabilizes.

The earlier note in this file ("surviving targets are essentially one-hot, so the
target carries the teacher's ranking, not its uncertainty") was recorded as a
curiosity. It is the headline.

### The fix, and it is small

Depth 0 does **not** have to be degenerate. The paper's Case I builds the
first-byte distribution by aggregating tokens that *start with* B1 — it does not
condition on the gold byte. Measured on the same dump, unconditioned:

| depth-0, unconditioned first-byte marginal | value |
|---|---:|
| entropy (nats) | mean **0.306**, p95 1.197 |
| positions with entropy > 0.01 | **64.4 %** |
| mean max marginal | 0.888 |

So the teacher *does* carry a real soft byte distribution — and my implementation
threw it away at depth 0 by applying the gold-prefix filter there. Corrected
design: **depth 0 → unconditioned first-byte marginal (soft, real); depths ≥ 1 →
drop from KD entirely** (one forward pass plus gold-prefix conditioning cannot
represent byte uncertainty there; restoring it needs either the EoT channel or
extra teacher passes). That is a real distillation signal on ~1 position in 4.1
— about 24 % of bytes — and nothing anywhere else. Whether 24 % is enough to move
the ruler is an empirical question, and the only honest next experiment.

### What this does NOT retract

The retained-mass table (91.2/47.3/36.7/31.6 % coverage) is still correct as a
measure of **how often the gold prefix has any support in the top-k** — it just
is not a measure of target *fidelity*, because the target carries no
uncertainty to be faithful about. The gates, the dump format, the alignment
contract, the α=0 bit-exactness guarantee, and the KD term's algebra all stand.
The word "fidelity" in the earlier coverage text is what should have been
questioned: fidelity to a distribution that does not exist in the target.

---

## Step B — the corrected target is real signal, and the KD arm STILL diverges (2026-10-03)

The one-hot target was fixed first: **depth 0 only, the paper's Case I
first-byte aggregate, no gold conditioning; depths ≥ 1 declined.** Gate
`tests/check_mi.py` changed semantics deliberately and now PASSES, including a
new **tripwire (t6)** that fails if the depth-0 target is one-hot again.

On a real dump (788 kB of TinyStories, B=64, k=32, 129,024 positions):

| | gold-conditioned (old) | Case I (corrected) |
|---|---:|---:|
| coverage | 51.7 % (any depth) | **24.00 %** (depth 0 only — 1 byte in 4.17) |
| target entropy | **0.000000** nats | **0.3282** nats (max 2.74) |
| top-1 mass | 1.0000 | **0.8794** |
| top-1 == gold byte | 100.0000 % | **82.1420 %** |

So the corrected target carries genuine uncertainty, and **the teacher disagrees
with the gold byte on 17.9 % of token starts**. That is real signal — and real
label conflict.

### The A/B: 2,500 steps, B=64, seed 0, identical byte stream, laptop CPU

| step | CE arm | KD arm (α=0.5, corrected) |
|---:|---:|---:|
| 250 | 2.61 | 3.07 (**+0.46**) |
| 1,000 | 2.48 | 2.95 (**+0.47**) |
| 1,250 | 2.11 | 2.79 |
| 1,500 | 2.18 | **27.19** |
| 1,750 | 2.02 | **148.11** |
| 2,499 | **1.80** | **860.68** |

**The corrected arm is worse from step 250 and still diverges**, exactly as the
one-hot arm did — and harder. The control is stable and still improving at
step 2,499.

### Verdict: the KD arm is CLOSED as specified

Pre-registered criterion from the plan before running: *"If the corrected arm is
still ≈ CE, or still destabilizes, the answer is that distillation does not help
TMT at this scale and I stop rather than spend T4 hours confirming it."* It
destabilized. **No T4 time is spent on the science A/B.**

Two independent target definitions (gold-conditioned one-hot; Case I soft) both
lose to the control and both diverge, on CPU and on the T4 smoke. So the failure
is not the one-hot bug — that was real but not the cause.

**Best current explanation (a hypothesis, not a result).** The teacher's argmax
disagrees with the gold byte at 17.9 % of token starts (~4.3 % of all byte
positions), so CE and KD push in opposite directions on a few percent of
positions. AdamW's per-parameter second-moment normalisation divides by a *small*
√v exactly where the two gradients cancel, amplifying the oscillation instead of
damping it. Consistent with: stable to ~1,000 steps, then exponential blow-up.

Not established: one seed, one α (0.5), one corpus, B=64. This closes **KD as
specified**, not "distillation can never work here".

### What survives, and is reusable

- the corrected target and its gate: 24 % of byte positions, entropy 0.33 nats,
  teacher-vs-gold disagreement 17.9 % — the first genuine distillation signal
  measured in this repo;
- the depth-0/Case-I reading of the paper, which the earlier implementation had
  wrong;
- the dump format, alignment contract, completeness gate, α=0 bit-exactness
  (max|Δθ| = 0.0), and every gate in `tests/`.

### Recommended next direction

`PLAN-2026-10-04-ROADMAP.md`'s reframe — the standing blocker is gradient SNR
≈ 0.05, a *gradient-estimator* problem — is now better supported than the
distillation route, and the rig has just closed the role-specific-LR axis
(`5f5fe16`). The teacher-disagreement number above is itself evidence for that
reading: the thing a better teacher supplies is a *better gradient*, and at
4.5 M parameters with O(1) state there is a hard ceiling on how much of a
full-context teacher is usable.

---

## P0 item 1 — the trace-quality gate: the eligibility trace is NOT biased (2026-10-03)

`tests/check_trace_bptt.py`, free, CPU, seconds. Roadmap §4 action 1. The
reference is **exact truncated BPTT** built from the engine's own forward
functions (`TorchLayer.forward`, `_layer_norm`, `_sig`) with only one difference:
the state stays in the autograd graph instead of being detached each step. No
hand-written forward means no second implementation to disagree.

**Forward agreement is asserted first and is exact (max |ΔLoss| = 0.000e+00)**
— with the trace pass's LR set to 0 so the parameters are frozen for the window.
Without that freeze the engine's parameters drift between steps while the
reference holds them fixed, and the two forwards diverge by 1.4e-4 for reasons
that have nothing to do with the trace.

### Result

| layer | rel err | cosine | scale (trace / exact) |
|---|---:|---:|---:|
| 0 | 22.7 | 1.000000 | 8.00× |
| 1 | 17.8 | 1.000000 | 8.00× |
| 2 | 27.0 | 1.000000 | 8.00× |
| 3 | 15.0 | 1.000000 | 8.00× |

Cosine is 1.000000 and the scale is exactly B. Two readings, which must not be
conflated:

**(a) On equal footing the trace is EXACT.** Removing the scale, the trace
gradient matches the pathwise BPTT gradient to **3.7e-07 relative, cosine
1.000000**, at every layer, in **both** decay regimes:

| regime | σ | half-life | scale | equal-footing rel err |
|---|---:|---:|---:|---:|
| default init | 0.544 | 1.14 B | 8.00× | 2.5e-07 |
| `--decay-init 3` | 0.953 | 14.27 B | 8.00× | same to 6 figures |

**Issue #2's claimed bias (decay[0] rel 0.36 / cosine 0.934 at 4 layers) does
NOT reproduce.** The apparent bias was the scale factor below.

**(b) The real defect: the decay rows are not normalised by B.** The loss is
`(...)/B`, so every parameter's gradient carries `1/B` — except the decay
gradient, `(dlds * decaytrace).sum(axis=0)`, which does not. The ratio is
exactly B, verified by sweeping it: **8.00× at B=8, 16.00× at B=16, 32.00× at
B=32.** Direct confirmation: recomputing exact BPTT with the per-step loss *not*
divided by B reproduces the engine's trace gradient to **rel 0.0000, cosine
1.000000**.

### Consequences

- **Under AdamW: nothing.** Adam is per-parameter and scale-invariant, so a
  constant factor on the 8,192 decay coordinates is absorbed. That is why
  889,897 steps ran without incident and why nobody noticed.
- **Under any non-scale-invariant optimizer: a live trap.** Roadmap P4 proposes
  adding SGD as a factorial arm. Under SGD the decay rows would receive an
  effective learning rate **B× larger** than every other parameter — at the
  frozen B=1024, that is 1,024×. Any SGD arm must divide the decay gradient by
  B, or the decay-timescale experiment is measuring a learning-rate blowup
  rather than an optimizer effect.
- **For P0's premise:** §4 action 2 (the hybrid window + exact-TBPTT trainer) was
  conditional on *"if the gate shows the bias"*. It does not show a bias. The
  estimator is not the leading suspect for gradient SNR ≈ 0.05 or for the
  memory wall at this operating point, which converges with #3's own finding
  that decays collapse to the 1-byte regime deterministically and reyvvon's
  result that init + schedule fixed the probe without touching the estimator.
  **Do not build the hybrid trainer on this evidence.**

### Two methodology traps this gate had to clear first

1. **Random bytes make the comparison vacuous.** With iid bytes the recurrent
   states are effectively iid, every lag weighting collapses onto the same
   direction, and cosine comes out a clean 1.0000 for every layer at every
   setting — a confident false negative. Real line-bucketed corpus bytes are
   required, and the gate now says why in its docstring.
2. **Freeze the parameters for the window** (LR=0), or the comparison measures
   AdamW drift instead of the trace.

Not established: T=16–24 steps, dim 32, 2–4 layers, one corpus, B∈{8,16,32}. The
claim is about the estimator's *shape and scale*, which is what P0 needed; it is
not a statement about trained checkpoints, where σ is heterogeneous per
coordinate and the state gain is not ≈1.

---

## The MI target was wrong at every depth >= 1, and the gate that pinned it (2026-10-04)

Re-reading `docs/ideas/papers/2609.12303v1.paper.txt` §2.1 against `distill.py`
found that the 2026-10-03 "one-hot" fix and the depth-0-only target that replaced
it were **both** wrong, in a way that the then-current gate actively enforced.

### What the paper says, and what the code did

The paper conditions on the bytes **already emitted** and aggregates over the
**next** one:

> "To compute the first-byte distribution, the columns sharing the byte prefix B1
> are aggregated. Next, we subset the vocabulary by the ground-truth byte `i` and
> aggregate the probabilities over B2. Finally, we repeat this process for B3."
> — §2.1 / Figure 2

So at depth `d`, with `gold` the byte string of the token being predicted:

    S_d   = { c : bytes(c)[:d] == gold[:d]  and  len(c) > d }
    p_tgt = sum_{c in S_d, bytes(c)[d] == x} p(c) / sum_{c in S_d} p(c)

Two versions of the code got this wrong, in opposite directions:

| version | conditioning | aggregation | result |
|---|---|---|---|
| pre-2026-10-03 | `bytes(c)[:d+1] == gold[:d+1]` (includes byte d) | `bytes(c)[d]` | **one-hot at the gold byte**, entropy exactly 0.000000 nats |
| 2026-10-03 → 2026-10-04 | depth 0 only; **all** depth >= 1 declined | first byte | soft at depth 0, **nothing anywhere else**, coverage 23.1% |
| 2026-10-04 (now) | `bytes(c)[:d] == gold[:d]` (emitted prefix only) | `bytes(c)[d]` | the paper's target at every depth |

The off-by-one in the first version is real and was worth finding. But the
justification for the second — recorded in `distill.py`, `tests/check_mi.py` and
this file — was **false**:

> *"depth >=1 (inside a token): NO TARGET. Declined and counted, because one
> forward pass conditioned on the gold prefix cannot represent byte uncertainty
> there"*

The paper's own worked example contradicts this on the same page: at depth 2 it
gives P(`isu`)=0.8 / P(`isk`)=0.2, and EoT gives 0.667/0.167/0.167. Three-way
splits, from a single forward pass. What one pass genuinely cannot supply is the
mass of candidates that **end** mid-token — which is what `<eot>` is for — not
the uncertainty over the next byte.

The fix is to delete one term from the match, not to delete the depths. All of it
is computable from the dump that already exists (`ids`, `probs`, `tok_len`,
`streams`); zero extra teacher passes.

### Measured, real teacher, real corpus text (k=32)

Per byte depth, over 6000 bytes of `wiki_tinystories_001.txt`:

| depth | positions | target available | non-degenerate | mean entropy | top1==gold | mean retained mass |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1388 | 99.9% | 65.5% | 0.249 | 86.6% | 0.919 |
| **1** | **1261** | **98.7%** | **95.5%** | **1.108** | 44.9% | 0.798 |
| 2 | 1138 | 93.5% | 75.2% | 0.279 | 80.2% | 0.291 |
| 3 | 913 | 83.5% | 30.3% | 0.097 | 90.3% | 0.235 |
| 4 | 545 | 72.7% | 6.1% | 0.018 | 98.5% | 0.169 |
| 5+ | 1358 | ≤70% | ~4% | ~0.01 | ~99% | ≤0.14 |

- depth-0-only implementation: **1387 / 5999 = 23.1% coverage**
- paper's Marginalize-It: **5338 / 5999 = 89.0% coverage**, of which 59.5%
  non-degenerate
- **65.9% of all byte positions carried a usable teacher target that the
  implementation was discarding.**

Depth 1 is the richest band in the whole table (mean entropy 1.108 nats, 95.5%
non-degenerate) and it was entirely thrown away. Depth >= 4 degenerates toward
one-hot under k=32 truncation — that is the real fidelity limit, not the paper's
construction — and is handled by `floor`/`backoff`.

On a full 400 kB dump read through the actual `MiReader` (64 streams, k=32,
`floor=0.05`), after the fix:

| statistic | value |
|---|---:|
| coverage | **64.6%** (was 23.1%) |
| mean target depth | 1.26 |
| mean target entropy | 0.536 nats |
| mean retained prefix mass | 0.454 |
| top-1 == gold byte | 71.1% |
| declined (first byte, or retained ≤ floor) | 35.4% |

Verified against an independent brute-force implementation of the paper's
formula over 2137 positions of a real stream: **max |Δ| = 2.4e-07**.

### A gate that pinned the wrong definition

`tests/check_mi.py` t3 asserted *"only depth 0 is produced; depths >= 1 are
declined"*. A gate that encodes a wrong definition is worse than no gate, because
it makes the wrong thing survive review. It is rewritten, and two things were
added:

- **t7 pins the paper's own worked example**: teacher row `isu 0.5, is 0.125,
  isk 0.125`, gold token `isu` ⇒ B3 = **0.800 / 0.200**. The 2026-10-03 bug
  returns 1.000/0.000; the 2026-10-03→04 misdiagnosis returns no target at all.
  One test now fails for both.
- **t3c** asserts a depth-1 target keeps a non-gold branch
  (`c:0.778, x:0.222`), which is the specific failure mode.

Two fixture bugs surfaced while doing this and are worth recording, because both
were invisible to the depth-0-only test:

1. `stream "abcde"` with `tok_len=[1,2,2]` over a table of `["abc","de","fgh"]`
   — the gold token `"bc"` was **not in the vocabulary**, so the prefix test
   could never match. The old test passed because depth 0 never reads a gold byte.
2. `stream b"zz"+b"isu"` is **5 bytes** while `tok_len=[1,3]` is 4, shifting every
   gold byte by one.

Both fixtures now assert `sum(tok_len) == len(stream)` before use.

### The gradient was masked in one place and not the other

`gpu_port.py`, in `step_batch`:

```python
s_t = torch.where(covered, s_t, torch.zeros_like(s_t))     # masked by covered
...
doutput += self.distill_alpha * (p_s - p_t)                # NOT masked
```

At a declined position `p_t == 0`, so the loss there is exactly zero while the
applied gradient still received `alpha * p_s`. The logged loss was not the
objective being optimised. Fixed by masking `doutput` with `covered[:, None]`.

**Not a bug, and now documented so nobody files it:** the reported loss is a
per-batch mean (`... / B`) while the applied gradient is the un-divided sum, so
`applied == B x ∂loss/∂θ` for the CE path and the KD path alike. This was
verified at `alpha=0` too. It is a logging convention, not a defect, and AdamW
normalises it away.

`tests/check_kd.py` gains **g4**, which the docstring had promised since the file
was written but which was never implemented: finite differences of the reported
loss against the applied gradient, at **partial** coverage. Every earlier check
compared only losses, and the synthetic target it used is non-zero on every row,
so `covered` was always `True` and the mask never engaged.

| | cov 100% | cov 50% | cov 24% | cov 10% |
|---|---:|---:|---:|---:|
| before the fix | 8.62 | 14.65 | 14.37 | 17.26 |
| after the fix | 8.62 | 8.00 | 8.29 | 8.62 |

`B = 8`. Gate 3 (`alpha=0` bit-exactness, `max|Δθ| = 0.0` over 120 steps) still
passes — the mask is unreachable at `alpha=0`.

### The control arm that was missing

With `alpha=0.5` and a target covering only a fraction of positions, `CE` is
down-weighted to `0.5` at **every** position while KD fires on a subset. So a
"KD arm" was never "CE + distillation" — it was "half-strength CE everywhere plus
a conflicting soft label on some positions". Both earlier A/Bs (one-hot and
Case-I) are uninterpretable without this arm: their divergence could have been
the halved CE alone.

`train_gpu.py` gains `--distill-null-target`: `alpha > 0`, all-zero target, no
dump read. The CE weight is `1-alpha` exactly as in a real KD arm and the KD term
contributes nothing. It is the arm that separates "distillation hurts" from
"halving CE hurts", and it costs no teacher time.

### Not established

- Coverage 64.6% is with `floor=0.05`; the floor is now a documented, reported
  knob (`stats()["floor"]`, `retained_below_floor`), not a hidden default.
- Depth >= 4 remains mostly one-hot under k=32. Raising k would recover it at
  linear storage cost; untested.
- The top1==gold rate of 44.9% at depth 1 is MI's **not-ending bias**, and it is
  large. Feeding depth-1 targets in naively will fight CE. This is the
  word-length-drift ruler's job, and it is now measuring something.
- One teacher (LFM2.5-230M-Base), k=32, one corpus, `floor=0.05`.

### The A/B, re-run with the corrected target and the missing control

2000 steps, B=64, seed 0, `--data-seed 0`, identical byte stream, from scratch,
same dump, laptop CPU (498–506 B/s, three arms within 2% of each other). Corpus
is a 200 kB slice of `wiki_tinystories_001.txt`; dump is 46,449 tokens at k=32.

| arm | flags | objective |
|---|---|---|
| A | `--distill-alpha 0` | CE only |
| **C** | `--distill-alpha 0.5 --distill-null-target` | **half-CE control, zero target** |
| B | `--distill-alpha 0.5 --distill-dir …` | MI-KD, corrected target |

Coverage reported by arm B: **82,410 / 128,000 = 64.38%**.

`ce` is the same quantity in all three arms (same CE term, same data), so it is
directly comparable. The total `loss` is **not** — CE is down-weighted to 0.5 in
C and B by construction, which is exactly why C exists.

| step | A loss | A ce | C loss | C ce | B loss | B ce |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 9.242 | 5.951 | 6.323 | 6.063 | 7.293 | 6.029 |
| 200 | 4.015 | 2.638 | 2.800 | 2.963 | 3.254 | 2.815 |
| 1000 | 2.938 | 2.129 | 1.904 | 2.210 | 2.332 | 2.162 |
| 1400 | 2.835 | 2.049 | 1.833 | 2.114 | 2.262 | 2.082 |
| 1975 | — | **1.963** | — | **2.033** | — | **1.998** |

**1. The divergence does not reproduce.** Arm B's total loss peaks at 7.293 (at
step 25, before it settles) and then falls monotonically to ~2.23. The two closed
arms peaked at 16.77 and 860.68. Whatever drove the blow-up, it is not present
once the target is the paper's and `doutput` is masked like the loss is.

**2. The half-CE confound is real but small.** Halving the CE weight costs
**+0.070 nats** of CE (C 2.033 vs A 1.963). This was my leading hypothesis for
the old divergence and it is **not** supported: a 0.07-nat penalty cannot produce
a 860-nat blow-up. Recorded because it was the hypothesis, and because the control
is now a permanent gate on any future KD arm.

**3. KD is CE-neutral at this scale.** B − C = **−0.035 nats** (KD slightly
ahead of its own control); B − A = **+0.035 nats** (KD slightly behind plain CE).
Both are one-seed noise at ±0.035. The honest reading is *neutral*, not *better*.

**Verdict: the 2026-10-03 closure does not survive its own control.** It is not
replaced by "distillation works" — it is replaced by "no longer falsified", and
the experiment that falsified it had two defects in the target and one in the
gradient. A neutral result at ~2.5e13 FLOPs is also what the paper itself
predicts for byte models at low compute (§5.4: byte models are *worse* on
downstream tasks until the crossover at 6.33e22–2.04e23 FLOPs, which is 6–7
orders of magnitude above this run).

**Not established.** One seed, one α (0.5), one corpus, one τ (1), 128 KB of text,
no ruler evaluation, no SNR measurement, no word-length drift. CE here ends at
~2.0 nats against a ruler baseline of 1.17, so this run says nothing about
absolute quality — it is a confound experiment, not a science arm. The depth-1
label conflict (top-1 == gold at 44.9%) is still unmeasured on generated text and
is the most likely reason a longer arm could go either way.

---

## The mask bug's cost, measured in situ (2026-10-04)

The section above records **two magnitudes for one defect**, ~9x apart: the
`gpu_port.py` comment says the unmasked `alpha * p_s` term was "~8% of the decoder
gradient at 24% coverage", while g4 measures 14.37 vs 8.29 at cov 24% — a spurious
term at **73%** of the legitimate gradient. Whichever is right, one of them has to
carry the claim that the mask bug caused the 860.68-nat divergence, and at 8% it
plausibly cannot. So: measure the consequence directly instead of reconciling the
denominators.

### Setup

2000 steps, B=64, dim 512, L16, seed 0, `--data-seed 0`, laptop CPU (~510 B/s),
corpus = a 200 kB slice of `wiki_tinystories_001.txt`, k=32 MI dump.
**Two arms differing in exactly one line** — the `covered[:, None]` mask on
`doutput` — same dump, same seed, same byte stream, no other change.

The dump reproduced the rig's exactly: **46,449 tokens, L_step=2137, k=32**, and
the fixed arm reproduced their arm-B endpoint (**ce 1.998** at step 1875; theirs
1.998 final). So this is the same experiment, not a lookalike.

| step | fixed mask: loss / ce | buggy mask: loss / ce |
|---:|---|---|
| 625 | 2.500 / 2.306 | 2.605 / 2.459 |
| 1000 | 2.354 / 2.159 | 2.481 / 2.297 |
| 1250 | 2.166 / 2.101 | 2.441 / 2.331 |
| **1375** | **2.147 / 2.077** | 2.492 / 2.319 |
| 1500 | 2.135 / 2.049 | 2.526 / 2.287 |
| 1625 | 2.163 / 2.045 | 2.603 / 2.271 |
| 1750 | 2.194 / 2.042 | 2.704 / 2.265 |
| **1875** | **2.162 / 1.998** | **2.782 / 2.215** |

Coverage identical in both: 77,435 / 128,000 = 60.50%.

### What it says

1. **The bug has a real, measured cost: +0.217 nats of CE** (2.215 vs 1.998) — an
   11% gap, from a gradient term that the logged loss never mentioned.
2. **It flips the arm's sign.** The fixed arm's loss is flat-to-falling to step
   1875. The buggy arm's loss **bottoms at step 1375 and then rises monotonically**,
   2.147 -> 2.782, with CE turning up too. The same run, the same target, one line.
   This is the shape that precedes a blow-up.
3. **It is still not sufficient to explain 860.68.** The buggy arm reaches 2.78,
   not 860, in 2000 steps. So the old divergence needed the *old target* as well —
   the depth-0-only/one-hot target and the unmasked gradient were both present in
   the closed arms. **Both defects were load-bearing.** The corrected run has
   neither, which is why it trains.
4. **The 8%-vs-73% discrepancy is now moot as a severity claim** and should be
   treated as un-reconciled bookkeeping rather than two competing statements about
   how bad the bug was. The number that matters is +0.217 nats and a sign flip,
   and it is now measured in situ rather than inferred from gradient norms, which
   is what neither of the original figures was.

### Not established

- One seed, B=64, 2000 steps, one dump, α=0.5. The +0.217 has no SE — it is a
  single-point difference at step 1875, not a mean over seeds. The *sign flip in
  the loss trend* is the more robust observation here.
- The buggy arm's upturn means a longer run plausibly diverges. Untested; the
  860-nat run was a different configuration (the old target) at a different point
  in the loss landscape.
- **Unrelated discrepancy, flagged:** this run reports coverage **60.50%**
  (77,435/128,000) where the rig's reports **64.38%** (82,410/128,000) — same
  denominator, different numerator, same 46,449-token dump. Not explained. Most
  likely the `floor` default or an aggregation difference; worth pinning, since
  coverage is a headline statistic for the target.
- B=64 is 1/16 the frozen B=1024, and gradient SNR is batch-dependent. The mask
  bug's cost at the frozen batch is unmeasured; the spurious term scales with B
  while the legitimate one does not, so the ratio is batch-dependent by
  construction.

---

## The optimizer axis is answered, and the decay role turns out to be optimizer-dependent (2026-10-04)

Isolated study in `../scratch-optimizer-study/`; the repo was not modified. The
SGD optimizer is derived at runtime from this repo's `gpu_port.py`, and the
study's gate proves the derivation is **bit-identical in `adamw` mode** over 120
steps (theta, both moment buffers, every per-step loss, max |Δloss| = 0.0), so
the derived forward/backward is provably the code that produced the control.

**Control, reused not re-run:** `arm_lrdecay1x_t919897.npz` — AdamW, single LR,
30,000 steps from t=889,897, B=1024, data seed 0, taken from the role-specific-LR
factorial. Every arm resumes from the *pinned* `tmt_mixed_long889897_t889897.npz`
with the same seed and step count, so the arms are matched by construction.
Ruler: `--slices 3 --bytes 20000 --xd-bytes 8000 --xd-files 4`.

### The reason an SGD arm is step-matchable here at all

Measured on the incumbent before any arm ran: the per-coordinate normalizer Adam
divides by has quantiles **0.9995 at p1 through 1.001 at p99 — a ratio of
1.00×**, against raw per-coordinate gradients spanning 7.5 × 10²¹×. **Adam's
per-coordinate normalization is inert on this model**; its step collapses to
≈ `lr·sign(m)` and it spends a uniform ±lr budget on all 4,481,793 coordinates.
The companion number: the momentum EMA retains **3.8 %** of gradient magnitude
(`median |m| / RMS(g)`), i.e. the gradient direction flips nearly every step —
which is what SNR ≈ 0.05 means.

Because the normalizer ≈ 1, `lr = 5e-4` is simultaneously the repo default *and*
the lr that reproduces Adam's step on a typical coordinate. The quantiles of
"the SGD lr matching Adam's step" are all **5.005e-4**, so there is no
median→p90 span to grid over. Measured median step multiples:

| arm | median step | × AdamW(5e-4) | what it actually tests |
|---|---:|---:|---|
| SGD 1.25e-4 | 2.711e-06 | 0.25× | step size, **not** the optimizer |
| SGD 5e-4 | 1.085e-05 | **1.00×** | **the optimizer, step-matched** |

### The `/B` precondition, and a first arm that actually satisfies it

The P0 action-1 gate (`3f2faca`) found the decay rows' gradient is **exactly B×
too large** — the loss is `/B` but `(dlds * decaytrace).sum(dim=0)` is not.
Harmless under AdamW (per-parameter and scale-invariant: m → m/B and √v → √v/B
leave `m/(√v+ε)` unchanged, which is why 889,897 steps ran clean). **Not**
harmless under SGD, where the 8,192 decay coordinates would receive an effective
learning rate 1024× larger than everything else.

The correction (`_gradfix`: multiply the 16 `.decay` slices by 1/B, every other
coordinate **exactly 1.0**) is gated on both counts. It is also **free for the
control** — AdamW's trajectory is unchanged — so the control above needed no
re-run. Only the SGD arms did.

### Results at t=919,897, all 30,000 steps from t=889,897

| metric | AdamW control | SGD 5e-4 (1.00× step) | Δ |
|---|---:|---:|---:|
| held-out CE | 1.2094 ± 0.0143 | 1.2279 ± 0.0188 | **+0.0185** |
| shift CE | 1.1648 ± 0.0162 | 1.1744 ± 0.0183 | +0.0096 |
| cross-domain CE | 2.4176 ± 0.1372 | 2.4745 ± 0.1414 | +0.0569 |
| stop-head AUC | 0.9496 | 0.9673 | +0.0177 |
| **SNR trunk / decay** | — (lost) | **0.0649 / 0.0651** | — |
| σ_mean | 0.51911 | 0.51480 | −0.0043 |
| half-life | 1.162 B | 1.138 B | −0.024 B |
| hot dims | 172 | **104** | −68 |
| hot tail @B=1024 | 180.4 MB | 109.1 MB | −71.3 |

### Verdict

**SGD does not beat AdamW at matched step size.** CE is worse in-domain
(+0.0185) and worse cross-domain (+0.0569); both are inside the plan's 0.03 bar
so neither is a regression worth panicking over, but the direction is consistent
and there is no upside anywhere. Crucially **SNR did not move** (0.065, the same
0.05–0.07 band every arm in this project sits in).

That last point is the important one, because it **converges with the roadmap's
independent decision to hold P0 item 2** ("do not build the hybrid trainer",
same date, from the opposite direction). Two unrelated attacks — swapping the
optimizer, and auditing the gradient estimator — both leave SNR ≈ 0.05 exactly
where it was. The blocker is not an estimator problem.

**Do not spend another arm on the optimizer axis.** P4's precondition should be
read as now *satisfiable*, not hypothetical: a valid arm exists.

### The decay role is optimizer-dependent, and that is the real finding

Neither the `/B` note nor P4 states the consequence. AdamW's scale-invariance
means the 8,192 decay coordinates have trained at full `lr` for 889,897 steps
**regardless of their true gradient magnitude**, while a magnitude-proportional
optimizer gives them ~1024× less and freezes them outright:

| | σ_mean | hot dims | half-life |
|---|---:|---:|---:|
| start t=889,897 | 0.51479 | 104 | 1.138 B |
| AdamW | 0.51911 | **172** | 1.162 B |
| SGD (corrected) | 0.51480 | **104 — frozen** | 1.138 B |

σ moved by 1 × 10⁻⁵ over 30,000 SGD steps. So the memory timescale's entire
learned history is a property of **AdamW's normalization**, not of the gradient's
actual magnitude — under a magnitude-proportional optimizer the memory role does
not train at all. This is an **architecture question, not a bug**, and it is also
the honest reason the SGD arm lost: part of what it "lost" was the memory role
standing still.

### Confounds and gaps, stated

- **Hardware differs from the control.** The SGD arms ran on the MX150
  (`ring_credit=auto`→NVRTC); the control ran on a T4 (Triton). Those paths were
  gated numerically equivalent, but that was not re-verified for these
  checkpoints. Stated rather than assumed.
- **The control's SNR telemetry was lost** to an earlier reclaimed VM, so the
  SNR comparison has no matched control row. All other rows come from
  checkpoints that survived.
- One seed, one corpus, one τ-free setting, 30,000 steps of continuation only.
- **Not established:** anything about from-scratch behaviour, longer horizons, or
  whether a magnitude-proportional optimizer can learn a *useful* memory
  timescale from scratch — it cannot be inferred from a frozen-role
  continuation.
- A first, **invalid** SGD arm was run before the `/B` correction and is
  **VOID**; it is recorded in `../scratch-optimizer-study/RESULTS.md` §7 rather
  than here, because its numbers measure a decay-role learning-rate blowup.

### RESOLVED — the 60.50% vs 64.38% coverage gap was never a reproducibility failure

Two rigs reported coverage on the same 46,449-token dump with the same
denominator (128,000 = 2000 steps x B=64) and got **64.38%** and **60.50%**.
Flagged as unexplained. It is explained, and it is a defect in how the statistic
is *reported*, not in the reader or the dump.

**The reader is deterministic.** Walking steps 0..1999 of the one existing dump
through `MiReader` reproduces 82,410 / 128,000 = 64.3828% exactly. No run-to-run
drift.

**The teacher is not the cause.** Re-running the forward pass at 4 vs 8 threads:
4 of 4,495 rows differ, and the *set* difference is **empty** — only reordering
inside the top-32. The candidate set that the prefix filter sees is unchanged, so
coverage cannot move. (Also: 4 threads vs 4 threads twice = 0 rows differ.)

**Coverage is `(definition x floor)`, and `floor` was never printed.** On that
one fixed dump:

| floor | produced | coverage |
|---:|---:|---:|
| 0.00 | 109,042 | 85.19% |
| 0.02 | 90,034 | 70.34% |
| **0.05** | **82,410** | **64.38%** |
| 0.08 | 78,446 | 61.29% |
| **0.091** | **~77,435** | **~60.50%** |
| 0.10 | 76,673 | 59.90% |
| 0.20 | 70,731 | 55.26% |

A **30-point swing** across the plausible floor range, and the two reported
numbers are the same dump at floor 0.05 and floor ≈ 0.091. Both rigs were right.
The number that was missing is the one that determines the answer.

**Fixes.** `train_gpu.py` now prints `floor`, `retained_mean`, entropy, mean
depth and `top1==gold` on the same line as the coverage, and `--distill-floor`
is a real flag. Any future coverage claim must carry its floor.

**What this changes about the earlier claims.** Nothing about the depth-loop fix:
coverage moved 23.1% -> 64.4% *at a fixed floor of 0.05*, and at floor 0.05 the
old depth-0-only target would still be capped near 23% because it declined the
depths outright rather than backoff-skipping them. But every coverage number in
this file is now conditional on a floor, and the depth>=4 band in particular is
floor-sensitive: it is mostly below 0.05 retained mass, which is why it
disappears at 0.09 and not because the target construction changed.

---

## The A/B on a T4 at the frozen B=1024 — and the result replicates (2026-10-04)

`bundles/make_kd_ab_bundle.py` → `colab run --gpu T4 bundles/colab_kd_ab.py`.
10,000 steps at **B=1024** (the frozen config), seed 0, `--data-seed 0`, one
14.93 MB TinyStories slice, k=32, α=0.5, floor=0.05. 3,635,697-token dump.
**OVERALL: PASS**, 1899 s wall.

### Target coverage per depth, T4 / B=1024, floor 0.05

| depth | produced | coverage |
|---:|---:|---:|
| 0 | 253,665 / 253,671 | 100.0% |
| **1** | **207,576 / 225,559** | **92.0%** |
| 2 | 93,060 / 201,219 | 46.3% |
| 3 | 56,348 / 165,164 | 34.1% |
| 4 | 18,912 / 96,486 | 19.6% |
| 5+ | 11,252 / 96,099 | 11.7% |

Reader: coverage **61.2%**, mean target depth **1.103**, top-1 == gold **66.2%**.
Depth 1 was **0.0%** before the fix. Gate 3 on CUDA/CPU: `max|Δθ| = 0.0`.

### The three arms

`ce` is comparable across arms; total `loss` is not (CE is down-weighted by
`1-α` in C and B by construction, which is why C exists).

| arm | final loss | final ce | ms/step | B/s |
|---|---:|---:|---:|---:|
| A — CE only | 2.0593 | **1.3258** | 29.6 | 34,635 |
| C — half-CE control (α=0.5, zero target) | 1.4158 | **1.3735** | 29.7 | 34,497 |
| B — MI-KD, corrected target | 2.0480 (max) | **1.3469** | — | 24,306 |

Arm B covered **67.47%** of positions. **No divergence**: max total loss 2.048.

| | T4, B=1024, 10k steps | laptop, B=64, 2k steps |
|---|---:|---:|
| halved-CE penalty `C−A` | **+0.0477** | +0.070 |
| KD vs its own control `B−C` | **−0.0266** | −0.035 |
| KD vs plain CE `B−A` | **+0.0211** | +0.035 |

**The same three signs and roughly the same magnitudes on a different device, a
16× larger batch, 5× the steps and 75× the data.** That is what makes this a
result rather than an anecdote:

1. Halving the CE weight costs ~+0.05 nats. Reproducible.
2. **KD recovers about half of what the CE discount costs.** B−C is negative in
   both runs: the teacher target is worth roughly half the α=0.5 CE discount,
   leaving ~+0.02 nats versus plain CE.
3. The end state is **neutral, not better**, and the residual gap is the
   undiscounted half of the CE loss, not a distillation deficit.

This independently confirms the prediction recorded in the roadmap's P3: the
objective is linear in `α/(1−α)`, so KD claws back part of what α costs, and at
CE-only optimality (α=0) there is nothing to claw back. **An α sweep will confirm
a negative unless something CE-invisible (word-length drift) shows a benefit
first** — so the α sweep is not the next experiment; the ruler is.

### Why this is not a distillation win, stated plainly

Three independent reasons, all in this file: the arm is 10 MB of TinyStories at
CE 1.35 against a 1.1741 ruler baseline, so it is under-trained and
register-shifted; the paper's byte-model crossover is 6.33e22–2.04e23 FLOPs and
this run is ~1e16; and the objective is linear in α/(1−α), which makes plain CE
the optimum of this family. "Neutral" is the expected result at this scale and is
recorded as such.

### Operational notes (cost me two runs; recording so it costs one)

- **The Colab VM is GC'd even with `--keep`.** The session went 404 while I was
  retrying a download. Anything that must survive has to leave the VM during the
  run — the bundle now pushes from inside the VM via `checkpoint_hub.push`, with
  the token arriving as `script_args` so **no credential is in the bundle**.
- **Large `colab download` is unreliable**: a 142 MB checkpoint failed with
  `IncompleteRead`. Push in-VM instead of download-then-push.
- The TinyStories marker literal arrived as `b""` (empty) in the first bundle:
  `blob.count(b"")` returns `len(blob)+1` (it reported "stripped 16498690
  markers" for a 16.5 MB corpus) and `replace(b"", b"")` is a no-op, so nothing
  was stripped and `tools/mi_dump.py` refused the corpus. Now built from hex and
  asserted: `MARKER len == 13` and `occurrences left == 0`.
- `make_streams` splits by line round-robin, so the SHORTEST stream is 0.766×
  the mean. A corpus sized to exactly `(steps+200)×B` runs ~23% short of the
  requested steps; 1.45× clears it and the step count is clamped to the real
  `L_step` rather than silently truncated.
- `mi_report.py` wrote `by_depth[stem]` containing scalars plus a nested
  `per_depth`; walking it as depth-keyed dies on `int("positions")` — after a
  14-minute dump. Its closing message also still claimed *"depth 0 is the ONLY
  depth that yields a target"*, i.e. the bug this file documents. Both fixed.

### Addendum — the second SGD arm, and the complete picture (2026-10-04)

`armsgdlr0.000125_t919897.npz` (0.25× AdamW's step) completed at 117.9 min.
All three arms below are 30,000 steps from the same t=889,897, same data seed,
same B=1024, all at t=919,897.

| arm | held-out CE | shift CE | cross-domain CE | σ_mean | hot dims | tail | stop AUC | SNR trunk/decay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **AdamW 5e-4 (control)** | 1.2094 ± 0.0143 | 1.1648 | 2.4176 ± 0.1372 | 0.51911 | 172 | 180.4 MB | 0.9496 | — (lost) |
| SGD 5e-4 (**1.00× step**) | 1.2279 ± 0.0188 | 1.1744 | 2.4745 ± 0.1414 | 0.51480 | **104** | 109.1 MB | 0.9673 | 0.0649 / 0.0651 |
| SGD 1.25e-4 (0.25× step) | 1.2049 ± 0.0182 | 1.1632 | 2.2630 ± 0.0952 | 0.51479 | **104** | 109.1 MB | 0.9763 | 0.0564 / 0.0677 |

**Honest reading of the deltas — they are ~1 SE, not decisive.** Combined SEs are
≈0.024 on held-out CE, so: step-matched SGD is **+0.0185 (~0.8σ)** worse
in-domain and +0.0569 worse cross-domain; the 0.25× arm is **−0.0045 (~0.2σ)**
in-domain, i.e. a tie, and **−0.155** cross-domain (~0.9σ on marginal SEs; the
arms are paired on the same four novels, so the effective noise is lower, but
one seed and one ruler setting cannot settle it). The claim that survives is
directional, not significant: **step-matched SGD is not better, and the smaller
step is not worse in-domain while being consistently better cross-domain.**

**No arm moved SNR.** 0.0649 and 0.0564 trunk, 0.0651 and 0.0677 decay — all
inside the 0.05–0.07 band that has now survived the optimizer swap, the
decay-role LR sweep (1×/4×/16×), and the estimator audit. Three independent
attacks, one unchanged number.

**The decay-role result is now structural, not an LR effect.** Across *both* SGD
arms spanning 0.25×–1.00× AdamW's step, σ_mean stayed at **0.51479–0.51480 and
hot dims at 104 — the exact start values to 5 decimal places**, over 30,000
steps. AdamW's control moved the same coordinates to 0.51911 / 172. A
magnitude-proportional optimizer does not train the memory role *at any* step
size in this range; it is not a tuning artifact.

**What cannot be separated with two arms.** Both SGD arms have the decay role
frozen *and* a smaller tail (109 MB vs 180 MB) *and* differ in step size. The
cross-domain gain is therefore consistent with the repo's documented trend that
continued training on this corpus raises in-domain CE — a less-over-trained model
generalizes better — rather than with "SGD". Distinguishing "SGD" from "frozen
decay role" from "smaller step" needs one more arm (SGD at 0.25× step with the
decay role *unfrozen*, i.e. lr-decay scaled up to compensate), which is cheap on
the MX150. **Not run.**

**Not established:** significance of any delta above; from-scratch behaviour;
whether a magnitude-proportional optimizer can learn a useful memory timescale
at all from t=0. The frozen-role result is a statement about *continuation from
an AdamW-trained checkpoint*, and a from-scratch run could differ in exactly the
way that matters.

### Checkpoints pushed to the hub, and the run that pushed them

Pushed to **`bakali1/tmt-checkpoints` (private)** with `run_id kd-ab-2026-10-04`,
via `checkpoint_hub.push` from inside the Colab VM (token passed as
`script_args`; **no credential is in the bundle or in this file**).

| file | sha256 (first 12) | final ce |
|---|---|---:|
| `armA_ce_t010000.npz` | `6b2e4be2e1a2` | 1.3258 |
| `armC_halfce_t010000.npz` | `b4e12434c5a7` | 1.3735 |
| `armB_kd_t010000.npz` | `37919b09a3e8` | 1.3469 |

Each with a `.sha256` sidecar. Verified by pulling `armB_kd` back down:
`checkpoint_hub.validate_npz` accepts it (internal `t` = 10000 = filename),
107 keys / 75 parameter arrays, and the content hash matches the sidecar.

**These are from-scratch 10,000-step experiment arms, not the incumbent
checkpoint**, and their CE (1.33–1.37) is worse than the ruler baseline
(1.1741). They are banked so the A/B/C comparison is reproducible and inspectable,
not as a release. Do not quote arm B as TMT's best model.

### The Colab run is bit-deterministic — and that is the strongest form of this result

The T4 job was run twice end to end (once to measure, once to push). Both runs
agree to every printed digit:

| | run 1 (measure) | run 2 (push) |
|---|---:|---:|
| arm A loss / ce | 2.0593 / 1.3258 | 2.0593 / 1.3258 |
| arm C loss / ce | 1.4158 / 1.3735 | 1.4158 / 1.3735 |
| arm B ce | 1.3469 | 1.3469 |
| `C−A` | +0.0477 | +0.0477 |
| `B−C` | −0.0266 | −0.0266 |
| `B−A` | +0.0211 | +0.0211 |

So the conclusion — *halving CE costs ~0.05 nats, KD recovers about half of it,
and the end state is neutral rather than better* — now rests on three
independent agreements: this T4 run twice bit-for-bit, and the laptop B=64 run
at the same signs and magnitudes. That is the level of evidence this repo
should have had before the 2026-10-03 closure, and did not.
