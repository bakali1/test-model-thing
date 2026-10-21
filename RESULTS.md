# RESULTS — the trustworthy ruler (PLAN-2026-09-22 Phase 0)

Run id / data / seed / bytes / per-slice CE / histogram / chat verdict.
One row per checkpoint that a decision was based on. Never edit a row; append.

## Read this first — current state, 2026-10-04

**The ruler** (below) is unchanged and is the only thing a CE number should be
quoted against. Everything below is append-only history.

### The three standing blockers, as currently understood

| # | blocker | status | where |
|---|---|---|---|
| 1 | gradient SNR ≈ 0.05 | **largely a measurement artifact.** The same estimator at a **frozen** θ reads **0.633** — a **12.6× gap** caused by the iterate moving every byte. Diagnosis is now **over-large step**, not a noise floor. The direct lever (global `--lr`) has **never been swept**. | end of file |
| 2 | memory formation is real but small (mean half-life 1.14 B; 7 of 8192 dims > 8 B) | **unchanged — and now measured rather than assumed.** The state probe puts the memory horizon at **8 B under a uniform-over-256 cue** (`--cue uniform`: cue drawn over all 256 byte symbols, most of which never occur in TinyStories) and at **≥16 B under an in-distribution cue** (`--cue corpus-support`: uniform over the 102 byte values the corpus actually contains; d=16 reads 9.31% against a 0.94% mismatched-cue control). **The horizon is cue-distribution-dependent — it moves with the stimulus, not only with the model.** Against the mean half-life both are several-fold beyond it; against the `max σ^d > 1e-7` per-coordinate tail they are ~128× / ~16× short (that tail is an amplitude bound, not a prediction — see the retraction below). Gate verdict: **INCONCLUSIVE** (random-byte control veto) — the gate definition is an open decision. | end of file, `tools/state_probe.py` |
| 3 | distance-curve / retrieval ruler | **built** (Karpathy 2015 Fig. 5 cell in `tools/eval.py`) and the state-probe half is measured (row 2). | `tools/eval.py`, `tools/state_probe.py` |

### Claims that have been retracted or corrected

Read these before quoting anything older:

- **Adam does not degenerate to `sign(m)`** — it is normalized SGD with a constant
  normalizer (`|update|/|m|` spread 1.0020×, `|m|` spread 763×). AdamW ≡ SGD here.
- **`ρ(1) = +0.52` and SNR `0.96` are retracted** — they came from **uniform random
  bytes**, a trap this repo had already written down for the trace gate. Real text
  gives `ρ(1) = −0.171` and SNR 0.633.
- **The variance hinge is a floor, not a penalty** — `max(0, 1-√var)` pushes latent
  variance **up**; it is anti-collapse.
- **The decay histogram does not predict memory** — the mean half-life *understates*
  the measured horizon several-fold, and the `max σ^d > 1e-7` tail *overstates* it
  (~128× against the 8 B uniform-cue horizon, ~16× against the 16 B
  corpus-support horizon). The tail figure is a **per-coordinate amplitude bound**
  diluted across the 8192-dim state, so it locates nothing; it is not a
  prediction that failed, it is not a prediction. **CORRECTED 2026-10-05:**
  `tools/state_probe.py` used to print "CONSISTENT: … the decay histogram
  predicts the probe's answer", which is false on both clauses and contradicted
  this retraction. It now prints that the histogram makes **no** prediction, and
  why. The horizon itself is also cue-distribution-dependent — see
  "The state-probe ruler" at the end of file.
- **The eligibility trace is NOT biased** — confirmed at production depth (16
  layers, dim 512, both decay regimes): cosine 1.000000, rel err ≤1.05e-06. The
  hybrid-window trainer stays HELD.
- **Gradient accumulation is a weak lever** — efficiency 0.138 at k=64, not the
  1/k the literature framing assumes.

### The three optimizer/LR attacks, merged

All three returned *SNR did not move*. One merged table, three rows, every number
retained in its own section above:

| attack | what moved | CE verdict |
|---|---|---|
| per-role LR (1×/4×/16× on decay) | σ_mean rose 1.82×/2.14× then **saturated**; CE monotonically worse | null, +29% hot tail for zero gain |
| step-matched SGD | decay role **froze outright** (hot dims 104 vs 172; σ moves 1e-5 over 30k) | +0.0185 (~0.8σ) — now attributed to noise + kernel confound |
| distillation KD (A/B/C, 3 runs) | coverage 0%→92%, no divergence (2.048 vs 860.68) | KD recovers ½ its CE discount, **end state neutral** |


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


> **SUPERSEDED / CONSOLIDATED 2026-10-04.** This section is retained
> verbatim and is **not** the current summary. Three separate optimizer and
> learning-rate attacks all returned the same verdict -- *SNR did not move* --
> and are now merged into one table in **"The standing SNR blocker was a
> measurement artifact"** at the end of this file, which also records the
> two corrections that change how they must be read (Adam is not sign-like;
> the ~0.05 figure is ~12.6x pessimistic because theta moves every byte).
> Keep this section for its numbers; read the merge for the conclusion.

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
(`d62a756`). The teacher-disagreement number above is itself evidence for that
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


> **SUPERSEDED / CONSOLIDATED 2026-10-04.** This section is retained
> verbatim and is **not** the current summary. Three separate optimizer and
> learning-rate attacks all returned the same verdict -- *SNR did not move* --
> and are now merged into one table in **"The standing SNR blocker was a
> measurement artifact"** at the end of this file, which also records the
> two corrections that change how they must be read (Adam is not sign-like;
> the ~0.05 figure is ~12.6x pessimistic because theta moves every byte).
> Keep this section for its numbers; read the merge for the conclusion.

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

The P0 action-1 gate (`4e4fdc4`) found the decay rows' gradient is **exactly B×
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


> **SUPERSEDED / CONSOLIDATED 2026-10-04.** This section is retained
> verbatim and is **not** the current summary. Three separate optimizer and
> learning-rate attacks all returned the same verdict -- *SNR did not move* --
> and are now merged into one table in **"The standing SNR blocker was a
> measurement artifact"** at the end of this file, which also records the
> two corrections that change how they must be read (Adam is not sign-like;
> the ~0.05 figure is ~12.6x pessimistic because theta moves every byte).
> Keep this section for its numbers; read the merge for the conclusion.

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

---

## The standing SNR blocker was a measurement artifact — and the optimizer framing was wrong (2026-10-04)

This section supersedes three earlier claims. It is appended, not edited in
place, per the file's convention. Scripts: `tools/rho_probe.py`,
`tools/reconcile_snr.py`, `tools/random_vs_real.py`.

### CORRECTION 1 — "Adam collapses to `lr·sign(m)`" was wrong

`RESULTS.md:1919` reads:

> its step collapses to ≈ `lr·sign(m)` and it spends a **uniform ±lr budget** on
> all 4,481,793 coordinates.

Both halves are wrong. With `√v ≈ 1` uniformly, Adam is

```
Δθ = -lr · m / (√v + eps) ≈ -lr · m
```

i.e. **normalized SGD with a constant normalizer**, not `sign(m)`. Sign-like
behaviour needs `√v ∝ |m|`, which is not the case here. Measured on the
incumbent: the ratio `|update| / |m|` has a p10→p99 spread of **1.0020×** while
`|m|` itself spreads **763.66×** over the same range.

The magnitude claim is refuted by this file's own table two paragraphs below it:
AdamW's measured median step at `lr=5e-4` is **1.085e-05**, i.e. **0.0217 × lr**,
not ±lr.

**What survives, and is now better supported:** AdamW and step-matched SGD are
effectively the *same algorithm* at this operating point, which is *why* the arms
tracked to 0.8σ (1.2279 vs 1.2094). The `+0.0185` should be attributed to noise
plus the NVRTC/Triton hardware confound — **not** to the optimizer. "SGD does not
beat AdamW" stands; "SGD is a different optimizer here" does not.

### CORRECTION 2 — the ~0.05 SNR is mostly parameter motion, not gradient noise

`telemetry.py:27-40` measures `||EMA_b(g)||² / EMA_b(||g||²)` during training,
where **θ moves on every single byte**. Measuring the same estimator at a
**frozen** θ on the same checkpoint and the same B:

| condition | telemetry-EMA SNR | plain k-average, k=96 |
|---|---:|---:|
| θ **frozen** (`lr=0`), real TinyStories, B=1024, t=919,897 | **0.6330** | 0.0994 |
| θ **moving** (real training, same metric) | **0.05** | — |

**A 12.6× gap on the same metric, same checkpoint, same batch size.** At a fixed
point the gradient is ~63% self-consistent; in training it reads 5%. The
difference is that the iterate moves further between updates than the gradient's
spatial consistency length.

This is the **over-large-step regime**, and it is the best-supported explanation
in this file for the SNR blocker. It also retro-explains three results that were
previously attributed elsewhere:

- SGD 0.25× step tied in-domain and won cross-domain — less jitter.
- 16× decay-role LR was mildly worse — more jitter, in the memory role.
- SGD ≡ AdamW — same algorithm, so the delta was noise.

**Caveats, stated:** frozen θ makes the field stationary by construction, so 0.633
is an upper bound; consecutive steps read overlapping bytes, which inflates
agreement; and averaging during training will not fully recover it because the
iterate still drifts within the window.

### CORRECTION 3 — random bytes manufacture a confident result. ρ(1) flips sign.

An external check reported `ρ(1) = +0.52` and SNR 0.9637 at frozen θ. Its batch
was built as `torch.randint(1, 250, (STEPS+1, B))` — **uniform random bytes**.
Re-running that exact script (same checkpoint, B=1024, 96 steps, same telemetry
call) changing only the data:

| | random bytes | **real TinyStories** |
|---|---:|---:|
| telemetry-EMA SNR, k=96 | 0.9637 | **0.6330** |
| plain k-average, k=96 | 0.8525 | **0.0994** |
| ρ(1), per-coordinate | **+0.534** | **−0.171** |
| ρ(32), per-coordinate | +0.099 | −0.017 |

ρ(1) **changes sign**. On uniform random bytes there is nothing local to model,
so the response collapses onto one constant direction and every step agrees —
confidence manufactured by the data, not found in the gradient. The plain-average
SNR differs 8.6× between the two arms.

**This is the trap `PLAN-2026-10-04-ROADMAP.md` P0 item 1 already wrote down** for
the trace gate: *"with random bytes the states are iid and every lag weighting
collapses to the same direction, giving a confident cosine 1.0000 for free."* Same
trap, different measurement. **Every future gradient-quality gate must assert
real corpus bytes**, not synthetic ones.

The negative ρ(1) on real text is itself worth recording: consecutive real-text
gradients partially **cancel**, which is the opposite of the random-byte result.

### The accumulation ladder, measured on real text rather than guessed

Efficiency = achieved ÷ ideal `1/k` noise reduction, frozen θ, real bytes:

| k | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| efficiency | 0.589 | 0.357 | 0.242 | 0.185 | 0.154 | **0.138** |

k=64 buys ~8.8 effective samples, not 64. Accumulation is **real but ~7× weaker
than the independent-sample prediction**, and the plain k-average on real text is
non-monotonic (0.144 at k=8, 0.056 at k=32, 0.099 at k=96). So "just accumulate"
is not the lever the literature framing implies — **the direct lever on step size
is the learning rate**, and it has never been swept on the global axis.

### What is now established, in one table

| claim | status |
|---|---|
| the eligibility trace is exact at production depth (16 layers, dim 512, both decay regimes, cosine 1.000000, rel err ≤1.05e-06) | **established today** |
| AdamW ≈ SGD at this operating point; "sign(m)" was wrong | **established today** |
| SNR 0.05 is ~12.6× pessimistic; the cause is parameter motion | **established today** |
| ρ(1) = +0.52, SNR = 0.96 at frozen θ | **RETRACTED — random-byte artifact** |
| gradient accumulation is a strong lever | **not supported**; efficiency 0.138 at k=64 |
| the variance hinge penalizes latent variance | **RETRACTED** — `max(0, 1-√var)` is a floor, not a penalty |
| global learning rate above 5e-4 | **never tested** — the only LR factorial was decay-role |

### The one experiment this points at

The over-large-step diagnosis has a direct, cheap, never-run test: **sweep the
global learning rate.** Every LR factorial in this repo moved `--lr-decay`, the
decay role's rate, never `--lr`. Given that the measured step is already
0.0217 × lr, the nominal lr and the effective step are not the same quantity, and
nobody has established where the effective step should sit. This is ranked ahead
of accumulation because accumulation costs k× the updates and this costs none.

---

## The state-probe ruler: the memory horizon is 8 B under a uniform-over-256 cue and ≥16 B under an in-distribution cue — the number is cue-distribution-dependent, i.e. partly a property of the stimulus (2026-10-05)

`tools/state_probe.py`, CPU only, numpy_port, 8.4 MB of forward pass.
Roadmap v3 P0 item 2. Artifacts: `runs/state_probe/probe_final.log` (246 lines,
the original 16384-B sweep + 12 fold partitions), swept states in
`runs/state_probe/sweep_512b.npz` (**248 MB**, re-fits the probe without
re-paying the sweep), and the cue-distribution runs re-derived on 2026-10-05
under `/tmp/state_probe_verify/` (six logs: `uniform_seed{0..4}.log`,
`corpus_support_seed0.log` — `runs/` and `*.log` are gitignored, so copy them
into `runs/state_probe/` if you want them to outlive this machine).
**The gate returns INCONCLUSIVE and this section records it as such; it does not
close blocker 2.**

> **CORRECTED 2026-10-05 — the headline used to be unqualified and it was
> wrong.** The section title, the header index row and the `**8 bytes, firm.**`
> line all said "the memory horizon is 8 bytes", with no mention of the cue
> distribution. `grep -i corpus-support RESULTS.md` returned **zero hits**: the
> cue distribution was disclosed once, deep in the protocol, and never again.
> The 8 B number is specific to `--cue uniform`, in which the cue is drawn over
> all 256 byte symbols — **most of which never occur in TinyStories at all**.
> Under `--cue corpus-support` (uniform over the 102 byte values the corpus
> actually contains) the same tool on the same checkpoint decodes the cue
> **further** out. **8 B is therefore partly a property of the STIMULUS, not of
> the model**, and quoting it without the cue mode quotes half a measurement.
> Both numbers are now given, each attributed to its cue mode, below.

### The protocol, with its conventions, because an accuracy without these is not a result

Per trial: zero the recurrent state; feed **one cue byte drawn uniformly over all
256 symbols** (`rng.integers(0, 256)`, seed 0) — never a word, never a
corpus-common byte, because a surface-form cue reproduces #23's confound; feed
`d` bytes of **real held-out distractors**; snapshot the state after the model
has consumed cue + exactly `d` distractor bytes. Then fit a tiny **offline
closed-form ridge probe** (ridge onto one-hot targets, solved in the dual from an
n×n Gram — never an 8192×8192 system, no SGD, no TMT updates) and decode the cue.
**Chance = 1/256 = 0.390625%.**

**The cue distribution is the load-bearing variable, and it is a `--cue` flag.**
`--cue uniform` (the protocol as written above), `--cue corpus-support`
(uniform over the byte values the distractor file actually contains — 102 of them
for TinyStories), or `--cue corpus-freq` (∝ corpus frequency; reported with a
warning, because a constant "always answer the modal byte" predictor scores 19.1%
with zero memory). **The horizon this tool reports is a function of that flag**,
and the two measurements below differ by a factor of two in horizon on the same
checkpoint with everything else held fixed. A prior version of the header printed
`CUE : ONE byte, UNIFORM over all 256 symbols` **unconditionally**, so
`--cue corpus-support` and `--cue corpus-freq` runs printed a false banner 18
lines before the truth; the banner is now conditional on `--cue`.

| convention | value |
|---|---|
| checkpoint | `runs/lrroles/arm_lrdecay1x_t919897.npz` (t=919,897, dim 512, L16) |
| state read out | concatenated layer states, 16 × 512 = **8192 dims** |
| trials | **512** (random-byte control arm: 64) |
| CV | 4-fold, every trial tested exactly once |
| λ grid | 1 … 1e7, chosen per outer fold by inner 3-fold CV on training folds only |
| preprocessing | per-dimension z-scoring on pooled, label-free statistics |
| distractors | 1146 disjoint 16384-B windows of `data/eval/tinystories_valid.txt`, asserted outside the training glob |
| distances | 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 |
| calibration | 0, 1, 2, 4 B (probe power) |
| context CE on the probe's own streams | **not recoverable from the shipped run** — see the correction below |
| SE | binomial at n=512; stability = **12 independent fold partitions** |
| resolution | smallest detectable effect at 2σ = **0.778 points** |

**REMOVED 2026-10-05 — `1.1893 nats/byte`.** This table row carried a
context-CE that appears in **no log in the repo** (`grep -rn 1.1893` outside this
file: nothing). The shipped artifact `runs/state_probe/probe_final.log` carries
`0.0030 nats/byte`, which is the *broken* accumulator — it sat inside the
snapshot branch, summed ~13 values and divided by all 16385. Re-derived fresh
runs read **1.2214–1.2540 nats/byte** at the settings used for the cue
comparisons below (n=512, 256-B distractor windows); short windows read higher
still (1.65 at 32-B windows), and the ruler's ≈1.17 is a 1024-B-slice number, so
none of these are directly comparable to it. The correct statement is the
**measured re-derivation**, not the unsourced 1.1893.

### Decoding accuracy vs distance (single 4-fold partition, n=512)

| d (B) | accuracy | chance | mismatched-cue control | carried? |
|---:|---:|---:|---:|---|
| 0 * | 77.3438% | 0.3906% | 0.5859% | probe live |
| 1 * | 58.3984% | 0.3906% | — | probe live |
| 2 * | 46.2891% | 0.3906% | — | probe live |
| 4 * | 27.3438% | 0.3906% | — | probe live |
| **8** | **12.8906% ± 1.4809** | 0.3906% | **0.0000%** | **YES** |
| 16 | 0.7812% ± 0.3891 | 0.3906% | 0.7812% | no |
| 32 | 0.5859% ± 0.3373 | 0.3906% | 0.5859% | no |
| 64 | 0.5859% ± 0.3373 | 0.3906% | 0.1953% | no |
| 128 | 0.5859% ± 0.3373 | 0.3906% | 0.1953% | no |
| 256 | 0.3906% ± 0.2757 | 0.3906% | 1.3672% | no |
| 512 | 0.1953% ± 0.1951 | 0.3906% | 0.5859% | no |
| 1024 | 0.1953% ± 0.1951 | 0.3906% | 0.0000% | no |
| 2048 | 0.3906% ± 0.2757 | 0.3906% | 0.3906% | no |
| 4096 | 0.1953% ± 0.1951 | 0.3906% | 0.1953% | no |
| 8192 | 0.3906% ± 0.2757 | 0.3906% | 0.0000% | no |
| 16384 | 0.3906% ± 0.2757 | 0.3906% | 0.1953% | no |

\* calibration rows. "mismatched-cue control" = the identical pipeline on
permuted cue labels; it is the specificity test that the probe is reading the
cue and not trial identity.

### The same table under 12 independent fold partitions — which rows survive

A single 4-fold partition at a 1–2% accuracy is 4–8 correct decodes out of 512
and the control is 2–5 hits out of 512; whether the difference is 4 hits or 2 is
decided by the partition, not by the model. So every row was re-probed 12 times
with fresh partitions:

**The `±` in this table is WITHIN-SWEEP partition noise over ONE fixed set of 512
states — it is NOT a confidence interval on anything.** Five independent sweeps
of the same checkpoint (new cue draws, new corpus windows, fresh partitions) put
the d=8 mean at 12.73–15.43%, a **2.70-point spread, ~1.8× wider than the ±0.940
printed here**. Read the table as "this state set, this partition family", and
read the between-sweep table further down as the uncertainty. The line at the end
of this section that says "the horizon number does not have a CI" is correct and
this table must not be read against it.

| d (B) | accuracy (SD over 12 partitions) | control | acc − control | range over partitions | verdict |
|---:|---:|---:|---:|---:|---|
| **8** | 12.467% ± 0.940 | 0.407% ± 0.333 | **+12.061 pts** | min **+10.352** | **stable, above control** |
| 16 | 1.953% ± 0.529 | 0.472% ± 0.232 | +1.481 pts | min +0.781 | stable, above control **here** — marginal across sweeps (below) |
| 32 | 0.521% ± 0.351 | 0.374% ± 0.232 | +0.146 pts | −0.781 … +1.172 | sign flips — null |
| 64 | 0.407% ± 0.232 | 0.358% ± 0.262 | +0.049 pts | −0.586 … +0.586 | sign flips — null |
| 128 – 16384 | 0.130–0.505% | 0.260–0.342% | −0.212 … +0.228 pts | all cross 0 | null |

**d=8 is the only robust decoding result in this sweep. d=16 is ~8 excess decodes
out of 512 and is MARGINAL, not absent: it is `stable, above control` on this
sweep's 12 partitions, and on the five fresh sweeps below it is stable above
control in **4 of 5** and sign-flips in **1 of 5**. The honest sentence is
therefore: *at the shipped `--cue uniform` settings, d=16 is marginally but
inconsistently above control.* It is NOT established as a horizon, and it is NOT
honest to write it up as a null either.

**REMOVED 2026-10-05 — the sentence that used to close this paragraph.** It read
"changing which corpus windows are drawn moves its control from 0.47% to 0.98% and
flips its classification, so d=16 is NOT established." That run has **no artifact
anywhere in the repo** (`grep -rn` for the values finds nothing outside this
file), and it was the *sole* stated basis for excluding d=16. It is removed rather
than kept, because the five logged sweeps above give the same conclusion with
artifacts behind it — without pretending to a corpus-window run that was never
recorded.

### MEMORY HORIZON — it depends on the cue distribution

**The horizon is 8 B under a uniform-over-256 cue and ≥16 B under an
in-distribution cue. The number is cue-distribution-dependent, and that
dependence belongs to the STIMULUS.**

This number did not exist before 2026-10-05, and the first version of it was
wrong by omission. It was written as **"8 bytes, firm"**, unqualified, in the
section title, in the header index row and here — while the cue distribution
behind it was disclosed once, deep in the protocol, and never again
(`grep -i corpus-support RESULTS.md` returned **zero hits**). A reader landing on
any of those three places got a property of the model's memory and, in fact, got
a property of the byte sampler.

The two measurements, same checkpoint, same probe, same folds, 512 trials:

| `--cue` | cue is drawn | d=8 | d=16 | d=32 | horizon |
|---|---|---|---|---|---|
| `uniform` | over all **256** byte symbols; ~154 of them never occur in TinyStories | 12.73–15.43% (5 sweeps) vs 0.26–0.42% control | 1.0–2.7%, stable above control in **4 of 5** sweeps, sign-flips in 1 | sign-flips / null in all 5 | **8 B** |
| `corpus-support` | uniformly over the **102** byte values the corpus actually contains | **52.31% ± 1.63** vs **0.879% ± 0.293** control, diff **+51.43 pts**, stable on 6/6 partitions | **9.31% ± 0.72** vs **0.944% ± 0.175** control, diff **+8.37 pts**, stable on 6/6 partitions | diff +1.30 pts, min 0.000 — **sign flips**, not a finding | **≥16 B** |

That is a factor of two in horizon on the same weights, from the cue sampler
alone. The uniform-cue reading is *not wrong* — it is a clean, reproducible
result (d=8 decoding at 12.7–15.4% against a 0.3–0.4% mismatched-cue control,
margins **+12.3 to +15.2 points**, five sweeps, every one of them `stable, above
control`). It is a measurement of **how long a cue the model has essentially
never seen survives**, which is the harder case and was the design intent. The
corpus-support reading measures **how long a plausible cue survives**, and that
is the number a reader usually means by "the memory horizon". Both belong in the
headline; neither alone does.

Command lines (each log carries the full output):

```
.venv/bin/python tools/state_probe.py --cue uniform        --seed S \
    --distractors data/eval/tinystories_valid.txt \
    --distances 8,16,32,64,128,256 --calibration-distances 0,1,2,4 \
    --trials 512 --batch 512 --control-trials 64 --stability 6 --label cue-uniform-seedS
.venv/bin/python tools/state_probe.py --cue corpus-support --seed 0 \
    --distractors data/eval/tinystories_valid.txt \
    --distances 8,16,32,64,128,256 --calibration-distances 0,1,2,4 \
    --trials 512 --batch 512 --control-trials 64 --stability 6 --label cue-corpus-support-seed0
```

#### Between-sweep spread, `--cue uniform` (the number the ±0.940 must not be read as)

| seed | d=8 accuracy, mean of 6 partitions | control | acc − control | d=16 | horizon |
|---:|---:|---:|---:|---|---:|
| 0 | 12.956% ± 1.013 | 0.391% ± 0.113 | +12.565 pts | stable above control, +0.879 pts | 8 B |
| 1 | 15.430% ± 0.575 | 0.260% ± 0.184 | +15.169 pts | stable above control, +2.279 pts | 8 B |
| 2 | 15.072% ± 1.248 | 0.260% ± 0.146 | +14.811 pts | **sign flips** (+0.716 pts, min 0.000) | 8 B |
| 3 | 13.444% ± 1.066 | 0.391% ± 0.226 | +13.053 pts | stable above control, +1.042 pts | 8 B |
| 4 | 12.728% ± 0.847 | 0.423% ± 0.208 | +12.305 pts | stable above control, +1.302 pts | 8 B |

**d=8 → 12.73–15.43%, a 2.70-point between-sweep spread.** The horizon is 8 B in
all five. d=16 is the marginal row, and it is marginal *in the honest direction*:
above control most of the time, never absent.

#### The amplitude bound, which is what actually transfers

Because the state is `s_t = σ·s_{t-1} + enc_t`, the cue's relative amplitude at
distance `d` is `σ^d`, known in closed form rather than fitted. That converts the
probe's response curve into a bound that transfers: the probe's detection
threshold sits between `σ_mean^8 = 5.3e-03` (detected, 12.5%) and
`σ_mean^16 = 2.8e-05` (not detected). **Anything below ~5e-03 relative amplitude
is UNMEASURED, not absent** — the probe cannot see it, and a null there is a
statement about the instrument.

This is also the cleanest statement of the cue-dependence, and it is worth
separating from the horizon number because it is *not* cue-dependent: the
amplitude threshold below is a property of the **instrument**, at roughly
`σ_mean^8`, whichever cue you feed it. What changes with the cue is **where that
threshold is crossed**, because an in-distribution cue byte carries more
amplitude into the state than a byte the corpus has never contained. So the
honest decomposition of the headline is: **the instrument's threshold is fixed;
the stimulus decides where the threshold lands; the "8 B" was a threshold
crossing, not a wall.**

### The 2×2, and the two cells this design cannot reach

| state has cue? | output uses cue? | cell |
|---|---|---|
| no | no | no learned memory at this distance (d ≥ 32 under `--cue uniform`; d ≥ 64 under `--cue corpus-support`) |
| **YES** (d=8, and d=16 under an in-distribution cue) | **no** | **memory exists; the predictive head/objective does not use it** |

At d=8 the head's probability on the cue byte moves by a differential of
**−1.5e-03 [−5.3e-03, +1.5e-03]** against a matched mismatched-cue control
(stated bar: 0.01 in probability units). Nothing above 5.5e-03 anywhere in the
sweep, i.e. nothing.

**Two cells are structurally unreachable here and are not reported.** The
bottom-right "yes/yes" is not established, because with a *uniform random* cue
there is no correct continuation: leg B measures whether the output distribution
moves, which follows almost automatically from leg A since the byte head is a
fixed linear map of the state. And the "state no / head yes for **literal** cues"
row — the surface-form-shortcut row — cannot be produced by this design at all,
because it requires a cue with surface form, which is exactly what this probe is
forbidden to use. **A "yes/yes" from this tool would not mean task-relevant
retrieval.**

### VERDICT: INCONCLUSIVE, and the reason is the control, not the effect

The random-byte control arm (identical pipeline, uniform random distractors,
n=64) read **above chance at 8 of 16 distances**, **9.375–18.750%** against real
text's 0.2–1.6%, and at d=32 it reads **18.75% where the real arm reads 0.59%**.
Under the gate's stated rule that is a veto, so the tool returns INCONCLUSIVE.

**CORRECTED 2026-10-05.** This paragraph used to say the control arm read
"3–19% against real text's 0–13%". The logged control-arm values flagged above
chance span **9.375%–18.750%** (the 3–19% and 0–13% figures appear in no log);
both ends are now the values in `runs/state_probe/probe_final.log`.

**Recorded as an open decision, not silently re-specified.** Changing a gate
after seeing the data is the failure mode this file has shipped once. The case
that the veto is mis-specified: in that arm the cue is *genuinely present* in the
state (the recurrence is deterministic), and on random bytes the distractor
content sums to something nearly constant across trials, so a planted cue is a
larger share of the trial-to-trial variation than on real text. A high reading
there is a true positive, not a fabrication, so that arm cannot separate "the
probe is powerful" from "the probe is faking". The arm that can is the
mismatched-cue control, and it sits at chance at every distance (0.00–0.59% at
d=8). **Whether the random-byte arm should remain a hard veto is a gate
definition and it is unresolved.** Supporting evidence that the veto itself is
weak: the set of distances it flags moves with the run ([0,1,2,4,8,16,32,64] in
`runs/state_probe/probe_final.log` vs [0,1,2,4,16,32,64,128] in
`runs/state_probe/probe.log`), because at n=64 one lucky decode is 1.6%.

**CORRECTED 2026-10-05 — "on the same sweep" was false.** These two flagged sets
were described as coming from the same sweep. They do not:
`probe.log` reads d=8 at **13.8672%** and `probe_final.log` reads d=8 at
**12.8906%**, so they are two different runs of the tool on the same checkpoint.
The conclusion — that the veto flags an unstable set of distances — survives;
the sentence claiming a single sweep did not, and no claim is made here about
which corpus windows either run drew.

### Consistency with the decay histogram: they disagree, in both directions

Checkpoint decay: σ min 0.02781, mean **0.51911**, max **0.98474**; mean
half-life **1.162 B**; **8 of 8192** dims above 8 B.

| quantity | value | vs the 8 B (`--cue uniform`) horizon |
|---|---|---|
| mean half-life | 1.162 B | **understates ~7×** — the cue is decodable ~7× further than the mean |
| `max σ^d > 1e-7` survives to | **1024 B** (1 dim of 8192) | **overstates 128×** (16× against the 16 B corpus-support horizon) |

Neither summary number predicts the horizon, and the reason is a category error
worth stating once: the decay histogram's tail figure is a **per-coordinate
amplitude** (`σ_max^8 = 0.88`, i.e. the hottest coordinate keeps 88% of the cue
at 8 bytes) while the probe reads an **8192-dim concatenation** in which that
surviving signal is spread over a handful of coordinates. The two measure
different things; the probe measures the second. Anyone quoting "memory up to
1024 B" from the decay tail is quoting a statement about one coordinate, not
about the state.

**The tool and this section used to disagree about this, and the tool was
wrong.** `tools/state_probe.py` printed

> `CONSISTENT: the probe's horizon is at or below what the decay parameters
> permit, and the decay histogram predicts the probe's answer.`

while this file filed the opposite as a retraction. Both cannot stand, and the
tool's string is false on **both** of its clauses: the tail reaches 1024 B
against a measured 8–16 B, and a per-coordinate bound sitting 16–128× above the
measurement is not a prediction of it. The 1024 B figure is checked directly in
`probe_final.log` — `dims with σ^d > 1e-7` is 1 at d=1024 and 0 at d=2048 — and
it is a real property of the weights; it is simply a statement about one
coordinate, not about the horizon. **As of 2026-10-05 the tool prints the honest
version**: "NO CONTRADICTION, BUT THE DECAY HISTOGRAM DOES NOT PREDICT THE
PROBE … the tail is an UPPER BOUND ON A SINGLE COORDINATE'S AMPLITUDE, not a
prediction … A bound N× above the measurement cannot locate the measurement."
This retraction and the tool now say the same thing.

### CHECKPOINT IDENTITY — the file probed is not the checkpoint behind the quoted statistics

`runs/lrroles/arm_lrdecay1x_t919897.npz` is **not** the model behind the
"mean half-life 1.14 B; 7 of 8192 dims > 8 B" figure quoted in blocker 2 and in
`PLAN-2026-10-04-ROADMAP.md`. It is a 30,000-step AdamW continuation from
t=889,897 and measures **1.162 B / 8 dims > 8 B**. The 1.138 B / 7 dims pair
belongs to `armsgdlr0.000125_t919897.npz`, whose σ_mean 0.51479 is the
incumbent's value exactly — consistent with this file's finding that the decay
role is frozen under a magnitude-proportional optimizer.

Decay statistics for all four t=919,897 checkpoints (re-derived from the
checkpoint files themselves, so these columns are attributable):

| checkpoint | σ_mean | mean half-life | dims > 8 B |
|---|---:|---:|---:|
| `arm_lrdecay1x` | 0.51911 | 1.162 B | 8 |
| `arm_lrdecay4x` | 0.52264 | 1.179 B | 9 |
| `arm_lrdecay16x` | 0.52404 | 1.189 B | 9 |
| `armsgdlr0.000125` (SGD, decay role frozen) | 0.51479 | 1.138 B | 7 |

**REMOVED 2026-10-05 — the `d=8 accuracy` column (12.1094 / 12.5000 / 12.3047 /
11.5234 %).** None of those four numbers appears in any log in the repo
(`grep -rn` across the tree: nothing), and the three checkpoints other than
`arm_lrdecay1x` have no probe log at all. The σ_mean / half-life / `dims > 8 B`
columns of the same table are computed directly from the checkpoint weights and
are exactly correct, so they are kept; the accuracies are not, and an
unattributable number next to an attributable one is worse than a gap.

Consequently the previous closing sentence here — "the 8 B horizon is the same
on all four" — is also removed: it rests only on the removed column. What
survives is the checkpoint-identity finding itself, which is about the decay
statistics and needs no probe run: the file named in blocker 2 is not the file
behind the half-life figure that blocker 2 quotes.

### Four defects found in the probe while building it

Recorded because each produced a plausible-looking wrong number, which is the
failure class v3 P0 item 3 exists to prevent:

1. **`p_cue` indexed the output distribution by the next REAL byte, not the
   cue.** The "does the output use the cue" leg was silently measuring ordinary
   byte predictability. It read a believable 0.52 at d=8 — closely matching the
   distance-curve cell's `p_close` at the same distance, which is exactly why it
   survived review. That coincidence is the whole lesson of this file applied to
   me.
2. **Calibration rows reused the previous iteration's mismatched-cue control**,
   judging d=2 and d=4 against a control belonging to a different distance.
3. **The context-CE sanity accumulator sat inside the snapshot branch**, summing
   ~13 values and dividing by 16385. It printed "0.0030 nats/byte" for a stream
   whose true CE is ~1.2 — the check was hiding the bug it existed to catch.
4. **A significance-only bar** on leg B classified a 2e-3 shift in probability as
   a result. The bar is now stated in probability units (0.01).

Two methodological corrections are also recorded because they changed numbers:
the leg-A gate compares against the pipeline's **own empirical null** (permuted
labels) rather than the analytic chance rate — at n=256 one lucky control decode
is enough to clear chance + 2σ and would have vetoed a real 5% reading — and the
`--stability` sweep exists because a single fold partition cannot decide a
1–2% row.

### Seven more defects, found by adversarial audit (2026-10-05)

Same failure class as the four above: each produced a plausible-looking wrong
number, a wrong *label*, or a destroyed sweep. All reproduced before fixing.

1. **`max()` on an empty generator destroyed the sweep.** The
   "horizon recomputed on stable rows" line was guarded only by `if rowsA:`, so
   when no row had `has` set it raised `ValueError: max() iterable argument is
   empty`. Reproduced live. Worse: `if args.save:` sat *after* that block, so a
   completed multi-minute sweep was discarded with nothing on disk. **Fixed
   twice over** — the horizon line is guarded, and **`--save` now runs
   immediately after both arms exist, before the 2×2, the horizon, the amplitude
   table and the consistency section**, so nothing printed after the sweep can
   now cost you the sweep.
2. **Rows with a negative or zero differential were announced as "stable, above
   control."** The `--stability` bucket was gated on `flips` alone, so a row the
   tool had just labelled "consistently null" landed in the same list as a real
   result. Reproduced live (`stable, above control: [8, 16, 32]` on rows reading
   −3.125 / 0.000 / 0.000 pts). **Fixed**: the append is gated on
   `d_r.mean() > 0`, the "consistently null" rows are reported in their own
   bucket, and the horizon is recomputed from the **partition-aggregated**
   decision rather than the single-partition `has` it used before — which is the
   very quantity `--stability` exists to distrust.
3. **The `CUE : ONE byte, UNIFORM over all 256 symbols` banner was
   unconditional**, so `--cue corpus-support` and `--cue corpus-freq` runs
   printed a false header 18 lines before the truth. Since the cue distribution
   turns out to be the load-bearing variable in *every* conclusion this tool
   draws, this is not cosmetic. **Fixed**: the banner is selected on `args.cue`,
   `CUE MODE : --cue <mode>` is printed, and the closing reminder names the cue
   the horizon belongs to.
4. **`--load` died with a bare `KeyError`** when `--distances` asked for
   distances the archive never held — e.g. re-loading a 4-distance `.npz` with
   the default distance list. **Fixed**: the request is intersected with the
   archive's `X_<d>` / `p_cue_<d>` / `p_mis_<d>` keys, the mismatch is printed
   before the header, and a run whose *main* distances are entirely absent
   refuses to print a horizon.
5. **Off-by-one in the context-CE accumulator.** `nxt = seq[:, min(t+1, Dmax)]`
   meant the final step scored the byte the model had *just consumed* — a
   trivially high-probability token — dragging the mean down. The step is now
   skipped and the divisor counts scored positions only.
6. **The tool claimed the decay histogram "predicts the probe's answer."**
   Reconciled above: the string is corrected and this retraction now agrees with
   what the tool prints.
7. **`INCONCLUSIVE` exits 2, undocumented.** The code is right — it is the tool
   correctly refusing to call a result — but a `set -e` wrapper treats the
   primary documented outcome as an error. **Documented rather than changed**:
   the verdict block now prints the exit code and states that 2 is the
   documented INCONCLUSIVE status, with the wrapper to use if it matters.

Two more RESULTS.md defects were numbers with **no artifact anywhere in the
repo**, removed rather than kept: the `1.1893 nats/byte` context CE (the shipped
log carries the broken `0.0030`; fresh runs read 1.22–1.25 nats/byte at these
settings) and the four-checkpoint `d=8 accuracy` column. The `0.47% → 0.98%`
corpus-window sentence and the `3–19%` control-arm span were likewise removed or
corrected above.

### Not established

- The verdict is **INCONCLUSIVE** and the gate definition is unresolved. This
  section does **not** close blocker 2 and does not establish "memory exists but
  is unused" as a property of the model — only that at 8 bytes under a uniform
  cue (and at 16 bytes under an in-distribution cue) the cue is linearly
  decodable from the state and the head does not respond to it.
- **The horizon is a property of the stimulus as much as of the model.** It is
  8 B under `--cue uniform` and ≥16 B under `--cue corpus-support`, on the same
  weights. A horizon quoted without its cue distribution is not a measurement.
  Neither number has been swept beyond two cue distributions; `--cue
  corpus-freq` exists and warns about its modal baseline but has not been run at
  scale.
- A **linear** readout can miss information that is present but not linearly
  available. That is the main alternative to "the state does not carry it".
- The probe's threshold is an **amplitude** bound (≈5e-03 relative); every
  distance row beyond that is UNMEASURED, not null.
- **The cue is planted first**, so it is always the oldest and most attenuated
  element of the state — the hardest possible retention case. Deliberate (it is
  the only way to plant a cue whose presence is not predictable from surface
  form), and it is a real limitation on the horizon number.
- One checkpoint family, one training corpus, one stream length, 512 trials. The
  d=8 result reproduces across 12 fold partitions and 5 independent sweeps
  (12.73–15.43%, a 2.70-point spread) and the horizon is 8 B in all five. **The
  horizon number has no CI**, and the `±` printed on any single sweep is
  within-sweep partition noise, ~1.8× narrower than the between-sweep spread.
- The `d=16` row is marginal rather than null at the shipped `--cue uniform`
  settings: stable above control in 4 of 5 sweeps, sign-flipping in 1 of 5.
- Says nothing about literal-cue induction (#23's domain), obfuscated or
  parallel-tracking retrieval (never tested by anything in this repo), or
  language-modelling quality.
- The saved state snapshots are **248 MB**, not 2.4 GB (`du -h
  runs/state_probe/sweep_512b.npz` → 248M; the earlier 2.4 GB figure was off by
  10×). `runs/` and `*.log` are gitignored, so neither the snapshots nor the
  logs are in git — delete them freely.

---

## RETRACTION — the `/B` decay-role "defect" was not a defect; the SGD memory-role finding is an artifact of its own harness (2026-10-05)

This retracts the claim first recorded in `4e4fdc4` and repeated in the
ROADMAP and in the "optimizer axis" section above: that the decay rows'
gradient was "exactly B× too large relative to every other parameter group".

**It was not.** The engine has a single `/B` in the whole backward path and it
is on the loss (`gpu_port.py:787`); `dWd`, `dW`, `dx` and the decay rows carry
none. The entire 4,481,793-coordinate gradient buffer is B× the mean gradient,
**uniformly**. There was never a decay-role asymmetry.

Consequence that matters more than the correction itself: the SGD arm's
`_gradfix` divided **only** the 16 `.decay` slices by B
(`scratch-optimizer-study/_derived_gpu_port_sgd.py:1107`), so the decay role
ran at an effective learning rate **1024× smaller** than every other parameter.
The result that followed — "σ moved 1e-5 over 30,000 SGD steps; the memory role
does not train under a magnitude-proportional optimizer; this is an
ARCHITECTURE question" — is an artifact of that correction. The arm measured
its own harness.

What stands, restated correctly:

- "Harmless under AdamW" (scale-invariance of `m/√v`) is real, but it is a
  property of every group, not of the decay role.
- SNR ~0.05 vs 0.633 frozen — unchanged and unrelated.
- The decay parameters still did not move in the ladder, but per the strategic
  review (`docs/reviews/STRATEGIC-REVIEW-2026-10-05.md`) that is because the
  corpus rewards ≈0.03 nats for any horizon beyond 8 B — not because an
  optimizer choice froze them.

The reviewer's n-gram measurement, which is the reason to read the rest of the
strategic review: an order-8 byte n-gram (0 parameters, 8 MB train) scores
**0.81 nats/byte** on TMT's held-out set against TMT's **1.2094** — the model
loses to a memorisation table on its own training distribution by 0.40 nats
using 0.9% of the data, and the CE curve is flat (12 B and 16 B are *worse*
than 8 B). The ruler cannot see memory; the experimental programme was aimed at
a bottleneck the corpus does not present.

---

## The memory ruler: the first direct measurement of retention (2026-10-12)

Every prior statement about memory in this log was inferred from a CE curve,
which cannot see memory. This is the first direct measurement, and it was taken
on the trained checkpoint `runs/lrroles/arm_lrdecay1x_t919897.npz` (t=919,897,
dim 512, 16 layers, ring_k 18) through a frozen numpy_port forward — no weight
updates, no optimizer, state reset per sample. The prior state-probe used a
fresh init, which is why its verdict was inconclusive; a fresh init has all
decays near 0.5 and the tail is inactive, so it was a ring-only measurement
dressed as an engine measurement.

**The checkpoint says, in its own weights, that it cannot remember.**
sigma_mean 0.5191, sigma_max 0.9847, mean half-life **1.16 B**, and only **8 of
512 dims** hold more than 8 B while **5** hold more than 16 B. The longest-lived
dimension in the model forgets in ~45 bytes.

**Retention curve** — matched-pair state-norm difference against its control
floor. Both sequences share the same distractors and differ only in a 4-byte key
at position 0; the control is the iid random-byte arm and the mismatched-cue
veto, which agree with each other at ~39–46.

| d | matched | control | ratio |
|---:|---:|---:|---:|
| 8 | 2.045 | 42.93 | 4.8% |
| 32 | 0.751 | 43.01 | 1.7% |
| 128 | 0.140 | 39.04 | 0.36% |
| 512 | 0.0003 | 43.80 | 7e-6 |

A clean geometric decay. At 128 B the retained signal is a third of a percent of
the noise floor.

**Retrieval tasks — the readout never beats its own control.** This is the
result that matters, because a readout can manufacture confident-looking
numbers. Selective copy: d=8 gives 0.0312 against a permuted-label veto of
0.0156 (se 0.0217, so ~1.4 sigma — within noise), and at d=512 readout and veto
are *identical* at 0.0156, i.e. the same artifact. Associative retrieval:
0.0000 at d=8, 128 and 512; at d=32 the readout is 0.0156 and its veto is
0.0156. Zero at every distance, with no arm where signal exceeds control.

**Long-range language CE 10.4767 nats/B against a chance level of 5.5248** on a
509-byte filler task — roughly twice *worse* than the generating distribution.
A model that had retained the marker would beat chance; this one is actively
miscalibrated at that range.

**The ruler reconciles with the incumbent on real text exactly.** Held-out CE
1.2094 +/- 0.0143 nats/byte (6 slices, 120 kB) — matching the standing 1.2094
figure to four decimals, which is the sanity check that this instrument is
measuring the same model as the rest of the log. Out-of-domain novels
2.6330 +/- 0.1538, range 2.048 (alice) to 3.378 (moby_dick), against unigram
3.063 and bigram 2.291.

**Verdict: the model does not retain information past ~64 bytes, and five
independent arms agree.** This was the question every prior measurement failed
to answer. The corpus does not reward it (the strategic review's ~0.03 nats for
any horizon beyond 8 B), and the weights agree with the corpus.

### The cause is the init, and it is a capacity limit rather than a training one

The incumbent decay init is `rng.uniform(-1/sqrt(dim), 1/sqrt(dim))`, so every
logit is within 0.044 of zero and every sigma within 1% of 0.5. Measured on the
actual engine, that puts **every half-life in [0.919, 1.094] bytes**. There is
no dimension in this model capable of holding 512 B, and no amount of training
builds one it was never given. This is what P4 of the T-axis plan addresses.

### ring_k caps at 256, and that caps P4 at ~32 B

`ring_k = min(256, ceil(ln(RING_TOL) / ln(sigma_max)))`. Measured across the
init: half_max 8 B gives ring_k 107, but half_max 32 B and above all return
**256, saturated at the hardcoded cap**.

So the plan's `--half-min 2 --half-max 512` buys nothing past ~32 B — a dim with
a 512 B half-life receives credit from only 256 bytes of history. This inverts
the plan's dependency order: P2 (the dense embedding trace, which deletes the
ring and its cap) is not an independent efficiency item, it is a **prerequisite
for P4's benefit**. The honest P4 A/B until then is `--half-max 32`, fully
served by ring_k 256; running `--half-max 512` would spend the full ring memory
for coverage the ring cannot deliver and any null result would be unattributable.

### Four bugs the ruler exposed, none caught by the gates

The gate suite never executes `tools/ruler.py`, so all of this shipped uncaught:

1. `_REPO` computed one directory too high — `import numpy_port` failed outright.
2. `_print_meta` indexed `n_gt_8B`, `sigma_mean`, `half_life_mean_B`, keys no
   metadata dict provides, crashing *after* the numbers were computed; `dim` and
   `layers` also reported 0.
3. `_frozen_forward` never sized state to its batch. `load_frozen` leaves
   `new_states(1)`, so batched calls died on `(64,512) vs (128,512)`. Fixing it
   with `m.new_states(B)` also implemented the documented per-call reset, and
   **the retention numbers dropped** (d=32 1.83 -> 0.75, d=128 0.34 -> 0.14)
   because the leak had been inflating the signal.
4. `_eval_corpus` returned `int(ces.sum())` under the name `total_bytes` — the
   sum of CE values, not a byte count. A 120 kB run reported "7 B". It was
   printed as `{total:,} B` and surfaced as `"total_bytes"`, so **every dose
   figure derived from it was wrong**.

Plus two in the ruler itself: `E._baselines` should have been `mb._baselines`
(same line already used `mb._eval_corpus`), and `sorted(glob("data/*.txt"))[:9]`
swept `data/sample.txt` — the training corpus — into the **out-of-domain** arm,
presenting an in-domain score as domain shift.

**Standing measurement-integrity note:** the retention numbers moved between
runs, and the direction was toward *less* signal. A measurement that only ever
moves toward the conclusion you wanted should be treated as suspect; here the
fix that reduced the signal was the one that implemented the documented state
reset, so the post-fix numbers are the correct reading and the reduction is
conservative rather than convenient.

---

## The CPU is usable, and the ring was never load-bearing (2026-10-12)

Two corrections to assumptions this log had been carrying.

### The CPU can run small experiments

Measured on the current engine, `numpy_port`, B=1024, dim=512, layers=16,
median of the last 10 of 12 steps:

| OPENBLAS_NUM_THREADS | ms/step | B/s |
|---:|---:|---:|
| 1 | 1,310 | 782 |
| **2** | **1,202** | **852** |
| 4 | 1,242 | 824 |
| unset (default) | 1,495 | 685 |

Default is genuinely the worst setting and 2 threads is the optimum, worth
**1.24x** end to end with no code change. The magnitude is smaller than a
loaded-box measurement suggested; the direction holds.

**sigmoid is ~20% of the CPU step** (cProfile, 5 steps, 10.7 s: step_batch 28%,
backward 24%, sigmoid 20%, forward 14%). It is called 16x forward and 16x
backward per step on `(B, dim)` arrays, and `numpy_port.sigmoid` built a boolean
mask plus two fancy-indexed writes per call. Replacing that with a branchless
`np.where` over the same two expressions:

    median 1,715 -> 1,402 ms/step = 1.22x, and BIT-IDENTICAL

verified over **131,072,000 elements** spanning N(0,1), N(0,8) and U(-120,120):
**0 differing bits**, max|delta| = 0.0, dtype preserved. The U(-120,120) arm is
what exercises the overflow branches, where the discarded branch produces
`inf/(1+inf) = nan` and is dropped by `np.where`.

A clip-based variant is faster still (5.0 ms vs 8.7 ms per call at (1024,512))
but is **not** bit-identical (CE delta 9e-6), so it is an algorithm migration
under this repo's exactness rules and was not shipped.

Both compose: 1.24x x 1.22x on the same path. **This makes small A/Bs feasible
here**, which the earlier "no GPU, so no experiment" reading got wrong — the
ruler worked because it is a frozen forward, but so does a short training run.

### The ring was an approximation, and the dense trace reproduces it exactly

The ring buffer and `tail_credit.py`'s private EMA were both approximations of
one object: an exact per-byte-id embedding eligibility trace. `tail_credit`
already materialised it for the dims where `sigma**ring_k` would drop more than
1% of the credit mass. So the dense `(B, 256, dim)` trace is *the tail with
every dim hot* — the approximation removed, not a new mechanism.

The plan's P2 gate compares them where both are computable:

| regime | max rel dev |
|---|---|
| dense vs TailCredit with all dims hot | **0.00e+00** |
| dense vs ring, `T < ring_k`, zeroed ring | **0.00e+00** |

Bit-level agreement, so the deletion is justified rather than hopeful. **The
ring and `tail_credit.py` are not yet deleted** — the dense path runs behind
`--trace-mode dense` so the incumbent and `check_parity` are untouched; deletion
is the remaining mechanical step.

Three consequences:

1. **P3's blocker dissolves.** The ring forced a per-dim `(dim,)` decay vector
   through `_scatter_ring_credit`, `TorchTail.begin_step` and `ring_k`
   inference. A dense trace has no per-dim addressing constraint, so a
   per-sample `a = sigmoid(decay + gate*x)` is no longer a problem for the
   credit path. Only the gradient remains, and it was already wrong.
2. **P4's cap disappears.** `ring_k = min(256, ...)` saturates at
   `half_max ~= 32 B`, so `--half-max 512` bought nothing. With no ring,
   `--half-max 512` runs end to end.
3. **The cost is real but bounded**: B*256*dim*4 per layer, i.e. 0.13 / 0.54 /
   2.15 GB at B=16/64/256 for dim=512 x 16 layers. All fit an 8 GB box. The
   trace is persisted at the same order of size the ring's `ring_w` already
   cost, and **cross-mode checkpoint loads refuse** rather than starting a cold
   trace — `sigma**50` is ~1e-15 for cold dims but **0.36** at sigma=0.98, so a
   zeroed trace would silently lose the history of exactly the dims it exists
   for.

**Method note.** The first draft of this gate compared the engine's hand-built
gradient against a second hand-written backward and disagreed by 12-40x, which
proved nothing — two hand-writtens sharing an omission agree perfectly. Rebuilding
the reference on torch autograd is what produced a trustworthy verdict. The same
lesson applies to the dense trace: it was checked against `TailCredit`, an
independent implementation, rather than against a restatement of itself.

---

## The T axis on a real GPU, and the P4 capacity probe (2026-10-12)

Three things this log did not have before: a GPU to measure on, an
instrument that separates "the init is the capacity limit" from "the init is
not", and a record of what happens when a gate suite that has only ever run
on a CPU is run on the device the arms will actually run on.

### The T axis is free at constant B — the plan's B reduction is what costs

`tools/seqlen_probe.py --device cuda` on an MX150 (4 GB, 142 GFLOP/s fp32,
sm_61), dim 512, layers 16, 20 timed windows per cell:

| cell (B×T) | B·T | µs/byte | vs (1024,1) |
|---|---:|---:|---:|
| 1024×1 | 1024 | 148.6 | 1.00× |
| 1024×2 | 2048 | 140.7 | 1.06× |
| 1024×4 | 4096 | 137.3 | 1.08× |
| **1024×16** | 16384 | **136.3** | **1.09×** |
| 256×16 | 4096 | 157.3 | 0.94× |
| 64×16 | 1024 | 551.5 | 0.27× |
| 16×64 | 1024 | 2153.9 | 0.07× |
| 8×128 | 1024 | 4277.1 | 0.03× |

**At constant B, T is not a cost — it is a 9% speedup.** The per-window
overhead (the optimizer, the gradient assembly) amortizes over the window's
bytes while the per-byte work is unchanged. A live training run agrees:
**7400 B/s at T=16 against 6116 B/s at T=1**, with the loss falling
62.6 → 40.7 and the CE 2.673 → 1.777 over 200 windows, no NaN.

The collapse in the lower half of the table is **not** the T axis: it tracks
the GEMM's row count, so at matched B·T the small-B cells are
launch/occupancy-bound. That is the opposite of the CPU reading, where the
same trade cost 1.15×, and it is why the T-axis plan's §5 falsifier — "if
(B=64,T=16) is more than 2× worse, the ring stays and P2 is abandoned" —
looked safe when it was decided on a CPU. On the GPU it fires at **3.7×**.

**What that changes.** The falsifier was written for the ring-deletion
decision and it also decided the T-axis decision, which it had no business
deciding. The two separate cleanly now: the **ring stays** (the B reduction
that would have made it deletable costs 3.7×) and **T is adopted at
B=1024** (where it is 1.09× *faster*). The plan's §1 bet — "lowering B in
exchange is what makes the whole ring/tail-credit apparatus deletable" — is
the part the measurement kills, and P1 survives it.

### The P4 init buys the capacity. Training gives it back.

The plan's causal story is "the cause is the init, and it is a capacity limit
rather than a training one". The capacity half had never been tested. It is
testable without training anything, because retention is a property of the
recurrence's dynamics: two sequences differing only in a 4-byte key at
position 0, the same d distractor bytes after it, and the state's norm
difference between the two.

`tools/capacity_probe.py` (new) measures exactly that on an untrained model;
`tests/check_p4_capacity.py` (new) gates it. At dim 512, layers 16, 64
samples, the full ruler:

| init | d=8 | d=32 | d=128 | d=512 |
|---|---:|---:|---:|---:|
| default (~1 B half-lives) | 0.40% | 0.00% | 0.00% | 0.00% |
| **P4 (2…512 B)** | **49.2%** | **21.0%** | **5.8%** | **1.05%** |
| TRAINED incumbent | 4.8% | 1.7% | 0.36% | 0.0007% |

(matched-pair signal ÷ random-byte control floor. The fast probe at dim 128,
layers 4, 32 samples reproduces it: 0.42%/0%/0% and 49.8%/22.4%/6.0%.)

**Reading 1 — the plan is right about the capacity.** A model *given* 2…512 B
half-lives holds a signal 512 B out; one with 1 B half-lives is dead by 32 B.
"There is no dimension in this model capable of holding 512 B, and no amount
of training builds one it was never given" is supported, and P4 is
**necessary**.

**Reading 2 — the plan is wrong about the sufficiency, and this is the
expensive one.** The TRAINED incumbent retains **less** than a fresh P4 model
at every distance: 4.8% vs 49.2% at d=8, 0.36% vs 5.8% at d=128. The
capacity is real and training gave it back. So P4 is a 2×2, not a step —
`(wd_decay 0.01 / 0.0) × (default init / P4 init)` — with P6 in the first
factor, because the weight decay on the decay logits pulls them back toward
`σ=0.5`, i.e. back to a 1-byte half-life, which is precisely the mechanism by
which the spread decays away (finding 7 says gradient descent undoes the
log-spaced init within an epoch; P6 says weight decay makes sure it cannot
grow back either). **Neither is fixed by the init alone.**

The instrument now costs seconds and no GPU, so it belongs in the gate suite
rather than in a training A/B: `tests/check_p4_capacity.py` gates the
capacity half; the training A/B gates the usage half.

### The gate suite had never run on a GPU

Running the suite on the box that will run the arms found four device bugs
and five hardcoded absolute paths, all of them invisible on a CPU box:

* `tools/seqlen_probe.py` built its input windows on the CPU and never moved
  them to the device — so **the plan's throughput measurement had never been
  taken on a GPU**, and the probe crashed on CUDA. Fixed.
* `tests/check_seqlen_credit.py` (the P1 gate) had the same bug, and had
  therefore never run on a GPU. Fixed, and it now takes `--device`: it
  **passes on CUDA with numbers identical to CPU** (worst embed-mass
  deviation 1.177e-07, worst elementwise 6.481e-06).
* `tests/check_parity.py` (P1's Gate 1) pinned its inputs to CPU while the
  engines took the device automatically, so it crashed on CUDA. Now pinned
  deliberately, with the reason recorded: bit-identity is a *CPU* claim
  (CUDA's atomic ordering makes it the wrong bar there), and the
  tolerance-based CUDA version of the same question belongs to
  `check_trace_bptt`.
* `tests/check_dense_trace.py`, `check_halflife_init.py` and
  `check_gated_recurrence.py` built their engines with `device=None`, so on a
  GPU box they silently became CUDA gates with CPU inputs — crashes, not
  tests. Pinned to CPU, with the convention written down.
* `tools/membench/model_backend.py` (×2) and three other tools hardcoded
  `/home/baku/programing/opensource/new-llm/test-model-thing`, so the ruler
  could not run anywhere else at all. Now derived from `__file__`.

`check_parity` passes on the box (numpy reference against torch-on-CUDA), and
so does the rest of the suite. The generalisable lesson: **a gate that has
only ever run on one device is a statement about that device.**

### The safety features, and what clipping actually does

`tests/check_safety.py` (new) gates the plan's last open P0 item and P6:

* A non-finite gradient is **REFUSED**: theta, m and v are left
  bit-identical, the refusal is counted and logged, `t` still advances (the
  data was consumed; rewinding the LR schedule would be a second, quieter
  way to change the run), and training continues.
* The guard is a **no-op on finite gradients, bit-exactly** — the same engine
  with the guard on and off produces identical theta, m and v over 12 steps.
  That is what makes it safe to enable by default.
* The global grad clip is **off by default** and its threshold is in
  *absolute* gradient units, which are **not comparable across T** (the
  window's gradient is a sum over T bytes), so it must be off for a T A/B.
* A measured correction to the assumption behind clipping: under AdamW the
  clip does **not** shrink the parameter step — the update is
  `m/√v ≈ sign(g)`, nearly scale-invariant in the gradient (max |Δθ| was
  2.080e-05 either way in the gate). What the clip bounds is the **moments**,
  and that is what matters: one enormous gradient inflates `v`, and every
  later small gradient is then divided by that inflated `√v` and suppressed
  for ~1/(1−β₂) = 1000 steps.
* P6's per-group weight decay: the wd vector is uniform by default
  (bit-identical to the incumbent's scalar `wd * th`) and exempts the 64
  decay rows at `wd_decay=0`; the update then matches a hand-written AdamW
  to **0.00e+00**. With the exemption, `--lr-decay` scales only the adaptive
  term — the "knob wired to a trap" is gone.

### The A/B that is running now

Three arms, all resumed from the **same trained incumbent** (t=919,897, B=1024,
ring_k=18, held-out CE 1.2094 ± 0.0143), each consuming the **same 8.39 MB**
of the same corpus in the same order, differing only in T:

| arm | windows | bytes | optimizer updates | within-window credit |
|---|---:|---:|---:|---:|
| T=1 | 8192 | 8.39 MB | 8192 | 0 bytes (the incumbent's 1-step regime) |
| T=4 | 2048 | 8.39 MB | 2048 | 4 bytes, exact |
| T=16 | 512 | 8.39 MB | 512 | 16 bytes, exact |

The verdict is the held-out CE from the ruler, with the retention curve
reported alongside, and the update count reported with it: at matched bytes
T=16 buys **16× fewer updates**, and the credit effect and the update-count
effect are not separable at one byte budget. That is the honest shape of the
experiment, and it is why the arms are T=1, 4, 16 rather than 1 and 16: the
monotone ordering across three points is what distinguishes "the package is
better" from "the credit effect is real".

---

## The T-axis A/B: T=1 wins at this scale, and the reason is the updates (2026-10-13)

The plan's P1 puts `seqlen` (T) first because the T axis is the prerequisite
for multi-token prediction, and because lowering B in exchange was supposed
to make the ring deletable. The GPU measurement killed the second half
(finding 9 above) and left the first. This is the quality half: three arms
from the same trained incumbent, all consuming the same 8.39 MB of the same
corpus in the same order, differing only in T.

**Family A — matched BYTES (8.39 MB each, resumed from the incumbent at
t=919,897, ring_k=18, held-out CE 1.2094 ± 0.0143):**

| arm | windows | bytes | optimizer updates | within-window credit | held-out CE |
|---|---:|---:|---:|---:|---:|
| incumbent | 0 | 0 | 0 | — | 1.2094 ± 0.0143 |
| **T=1** | 8192 | 8.39 MB | **8192** | 0 B | **1.1272 ± 0.0225** |
| T=4 | 2048 | 8.39 MB | 2048 | 4 B | 1.2210 ± 0.0211 |
| T=16 | 512 | 8.39 MB | 512 | 16 B | 1.1867 ± 0.0203 |

**Family B — matched UPDATES (512 each, resumed from the same incumbent):**

| arm | windows | bytes | updates | held-out CE |
|---|---:|---:|---:|---:|
| T=1 | 512 | 0.52 MB | 512 | 1.1377 ± 0.0180 |
| T=4 | 512 | 2.10 MB | 512 | 1.1350 ± 0.0170 |
| T=16 | 512 | 8.39 MB | 512 | 1.1867 ± 0.0203 |

**What this says.**

1. **T=1 wins, and by more than noise.** T=1 at 1.1272 is 0.059–0.094 nats
   better than T=4 and T=16 at matched bytes (2.6–3.8σ), and T=16 is also
   behind at matched updates (0.050, 2.5σ). The two families agree, and they
   were built to be biased in opposite directions (matched bytes gives T=16
   16× fewer updates → less overfitting → *for* T=16; matched updates gives
   it 16× more bytes of an already-seen corpus → more repetition → *against*
   T=16). A split verdict would have told us which effect dominates; both
   point the same way.
2. **The effect is the update count, not the credit.** At matched updates,
   T=1 (0.52 MB) and T=4 (2.10 MB) are indistinguishable (1.1377 vs 1.1350,
   0.15σ) — the extra bytes bought nothing at equal updates — while the arm
   with 16× fewer updates per byte is behind. The within-window credit is
   exact (tests/check_trace_bptt.py: 5e-3 within the window, verified on CPU
   and CUDA), and it does not pay for the updates it costs.
3. **The regime caveat, which is the honest headline.** The incumbent sits at
   ~5.7 epochs of its 165 MB corpus, and T=1's extra 0.05 epoch *improved*
   the held-out CE by 0.082 nats — so this model is still UNDERFITTING, not
   overfitting. In that regime, starving an arm of optimizer updates is
   simply a handicap, and the T axis pays that handicap for a credit effect
   that does not show up. The regime where T's update-starvation becomes an
   ADVANTAGE is the overfitting one — upstream measured 14 epochs > 24 epochs
   (#23) and 1.63 BPB > 1.82 BPB — and the incumbent is 2.4 epochs short of
   it (941 MB more, ≈ 3 days on the MX150, ≈ 1 day on a T4). **This A/B
   therefore does not rule the T axis out; it locates the regime where the
   question is worth asking.** That is the next experiment, not this one.
4. **The T=1 arms' own gain is mostly early.** T=1 at 512 updates already
   reads 1.1377, against 1.1272 at 8192 updates: the difference (0.0105) is
   inside the error bars, so 0.05 epoch of extra training bought ~0.08 nats
   and the first 0.03 epoch bought most of it.

**What this does NOT change.** The T axis is still free (1.09× per byte at
constant B), still the prerequisite for a *windowed* MTP, and its gradient is
still exact within the window. It just does not pay for itself as a way to
buy memory in a model that is still underfitting.

### The P4 init has the memory bill the plan priced for the dense trace

Found by an OOM while launching the P4 arms, and it belongs in the plan's
section 1 table:

`ring_k` is inferred from `sigma_max`, and the log-spaced init pushes
`sigma_max` to 0.9986, so **ring_k saturates at 256** and the ring's
per-layer buffer becomes `(B, 256, dim)` — **8.6 GB at B=1024**, 2.15 GB at
B=256. The P4 arm OOM'd on a 4 GB card at B=1024. The plan priced that bill
for the *dense* trace and did not notice that the **ring** costs the same
once `ring_k` saturates, so "delete the ring and lower B" was never going to
be free at `--half-max 512`.

And the dense path was paying it twice: `new_states` allocated the ring's
`(B, 256, dim)` buffer *and* the dense trace *and* the tail's `(B, 256, H)`
accumulator — 2.15 GB + 2.15 GB + 1.0 GB at B=256 — for buffers the dense
path never reads. All three are now allocated empty on that path. Measured
peak CUDA with the P4 init:

| B | ring | dense |
|---|---|---|
| 256 | 3.55 GB | **2.46 GB** (was OOM) |
| 512 | OOM | OOM |
| 1024 | OOM | OOM |

`check_dense_trace.py` and `check_parity.py` both still PASS (the dense
path's 0.00e+00 agreement with ring+tail is unchanged — the gate compares a
dense model against a ring+tail model, and it is the dense model that no
longer carries a dead buffer).

**This changes the P4 experiment, not just its budget.** With the ring, the
default init gets `ring_k=14` and the P4 init gets `ring_k=256` — so the 2×2
would have measured the *credit reach* as well as the init. The dense trace
has no cap, so at B=256 both arms get the same exact credit and the only
difference is the init and the weight decay. That is the 2×2.

---

## The P4 usage arms: the init buys the capacity, the weight decay destroys it, and training does not (2026-10-13)

The capacity probe (above) measured the CAPACITY half of P4 with no training.
This is the USAGE half: the first training run of a log-spaced-init model that
has ever happened, and the first measurement of whether the spread survives
the optimizer. Three arms, all FROM SCRATCH (a different init cannot resume an
incumbent), 8.39 MB each = 0.2 epoch of the 41 MB file, B=256, ring_k=14,
tail off, same seed, same file:

| arm | init | `wd_decay` | mean half-life | dims > 8 B | held-out CE |
|---|---|---:|---:|---:|---:|
| (reference) fresh P4, untrained | 2…512 B | — | 92.30 B | 6144 | — |
| **`armFS_p4_wd0`** | 2…512 B | **0.0** | **89.48 B** | **6103** | 1.8834 ± 0.0309 |
| `armFS_p4_wd001` | 2…512 B | 0.01 | 39.07 B | 5485 | 1.6907 ± 0.0333 |
| `armFS_default_wd001` (control) | default | 0.01 | 1.02 B | 0 | 1.4085 ± 0.0271 |

### The retention curve — the memory verdict

matched-pair signal ÷ random-byte control floor, the ruler's own instrument:

| model | d=8 | d=32 | d=128 | d=512 |
|---|---:|---:|---:|---:|
| fresh P4, untrained | 49.2% | 21.0% | 5.8% | 1.05% |
| **trained P4, `wd_decay=0`** | **36.6%** | **17.1%** | **5.5%** | **0.91%** |
| trained P4, `wd_decay=0.01` | 35.4% | 15.4% | 4.0% | 0.34% |
| trained default init (control) | 0.43% | 0.00% | 0.00% | 0.00% |
| TRAINED incumbent (t=919,897, 5.7 epochs) | 4.8% | 1.7% | 0.36% | 0.0007% |

**Three results, and they re-order the plan.**

1. **The bottleneck has moved.** A model trained with the P4 init retains a
   signal **128 bytes out** — 5.5% of the control floor, **15× the
   incumbent's 0.36%**, and against **0.00%** for the default-init control
   trained on the same bytes with the same seed. The incumbent's ruler
   verdict ("the model does not retain information past ~64 bytes") was a
   statement about the *incumbent's init*, and it is no longer the ceiling of
   what this architecture can do.
2. **P6 is the mechanism, and it is now measured.** The weight decay on the
   decay logits is what erodes the spread: without the exemption the mean
   half-life falls 92.30 → 39.07 B (−58%) in 0.2 epoch, and the d=128
   retention falls from 5.5% to 4.0% (−27%); with it, 92.30 → 89.48 B (−3%)
   and 5.8% → 5.5%. `sigma(0) = 0.5` is exactly a 1-byte half-life, so that
   weight decay is a systematic pull toward no memory — the plan's argument,
   confirmed by a 2×2 rather than by a mechanism sketch. **P4 and P6 are one
   change, not two.**
3. **Finding 7 does not reproduce here.** "Gradient descent undoes the
   log-spaced init within one epoch" was measured on TinyStories with TBPTT
   K=32 and 3 seeds. With the weight decay exempted, the spread is
   essentially intact after 8.39 MB (0.2 epoch): 89.48 B against a fresh
   92.30 B, and 5.5% against a fresh 5.8% at d=128. Whatever destroyed the
   spread in that experiment, the weight decay is a sufficient explanation
   for it here.

### What it does NOT fix

**The model now holds a long-memory signal and still cannot use it.** Every
copy/retrieval arm is at chance for every model in this log, including the
P4 arms: the readout never beats its own permuted-label veto (selective copy
and associative retrieval, d=8…512). And the long channels cost local CE:
the P4 arms read 1.8834 and 1.6907 nats/byte against the default-init
control's 1.4085 at the same byte budget. So:

* the **capacity** axis is now closed — the model has channels that can hold
  a 512 B signal, and they survive training;
* the **usage** axis is untouched, and the bottleneck has moved to the
  WRITE/READ path: a state that is a linear, order-losing EMA of raw byte
  embeddings can hold a signal's *magnitude* out to 128 B, but a linear
  readout of an order-losing sum cannot answer "what did I see 128 bytes
  ago". This is the plan's §5 fallback arriving on schedule — the remaining
  deficit is architectural, not a batching, credit or optimization problem —
  and it names two candidates: **P3's gate** (an input-dependent retention
  factor, i.e. the write side) and the **hierarchical wiring** (the CUDA
  fork's answer, where the state carries features rather than bytes, the read
  side).

**The honest one-line reading:** the plan's P4+P6 is done and measured, the
memory problem is no longer a capacity problem, and the next month is about
the write path and the readout — not about the init, not about the T axis,
and not about the ring.

---

## A paper, an audit, and the P3 gate gradient finally explained (2026-10-13)

### The paper: FlashLM (Chang, May 2026)

`~/Downloads/output.pdf` is **"FlashLM: CPU-Native Language Models Trained
From Scratch on Free-Tier Hardware"** — 8 pages, no figures, three tables, a
self-published preprint. Two notes are in `docs/papers/`:
`PAPER-NOTE.md` (a full transcription with every equation and every number
re-checked) and `RELEVANCE-NOTE.md` (the audit of it against this codebase,
with the cost table and the experiment).

Its thesis is **"the training objective matters more than the architecture"**,
and its lever is an auxiliary loss, **Future State Prediction (FSP)**: at every
r-th position (r=16) the model predicts *which tokens appear in the next τ
positions* (τ=64) as a binary multi-hot, with BCE (pos_weight 50), and
L = L_CE + 0.1·L_FSP. The claimed result is a 2.5× perplexity improvement
(25.08 → 10.24) for 1.7% extra parameters.

**The paper's own evidence is weaker than its abstract**, and the audit says so
plainly: it is a single unseeded run; its own Table 2 contains an
architecture-only arm (v7.4, Gated DeltaNet + SWA) that reaches **PPL 2.33** —
a 10.8× improvement, larger than FSP's 2.45× — so the table contradicts the
thesis printed two pages earlier; another architecture-only arm (v5.2,
Attention + RoPE, 10.56) is never discussed; the FSP model's coherence verdict
is "Partial"; three rows share an identical 2,750 tok/s across different model
sizes; and the FSP head's own shape ("one d×d projection … sharing the output
embedding matrix") is self-contradictory.

**And yet the idea transfers, because it is aimed at exactly our measured
gap.** A recurrent byte model has exactly one place where context can live: the
state. Next-byte CE asks the *output* (one step ahead) to be right and never
asks the *state* to be usable later. Our measurement is that the state now
*holds* a signal 128 B out (5.5% of the control floor, 15× the incumbent) and
*cannot use it* (every copy/retrieval probe at chance). FSP is the cheapest
objective that can put the state to work: its gradient lands on the state, and
from there the engine's existing credit machinery carries it to the embedding
(the ring) and to the decay logits (the decaytrace) — i.e. it reaches the
**memory**, not just a read-out.

### The implementation, and what the audit found in it

`gpu_port.py` gained an FSP path: a head on the last layer's state
(`fsp.W`, `fsp.b`, and its own LayerNorm `fsp.g`/`fsp.b0`), a BCE-with-logits
against a byte-bag multi-hot built by the trainer from the next τ bytes, and
`--fsp-alpha/--fsp-r/--fsp-tau/--fsp-pos-weight/--fsp-layer`. The head's
parameters exist in theta **only when alpha > 0**, so the default engine is
bit-identical (the gate proves it: same loss number, same gradient vector,
max |diff| = 0).

The relevance audit found five defects in it, and all five are fixed:

| | defect | fix |
|---|---|---|
| **D1** | **the arm was unmeasurable**: the ruler runs on the numpy reference, which did not know the FSP keys, so the checkpoint's length check raised | `numpy_port.Model` gained the FSP layout (same order as the torch engine) and `model_backend.load_frozen` detects it; the gate now writes a checkpoint and reads it back through the ruler's own loader |
| **D2** | the auxiliary reached **one of sixteen** layers' memories (the layers' states are independent recurrences) | documented as the first arm's known limitation, with the per-layer variant pre-specified as a second arm (16× the head's flops) |
| **D3** | the auxiliary moved the **last layer's LayerNorm gain**, i.e. it was a knob on the local read-out, and the arm's CE could move for a reason unrelated to memory | the head got **its own** normalisation parameters |
| **D4** | `pos_weight 50` is the paper's number and is mis-calibrated for bytes: our corpus's 64-byte future window holds **22.0 distinct symbols of 256 (8.58% dense)**, not 1.6% as in a 4096-token vocabulary | solving for the paper's own positive-mass fraction gives **8.4**; the runbook uses 8 |
| **D5** | the FSP term was never logged | the training line prints `fsp <value>` beside the CE |

`tests/check_fsp.py` pins the path against **finite differences**, with no
reference implementation to trust:

```
(1) the head's own gradient vs a central finite difference
    d/dfsp.W[0]   engine -9.521905e-04 vs finite diff -9.727478e-04   rel dev 2.11e-02
    d/dfsp.b[200] engine -2.670877e-02 vs finite diff -2.669334e-02   rel dev 5.78e-04
(2) the auxiliary reaches the DECAY logits (the memory), not just the head
    d/dlayer0.decay[2] engine +3.024431e-02 vs finite diff +3.026962e-02  rel dev 8.36e-04
    d/dlayer1.decay[2] engine +6.488688e-02 vs finite diff +6.485939e-02  rel dev 4.24e-04
(3) alpha=0 leaves the default engine bit-identical   (gradient max |diff| = 0.000e+00)
(4) the reported FSP term equals the BCE it claims to be
(5) the arm is measurable: the ruler's loader reads the checkpoint
```

Two things had to be discovered to write that gate, and both are traps worth
recording:

* **The engine's gradients are the gradient of the batch SUM, while the
  reported loss is the batch MEAN.** The `/B` in the loss expression is for
  reporting; the CE, the embedding and the decay rows all follow the sum
  convention (the repo's P0 gate documents the same asymmetry for the decay
  rows). A finite difference of the reported loss must therefore be scaled by
  B, and the FSP gradient must NOT carry a 1/B — otherwise a correct engine
  looks wrong by exactly B, which is what the first version of this check
  reported.
* **The finite-difference step has to be large.** The loss is float32, so its
  ULP near 13 is ~1e-6 and a central difference of eps multiplies that
  quantum by 1/(2·eps): at eps=1e-3 the finite difference of a 1e-3 gradient
  is pure cancellation noise, and two different coordinates come out EXACTLY
  equal — which is the tell. At eps=0.05 the smallest gradient checked is
  still resolved to ~1%.

### The P3 gate gradient: the mechanism is now known

The gate's own docstring said the gate gradient was "wrong as written" and
listed the missing pieces as "the cross-time chain (what `decaytrace` already
does)" and "the state gradient has to pass back through the layer-norm and the
z·sigmoid(z) gated linear per layer, not through a single decoder-level dx".
Both are now measured, and the first one turns out to be exactly right — with
a precise structure.

The engine's `dL/ds_t` is the **readout-only** part of the state gradient, so
the true derivative wrt the retention carries the chain from every later
state, and the gate needs the decaytrace's twin:

```
gatetrace_t = a_t * gatetrace_{t-1} + a_t(1 - a_t) * x_t * s_{t-1}
dL/dgate    = sum_t dL/ds_t * gatetrace_t        (with the READOUT-ONLY dL/ds_t)
```

The four-way truth table (dim=3, B=2, T=5, float64, against torch autograd on
the same recurrence) is what makes the structure unambiguous:

| formula | state gradient fed in | max rel dev vs autograd |
|---|---|---:|
| plain sum (**the engine today**) | readout-only | **0.3467×** ← wrong |
| gate trace | readout-only | **0.0000×** ← correct |
| plain sum | total | 0.0000× (correct, but the engine does not have the total) |
| gate trace | total | 0.2723× (double counts) |

The engine's plain sum is the **local** effect of a_t on s_t only, and the
chain factor **grows as the retention approaches 1** — which is exactly the
regime the log-spaced (P4) init creates. A gate that is wrong by a factor that
scales with the memory it is supposed to create is worse than no gate at all.

There is a second, independent defect in the same line: with a *gated* (i.e.
per-sample) retention, `dec` has shape (B, dim), so `dec[None, :]` is
(1, B, dim); multiplying it against (B, dim) broadcast-promotes to (1, B, dim)
and `.sum(dim=0)` then sums an axis of size 1 — i.e. it sums **nothing over the
batch**, and the "gradient" comes out (B, dim) instead of (dim,).

So the P3 gate is not blocked by a mystery: it needs (i) the gatetrace, (ii) a
correct batch reduction, and (iii) — still, and still validly — the per-sample
plumbing for the ring credit, the tail and `ring_k` inference. (i) and (ii) are
small; (iii) is the real rework, and it is unchanged by this discovery.

### The instrument the hypothesis needs: the FSP bag cell

The audit's **D7** is the sharpest objection to the FSP arm, and it is a
property of the *byte-level* translation rather than of the paper: the target
is the bag of the next 64 bytes, and that bag holds ~22 distinct symbols of
256 — and those symbols are largely the common ones. So a head that ignores
the state entirely and learns the unigram distribution already scores well.
**The question the arm therefore has to answer is not "is the FSP loss low"
but "does the head beat the priors"** — because a head that does not was
satisfied by the prior, and the arm's null result would say nothing about the
hypothesis.

So the ruler grew a cell (section 3b) that scores the model's **own** head on
the future bag and puts it next to two static baselines measured on the same
positions:

```
[fsp_bag] head(threshold) F1=...  head(top-22) F1=...
            global prior F1=...     local 4-gram F1=...
```

The metric is instance-level **set F1** (precision/recall over the multi-hot),
each predictor using its own decision rule: the head at its trained 0.5
threshold and at top-k, the priors at top-k with k = the mean bag size. On our
corpus the global prior reaches **F1 ≈ 0.82**, so the head has to clear that
bar to have learned anything contextual at all. A 200-step smoke model scores
0.17 — far below, exactly as it should.

Two calibration decisions came out of the same audit and are applied to the
queued arm:

* **α = 1.0, not the paper's 0.1.** The measured α-ladder (dim 64, 4 layers,
  B=32, 200 steps on real bytes, max|Δembed| against the α=0 arm) is
  3.7e-05 / 1.8e-04 / 7.3e-04 / 3.5e-03 at α = 0.1 / 0.5 / 2 / 10 — linear in
  α, and small enough at the paper's value that the arm would be a null for a
  reason that has nothing to do with the hypothesis. The byte-level scale of
  the two loss terms differs from the token-level one the paper tuned on, so
  α is re-set by measurement here, not copied. The paper's value stays
  available as a flag.
* **pos_weight = 8.5, not 50.** The bag density on our corpus is 8.55% dense
  (21.9 of 256 symbols); the paper's 50 is tuned for a 4096-token vocabulary
  where the same window is ~1.6% dense. Solving for the paper's own
  positive-mass fraction on *our* density gives 8.5.

**The first arm's known limitation (D2), pre-registered.** The auxiliary
reaches **one of the sixteen** layers' memories (the layers' states are
independent recurrences, so a head on the last layer trains that layer's
decay logits and the shared embedding's credit, and no other layer's decay).
A null result is therefore ambiguous between "the objective does not work" and
"one of sixteen memories got the job". The per-layer variant — 16× the head's
flops — is pre-specified as a **second** arm, not a variant of this one.

---

## 2026-10-21 — the speed investigation: the hypothesis was WRONG, and the correction is the useful part

**The question.** The engine's optimizer step is dominated by one irregular
memory pattern: the ring credit's atomic scatter. At the production shape
(B=1024, K=18, dim=512, 16 layers) it issues **150,994,944 fp32 atomic
element-adds per step** into a 131,072-address destination — 72 contending
adds per address on average, 3,012 on the busiest. The hypothesis was that
*contended atomics* are the wall, and that a fused or segmented credit path
could raise the training throughput.

**The measurement (Tesla T4, cc 7.5, the production shape).** The credit,
timed alone (`tools/bench_ring_credit.py`):

| path | ms/call | ms/step (16 layers) | atomics/layer | passes over the ring |
|---|---:|---:|---:|---:|
| `full` (portable reference) | 1.285 | 20.56 | 9,437,184 | 5 |
| `compiled` (Inductor — the CUDA default) | 0.671 | 10.74 | 9,437,184 | ~2 + scatter |
| `triton` | 1.024 | 16.38 | 9,437,184 | 2 + scatter |
| `nvrtc` (hand-written fused kernel) | **0.475** | **7.61** | 9,437,184 | 1 (read+write) |
| traffic floor at 320 GB/s | — | 1.89 read / 3.77 read+write | — | — |

All four paths agree **bit-for-bit** on the T4, and the `full` baseline's
20.56 ms/step matches the profiler's independent 21.3 ms/step for the
ring-credit phase — the harness and the profiler agree.

**The verdict: the wall is the BYTES, not the atomics.** The credit moves
604 MB/step of ring traffic (1,208 MB with the decay's write-back) — the
traffic floor is 1.89–3.77 ms/step, and the *best* path is 7.61 ms. The
atomics are not what is left over: the fused `nvrtc` path already moves the
ring exactly once and is 2.7× faster than the portable reference **while
issuing the same 151M atomics per step**. So the atomics are cheap, and
what costs is reading and writing the ring.

**Two corrections to what I believed going in.**

1. *The removed `segmented` candidate was never measured on the MX150.* Its
   0.81× (RESULTS, Board run 4) and 0.71× (Phase-0 board) are **both T4
   numbers**; the MX150 work is from 2026-10-03, a month after the candidate
   was removed. I had briefed the investigation on a false premise, and the
   agent corrected it in the record rather than answering the question as
   asked.
2. *The candidate did not lose because of the atomics either.* It traded
   9.44M atomics for a full extra round trip over the (B,K,dim) tensor
   (~113 MB/step): the traffic it **adds** cost more than the atomics it
   removed (0.38 ms vs 0.23 ms on the T4; 3.0 vs 1.8 ms on the MX150). The
   margin gets *worse* on the faster GPU — the same extra traffic buys less
   time while the atomics it removed were already cheap. The isolated
   micro-benchmark agrees: `index_add_` 0.320 ms vs sort+segment-sum
   0.661 ms. It also moved the float (state RMS 1.8e-03 against 1.3e-04 for
   the kept paths), because `segment_reduce` reassociates.

**What survives the analysis: the `grouped` credit.** The reduction axis of
the credit is the **byte id**, never the lag: the K lag terms of a (b,d)
land on K *different* byte ids (99.9% of lag pairs differ, measured), so
reducing the lag axis in registers is illegal — it sums contributions
destined for different bytes. The legal move is to group the rows by byte
id and fold each group **left, in increasing row order** — which is the
reference's own order (`index_add_` is bit-identical to numpy's walk,
measured, not assumed) — and write each address exactly once:

* **zero atomics** per layer (the other four paths issue 9,437,184);
* **no extra traffic**: one pass over the ring, read and written back, the
  same traffic the fused `nvrtc` path already moves;
* **the same float**, term for term.

Gated by `tests/check_ring_grouped.py` (new): bit-identical to `full`, to
numpy's own scatter, and to a 12-step engine run's theta/m/v — and the gate
also pins the two *illegal* moves (the lag reduction, the reassociation) so
they cannot be re-introduced by accident. `tests/check_parity.py`,
`tests/check_safety.py` and `tests/check_seqlen_credit.py` all pass, and the
default path (`auto` → `compiled` on CUDA) is untouched.

**Still open:** the `grouped` path's T4 wall-clock (the free tier refused a
second allocation), and the cost of the sort of 18,432 keys per byte.

**The bigger prizes are outside the credit.** The `masked_fill` that zeroes
the 1.4% hot columns of the ring is 5.1 ms/step, and the 1,072 kernel
launches per step cost 10.1 ms of *CPU* time — both larger than anything
the credit's atomics can save. The engine's launch overhead is the next
thing to look at.

**What this changes about the plan.** The T-axis and P4/FSP work is a
*quality* question and the throughput question was supposed to be a
separate axis. The measurement says the throughput axis is not where the
151M atomics are: the engine already moves the ring once, and the remaining
cost is bytes and launches. So a "make the credit faster" programme has a
small ceiling (the credit is 7.6 ms of a ~63 ms step at best), and the
"make the step cheaper" programme (fewer launches, a smaller ring) has a
large one.

**The decision that follows: the default credit path is now the fastest
bit-identical one.** The default was `compiled` (33.68 ms/step end to end on
a T4) while the already-shipped `nvrtc` measured 26.72 ms/step — **1.26× the
throughput, moving no float** (all four paths produced the identical loss
7.58001, and all five are bit-identical on the device with max|Δ| = 0.0).
`auto` now resolves to `nvrtc` on CUDA, and if the kernel cannot be compiled
the resolution degrades to `compiled` (cc ≥ 7.0) or the portable reference,
never straight to the slow path. This is not new code: the fused NVRTC
kernel was already shipped — it is the only fused option below cc 7.0, where
it measured 2.2–2.7× on the MX150 — it was simply not the default where
Inductor happened to be available.

**The `grouped` credit stays as an option, not a default.** It is the legal,
bit-identical way to remove every atomic, and it *wins* when the byte ids are
uniform (0.486 vs 0.559 ms/call) — but on a real byte histogram the
commonest byte owns 16.3% of the rows, and the legal grouping hands all 3,012
of them to one byte id, one block, one SM, while the atomic scatter spreads
every row over every SM. So it is **slower on real data** (0.693 vs 0.497
ms/call), and the reason is structural, not a tuning failure. The finding
that closes the investigation: **the contention was never the wall.**

**What the investigation says to do next.** The credit's floor is its
traffic (604 MB/step → 1.89 ms read, 3.77 ms read+write) and the best path
is at 7.95 ms/step of credit — so the credit is 24% of a ~63 ms step and
cannot be made much cheaper by fixing its atomics. The two larger prizes are
outside it: the `masked_fill` that zeroes the ring's 1.4% hot columns
(5.1 ms/step) and the **1,072 kernel launches per step** (10.1 ms of *CPU*
time — the launch overhead is larger than the whole credit's arithmetic).
A launch-count reduction is the next thing to measure.

---

## 2026-10-21 — C: the P4-vs-FSP A/B on a T4 (19,000 steps each) — FSP is a null, and the null has a *measured* cause

**The arms.** `A_p4_wd0_c` (the control: P4 init, `wd_decay 0`) vs
`A_p4_wd0_fsp_c` (the same, plus `--fsp-alpha 1.0 --fsp-r 16 --fsp-tau 64
--fsp-pos-weight 8.5 --fsp-layer -1`). B=1024, T=1, `ring_k=18`, `TMT_TAIL=0`,
seed 0, **from scratch**, 19,000 steps (19.5 MB) each — the salvaged control's
*exact* budget, so the two arms and the salvaged run are comparable. Colab T4,
corpus fetched on the box (so the held-out bytes differ from the local files:
the absolute CE below is Colab-vs-Colab only).

**The run survived the recycling.** The free tier refused allocations outright
for hours, and the runtimes that did come were recycled mid-run. The loop
(allocate → ship → fetch → *resume* both arms from the hub → wait → repeat)
worked because of the two pieces landed earlier: the checkpoint's **data
position** (a resume continues the corpus instead of replaying it) and
`--stop-at-t` (a resume finishes at exactly the same step, so the byte budget
cannot drift). Both arms reached t=19,000 and were pushed to the hub.

| metric | control | +FSP | Δ |
|---|---:|---:|---|
| held-out CE (Colab corpus) | 1.360 | 1.361 | **+0.001** |
| half-life mean | 93.97 B | 92.77 B | −1.2 B |
| retention d=8 | 38.3% | 37.5% | −0.8 |
| retention d=32 | 17.6% | 17.1% | −0.5 |
| retention d=128 | 5.3% | 5.0% | −0.3 |
| retention d=512 | 0.83% | 0.78% | −0.05 |
| selective copy / associative retrieval | chance | chance | — |
| **FSP bag F1 (head @ 0.5)** | — | **0.685** | vs the global prior **0.817** |
| **FSP bag F1 (head @ top-22)** | — | **0.792** | vs the global prior **0.817** |
| (local 4-gram prior, same positions) | — | 0.420 | — |

**The verdict: FSP changed nothing — and the D7 cell says exactly why.** The
head did **not** beat the static byte-frequency prior (0.685 and 0.792 against
0.817). The auxiliary loss was therefore satisfied by the prior: the objective
never asked the state for anything. This is the failure the cell was built to
catch, and it caught it.

**Three readings, in the order they constrain the next step.**

1. **The objective is saturated by a trivial solution, and this is a property
   of the byte-level port, not of the hypothesis.** "Predict the bag of the
   next 64 bytes" at byte level is mostly "predict which bytes are common":
   a 64-byte bag holds ~22 distinct symbols and they are largely the common
   ones. The paper's token-level setting does not have the same shape (a
   4096-symbol vocabulary at ~1.6% density). So this arm's null says *the
   auxiliary was free*, not *the state is useless*.
2. **The target is also beyond the state's reach, independently.** The
   retention curve says the state's usable horizon is short — 38% at 8 bytes,
   17% at 32, 5% at 128 — and the FSP target is a bag at **64 bytes** ahead.
   τ=64 was carried over from the paper's token setting, where one token is a
   word; here it is outside what the state can carry at all. A head at the
   prior is exactly what a state that carries nothing that far should produce.
3. **The pre-registered limitation (D2) is still in force** — only one of the
   sixteen layer memories received the direct gradient — so a null is
   ambiguous between "the objective was free" and "one of sixteen memories
   got the job". Reading (1) dominates: the head did not need the state.

**What the A/B does establish: the write path works, the read path does not.**
This is a reframe of where the bottleneck is, and it comes from three
independent measurements that agree:

* the **retention curve**: the matched cue's states are 2.6× closer together
  than the random control's at 8 bytes (matched Δh 23.4 vs control 61.1) —
  the cue **is** in the state;
* **every** copy/retrieval arm is at chance, in this pair and in all the
  earlier ones — the model cannot **read** the stored value out;
* the **FSP head** is a third, independent read-path failure: a head that
  reads the state *directly* and predicts the future bag gets the prior and
  nothing more.

So the failure is not that the state is empty; it is that nothing in the
architecture can extract what the state holds. That reorients the plan:

* the **read wiring** (how a layer's state reaches the output) has never been
  tested with an objective that is not saturated — the hierarchical
  feature-state wiring is the untested candidate;
* the **P3 gate** (an input-dependent *write*) is not mis-aimed — but its
  value is now understood differently: not "make the state hold the cue"
  (it does) but "make the write selective, so the read's signal-to-noise is
  good enough to decode".

**What an FSP v2 would have to change** (pre-registered before it is run):

* **τ inside the state's reach** — 8 or 16 bytes, not 64;
* **a prior-free score**: the head's F1 measured *on the symbols the prior
  gets wrong*, or against a prior conditioned the same way, so a head that
  learned the prior cannot look good;
* the **per-layer variant** (16× the head's flops) stays the second arm —
  but only after the objective stops being free.

**The other two salvaged snapshots, measured on the local corpus** (the
capacity axis's erosion curve, all at ~19 MB):

| arm | init | wd_decay | half-life mean | retention d=8 / 32 / 128 | held-out CE |
|---|---|---|---:|---|---:|
| default | default | 0.01 | 1.05 B | 1.1% / 0.10% / 0% | 1.3297 ± 0.0254 |
| P4 | P4 | 0.01 | 57.16 B | 37.0% / 16.3% / 4.6% | 1.6342 ± 0.0297 |
| P4 | P4 | 0 | 95.37 B | 37.5% / 17.1% / 5.3% | 1.7482 ± 0.0362 |

Weight decay erodes the P4 spread roughly in half (95 B → 57 B of half-life)
and costs CE, but it does not remove the capacity: the retention curve is
barely moved. The default init has no capacity at any budget.

---

## 2026-10-21 — the audit of the read path, the architecture, and the plan

Three findings from an audit of the read path and the model's structure. The
first two are **defects in instruments**, and the third is a **structural fact
about the architecture** that the previous plan was working around without
knowing it. Together they change what the next experiment should be.

### 1. D1 — the copy/retrieval readout was broken, so the read-path null was an artifact

`tools/membench/model_backend.py:_ridge_score` had a **swapped tuple
assignment** in its lambda selection:

```python
if acc > best_acc:
    best_acc, best = lam, acc     # <- SWAPPED
```

`best_acc` received the *lambda* and `best` received the *accuracy*, so
(a) only `lam=1.0` was ever eligible — after the first iteration
`best_acc` is 1.0 and `acc > 1.0` is never true, so λ=1e3 and λ=1e5 were dead
code; and (b) `chosen = best` was a **training accuracy**, so the final fit
solved `G + 0·I` — an *unregularized* solve of a rank-deficient Gram matrix
over 8,192 features.

**Verified by calling the project's own function**: it returns `lam = 0.0`.
The stored ruler output is the fingerprint — `runs/taxis/ruler_A_p4_wd0_c.txt`
prints `readout exact=0.0156 (se 0.0155) lam=0`, and the intended code
cannot print that (`chosen ∈ {1.0, 1e3, 1e5}`).

A second defect in the same cell: `y = targets.astype(np.int64).ravel()`
flattened the `(n, 4)` payload targets, and the split then indexed only
`idx ∈ [0, n)`, so **only payload byte 0 of each sample was ever decoded** —
while the metric was published as *"fraction of exact 4-byte payload
matches"* against a chance level of **2⁻³²** (the true chance for byte 0 is
1/256).

**Fixed** (same file): the assignment is corrected, the lambda is now chosen
on an **inner hold-out carved out of the training half** (the old code chose
it on the *test* half), the k-byte payload is fit with **k independent
readouts, one per byte position** (a single readout over shared features can
only emit one byte value per sample, so it cannot decode a multi-byte payload
at all), and the permuted-label veto now permutes **samples**, not bytes.

**The consequence for the record:** the claim *"the readout never beats its
own control"* — which the 10-11 plan's pivot to the readout rested on — was
**produced by a broken probe and is not established**. This is the fourth
instance of the same failure mode in this repo (the harness that never ran;
the contention measured 256× too high; the FSP arm that was unmeasurable;
now the readout). `tests/check_ridge_readout.py` is the new gate: it fails on
the pre-fix code and passes on the fixed code.

### 2. D3 — the copy probe is class-coverage limited, so its "chance" is not the right bar

The readout is a **256-way one-hot ridge**, so it can only ever *emit* a byte
value that occurred in its training half. The copy task's payloads are
**random**, so with `n` samples the training half contains only
≈ `1 − exp(−n_train/256)` of the 256 byte values. At the ruler's
`sample_count=128` that is **≈ 22 %** of the alphabet, so the probe's
per-byte accuracy is capped near 22 % *whatever the state contains*, and the
4-byte exact rate near `0.22⁴ ≈ 0.2 %` — against a stated chance of 2⁻³².
The published metric and its chance level are not on the same scale, and a
coverage-limited probe cannot distinguish "the state is empty" from "the
probe cannot name the answer".

**Consequence:** a verdict from a coverage-limited probe is `UNMEASURED`, not
a null. The re-measurement must report the **class coverage** next to every
accuracy, and must be powered (thousands of samples, or a matched alphabet).

### 3. The architecture, as the code actually is

From `gpu_port.py:228-256` and `gpu_port.py:1701-1707`:

```python
state = states * a + enc          # the ONLY thing that enters the state
...
enc = self.embed[c_arr]           # ONE embedding
x = enc
for l in self.layers: x, state, decay, cache = l.forward(enc, x, l.states)
```

* The **residual stream never enters the state**; the only inputs to the
  memory are the byte embedding and the decay.
* There is **no cross-layer state flow**: layer L's memory is not a function
  of layer L−1's memory.
* Therefore **the 16 layers are 16 nonlinear readouts of ONE 512-dimensional
  multi-scale EMA bank**, accumulated into a residual stream. They are not a
  feature hierarchy, and "16 layers" does not mean 16 memories.
* The **8,192 decay channels are 16 permutations of the same 512
  time-scales** (the P4 init builds `np.geomspace(hmin, hmax, 512)` inside
  the layer loop and reshuffles it per layer). The model's distinct memory
  time-scales number **512, not 8,192** — and under the default init every
  half-life is ≈1 byte, i.e. no multi-scale at all.

**Parameter budget** (`numpy_port.Model(dim=512, layers=16)` →
`theta.size = 4,481,793`, verified live):

| group | params | % |
|---|---:|---:|
| per-layer `W` (512×512) × 16 | 4,194,304 | **93.585** |
| per-layer `g`+`b` × 16 | 16,384 | 0.366 |
| per-layer **`decay` × 16** | **8,192** | **0.183** |
| `embed` (256×512) | 131,072 | 2.925 |
| decoder `Wd`+`bd` | 131,328 | 2.931 |
| `Ws`+`bs` | 513 | 0.011 |

**The structural limit.** `state_j(t) = Σ_k a_j^k · emb(byte_{t−k})_j` is a
*weighted average* of one embedding coordinate over time. It can represent
"there has been a lot of `e` recently"; it cannot represent "the value that
followed this key". A linear readout over a sum of EMAs cannot isolate one
term without a **key-conditioned read** — which is the 10-11 plan's own §7.3
item 8 and `STRATEGIC-REVIEW §3(iii)`, and it is why the read path fails:
**not** a shortage of capacity, and **not** a shortage of layers.

**Consequence for the size question.** 94.1 % of the parameters sit in the
readout stack — precisely the surface a test-time update overwrites, i.e. the
catastrophic-forgetting surface — while the entire control surface of the
memory is 0.18 %. Capacity is not the binding constraint (data-limited, not
capacity-limited; the CUDA fork's 34M run still shows no retrieval past
64 B). The plan's decision: **the core is trained once and then frozen; new
knowledge lives in experts, and an expert never copies the depth** — a
decay-set expert costs 8,192 params (0.18 %), a rank-r readout LoRA costs
1,024·r per layer, and a full depth copy costs 4,218,880 (which is why the
T-RTU design was deferred).

### 4. The plan

`docs/plans/PLAN-2026-10-21-CONTINUAL-LEARNING.md` is now the single current
plan. It supersedes the *ordering* of the 10-04 roadmap, the 10-06 memory
benchmarks and the 10-11 T-axis plan; their evidence and their closed axes
are retained. `docs/INDEX.md` was corrected: it had marked nine documents
"current — read before new experiments", omitted five live ones, and pointed
at two closed axes.

**The goal, and its ruler:** *continual learning*, measured as three
numbers — **ACQUIRE** (recall new facts), **RETAIN** (keep the old ones),
**NO REGRESSION** (general held-out CE does not degrade) — with thresholds
fixed in advance, a permuted-label veto on every cell, and a
whole-programme kill criterion that the repo has never had.

**Also recorded here, for the record:** the n-gram script that produced the
`0.81` figure (`lag5.py`) was never committed and no longer exists, so that
number is currently unreproducible; and the original comparison was not
byte-matched to the ruler's held-out set (the n-gram was scored on a 300 kB
head of `tinystories_valid.txt`; the model is scored on 6 × 20,001-byte
slices across both files in `data/eval/`). Both are fixed in the plan's
CL-0 step.

### 5. The read path, re-measured with a working instrument — and the verdict changes

The fixed `_ridge_score` was re-run against the C A/B's P4 checkpoint
(`runs/taxis/A_p4_wd0_c.npz`, frozen numpy_port, seed 0). **The `lam=0`
fingerprint is gone**: every cell now reports a real λ from {1, 1e3, 1e5}.
The old instrument's `exact=0.0156` excursions (1/64 — a single test sample)
were the unregularized-solve artefact, and they are gone with it.

**The instrument has power.** The same ridge over the same states decodes the
**trailing key** (the four bytes the sequence ends with) at **41.7 %/byte**
(d=8) and **21.4 %/byte** (d=32) against a ~0.5 % veto. So the readout, the
split, and the regularization all work; the state does hold recent bytes.

**Class coverage is NOT the binding constraint** (D3 refuted). At n=2048 the
training half covers **100 %** of the 256 byte values and 100 % of the
4-byte payloads, and the payload readout still reaches only **2.3 %/byte**.
Coverage is ~60 % at n=128 and the readout is ~0; coverage is 100 % at
n=2048 and the readout is 2.3 %. Coverage does not set the level in either
direction. **The state is the limit, and the instrument is not the problem.**

**The payload is decodable, but only for about a dozen bytes.** The
padding-free fixed-gap curve at n=2048 (matched null in brackets):

| gap (distance) | payload, per byte | null | trailing key, per byte |
|---|---|---|---|
| 8 (12 B) | **12.4 %** | 0.53 % | 41 % |
| 32 (36 B) | 1.69 % | 0.48 % | 27 % |
| 128 (132 B) | 0.66 % | 0.44 % | 25 % |
| 512 (516 B) | 0.56 % | 0.42 % | 25 % |

**A matched alphabet makes the metric usable**: at d=32, n=2048, a
16-symbol payload alphabet gives **21.3 %/byte** (null 6.2 %) and a 4-symbol
alphabet **44.9 %/byte** (null 24.7 %) with `exact` = 0.0293 (30/1024
samples). With the full 256-symbol alphabet the `exact`-4-bytes metric is
hopeless at any feasible n, because `per_byte^4 ≈ 3e-7` — **the exact-4-byte
metric is the wrong instrument for a 256-symbol payload and cannot report
this trace.**

**D4 — the copy cell read the state AFTER the padding.** `model_copy_readout`
pads the variable-length batch with **zero bytes** to the longest sample and
then reads `final_states`, which `_frozen_forward` takes at `t == L-1` — so
at d≥32 the short-gap samples are read after 24–504 padding bytes, not at the
cue. The d=8 cell is the only padding-free one. The fixed-gap curve above is
the padding-free re-measurement; the ruler's own d≥32 cells are not.

**What this changes.**

1. **"The read path does not work" is false.** It works, over a horizon of
   about a dozen bytes: 12.4 %/byte at 12 B of lag, 23× its own null, while
   the trailing key sits at 41 %. The claim was an artifact of the broken
   probe.
2. **"The write path works" is also too strong.** The state does not hold a
   *decodable* payload past ~32 B, and the trailing key is at 25–41 % even at
   516 B of lag — so what decays is the **payload's addressability**, not the
   state's ability to hold recent bytes.
3. **The sharpest open question in the programme is now a measurement
   paradox**: the P4+P6 init gives a **95.4 B half-life** and the Δh
   retention curve reports **37.5 % at 8 B and 17 % at 32 B**, yet the
   *decodable* payload horizon is **~12 B**. Capacity (Δh) and usable content
   (decodable payload) diverge by more than an order of magnitude. Which of
   the two is lying is the next experiment, and it is a question about the
   write, not about the readout.
4. **The ruler's read cells must be re-specified** before any of them is
   quoted again: read the state **at the cue** (not after padding), report
   **per-byte** (not exact-4-bytes) at 256 symbols, and use a **matched
   alphabet** when an `exact` rate is wanted. Every read-path number in this
   file before this section was produced by an instrument that was broken in
   at least one of those three ways.

### 6. D2 — the n-gram baseline is rebuilt, byte-matched, and the claim survives

`tools/ngram_ce.py` is the rebuild of the lost `lag5.py`. The original
script was never committed, so the `0.81` figure was quoted but not
reproducible, and it was not byte-matched to the ruler's held-out set.

**Protocol (now in the script and in `--help`).** Train: the first 8,000,000
bytes of `data/eval/wiki_tinystories_005.txt`. Held-out, byte-matched: for
each file matched by `data/eval/**/*` (two files), the 3 disjoint
20,001-byte windows that `tools/eval.py:_slices` uses, each scored as the
mean of -log P(byte[t+1] | byte[t]) over its 20,000 positions, the file's
score the unweighted mean of its 3 windows, and the final score the
unweighted mean of the per-file scores (the ruler's own aggregation,
`tools/eval.py:358`). 120,000 scored positions.

**Model:** Jelinek-Mercer interpolation with a geometrically falling
order-dependent weight, `lambda_k = 1 - r**(k-1)`. Counting is exact: an
n-gram is keyed by its bytes packed big-endian into a uint64, which is
lossless for orders ≤ 8, so there is no rolling hash and no aliasing.
`--selftest` checks the keying against a pure-Python dict of byte tuples
(order by order, exact agreement).

**The result, byte-matched, on the same bytes the model is scored on:**

| | CE (nats/byte) |
|---|---:|
| **order-8 byte n-gram, 0 parameters, 8 MB** (r=0.9) | **0.6512** |
| — per file: `tinystories_valid.txt` | 0.6617 |
| — per file: `wiki_tinystories_005.txt` | 0.6407 |
| the TMT model, same held-out set | **1.2094** |
| **the gap** | **0.5582 nats/byte in the n-gram's favour** (×1.74 in perplexity) |

**The result is insensitive to the interpolation hyperparameter.** The whole
sweep (byte-matched aggregate):

| r | 0.5 | 0.6 | 0.7 | 0.8 | **0.9** | 0.95 | 0.99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE | 0.9037 | 0.8315 | 0.7458 | 0.6582 | **0.6512** | 0.8482 | 1.7245 |

Absolute discounting is worse (0.9088 / 0.9292 / 0.9584 at D = 0.5 / 0.75 /
0.9). Every setting except the pathological λ→1 (r=0.99) beats the model.

**And the original figure is consistent with this.** On the original
protocol (the 300 kB head of `tinystories_valid.txt`, one segment) the same
model scores **0.6190** at r=0.9 and 0.7035 at r=0.7 — the quoted **0.81**
sits inside the sweep's range, so the original number is reproducible in
magnitude even though the script that made it is gone.

**What this does and does not say.** It says that on this corpus, at this
scale, a **zero-parameter lookup table with 8 bytes of context** is half a
nat per byte better than a **4.5-million-parameter trained recurrent
model**. It does **not** say the recurrent state is useless — the retention
and decode measurements above are about a *different* property (carrying a
specific payload across a gap) and they are not contradicted. But it does
mean that **the CE ruler cannot be the headline**, and it sets the bar the
model has to clear before any memory claim is worth making: the model's CE
must first approach the n-gram's.

**Caveats recorded, not hidden.** (a) The original script is documented as
MAXORD=12; this rebuild supports orders 1–8 exactly (an order-9+ gram does
not fit in a uint64), so it is a **weaker** baseline than the original — if
it still beats the model, the original claim survives. (b) The n-gram's
training file is also one of the model's scored eval files; the scored
slices are at strides of ~13.8 MB and ~27.5 MB, far past the 8 MB training
head, so there is no train/eval overlap, but the corpus is shared.
(c) The r=0.99 row shows the model is not robust to a λ schedule that
abandons back-off; the reported optimum (r≈0.85–0.9) is a real optimum,
not a monotone drift.

### 7. D5 — the measurement paradox is resolved: the state DOES hold an addressable payload

`tools/d5_probe.py` (new) re-measures the read path on the C A/B's P4
checkpoint with every instrument requirement from §5 applied: the state is
read **at the cue** (the D4 padding bug avoided, via the fixed-gap
generator, so every sequence is exactly 12+d bytes and no zero padding
exists), the readout is the **fixed** `_ridge_score`, the payload uses a
**matched 16-symbol alphabet** with a **matched null**, n=2048 (full class
coverage: 100 % of the 256 byte values and 100 % of the payload values are
in the training half), and every cell carries its own permuted-label veto
and a nuisance control (the payload region replaced by fresh random bytes
of the same alphabet).

**The result (n=2048, alphabet 16, per-byte accuracy and its own matched
null):**

| gap | payload | null | ×null | **cue key** | null | ×null |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 28.03 % | 6.05 % | **4.63×** | 47.95 % | 0.29 % | **165×** |
| 12 | 29.49 % | 6.42 % | **4.59×** | 43.21 % | 0.44 % | 98× |
| 32 | 22.09 % | 6.64 % | **3.33×** | 28.64 % | 0.32 % | 90× |
| 64 | 15.89 % | 5.66 % | 2.81× | 27.12 % | 0.42 % | 65× |
| 128 | 10.16 % | 6.67 % | 1.52× | 26.88 % | 0.32 % | 84× |

The nuisance control (the payload absent from the state) sits at its own
chance at every gap — so the payload signal is not an artefact of the
readout.

**The paradox dissolves.** The capacity ruler (Δh retention) and the
decodable payload now agree in shape:

| distance | Δh retention | decodable payload **above its own null** |
|---:|---:|---:|
| 8 B | 37.5 % | 21.98 % |
| 32 B | 17 % | 15.45 % |
| 128 B | 5.3 % | 3.49 % |

They are the same curve. The earlier "the readout never beats its own
control" and the "12-byte horizon" were both artifacts of instruments that
were broken in three separate ways at once (the λ swap, the byte-0 ravel,
and the post-padding read), each of which pushed the measured payload
toward its own null.

**The new picture, which is stronger than the old one:**
- **the ADDRESS is strongly present and persistent.** The cue key decodes
  **65–165× over its own null** and holds ~27 % even at 128 B of lag. The
  state *knows which key it is being asked about*, and keeps knowing it.
- **the CONTENT is present and decays.** The payload decodes 4.6× over null
  at 8–12 B, falling to 1.5× at 128 B. So the payload is genuinely in the
  state, and what decays with distance is the payload's *separability*, not
  its presence.
- **this is the "address vs content" split the plan asked for**, and it
  says the write path is *not* the problem in the way the old framing
  assumed: the state holds an addressable, persistent address and a
  decaying but real content.

**What this does NOT say.** A 4.6× over-null per-byte rate is not a model
that can *use* the memory: 28 %/byte on a 4-byte payload is 0.6 % of
payloads fully recovered, against a chance of 2⁻³² — better, but the
readout is still far from reading. And the fact that the ADDRESS survives
where the CONTENT decays is exactly what an order-losing weighted average
would do: the most recent key dominates the state, so the key is easy to
read, while the older payload is averaged into a mixture and is only
partially separable.

**So D5's answer is (a) and (c) together, and not (b).** The write is
partly order-losing (the content is mixed with the address and with the
distractors), and the Δh ruler overstated the *usable* content even though
it did not overstate the *capacity*. The readout is **not** key-blind —
the global readout already recovers the payload at 4.6× null, so a per-key
readout is not the missing piece. **The next lever is the write's
selectivity, not the read's wiring** — which is a different conclusion from
the one the 10-11 plan reached, and it points at the P3 gate rather than at
the hierarchical feature-state wiring.

Runtime: 1,558.9 s total on CPU (the forward is a numpy loop over
timesteps x 16 layers; the ridge is 45–109 s per gap).

### 8. The speed question, re-examined: the CUDA graph is NOT the prize

A feasibility assessment of CUDA-graphing the step (2026-10-21) reached a
conclusion that contradicts the standing assumption, so it is recorded here
in full.

**The premise mixed two arms.** The 26.72 ms/step figure is the **`nvrtc`**
arm. The "1,072 launches = 10.1 ms of CPU" and the "`masked_fill` = 5.1
ms/step" figures are from the **OLD `compiled` arm** (a 51 ms step). The
`masked_fill` does **not** exist in the default arm: the NVRTC kernel
applies the hot-column mask inline (`gpu_port.py:383`). "Fuse the
masked_fill into the credit kernel" is **already shipped**; it only exists
in the `compiled` and `full` paths.

**The 10.1 ms of launch time is already overlapped, not on the critical
path.** The bench synchronises once after the loop
(`tools/bench_step_paths.py:49-55`), so 26.72 ms = max(CPU, GPU). The
device work alone is >20 ms, i.e. 2–3× the CPU launch time, so the wall is
GPU-bound and the CPU time is fully hidden. The per-step syncs do not put
the CPU on the path either: the CPU emits launches every ~9.4 µs while the
GPU consumes them every ~24.9 µs, so the GPU never idles.

**A CUDA graph would therefore buy ~0.5–1.5 ms/step (2–6 %)**, not
10–15 %: it can only recover the *GPU-side dispatch* of 1,072 kernels.
`RESULTS.md:999` ("CUDA graphs would have bought nothing") was the correct
conclusion; the repo's 10–15 % estimate in
`docs/optimization/OPTIMIZATION_TEST_STATUS.md:123` treats CPU time as wall
time and is wrong.

**What is capturable, if anything.** The largest capturable subgraph is
**two graphs**: (A) forward + backward + credit + gradient assembly
(`gpu_port.py:1701`→`2049`), and (B) the AdamW update
(`:2093-2108`). Outside them: the finiteness read (it is a *host decision*
between A and B), the scalar updates, the telemetry, and the tail rebase.
It needs six changes: a device-side ring write slot (today `w[:,p]=1.0`
bakes a host index into the kernel), fixed input buffers, an async loss, a
device-side bias-correction scalar, a side-stream warmup for the cuBLAS
workspace, and an in-place tail-mask update at the rebase. The unsafe ops
are `float(loss_tensor)` mid-step (`:1810-1814`, **on by default**), the
nan-guard `.item()` (`:2078`), the grad-clip `.item()` + host branch
(`:2089`), and the host bias-correction `b1**t` (`:2106`).

**The cheaper wins, ranked (none needs a graph):**
1. **Fused AdamW as one NVRTC kernel** — the optimizer region is 3.0 ms
   and currently makes ~25–30 array passes; a fused kernel makes 7.
   **~1.5–2.2 ms (6–8 %)**, and bit-identical if FMA contraction is off.
2. **Fused tail decay+credit** — 3 element-passes become 2; the tail is
   10.2 ms, so **~3–4 ms (12–15 %)**. It reassociates the einsum, so it
   must land in `numpy_port` too or the parity gate breaks.
3. `--async-metrics` plus taking the nan-guard off the per-step path —
   small alone (~0–0.5 ms) but the **prerequisite for any graph**; and
   folding `w[:,p]=1.0` into the NVRTC kernel (~0.2 ms).

**`torch.optim.AdamW(fused=True, capturable=True)` is NOT applicable.** The
engine hand-builds AdamW on flat 4,481,793-fp32 vectors
(`gpu_port.py:1052-1057`, `:2062-2110`) with per-coordinate LR and weight
decay vectors. `torch.optim` cannot express per-coordinate LR/WD without
splitting the flat layout, which would change the checkpoint format, and
the arithmetic differs anyway (eps added after bias correction and inside
the role-LR bracket; decay scaled by the role LR). The engine's **own**
fused/capturable AdamW is the one piece that would pay twice — once for
the speed and once as the enabler of a graph.

**The decisive measurement, if this is ever pursued:** `tools/bench_variants.py
profile_baseline` (device time sum vs wall) plus a no-sync CPU-time probe
plus an `--async-metrics` A/B. If the device time is already >95 % of the
wall, the graph is worth nothing and the three wins above are the work.

**This also revises the earlier claim in this file** that the biggest
remaining prizes were the `masked_fill` (5.1 ms) and the launches
(10.1 ms). Both are artefacts of the superseded `compiled` arm. The real
remaining prizes are the **traffic** (the credit's floor is 3.77 ms/step
against 7.6–8.0 ms measured) and the **optimizer** (3.0 ms in ~25–30 array
passes).
