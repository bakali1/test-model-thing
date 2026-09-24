# Optimization Test Status — what is proven, what is not

Written 2026-09-24 after the Phase-0/1 benchmark cycle. Purpose: so the next
iteration starts from measurements rather than from the plan's predictions —
several of which were wrong.

**How to reproduce any number here:**
```bash
.venv/bin/python make_bench_bundle.py
colab run --gpu T4 --timeout 1800 colab_bench.py
```
6–8 minutes: 200 timed steps, real TinyStories text, the live HF checkpoint.

---

## 1. What ships by default

| setting | value | why |
|---|---|---|
| `ring_credit` | `auto` → `compiled` on CUDA, `full` on CPU | measured **+26.5%** (27,560 vs 21,790 B/s) |
| `grad_assembly` | `cat` (one fused gradient-buffer write) | bit-identical; +3–6% |
| math | legacy numpy-identical, fp32, TF32 off | trajectory fidelity to the live run |

The default path was verified **bit-identical** to the pre-optimization engine
(`max|Δθ| = 0.0` over 150 steps on CPU), so nothing about the model's
mathematics changed.

---

## 2. TESTED — measured, with numbers

| change | result | evidence |
|---|---|---|
| Inductor-fused ring credit (`compiled`) | **+26.5%**, 27,560 B/s | board run 4 + final run |
| Triton fused ring kernel (`triton`) | +12%, 24,505 B/s | final board run |
| Fused gradient-buffer assembly (`cat`) | +3–6%, bit-identical | board runs 3–4 |
| Compiled path across a tail rebase | **no recompile stall** (28,696 B/s in the 500–600 window) | `rebase-check`, 620 steps |
| 200-step real-data benchmark | noise floor **~1%** (vs ~3–4% at 12 steps) | two independent runs |
| Real-text EOS density | **0.63%** (synthetic bench assumed 2%) | `real_bytes_for` |

Correctness gates, all green, all free (CPU):
`check_gradients`, `check_ring_k`, `check_parity`, `check_trainer`,
`check_variants`, `test_tail`, `check_fetch`.

`check_variants` now gates the three surviving ring paths against the `full`
reference at 1e-4 (measured: 3.0e-07 for `compiled`).

---

## 3. NOT TESTED — explicit gaps

These are open. Do not assume any of them works.

| gap | why it matters | suggested next step |
|---|---|---|
| **B=4096 with the compiled path** | all measurements are B=1024; compile specializes on shape, so a new B triggers recompilation | board run at `--B 4096` |
| **A long leg (>2k steps) on the compiled path** | the rebase check covers 620 steps; a real leg is 30k+ | run `colab_leg.py` |
| **Multi-file / file-transition training** | every benchmark used a single in-memory corpus slice; the trainer loops over files | leg with ≥2 corpus files (needs >40k steps at B=1024) |
| **`compiled` with a changed tail hot-set size** | membership changes alter `hot_mask` between None/tensor — two graph variants; only the *shape* is stable, not proven across many rebases | rebase-check with 3+ rebases (1500+ steps) |
| **CPU fallback parity under `auto`** | `auto` picks `full` on CPU; verified bit-identical, but not benchmarked for speed | not needed — CPU is the reference |
| **Kaggle T4/P100** | every number is Colab T4; Kaggle may be a different T4 revision or a P100 | one short Kaggle probe when quota is authorized |
| **Determinism across runs on GPU** | CUDA atomics make `index_add_` non-deterministic; state/RMS ~3e-04 is expected, but run-to-run *identical* bytes were not tested | two runs of the same seed, compare theta bitwise |
| **A100 / larger B economics** | `--B 4096` on A100 untested; the compiled path may behave differently | deferred |
| **Energy / thermal throttling** | 30-min sessions were fine; 12 h Kaggle legs may throttle | monitor during the first long leg |
| **The `g-loop` reference path** | kept only as the A/B reference for `cat`; never intended for production | — |

---

## 4. REJECTED — measured and removed (do not re-add without new evidence)

| candidate | measured | note |
|---|---|---|
| Factored ring decay (b1) | **0.84x**, +0.61 GB VRAM | the factor is `(B,K,D)`, not `(B,K)` — no traffic saved |
| Factored tail decay (b2) | 0.98x | cost **four** correctness bugs; code removed |
| Cold-column ring credit (a1) | 0.95x | hot dims are 0.7–1.4% of the ring, and the masked write is slow regardless of size |
| K-loop ring credit (a2-kloop) | 0.90x | 18× more launches, same traffic |
| Sort + segment-sum (c-segmented) | 0.81x | the gather costs more than the contention it saves |
| `torch.compile(mode="reduce-overhead")` on the whole step | 0.003x (68 B/s) | recompiles instead of replaying; **never retry in this shape** |
| Whole-step CUDA graph | not attempted | would need the same static-shape discipline as `compiled`, but the win is unproven |

---

## 5. Bugs found by the benchmark (all fixed, all now gated)

Recorded because each one passed a shorter test first:

1. **Factored tail added `a` instead of `a/c`** → 35× RMS divergence; the loss
   moved only 3.8e-04. Invisible without a scale-aware state metric.
2. **Factored-ring tail seeding** multiplied an adopted `(B,K,H1)` gather by an
   ungathered `(B,K,D)` factor → crash, or silent corruption if `H1 == D`.
3. **Triton kernel loaded the `(B,K,D)` factor as a scalar** → 11× RMS error.
4. **Renormalization `E.div_(c)` instead of `E.mul_(c)`** → ~50× RMS jump on
   the first renorm; the 60-step gate had passed at 3e-07 only because no
   renorm landed inside its window.
5. **Fixed renorm interval underflowed** (100 steps × sigma 0.205 ≈ 1e-70) →
   NaN losses, visible only at 200 steps.
6. **Board forced a host sync per timed step**, silently cancelling the
   async-metrics candidate's premise.
7. **Profiler compared whole-window totals against a per-step wall clock** →
   phase table summed to 309% of wall.

**Lessons to carry forward:** compare *effective* values in scale-aware units
(max abs / RMS); run long enough to cross a rebase/renormalization boundary;
and treat a test that passes as a hypothesis, not a proof.

---

## 6. Next iteration — ranked candidates with their evidence

| candidate | evidence it targets | expected |
|---|---|---|
| **Cold-only ring layout** (store only the ~8,077 cold columns) | profiler: `masked_fill` = 5.1 ms/step (8%) to zero 1.4% of data; the cost is strided indexing, not bytes | 5–10% |
| **CUDAGraph, hand-rolled** (not `torch.compile`) | `cudaLaunchKernel` = 10.1 ms/step, 1,072 launches | 10–15% |
| **Compile the layer function only** | fwd+bwd layers = 41% of the step, only 8.3 ms of it sgemm | unknown |
| **Ring decay via a fused kernel** (avoid the 37.7 MB `w.mul_` without a `(B,K,D)` factor) | ring decay multiply = 13.7 ms/step | 5–10% |

Rules for the next round:
- Measure at **≥200 steps** on **real text** with the **live HF checkpoint**.
- Any candidate crossing a rebase must pass `rebase-check`.
- Anything that reassociates floating point must land in `numpy_port` too and
  re-clear the **0.03 CE** ruler bar.
- Science changes (lowering `ring_k`, capping the hot set, bf16 ring storage)
  are **not** optimizations — they need an explicit decision, and they are the
  only things likely to give a step change rather than 5–15%.
