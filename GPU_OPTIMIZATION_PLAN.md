# GPU Optimization Plan — TMT (4.5M byte-level continual LM)

**Status:** ready to execute. Written 2026-09-24 after the first live 30k-step
Colab T4 leg. Supersedes the "next lever" note in
`PLAN-2026-09-23-GPU.md` §0.3.

**Baseline to beat:** **21,477 B/s** last-window (47.6 ms/step at B=1024),
measured on the live checkpoint (t=94,747, `ring_k=18`, 115 hot dims, dim=512,
16 layers, 4,481,793 params) on a T4, fp32, TF32 off.
Reference points: the old tail-sync bug was 2,655 B/s; the synthetic-checkpoint
micro-bench says 32,353 B/s but does **not** reproduce on the live checkpoint
(23,796 B/s there), so plan against 21.5 kB/s.

---

## 1. Where the time actually goes

Profiled analytically at the live config, and confirmed by the leg's own
diagnostics (end-of-stream flags cost 0.8%, so EOS is *not* a factor).

| component | traffic per step | share |
|---|---|---|
| **ring** — `w.mul_(dec)` 75.5 MB + `ring_vals` temp 75.5 MB + zero-hot 17 MB + scatter read 37.7 MB, **per layer × 16** | **3,325 MB** | **87%** |
| tail — `E.mul_(sig)` 241 MB + credit einsum read 121 MB | 362 MB | 9% |
| optimizer (8 passes over 4.5M params) + layer activations | 814 MB | — |
| matmul | 26.6 GFLOP ≈ 7.6 ms at ~3.5 TFLOPS fp32 | — |

Budget for the observed 47.6 ms/step:

```
  ~15 ms   elementwise bandwidth (3.83 GB at ~250 GB/s effective)
  ~8 ms    matmul
  ~25 ms   151M fp32 atomics/step from dembed.index_add_ + ~300-400 launches
```

**The ring is the target, not the tail.** The ring carries full `dim`-wide
embedding vectors for `K` lags, so its cost scales with
`B x K x dim x layers` (1024 × 18 × 512 × 16). The tail scales with
`B x 256 x hot` and is 3× smaller today. This inverts the assumption we made
when hot dims doubled 57 → 115 and the tail looked like the problem.

The hot-dim growth still matters — it makes A1 more valuable over time — but
the tail decay itself is only 6% of traffic.

---

## 2. Phase 0 — measurement harness (do this first, ~30 min)

**Problem:** every optimization currently costs a 25-minute leg to evaluate.
**Fix:** `bench_variants.py`, a self-contained script that benchmarks the
baseline and every candidate variant **in one process on one T4**, with
`torch.cuda.synchronize()` around each timing region and peak-allocation
reported. ~3 minutes for the whole board.

Requirements:
- Each variant is a patch applied to the *same* `GpuModel` (a flag or a
  subclass), so comparisons are apples-to-apples in one process.
- Report: B/s, ms/step, peak MB, and a scale-aware parity number
  (`max|a-b| / RMS`) of the resulting `theta` against the baseline after N
  steps from the same checkpoint and bytes.
- Print a one-line table sorted by B/s. This table becomes the RESULTS.md row.

This is the single highest-leverage half hour available: it turns every later
phase from a 25-min experiment into a 3-min one.

---

## 3. Phase A — no math change (safe; target 1.3–1.5×)

Identical arithmetic, identical results. Any drift here is a bug, not
reassociation.

### A1. Stop computing ring credit for hot columns (best risk/reward)

Today, per layer:

```python
w.mul_(dec[None, None, :])                    # decay the whole ring
torch.mul(w, dlds[:, None, :], out=ring_vals) # full dim-wide multiply
ring_vals[:, :, hot_mask] = 0.0               # ...then discard 22% of it
dembed.index_add_(0, rows, ring_vals.view(B*K, -1))
```

The hot columns are multiplied and scattered, then **zeroed and thrown away** —
22% of the ring's two most expensive ops, and that fraction grows as the hot
set grows. Fix: index only the cold columns in the credit path.

```python
cold = self.cold_idx[i]                       # int64 (Dc,) built at rebase
torch.mul(w.index_select(2, cold), dlds[:, cold][:, None, :], out=vals_cold)
dembed.index_add_(0, rows, vals_cold.reshape(B*K, -1))   # then scatter into [:, cold]
```

**Keep the full ring stored.** The hot columns' stored values are still needed:
`TorchTail.sync` seeds newly-hot dims from `ring_w[:, :, adopt]`, and a dim that
later drops *out* of the hot set resumes with its ring history intact. Dropping
them from storage is a *science* change (see §6), not an optimization.

Expected: −22% of ring multiply + scatter traffic, growing with the hot set.

### A2. Fuse the `ring_vals` temp (removes ~1.2 GB/step)

`torch.mul(w, dlds, out=ring_vals)` **writes** 37.7 MB/layer that `index_add_`
immediately **reads back** — 1.2 GB/step of pure round-trip. Fuse the
multiply into the scatter with `torch.compile` (or a `K`-loop of
`index_add_(0, ring_b[:, k], w[:, k, :] * dlds)`), so `w` is read once.

Watch for: `torch.compile` reassociating the reduction inside `index_add_`.
Check the parity number from Phase 0; if it moves, fall back to the explicit
k-loop (more launches, same traffic saving).

### A3. CUDA-graph the step (removes launch + sync overhead)

~300–400 launches per step plus a `float(loss)` host sync per step. Capture the
step as a CUDA graph (static shapes, static input buffers) and read the loss
from a device tensor instead of syncing. Low risk, no math change.

---

## 4. Phase B — factored decay (float reassociation; another 1.3–1.5×)

These change the *order* of floating-point operations. Mathematically
identical products, different rounding. See §7 for the mandatory gate.

### B1. Ring decay factoring — biggest single win (1.2 GB/step)

The ring is decayed in full every step:

```python
w.mul_(dec[None, None, :])        # 37.7 MB read + 37.7 MB write, per layer
```

but each slot is overwritten every `K` steps (`w[:, p] = 1.0`), so a slot only
ever accumulates `K` decay factors. Store `w` **undecayed** and keep the decay
product per `(b, k)` instead:

```python
# c[b,k] = product of dec[b] since slot (b,k) was written
c.mul_(dec[:, None])              # 18k floats = 147 KB  (was 75.5 MB)
c[:, p] = 1.0                     # this slot is fresh
# contribution uses the scaled gradient, same product, no stored decay:
w_scaled = w * (c.reshape(B, K, 1) * dlds[:, None, :])
```

- No underflow: `c` resets every `K = 18` steps; worst case ~0.5¹⁸ ≈ 4e-6.
- Touches 147 KB instead of 604 MB per step. Also removes A2's temp problem
  from the decay side entirely.
- `torch.compile` fuses `c` scaling into the multiply for free.

### B2. Tail decay factoring (saves ~240 MB/step)

Same trick for `E.mul_(sig_hot)`, with one difference: `E` is never
overwritten, so an undecayed `c` underflows to zero within ~50–100 steps.
Renormalize periodically:

```python
c.mul_(sig)                       # B x H scalars
# every N=100 steps: E /= c; c = 1
# credit folds c into the gather: einsum(E, g * c)
```

Amortized cost: one 120 MB pass per 100 steps ≈ 1.2 MB/step, replacing
241 MB/step.

---

## 5. Phase C — the atomics and the fused kernel (highest ceiling)

The ~25 ms unaccounted for is 151M fp32 atomic adds/step:
`dembed.index_add_` scatters 9.4M floats **per layer** into only **256** target
rows, so ~72 atomic row-adds contend per target row per layer.

**Measure before building.** Phase 0 should add a micro-bench of the isolated
`index_add_` at these exact shapes. If it is 5 ms, sort+segment-sum is not
worth it; if it is 20 ms, it is.

Options, in increasing ambition:

1. **Sort + segment-sum** (the original plan's lever). Sort the `B*K = 18,432`
   row indices by target byte, then a segmented reduction replaces atomics.
   Honest caveat: the gather of `ring_vals` in sorted order costs about as much
   traffic as it saves, so the win is contention, not bandwidth.
2. **One fused Triton kernel per layer** — the real fix. Read `w` **once**,
   apply `c * dlds` scaling, mask to cold columns, and atomically accumulate
   into `dembed`, all in a single pass:

   ```
   ring traffic today:   208 MB/layer  (3 passes: decay r+w, multiply r+w, scatter r)
   fused kernel:         ~38 MB/layer  (1 pass)  => ring 3.3 GB -> ~0.7 GB/step
   ```

   This is the single change that could take the step from 47.6 ms toward the
   ~23 ms floor implied by bandwidth + matmul alone. It is also the most
   likely to introduce a subtle bug, so it goes last, behind the parity gate.

---

## 6. Explicitly not optimization levers

| idea | measured / contractual verdict |
|---|---|
| B=4096 instead of 1024 | only ~6% better per byte (34.3k vs 32.4k synthetic) — per-step costs scale with B, so it buys nothing |
| TF32 / bf16 / autocast | breaks the fp32 parity contract; would require re-baselining the whole ruler |
| lowering `ring_k` (18 → smaller) | cuts ring cost *linearly* — but it changes the model's approximation. A science decision |
| capping the hot set (top-N by usage) | makes A1 more valuable and bounds tail growth — also a science decision |
| cold-only ring storage | saves 604 → 470 MB of VRAM, but a dim dropping out of the hot set loses its ring history (K-step warmup). Science, not engineering |
| multi-stream / larger effective B via vmap | changes trajectory; revisit only if Phase C lands and throughput still binds |

---

## 7. Protocol (non-negotiable)

1. **Local CPU gates after every single change**, before any quota:
   `tests/check_gradients.py`, `tests/check_parity.py`, `tests/check_trainer.py`.
   Free, and they catch semantic errors immediately.
2. **One `bench_variants.py` T4 run per batch of changes** (~3 min), never one
   25-min leg per idea.
3. **Phase A must be bit-identical.** Any drift in theta means a bug — stop.
4. **Phase B/C reassociate floating point and therefore move the trajectory.**
   For each: apply the *identical* change to `numpy_port.py` so CPU ≡ GPU
   parity is preserved, snapshot the current checkpoint, then re-run the ruler.
   Accept only if in-domain CE stays within the **0.03 CE** bar of the
   t=94,747 baseline (1.1741 ± 0.0415 in-domain, 1.1425 ± 0.0165 shift).
   Record the ruler delta in RESULTS.md alongside the speed delta.
5. **Rollback discipline:** tag the commit before each phase
   (`git tag opt-before-B1`). Checkpoints are immutable and content-verified,
   so the worst case is "revert the code, keep training from t=94,747".
6. **Nothing is recorded as an improvement without before/after numbers** in
   RESULTS.md, from the Phase 0 harness.

---

## 8. Work order and expected payoff

| # | item | effort | risk | expected B/s | cumulative |
|---|---|---|---|---|---|
| 0 | `bench_variants.py` harness | 30 min | none | — | enables everything |
| A1 | skip hot columns in ring credit | 1–2 h | none (bit-identical) | ~24 k | 1.1× |
| A2 | fuse the `ring_vals` temp | 1–2 h | low | ~28 k | 1.3× |
| A3 | CUDA-graph the step | 2–3 h | low | ~31 k | 1.45× |
| B1 | factored ring decay | 2–3 h | medium (trajectory) | ~37 k | 1.7× |
| B2 | factored tail decay | 1–2 h | medium (trajectory) | ~39 k | 1.8× |
| C | measure atomics, then fused Triton kernel | 1 day | high | ~45 k+ | 2.1×+ |

Consequences at each level:

| throughput | 12 h Kaggle session | 4.5 GB one-epoch pass |
|---|---|---|
| 21.5 kB/s (now) | 0.93 GB | 58 h |
| 31 kB/s (after A) | 1.34 GB | 43 h |
| 39 kB/s (after B) | 1.69 GB | 32 h |

**Recommended order given limited time:** Phase 0 → A1 → A2 → A3 → B1 → B2 →
C. A1–A3 are worth doing even if nothing else happens: they are pure deletion
of wasted work, they cannot change the model's mathematics, and together they
recover the throughput the synthetic bench promised. B1 is the best
speed-per-line in the plan but is the first change that requires a ruler
re-baseline, so do it only with time left to validate properly.
