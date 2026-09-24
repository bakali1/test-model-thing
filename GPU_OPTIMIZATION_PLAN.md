# GPU Optimization Plan — TMT (4.5M byte-level continual LM)

**Status: EXECUTED on 2026-09-24. Read §0 first — it supersedes the phase
targets below.** One candidate is worth adopting (`a2-compiled`, +20%); the
headline traffic-arithmetic predictions in this plan were refuted by
measurement. The phase text is kept as the record of what was believed when
written, with corrections marked. This plan supersedes the "next lever" note
in `PLAN-2026-09-23-GPU.md` §0.3/§0.4.

**Baseline to beat:** **21,477 B/s** last-window (47.6 ms/step at B=1024),
measured on the live checkpoint (t=94,747, `ring_k=18`, 115 hot dims, dim=512,
16 layers, 4,481,793 params) on a T4, fp32, TF32 off.
Reference points: the old tail-sync bug was 2,655 B/s; the synthetic-checkpoint
micro-bench says 32,353 B/s but does **not** reproduce on the live checkpoint
(23,796 B/s there), so plan against 21.5 kB/s.

**Arithmetic correction:** at the live H=115, the hot fraction is
`115 / (16 × 512) = 1.40%`, not 22%. A1's speedup must therefore be measured,
not predicted from the old 22% estimate. The traffic table below is a
hypothesis until the Phase 0 CUDA profiler confirms its convention and
kernel-time attribution.

---

## 0. MEASURED OUTCOME (supersedes the targets below)

Board run on the live checkpoint, B=1024, 12 timed steps, one T4 process
(`bench_variants.py`; full table in RESULTS.md):

| variant | B/s | vs baseline | verdict |
|---|---|---|---|
| **a2-compiled** | **23,933** | **1.20x** | **ADOPT** (state/RMS 3.6e-05) |
| a1-cold | 20,679 | 1.03x | noise — as the 1.4% arithmetic predicts |
| b2-tail | 19,791 | 0.99x | no win (tail is 9% of traffic) |
| a2-kloop | 19,298 | 0.96x | worse than the compiled path |
| c-triton | 18,287 | 0.91x | the fused kernel is slower |
| b1-triton | 18,010 | 0.90x | — |
| b1-b2 | 17,170 | 0.86x | regression |
| b1-ring | 17,014 | 0.85x | **regression**, +0.61 GB VRAM |
| c-segmented | 14,105 | 0.71x | 2x slower than the atomics it replaces |
| a3-compile | 68 | 0.003x | catastrophic; do not retry in this shape |

Isolated ring scatter: `index_add_` **0.320 ms/layer** (~5.1 ms/step across 16
layers); sort+segment **0.661 ms**.

**Three corrections to this plan, all from measurement:**

1. **A1 was never 22%.** Hot dims are 57–115 *total* across 16×512 = 8,192
   ring dimensions (0.7–1.4%). The original text divided by 512 as though the
   count were per-layer. Real value ≈ 0.5% of traffic; measured 1.03x (noise).
2. **B1's premise was wrong.** The decay factor is `(B,K,D)`, not `(B,K)`,
   because `dec` is per-(stream, dim). Factoring trades a 37.7 MB multiply for
   an equally large factor multiply *plus* an extra multiply in the credit
   path, and costs +0.61 GB of VRAM. The planned "biggest single win" is a 15%
   regression.
3. **The ~25 ms/step atomic budget does not exist** — it is 5.1 ms — and the
   sort+segment replacement is 2x slower than the atomics it targets, because
   the gather costs more than the contention it saves. The ring is not
   bandwidth-bound as the traffic model implied: a single-pass fused kernel is
   *slower*.

**What actually worked** is the one prediction that did not rest on the
traffic model: deleting the materialized `ring_vals` temporary, which Inductor
fuses into the scatter. That single change is +20%, and it is the whole of the
remaining Phase-A win — not the 1.5x the phase targeted.

### 0.1 What the profiler changed (measured, T4, `bench_variants.py --profile`)

`step_batch` now carries `record_function` phase regions (bit-identical output
with them on). Per-step attribution:

| phase | ms/step | share |
|---|---|---|
| **ring_credit** | 21.3 | 34% |
| bwd_layers | 15.7 | 25% |
| **tail_credit** | 10.2 | 16% |
| fwd_layers | 10.2 | 16% |
| decaytrace | 3.6 | 6% |
| optimizer | 3.0 | 5% |

Kernel-level facts that redirect the work:

- **`masked_fill` costs 5.1 ms/step (8%)** to zero hot columns that are **1.4%
  of the data**. The ring's cost is *strided indexing*, not bandwidth — which is
  why `a1-cold` gained nothing (it swapped a slow masked write for an equally
  slow gather). The fix is a **contiguous cold-only ring layout**.
- **`cudaLaunchKernel`: 10.1 ms/step, 1,072 launches/step.** Real overhead, and
  the right target for A3 — but via a hand-rolled `torch.cuda.CUDAGraph`, not
  `torch.compile` over the whole step (which was 0.003x).
- **The layer stack is 41% of the step and was never a target.** Only 8.3 ms of
  its 26 ms is sgemm; the rest is layer_norm/activation/state elementwise.
  Compiling just the layer function is the next candidate.
- **The tail is 16%, not the 9% the traffic model implied**, which is why
  `b2-tail` landed at 0.99x.

---

## 1. Where the time goes (as believed at writing time — corrected by §0)

The traffic figures below are an analytical hypothesis at the live config;
the leg diagnostics establish that end-of-stream flags cost about 0.8%, so EOS
is not the factor. Phase 0 must confirm the actual kernel-time split.

| component | traffic per step | share |
|---|---|---|
| **ring** — `w.mul_(dec)` 75.5 MB + `ring_vals` write 37.7 MB + scatter read 37.7 MB + hot-column zeroing ~0.53 MB average/layer, **per layer × 16** | **~2,420 MB** | **hypothesis; verify with profiler** |
| tail — `E.mul_(sig)` 241 MB + credit einsum read 121 MB | 362 MB | 9% |
| optimizer (8 passes over 4.5M params) + layer activations | 814 MB | — |
| matmul | 26.6 GFLOP ≈ 7.6 ms at ~3.5 TFLOPS fp32 | — |

Budget for the observed 47.6 ms/step:

```
  ~15 ms   elementwise bandwidth (about 3.6 GB at ~250 GB/s effective)
  ~8 ms    matmul
  ~25 ms   151M fp32 atomics/step from dembed.index_add_ + ~300-400 launches
```

**The ring is the target, not the tail.** The ring carries full `dim`-wide
embedding vectors for `K` lags, so its cost scales with
`B x K x dim x layers` (1024 × 18 × 512 × 16). The tail scales with
`B x 256 x hot` and is roughly 6–7× smaller today. This inverts the
assumption we made
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
- Each variant is a patch applied to a fresh `GpuModel` loaded from the same
  immutable checkpoint and the same input bytes. Do not reuse a mutated model
  object between variants; reset peak-memory counters and release state.
- Report: B/s, ms/step, peak MB, and scale-aware state parity metrics
  (`max|a-b| / RMS`, plus max absolute error) for theta, gradients, moments,
  ring, and tail state after N steps.
- Print a one-line table sorted by B/s. This table becomes the RESULTS.md row.

This is the single highest-leverage half hour available: it turns every later
phase from a 25-min experiment into a 3-min one.

---

## 3. Phase A — no intended model change (measure; no fixed speed target)

Identical arithmetic, identical results. Any drift here is a bug, not
reassociation.

### A1. Stop computing ring credit for hot columns (small at current H; measure)

Today, per layer:

```python
w.mul_(dec[None, None, :])                    # decay the whole ring
torch.mul(w, dlds[:, None, :], out=ring_vals) # full dim-wide multiply
ring_vals[:, :, hot_mask] = 0.0               # discard the hot subset (1.4% at H=115)
dembed.index_add_(0, rows, ring_vals.view(B*K, -1))
```

The hot columns are multiplied and scattered, then **zeroed and thrown away** —
1.4% of the ring's two most expensive ops at the current live configuration,
and that fraction grows as the hot set grows. A cold-only implementation must
accumulate a `(256, D_cold)` temporary and then add it to `dembed[:, cold]`;
it cannot pass a `D_cold` source directly to a `D`-column `dembed.index_add_`.

```python
cold = self.cold_idx[i]                       # int64 (Dc,) built at rebase
vals_cold = w.index_select(2, cold) * dlds[:, cold][:, None, :]
cold_dembed = torch.zeros((256, len(cold)), dtype=F, device=w.device)
cold_dembed.index_add_(0, rows, vals_cold.reshape(B * K, -1))
dembed[:, cold] += cold_dembed
```

**Keep the full ring stored.** The hot columns' stored values are still needed:
`TorchTail.sync` seeds newly-hot dims from `ring_w[:, :, adopt]`, and a dim that
later drops *out* of the hot set resumes with its ring history intact. Dropping
them from storage is a *science* change (see §6), not an optimization.

Expected: only the measured hot fraction of the credit path, about 1.4% at
H=115; the speedup is currently **TBD**, not a 1.1× claim. It becomes more
valuable only if the hot set grows substantially.

### A2. Fuse the `ring_vals` temp (potentially removes ~1.2 GB/step)

`torch.mul(w, dlds, out=ring_vals)` **writes** 37.7 MB/layer that `index_add_`
immediately **reads back** — 1.2 GB/step of pure round-trip. Fuse the
multiply into the scatter with `torch.compile` (or a `K`-loop of
`index_add_(0, ring_b[:, k], w[:, k, :] * dlds)`), so `w` is read once.

Watch for: `torch.compile` reassociating the reduction inside `index_add_`.
An explicit K-loop lowers temporary peak but does **not** by itself remove
the multiply-write/scatter-read round trip. Only a genuinely fused kernel
should be credited with that traffic reduction. Check the full parity state
from Phase 0; if it moves, keep the K-loop as a separately measured fallback.

### A3. CUDA-graph the step (removes launch + sync overhead; medium risk)

~300–400 launches per step plus a `float(loss)` host sync per step. Capture the
step as a CUDA graph only after the host sync is removed/isolated. Measure
launch overhead first: a graph cannot remove ring bandwidth or atomic
contention. Treat graph capture as medium risk, not low risk.

---

## 4. Phase B — factored decay (float reassociation; another 1.3–1.5×)

These change the *order* of floating-point operations. Mathematically
identical products, different rounding. See §7 for the mandatory gate.

### B1. Ring decay factoring — **corrected: not the advertised 1.2 GB win** (format change)

The ring is decayed in full every step:

```python
w.mul_(dec[None, None, :])        # 37.7 MB read + 37.7 MB write, per layer
```

The first draft incorrectly described the decay as one scalar per `(b,k)`.
That is false: `dec` is a `(D,)` vector of dimension-specific decay values, so
the factored state must retain a factor for every `(b,k,d)`:

```python
# c[b,k,d] = product of dec[d] since slot (b,k,d) was written
c.mul_(dec[None, None, :])        # B*K*D floats, the same ring-sized state
c[:, p, :] = 1.0
w_scaled = w * c * dlds[:, None, :]
```

This removes the explicit `w` mutation and is useful for state-space
experiments, but it does **not** reduce the dominant memory traffic: both the
legacy decay and the factored factor read/write `(B,K,D)` values. It therefore
must not be credited with the claimed 1.2 GB/step saving. A future compressed
age representation would need a different proof and measurement.

The implementation keeps this corrected `(B,K,D)` representation behind the
versioned `ring_algorithm='factored'` flag. The ring is still ephemeral under
the existing checkpoint contract, so resume explicitly cold-starts `c=1` and
`w=0`; the factor is saved to prevent a future state-format change from
silently reinterpreting it.

### B2. Tail decay factoring (potential ~240 MB/step; versioned state required)

Same trick for `E.mul_(sig_hot)`, with one difference: `E` is never
overwritten, so an undecayed `c` underflows to zero within ~50–100 steps.
The per-layer/B-column factor must be persisted with the checkpoint; the
current saved E is already decayed and cannot be reinterpreted as factored
state. Renormalize periodically:

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
rows, so the average source-to-target contribution count is about
`9.4M / 256 ≈ 36,864` per target row per layer; actual contention must be
profiled rather than inferred from an incorrect factor of 72.

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
   ring traffic today:   ~151 MB/layer  (decay r+w, multiply write, scatter read)
   fused kernel:         ~38 MB/layer  (1 pass)  => ring ~2.4 GB -> ~0.7 GB/step
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
   The CUDA profiler/harness must also compare all state tensors, not only theta.
   These gates catch semantic errors before cloud time is spent.
2. **One `bench_variants.py` T4 run per batch of changes** (~3 min), never one
   25-min leg per idea.
3. **Phase A uses the strictest feasible state parity.** A true fused or
   cold-subset scatter may change CUDA atomic scheduling, so use a documented
   tolerance for the complete state rather than claiming universal bitwise
   equality. Any large drift is a bug — stop.
4. **Phase B/C reassociate floating point and therefore move the trajectory.**
   Keep the old engine as the reference; implement the candidate behind a
   versioned flag, compare it against the old CPU/Torch trajectory, snapshot
   the old checkpoint, then run the ruler. Do not silently replace the old
   NumPy semantics. Accept only if the documented state tolerance and the
   **0.03 CE** quality bar pass. Factored state must add an algorithm/checkpoint
   version and persist or explicitly migrate its new decay factors.
5. **Rollback discipline:** tag the commit before each phase
   (`git tag opt-before-B1`). Checkpoints are immutable and content-verified,
   so the worst case is "revert the code, keep training from t=94,747". A
   factored-decay variant must never silently load an old checkpoint: require
   an explicit version check and migration/warmup policy.
6. **Nothing is recorded as an improvement without before/after numbers** in
   RESULTS.md, from the Phase 0 harness.

---

## 8.1 Implementation status in this worktree

The candidate paths are implemented behind explicit flags, with the legacy path
unchanged as the default:

- `bench_variants.py` provides the fresh-model, synchronized Phase-0 board,
  profiler, and isolated `index_add_`/segmented microbenchmark.
- A1 (`--ring-credit cold`), A2 controls (`kloop` / `compiled`), and C
  (`segmented` / CUDA-only `triton`) are available for T4 measurement.
- A3 is exposed as `--compile-step` plus `--async-metrics`; it deliberately
  refuses to run with the per-step host metric sync.
- B1/B2 are versioned candidates with explicit migration and saved factors.
  B1's factor is `(B,K,D)`, correcting the invalid scalar `(B,K)` proposal;
  therefore it is not credited with a traffic win.
- `tests/check_optimizations.py` is the local CPU gate for all candidate state,
  loss, and checkpoint-version behavior.

No GPU result is claimed in this worktree: the laptop environment has
`torch 2.14.0+cpu` and no CUDA device. The T4 board and the documented 0.03 CE
ruler still need to be run before accepting any candidate as an improvement.


## 9. Work order and expected payoff

| # | item | effort | risk | expected B/s | cumulative |
|---|---|---|---|---|---|
| 0 | `bench_variants.py` harness | 30 min | none | — | enables everything |
| A1 | skip hot columns in ring credit | 1–2 h | low | TBD (small at H=115) | measure |
| A2 | fuse the `ring_vals` temp | 1–2 h | low/medium | TBD | measure |
| A3 | CUDA-graph the step | 2–3 h | medium | TBD | measure |
| B1 | factored ring decay (corrected D-factor, no promised traffic win) | 2–3 h | medium (trajectory + format) | TBD | measure |
| B2 | factored tail decay | 1–2 h | medium (trajectory + format) | TBD | measure |
| C | measure atomics, then fused Triton kernel | 1 day | high | TBD | measure |

Consequences at each level:

| throughput | 12 h Kaggle session | 4.5 GB one-epoch pass |
|---|---|---|
| 21.5 kB/s (now) | 0.93 GB | 58 h |
| 31 kB/s (hypothetical) | 1.34 GB | 43 h |
| 39 kB/s (hypothetical) | 1.69 GB | 32 h |

**Recommended order given limited time:** Phase 0 → A2 → B1 (versioned
experiment) → A3 → A1 → B2 → C. A1 is small at the current H=115; A2 and B1
are the first candidates worth measuring, while B1 also requires a format and
ruler migration plan.
