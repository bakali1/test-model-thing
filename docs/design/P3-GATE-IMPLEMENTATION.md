# P3 gate — implementation note and measurements (2026-10-22)

> ### READ THIS FIRST (added 2026-10-22, after the measurements)
>
> **The implementation is sound and every number in it is valid.** The
> dense-credit reassociation (12× on the contraction, 6.8× end to end,
> measured against a float64 truth, with the dense-vs-ring bit-identity pin
> honestly restated as a measured float32 floor) is real work and it
> stands.
>
> **But the P3 LINE IS CLOSED AS A LEVER, and this document is now the
> record of a closed line rather than a proposal.** `RESULTS.md` §23 and
> M1/M6: the gate **engages** (it reaches 0.110 max) but is **not
> selective** — its between-sample spread is **2.4e-05** of its
> within-sample spread, so it is a global per-dimension re-scaling, not a
> per-sample choice. **And a selective *state* write could not have fixed
> the programme's failure anyway**, because the failure was **a table in
> the weights being overwritten** (§22), not a state being overwritten.
>
> **The replacement is the plan's §4/M2 — an explicit keyed store** — plus
> a training-signal arm (CL-6b: the D5-style copy circuit trained
> explicitly). **The dense path is now 6.8× faster, so the arm that
> matters — the keyed store — is affordable.**

**Status:** implemented, gated, measured. No commit; the working tree is dirty
on purpose.

This is the companion to `docs/design/P3-GATE-ANALYSIS.md` (the design). The
design's §5.1 recommendation is built here verbatim, the missing credit gate
is written, and §5.3's one unmeasured number (the dense credit's throughput
and memory) is measured on the MX150.

---

## 1. What was built

| # | item | where |
|---|---|---|
| 1 | forward: **no change**. `a = sigmoid(decay + gate * x_in)` already returned `(B, dim)`; `gate = 0` ⇒ `a == sigmoid(decay)` bit-for-bit | `gpu_port.py` `TorchLayer.forward`, `numpy_port.py` `Layer.forward` |
| 2 | gradient: **no change**. The traced form was already implemented and pinned | `gate_trace_update` / `gate_gradient` |
| 3 | credit: the gated arm runs the **dense** trace. The decay broadcast is ndim-guarded (`dec[:, None, :]` when `dec` is `(B, dim)`, `dec[None, None, :]` otherwise); the einsum and the add were already per-stream | `gpu_port.py` (dense credit block), `numpy_port.py` (new dense path) |
| 4 | decaytrace: ndim-guarded (`dec` vs `dec[None, :]`) — the silent `(1, B, dim)` broadcast | `gpu_port.py`, `numpy_port.py` |
| 5 | the refusal: **lifted in both engines**, and its comment rewritten to state the real history (three of the four stated blockers were broadcast changes; the fourth — the ring's per-dim truncation ceiling — is dissolved by the dense trace) | `gpu_port.py` `step_batch`, `numpy_port.py` `step_batch` |
| 6 | **the missing gate**: `tests/check_gated_credit.py` — the gated `step_batch`'s FULL gradient (`dembed`, `ddecay`, `dgate`) against torch autograd, plus the dense-vs-ring equivalence under a per-sample `dec`, plus the numpy reference's gated arm | new file |
| 7 | `tests/check_gated_recurrence.py`: the refusal assertion became a **run** assertion (and `tests/check_gate_grad.py`'s, which asserted the same refusal) | both files |
| 8 | `train_gpu.py --gated` | `train_gpu.py` |
| 9 | `tools/membench/model_backend.py`: `load_frozen(ckpt, gated=False)` (and `tools/d5_probe.py --gated`, so the probe can score a gated checkpoint) | both files |

Two things the design did not list, and why they were needed:

* **the gate trace is now persisted** (`gate_trace.{i}` in both engines' save
  and load). It was deliberately not persisted *because the arm refused to
  run*, so the buffer was always zero — an artefact of the refusal, not a
  property of the buffer. A resumed gated run that started the trace at zero
  would silently lose the chain across the resume, which is exactly the bug
  the trace exists to fix.
* **the numpy reference got the dense path** (~35 lines, as costed): the
  reference is ring + `TailCredit` only, so without it the gated arm had no
  parity gate at all. It mirrors the torch side's dense branch, the decay
  broadcast, the `trace_mode` argument and the "gated forces dense" rule.

The default engine is untouched: `theta`'s layout, the numerics and
`tests/check_parity.py` are exactly as they were. Every change sits behind
`self.gated` or `trace_mode == "dense"`.

---

## 2. The gates

| gate | result |
|---|---|
| `tests/check_gated_credit.py` (new) | **PASS** |
| `tests/check_gated_recurrence.py` | **PASS** |
| `tests/check_gate_grad.py` | **PASS** |
| `tests/check_parity.py` | **PASS** (T=1 bit-identity: 114 comparisons, worst differing element 0) |
| `tests/check_dense_trace.py` | **PASS** |
| `tests/check_gradients.py` | **PASS** (worst 1.0e-07) |
| `tests/check_trainer.py` | **PASS** |
| `tests/check_optimizations.py` | **PASS** |
| `tests/check_safety.py` | **PASS** |

(the whole `tests/` suite passes: 32 files, `.venv/bin/python`, CPU.)

### 2.1 The credit's deviation from autograd — the number, not just the PASS

`check_gated_credit.py` compares the gated `step_batch`'s gradient against
torch autograd over the SAME window, with the FULL cross-time chain (the
engine's credit and its two traces *reconstruct* the chain the engine's
per-byte backward does not have, so the reference must carry it).

Two references are computed: a float32 one (like-for-like) and a float64 one
(the truth), and the float32 rounding floor `|ref32 − ref64|` is measured, so
the tolerance can only be loosened by a measurement.

| slice | CPU (fp32 vs fp32) | CPU (vs float64) | fp32 floor | MX150 (fp32 vs fp32) |
|---|---|---|---|---|
| `embed` | 1.90e-07 | 2.70e-07 | 1.83e-07 | 3.49e-07 |
| `layer0.decay` | 9.79e-08 | 2.86e-07 | 2.86e-07 | 1.31e-07 |
| `layer0.gate` | 1.57e-07 | 2.84e-07 | 2.44e-07 | 1.96e-07 |
| `layer1.decay` | 1.46e-07 | 4.94e-07 | 4.94e-07 | 1.46e-07 |
| `layer1.gate` | 9.77e-07 | 6.19e-07 | 7.97e-07 | 6.74e-07 |

**Max deviation 9.8e-07 relative (CPU), 6.7e-07 (MX150) — at the float32
rounding floor of the same computation, not at a hand-set tolerance.** The
tolerance in the gate is `max(1e-4, 3 × floor)`; the measured deviations are
~100× inside the floor-scaled bound.

Other pinned facts, all in the same file:

* `gate = 0` is **bit-identical** to the incumbent: the loss, the CE, the
  recurrent states and **every non-gate gradient slice** (max rel dev
  0.000e+00, against the same trace mode). The gate's own gradient is
  *non-zero* at `gate = 0` — by design: the gate's effect is zero (the
  recurrence reduces exactly to `sigmoid(decay)`) but its derivative is not.
* the dense credit vs the ring incumbent: `dembed` rel dev 0.000e+00 (the
  ring is exact at `T < ring_k`).
* a **gated checkpoint round-trips**: the gate's parameters, the gate's
  cross-time trace, the dense credit trace and the parameters themselves all
  come back bit-exactly, and the resumed run continues **bit-identically**
  (the same gradient on the next step, from the checkpoint) -- in the numpy
  reference too.
* the design's §2 measurement, now permanent: with no truncation, ring ==
  dense == the explicitly enumerated survival products at **0.000e+00** under
  a per-sample, time-varying retention; with truncation, the ring drops
  credit (3.8e-04) and the dense trace does not (1.8e-08 vs the truth).

---

## 3. The measurement §5.3 left open: the dense credit's cost

**MX150 (4 GB, cc 6.1), dim=512, layers=16, T=1, mean of 20 timed steps
after 5 warmup, `torch.cuda.max_memory_allocated`.**

| B | path | ms/step | peak |
|---|---|---|---|
| 64 | ring, default init (ring_k=14) | 22.2 | 0.222 GB |
| 64 | **dense (the P3 credit path)** | **500.2** | **0.730 GB** |
| 64 | **gated arm (dense + gate)** | **501.2** | **0.735 GB** |
| 64 | ring, P4 init (ring_k=256) + tail | 367.0 | 1.094 GB |
| 64 | dense, P4 init | 504.4 | 0.731 GB |
| 256 | ring, default init | 44.1 | 0.368 GB |
| 256 | **dense** | **2004.5** | **2.500 GB** |
| 256 | **gated arm** | **2012.2** | **2.516 GB** |
| 256 | ring, P4 init + tail | 1430.2 | 3.926 GB |
| 256 | dense, P4 init | 2004.3 | 2.500 GB |
| 512 | ring, default init | — | dense/gated/P4-ring **OOM at construction** |
| 1024 | ring, default init | 139.1 | 0.967 GB — dense/gated/P4-ring **OOM** |

Three things fall out of this:

1. **The design's 2–4 ms/step projection is not supported.** The measured
   dense credit is 2.0 s/step at B=256 on this card — ~45× the ring's
   44 ms, and ~500× the projection.
2. **The cost is NOT the trace's memory — it is the einsum's SHAPE.**
   Per layer at B=256 (E is 134 MB):

   | op | ms |
   |---|---|
   | `E.mul_(dec[:, None, :])` (0.27 GB moved) | 6.45 |
   | `einsum("bvd,bd->vd", E, dlds)` (0.13 GB read) | **116.10** |
   | `E[ar, c_arr, :] += 1.0` | 0.13 |
   | *ALT:* `(E * dlds[:, None, :]).sum(0)` — the same contraction | **9.52** |

   The einsum contracts over `b` with `(v, d)` free: torch turns that into
   131 072 batched 1×1×256 GEMVs, and pays **18× the cost of the memory
   traffic it implies**. The elementwise-multiply-then-sum formulation is
   **12× faster** and would take the dense path from 1963 ms/step to
   **258 ms/step**. It is *not* bit-identical (a different summation order),
   so it is a measurement here, **not a change** — but it is the single
   highest-value line left in the engine.
3. **At the P4 init (the D5 configuration) the dense trace is the CHEAPER of
   the two in memory** (2.500 GB vs the P4 ring's 3.926 GB, which carries the
   tail's 1.33 GB accumulator on top) — as the design predicted — and 1.4×
   slower in step time. And the gated arm **cannot run at B=1024 on a 4 GB
   card** (neither can the P4 ring): the dense trace alone is 8.6 GB at
   B=1024, dim=512, 16 layers. The gated arm's ceiling on the MX150 is
   **B=256**.

The gated arm's *gradient* deviation from autograd on the MX150 is the same
as on the CPU: max **6.7e-07** relative, at the float32 floor (§2.1).

---

**The ALT is no longer a measurement: it is ADOPTED. See section 5** -- the
formulation, the numerics against a float64 truth, the 5.5x/6.8x end-to-end
speed on the MX150, and what happened to the dense-vs-ring bit-identity pin.

## 4. What is still UNMEASURED

* **The dense credit's cost on a T4.** The MX150 is cc 6.1 with no tensor
  cores and ~1/8 of a T4's bandwidth; the 116 ms einsum is shape-bound, not
  bandwidth-bound, so a T4 would be *faster* but not 45× faster. The number
  that matters for scheduling the P3 arm is the T4's, and it is still
  unmeasured (a Colab T4's free-tier quota is spent).
* **Whether the einsum's ALT formulation is acceptable.** It is 12× faster
  and it breaks the dense-vs-tail bit-identity (0.00e+00 → ~1e-7). That is a
  decision, not a measurement: the pin's *value* is that the two paths are
  the same object, and a 1e-7 relative difference preserves that claim, but
  the gate would have to be re-stated.
  **RESOLVED in section 5 below: measured, adopted, and the pin re-stated
  with the measured number.**
* **Whether the gate's mechanism does anything at all.** §2 of the design
  (M1) is untouched by this work: the per-sample spread of `a` on a TRAINED
  model is unmeasured, and a null there would make the arm a no-op.
* **A trained gated run.** Nothing here has trained a gated model; the arm
  has been run for a handful of steps in gates only.

---

## 5. The einsum, REFORMULATED (2026-10-22)

**Status: adopted. The dense path is 5.5x (B=64) to 6.8x (B=256) faster on
the MX150, the peak memory is UNCHANGED, and the numerics move by at most
one float32 ulp of the credit's scale.**

### 5.1 The change

`gpu_port.py`'s dense credit block:

    - dembed += torch.einsum("bvd,bd->vd", E, dlds)
    + dembed += (E * dlds[:, None, :]).sum(0)

`numpy_port.py`'s dense path keeps `np.einsum("bvd,bd->vd", E, dlds)`, and
the reason is measured, not stylistic: **in numpy the two formulations are
BIT-IDENTICAL** (max|diff| = 0.000e+00 at B in {3, 4, 7, 64, 256, 1024} x
dim in {8, 13, 16, 64, 512}, with positive and with cancelling data), because
numpy sums the contracted axis in the same left-fold order for both -- so the
multiply-then-sum semantics the reformulation buys on the GPU is ALREADY what
the numpy line computes -- **and numpy's einsum is 3x FASTER there** (27.9
ms against 78.2 ms per layer at B=256, dim=512), because numpy's einsum
never materialises the (B, 256, dim) product. Changing it would have been a
pure regression for a zero numerical change.

The dtype contract: the einsum does **not** promote (float32 in, float32 out)
and neither does the reformulation, and both accumulate in float32. The
reformulation's `sum(0)` is a float32 reduction of a float32 product, so the
accumulation precision is unchanged.

### 5.2 The numerics: new vs old vs the float64 truth

Two references, because they answer different questions:

* the **float64 truth** -- the same contraction in float64: "how far is the
  float32 answer from the mathematical answer";
* the **exact sum of the float32 products** -- every float32 product
  E[b,v,d]*dlds[b,d] is exact in float64, so a float64 sum of them IS the
  exact sum of the float32 products: "how much of the error is the
  SUMMATION ORDER", which is the ONLY thing the reformulation changes.

Measured by `tools/measure_dense_credit_sum.py` (the table is reproduced by
`tests/check_dense_credit_sum.py`). Max abs / max rel against the float64
truth:

| device | B | dim | einsum (old) | reformulation (new) | order-only error, old -> new |
|---|---:|---:|---|---|---|
| MX150 | 64 | 512 | 5.290e-06 / 1.428e-07 | **2.903e-06 / 7.838e-08** | 5.290e-06 -> 2.903e-06 (**0.55x, BETTER**) |
| MX150 | 256 | 512 | 1.051e-05 / 1.392e-07 | **9.246e-06 / 1.224e-07** | 1.051e-05 -> 9.246e-06 (**0.88x, BETTER**) |
| MX150 | 256 | 64 | 7.496e-06 / 1.152e-07 | **6.006e-06 / 9.234e-08** | 7.496e-06 -> 6.006e-06 (0.80x, better) |
| MX150 | 16 | 512 | 3.488e-06 / 1.879e-07 | **1.533e-06 / 8.258e-08** | 3.488e-06 -> 1.533e-06 (0.44x, better) |
| MX150 | 3 | 8 | 4.278e-07 / 5.202e-08 | **4.278e-07 / 5.202e-08** | 0.000e+00 (identical) |
| CPU | 64 | 512 | 2.970e-06 / 8.019e-08 | 5.198e-06 / 1.404e-07 | 2.970e-06 -> 5.198e-06 (**1.75x, WORSE**) |
| CPU | 256 | 512 | 8.503e-06 / 1.126e-07 | 9.535e-06 / 1.263e-07 | 8.503e-06 -> 9.535e-06 (1.12x, worse) |
| CPU | 16 | 512 | 1.038e-06 / 5.594e-08 | 3.488e-06 / 1.879e-07 | 1.038e-06 -> 3.488e-06 (3.36x, worse) |
| CPU | 3 | 8 | 4.278e-07 / 5.202e-08 | 4.278e-07 / 5.202e-08 | 0.000e+00 (identical) |
| numpy (reference) | any | any | -- | -- | **0.000e+00 -- bit-identical** |

**The verdict, stated plainly, per device:**

* **On the GPU -- the device the 12x is for -- the reformulation is CLOSER
  to the truth than the einsum it replaces**, in every setting measured
  (0.44x-0.88x of the incumbent's own error). The reason is visible in the
  arithmetic: the einsum lowers to a per-dim GEMV that accumulates with FMA
  (the product is never rounded before the addition), while the
  reformulation rounds each product to float32 and then sums.
* **On the CPU the reformulation is 1.12x-3.36x WORSE** than the einsum,
  because torch's CPU einsum happens to use a blocked accumulation. But both
  are at the float32 epsilon: the RELATIVE deviations are 8.0e-08 (old) and
  1.4e-07 (new) against a float32 eps of 1.19e-07 -- a difference of about
  one ulp of the credit's scale.
* **In numpy nothing moves at all**: the two formulations are the same bits.
* at **B=3** (the size the P3 credit gate runs at) the two are bit-identical
  everywhere, which is why **every existing gate's numbers are unchanged**.

### 5.3 The speed, on the MX150 (dim=512, layers=16, T=1, mean of 20 timed
steps after 5 warmup, `torch.cuda.max_memory_allocated`; the card idle, one
ssh at a time)

| B | path | before | after | peak before -> after |
|---|---|---:|---:|---|
| 64 | ring, default init | 22.2 ms | 21.4 ms | 0.222 -> 0.222 GB |
| 64 | **dense (the P3 credit path)** | **499.3 ms** | **90.4 ms** | **0.730 -> 0.730 GB** |
| 64 | **gated arm (dense + gate)** | **504.4 ms** | **85.2 ms** | 0.735 -> 0.735 GB |
| 64 | dense, P4 init | 508.0 ms | 89.2 ms | 0.731 -> 0.731 GB |
| 256 | ring, default init | 45.1 ms | 44.7 ms | 0.368 -> 0.368 GB |
| 256 | **dense** | **2005.3 ms** | **296.8 ms** | **2.500 -> 2.500 GB** |
| 256 | **gated arm** | **2012.3 ms** | **304.6 ms** | 2.516 -> 2.516 GB |
| 256 | dense, P4 init | 2004.8 ms | 294.2 ms | 2.500 -> 2.500 GB |

**5.5x at B=64, 6.8x at B=256** end to end (the credit was 96% of the step).
The **peak memory is UNCHANGED**: the reformulation's transient product
(B*256*dim*4 = 134 MB at B=256) is allocated and freed inside the trace's
own footprint, so it never raises the high-water mark. The per-op numbers
(`tools/measure_p3_credit.py`, unchanged): the credit op goes from
**116.08 ms to 9.52 ms** per layer -- 12.2x -- and the dense path's credit
total from 1962 ms to 258 ms per step.

### 5.4 What happened to the dense-vs-ring bit-identity pin

**The pin moved, and here is exactly how much and what it now guarantees.**

`tests/check_dense_trace.py` (a) pins "dense == ring+tail when the tail
covers every dim". Before the change that comparison was **0.00e+00** --
not because the two paths were different implementations of one object, but
because they ran the **same einsum on the same numbers** (the torch tail's
credit is `torch.einsum("bvh,bh->vh", E, g)`, the dense path's was
`torch.einsum("bvd,bd->vd", E, dlds)`).

After the change the two paths sum the same terms in a different order:

| comparison | before | after |
|---|---:|---:|
| dense vs ring+tail at the gate's size (B=4, dim=16) | 0.00e+00 | **0.00e+00** (unchanged: the two orders coincide at B=4) |
| dense vs ring+tail at B=64, dim=512, layers=2, T=4 | 0.00e+00 | **1.084e-07** (max abs 1.192e-07 on a scale of 1.100) |

**What the pin now guarantees, stated plainly:** the dense path and the
ring+tail path are the same OBJECT -- the dense trace is the tail with every
dim hot, exactly as the design says -- and they agree to **the float32
rounding floor of the credit's summation, about one ulp** (1.084e-07
relative, against a float32 ulp of 1.192e-07 at that scale). The gate says
so; it does not claim bit identity any more. `check_dense_trace.py`'s
docstring and its (a) check were re-stated accordingly, and a new section
(f) measures the same comparison at a size where the two orders really do
differ, with the tolerance stated as a measured float32 floor.

**The ring path was NOT touched to preserve the pin.** The tail's credit
could have been reformulated the same way -- and then the two paths would be
bit-identical again -- but the tail is on the DEFAULT path, and the default
engine's numerics must not move (`check_parity`'s "worst differing element
0" is the pin for that). So the pin was re-stated, not defended by changing
the incumbent.

### 5.5 The gates

| gate | result |
|---|---|
| `tests/check_dense_credit_sum.py` (**new**) | **PASS** (CPU and MX150) |
| `tests/check_dense_trace.py` | **PASS** (the re-stated pin, section (f) included) |
| `tests/check_gated_credit.py` | **PASS** -- and every number in it is UNCHANGED (the gate runs at B=3, where the two formulations are bit-identical) |
| `tests/check_parity.py` | **PASS** -- "T=1 bit-identity: 114 comparisons, worst differing element 0 (none)" |
| `tests/check_gradients.py` | **PASS** (worst 3.9e-08) |
| `tests/check_trainer.py` | **PASS** |
| the whole `tests/` suite (33 files) | **PASS** (CPU) |

`tests/check_dense_credit_sum.py` is the new gate: it measures the two
formulations against the float64 truth and against the exact sum of the
float32 products, with the **tolerance being the MEASURED float32 floor of
the same computation** (the incumbent einsum's own deviation), and it pins
that the ENGINE ships the reformulation at the bit level -- the credit's
operands are the engine's own decayed trace and the state gradient its
backward produced, the credit is a reduction over the batch axis of their
elementwise product, and it is NOT the einsum's bytes.

### 5.6 The recommendation

**ADOPT, unconditionally, in the torch engine.** The reason, in one line:
on the device the engine runs on, the reformulation is 12x faster per layer
and CLOSER to the float64 truth, at an unchanged peak memory and an unchanged
default path. The CPU's 1.12x-3.36x loss is one ulp of relative error on a
path that is 4.6x faster on the CPU too, so it is not worth a flag.

**The numpy reference keeps its einsum**, because there the reformulation is
a measured no-op numerically (bit-identical) and a 3x regression in time.

---

## 6. The next experiment

**The P3 arm itself, with the D5 protocol** (the design's §6):

1. Train the gated arm (`train_gpu.py --gated --trace-mode dense`) from the
   P4 init on the same corpus, same seed, same byte stream as the incumbent
   — the gate starts at exactly 0, so the two arms share a starting point
   bit-for-bit.
2. Score it with `tools/d5_probe.py --gated --ckpt <gated ckpt>` at the same
   gaps (8/12/32/64/128 B), the same matched 16-symbol alphabet, n=2048 and
   the same matched null per cell.
3. **M1 first, because it is the cheapest and the most decisive**: the
   per-dim AND the per-sample spread of `a_{b,t,d} = σ(decay_d + gate_d·x_t)`.
   A per-sample spread of ≈ 0 means the gate never engaged and the whole arm
   is a no-op — a null with a measured cause.
4. **M2**: the payload's ×null at 32–128 B (today 3.33× / 2.81× / **1.52×**
   at 128 B) and the cue key's ×null (today 90× / 65× / 84×).
5. **M3**: the clean control is the **same arm with the gate frozen at 0**,
   which is bit-identical to the incumbent — the 8 192 gate parameters
   (0.18 %) must not be confounded with the gate's function.
6. **M6** (the direct selectivity measurement): the payload's decode after
   `d` distractors, random vs maximally-confusable. If (i) and (ii) are
   indistinguishable, the write is still an indiscriminate weighted average,
   whatever the gate did to `a`.

And the engineering prerequisite, which is cheap and independent of the
result: **the dense credit's einsum is 18× more expensive than its memory
traffic** (§3.2). Fixing that (one line, at the price of a 1e-7 change in
the summation order) is worth doing before the arm is scheduled at any batch
where the step time matters.
