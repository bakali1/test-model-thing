# P3 gate — implementation note and measurements (2026-10-22)

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

(the whole `tests/` suite passes: 33 files, `.venv/bin/python`, CPU.)

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
* **Whether the gate's mechanism does anything at all.** §2 of the design
  (M1) is untouched by this work: the per-sample spread of `a` on a TRAINED
  model is unmeasured, and a null there would make the arm a no-op.
* **A trained gated run.** Nothing here has trained a gated model; the arm
  has been run for a handful of steps in gates only.

---

## 5. The next experiment

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
