# Review of `docs/optimization/GPU_OPTIMIZATION_PLAN.md` and `docs/plans/PLAN-2026-09-23-GPU.md`

> **Historical review — do not execute as current status.** The optimization board has since run.
> `docs/optimization/GPU_OPTIMIZATION_PLAN.md` §0 and `docs/optimization/OPTIMIZATION_TEST_STATUS.md` record the measured outcome:
> `a2-compiled` was adopted, the other listed candidates were rejected/neutral, and the remaining
> gaps are listed there. The implementation and cloud-run state is current in
> `docs/plans/PLAN-2026-09-23-GPU.md` §8.2 and `RESULTS.md`. The findings below are preserved as the review that
> caught the earlier arithmetic/state-format problems.

**Review basis:** current implementation at commit `53db379`, the live 30k-step T4 leg recorded in `RESULTS.md`, current `gpu_port.py`/`train_gpu.py`, and the current CPU gates.  
**Reviewed plans:**

- `test-model-thing/GPU_OPTIMIZATION_PLAN.md`
- `test-model-thing/PLAN-2026-09-23-GPU.md`

## Fixes applied after this review

The following review findings have now been corrected in the worktree:

- the generated Colab leg no longer embeds an HF token and uses a runtime
  Colab secret;
- the live leg defaults to 10k steps, pulls before trusting a stale local
  checkpoint, validates expected parameter count/digest sidecars, and bounds
  its HTTP download;
- `train_gpu.py` passes `expect_n` and requires digest sidecars on Hub pulls;
- Hub sidecars upload before NPZ publication and are removed by remote prune;
- Kaggle and the fetcher use the same `corpus/current` path;
- corpus downloads are pinned to the recorded repository revision;
- tail memory checks include replacement peak memory and restored E;
- the optimization/parent plans now use H=115 => 1.40% rather than 22%;
- the parent plan uses the real ~21.5–22.1 kB/s baseline and marks the
  factored-decay format/version requirement.

The optimization implementations themselves have not been added yet; this
review still requires Phase 0 profiling before A2/B1/C experiments.

## Executive answer

### Is the real problem GPU throughput?

**Yes, if the immediate goal is to train on more bytes and run experiments faster.** The live T4 leg establishes a real throughput ceiling of approximately **21.5–22.1 kB/s** at B=1024, versus roughly 1 kB/s on the CPU. The ring-credit path is a credible target because it performs large repeated elementwise work and a large scatter reduction.

However, the optimization plan currently overstates how much is known about the bottleneck and contains at least one important arithmetic error:

- The live configuration has **115 hot dimensions total** across 16×512 = 8,192 ring dimensions.
- That is only **1.40%**, not 22%.
- Therefore A1 cannot plausibly save 22% of ring work or deliver the stated 1.1× improvement by itself.

The real immediate action should be:

1. verify the traffic/time breakdown with an actual CUDA profiler;
2. implement a correct benchmark harness;
3. test A2/A3 and B1 behind flags;
4. treat A1 as a small optimization, not the headline win;
5. add explicit state-format/version changes for any factored-decay algorithm.

The optimization plan is directionally correct, but it should not be executed exactly as written.

## What is verified and what is not

| Claim | Assessment |
|---|---|
| Real GPU throughput is much higher than CPU throughput | **Verified** by the recorded live T4 leg. |
| Live planning baseline should be ~21.5 kB/s, not synthetic 32.4 kB/s | **Correct and important**. The plan already incorporates this. |
| Ring work is a major cost | **Plausible and likely**, but the claimed percentage is analytical, not a measured kernel profile. |
| Tail is not the dominant bottleneck at the current hot count | **Likely correct**. At 115 total hot dims, tail E is about 120.6 MB fp32. |
| Ring is 87% of step time | **Not yet verified**. Traffic is not time, and the table has a likely double-counting/definition problem. |
| `index_add_` contributes roughly 151M atomic operations per step | **Operation count is plausible** for 16 layers × B×K×D = 151M source values. The stated contention factor (~72 per target row) is not consistent with the current shapes. |
| A1 saves 22% of ring work | **Incorrect** at the recorded H=115. |
| A2 removes a full temporary round trip | **Correct as a potential fused implementation**, but the proposed K-loop does not by itself remove the round trip. |
| A3 is low risk | **Overstated**. CUDA graph capture is nontrivial with the current Python loop, host synchronization, tail rebase, and in-place optimizer. |
| B1 is mathematically equivalent | **Conceptually valid, but requires a new persisted state format and a migration/version decision.** |
| B2 is mathematically equivalent | **Conceptually valid, but the decay factor must be persisted and the checkpoint must be versioned.** |
| Expected B/s after each phase | **Unverified estimates**. They must not be used as acceptance criteria before measurement. |

## Critical arithmetic correction: A1 is much smaller than claimed

The optimization plan says the live configuration has 115 hot dimensions and that hot columns account for 22% of ring work. The actual fraction is:

```text
H / (layers × dim) = 115 / (16 × 512) = 0.0140 = 1.40%
```

The current ring credit source has:

```text
B × K × dim = 1024 × 18 × 512 = 9,437,184 values per layer
B × K × H   = 1024 × 18 × 115  = 2,119,680 hot values per step across all layers
```

The hot portion is therefore about **1.4%**, not 22%. The current full `ring_vals[:, :, hot_mask] = 0` also writes only about 2.12M float values, approximately 8.5 MB per step, not 17 MB by the simple source-size calculation.

A1 may still be useful because it avoids work for a growing hot set, but its expected effect at H=115 should be close to the measured fraction of the relevant operations, not 22%. The current A1 expected value of approximately 24 kB/s is unsupported.

### A1 pseudocode also needs a dimension mapping

The proposed snippet:

```python
dembed.index_add_(0, rows, vals_cold.reshape(B*K, -1))
```

cannot be used directly when `vals_cold` has only `D_c` columns and `dembed` has `D` columns. It must accumulate into a `(256, D_c)` temporary and then add it to `dembed[:, cold]`, or use an equivalent scatter into the selected columns. The plan should show the complete correct mapping, not just the conceptual subset.

## Problems in the traffic budget

The ring table needs to be rewritten with an explicit traffic convention. It currently lists:

- `w.mul_`: 75.5 MB/layer, apparently read + write;
- `ring_vals` temp: 75.5 MB/layer;
- zero-hot: 17 MB;
- scatter read: 37.7 MB/layer.

If `ring_vals` at 75.5 MB already means read + write, then the separate 37.7 MB scatter read is counted twice. If 75.5 MB means something else, it should be labeled explicitly. The same ambiguity affects the claimed 3.325 GB/step ring total and the claimed 1.2 GB A2 saving.

Before prioritizing phases, run a profiler that reports:

- actual CUDA kernel time by operator;
- bytes read/written where available;
- `index_add_` time in isolation;
- time for ring decay, ring multiply, zeroing, and scatter separately;
- matmul and optimizer time;
- hot-set count and tail-on/tail-off comparison.

The analytical traffic estimate is useful for hypothesis generation, but it is not a substitute for a measured time breakdown.

## Review of each optimization

### Phase 0 — measurement harness: correct priority, incomplete specification

This is the right first step. The harness should be strengthened:

- use an actual CUDA profiler, not only analytical traffic;
- use CUDA events or synchronized timing per variant;
- reload the same checkpoint for every variant;
- use identical input bytes, B, ring K, hot set, and `TMT_TAIL` setting;
- reset peak memory statistics and release variant tensors;
- compare all relevant state, not only theta;
- report per-variant `loss`, `theta`, gradients, m/v, ring state, and tail state;
- report a cold-start and post-rebase case, not only the steady state;
- include a no-op graph/compile control to separate compilation overhead from steady-state speed.

“Each variant is a patch applied to the same GpuModel” is ambiguous. The same model object cannot be reused after a variant changes its state. Use a fresh model loaded from the same immutable checkpoint for each variant, or an explicit state snapshot/restore.

### A1 — skip hot columns: valid idea, wrong expected payoff

A1 is mathematically reasonable if all state updates remain unchanged and the cold credit is accumulated into the correct columns. It is not necessarily bit-identical because removing zero-valued atomic operations can change scheduling/order. The plan currently says Phase A must be bit-identical, which is too strong for a GPU scatter rewrite.

Change the acceptance language to one of:

- exact for the pure elementwise portion and within the established parity tolerance for the complete state; or
- a separate bit-exact CPU reference plus a CUDA numerical gate.

Measure the actual hot fraction and the actual speedup. Do not promise 1.1× from this item.

### A2 — fuse `ring_vals`: likely valuable, but the fallback is misdescribed

A fused multiply/scatter can remove a large temporary round trip. However, this K-loop:

```python
for k in range(K):
    index_add_(ring_b[:, k], w[:, k, :] * dlds)
```

still writes a temporary product and reads it back for each k. It reduces peak temporary memory and may improve locality, but it does **not** remove the total multiply-write/scatter-read traffic by itself.

The plan should distinguish:

- explicit K-loop: lower temporary peak, more launches, likely little traffic reduction;
- compiled/fused kernel: can remove the round trip, but may reassociate or change atomic behavior;
- custom fused CUDA/Triton kernel: strongest traffic reduction, highest correctness risk.

A2 should remain a benchmark candidate, but its expected 28 kB/s is not established.

### A3 — CUDA graph: potentially useful, but not low risk yet

CUDA graphs can reduce launch overhead, but the current step contains:

- Python loops over 16 layers;
- host-to-device input conversion in the trainer;
- `float(loss)` and `float(ce)` synchronizations;
- occasional tail membership rebase;
- in-place optimizer and ring updates;
- dynamic control flow around checkpointing.

A graph cannot remove the bandwidth cost or the atomic contention. It can only remove launch/synchronization overhead. Measure the launch fraction first. If it is only a few milliseconds, A3 should not be prioritized over A2/B1.

Also, the current `float(loss)` synchronization must be removed or isolated before graph capture; otherwise the graph cannot be captured around the full step.

### B1 — factored ring decay: promising, but state and checkpoint design must change

The mathematical idea is sound: store an undecayed ring and maintain a decay factor per slot. The plan’s wording should say that **each `(b,k)` slot is reset when overwritten**, which occurs once every K steps per stream. It is not a single global reset every K steps.

This is not just a GPU implementation change. It changes the meaning of persisted ring state. The current checkpoints do not contain ring buffers, and the proposed factored implementation would also need a per-slot decay factor. Without a format/version field, a checkpoint written by the old engine cannot safely be loaded by the factored engine.

Required changes:

1. add an algorithm/checkpoint format version to the NPZ;
2. either persist the factored ring and factors, or explicitly define a warmup/migration rule;
3. refuse loading old checkpoints into the new factored state unless an explicit migration is used;
4. keep the old engine available as the reference;
5. compare the new trajectory against the old engine with the ruler, not only CPU-vs-new-GPU parity.

Changing `numpy_port.py` to the factored algorithm would preserve CPU/GPU equivalence for the new algorithm, but it would no longer preserve the live old trajectory. That is a deliberate algorithm migration, not a pure optimization.

### B2 — factored tail decay: valid idea, but checkpoint state is incomplete

The tail has the same ordering problem. An undecayed E requires a per-stream/hot-column decay factor, and that factor must survive a checkpoint. The current checkpoint stores decayed E but no factored decay factor. A checkpoint written before B2 cannot be interpreted as factored state.

The plan must add:

- versioned E representation;
- persisted decay factors or an explicit migration;
- periodic renormalization state;
- tests across save/load at a non-renormalization step;
- CPU/GPU parity and ruler validation after the change.

### Phase C — sort/segment reduction: reasonable research direction, uncertain payoff

The plan is appropriately cautious here. A sort/segment reduction can reduce atomic contention, but it adds sorting, gather, and segmented-read traffic. It may not help if the measured `index_add_` cost is bandwidth rather than contention.

The current estimate of 151M atomic operations is useful, but the stated contention count should be recalculated. For one layer, the source count is about 9.44M and there are 256 target rows, or roughly 36,864 source contributions per target row on average, not 72. The exact contention depends on the byte distribution and kernel scheduling, so profile the actual shapes.

Do not build the fused Triton kernel until:

- isolated `index_add_` timing is recorded;
- a simple segmented reduction prototype is benchmarked;
- numerical parity is checked on CPU and CUDA;
- the checkpoint/ring state format is versioned.

## Problems in `docs/plans/PLAN-2026-09-23-GPU.md`

The older plan is directionally correct but is now partly stale and should be made a short parent/route document rather than a second optimization specification.

### Changes needed in the parent plan

1. **Make `docs/optimization/GPU_OPTIMIZATION_PLAN.md` authoritative for optimization.** Add a clear link and remove the duplicated §0.4 optimization table, especially the stale 22% A1 claim.
2. **Correct the plan’s optimization summary.** The parent plan still says the remaining work is mainly ring scatter; the live profile says the ring and its temporary/scatter path dominate, while A1’s 22% claim is wrong.
3. **Use the real live baseline everywhere.** 21.5–22.1 kB/s is the planning number; 32.4 kB/s is synthetic-only. The parent plan already warns about this, but its §1.2 still contains the old aspirational 5–10 ms/100–200 kB/s language. Remove or label that as obsolete.
4. **Update the corpus path contract.** The current fetcher publishes `out/current/`, while the Kaggle script defaults to `data/corpus/current/`. The parent plan should define one canonical path and require the Kaggle wrapper to use it.
5. **State the checkpoint limitation consistently.** The route plan should say “recoverable, not bit-identical” everywhere because ring state is not persisted.
6. **Do not call optimization acceptance “bit-identical” universally.** A1/A2/C can alter atomic order; use explicit state/parity thresholds and ruler deltas.
7. **Require an algorithm version for factored state.** B1/B2 cannot be added to the existing NPZ contract without a version/migration decision.
8. **Separate infrastructure gates from science gates.** CUDA parity proves implementation agreement, not model quality. The ruler/CE and held-out evaluation remain mandatory for B/C changes.
9. **Fix the secret-management instruction before cloud use.** The generated live-leg artifact currently embeds a credential; the parent plan should explicitly prohibit credential embedding and require rotation.
10. **Update completion status.** The live 30k leg is now recorded, so the parent plan’s “not yet run” items and old “first leg not run” wording need reconciliation with the current `RESULTS.md` and `docs/runbooks/SCHEDULE.md`.

## Recommended corrected optimization order

1. **Phase 0 profiler and reproducible benchmark harness.**
2. **A2 fusion experiment**, with a correct full-dimension mapping and a controlled K-loop baseline.
3. **B1 factored ring decay**, behind a versioned experimental flag, with old/new checkpoint and ruler migration tests.
4. **A3 graph capture**, only after measuring launch overhead and removing per-step host sync.
5. **A1 cold-column optimization**, measured at the actual hot fraction; likely small at H=115.
6. **B2 factored tail decay**, with persisted decay factors and a versioned checkpoint.
7. **Phase C atomics/segmented reduction**, based on isolated profiler evidence.

This order is different from the current plan because A1 is currently overestimated and B1 has a state-format implication.

## Required acceptance gates

For every optimization:

- CPU reference gate;
- CUDA/CPU state parity;
- finite/non-finite loss check;
- exact parameter and ring/tail state comparison;
- same checkpoint, same bytes, same B, same hot-set state;
- before/after throughput with synchronized timing;
- peak allocated and reserved memory;
- ruler result after the required resume warmup exclusion;
- new algorithm/checkpoint version where state representation changes;
- rollback checkpoint and commit tag.

For Phase A, use the strictest feasible bitwise/tolerance contract. For B/C, do not require bitwise equality if reassociation is intentional; require the documented numerical tolerance and a quality gate. The plan should not combine “bit-identical” and “float reassociation” as if both apply to the same phase.

## Final verdict

**The real problem is real:** GPU throughput is currently the practical bottleneck for consuming more data, and the ring-credit implementation is a legitimate optimization target.

**The plan needs changes before execution:** the A1 hot-dimension percentage and expected speedup are wrong, the traffic table needs reconciliation, A2/A3 risk levels are overstated, B1/B2 require new checkpoint/state versions, and the parent plan still contains stale throughput and completion language. Correct those items first; then execute the optimization plan behind flags and measured gates.
