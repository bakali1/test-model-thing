# SPEED — the kernel: the ring credit's atomic scatter (2026-10-21)

**Date:** 2026-10-21
**Status:** the legality analysis is DONE and gated; the candidate is
implemented and gated; the T4 measurement of the candidate is PENDING (the
free-tier T4s are busy — see §6).

> **One sentence:** the ring credit's cost is **not** mostly the atomics —
> it is mostly the bytes, and the bytes can be halved by fusing (which the
> shipped `nvrtc` path already does, 2.7x on the credit) — but the atomics
> that remain are **removable without changing a single float**, by grouping
> the rows by byte id and folding each group in the reference's own order.

**What is in here:** §1 the credit, §2 the legality analysis (the core),
§3 the removed `segmented` candidate, §4 the candidate that survives the
analysis, §5 the arithmetic per path, §6 the measurement, §7 the gates,
§8 what is still open.

---

## 1. the kernel: what the ring credit actually is

The engine keeps a ring buffer `w` of shape `(B, K, dim)`: for stream `b` and
lag `k`, `w[b,k,:]` is the embedding of the byte that stream `b` saw `k`
steps ago, decayed. `ring_b[b,k]` is that byte's **id**. The credit pays the
embedding's gradient to the bytes that caused it:

```
dest[ rows[b*K+k] , d ]  +=  w[b,k,d] * dlds[b,d]        for every (b,k,d)
```

`dest` is the embed gradient buffer, `(256, dim)`: 256 byte ids × 512 dims.

At the production shape (B=1024, K=18, dim=512, 16 layers):

| quantity | value |
|---|---:|
| ring elements per layer | 1024·18·512 = **9,437,184** |
| atomic element-adds per layer | **9,437,184** (every element) |
| atomic element-adds per step (16 layers) | **150,994,944** |
| destination addresses | 256·512 = **131,072** |
| mean contention per address | 9,437,184 / 131,072 = **72** |
| busiest address (realistic byte histogram) | **3,012** adds |
| ring bytes per layer (fp32) | 37.75 MB |
| ring bytes per step | **604 MB** (read) / 1,208 MB (read + the decay's write-back) |
| traffic floor at the T4's 320 GB/s | **1.89 ms** read / **3.77 ms** read+write |

The scatter is the engine's only *irregular* memory pattern: everything else
in the step is a dense matmul or a dense elementwise op.

### Where the time goes (the T4 profiler, `RESULTS.md` 2026-09-24, baseline)

Profiled wall 63 ms (51 ms unprofiled), 3 profiled steps:

| phase | ms/step | share |
|---|---:|---:|
| **ring_credit** | **21.3** | **34%** |
| bwd_layers | 15.7 | 25% |
| **tail_credit** | 10.2 | 16% |
| fwd_layers | 10.2 | 16% |
| decaytrace | 3.6 | 6% |
| optimizer | 3.0 | 5% |

and the device kernels behind them:

| kernel | ms/step | what |
|---|---:|---|
| `elementwise_kernel` (MulFunctor) | 13.7 | the ring decay, the `w*dlds` multiply, the tail's `E.mul_` |
| **`indexFuncLargeIndex ReduceAdd`** | **9.0** | **`dest.index_add_` — the atomic scatter (17 calls/step)** |
| `cudaLaunchKernel` (CPU) | 10.1 | 1,072 launches/step |
| `direct_copy_kernel` | 5.8 | `w[:, p] = 1.0`, the tail adds |
| **`masked_fill_kernel`** | **5.1** | **zeroing the 1.4% hot columns of the ring** |
| `sgemm` | 8.3 | the actual matmuls |

Read that table carefully, because it settles the framing: **of the ring
credit's 21.3 ms/step, the atomic scatter is 9.0 ms and the other 12.3 ms is
bytes and launches.** The `full` path makes FIVE passes over the ring per
layer (`w.mul_` = read+write, `w*dlds` = read+write, `index_add_` = read),
and two of them are re-reads of the same bytes. Fusing them is the bigger
win, and it is already shipped.

The MX150's own record says the same thing in its own numbers (RESULTS.md,
2026-10-03), per layer at B=1024, K=14, D=512 against a measured 39.1
GiB/s copy ceiling:

| per layer, MX150 | ms | vs bandwidth-optimal |
|---|---:|---|
| `w.mul_(dec)` | 1.455 | optimal (2 passes) |
| `torch.mul(w, dlds, out=vals)` | 1.456 | optimal (2 passes) |
| `dest.index_add_(rows, vals)` | **2.521** | **3.5x worse** (atomics) |
| **×16 layers** | **86.9 ms of 195** | **45% of the step** |

So the atomics are 40 ms/step of a 195 ms step on Pascal — a fifth of the
step — and the bytes are the other half. Both are attackable; only one of
them is attackable *without touching the arithmetic*.

---

## 2. the legality analysis (the core)

### 2.1 What the reference computes, and in what order

The reference is `numpy_port.py`'s `_scatter_credit` (numba, or the BLAS
fallback):

```python
for i in range(n):            # n = B*K rows, in increasing index
    r = rows[i]
    for d in range(dim):
        dembed[r, d] += vals[i, d]
```

So for a fixed address `(r, d)`, the reference's value is the **left fold, in
increasing row index**, of the values of the rows that map to `r`:

```
dest[r,d] = (((dest[r,d] + v_{i1}) + v_{i2}) + ... )      i1 < i2 < ...
```

**MEASURED, not assumed** (`tests/check_ring_grouped.py`): torch's
`index_add_` on CPU is **bit-identical** to that walk, and to numpy's
`np.add.at`. So the order is well defined, and there is a float to preserve.

### 2.2 The illegal reduction (the one a naive implementation reaches for)

The obvious way to cut the atomics is to notice that the credit is a sum over
`(b, k)` and reduce **the K axis in registers**, issuing one atomic per
`(b, dim)`: 9.44M → 524K atomics per layer, an 18x cut.

**This is wrong, and not by a rounding.** `rows[b,k]` is the byte id that
stream `b` saw `k` steps ago, so the K lag terms of one `(b,d)` land on **K
different byte ids**. MEASURED: 99.9% of the lag pairs of a stream hold
different byte ids (1 of 896 coincide in the gate's draw). Summing them
pays one byte's credit onto another byte; the gate's control shows the
result landing on the wrong bytes with `max|Δ| = 2.6e+01`.

**The reduction axis of this scatter is the BYTE ID. Never the lag.** The K
axis is a *history* axis, not a summable axis: the K terms of `(b,d)` are K
different payments to K different creditors.

### 2.3 The legal reductions

| transformation | legal? | why |
|---|---|---|
| reduce over the **lag** axis (one atomic per `(b,d)`) | **NO** | the terms belong to different byte ids (§2.2) |
| reduce over the **dim** axis | **NO** | each dim is a different destination address |
| sum the group's values with a **tree / `segment_reduce`** | **NO** | reassociation changes the float: `(d+v1)+v2 ≠ d+(v1+v2) ≠ ((v1+v2)+v3)`. MEASURED on 72 values: the left fold and the pairwise sum differ by 1.9e-06 (1.1e-07 relative) — a different float, and the project's rule is "not a single float" |
| start the fold from **0** instead of the destination's current value | **NO** | the destination already holds the previous bytes' and layers' credit; `(d+v1)+v2 ≠ (v1+v2)` |
| add the group's values in a **different order** within the group | **NO** | same reason: the fold's order is the reference's order |
| **group the rows by byte id**, fold each group left in increasing row order, write the address **once** | **YES** | each address is touched by one group; the group's values are added in exactly the reference's order, starting from exactly the reference's starting value. The result is the same float, bit for bit |
| permute the **addresses** (which group is computed first) | **YES** | the addresses are independent |
| fuse the passes (decay, multiply, scatter in one kernel) | **YES** | per-element arithmetic, unchanged order — this is what `compiled`/`triton`/`nvrtc` already do, and it is why the ring buffer stays bit-identical |

### 2.4 The one thing that makes the legal move possible

The reference's order is the **row** order, and a group of rows (the rows
that share a byte id) is a *subsequence* of the rows. So the group's values
must be added in increasing row index — and the natural way to find the
groups (a sort by byte id) destroys that order unless the sort is **STABLE**.
A stable sort keeps the rows of a group in their original relative order,
which is exactly what makes the grouped fold the reference's fold.

This is the whole difference between the candidate that works (§4) and the
one that was removed (§3).

### 2.5 The cost of the grouping

The plan is: a stable sort of the `B*K = 18,432` byte ids, a count per byte
id, and the offsets. Per byte, i.e. **once per step**, not once per layer
(the 16 layers share the ring's byte ids):

| item | size | cost |
|---|---:|---|
| the sort (18,432 int64 keys) | 147 KB | one small radix sort: ~30–60 µs on a T4 |
| the counts + offsets | 2 KB | two tiny kernels |
| the per-row offsets the kernel wants (`w` offset, `dlds` offset) | 2 × 73 KB | two elementwise ops |
| **total per step** | ~0.3 MB | **~0.1 ms, i.e. 1/16 of a layer's credit** |
| **amortized per layer** | | **~6 µs** |

Against what it removes: 9,437,184 atomics per layer, 9,437,184 → **0**.
The grouping is ~2% of what it removes, and it is paid once per step instead
of once per layer.

### 2.6 What the atomics actually cost (the arithmetic)

Per layer, at the production shape, on a T4 (320 GB/s):

| | bytes moved | floor | measured | the gap |
|---|---:|---:|---:|---:|
| `full` | 5 passes = 189 MB | 0.59 ms | **1.285 ms** | 0.70 ms: the atomics + the extra launches |
| `compiled` | ~2 passes + atomics | ~0.25 ms | **0.671 ms** | ~0.42 ms |
| `triton` | 2 passes + atomics | ~0.25 ms | **1.024 ms** | ~0.77 ms |
| `nvrtc` | 1 pass (read+write) + atomics | 0.244 ms | **0.475 ms** | **~0.23 ms: the atomics** |
| `grouped` | 1 pass (read+write) | 0.244 ms | **pending** | the target is ~0 |

So on a T4 the contended fp32 atomics cost **~0.23 ms per layer, ~3.7 ms per
step**, and the ceiling for a contention-free credit is the traffic floor of
**3.77 ms per step**. The `nvrtc` path is already within 2x of that floor;
the grouped path's whole prize is the remaining 3.7 ms/step.

**The hypothesis in the task brief — "contended fp32 atomics are the wall" —
is half right and half wrong, and the measurement says which half.** The
atomics are a fifth of the step's ring-credit phase and ~7% of a 51 ms step
(the profiler's unprofiled wall); the *bytes* are the other four fifths of
the credit's cost, and they are what the shipped fused paths already fixed.
The MX150 is where the atomics hurt most in relative terms (2.52 ms/layer,
3.5x the bandwidth-optimal, 21% of the step) because its memory system is 8x
slower while its L2 atomics are no faster.

---

## 3. the removed `segmented` candidate: where it was measured, and why it lost

`git log -p --all -- gpu_port.py | grep -A40 segmented` → the candidate is in
commit `d2b6341` ("variant verification") and was removed in `c41f457`
("ship the +27% ring path as the default; remove everything that lost"). Its
implementation:

```python
if mode == "segmented":
    vals = self.ring_vals.reshape(-1, self.dim)
    counts = self._segment_lengths
    summed = torch.segment_reduce(
        vals.index_select(0, self._segment_order), "sum", lengths=counts)
    self._ring_dest.add_(summed)
```

with `self._segment_order = torch.argsort(rows, stable=True)` and
`self._segment_lengths = torch.bincount(rows, minlength=256)`, rebuilt every
step ("Rows change every byte; caching this order across steps is a silent
correctness bug").

**WHERE IT WAS MEASURED — the brief's premise is wrong.** The 0.81x is in
`RESULTS.md`'s **"Board run 4"** table, and the Phase-0 board measures it at
**0.71x**; both boards are the **T4** board (`RESULTS.md`: "Phase-0
optimization board — MEASURED on T4 (2026-09-24)", "Board run 3 + the
measurement noise floor (T4)"). The MX150 work is from **2026-10-03**, a
month *after* the candidate was removed, so the MX150 never saw it. The
claim "it lost on the MX150, where the atomics are not the wall" is not
supported by the record: **it lost on the T4, twice, at 0.71x and 0.81x.**

**WHY IT LOST, and it is not "the MX150 hid it" — it loses on every GPU,
and it loses WORSE on the faster one.** The candidate trades 9.44M atomics
for a full extra round trip over the `(B,K,dim)` tensor:

| step | bytes | T4 (320 GB/s) | MX150 (39.1 GiB/s) |
|---|---:|---:|---:|
| `index_select` (the gather into sorted order) | read 37.7 + write 37.7 MB | 0.24 ms | 1.9 ms |
| `segment_reduce` (the grouped sum) | read 37.7 MB | 0.12 ms | 0.95 ms |
| the sort of 18,432 keys | ~0.1 MB | 0.02 ms | 0.1 ms |
| **the traffic it ADDS** | **~113 MB** | **~0.38 ms** | **~3.0 ms** |
| the atomics it REMOVES | | **~0.23 ms** | **~1.8 ms** |
| **net** | | **−0.15 ms (worse)** | **−1.2 ms (worse)** |

and the isolated micro-benchmark on the T4 agrees: `index_add_` **0.320 ms**
vs **sort+segment-sum 0.661 ms** — the segmented form is **2.06x slower**
doing the same arithmetic. The reason is structural: the segmented form
moves the data three more times to avoid an operation that does not move
data at all. On a faster GPU the *same* extra traffic buys less time while
the atomics it removes were already cheap — so the trade gets worse, not
better, as the GPU gets faster. (This is the mirror image of the brief's
hypothesis, and it is the reason "the MX150 would have liked it" is exactly
backwards.)

**And it was not bit-identical anyway.** `segment_reduce("sum")` does not
promise a left fold: it reassociates, so it is a different float (the
`RESULTS.md` entry for `c-segmented` carries a 1.8e-03 state RMS, 10x the
1.3e-04 of the paths that were kept). The `segmented` candidate therefore
failed BOTH tests: it was slower AND it moved the float. The candidate in
§4 is the same idea with the two differences that matter — the group's
values are combined **in the reference's order** (so the float is
identical), and they are combined **in the kernel that is already reading
the ring** (so no extra traffic is added).

---

## 4. the candidate that survives the analysis: the grouped credit

`ring_credit="grouped"`. `gpu_port.py`, `_NVRTC_GROUP_SRC` + the path
plumbing; the portable reference implementation is
`_ring_credit_grouped_torch`.

**What it does.** Each thread owns exactly one `(byte id, dim)` address. It
reads the destination, then folds that address's group of rows **in
increasing row order**, then writes the address **once**:

```cuda
float s = dest[r*D + d];                       // the reference's starting value
for (j = 0; j < m; ++j) {                     // the group's rows, in row order
    float wv = w[woff[j]*D + d] * dec[d];
    w[woff[j]*D + d] = wv;                     // the decay, in place
    float v = wv * dlds[dlsoff[j] + d];
    if (has_mask && hotmask[d]) v = 0.0f;
    s += v;                                   // LEFT FOLD -- the reference's order
}
dest[r*D + d] = s;                            // one write, no atomic
```

* **zero atomics** per layer (the four other paths issue 9,437,184);
* **no extra traffic**: the kernel reads the ring exactly once and writes it
  back once (the decay), which is the same traffic the fused `nvrtc` path
  moves — the *plan* costs ~0.1 ms per step, 1/16 of a layer;
* **the same float**: the fold is left, in row order, from the destination's
  current value, so it is the reference's fold, term for term;
* the destination is written by exactly one thread per address, so there is
  no read-modify-write race and no second writer.

The rows of a group are found with a **stable** sort of the byte ids (§2.4),
so the group's rows keep their relative order.

**Fallbacks.** On CPU the path uses `_ring_credit_grouped_torch`, the same
fold expressed in torch (a loop over rounds, each round adding every
address's next row) — slow, but it is the reference semantics, and it is
what makes the kernel testable without a GPU. If the NVRTC kernel cannot be
compiled, the path falls back to `full` and says so.

**Gated** by `tests/check_ring_grouped.py` (new): the grouped credit is
bit-identical to `full`, to numpy's own scatter, and to a 12-step engine
run's `theta`/`m`/`v`; and the gate also pins the two illegal moves (the lag
reduction, the reassociation) so nobody re-introduces them by accident.

**The default path is untouched.** `auto` still resolves to `compiled` on
CUDA and `full` elsewhere.

---

## 5. the arithmetic per path (at B=1024, K=18, dim=512, 16 layers)

| path | atomic element-adds / layer | per step | plain RMW writes / step | passes over the ring | notes |
|---|---:|---:|---:|---:|---|
| `full` | 9,437,184 | 150,994,944 | — | 5 | the portable reference |
| `compiled` | 9,437,184 | 150,994,944 | — | ~2 + scatter | Inductor; `auto` on CUDA |
| `triton` | 9,437,184 | 150,994,944 | — | 2 + scatter | needs cc ≥ 7.0 |
| `nvrtc` | 9,437,184 | 150,994,944 | — | 1 (read+write) | no cc floor |
| **`grouped`** | **0** | **0** | 256·512·16 = 2,097,152 | 1 (read+write) | the candidate |
| `segmented` (removed) | 0 | 0 | 2,097,152 | **4** | +113 MB/step: why it lost (§3) |

Traffic floor: 604 MB/step read, 1,208 MB/step read+write → **1.89 ms** and
**3.77 ms** at 320 GB/s.

---

## 6. the measurement

**Harness** (`tools/bench_ring_credit.py`), which measures the credit ALONE
(so the step time is not diluted by everything else) and prints the atomic
arithmetic next to the measurement:

```
.venv/bin/python tools/bench_ring_credit.py --batch 1024 --dim 512 \
    --layers 16 --ring-k 18 --iters 50
```

It reports `unavailable` (not a crash) for a path the device cannot run —
Triton and Inductor need compute capability ≥ 7.0, which is why the MX150
(a cc 6.1 Pascal) can only run `full` and `nvrtc`. That fallback is part of
the story, so it must not be a crash.

**The harness had two defects that would have made its numbers wrong**, and
both are fixed:

1. it never ran at all — `_ring_dest` is allocated by `step_batch`, and the
   harness calls the credit outside a step, so every path died with
   `AttributeError: 'GpuModel' object has no attribute '_ring_dest'`;
2. the ring's byte ids were left at their initial value (**all zeros**), so
   all 18,432 rows addressed byte id 0: **3,012× the real contention** on
   one address (the real mean is 72, the real busiest address takes 16.3%
   of the rows in a Zipf-like byte histogram). The byte ids now come from a
   realistic histogram, and the contention is printed beside the timing.

### T4 (Tesla T4, cc 7.5, torch 2.11+cu130), B=1024, K=18, dim=512, 16 layers

| path | ms/call | ms/step (×16) | atomic adds/step | G adds/s | adds/address |
|---|---:|---:|---:|---:|---:|
| `full` | 1.285 | 20.56 | 150,994,944 | 7.34 | 72 |
| `compiled` (the default on CUDA) | 0.671 | 10.74 | 150,994,944 | 14.07 | 72 |
| `triton` | 1.024 | 16.38 | 150,994,944 | 9.22 | 72 |
| `nvrtc` | **0.475** | **7.61** | 150,994,944 | 19.87 | 72 |
| `grouped` | **pending** | | **0** | — | 72 (folded in-register) |

Traffic floor for the same work: 604 MB/step read → 1.89 ms at 320 GB/s;
1,208 MB/step read+write → 3.77 ms.

**All four existing paths agree BIT-FOR-BIT on the T4** (max|Δ| = 0.0 against
`full`, on the same device, in the same process) — including `full` against
itself. That is a stronger result than the MX150's record, which found the
same path differing from itself by 8.8e-07 of RMS over a 150-step run: the
atomic order is the hardware's, so "bit-identical" is a CPU claim, and on a
GPU it holds for a single call and not necessarily for a run. The harness
now prints `full vs full` so the difference between those two claims is
visible on the device the measurement runs on.

**Reading the table.** The credit is 20.6 ms/step in the baseline (`full`,
which matches the profiler's 21.3 ms/step for the ring-credit phase) and
7.6 ms/step with the best shipped path. The `grouped` path's target is
**~4 ms/step** (the traffic floor plus the fold) — i.e. it would take the
credit from 7.6 to ~4-5.5 ms/step, and from the *default* path's 10.7 to
~4-5.5 ms/step. On a 51 ms step that is 10-13% of the step, and it is the
LAST of the credit's costs: after it, the credit is at its byte floor and
no further kernel can help.

**Pending:** the `grouped` row. The free-tier T4s were busy (one running the
P4-vs-FSP A/B), and the allocation of a second one was refused repeatedly.
The kernel compiles (checked for `compute_75` by `tools/check_nvrtc.py`,
which compiles CUDA to PTX with no GPU at all), and its arithmetic is gated
on CPU; what is missing is its wall-clock on a T4.

---

## 7. the gates

| gate | result |
|---|---|
| `tests/check_parity.py` | **PASS** (numpy ≡ torch; the T=1 bit-identity arm: 114 comparisons, 0 differing elements) |
| `tests/check_safety.py` | **PASS** (8/8) |
| `tests/check_seqlen_credit.py` | **PASS** (worst embed L1-mass deviation 1.2e-07, worst elementwise 6.5e-06) |
| `tests/check_ring_grouped.py` (new) | **PASS** (the legality argument + the bit-identity of the candidate) |
| `tools/check_nvrtc.py` (new) | both credit kernels compile for `compute_75` |

The default path is unchanged: `auto` → `compiled` on CUDA, `full` on CPU,
and no number on the default path moved.

---

## 8. what is still open

1. **the `grouped` path's T4 wall-clock** (§6) — the one number missing.
2. **the plan's cost on the T4** — the sort of 18,432 keys per byte; the
   arithmetic says ~0.1 ms/step, but it should be measured, and if the sort
   is expensive the plan can be built once per byte *and* reused by the 16
   layers (it is, by construction) or replaced by a counting sort.
3. **whether the grouped path is worth the default** — it is a new default
   only if it is measurably faster than `nvrtc` on a T4 *and* the bit-identity
   claim survives a full training run (the MX150's 8.8e-07 run-to-run spread
   says a GPU run cannot promise bit-identity anyway; the claim that can be
   made is "within the reference's own spread", which is the standard the
   `nvrtc` path was adopted under).
4. **the other half of the credit's cost is not the credit's fault**: the
   `masked_fill` (5.1 ms/step) that zeroes the 1.4% hot columns of the ring,
   and the 1,072 launches/step (10.1 ms of CPU time). Both are bigger
   prizes than the atomics, and both are outside this investigation.
