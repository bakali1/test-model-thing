# P3 gate — obstacle analysis (2026-10-22)

**Status:** analysis only. No repo file other than this one was changed; no
training, no long job, no commit.

**Question.** `PLAN-2026-10-21-CONTINUAL-LEARNING.md §6 D5` found the recurrent
state holds a *persistent, strongly addressable address* (the cue key decodes
65–165× over its own null and holds ~27 % at 128 B of lag) and a *decaying but
real content* (the payload decodes 4.6× over null at 8–12 B, 1.5× at 128 B),
and that **the address survives where the content decays**. That is the
signature of an order-losing weighted average. The plan therefore moved the
lever from the read's wiring (M1) to **the write's selectivity (the P3 gate)**.

The gate is implemented, its gradient is implemented and pinned
(`tests/check_gate_grad.py`), and the arm **refuses to run** at `step_batch`
(`gpu_port.py:1847`, `numpy_port.py:450`). This document says exactly what
blocks it, item by item, and what the smallest safe change is.

---

## 0. TL;DR

1. **The refusal's stated reason is half wrong.** The plan and the code both
   say the ring credit, the tail and the decaytrace "all assume a per-dim
   `(dim,)` decay vector". The *shape* part is true and trivial (broadcast
   changes). The *substance* — "the ring's stored value and the credit's decay
   no longer agree" (`PLAN-2026-10-11-T-AXIS.md §2 P3`) — **does not survive a
   direct numerical check**: the ring's per-step `w *= a` **is** the exact
   survival product for a time-varying, per-sample retention. Measured, §2
   below: max rel dev **0.000e+00** against the explicitly enumerated
   per-sample products.
2. **The real blocker is the TRUNCATION BOUND, not the shape.** The ring is
   allowed to drop the credit beyond lag K because `σ_d^K ≤ RING_TOL = 1e-4`
   for every dim (`gpu_port.py:1197`). A two-sided per-sample gate can push the
   effective retention `a` for *any* dim arbitrarily close to 1, so that bound
   evaporates: with `a = 0.99` and `K = 14`, the ring silently drops **87 % of
   the credit mass** for that (stream, dim). The hot set
   (`hot_from_np`, `gpu_port.py:144`) is what protects the hot dims, and it is
   computed from the **decay logits**, i.e. from the ungated `σ_d` — so it is
   per-dim and *input-independent*, which is exactly the invariant that a
   two-sided gate destroys.
3. **There is already an exact credit path in the repo that has no such
   bound.** The dense trace (`trace_mode="dense"`, `gpu_port.py:2166-2186`) is
   the tail with every dim hot; `tests/check_dense_trace.py` shows it is
   **bit-identical** (max rel dev `0.00e+00`) to ring+tail. It has **no
   `ring_k`, no hot set, no mask, no rebase and no truncation**, so it is exact
   for *any* per-sample `a` — including a two-sided gate — with a **one-line
   broadcast change**.
4. **The gate gradient needs no change at all.** `gate_trace_update`
   (`gpu_port.py:374`) already handles `a` of shape `(dim,)` and `(B, dim)`, and
   `check_gate_grad.py` already pins it against torch autograd **and** a
   central finite difference. It is already per-sample-correct.
5. **Recommendation (§5):** implement the gate fully, two-sided, with the
   credit re-derived on the **dense trace** — **~10 lines in `gpu_port.py`**,
   plus a dense path in `numpy_port.py` for the reference/parity gate
   (~35 lines), plus a **new gate** that checks the gated `step_batch`'s full
   gradient against autograd (~120 lines, the test that does not exist today
   and the one whose absence is why the arm refuses). Memory is the dense
   bill, `B*256*dim*4` per layer — **the same bill the P4 ring already pays**,
   because `ring_k` saturates at 256 under the P4 init (`RESULTS.md`, "The P4
   init has the memory bill the plan priced for the dense trace"). The default
   engine and `check_parity` are untouched.
6. **The runner-up, and the trigger to switch to it (§5.3):** a **one-sided
   gate** (`a ≤ σ_d`, i.e. the gate may only *shorten* retention) on the
   existing ring path. It preserves the hot set, the mask and the `ring_k`
   inference *exactly* (because `a_{b,d} ≤ σ_d` keeps `a^K ≤ σ^K`), so the
   whole credit machinery keeps working — but it changes the pinned gate
   gradient (a clamp indicator enters the trace) and it restricts the
   mechanism to "flush, never extend".

---

## 1. What the per-dim assumption actually is

There are **two** assumptions hiding inside "the decay is a per-dim `(dim,)`
vector". They have very different costs.

**(A) SHAPE.** The retention is a vector over dims, broadcast over the batch.
*Cost of breaking it:* every consumer must broadcast a `(B, dim)` instead of a
`(dim,)`. Every one of those sites is a **broadcast**, not an arithmetic change:
`(dim,)` → `(1,1,dim)` → `(1,dim)` and `(B,dim)` → `(B,1,dim)` → `(B,dim)` are
the same elementwise product. **Where it bites:** the three CUDA/numba
*kernels* index the decay by **dim only** through a raw pointer, so they need
an index change (`dec[d]` → `dec[b*D+d]`); and the numpy reference needs an
explicit `[..., None, :]` because numpy broadcasts from the right.

**(B) TIME-INVARIANCE.** The retention is the *same at every step*, so that
"multiply the buffer by today's σ" reproduces the survival product. *Cost of
breaking it:* **zero, if the buffer is multiplied once per step** — because the
product accumulates. This is the part the plan believes is broken, and it is
not.

### 1.1 Why (B) is not broken — the algebra

The engine's ring holds, per stream, the survival weight of the byte written
`k` steps ago, and multiplies the whole buffer by the current retention once
per step:

```
w ← w * a_t          (every step, per stream)
w[:, p] ← 1.0        (this byte becomes lag 1)
```

Unroll it: the slot that held the byte written at step `s` carries
`Π_{j=s+1..t} a_j` at step `t` — **exactly** the survival product
`∂s_t/∂e_s` of the recurrence `s_t = a_t s_{t-1} + e_t`, for *any* sequence
`a_1 … a_t`, per-sample or not. Same for the tail (`tail_credit.py:93-160`):
`E ← E * a_t; E[b,c_t,:] += 1` expands to
`E[b,v,d] = Σ_{s: c_s = v} Π_{j=s+1..t} a_{b,j,d}` — the exact survival-weighted
embedding trace. The dense trace is the same recursion with all dims hot.

So the ring and the tail are **not** approximations of the credit under a
per-sample gate; they are the *same recursion as the state itself*. The one
thing they approximate is the **lag horizon** (the ring drops bytes older than
`K`), and that approximation is governed by the retention's magnitude, not by
its shape.

### 1.2 Why (B) is *nearly* broken — the one real consequence

The ring's truncation is bounded by `a^K`. The engine's contract
(`RING_TOL = 1e-4`, `gpu_port.py:36`) is: for every dim, `σ_d^K ≤ 1e-4`, i.e.
the credit dropped beyond lag `K` is ≤ 0.01 % of the mass. The **hot set**
(`HOT_FRAC = 0.01`) moves the dims whose `σ_d^K > 1 %` off the ring and onto
the tail, which is exact at every lag.

A two-sided per-sample gate makes `a_{b,d}` **input-dependent**, so:

* `σ_d^K ≤ RING_TOL` no longer bounds `a_{b,d}^K` — it can be `≈ 1`;
* the hot set, computed from the **decay logits** (`hot_from_np(sigmas, ...)`,
  `gpu_port.py:144`; `tail_credit.py:181-183`), is per-dim and stale by up to
  `REBASE_EVERY` steps — a dim that just became hot is still credited by the
  ring *and* by the tail (double count) and, worse, its ring credit is
  truncated at `K` with up to ~87 % of the mass missing.

This is a **correctness** requirement, not a performance choice, and it is the
one thing the refusal's comment does not name.

**Two ways out, and only two:**

* **give the effective retention an input-independent per-dim ceiling**
  (`a_{b,d} ≤ σ_d`) — the one-sided gate (§4a′); the hot set, the mask and
  the `ring_k` inference then all stay *exactly* valid;
* **remove the truncation** — the dense trace (§4c′); no `ring_k`, no hot set,
  no mask, no rebase, no truncation.

---

## 2. The measurement that settles §1.1

Reproduced with a faithful transcription of the engine's own ring and dense
branches (`w *= dec[:, None, :]`; `E *= dec[:, None, :]`; `E[b,c,:] += 1`),
against a ground truth that enumerates the per-sample survival products
explicitly, byte by byte. `B=3, D=5`, `a ~ U(0.3, 0.99)` **per stream, per
step, per dim** (i.e. exactly what the gate produces).

| regime | ring vs truth | dense vs truth | ring vs dense |
|---|---|---|---|
| `T=12, K=16` (no truncation) | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| `T=12, K=8` (truncation) | 5.213e-02 | **0.000e+00** | 5.213e-02 |
| `T=20, K=16` (truncation) | 3.811e-04 | 1.792e-08 | 3.811e-04 |

Reading: with a **per-sample, time-varying** retention and no truncation, the
ring's credit is **bit-exact** against the explicitly enumerated survival
products. The only disagreement is the ring's lag-`K` truncation, which is the
pre-existing, documented approximation — and whose *size* is what the gate
breaks (§1.2). The dense trace is exact in every regime.

Reproduce (≈30 lines, CPU, < 1 s):

```python
import numpy as np
rng = np.random.default_rng(0); B, K, D, T = 3, 16, 5, 12
w = np.zeros((B, K, D), np.float32); ring_b = np.zeros((B, K), np.int64)
E = np.zeros((B, 256, D), np.float32)
A = rng.uniform(0.3, 0.99, (B, T, D)).astype(np.float32)
cs = rng.integers(0, 256, (B, T)); dlds = rng.standard_normal((B, D)).astype(np.float32)
dr = np.zeros((256, D)); dd = np.zeros((256, D))
for t in range(T):
    a = A[:, t, :]
    w = w * a[:, None, :]; vals = w * dlds[:, None, :]
    np.add.at(dr, ring_b.ravel(), vals.reshape(B*K, D))
    p = t % K; w[:, p, :] = 1.0; ring_b[:, p] = cs[:, t]
    E = E * a[:, None, :]; dd += np.einsum("bvd,bd->vd", E, dlds)
    E[np.arange(B), cs[:, t], :] += 1.0
dt = np.zeros((256, D))
for b in range(B):
    for t in range(T):
        for s in range(t):
            prod = np.ones(D, np.float32)
            for j in range(s+1, t+1): prod = prod * A[b, j, :]
            dt[cs[b, s]] += prod * dlds[b]
rel = lambda x, y: float(np.abs(x-y).max() / (np.abs(y).max() + 1e-12))
print(rel(dr, dt), rel(dd, dt))     # -> 0.0 0.0
```

---

## 3. Itemised inventory of every per-dim assumption

Classification key: **(i)** correctness requirement — the site is wrong if the
retention stops being per-dim / per-sample / time-invariant; **(ii)** performance
choice — it could be relaxed at a cost; **(iii)** implementation convenience —
it is incidental and the fix is mechanical.

### 3.1 The forward — the retention itself

| # | site | what it assumes | class | what breaks |
|---|---|---|---|---|
| F1 | `gpu_port.py:322` `a = _sig(self.decay + self.gate * x_in)` (TorchLayer.forward) | nothing per-dim: the gated forward **already returns `(B, dim)`** | (iii) | nothing — the forward is already per-sample-ready. `gate = 0` ⇒ `0.0 * x_in` is an exact zero ⇒ `a == sigmoid(decay)` bit-for-bit, which is what makes the A/B resumable. |
| F2 | `numpy_port.py:135-140` `a = sigmoid(self.decay[None,:] + self.gate[None,:] * x_in)` | same | (iii) | nothing; already mirrors F1. |
| F3 | `gpu_port.py:257` / `numpy_port.py:141` `state = states * a + enc` | the write weight is **1.0** | (iii) | nothing. (The plan's `s = a*s + (1-a)*x` form would break the bit-identity and is deliberately not used.) |

### 3.2 The ring credit path — `gpu_port.py:_scatter_ring_credit` (1534-1650)

| # | site | what it assumes | class | what breaks |
|---|---|---|---|---|
| R1 | `gpu_port.py:1623` (compiled path) `w.mul_(dec[None, None, :])` | decay broadcast over `(B,K,dim)` | (iii) | **nothing arithmetically.** `dec[:, None, :]` is the same elementwise product. Bit-identical in the ungated case. |
| R2 | `gpu_port.py:1630` (triton path) `w.mul_(dec[None, None, :])` | same | (iii) | same as R1. |
| R3 | `gpu_port.py:1640` (`full`, the portable reference) `w.mul_(dec[None, None, :])` | same | (iii) | same as R1. This is the path the bit-identity gates compare against, so it is the reference for any kernel change. |
| R4 | `gpu_port.py:1583-1585` `if dec.dim() != 1: mode = "full"` (the grouped path's guard) | the grouped kernel needs a per-dim decay | (iii) | **already handled**: the code anticipates a per-sample `dec` and degrades to the portable path. Evidence that the change is smaller than the refusal implies. |
| R5 | `gpu_port.py:893-910` `_ring_credit_grouped_torch`, `w.mul_(dec[None, None, :])` at 906 | same | (iii) | same as R1; the path already refuses per-sample `dec` via R4. |
| R6 | `gpu_port.py:442-465` `_NVRTC_RING_SRC` — `float wv = w[idx] * dec[d];` with the comment *"dec is (D,), NOT (B,D)"* | the raw pointer is indexed by **dim only** | **(i)** | **the fused nvrtc path is a hard blocker for a per-sample `dec`.** Fix: `dec[b*D+d]` plus a contiguous `(B,D)` decay on the call site. ~3 lines of CUDA. **Risk:** this is the 1.26× end-to-end winner path, and a wrong index is a silently wrong gradient, not a crash. Needs a bit-identity gate against the `full` path. |
| R7 | `gpu_port.py:639-720` the grouped nvrtc kernel — `wv[u].x *= dec[d0];` (691-692, 711-712) | same | (i) | same as R6, for the `float4` and scalar variants (`d0` → `b*D + d0`). |
| R8 | `gpu_port.py:757` `const float wv = w[(long)p.x + d] * dec[d];` | same | (i) | same as R6. |
| R9 | `gpu_port.py:1653-1670` `_ring_hot_mask_arg`; the kernel's `hotmask[d]` (461); the portable `masked_fill_(hot_mask[None, None, :], 0.0)` (1642-1643) | the hot mask is a per-**dim** bool | **(i) for a two-sided gate; (iii) for the one-sided gate** | under a two-sided gate the hotness is per-**stream**, so the mask must be `(B, dim)`: `hotmask[b*D+d]` in the kernels, `hot_mask[:, None, :]` in the portable path. Under the one-sided gate (§4a′) the mask stays per-dim and **nothing changes**. |
| R10 | the ring's **truncation contract**: `ring_k` inferred so `σ_max^K ≤ RING_TOL` | the retention is bounded by the per-dim `σ_max` | **(i)** | **THE real blocker.** `gpu_port.py:1197`, `numpy_port.py:244`. A two-sided gate has no per-dim ceiling on `a`. See §1.2 and §5.2. |

### 3.3 The tail credit path — `gpu_port.py:TorchTail` (95-215), `tail_credit.py:TailCredit` (165-300)

| # | site | what it assumes | class | what breaks |
|---|---|---|---|---|
| T1 | `gpu_port.py:228` `E.mul_(sig_hot[None, None, :])` (begin_step) | `sig_hot` is `(H,)`, broadcast over `(B,256,H)` | (iii) | **nothing arithmetically** — `sig_hot[:, None, :]` is the same product. The tail's `E` is already per-stream. |
| T2 | `gpu_port.py:283` `E.mul_(sig_hot[None, None, :])` (the unfused fallback) | same | (iii) | same as T1. |
| T3 | `gpu_port.py:258-262` the nvrtc fused tail kernel `tmt_tail_fused` (`args=[E, sig_hot, g, out, B, V, H]`) | the raw pointer is indexed by **hot-dim only** (`sig[h]`) | **(i)** for the fused tail path | the kernel must index `sig[b*H+h]`; ~4 lines of CUDA + a contiguous `(B,H)` argument. The fallback (T2 + the einsum) is always correct — the kernel is a speed item, not a correctness item, *provided the fallback is used when the kernel is unavailable*. |
| T4 | `tail_credit.py:93-160` the numba kernels `_tail_decay` / `_tail_credit` / `_tail_fused_decay_credit` (`ev[h] *= sig[h]`, `d = E[b,v,h] * sig[h]`) | same | **(i)** for the fused path | same as T3; the numba path is the CPU's "fused" equivalent and the CPU parity path. |
| T5 | `gpu_port.py:144-145` `hot_from_np(sigmas, ring_k, hot_frac) = sigmas ** ring_k > hot_frac`; `gpu_port.py:169-175` (mask + cols); `tail_credit.py:181-183` `hot_from` | the hot set is a **per-dim** function of the **decay logits** | **(i)** | **The second half of the real blocker.** Under a two-sided gate, hotness is per-stream and time-varying. Either the set becomes the union over the batch and is refreshed every step (cost: a per-step rebase, which `gpu_port.py:1912-1914`'s comment says was 99 % of GPU step time when it ran per step), or the retention gets a per-dim ceiling. |
| T6 | `tail_credit.py:186-224` `sync()` — rebuilds the hot set every `REBASE_EVERY` steps, adopts/drops dims, seeds newly-hot dims from the ring | the hot set changes slowly | **(i)** under a two-sided gate | a stale hot set double-counts a dim that has just become hot (ring + tail) *and* leaves a dim hot-by-σ cold-by-`a` on the ring where it is truncated at `K`. |
| T7 | `tail_credit.py:225-240` `_seed_from_ring` (`np.add.at(E, (bidx, ids[:,k,None], hh), w[:,k,:])`); `gpu_port.py:205-218` `TorchTail.sync` (`ring_ws_t[l][:, :, adopt_t]`, `ring_b_t[:, k]`) | the ring's weights are per-stream | **(iii)** | **nothing** — both seed the tail's `E` from the ring's *per-stream* weights, which under a per-sample `a` already carry the per-sample survival products. Correct as written. |
| T8 | `gpu_port.py:2189` / `numpy_port.py:575` `sig_hot = dec[self.tail.cols_of(i)]` | `dec` is `(dim,)` ⇒ `sig_hot` is `(H,)` | **(i)** | must become `dec[:, cols]` ⇒ `(B, H)`; the tail's `begin_step`/`credit` then take a per-sample `sig_hot`. One line at each call site. |
| T9 | `gpu_port.py:236-244` / `numpy_port.py:267-280` `credit(layer, dlds_full, sig_hot)` — `g = dlds_full[:, cols]`, `out[v,h] = Σ_b E[b,v,h] g[b,h]` | the tail credit is summed over the batch | **(iii)** | **nothing** — the embedding gradient is a sum over the batch by construction, so a per-stream `E` is already right. |

### 3.4 The decaytrace and the gradient assembly

| # | site | what it assumes | class | what breaks |
|---|---|---|---|---|
| G1 | `gpu_port.py:2217-2218` `l.decaytrace = (dec[None,:] * l.decaytrace + dec[None,:] * (1.0 - dec[None,:]) * l.states)` | `dec` is `(dim,)` ⇒ `dec[None,:]` is `(1,dim)` | **(i)** | with `dec` of shape `(B, dim)`, `dec[None,:]` is `(1,B,dim)` — **the exact broadcasting bug the gate gradient already had** (the engine says so at `gpu_port.py:2248-2252`). Fix: an ndim-guarded branch (`dec` vs `dec[None,:]`). **Arithmetically identical in the ungated case.** |
| G2 | `numpy_port.py:594-595` — the same line | same | (i) | same as G1. |
| G3 | `gpu_port.py:2219` / `numpy_port.py:596` `decay_grads[i] += (dlds * l.decaytrace).sum(dim=0)` | — | (iii) | **nothing** — already sums over the batch; the `(dim,)` decay parameter's gradient is the batch-summed `(B,dim)` product. |
| G4 | `gpu_port.py:2245-2256` / `numpy_port.py:608-613` `gate_gradient(dlds, l.gatetrace, dec, gate_xin[i], l.states)` | `a` may be `(dim,)` or `(B,dim)` | **(iii)** | **nothing.** `gate_trace_update` (`gpu_port.py:374-388`, `numpy_port.py:182-192`) branches on `a.ndim == 1`, and `check_gate_grad.py` pins the formula against torch autograd **and** a central finite difference. The gate gradient is *already* per-sample-correct and is the strongest asset in this whole analysis. |
| G5 | `gpu_port.py:2255` / `numpy_port.py:613` `gate_grads[i] += contrib` — the contribution is already batch-summed inside `gate_gradient` | — | (iii) | nothing. |
| G6 | `gpu_port.py:2166-2186` the **dense** branch: `E.mul_(dec[None,None,:])` (2183), `torch.einsum("bvd,bd->vd", E, dlds)` (2185), `E[ar, c_arr, :] += 1.0` (2187) | `dec` is `(dim,)` | **(iii) for the multiply; (iii) for the einsum and the add** | the multiply becomes `dec[:, None, :]` — **one line**. The einsum is already per-stream (`bvd,bd->vd`); the add is already per-stream (`E[ar, c_arr, :]`). **This is the only credit path in the engine that needs no structural change at all.** |

### 3.5 `ring_k` inference

| # | site | what it assumes | class | what breaks |
|---|---|---|---|---|
| K1 | `gpu_port.py:1183-1199` `dmax = max_d σ_d`; `inferred_ring_k = min(256, ceil(ln(RING_TOL)/ln(max(dmax,1e-3))))`; `numpy_port.py:236-244` (mirror) | the *hottest* dim's `σ` bounds every dim's `σ` | **(i)** | under a two-sided gate there is no `σ_max` that bounds the effective `a`. Either a per-dim ceiling on `a` (one-sided gate), or the dense path (no `ring_k` at all), or an explicit `--ring-k` with a documented truncation. Note `RING_TOL = 1e-4` (`gpu_port.py:36`, `numpy_port.py:36`) — a *tight* contract, and `ring_k` saturates at 256 under the P4 init, which is why the P4 ring costs the same as the dense trace anyway. |
| K2 | `gpu_port.py:2556-2569` / `numpy_port.py:695, 718-721` — `ring_k` stored in / restored from the checkpoint; an explicit `ring_k` refuses a mismatched checkpoint | a scalar | (iii) | nothing for the shape; but under the dense path there is no `ring_k` to store. `tests/check_ring_k.py` pins this contract and is unaffected if the gated arm uses the dense path (which has its own `trace_mode` key, already cross-mode-refusing at `gpu_port.py:2597-2604`). |
| K3 | `numpy_port.py:786-803` — the hot/cold truncation *warning* at load (`hot.max() ** ring_k`, `cut = sigma_max ** ring_k`) | a per-dim `σ` bound | (i) | the warning's premise (σ_max bounds the truncation) is exactly what a two-sided gate breaks. Under the dense path there is no truncation to warn about. |

### 3.6 The checkpoint save/load — **no per-dim assumption anywhere**

| # | site | finding | class |
|---|---|---|---|
| C1 | `gpu_port.py:2470` `out[f"decaytrace.{i}"]` | the decaytrace is already `(B, dim)` — per-stream already. | (iii) benign |
| C2 | `gpu_port.py:2493` `out["ring_k"]`; `2494-2506` `dense_E.{i}` `(B,256,dim)`, `tail_cols.{l}` `(H,)`, `tail_E.{l}` `(B,256,H)` | all per-stream or per-dim-**parameter** shapes; no assumption about the decay's shape. | (iii) benign |
| C3 | `gpu_port.py:2605-2620` the `dense_E` load (B-resize); `2620-2645` the parameter load `for name, shape in self.shapes: if name in d: copy_` | the gate is a `(dim,)` **parameter** in `shapes`, so it is saved and loaded generically; a legacy checkpoint with no `layer{i}.gate` key leaves the gate at its init (0), which is exactly the "resumable A/B" property. | (iii) benign |
| C4 | `numpy_port.py:678-695` (mirror), `numpy_port.py:680-682` (the comment that the gated arm's buffers are always zero because it refuses to run today) | same. | (iii) benign |

**Conclusion: the checkpoint path imposes no per-dim constraint.** A gated
checkpoint is a normal checkpoint with 8192 extra parameters.

### 3.7 The parity and regression gates

| # | site | finding | class |
|---|---|---|---|
| Q1 | `tests/check_parity.py` — numpy ≡ torch on the **default** engine | the default engine's `theta` layout has **no** `layer{i}.gate` key, so the parity gate is untouched by the gate's existence. | (iii) — **the risk to the parity gate is LOW**, provided every change is behind `self.gated` / `trace_mode == "dense"`. |
| Q2 | `tests/check_gate_grad.py` — pins the gate gradient against autograd **and** a central finite difference | it drives the engine's own `Layer.forward` and `gate_gradient` — but **not** `step_batch`, so it does **not** exercise the embedding credit at all. | **(i) — this is the missing test.** The whole reason the arm refuses is "a plausible-looking gradient that is quietly wrong". A gradient test that never touches the credit path cannot certify the credit. A new gate must compare the gated `step_batch`'s **full** gradient (including `dembed`) against torch autograd on the same window. |
| Q3 | `tests/check_gated_recurrence.py:202-215` — asserts the refusal (`NotImplementedError`) and that it fires before `theta`/moments/`t` move | the refusal is pinned. Lifting it requires editing this file. | (iii) — a test edit, and the edit must *replace* the refusal assertion with the **run** assertion (the arm must now step, and the gate must now change the loss). |
| Q4 | `tests/check_dense_trace.py` — dense ≡ tail-with-all-dims-hot and dense ≡ ring at lags `< K`, both at `0.00e+00` | **this is the template for the gated credit gate**: the same comparison, but with a per-sample `dec`. | (iii) — reusable as the reference for the new gated credit gate. |
| Q5 | `tests/check_ring_k.py` — the checkpoint's `ring_k` contract | unaffected by the gate; affected only if the gated arm changes the trace mode. | (iii) |

### 3.8 Everything else

| # | site | finding | class |
|---|---|---|---|
| X1 | `train_gpu.py:1055` — `gated=False` is **hard-coded** | **the CLI has no `--gated` flag**; the gated arm has never been reachable from the trainer. Any run needs a flag (or a script). | (iii) — a CLI addition, on top of the engine work. |
| X2 | `tools/membench/model_backend.py:69` — `N.Model(dim=dim, layers=layers, temp=0.75, lr=5e-4, fsp=fsp)` | the D5 probe's model builder has **no `gated` argument**; the selectivity probe needs one (and the ability to load a gated checkpoint). | (iii) — a small addition to the probe. |
| X3 | `gpu_port.py:1362-1420` `new_states` — the ring's buffers are allocated **empty** on the dense path, and `dense_E` is allocated only there | the dense path is already memory-optimised; switching the gated arm to `trace_mode="dense"` inherits that. | (iii) — no change, an asset. |
| X4 | `gpu_port.py:1804-1856` / `numpy_port.py:435-456` — the refusal and its (partly incorrect) comment | the comment lists four blockers; §3 shows three of them are broadcast changes and one (`ring_k`/hot set) is the real one. | — the comment must be rewritten when the refusal is lifted, or it will mislead the next reader exactly as this analysis found it misleading. |
| X5 | `tools/gate_grad_probe.py` — the four-way truth table for the gate gradient | exists and is the right reference for Q2's new gate. | (iii) — reusable. |

### 3.9 Summary of the classification

* **(i) correctness — real blockers, 5 sites:** R6/R7/R8 (the three CUDA decay
  indexings), R10 + T5 + T6 + K1 (the truncation bound and the hot set — *one*
  problem, four manifestations), G1/G2 (the `dec[None,:]` broadcast), T8 (the
  `sig_hot` gather), Q2 (the missing full-gradient gate).
* **(iii) implementation convenience — 20+ sites**, every one of them a
  broadcast change with **identical arithmetic** in the ungated case.
* **(ii) performance choice — 2 sites:** the fused nvrtc ring path and the
  fused tail path are speed items (the fallbacks are correct); the dense
  trace's memory bill is the price of the exactness that dissolves R10/T5/T6/K1.
* **(i) — already correct, no change:** the gate gradient (G4/G5), the
  checkpoint path (C1-C4), the tail's per-stream credit reduction (T9), the
  tail's ring seeding (T7), the dense einsum (G6).

---

## 4. The central question: is the gate implementable without changing the arithmetic?

**Short answer: yes — the *arithmetic* is already right; the *bookkeeping* is
not. The engine's credit paths are the same recursion as the state recurrence,
so a per-sample retention needs broadcasts, not new mathematics. What genuinely
must change is one thing: the truncation bound.**

The refusal's own words (`gpu_port.py:1845-1856`) list four blockers. Scored
against the code:

| stated blocker | verdict |
|---|---|
| `_scatter_ring_credit` does `w.mul_(dec[None,None,:])` at three sites, `w` is `(B,K,dim)` | **overstated** — the broadcast becomes `dec[:,None,:]`, the same elementwise product. Only the three *kernels*' raw-pointer indexing (R6-R8) needs real work. |
| `TorchTail.begin_step` takes `sig_hot` for the hot dims, i.e. a vector | **overstated** — `sig_hot` becomes `(B,H)`, the same elementwise product. The kernels need an index change; the fallback does not. |
| the `decaytrace` recurrence multiplies by `dec[None,:]` | **correct as stated, and the fix is one line** — but it is a genuine *silent* bug (the `(1,B,dim)` broadcast), the same one the gate gradient had. |
| `ring_k` is inferred from `sigma_max`, which stops being a scalar | **correct, and it is the real blocker** — but the plan over-diagnoses it as "not a scalar". `sigma_max` does not have to be a scalar: it has to be a **valid upper bound on the effective retention**, and that is exactly what the dense trace removes (no `ring_k` at all). |

### 4.a (option a) A per-DIM gate driven by a per-dim statistic of the residual

*Shape*: `a = σ(decay + gate · mean_b(x_in[b,:]))` — a `(dim,)` vector, shared
across the batch.

*What it buys*: the retention becomes **content-adaptive** (it depends on what
was just written), which is the qualitative property the gate is for. What it
does **not** buy: per-sample selectivity. Two streams in the same batch get the
same retention; a stream whose byte is "important" and a stream whose byte is
"noise" are treated identically. Since the D5 measurement is a *per-sample*
retrieval task, this may be enough or may not.

*What it costs*: the same truncation problem, because the retention is still
**time-varying** (the statistic changes from step to step). A per-dim gate does
**not** make the credit machinery work — it only avoids the `(B,dim)` shape.

*Verdict*: **rejected as the primary lever.** It keeps the whole per-dim
bookkeeping (hot set, `ring_k`, mask) *intact* (a time-varying but per-dim
retention still needs the per-dim ceiling argument, so it has the same
truncation problem) while giving up the per-sample selectivity that the D5
story is about. It is strictly worse than the alternatives on both axes.

### 4.b (option b) A per-sample gate applied to a SEPARATE small "selectivity" state

*Shape*: the main recurrence's decay stays `(dim,)`; a separate, small
per-sample state modulates the **write** rather than the retention, e.g.
`s_t = a·s_{t-1} + g_t·e_t` with `g_t` a per-sample write weight.

*What it buys*: the decay spectrum (the 8,192 logits, 0.18 % of the weights)
stays exactly as it is, and the "selectivity" mechanism is isolated and
measurable.

*What it costs*:

* the embedding credit must be scaled by the write weight **at the write's
  time**: `dembed[v] += Σ_{lags k} w[b,k,d]·wt[b,k]·dlds[b,d]`. That needs a
  per-`(stream, lag)` write-weight buffer `(B,K)` (cheap: 1 MB at B=1024,K=256)
  **and** the ring's `w` must no longer be the whole story — the credit is no
  longer a pure function of `w`, so the fused nvrtc ring kernel needs a second
  pointer and a new index (`R6` becomes a bigger change);
* the tail's `add_byte` must add `wt` instead of `1.0` (`tail_credit.py:133,
  154` and `gpu_port.py:290-294`), and the dense path's `E[ar,c,:] += 1.0`
  likewise;
* a **new gradient**: `∂L/∂gate` for a write weight needs its own trace
  (`wt_trace_t = a_t·wt_trace_{t-1} + (∂g_t/∂θ)·e_t`) — the same shape as the
  existing `gate_trace_update`, but the existing pinned one is for the
  *retention* form and would not be reused;
* and the mechanism is **not** "adaptive forgetting": it scales what is
  written, not what is kept. The D5 diagnosis ("the address survives where the
  content decays") is a statement about **retention over time**, so a
  write-weight gate attacks a different quantity.

*Verdict*: **rejected, but it is the fallback if the dense path's memory is
prohibitive at the working batch** — and it is *more* work than the dense
path, not less.

### 4.c (option c) The per-sample gate done fully, with the credit re-derived

This is the plan's own option, and it is right — but the plan's framing of it
("the ring would then have to store the undecayed value plus the per-element
factor (twice the memory), or the credit has to be re-derived") **mis-describes
the ring**. The ring does **not** need to store the undecayed value: its
`w[b,k,d]` already *is* the per-sample survival product `Π a` (§1.1, §2, both
verified at `0.00e+00`). The ring needs no second buffer. What it needs is a
**valid truncation bound**, and there are exactly two ways to get one.

**c′ — re-derive the credit on the DENSE trace (recommended, §5).**
`trace_mode="dense"` already exists (`gpu_port.py:2166-2186`), is already
**bit-identical** to ring+tail (`tests/check_dense_trace.py`, `0.00e+00`), and
has **no `ring_k`, no hot set, no mask, no rebase and no truncation**. It is
therefore **exact for any per-sample retention**, two-sided gate included, with
a **one-line broadcast change** (`E.mul_(dec[:, None, :])`). The plan's own
`RESULTS.md` says so: *"P3's blocker dissolves. … A dense trace has no
per-dim addressing constraint, so a per-sample `a = sigmoid(decay + gate*x)` is
no longer a problem for the credit path."*

**c″ — re-derive the credit on the ring, under a ONE-SIDED gate.**
Constrain the effective retention to `a_{b,d} ≤ σ_d` (the gate may only
*shorten* retention). Then:
`a_{b,d}^K ≤ σ_d^K ≤ HOT_FRAC/RING_TOL`, so the **existing per-dim hot set,
the existing per-dim mask and the existing `ring_k` inference stay exactly
valid** — no rebase, no union, no memory change, no `ring_k` change. The only
changes are the broadcasts (§3.2, §3.3) and the kernel indexings (R6-R8, T3-T4).
The price: the gate's functional form gains a one-sided constraint, which puts
an **indicator** into the gate's local gradient term, so the pinned
`check_gate_grad.py` formula must be re-derived and re-pinned; and the model
can never make a dim retain *longer* than its decay logit says (the decay
logit remains the ceiling; the gate provides the dynamic range below it).

**c‴ — re-derive the credit on the ring with a per-stream hot set.**
Theoretically the most faithful (the hot set becomes
`{d : max_b a_{b,d}^K > HOT_FRAC}`, refreshed every step) but it costs a
per-step rebase (the engine's own comment, `gpu_port.py:1912-1914`, records a
per-step sync at 99 % of GPU step time), a per-stream mask (R9), and a hot set
whose union can grow toward `dim` — at which point the tail *is* the dense
trace, with the dense memory bill and none of the dense path's simplicity.
**Rejected: it converges to c′ while paying more.**

### 4.d (option d) Anything better, argued from the code

**Yes: the dense trace *is* the better mechanism, and it is already in the
repo.** Three findings from the code support it:

1. **The credit recursions are already the state recurrence.** The dense
   trace's `E[b,v,d] ← E[b,v,d]·a_t + onehot(c_t)` is literally
   `s_t = a_t s_{t-1} + e_t` with the byte id in place of the embedding. It
   is the *same* object the state is, and it is exact for a per-sample `a`
   (§1.1, §2). Nothing about the gate is foreign to it.
2. **The dense path is the only credit path with no per-dim addressing at
   all** — so it is the only one that needs no hot set, no mask, no `ring_k`
   and no rebase. That is four of the five real blockers, gone.
3. **The memory is not a new bill.** Under the P4 init, `ring_k` saturates at
   256, so the ring's buffer is `(B,256,dim)` — the same shape and size as the
   dense trace (`RESULTS.md`: *"the ring costs the same once `ring_k`
   saturates"*; measured peak CUDA at B=256: ring 3.55 GB vs **dense 2.46 GB**).
   And the D5 measurement was made on a **P4** checkpoint
   (`runs/taxis/A_p4_wd0_c.npz`), i.e. in the regime where the dense path is
   the *cheaper* of the two. The dense path's cost is a **throughput** question
   (2.15 GB of traffic per step at B=256, dim=512, 16 layers ≈ 2–4 ms/step at
   T4 bandwidth), not a **capacity** question.

A second, smaller finding: **the one-sided gate (c″) and the dense path (c′)
are not mutually exclusive**, and the combination is worth naming because it
changes the *shape* of the A/B: c′ alone keeps the two-sided gate (the pinned
gradient, the full mechanism) and pays memory; c″ alone keeps the ring's
throughput and pays a restricted mechanism plus a re-pinned gradient. **c′ is
recommended because the gradient is the thing this project has already got
right and verified twice.**

---

## 5. Recommendation

### 5.1 The recommendation

**Implement the P3 gate fully and two-sided, with the credit re-derived on the
existing dense-trace path (`trace_mode="dense"`), i.e. option (c′).**

Concretely:

1. **Forward** — no change. `gpu_port.py:322` already returns `(B, dim)`;
   `gate = 0` remains bit-identical to the incumbent.
2. **Gradient** — no change. `gate_trace_update` / `gate_gradient` already
   handle `(B, dim)` and are pinned against autograd and finite differences.
3. **Credit** — the gated arm runs the **dense** trace:
   `E.mul_(dec[:, None, :])` (one line, `gpu_port.py:2183`), the einsum and
   the add are already per-stream. No `ring_k`, no hot set, no mask, no
   rebase, no truncation, no `ring_k` inference to re-derive.
4. **decaytrace** — an ndim-guarded branch at `gpu_port.py:2217-2218` and
   `numpy_port.py:594-595` (`dec` vs `dec[None,:]`). Arithmetically identical
   in the ungated case; fixes a silent `(1,B,dim)` broadcast otherwise.
5. **The refusal** — lifted, and its comment rewritten to state the real
   history (the dense trace's existence dissolves the per-dim constraint).
6. **The missing gate** (this is the real deliverable): a new
   `tests/check_gated_credit.py` that compares the gated `step_batch`'s **full**
   gradient — `dembed`, `ddecay`, `dgate` — against **torch autograd** on the
   same window, and against the dense-vs-ring equivalence of `Q4` under a
   per-sample `dec`. `check_gate_grad.py` pins the recurrence's gradient but
   never touches the credit; the arm's whole reason for refusing is the risk
   of a plausible-but-wrong credit, so the credit must be pinned before the
   refusal is lifted.

### 5.2 Why this and not the alternatives

* It is the only option that needs **no change to the arithmetic**: the
  forward is untouched, the gate gradient is untouched, the credit's *product*
  is untouched (only its broadcast changes), and the incumbent engine's every
  byte is untouched.
* It is the only option where the **pinned** gradient survives as-is.
* It dissolves the **one real blocker** (the truncation bound) instead of
  working around it with a mechanism restriction.
* It reuses machinery the repo already trusts: `trace_mode="dense"` is
  bit-identical to ring+tail (`0.00e+00`) and its buffers are already
  memory-optimised (`gpu_port.py:1362-1420`).
* At the P4 init — the D5 configuration — it is **not** more memory than the
  ring (both are `(B,256,dim)`; measured 2.46 GB vs 3.55 GB peak at B=256).

### 5.3 Cost

| item | estimate | notes |
|---|---|---|
| `gpu_port.py` — dense decay broadcast + decaytrace ndim guard + refusal/comment + gated-mode plumbing | **≈ 12 lines changed** | the only *arithmetic* edits are two broadcasts and one guard, each arithmetically identical in the ungated case |
| `numpy_port.py` — the reference implementation has **no dense path** (it is ring + `TailCredit` only): add one | **≈ 35 lines** | the numpy reference is what `check_parity` compares against; without it the gated arm has no parity gate |
| `tests/check_gated_credit.py` — full-gradient-vs-autograd + dense-vs-ring under a per-sample `dec` | **≈ 120 lines** | the gate that does not exist today and the one the refusal is really waiting for |
| `tests/check_gated_recurrence.py` — replace the refusal assertion with a run assertion | **≈ 15 lines** | the file's §(c) block currently *asserts* `NotImplementedError` |
| `train_gpu.py` — a `--gated` flag (`gated=False` is hard-coded at 1055) | **≈ 5 lines** | without it the arm is unreachable from the trainer |
| `tools/membench/model_backend.py` — a `gated` argument for the probe | **≈ 5 lines** | the D5/probe model builder |
| **memory** | dense trace `B·256·dim·4` per layer = **2.15 GB at B=256**, 536 MB at B=64, 134 MB at B=16 (dim=512, 16 layers) | **the same bill as the P4 ring at `ring_k = 256`**; the gated arm cannot run at B=1024 on a 4 GB card (neither can the P4 ring) |
| **throughput** | the dense credit moves `B·256·dim·4` bytes per layer per step ≈ 2.15 GB/step at B=256, dim=512, 16 layers → ~2–4 ms/step at T4 bandwidth | **not measured**; this is the one number that must be measured before committing. If it is prohibitive, switch to (c″) below |
| **risk to the parity gate** | **LOW** | every change sits behind `self.gated` / `trace_mode == "dense"`; the default engine's `theta` layout, its numerics and `check_parity.py` are untouched. The *new* risk is a **new** gate (the full-gradient one) — which is the point |
| **risk to the existing gates** | `check_gate_grad.py` **passes unchanged** (the gradient is untouched); `check_dense_trace.py` **passes unchanged** (the dense path is untouched in the ungated case); `check_gated_recurrence.py` **must be edited** (it asserts the refusal) | — |

**Fallback if the dense path's throughput or memory is prohibitive at the
working batch:** option **(c″), the one-sided gate on the ring**
(`a ≤ σ_d`). Cost: ≈ 8 lines of broadcast changes (`R1-R3`, `T1-T2`, `T8`),
≈ 15 lines of CUDA index changes (`R6-R8`, `T3-T4`), one clamp indicator in
the gate trace plus a re-pinned `check_gate_grad.py` (≈ 40 lines), and **no
memory change** and **no hot-set / `ring_k` / rebase change**. The price is
the mechanism restriction: the gate may only shorten retention.

### 5.4 What this analysis does **not** claim

* It does **not** claim the gate will fix D5. It claims the gate can be made
  to *run* without touching the arithmetic, and that the credit can be made
  exact. Whether selectivity is the binding constraint is the probe's question
  (§6).
* It does **not** claim the fused nvrtc ring path survives a per-sample
  retention unchanged — it does not, and the dense path sidesteps it.
* It does **not** measure the dense path's step time. That number is the one
  open engineering risk in the recommendation, and it should be measured
  (a 20-step timing on an existing checkpoint is enough) before the arm is
  scheduled.

---

## 6. What the selectivity probe (part 2) must measure

The probe must be run on the **D5 instrument** (`tools/d5_probe.py`), with the
state read **at the cue**, a **matched 16-symbol alphabet**, `n = 2048`, and a
**matched null per cell** — the same fixes that made D5 legible (D1, D4). It
must answer, in this order:

**M1 — Did the gate engage at all?** (the cheapest and the most decisive)
Measure the distribution of the effective retention `a_{b,t,d} = σ(decay_d +
gate_d·x_in)` over (stream, step, dim): the per-dim spread and the **per-sample
spread**. If the per-sample spread of `a` is ≈ 0, the gate is inert and the
arm is a no-op — a null with a measured cause, and the recommendation changes
(§6.2). This is the probe's own "is the instrument powered" check and it must
come first, because a null downstream is uninterpretable without it.

**M2 — Did the content stop decaying?** The D5 cells, per gap
(8/12/32/64/128 B): the **payload's ×null** and the **cue key's ×null**. The
pre-registered expectation for the gate hypothesis: the payload's ×null at
32–128 B **rises** (today: 3.33× at 32 B, 2.81× at 64 B, **1.52× at 128 B**)
while the key's ×null does not fall below its own null (today: 90×/65×/84×).
The headline number is the **payload ×null at 128 B**; today it is 1.52×,
i.e. barely above its own null.

**M3 — Is the improvement attributable to the gate?** The gate adds
`layers·dim = 8,192` parameters (0.18 %). The clean control is the **same arm
with the gate frozen at 0**, which is bit-identical to the incumbent
(`check_gated_recurrence.py` §(a)) — same code path, same seed, same step
count, same data. If the probe compares only "gated vs ungated checkpoint",
the 8,192 parameters are confounded with the gate's function; the
gate-frozen control removes that.

**M4 — Is the credit right?** This is not a probe question but a **gate**
question and it must be answered *before* the probe is believed: the gated
`step_batch`'s full gradient (`dembed`, `ddecay`, `dgate`) against torch
autograd, plus the dense-vs-ring equivalence under a per-sample `dec`. A probe
run on a wrong credit measures the wrong thing — this is the project's
recurring failure mode (D1, D4, and the `dec[None,:]` broadcast bug).

**M5 — What did the gate do to the retention spectrum?** Report the effective
half-life distribution `−ln2/ln a` per dim, gated vs ungated. A gate that
collapses the model's ability to hold a long half-life would show up here and
would explain a null in M2 without the mechanism being wrong.

**M6 — A DIRECT selectivity measurement** (the one `STRATEGIC-REVIEW §3(iii)`
asks for: *"can the state transition protect relevant information from
irrelevant writes?"*). The D5 readout measures retrieval; it does not measure
selectivity. The direct form: compare the payload's decode after `d`
distractors of two kinds — (i) random distractors and (ii) distractors chosen
to be *maximally confusable* with the payload (same byte values, same
position in the sequence). If the write is selective, (ii) should damage the
payload substantially more than (i); if the write is an indiscriminate
weighted average, (i) and (ii) are the same. This is the measurement that
distinguishes "the gate made the state hold more" from "the gate made the
write selective".

### 6.2 What would change the recommendation

| observation | conclusion | what changes |
|---|---|---|
| **M1: the per-sample spread of `a` is ≈ 0** | the gate never engaged — the mechanism is inert (a training/objective problem, not an arithmetic one) | **the gate is not the lever.** Stop the P3 line; go to M2 (an explicit keyed store, the plan's cheaper candidate) or M1 (the read's wiring). The engine work in §5 is not wasted (it makes the arm runnable), but it is not the priority |
| **M1: the per-sample spread of `a` is large, but M2 shows no rise in the payload's ×null at 32–128 B** | the write's selectivity is **not** the binding constraint | **the recommendation flips to the read.** The D5 interpretation ("order-losing weighted average ⇒ selectivity is the lever") is wrong; the plan's own M1 (hierarchical feature-state wiring) or M2 (keyed store) becomes the next arm |
| **M2: the payload's ×null rises but the key's ×null collapses** | the gate is buying content by destroying the address | the mechanism needs a constraint (e.g. the one-sided gate, or a gate on a *separate* selectivity state so the main spectrum is untouched) — i.e. option (b) becomes relevant after all |
| **M6: (i) and (ii) are indistinguishable** | the write is still an indiscriminate weighted average, whatever the gate did to `a` | the gate's mechanism is not the one the diagnosis calls for; re-derive before spending more |
| **M4: the full-gradient gate fails** | the credit is wrong | the arm is not runnable; the dense path's broadcast (or the numpy mirror) is wrong; fix before any probe |
| **the dense path's measured step time is prohibitive at the working B** | the recommendation is infeasible as stated | switch to (c″), the one-sided gate on the ring (§5.3) — accepting the restricted mechanism and the re-pinned gradient — before abandoning P3 |

---

## 7. Sources

* `docs/plans/PLAN-2026-10-21-CONTINUAL-LEARNING.md` §2, §4, §6 (D5).
* `docs/plans/PLAN-2026-10-11-T-AXIS.md` §2 P3 (the original design and the
  deferral), and the "P3's blocker dissolves" note in `RESULTS.md`.
* `docs/reviews/STRATEGIC-REVIEW-2026-10-05.md` §3 (i)(ii)(iii) and §4 (the
  copy task, and the "testable form of the objection is **selectivity**").
* `gpu_port.py` — 95-215 (`TorchTail`), 306-330 (`TorchLayer.forward`),
  374-402 (the gate trace/gradient), 442-465 / 639-720 / 757 / 893-910 (the
  CUDA ring kernels), 1183-1199 (`ring_k`), 1362-1420 (`new_states`),
  1534-1650 (`_scatter_ring_credit`), 1804-1856 (the refusal),
  2166-2186 (the dense credit), 2189-2219 (the credit + decaytrace),
  2470-2645 (the checkpoint).
* `numpy_port.py` — 131-160, 182-208, 236-244, 435-456, 563-596, 678-721.
* `tail_credit.py` — 67 (`HOT_FRAC`), 93-160 (the numba kernels), 165-300
  (`TailCredit`).
* `tests/check_gated_recurrence.py`, `tests/check_gate_grad.py`,
  `tests/check_parity.py`, `tests/check_dense_trace.py`, `tests/check_ring_k.py`.
* `RESULTS.md` — "The ring was an approximation, and the dense trace
  reproduces it exactly"; "The P4 init has the memory bill the plan priced for
  the dense trace".
* The §2 measurement was run as a scratch script (not committed) and is
  reproduced inline above.
