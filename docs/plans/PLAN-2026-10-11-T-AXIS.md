# PLAN — the T axis: from stream-parallel to window-parallel (2026-10-11)

**Date:** 2026-10-11
**Status:** plan, not execution. Ordered so every step is independently gated and
revertible. Supersedes the ordering of `PLAN-2026-10-04-ROADMAP.md` v3 and of
`PLAN-2026-10-06-MEMORY-BENCHMARKS.md`; retains their evidence.

> **UPDATE 2026-10-12 (GPU + the P4 capacity probe).** The plan has been
> executed as far as P0, and four of its claims changed. What was measured:
> (1) **the T axis is free on a GPU at constant B** — 1.09x the incumbent's
> throughput per byte at T=16 — while the plan's §5 falsifier actually
> measures the *B reduction* (3.7x worse at B=64), so §1's bet is what fails,
> not P1; (2) the whole gate suite had **never run on a GPU** and four device
> bugs were hiding there; (3) a **0-training capacity probe** shows the P4 init
> really does buy retention capacity (49%/21%/5.8% of the control floor at
> d=8/32/128, against 0.4%/0%/0% for the default init) — and that the
> TRAINED incumbent retains *less* than a fresh P4 model, so P4 is necessary
> but not sufficient; (4) upstream's MTP needs **no** T window, so P5 does not
> wait for P1. See §0 findings 9–12 and §7 for the corrections.

> **One sentence:** stop batching over 1024 streams and start batching over time —
> add `seqlen` (T) as a first-class knob with `T=1` as the floor — because T is the
> prerequisite for multi-token prediction, and because lowering B in exchange is
> what makes the entire ring/tail-credit apparatus deletable.

**Sources pooled:** this repo's own `RESULTS.md` and reviews; upstream issues
`#17 #20 #23 #25 #28 #30 #31 #34` (`jrz97619761/test-model-thing`); and the CUDA
fork `SimonVitzethum/test-model-thing-CUDA` at `2cfdf41` (`tmt-cuda/`,
`tmt-cuda/src/{config.h,cell.cu,model.cu}`, `ARCHITECTURE.md`, `CHANGES.md`).

---

## 0. What we now know, and what it rules out

| # | finding | source | consequence |
|---|---|---|---|
| 1 | Our gradient is **already exact for every parameter that has a cross-time path.** In `gpu_port.py:TorchLayer.forward`, `enc` is passed unchanged to all 16 layers; `W`/`g`/`b` produce only `y`, which reaches only the residual `x` and the loss. Only `embed` and `.decay` touch `state` — and both are traced exactly. | this repo, verified | The 1-step gradient **is** the true gradient for the FF/LN/decoder weights. The whole SNR/trace-quality programme was solving a problem this architecture does not have. **Closed.** |
| 2 | The CE ruler is blind to memory: an order-8 byte n-gram (0 params, 8 MB) scores **0.81** nats/byte on our held-out set vs our **1.2094**. | `STRATEGIC-REVIEW-2026-10-05.md` §1 | No optimizer, estimator or init change can produce long memory that is *used* on this corpus. |
| 3 | SVC's 34M-param run independently reproduces our #23 result: traces **0.5% worse** than plain TBPTT at equal wall-clock at full context; **+7.3%** only when state resets every 128 bytes. | `#23`, 4h×2 arms, RTX 5070 | Our P0 demotion was right and is now externally corroborated by someone who had exact gradients and chose not to use them. **Stop spending on credit assignment.** |
| 4 | SVC's converged default after 420M-param runs: `gated=1 half_min=2 half_max=512 seqlen=128 batch=8 experts=1 topk=1 traces=0 ce=1 var=0 latent=0 stop=0 maxcarry=0`. | `tmt-cuda/src/config.h` | **Everything exotic is opt-in and measured-off.** The architecture he landed on is *gated recurrence + log-spaced half-lives + a TBPTT window + CE-only*. That is a far stronger prior than any of our speculation, and it is close to what we already have. |
| 5 | Data-limited, not capacity-limited: 14 epochs over 95 MB beats 24 (1.63 vs 1.82 BPB). | `#23` | More parameters mostly widens the generalization gap. Scale is not the first lever. |
| 6 | No measurable retrieval past ~64 bytes, in either trace condition, with a real-text induction probe. | `#23` comment, 48 trials/cell | Consistent with our own state-probe ruler (8 B uniform-cue, ≥16 B corpus-support). |
| 7 | On real text, gradient descent **undoes** log-spaced init within one epoch: TinyStories, TBPTT K=32, 3 seeds — original 1.539 ± 0.010 vs log-spaced 1.537 ± 0.029, median half-life 0.96 vs 0.93. | mar-i0, `#3` comment | Do the init because it is the right prior and costs nothing, **not** because it persists. |
| 8 | `tools/eval.py`'s ruler is already hygienic: `cross_entropy` calls `m.new_states(1)` per slice and `forward_logits` performs no weight update. | verified here | Priority 4 of `PLAN-2026-10-06` is **already satisfied**. Do not re-do it. |
| 9 | **On a GPU, T is free at constant B, and B reduction is what costs.** MX150: (1024,16) = **1.09x** the incumbent's bytes/s; (64,16) = 0.27x, (16,64) = 0.07x, (8,128) = 0.03x. The degradation tracks the GEMM's row count, not T. | 2026-10-12, `tools/seqlen_probe.py --device cuda` | **§1's bet fails; P1 survives.** Take T at B=1024 and KEEP the ring. The §5 falsifier as written conflates the B reduction with the T axis. |
| 10 | The gate suite had **never run on a GPU**: four device bugs (the probe, the P1 gate, the parity gate, and three gates that silently became CUDA gates) and five hardcoded absolute paths. | 2026-10-12 | Every gate in §2 is a **CPU** gate until it says otherwise; "the P1 gate passes" meant "on a CPU". `check_seqlen_credit.py --device cuda` now exists and passes with identical numbers. |
| 11 | **The P4 init buys the capacity the plan says it buys.** 0-training probe (fresh model, default vs P4 init): retention 0.4%/0%/0% vs **49%/21%/5.8%** of the control floor at d=8/32/128. And the TRAINED incumbent retains **4.8%/1.7%/0.36%** — *less than a fresh P4 model at every distance*. | 2026-10-12, `tools/capacity_probe.py`, `tests/check_p4_capacity.py` | The capacity half of "the cause is the init" is **confirmed**. The sufficiency half is **false**: training gives the capacity back. P4 must be paired with P6 (and probably P3) or the spread is trained away. |
| 12 | **MTP needs no T window.** Upstream's `50338c8` adds `mtp=K` heads that predict the byte K positions ahead *from the same representation* — no window is involved. | 2026-09-22, `tmt-cuda` `origin/zig-kernels` | P5's premise ("possible only because P1 exists") is **false**. P5 can run in parallel with P1, and upstream's implementation is a reference. |

---

## 1. The cost of B=1024 that nobody has priced

Exact all-lag embedding credit costs `B · 256 · D · 4` bytes **per layer**. At
D=512, 16 layers:

| B | dense embed-trace cost | what it forced |
|---:|---:|---|
| 1024 (today) | **8.6 GB** | the ring buffer + `tail_credit.py` |
| 256 | 2.1 GB | still over the 4 GB cap with everything else |
| 64 | 537 MB | comfortable |
| 16 | **134 MB** | trivial |
| 8 (SVC's default) | 67 MB | trivial |

**`tail_credit.py` and the whole ring machinery exist only because B is large.**
SVC runs `batch=8`/`16`/`32` by default and never needed it: he carries a dense
`(B,256,D)` trace instead. Raising T and lowering B may therefore **delete ~400
lines and an entire class of correctness risk** — the four ring-credit kernel
paths, hot/cold membership, `ring_k` inference, the `REBASE_EVERY` cadence, the
4 GB tail cap, and the `cold`/`kloop`/`segmented` candidates that are already dead
but still referenced by a gate.

This is the load-bearing bet of the plan, and §5 states its falsifier.

---

## 2. Priority queue

### P0 — make the new engine trustworthy before measuring anything in it (~0 GPU)

**DONE 2026-10-11.** All three items, with gates:

1. **`ce_only` finished.** `tests/check_ce_only.py` now exists and proves four
   things: `ce_only=False` leaves every auxiliary term bit-identical
   (`masked == raw` over 60 steps, 180 non-zero term-steps) and is deterministic
   at a fixed seed; `ce_only=True` masks all three terms to **exactly** 0.0 while
   asserting the raw values are non-zero in the same breath (a zeroed zero proves
   nothing) and leaves the stop-head gradient exactly 0.0; KD and CE-only refuse
   each other at construction. Wired as `--ce-only` in `train_gpu.py`, and the
   trainer now prints `loss == ce` exactly, which is the visible proof. This is
   priority 3 of `PLAN-2026-10-06`, now runnable.
2. **`tests/check_optimizations.py` repaired.** It constructed
   `ring_credit="cold"/"kloop"/"segmented"` — all measured and deleted — so it
   died with `ValueError` instead of testing anything. It now covers the two
   axes that are live and unguarded: `grad_assembly="loop"` is bit-exact vs
   `"cat"` (documented in `gpu_port.py`, previously untested), and the four
   removed candidates plus stale factored checkpoints must be **refused** rather
   than silently downgraded. 16/16 gates were passing before this; it is the
   reason they were 15/16.
3. **NaN guard + gradient clipping — DONE 2026-10-12.** The engine now
   REFUSES a non-finite gradient: theta, m and v are left bit-identical, the
   refusal is counted and logged, `t` still advances (the data was consumed;
   rewinding the LR schedule would be a second, quieter way to change the
   run), and training continues. The guard is a **no-op on finite gradients,
   bit-exactly** (gated), so it is safe to enable by default. The global clip
   is off by default (`--grad-clip 0`) and its threshold is in ABSOLUTE
   gradient units, which are **not comparable across T** (the window's
   gradient is a sum over T bytes): a clip is a safety net, not a knob, and
   it must be off for a T A/B. A measured caveat worth keeping: under AdamW
   the clip does **not** shrink the parameter step (the update is
   `m/sqrt(v) ~ sign(g)`, nearly scale-invariant); it bounds the MOMENTS,
   which is what stops one enormous gradient from suppressing the next
   thousand steps. Gate: `tests/check_safety.py`.

### P1 — `seqlen` as a knob, `T=1` as the floor

- Signature becomes `step_batch(cs (B,T), ns (B,T), ends (B,T))`; the t-loop lives
  inside the step; **one** optimizer update per window.
- **Gate 1:** at `T=1`, bit-identical to today's engine. The existing
  `tests/check_parity.py` harness already proves numpy ≡ torch; extend it to prove
  `T=1` ≡ `T>1`-with-T=1.
- **Gate 2 (the measurement we have never had):** `tests/check_trace_bptt.py` at
  `T>1`. Within the window the gradient must be exact; the boundary is where it
  degrades. Print both, per parameter group.
- **Measure `(B,T)` throughput before tuning anything.** **MEASURED 2026-10-12
  on the MX150 (CUDA) and on CPU.** The plan's own prediction ("expect the
  useful range to be `T=8…32`, not 128", "torch autograd will not match
  SVC's fused-kernel numbers") is wrong in our favour on one axis and wrong
  in the other direction on the other:

  | cell (B×T) | B·T | MX150 µs/byte | vs (1024,1) |
  |---|---:|---:|---:|
  | 1024×1 | 1024 | 148.6 | 1.00x |
  | 1024×2 | 2048 | 140.7 | 1.06x |
  | 1024×4 | 4096 | 137.3 | 1.08x |
  | **1024×16** | 16384 | **136.3** | **1.09x** |
  | 256×16 | 4096 | 157.3 | 0.94x |
  | 64×16 | 1024 | 551.5 | 0.27x |
  | 16×64 | 1024 | 2153.9 | 0.07x |
  | 8×128 | 1024 | 4277.1 | 0.03x |

  * **T is FREE at constant B** — 1.09x FASTER per byte at T=16, because the
    per-window overhead (optimizer, gradient assembly) amortizes over the
    window's bytes. A live run confirms it: 7400 B/s at T=16 vs 6116 B/s at
    T=1.
  * **The cost in this table is the B reduction, not T.** The degradation
    tracks the GEMM's row count (1024 rows -> 8 rows), so at matched B·T the
    small-B cells are launch/occupancy-bound. This is the opposite of the
    CPU reading, where the same trade cost only 1.15x — which is why §5's
    falsifier, decided on a CPU, looked safe.
  * **The autograd worry is dissolved: this engine has no autograd.**
    `theta` is a plain buffer and every gradient is hand-built, so there is no
    T-deep graph and activation memory stays O(B·D) per layer, not O(B·T·D).
    The plan's P1 memory prediction is wrong *for this engine*, and its
    "torch autograd at T>1 is the single biggest unknown" is answered by
    construction. The real unknown is the GPU launch overhead at small B,
    which is what the table above measures.
- **Honest cost:** 16 layers × T forward ops + T backward ops per window, and
  `B·T` positions per optimizer update instead of `B`.

### P2 — the dense trace replaces the ring (IMPLEMENTED 2026-10-12)

`--trace-mode dense`, gate `tests/check_dense_trace.py`. **The ring was only
ever an approximation of one object** — an exact per-byte-id embedding
eligibility trace — kept for memory. `tail_credit.py` already materialised that
object for the dims where truncation costs >1% of credit mass, so the dense
trace is *the tail with every dim hot*, not a new mechanism.

**The plan's gate passes, and the agreement is exact (0.00e+00), not
approximate:**

| regime | result |
|---|---|
| dense vs TailCredit with all dims hot | max rel dev **0.00e+00** |
| dense vs ring, `T < ring_k`, zeroed ring | max rel dev **0.00e+00** |

Because the agreement is bit-level, deleting the ring is now justified rather
than hopeful. The ring and `tail_credit.py` are **not yet deleted** — the dense
path runs alongside them behind the flag, so the incumbent is untouched and
`check_parity` still passes. Deletion is the remaining mechanical step, listed
at the end.

**This also dissolves P3's blocker.** The ring forced a per-dim `(dim,)` decay
vector through `_scatter_ring_credit`, `TorchTail.begin_step` and `ring_k`
inference from `sigma_max`. A dense trace has no per-dim addressing constraint,
so `a = sigmoid(decay + gate*x)` being per-sample `(B,dim)` is no longer a
problem for the credit path. What remains for P3 is only the gradient, which was
already wrong for independent reasons (see P3).

**And it unblocks P4.** `ring_k = min(256, ceil(ln(RING_TOL)/ln(sigma_max)))`
saturates at `half_max ~= 32 B`, so the plan's original `--half-max 512` bought
nothing. With the dense trace there is no `ring_k`, and `--half-max 512` runs
(verified end to end).

**Cost, priced rather than assumed** (`B*256*dim*4` bytes per layer, dim=512,
16 layers): B=16 -> 0.13 GB, B=64 -> 0.54 GB, B=256 -> 2.15 GB. All fit an 8 GB
box. The trace is persisted in checkpoints (`dense_E.{i}`) at the same order of
size the ring's `ring_w` already cost, and cross-mode loads **refuse** rather
than silently starting a cold trace — `sigma**50` is ~1e-15 for cold dims but
**0.36** at sigma=0.98, so a zeroed trace would quietly lose the history of
exactly the dims it exists for.

### P3 — gated recurrence, behind a flag, gate initialized to zero

**PARTIAL 2026-10-12.** Implemented and gated, but the arm REFUSES at
`step_batch` rather than running. Gate: `tests/check_gated_recurrence.py`
("part 1").

**What is verified:** `gate=0` is bit-identical to the incumbent (parameters
bit-equal, and the recurrence reduces EXACTLY to `sigmoid(decay)`); the gate
parameter exists only in the gated arm, is exactly zero at init and is
ELEMENTWISE (`layers*dim`, not a matmul — so the "~16k parameters" note in this
plan was right); its absence from the default engine is what keeps `theta`'s
layout and therefore `check_parity`'s numpy==torch identity intact; the gate has
real authority over the loss; and the refusal fires before theta, the AdamW
moments or `t` move.

**The blocker is architectural, not a missing line.** Gating makes the retention
factor input-dependent, so it becomes PER-SAMPLE `(B, dim)` instead of the
per-dim `(dim,)` vector every consumer assumes:

- `_scatter_ring_credit` does `w.mul_(dec[None, None, :])` at three sites,
  including the fused nvrtc/kloop paths, and `w` is `(B, K, dim)`;
- `TorchTail.begin_step` takes `sig_hot` for the hot dims, i.e. a vector;
- the `decaytrace` recurrence multiplies by `dec[None, :]`;
- `ring_k` is INFERRED from `sigma_max`, which stops being a scalar once sigma
  depends on the input.

**The gate gradient is implemented but UNVERIFIED, and is wrong as written.**
Two independent references agree on that: torch autograd on the same recurrence,
and a central finite difference of the loss, both disagree with the engine's
formula. At `dim=8 layers=2 B=3 T=4` with the gate perturbed to
`linspace(-0.7, 0.9)`:

| | max rel dev layer0 | layer1 |
|---|---|---|
| dL/dgate vs autograd | 4.59 | 8.19 |
| dL/ddecay vs autograd | 2.99 | 4.74 |
| dL/dgate vs finite-diff | 11.7 | 22.4 |

The missing pieces, for whoever picks this up: `ds_t/da_t` must carry the
CROSS-TIME chain (what `decaytrace` already does for the ungated engine and what
a naive per-byte transcription drops), and the state gradient must pass back
through the layer-norm and the `z*sigmoid(z)` gated linear PER LAYER, not through
a single decoder-level `dx`.

**A note on form.** This plan writes the update as `s = a*s + (1-a)*x`. That
cannot satisfy the paragraph's own "bit-identical starting point / resumable
A/B" requirement, because `(1-a)` is not 1.0 — at `gate=0` it would still not
reproduce the incumbent, so an incumbent checkpoint could not be resumed into
the gated arm as its own control. Implemented as `s = a*s + x`: retention is
gated, the write stays weight 1.0, and `gate=0` is exactly the incumbent. If the
bounded-state / non-unit-write property is wanted, it needs a separate arm with
its own control, not this one.

**A note on method.** The first attempt at this gate compared the engine's
hand-built gradient against a second hand-written backward and "failed" at
12–40x. That comparison is nearly worthless: two hand-writtens agreeing proves
only that they share the same omission. Rebuilding the reference on torch
autograd is what produced a trustworthy verdict. Worth remembering for any
future gradient work here — including the rework below.

**Remaining for P3:** (1) rework the ring/tail/decaytrace machinery for
per-sample decay, (2) re-derive `ring_k` (it can no longer be inferred from a
scalar `sigma_max`; a per-sample bound over the batch, or an explicit
`--ring-k`, are the candidates), (3) re-verify the gate gradient against
autograd — the numbers above are the fixed target.

```
a = σ(decay + gate·x);   s = a·s + (1-a)·x
```

`gate` init 0 ⇒ `a = σ(decay)` at init ⇒ **bit-identical starting point**. This
makes it a clean A/B against the incumbent: resumable, no new arm, one flag.
Cost ≈ 16 × 512 × 2 ≈ 16k parameters. The state becomes bounded and the write is
no longer weight 1.0 — log state RMS, which currently drifts 0.40 → 0.46.

This is our roadmap P1.1 + P1.4, it is the owner's explicit `#25` request, and it
is already finite-difference verified in the CUDA fork (gated and ungated, with
nonzero carry).

### P4 — log-spaced half-life init

**DONE 2026-10-12 (implementation + gate).** `--half-min/--half-max` wired to a
new `half_life_init=(hmin,hmax)` kwarg on `GpuModel`; gate
`tests/check_halflife_init.py`. Additive: `half_life_init=None` is bit-identical
to the default build, so `check_parity.py` and every existing A/B arm are
untouched. The *set* of half-lives is geometric, the *assignment to dims* is
shuffled by the seeded rng so dim index stays meaningless.

Measured on the engine: the incumbent init puts every half-life in
**[0.919, 1.094] B**; the P4 arm spans 2.000 -> 512.015 B. The ruler (see
`RESULTS.md` 2026-10-12) is what makes this urgent rather than merely tidy — the
trained checkpoint has only 8 of 512 dims holding more than 8 B, so the deficit
is capacity, not optimisation.

**BLOCKED ABOVE `--half-max 32` BY THE RING CAP — see P2.** `ring_k =
min(256, ceil(ln(RING_TOL)/ln(sigma_max)))`: half_max 8 B gives ring_k 107, but
half_max 32 B and above all saturate at the hardcoded 256. A dim with a 512 B
half-life receives credit from only 256 B of history, so the plan's original
`--half-max 512` is unservable. The honest A/B until P2 lands is
`--half-min 2 --half-max 32`, which ring_k 256 covers exactly.

`--half-min 2 --half-max 512`, geometric, as an init default. **Expect it to decay
away on real text** (finding 7). Do it because it is free and it is the right
prior. If P3's gate lets the model actually *use* the spread, that is the
interesting result and the reason to have done it.

**THE CAPACITY HALF IS NOW MEASURED, AND THE SUFFICIENCY HALF IS NOT**
(2026-10-12). A 0-training probe settles the first half of the plan's causal
story, and it settles it in the plan's favour: `tools/capacity_probe.py` and
the gate `tests/check_p4_capacity.py` measure the retention curve of an
UNTRAINED model with the default init against one with the P4 init, at
seconds of cost and no GPU.

| init | d=8 | d=32 | d=128 | d=512 |
|---|---:|---:|---:|---:|
| default (~1 B half-lives) | 0.40% | 0.00% | 0.00% | 0.00% |
| **P4 (2…512 B)** | **49.2%** | **21.0%** | **5.8%** | **1.05%** |
| TRAINED incumbent | 4.8% | 1.7% | 0.36% | 0.0007% |

(matched-pair signal ÷ random-byte control floor; the full ruler at dim 512,
layers 16, 64 samples. The fast probe at dim 128, layers 4, 32 samples
reproduces it: 0.42%/0%/0% and 49.8%/22.4%/6.0%.)

Two readings, and the second is the one that changes the plan:

1. **The init really is the capacity limit.** A model *given* 2…512 B
   half-lives holds a signal 512 B out; one with 1 B half-lives is dead by
   32 B. So the plan's claim — "there is no dimension in this model capable of
   holding 512 B, and no amount of training builds one it was never given" —
   is supported, and P4 is **necessary**.
2. **P4 is NOT sufficient, and the plan's phrasing hides this.** The TRAINED
   incumbent retains *less* than a fresh P4 model at every distance. The
   capacity is real, and training gave it back. So P4 is a 2×2, not a step:
   `(wd_decay 0.01 / 0.0) × (default init / P4 init)`, with P6 in the first
   factor, because the weight decay on the decay logits pulls them back
   toward `σ=0.5` — back to a 1-byte half-life — which is the *mechanism* by
   which the spread decays away. Finding 7 says gradient descent undoes the
   log-spaced init within an epoch; P6 says weight decay makes sure it
   cannot grow back either. Neither is fixed by the init alone.

The gate is now part of P4's definition: `tests/check_p4_capacity.py` (the
capacity half, seconds) alongside the training A/B (the usage half, hours).

### P5 — multi-token prediction (the reason we wanted T)

Heads at t+1…t+K from the state at t, K ∈ {2,4}.

**PREMISE CORRECTED 2026-10-12: MTP does NOT need T.** Upstream implemented it
in `50338c8` (*"feat: multi-token prediction and the Muon optimizer"*, branch
`origin/zig-kernels`): `mtp=K` gives the decoder K more heads, each predicting
the byte K positions ahead **from the same representation**. No window is
involved, so P5 does not wait for P1 — and upstream's implementation is a
reference (they rate it 50% in their `TODO.md` and never measured it).

This is upstream's current quality frontier (`#28`, `#30`) and it is the direct
answer to our single largest measured deficit: an order-8 lookup table beats us by
0.40 nats, and MTP is how a byte-native model learns multi-byte structure **without
a tokenizer**. If the user prefers T over BPE — and I agree that preference — P5 is
what makes that preference pay off.

### P6 — separate weight decay for the decay logits

**DONE 2026-10-12.** `--wd-decay` (default 0.01 = the incumbent's uniform
weight decay, so every existing arm and checkpoint keeps its exact dynamics;
0.0 exempts the `.decay`/`.gate` groups). The engine now carries a
per-coordinate weight-decay vector; with the default it is uniform and
bit-identical to the incumbent's scalar `wd * th` (gated, 0.00e+00 against a
hand-written AdamW). The `--lr-decay` trap is gone with it: with the decay
group's weight decay at 0, the role's LR scales only the adaptive term.

`gpu_port.py:955` applies `wd=0.01` to the flat vector with no exception for
`.decay`, and `σ(0)=0.5` is exactly a 1-byte half-life, so weight decay is a
systematic pull toward *no memory*, anchored at init. Two parameter groups: `wd`
0.01 for ordinary weights, `wd` 0 for `.decay` (and `.gate` once P3 lands).

Also un-trap `--lr-decay`: `_role_lr_vec` currently scales the decay role's weight
decay along with its learning rate — the "knob wired to a trap" our own review
flagged.

### P7 — document-boundary state resets

**PROMOTED 2026-10-12.** This is the only condition under which the exact
credit our ring/tail/dense machinery represents has ever been measured to
pay: upstream's #23 finds hybrid traces 0.5% *worse* at full context and
**+7.3% better** when the state is reset every 128 bytes. P2 (the dense
trace) is the engineering half of the same object and is now a cleanup, not a
lever; P7 is where the credit is worth anything, and it is cheap (a reset at
a separator byte, `docsep` in the CUDA fork's config).

SVC's `docsep`: force `a=0` at a separator byte so state, gradient and traces are
cut by the same recurrence. Our `make_streams` already round-robins whole lines, so
per-document resets are natural here. This is the **+7.3%** condition from `#23` —
the one regime where exact traces measurably pay. It is also the condition our
`bench_variants.py`/`state_probe.py` arms should be judged under.

---

## 3. Explicitly not doing

| idea | why |
|---|---|
| further optimizer / LR / SNR arms | three axes, all null; SVC reproduces independently |
| trace-quality archaeology | finding 1: the gradient is exact for every parameter with a cross-time path |
| BPE | T is the prerequisite and the more defensible change; revisit only if P5 stalls. Note the owner's own reframing (byte-level BPE, base 256 + merges) is better than our flat rejection, so this is "not yet", not "never" |
| MoE | the owner has not decided; `claws61821` argues opt-in; and routing is what collapsed SVC's 16-expert run |
| MLA / attention | violates "no bounded size"; it is off by default even in his own fork |
| `membench` P1–P10 as a programme | SVC's 34M run already answered most of it (no retrieval past 64 B, traces null, data-limited). Keep only the Δh retention probe — it is ~0 GPU and nobody has run it |
| re-doing eval state hygiene | finding 8: already correct |

---

## 4. What we keep from our own record

The ruler, the append-only discipline, the gates, the n-gram ceiling, and the
state-probe cue-distribution result. SVC retracted a 253× delimiter claim
("a plausible mechanism is not evidence") and then an expert-router collapse that
turned out to be a `te / 8` indexing bug. Upstream is about to start agent-driven
parallel layer-block search (`#34`) with no `RESULTS.md` and no equivalent.

**Our comparative advantage is exactly what they are about to need**, and it is
worth more than any single architecture experiment in this plan.

---

## 5. Falsifiers, stated in advance

- **P1/P2:** if throughput per byte at `B=64, T=16` is more than 2× worse than
  today's `B=1024, T=1` on the same device, the ring stays and P2 is abandoned.
  Record the number; do not tune T to rescue it. **MEASURED 2026-10-12 on the
  MX150: 3.7× worse, so the ring stays. But the falsifier was written for the
  ring-deletion decision and it also decided the T-axis decision, which it
  should not have: at constant B, T=16 is 1.09× the incumbent's throughput per
  byte. The two decisions are now separate — the ring stays, P1 proceeds at
  B=1024 — and the falsifier is restated below.**
- **P1 (restated):** the T axis is adopted at B=1024 if, on the same device, it
  is not more than 2× worse per byte than T=1 *at the same B*. It is 1.09×
  better. Independently of speed, it is adopted only if the held-out CE at
  matched BYTES does not fall, and the arm's reading is reported with its
  update count (T=16 buys 16× fewer updates for the same bytes — the two
  effects are not separable at one byte budget, and the plan should say which
  one the number is about).
- **P4 (capacity half, 0 GPU):** if the P4 init does not raise the untrained
  model's retention at d=128 above the default init's, the init was never the
  binding constraint and the WRITE PATH is (the state is a linear,
  order-losing EMA of raw byte embeddings). **MEASURED: 5.8% vs 0.00% — the
  capacity exists.** Gate: `tests/check_p4_capacity.py`.
- **P4 (usage half, the real one):** if a TRAINED model with the P4 init (and
  P6's `wd_decay=0`) still shows no retention past ~64 B, then the init was
  necessary but not sufficient, and the next lever is the write path
  (hierarchical wiring), not the init. This is the falsifier that decides
  whether to spend the next month on P3 or on wiring.
- **P2:** if the dense trace does not reproduce ring+tail credit where both are
  computable, the ring is doing something the closed form misses and the deletion
  is blocked.
- **P3:** if the gated arm does not beat its own `gate=0` control on the two-ruler
  gate at matched wall-clock, the ungated write is not the blocker and §3(iii) of
  the strategic review is wrong.
- **P5:** if MTP does not beat single-target CE at matched bytes, the n-gram
  deficit is not a multi-byte-structure problem and BPE deserves a second look.
- **Defined non-success for the whole plan:** if P1 lands, P2 deletes the ring, and
  the two-ruler CE does not move, then the remaining deficit is architectural
  (16 parallel readouts rather than 16 layers of depth) and the next plan is about
  layer wiring, not about batching, credit or optimization.

---

## 6. Not established

- **Nothing in this plan has been run.** Every number is from the CUDA fork, from
  upstream issues, or from our own `RESULTS.md`.
- **torch autograd at `T>1` is unmeasured here, and it is the single biggest
  unknown.** SVC's throughput comes from fusing the T-loop into 3 kernels per
  window; torch cannot do that. The expected gap is what P1's first measurement
  exists to price. **RESOLVED 2026-10-12, by construction: this engine has no
  autograd.** `theta` is a plain buffer, every gradient is hand-built, so there
  is no T-deep graph and no `B·T·D` activation memory; the plan's P1 memory
  prediction is wrong for this engine, and the "biggest unknown" was a
  property of a different engine. The unknown that replaced it is the GPU
  launch overhead at small B, which is now measured (finding 9).
- **The CUDA fork's MLX branch is Zig + MLX, not Python** (`tmt-mlx/src/cell.zig`,
  351 lines). The *algorithm* is portable; the code is not.
- The CUDA fork is 9,080 lines of CUDA and Zig with no torch equivalent. We take
  the math and the converged config, not the code.
- **THIS MACHINE HAS NO GPU** (torch 2.14.0+cpu, `torch.cuda.is_available()`
  False, no `nvidia-smi`), measured 2026-10-12. The incumbent at t=919,897 was
  trained elsewhere. **No training A/B is feasible here** — the ruler works only
  because it is a *frozen* forward. What this machine can still verify is
  correctness gates at small scale: P2's dense-trace-vs-ring equivalence is
  explicitly specified as "few hot dims, small B", and P3's gate is checkable by
  finite difference exactly as the CUDA fork did. The decisive P3/P4 A/Bs and
  every throughput number need GPU time; do not read a CPU number as the verdict.
  **SUPERSEDED 2026-10-12: there IS a GPU** — the other laptop, an MX150 with
  4 GB (142 GFLOP/s fp32, sm_61, driver 580, torch 2.14.0+cu126), reachable at
  `192.168.1.13`. The whole gate suite passes there, the T axis trains there
  (7400 B/s at T=16, 6116 B/s at T=1), and the A/B runs there. What it cannot
  do is the *full* incumbent run: the MX150 is ~3.3× slower than the T4 the
  889,897-step run used (~36 h here vs ~11.5 h there), so a full-scale arm
  belongs on the T4 and a one-hour arm belongs here.
- **We have never run above 4.5M parameters.** SVC has 34M trained (2.46B bytes,
  1.82 BPC) and 420M configured. Scale is the largest unmeasured axis in the
  project, and finding 5 says it is not the first lever.

---

## 7. What changed on 2026-10-12 (execution of P0, and four corrections)

The plan's own gates were run, and the plan was corrected four times. The
corrections are recorded here because **a plan that has been executed is a
different document from the plan that was written**, and the second one is
what the next person should read.

### 7.1 Landed

| item | state | gate |
|---|---|---|
| P0-1 CE-only | DONE | `tests/check_ce_only.py` |
| P0-2 `check_optimizations.py` repaired | DONE | 16/16 |
| **P0-3 NaN guard + grad clip** | **DONE** | `tests/check_safety.py` |
| **P6 per-group weight decay** | **DONE** (`--wd-decay`) | `tests/check_safety.py` |
| P1 `seqlen` as a knob, T=1 bit-identical | DONE | `tests/check_seqlen_credit.py` (CPU **and CUDA**), `tests/check_parity.py` |
| P2 dense trace (implemented; **the ring stays**) | DONE, rationale revised | `tests/check_dense_trace.py` |
| P4 log-spaced half-life init | DONE, **plus a capacity gate** | `tests/check_halflife_init.py`, `tests/check_p4_capacity.py` |
| P3 gated recurrence | PARTIAL — implemented, gradient UNVERIFIED, arm refuses | `tests/check_gated_recurrence.py` |

### 7.2 The four corrections

1. **§1's bet is backwards; P1 survives it.** Lowering B to delete the ring
   costs 3.7× per byte on a GPU, while the T axis it was buying is 1.09×
   *faster* per byte at constant B. Take T at B=1024; keep the ring; treat
   the dense trace as the `ring_k ≤ 256` fix (which the P4 capacity probe
   shows is a real limit) rather than as the point of the exercise.
2. **P4 is necessary, not sufficient.** The capacity probe confirms the
   capacity half of "the cause is the init" and refutes the sufficiency half:
   the trained incumbent retains *less* than a fresh P4 model. P4 becomes a
   2×2 with P6, and the usage half is the falsifier that decides whether the
   next month goes to P3 or to the write path.
3. **P5 does not need P1.** Upstream's MTP predicts K bytes ahead from the
   same representation; no window is involved. P5 can run in parallel, and
   upstream's implementation is a reference.
4. **The gate suite was a CPU suite, and nobody said so.** Four device bugs
   (the throughput probe, the P1 gate, the parity gate, and three gates that
   silently became CUDA gates on a GPU box) and five hardcoded absolute paths
   were found by running on the device the arms will actually run on. The
   lesson generalises past this repo: a gate that has only ever run on one
   device is a statement about that device.

### 7.3 The corrected priority order

1. **P0-3** (NaN guard + clip) — DONE.
2. **P6** (per-group wd) — DONE. It is a prerequisite for a P4 arm, not a
   cleanup: without it, the weight decay undoes the P4 init and the A/B
   measures the wrong thing.
3. **The P4 usage A/B** — `(wd_decay 0.01/0.0) × (default init / P4 init)`,
   with the retention ruler as the verdict, not only CE.
4. **P1's A/B at B=1024** — the T axis, measured at matched bytes with the
   update count reported.
5. **P7** (document resets) — the only credit condition upstream measured as
   paying.
6. **P3** (the gate) — the big one, and the one that answers #36's "pivot to
   Mamba" question, since a data-dependent retention factor IS the Mamba
   mechanism. Its gradient must be re-derived against the autograd reference
   before it is trusted.
7. **P5** (MTP) — independent of P1, and the answer to the n-gram deficit and
   to #28/#35.
8. **The write path** — promoted from §5's fallback to a named item, because
   the CUDA fork's hierarchical wiring is where its memory came from, and
   because a linear order-losing EMA of byte embeddings cannot represent
   "that name appeared 200 bytes ago" no matter how long its half-life is.
