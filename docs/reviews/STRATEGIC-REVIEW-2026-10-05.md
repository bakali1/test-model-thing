# Strategic review — the memory programme is aimed at a phantom (2026-10-05)

**Author:** deep-review subagent (agent "synthesis2"), verified by the orchestrator.
**Status:** review, not a plan. No experiment in this document has been run except
the n-gram ceiling, which was run by the review itself and every number
re-derived from the repo; reproductions are cheap and given.

This review supersedes the framing of `PLAN-2026-10-04-ROADMAP.md` v3. Where they
disagree, this review wins. The architecture judgement is reasoned, not measured —
the copy task in §4 is the falsifier.

---

## 0. Two premises in the project's own records are wrong

### 0a. The `/B` "decay-role defect" is not a defect — every group is B× the mean

There is no `/B` in the backward pass. Verified in both ports:

```
gpu_port.py:787   loss = (... + ce.sum() + ...) / B        ← the ONLY /B
gpu_port.py:906   decay_grads.append((dlds * l.decaytrace).sum(dim=0))   ← no /B
                  dWd = doutput.t() @ x ;  dx = doutput @ self.Wd        ← no /B
```

The entire 4,481,793-coordinate gradient buffer is **B× the mean gradient,
uniformly**. `RESULTS.md:1606-1616`'s claim that the decay rows are "exactly B×
too large *relative to every other parameter group*" is false — every group
shares the property.

Consequences:

- The SGD arm's `_gradfix` (`scratch-optimizer-study/_derived_gpu_port_sgd.py:1107`)
  divided **only the 16 .decay slices** by B, on a premise its own docstring
  states and that is false. The decay role therefore ran at an effective
  learning rate **B = 1024× smaller** than every other parameter — by
  construction of the harness.
- The celebrated finding — *"σ moved 1e-5 over 30,000 SGD steps; the memory role
  does not train under a magnitude-proportional optimizer; this is an
  ARCHITECTURE question"* — is an artifact of that harness. The experiment
  measured its own correction.
- The `1024×` figure quoted throughout `RESULTS.md` and the ROADMAP is the wrong
  sign. Commit `3f2faca`'s message ("the decay rows miss a /B") is wrong.
- "Harmless under AdamW" (scale-invariance of `m/√v`) remains true, but it is
  true of every group, not a decay-role property.

SGD is not established as "the optimizer that froze the memory role". If that
question is ever reopened, the SGD arm must run without `_gradfix` or with it
applied uniformly.

### 0b. The 2015 external anchor is a different paper on a different corpus

arXiv:1506.02078v2 is **Karpathy, Johnson & Fei-Fei, "Visualizing and
Understanding Recurrent Networks"** — corpora: **War and Peace + Linux Kernel
source**, not Penn Treebank. The ~230-character figure is one cell off a figure;
the ~60-character brace figure is on **Linux kernel source**, and closing a brace
is a **depth-counter** task — the paper's own words: *"one cell is clearly acting
as a line length counter."* It is not long-range *associative* retrieval; it is a
small-integer counter with ~60-step range, on the one corpus of the pair with
genuine long-range structure.

The ROADMAP's treatment of it as a ceiling for *associative* memory is stronger
than the paper supports. (The project's own shelf note — `1506.02078v2.IDEA.md`
§3 — already said this; it was not propagated.)

---

## 1. The actual problem: the ruler is blind to memory

Every arm in the memory programme wrote to `--lr` and read a number that is
structurally insensitive to the quantity under study. That is why σ never moves,
and it would not move at any learning rate.

Measured (the experiment the repo had never run — what memory does the corpus
actually reward?):

An order-*k* byte n-gram is the sufficient statistic for a model with exactly
*k* bytes of memory, so its held-out CE is the Bayes-optimal score for a
*k*-byte-memory model. Held-out TinyStories, 8 MB train / 300 kB test, disjoint:

| memory k | held-out CE (nats/byte) |
|---|---|
| 1 B | 2.6455 |
| 2 B | 2.1687 |
| 4 B | 1.3824 |
| 6 B | 1.0284 |
| **8 B** | **0.9420** |
| 12 B | 0.9687 (worse) |
| 16 B | 0.9882 (worse) |

(With more training data the curve is **0.8096 at k=8**, **0.7737 at k=12** —
flat past 8 bytes.)

> **An order-8 byte n-gram — a lookup table, zero learned parameters, 8 MB of
> training — scores 0.81 nats/byte on TMT's held-out set.**
> **TMT — 4,481,793 parameters, 889,897 steps, ~912 M bytes seen — scores 1.2094.**
> The model loses to a memorisation table on its own training distribution by
> **0.40 nats (33%)**, using 0.9% of the data. The gap is a lower bound (8 MB vs
> 912 MB).

Directly observable cause — fraction of held-out positions whose *k*-byte context
already occurs in training (2 MB train, official `tinystories_valid.txt`, disjoint):

| k | context seen in train | next byte determined by it |
|---|---|---|
| 4 | 99.4% | **52.0%** |
| 8 | 87.1% | **58.0%** |
| 12 | 57.4% | 42.5% |
| 16 | 34.2% | 25.9% |

**Over half of all held-out positions have their next byte determined by four
bytes of context from 2 MB of training.** TinyStories is a template library.
Next-byte prediction on it is a lookup that saturates at ~8 bytes.

Consequences:

- The entire reward for horizon 8 B → 64 B is **< 0.03 nats** — below the 0.03
  bar the repo uses for "real". The ruler cannot see memory.
- No optimiser, estimator or architecture change can produce long memory that is
  *used*, because the training signal that would reward it is ≈0.03 nats. The
  ladder was aimed at a nonexistent bottleneck.
- The in-domain/cross-domain gap (FS-W vs FS-N: −0.5469 in-domain, +0.96 on
  novels) is a **repetition gap, not a long-range-structure gap** — TinyStories
  is more memorisable, and memorisation does not transfer.
- The shelf's `INDEX.md` says it: *"No paper on this shelf tests a byte-level
  recurrent model."* S4D's long-range results come from tasks with *enforced*
  long-range structure (LRA, BIDMC 4000-step, sCIFAR 1024). Nothing on the shelf
  measures a recall horizon on language.

---

## 2. Two problems, and the second is bigger than the memory question

1. **The task does not require long memory** — so the memory programme is aimed
   at nothing.
2. **The model is far from the ceiling on the easy task.** It loses to an 8 MB
   lookup table by 0.40 nats. That is a larger deficit than the memory question
   and nothing in the programme has measured it.

---

## 3. Architecture judgement

**Sound as a short-horizon smoother; structurally unsound as a long-horizon
store.** Sound: diagonal linear recurrence with per-channel learned decay is a
legitimate S4D-shaped choice; the half-life framing is correct and interpretable;
the eligibility-trace ring buffer is exact (cosine 1.000000 at 16 layers, both
regimes) and is the best engineering in the repo.

Three mechanisms kill the long-memory thesis:

**(i) Initialised at, and weight-decayed toward, the degenerate point.**

```
gpu_port.py:485  lim = 1.0/sqrt(dim)          # 0.0442
gpu_port.py:499  d0 = uniform(-lim, lim, (dim,))
gpu_port.py:955  th.sub_(lv * (adaptive + self.wd * th))   # wd=0.01 UNMASKED
```

`sigmoid(±0.0442) ∈ [0.489, 0.511]` → half-life 0.965–1.035 B. All 8,192
channels start at 1.000 ± 0.035 bytes. Weight decay applies to the flat vector
with no exception for the decay slices, and `sigmoid(0) = 0.5` = half-life exactly
1 byte, so wd is a systematic pull toward *no memory*, anchored at init.

Re-derived from `tmt.npz` (not quoted): logit min −1.00, mean 0.114, std 0.227
(init std 0.0255) — the logit distribution widened 5×, but `n > 2 B: 135 (1.6%),
n > 4 B: 11, n > 8 B: 2, n > 32 B: 1`. **This is a random walk around a
weight-decay anchor at its initialisation value** — noise-driven diffusion under
a restoring force. It responds to nothing because there is nothing in the
gradient to respond to: the corpus says 0.03 nats.

So "the decay parameters essentially never move" is right in effect and wrong in
mechanism: they wander, diffusively, around the point where they started.

**(ii) Layer-norm cancels the amplitude advantage of a long half-life — 
closed form.**

For `s_t = a·s_{t−1} + x`, `Var(s) = Var(x)/(1−a²)`. Layer-norm divides exactly
that back out, so a cue written at t=0 contributing `a^d·c` has post-LN relative
amplitude

    f(a,d) = a^d · sqrt(1−a²),    maximised at  a*² = d/(d+1)

| distance d | optimal half-life | max cue amplitude | at σ≈0.55 |
|---|---:|---:|---:|
| 8 B | 11.8 B | 0.208 | 0.0070 |
| 16 B | 22.9 B | 0.149 | 0.000059 |
| 32 B | 45.1 B | 0.106 | ~0 |
| 64 B | 89.4 B | 0.0755 | ~0 |
| 1024 B | 1420 B | 0.0189 | ~0 |

(Scaling constant: the report said `1.39/√d`; the correct constant is ≈ `0.59/√d`.
The table values are right; only the stated constant was off, and no conclusion
depends on it. The scaling law `~1/√d` is the point.)

Two consequences, both fatal:

1. The best available amplitude decays like **~1/√d no matter what**. Tuning σ
   cannot buy long memory. This is an architecture-intrinsic horizon — a
   property of layer-norm applied to an ungated linear recurrence — that no decay
   initialisation, gate, or log-spacing can beat.
2. At the model's σ=0.55 the amplitude at d=8 is **30× below its own ceiling**,
   and at d=16 is **2500× below**. Even perfect tuning only buys 0.007 → 0.208 at
   d=8; it does not reach d=64.

A longer half-life is a **net loss** under this readout: zero amplitude gain (LN
removes it), full noise-accumulation cost. You must gate the write (Mamba's
`Δ_t`) or move memory outside the normalisation.

**(iii) 8,192 diagonal channels is the wrong shape; the budget is a red herring.**

`state = states*decay + enc` writes every byte into all 512 channels with weight
exactly 1.0 — no gate, no key, no `B_t`/`Δ_t` anywhere in the forward pass
(verified in full). A diagonal EMA is a **low-pass filter, not an associative
store**. Retrieval requires addressing ("what did I write where `XQ7P`
appeared?"); TMT can only ask "what is the exponentially-weighted average of
everything recent?" Those are different questions, and no allocation of 8,192
diagonal channels answers the second. `ARCHITECTURE_IDEAS2.md §16.1`'s "secret
symbol at d=10⁶" is an attention test wearing a memory costume: there is no key
in the model to look up with.

Verdict: **the defect is not σ. It is the ungated, non-addressable write.**

---

## 4. The breakthrough experiment: the copy task

Byte stream = uniform random bytes; a **4-byte key** planted at position 0; then
*d* random distractor bytes; then the **same 4-byte key again**; the model must
predict the 4 bytes that follow the second occurrence. Sweep d ∈ {8, 32, 128,
512, 2048}. Chance = 1/256 per byte.

Why: it is the only instrument that makes the ruler sensitive. Uniform random
distractors make the task memory-obliterating — local frequency statistics are
worthless *by construction*, so a model that ignores the d=2048 cue scores exactly
chance and cannot game the loss. It is ~10 MB and trains in minutes. It
adjudicates §3(ii) directly.

**Stage 1 — one hour of CPU, no training:** take the existing `tmt.npz`, zero the
state, plant the 4-byte key, feed *d* random distractors, probe with
`tools/state_probe.py`.

| Result | Conclusion | Action |
|---|---|---|
| above chance at d=512+ | architecture fine; corpus was the whole problem | fix the data; nothing architectural |
| dies at d≈8–16 matching `f(a,d)` | architecture is the constraint; LN + ungated write is the mechanism | gated write (`Δ_t`), strip wd from decay logits, log-spaced init (S4D), memory outside the LN |

**Stage 2** (only if stage 1 fails): train on it, ~2 GPU-hours across the sweep.

**Falsifier, stated in advance:** if a trained model beats chance at d=2048, the
layer-norm ceiling is wrong and §3(ii) collapses.

---

## 5. Stop doing these now

1. **The global `--lr` sweep.** Dead. SNR's 12.6× frozen/moving gap is a genuine
   finding about jitter, but it is not the memory gate. The sweep has already
   been run six times; it moves CE, not σ.
2. **The SGD optimizer axis.** Already marked "no further arms" — and now doubly
   dead: its central conclusion rests on the false `/B` premise that crippled the
   decay role by 1024× (§0a). The arm measured its own harness.
3. **Any further `--lr-decay` work.** The knob moves σ, and σ is not the blocker.
   Worse, `--lr_decay` scales *weight decay with it* (`gpu_port.py:960-966`) —
   the one knob that could help is wired to a trap.
4. **All T-RTU / `ARCHITECTURE_IDEAS2.md` work.** Built on the "8–64 B needle
   wall" premise that §1 kills; §16.1 is an attention test. At K=16 the write
   projection alone is 4.2 M params — nearly doubling the model — and `latent =
   Σ read·state` sums slots before any normalisation, so a slow slot numerically
   drowns a fast one. Defer until the copy task exists.
5. **Further `--cue uniform` state-probe sweeps** without recording the cue mode.
   Fix `--cue corpus-support` as the default first; the cue distribution sets the
   horizon, so probing without recording it is measuring the stimulus.

---

## 6. Log hygiene (cheap, two items actively misleading)

- `RESULTS.md` header index still quotes "mean half-life 1.14 B; 7 of 8192 dims
  > 8 B" from `armsgdlr0.000125`, a checkpoint that was never probed. Update the
  header or label the source.
- Three different quantities share the label "hot dims" in the record (104–172 /
  1669–1895 / 7–9). Pick one name per quantity.
- The 8 B horizon claim: the repo's own `--cue` correction notes it is partly a
  property of the stimulus. Quoting "8 B" unqualified is quoting a withdrawn
  claim; quote it with its cue mode or not at all.

---

## 7. Verification

The orchestrator independently verified §0a (the only `/B` is on the loss; the
whole buffer is uniformly unscaled; `_gradfix` divides only the decay slices) and
the §3(ii) closed form (`a*² = d/(d+1)`, table values, ~0.59/√d constant). The
n-gram numbers come from `lag5.py` (pure-Python, no hashing; `lag2/3.py` contain a
rolling-hash aliasing bug and are superseded). Reruns: build `data/eval` + 8 MB of
`wiki_tinystories_*.txt` head as train, 300 kB of `tinystories_valid.txt` as
held-out, MAXORD=12, interpolation weight λ with geometric falloff.