# Response to external critique of the strategic review (2026-10-06)

**Status:** accepted in part. This document records the point-by-point verdict on an
external review of `STRATEGIC-REVIEW-2026-10-05.md` (rev 1, commit `d11e7ee`). The
review document was revised in place (rev 2) to remove the overclaims; this file
keeps the full record of what was challenged, what stands, and what is now open.

**Bottom line:** the critique is accepted on its three main corrections and rejected
on none of substance. The single most important rewrite it forces:

> **Do not replace "the optimizer is the problem" with "the architecture is the
> problem" yet. Replace both with "we have not been measuring the capability we
> care about."**

That replaces the rev-1 headline ("the memory programme is aimed at a phantom") with:

> **The programme conflated three questions and almost everything measures only
> the first.**
>
> 1. Does the corpus contain enough long-range predictive structure to reward
>    memory? — the TinyStories CE experiment answers mostly this.
> 2. Does TMT's state physically preserve information over long delays? — almost
>    nothing directly answers this.
> 3. Can TMT select and retrieve the relevant information after distractors? —
>    the literal-cue probe answers a narrow version.
>
> The programme should become **retention → selectivity → retrieval**, and each
> leg needs its own ruler.

---

## 0. Facts first: one measurement the critique treats as closed never landed

The corrected copy probe — `tools/copy_probe_fix.py` (shared distractor stream,
same second key, differing only in the first 4 bytes, permutation null) — is the
decisive retention measurement for the layer-norm question. It was shipped to the
MX150 at 19:33 on Oct 5, and **its results never returned**: the runner session
(`2026-10-05T19-03-43…)` ends with `Command aborted`; the numbers are absent from
`RESULTS.md`, from `runs/`, and from every later session. The scripts sit
uncommitted in `tools/`.

Consequence: the architecture verdict is **OPEN**, not decided — which is
consistent with the critique's main correction.

---

## 1. What stands, verified against the repo

### 1.1 `/B` invalidates the SGD result — remove it, don't downgrade it

Verified independently: `gpu_port.py:787` is the *only* `/B` in the backward pass
and it is on the loss; `decay_grads` (`:906`) and the `dWd`/`dW`/`dx` lines carry
no `/B`. The whole 4,481,793-coordinate buffer is uniformly B× the mean gradient.
`_gradfix` (`scratch-optimizer-study/_derived_gpu_port_sgd.py:1107`) divided only
the 16 decay slices by B, giving the decay role an effective learning rate
**1024× smaller** than every other parameter. The result that followed — "σ moved
1e-5 over 30,000 SGD steps → architecture question" — measured the harness's own
correction. **Removed from the evidence base entirely** (not merely downgraded),
matching the critique's instruction. Runnable as a check:
`git show 4e4fdc4` (the commit claiming "the decay rows miss a /B").

Adam's per-coordinate scale invariance is why the bug hid under AdamW for 889,897
steps and exploded only under plain SGD — exactly the critique's secondary point.

### 1.2 Karpathy correction — agreed, with the critique's better use of the paper

`arXiv:1506.02078v2` is Karpathy, Johnson & Fei-Fei on War and Peace + Linux
kernel. The ~60-char brace cell is a **depth counter**, not associative retrieval.
The critique's upgrade is adopted: the paper is still relevant to TMT, but as
evidence that **a small recurrent model can learn a low-dimensional persistent
variable whose lifetime far exceeds the local prediction statistics** — supporting
the "learned timescale" thesis without being a quantitative 230-byte ceiling.

### 1.3 The n-gram result is the biggest strategic finding — with its correct reading

The important statement is:

> The current held-out CE benchmark is dominated by dependencies that an 8-byte
> finite-context model already captures.

not "8-byte memory is all TinyStories needs". The curve
(2.6455 → 2.1687 → 1.3824 → 1.0284 → 0.9420 through 8 B, then flat/worse) means
**ordinary next-byte CE has stopped being a useful memory-horizon ruler** by the
regime the programme cares about. The valid conclusions:

- "TMT has not demonstrated memory beyond the part of the corpus that contributes
  substantial average CE" — correct, defensible.
- "TMT has a large easy-task optimization/modelling deficit" (loses to an 8 MB
  n-gram by 0.40 nats on 0.9% of the data) — a **separate problem** from memory,
  and a larger one.
- "No optimiser can make long memory show up in this number" — yes.
- "Long memory cannot be learned" — **no**, not established (see §2.1). A
  dependency that is decisive when it occurs but occurs at only 0.5% of positions
  adds almost nothing to average CE yet can be a real capability. This is exactly
  why synthetic task-level rulers exist.

### 1.4 Variance hinge is a floor, not a penalty — already in the record

`gpu_port.py:776-785`: `var_active = (1.0 - sqrt(xv)) > 0`; the term is
`max(0, 1−√var)`, which pushes latent variance **up**. RESULTS.md already
retracted "the hinge penalizes variance". The critique's refinement is kept:
"the hinge may indirectly oppose CE's desired state dynamics" remains an open,
cheap check (`g_hinge^{decay}` vs `g_CE^{decay}`).

### 1.5 Weight-decay anchor: plausible, not demonstrated — discriminating test specified

The facts (init `lim = 1/sqrt(dim)` = 0.0442, `sigmoid(±0.0442) ∈ [0.489,0.511]`
→ half-life ≈1 B; `wd=0.01` applied uniformly to `th` including decay logits) are
confirmed. So is the trained distribution: logit std widened 0.0255 → 0.227 but
only 1.6% of channels exceed 2 B. But "diffusion around a weight-decay anchor"
remains **one interpretation of two**: the alternative is `E[g_θ|θ]` itself
pointing toward θ=0 (a loss-gradient attractor). Those need different fixes. The
critique's discriminating measurement is adopted:

> Bin θ, plot `E[g_data | θ ∈ bin]` against `g_wd = wd·θ`; if
> `E[g_data] ≈ −g_wd` it is an optimization balance, if `E[g_data] ≈ 0` it is a
> loss-gradient attractor.

Corroboration that the architecture *can* hold long state: the repo's own trace
gate measured a **14.27 B half-life with exact trace cosine 1.000000** under
`--decay-init 3` (`RESULTS.md`), versus 1.14 B at default init — long half-lives
are reachable; the search just never goes there. Consistent with the critique's
point that nothing here proves architectural impossibility.

---

## 2. The overclaims, corrected in rev 2

### 2.1 "≈0.03 nats ⇒ no optimiser/estimator/architecture change can produce used long memory" — rejected

Valid: *average TinyStories CE provides very little aggregate incentive for
dependencies beyond ~8 bytes*. Invalid as written: average CE underweights
rare-but-decisive long dependencies, so the sentence overreaches from "the ruler
does not reward it" to "it cannot exist". Rev 2 language:

> The CE ruler provides very little aggregate incentive for dependencies beyond
> ≈8 bytes; this is a statement about the ruler, not a proof that long-range
> capability cannot be learned. Frequency dilution is why synthetic memory tasks
> are required.

### 2.2 The layer-norm "absolute 1/√d ceiling" — rejected as a network proof

The calculus is right: for `s_t = a·s_{t−1} + x`, `f(a,d) = a^d·√(1−a²)`,
maximiser `a*² = d/(d+1)`, asymptotic `f_max(d) ≈ 0.607/√d`. What it does **not**
establish:

- The scalar-channel bound ignores the channel count. The state has **512 channels
  per layer** (`gpu_port.py:408`: "512 coords each = 8,192 total");
  a cue distributed over r independent channels gains ≈√r in aggregate signal.
  At r = 512, d = 64: `0.607·√(512/64) ≈ 1.72` — above the noise floor. The
  "cannot survive" reading is destroyed by the critique's own arithmetic.
- The model has **heterogeneous decays** `a₁…a₅₁₂` per layer: one channel can be
  tuned for d=8, another for d=64, another for d=512 — the S4D parameterisation
  point (Gu et al. 2206.11893, already on the shelf as
  `docs/ideas/papers/2206.11893v2.IDEA.md`).
- 16 layers of nonlinear mixing and a dense readout follow the recurrence.

The honest residual concern, kept in rev 2: layer-norm normalises the **aggregate**
state vector, and long-decay channels carry the largest variance
(`Var ∝ 1/(1−a²)`) — so the network's real competitor may be
**variance-channel-swamping under aggregate normalisation**, a different and
weaker claim than "no information survives". The Δh retention probe (§3.1) is
exactly what separates these, and it is unrun.

### 2.3 "Ungated diagonal memory cannot be associative" — too strong; selectivity replaces impossibility

`stateᵗ = decay⊙stateᵗ⁻¹ + enc(xᵗ)` (verified `gpu_port.py:218`) is not a
key→value→slot store, and irrelevant writes do interfere. But "not explicitly
addressable" ≠ "cannot perform associative recall": a learned encoder can
distribute key/value structure into the state and downstream nonlinearity can
operate on the superposition. The question is **capacity and interference**, not
logical possibility. The testable replacement (Mamba's selective-copy, §2312.00752)
is adopted:

> **TMT may have enough passive temporal persistence but insufficient selectivity
> to protect relevant information from irrelevant writes.**

### 2.4 Stage-1 (frozen checkpoint fails copy ⇒ architecture constrained) — not a valid falsifier

A TinyStories-trained checkpoint was never optimised to encode arbitrary random
4-byte keys; chance-level copy accuracy proves almost nothing about architectural
possibility, and a pass wouldn't be clean either. Rev 2 replaces it with the
retention measurement below.

### 2.5 Copy task in the review was incomplete — payload + selective version adopted

The critique's two corrections to the copy task are correct and adopted:

1. **Add a payload:** `KEY → PAYLOAD → distractors → KEY → predict PAYLOAD`,
   payload random and independent of the key (otherwise it can collapse into
   pattern recognition). The review's bare "repeat the 4 bytes after the key" was
   a re-appearance pattern, not associative retrieval.
2. **Randomise the gap and add distractor key/value pairs:** vanilla fixed-gap
   copying can be solved by time-aware fixed dynamics; **selective copy** — random
   gaps, irrelevant pairs — forces the model to decide what to preserve. Mamba's
   S4 baseline scores 18.3% on the induction-style version vs 97%+ for the
   selective SSM; that is the discrimination TMT's architecture question needs.

---

## 3. What this changes operationally

### 3.1 The proper Stage 1: retention (Δh), not copy — no training, ~30 min

Frozen checkpoint. Two identical sequences differing only in the initial key.
Measure Δh_d, the state difference, as a function of d — before layer-norm, after
layer-norm, after the nonlinear block, and at the logits — and compare against the
scalar prediction `a^d·√(1−a²)`.

- dies by d≈16 → the architecture genuinely suppresses temporal perturbations
  (LN ceiling strengthened);
- alive at d=512 → the LN-ceiling hypothesis is weakened;
- the before/after-LN comparison localises the suppression.

This is the measurement the rev-1 review needed but did not run. `copy_probe_fix.py`
was the first attempt and its result is missing; the corrected run is the Δh
version of the same idea. **This is the highest-priority next experiment.**

### 3.2 The programme: retention → selectivity → retrieval

1. **Retention** — frozen model, Δh_d retention curve (above).
2. **Selectivity** — train on selective-copy data with random gaps and irrelevant
   distractors; tests whether the architecture can protect relevant writes.
3. **Retrieval** — exact payload accuracy and CE at d ∈ {8, 32, 128, 512, 2048}
   with random gaps. This finally creates a ruler whose target **is memory
   itself**, replacing the hope that memory happens to move average TinyStories CE.

### 3.3 Still stopped (unchanged from rev 1)

Global `--lr` sweeps (move CE, not σ), the SGD axis (its conclusion rests on the
false `/B` premise — dead on its own evidence), `--lr-decay` (it scales weight
decay with it), T-RTU/IDEA2 build-out (premise unmeasured, not refuted as a
capability; test selectivity first), unlabeled `--cue uniform` probes.

---

## 4. Final judgement table (rev 2)

| Claim in rev-1 review | rev-2 verdict |
|---|---|
| `/B` decay-SGD finding was a harness artifact | **Agree; removed from evidence base** |
| Karpathy misused as memory ceiling | **Agree; kept as timescale-persistence evidence** |
| TinyStories CE is a poor long-memory ruler | **Strongly agree** |
| 8-byte n-gram exposes a huge easy-task deficit | **Agree** |
| TinyStories therefore contains no useful long-range signal | **Too strong → "unmeasured by the ruler"** |
| Decay logits pulled toward 0 by weight decay | **Agree** |
| Trained decay distribution = diffusion around that anchor | **Plausible, not demonstrated → E[g\|θ] check** |
| LN cancels raw benefit of high decay | **Conditionally agree (scalar channel)** |
| LN creates an absolute 1/√d architectural ceiling | **Reject as network proof; √r + heterogeneity reopen it** |
| Ungated diagonal memory is weak for associative recall | **Agree (capacity/interference)** |
| Ungated diagonal memory cannot perform associative recall | **Too strong → selectivity hypothesis instead** |
| Copy/selective-copy is the right benchmark | **Strongly agree (with payload + random gaps)** |
| Frozen checkpoint failing copy proves architecture failure | **Reject** |
| Corrected retention probe can discriminate the architecture | **Yes — highest priority, currently unrun** |

---

## 5. Open items (checked against the repo, still missing)

1. **Corrected copy-probe result** — shipped, never returned (`Command aborted`).
2. **Δh retention curve** — proposed, never run.
3. **E[g_θ | θ] vs −wd·θ** — proposed, cheap, never run.
4. **`g_hinge^{decay}` vs `g_CE^{decay}`** — proposed, cheap, never run.
5. **Selective-copy training run** — not started.