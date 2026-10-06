# PLAN — answer the architectural question: retention → selectivity → retrieval (2026-10-06)

**Author:** external review (accepted in part, see
`docs/reviews/RESPONSE-2026-10-06-EXTERNAL-CRITIQUE.md`; planning notes by the
orchestrator session).
**Status:** plan, not execution. Priority 1 (retention probe) is implementable
with ~0 GPU and is the only item with an implementation started (`tools/copy_probe*.py`
were the first attempt; their result is missing and the design below supersedes them).

This plan supersedes the "improve TinyStories CE" framing of
`PLAN-2026-10-04-ROADMAP.md` for the memory axis. It is the operational half of
rev 2 of `docs/reviews/STRATEGIC-REVIEW-2026-10-05.md`.

> **One sentence:** TMT stops asking *"does TinyStories CE improve when memory
> changes?"* and starts asking *"can the recurrent state retain, protect, and
> retrieve information?"* — with a ruler for each verb.

---

## 0. Why this direction

Corrected evidence (rev 2 of the strategic review):

- The SGD memory result was invalidated by the decay-only `/B` fix (harness
  artifact). No optimizer conclusion survives that event.
- The LN argument is a per-scalar-channel bound, not a network proof — the
  network question is open.
- The n-gram ceiling shows the CE ruler is blind to memory, not that memory is
  absent.
- The decisive corrected copy probe never returned; the architecture question
  is **OPEN**.

None of this justifies more optimizer variants or decay-LR tweaks. It justifies
measuring the capability directly.

---

## 1. Priority 1 — causal state-retention probe (~0 GPU)

Highest-value change. No training.

**Protocol.** Take an existing checkpoint and feed two sequences that are
identical except for the initial 4-byte key:

```
sequence A: KEY_A + d random distractor bytes
sequence B: KEY_B + exactly the same distractor bytes
```

At each distance d, measure the causal state difference

```
Δh_d = ‖h_d^A − h_d^B‖
```

at four stages independently:

1. raw recurrent state (before layer-norm)
2. normalized state (after layer-norm)
3. post-nonlinearity output (after the SiLU block)
4. final logits

Plot `d → Δh_d` and compare against the scalar prediction `a^d·√(1−a²)`
(half-life from the checkpoint's actual σ).

**Reading.** Three different possibilities, three different conclusions:

```
cue disappears from state          → the representation is destroyed (LN/reset suspect)
cue survives in state, output      → retrieval/training problem, not storage
ignores it
cue survives and output responds   → architecture already has memory; TinyStories
                                    simply was not rewarding it
```

**Decision rule:**

- Δh dies by d≈16 → the architecture genuinely suppresses temporal perturbations;
  the LN-ceiling hypothesis is strengthened; investigate normalization placement.
- Δh alive at d=512 → the LN-ceiling hypothesis is weakened; LN is not the
  bottleneck; move to priority 2 immediately.
- The before/after-LN comparison localizes exactly where information disappears.

**Status of prior art:** `tools/copy_probe.py` (confounded control) and
`tools/copy_probe_fix.py` (corrected control, shipped 2026-10-05 19:33) ran on the
MX150; the corrected result **never returned** (runner aborted). This probe is the
Δh version of the same idea and supersedes both scripts — the KL-over-conditions
design is replaced by a direct state-norm difference, which needs no control
conditions at all.

## 2. Priority 2 — synthetic memory benchmark suite (~0 to set up)

Keep TinyStories as the language benchmark; add a dedicated long-memory suite:

```
A. delayed copy          — fixed known gap
B. associative copy      — KEY → PAYLOAD → distractors → KEY → predict PAYLOAD
C. selective copy        — random gaps + irrelevant key/value pairs
D. long-range language   — structured sequences with non-local dependencies
```

**The key→payload task:**

```
KEY      PAYLOAD
AB7Q     91F2

[random 8..2048 bytes]

AB7Q  →  predict 91F2
```

- gap d random per sample, drawn from {8, 32, 128, 512, 2048};
- payload **random and independent of the key** (prevents collapse into pattern
  recognition);
- distractor bytes uniform random, so local frequency statistics are useless
  and chance = 1/256 per byte cannot be gamed.

**The selective variant** (the Mamba-style discriminator — S4 18.3% vs selective
SSM 97%+ on the induction-style task):

```
A → X
B → Y
C → Z

[random noise]
[random unrelated keys]

B → ?
```

Random gap + distractor key/value pairs turn the question from "did the state
wait the right number of steps" (fixed dynamics can do that) into "did the state
decide what to preserve" (requires content-dependent retention). That is the
explicit hypothesis under test.

## 3. Priority 3 — CE-only training for the memory task (low GPU)

For the synthetic memory benchmark use **CE only**:

```
L = CE
```

Temporarily strip: variance hinge, JEPA/embed-MSE, KD, end-of-text loss.

Not because these losses are wrong — because they make the experiment ambiguous.
With L = CE only, "copy task succeeds" means *the recurrence + optimizer can
learn the capability*. Add auxiliary losses back one at a time only after that
works. This also resolves the Q3 gradient-interaction questions more cleanly than
any TinyStories run could.

## 4. Priority 4 — checkpoint / eval / state handling guarantees (~0 GPU)

Make these explicit, everywhere:

```
checkpoint:
    model state
    optimizer state
    recurrent state
    step
    data position
    RNG state
```

```
every evaluation:
    reset recurrent state
    never update weights
    fixed corpus split
    fixed sample IDs
```

Motivation: the public issue trackers documented (a) `benchmark.py` leaking
recurrent state across independent examples and accumulating MCC on the training
stream, and (b) the independent PyTorch port finding that overwriting
`decay.grad` with `dlds * trace` breaks gradient accumulation — the historical
trace contribution must be accumulated correctly, and the port added resumable
accumulation + atomic checkpointing. An evaluator that leaks state makes a
recurrent project look better or worse than it is; none of the memory numbers
below are trustworthy without these guarantees.

## 5. Priority 5 — separate decay weight decay (small controlled experiment)

Current: `θ_decay → σ(θ)`, AdamW adds `+λθ`. Since `σ(0) = 0.5` → one-byte
half-life, this is an explicit prior toward the *shortest*-memory point — odd
for a model whose central feature is learned memory.

Experiment: **two parameter groups**

```
ordinary weights: wd = 0.01
decay logits:     wd = 0
```

Do **not** interpret success/failure immediately. The only question answered:
*does removing the artificial restoring force change the learned timescale on a
task that actually requires long memory?* (See the `E[g_θ|θ]` vs `−wd·θ`
discriminator in `RESPONSE-2026-10-06-EXTERNAL-CRITIQUE.md` §0/1.5 — run that
measurement on TinyStories before or alongside, it is ~free.)

Note: do **not** do more initialization-only experiments on TinyStories (the
log-spaced init result already exists; the ruler was the problem). The S4D
parameterization lesson applies once the benchmark rewards memory, not before.

## 6. Priority 6 — long-memory real data (medium GPU, only after synthetic works)

Then change the training distribution — do not abandon TinyStories:

```
Stage 1: synthetic associative memory
Stage 2: synthetic + long structured sequences
Stage 3: long-form language / books / code
Stage 4: TinyStories + long-memory mixture
```

Hard requirement: the long-memory component must contain dependencies whose
answer **cannot be determined from the last few bytes** — otherwise this recreates
the TinyStories problem under another name.

## 7. Priority 7 — selective-write gate (medium GPU, only if 1–3 fail)

**Do not implement first.** First establish whether the existing recurrence can
retain information on the controlled task — otherwise a gate may just be
compensating for a bad training distribution.

Current write: `s_t = a ⊙ s_{t−1} + x_t` — every byte into every channel with
weight 1; noise continually injected.

Minimal extension (content-dependent writing, Mamba-flavoured):

```
g_t = σ(w_g ⊙ x_t + b_g)          ← diagonal gate
s_t = a ⊙ s_{t−1} + g_t ⊙ x_t
```

Cost: `16 × 512 × 2 ≈ 16k` parameters. **Not** a full `512×512 W_g` initially
(that is ~4.2M params across 16 layers — nearly doubling the model on a first
experiment). Preserves: streaming, O(1) state in sequence length, tiny parameter
overhead, no attention, diagonal recurrence. Adds: content-dependent writing.

## 8. Priority 8 — normalization (conditional)

Do **not** change layer-norm yet. The `f(a,d) = a^d·√(1−a²)` result is a useful
sanity calculation, not an impossibility theorem for a 512-dim nonlinear
network. Decision tree:

```
measure cue retention
   ↓
does information disappear at LN?
   ↓
yes → investigate normalization placement
no  → LN is not the bottleneck
```

## 9. Priority 9+ — optimizer and scale (later)

Freeze the optimizer question now:

```
AdamW
beta1 = 0.9
beta2 = 0.999
eps   = 1e-8
wd    = 0.01
```

One change at a time, after the memory capability is established. New optimizer /
model scaling only after priorities 1–8.

## 10. Memory evaluation API

`eval.py` stops having one concept called "memory". Separate functions:

```
retention_curve(d)
selective_copy_accuracy(d)
associative_retrieval_accuracy(d)
language_ce()
transfer_ce()
cross_domain_ce()
```

Every memory result reports:

```
cue type
gap distribution
distractor type
state reset policy
seed
sample count
chance level
```

The cue type matters enormously — **"8 B memory" without saying what stimulus
produced that number disappears from the project vocabulary** (rev-2-corrected,
per the strategic review).

---

## Priority summary

| # | Change | GPU cost | Priority |
|---|---|---|---|
| 1 | Causal state-retention probe (Δh) | ~0 | **1** |
| 2 | Synthetic associative/selective-copy benchmark | ~0 initially | **2** |
| 3 | CE-only memory training | low | **3** |
| 4 | Correct checkpoint/eval/state handling | ~0 | **4** |
| 5 | Decay weight-decay separation | low | **5** |
| 6 | Long-memory real-data mixture | medium | **6** |
| 7 | Selective write gate | medium | **7** |
| 8 | Change normalization | only if probe points there | **8** |
| 9 | New optimizer | later | **9** |
| 10 | Scale model | much later | **10** |

## Architecture (unchanged until 1–3 say otherwise)

```
byte → encoder → recurrent state (s = a*s + write) → normalization
     → nonlinear layer → output
```

Target (only after the benchmark exists):

```
featured: MEMORY TEST SUITE  →  retention / selective / retrieval  →  language eval
hedged:   current write → content-dependent gate → selective memory
```