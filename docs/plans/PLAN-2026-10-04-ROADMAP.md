# PLAN — roadmap v2: credit → SSM → memory ruler (2026-10-04)

**Date:** 2026-10-04
**Status:** synthesis, not execution. Every item below is ranked against the
measured blockers in `../RESULTS.md` and `../ideas/ARCHITECTURE_IDEAS2.md`.

> ## ROADMAP v3 — 2026-10-04, supersedes the v2 ordering below
>
> v3 is a **reordering**, driven by two measurements taken after v2 was written.
> The v2 body is retained below for its evidence chains, its rejected-ideas
> table, and its sources map. **Where the two disagree, v3 wins.**
>
> ### What changed the order
>
> **1. The SNR blocker is largely a measurement artifact.** The standing
> "SNR ≈ 0.05" is read by `telemetry.py` while θ moves every byte. The *same
> estimator* at a **frozen** θ on the same checkpoint and batch reads **0.633** —
> a **12.6× gap**. The diagnosis is therefore **over-large step**, not a noise
> floor. This removes "fix the gradients" from first place and *adds* an axis
> nobody has run: the **global** learning rate. See `RESULTS.md`, end of file.
>
> **2. The ruler is now genuinely missing, not merely unbuilt.** Both blockers
> left standing — memory formation, and whether step size is the cause — are
> **undecidable without a measurement that can distinguish them**. v2 ordered
> credit work first because it was assumed measurable. It turns out it wasn't
> measured correctly. So v3 starts by making the goal measurable.
>
> ### v3 priority queue
>
> **P0 — Make the goal measurable.** *Do first; everything else is undecidable
> without it.*
> 1. **Distance-curve cell** (Karpathy 2015 §4.2/Fig. 5) in `tools/eval.py`.
>    Already in progress. Must ship with its two confounds stated in the output:
>    same-byte delimiter classes (quote/apostrophe) are satisfiable by parity
>    counting alone, and the two-byte classes have ~0 events in this corpus.
> 2. **State-probe ruler** — the part v2 never had. Freeze a checkpoint, plant a
>    *random* cue byte, let `d` bytes of distractors pass, then decode the cue
>    from the recurrent state with a tiny offline linear probe (chance 1/256).
>    No TMT weight updates, no surface-form shortcut. This is what separates
>    *"no memory"* from *"no surface-form-dependent retrieval"*, which the #23
>    needle probe structurally cannot do.
> 3. **Definition discipline, enforced in `tests/`.** Every gradient-quality
>    number prints its estimator convention and its ruler setting beside it, and
>    every such gate asserts **real corpus bytes**. Random bytes manufacture a
>    confident result — measured: `ρ(1)` flips `+0.534 → −0.171` and the SNR
>    moves 8.6× on one variable. v2 already wrote this warning down for the
>    trace gate and it was re-triggered anyway, so it is now a test, not a note.
>
> **P1 — Step size.** *Cheapest decisive experiment left; never run.*
> Sweep the **global** `--lr`. Every prior LR factorial moved `--lr-decay`, the
> decay role's rate; `--lr` itself was never swept, and the measured AdamW step
> is **0.0217 × lr**, so nominal and effective step are not the same quantity.
> This is the direct test of the over-large-step diagnosis and costs no extra
> updates, which is why it is ranked above accumulation (measured efficiency
> 0.138 at k=64 — accumulation is ~7× weaker than the 1/k framing assumes).
> Judge at **equal wall-clock**, arms paired on the same seed and byte stream.
>
> **P2 — Memory formation.** SSM design iteration (was v2's P1), now *downstream
> of a ruler that can see it*. Unchanged contents: input-dependent gates,
> multi-slot RTU + read routing, log-spaced half-life init, bounded recurrence.
>
> **P3 — Distillation.** **Closed as specified, infrastructure retained.** After
> the off-by-one fix and the added control arm, KD recovers ~half of what its CE
> discount costs and the end state is **neutral** — three independent runs agree,
> two of them bit-identical. Plain CE is the optimum of this family, so the α
> sweep is **not** the next experiment. Do not reopen without new evidence.
>
> **P4 — Optimizer.** **Answered 2026-10-04; folded into P1.** AdamW and
> step-matched SGD are the same algorithm at this operating point
> (`|update|/|m|` spread 1.0020×). The earlier `+0.0185` is noise plus a kernel
> confound. *Further optimizer search is a closed axis.*
>
> **P5 — Throughput.** Mamba parallel scan / ParaRNN. Unchanged, still later.
>
> **P6 — Community.** CLI first (onboarding, issue #21), then AMD/ROCm, then a
> public HF release. Note the release wants the **incumbent** checkpoint, not the
> banked A/B/C distillation arms (CE 1.33–1.37 vs the 1.1741 ruler).
>
> ### Two questions this plan can no longer ask
>
> - *"Is the eligibility trace biased?"* — **settled, no.** Confirmed at
>   production depth today: 16 layers, dim 512, both decay regimes, cosine
>   1.000000, rel err ≤1.05e-06. v2's P0 item 2 stays **HELD**; its conditional
>   ("if the gate shows the bias") did not fire.
> - *"Does the variance hinge suppress memory?"* — **the premise was false.**
>   `max(0, 1-√var)` penalises variance **below** a floor and pushes it up.
>   It is anti-collapse. Ask instead whether it is *active at all* at the learned
>   residual-stream scale, and whether its decay-coordinate gradient aligns with
>   or opposes CE's.

> **CORRECTION 2026-10-04 (P0's premise, measured).** P0 action 1 was built and run:
> `tests/check_trace_bptt.py`. **The eligibility trace is NOT a biased estimator.**
> Against exact truncated BPTT, built from the engine's own forward functions, at 4
> layers and in both decay regimes (σ=0.544, 1.14 B half-life; σ=0.953, 14.27 B):
> **cosine 1.000000, equal-footing relative error 3.7e-07**. What #2 measured as bias
> (0.36 / 0.934) is a `1/B` normalization asymmetry on the decay rows — see below.
> Consequences, all applied in place in this document: §1's evidence chain loses its
> first bullet, §4 action 2's conditional did not fire, and §6(a) was unsatisfiable
> as written and is rewritten. See `../RESULTS.md`, "P0 item 1 — the trace-quality
> gate". **P0 item 2 (the hybrid window + trace trainer) is not justified by current
> evidence** and must not be built on it.
**Sources pooled:** YouTube feedback (2026-09-16…), the open GitHub issues
(#2/#3/#7/#8/#9/#10/#13/#14/#15/#23/#24/#25/#28/#30), Karpathy et al.
*Visualizing and Understanding Recurrent Networks* (arXiv:1506.02078v2, 2015),
the Mamba (S6) design, and the 2026-10-03 follow-up video decisions.

This plan exists to answer one question the video left open: **in what order**
do we attack the two standing blockers —

1. gradient **SNR ≈ 0.05** (both shared and decay groups, whole 890k run), and
2. **memory formation is real but small** (mean half-life ~1.14 B; only 7 of
   8192 dims with half-life > 8 B at 890k — the "~24 dims" figure sometimes
   quoted is not supported by `RESULTS.md`, which reports 7), with an
   independent needle probe showing **no measurable retrieval past ~64 bytes**
   (#23; literal-cue induction only — obfuscated and parallel-tracking
   retrieval were not tested by that probe).

## 0. Situation review (why these are the blockers)

| evidence | value | source |
|---|---|---|
| held-out CE (clean baseline, t=64,747) | 1.1741 ± 0.0415 | `RESULTS.md` 2026-09-23 |
| gradient SNR, entire 889,897-step run | ~0.05 (0.10 @ 2k, never recovered) | `RESULTS.md` long run |
| mean half-life / long dims (890k) | 1.14 B / 7 dims >8 B | `RESULTS.md` long run |
| needle probe: retrieval benefit | only at d=64 ±2σ; nil ≥512 B (literal-cue induction; obfuscated/parallel-tracking untested) | issue #23 |
| teacher nats/byte vs TMT | 0.6853 vs 1.1741–1.1902 | `PLAN-2026-10-03-DISTILL.md` gate 0 / teacher eval |
| 2015 LSTM ceiling: quote cell | ~230 chars tracked, BPTT-100 | Karpathy §4.2 |

The 2015 number is the single most important external anchor: a 1M-parameter
LSTM trained with truncated BPTT-100 + RMSProp **does** track 230-char quotes
and boosts closing braces to ~60 chars. So long memory is achievable by this
architecture class with simple gradients. TMT's measured <<64-byte retrieval is
therefore *not* an inherent limit of the RNN+SSM direction — it is a training
(credit-assignment) deficit. That is what makes the order below load-bearing.
(One honesty guard on that reading: the only clean estimator-only A/B in the
archive is #3's K=8/K=32 variance result; #23's own end-to-end A/B found the
hybrid traces buy nothing over a long stream at matched wall-clock — the
caveat is stated in §1 and wired into the P0 gate.)

## 1. The core reframe

**Working hypothesis — the gradient estimator is a leading suspect, not a
settled root cause.** The evidence justifies making credit quality the *first*
variable to control (P0), but two results in our own archive stop us from
asserting a single root: #23's end-to-end null and reyvvon's estimator-free
fix (both below).

**SUPERSEDED IN PART, 2026-10-04.** The gate that was supposed to settle this
(P0 action 1) came back negative: the trace is exact, so gradient *exactness* is
not the deficit. Two results now carry the whole hypothesis — #3's collapse and
reyvvon's estimator-free fix — and both point at **init + schedule** rather than
at the estimator. #3's own A/B remains the only controlled estimator-only
comparison in the archive, but its 176x variance reduction was measured with
exact gradients *at fixed init*, and the gate says the two were confounded:
exact gradients cannot be what fixed the collapse, because the trace was already
computing them.

Chain of evidence (for):
- ~~#2 (mar-i0): the eligibility trace is exact for the last layer only. At 4
  layers: `decay[0]` relative error 0.36, cosine 0.934; at 16 layers most
  layers are in the biased regime. `check_gradients.py` only checks against
  **finite differences on a frozen model** — this bias is currently invisible
  to the test suite.~~
  **RETRACTED 2026-10-04 by measurement.** `tests/check_trace_bptt.py` finds
  cosine 1.000000 and equal-footing relative error 3.7e-07 at 4 layers, at
  σ=0.544 **and** σ=0.953. The 0.36/0.934 figures are reproduced exactly by the
  engine's `/B` asymmetry on the decay rows (ratio 8.00x / 16.00x / 32.00x at
  B=8/16/32), which is a scale, not a direction. The concern in #2 was correct —
  the suite really did not check this — but the answer is that there is no bias to
  find. Note what this means for #3, below.
- #3 (mar-i0): with traces, the only regime that trains stably is the
  one-byte-memory regime, so decays collapse into it **deterministically**
  (0.889 ± 0.009 half-life across seeds). Swapping the estimator to exact
  truncated BPTT at *fixed* depth and init removes the collapse: K=8 → 118×,
  K=32 → 176× variance reduction; long timescales survive. **This is the only
  controlled estimator-only A/B we have, and it is the reason P0 is first.**
  **RE-READ IN LIGHT OF THE 2026-10-04 GATE:** that A/B cannot be explained by
  gradient exactness — the trace already computes the exact decay gradient — so
  what it actually varied was something else in the fork. #3's variance result
  stays on the books as an unexplained effect, not as evidence about the
  estimator.

Evidence that must be stated alongside the above:
- **#23 end-to-end null:** the same fork's exact-trace/hybrid trainer bought
  **nothing over a long stream at equal wall-clock** (0.5 % worse than plain
  truncated BPTT; only +7.3 % when the state resets every 128 bytes). #23 also
  measured that **gradient cosines do not predict the outcome** ("a cosine of
  0.97 against full BPTT is apparently not the quantity that decides the
  outcome") — so the P0 gate must not rest on cosine alone.
- **reyvvon closed #3 without changing the estimator:** 1–32 log-spaced decay
  init + an LR cooldown fixed the delimiter probe on the *original* trace
  code (12 experiments). The collapse is therefore also a function of
  init + schedule, not only of gradient exactness.
- **#23 needle probe (scoped):** retrieval is absent past ~64 bytes for
  *literal-cue induction*; the surface-form-independent retrieval that SSMs
  claim was never tested in its own right by that probe, so it is
  "not measured" rather than "refuted".
- Karpathy 2015 (ceiling): the class can do it — which shows TMT's RNN+SSM
  direction is not the limit, without isolating the estimator as *the* cause.

**Consequence:** the video's cornerstone (iterate the SSM/RTU design) is right,
but the *first* variable to nail down is credit quality: gates/slots/init are
cheap to mis-tune on top of biased gradients (the exact trap #3 documents),
and the estimator A/B is cheaper than the SSM work. **Pre-committed branch:**
if the P0 gate moves SNR but not ruler CE / needle (the #23-observed outcome
shape), stop estimator archaeology, keep the P0 hybrid only if it is neutral
relative to the trace trainer, and proceed to P1 — recording the falsification
in `RESULTS.md` as a first-class result.

## 2. Priority queue (ranked; # of items = order of attack)

### P0 — Credit quality (do first; cheapest decisive experiment)
1. **Trace-quality gate**: port mar-i0's `verify_traces.py` cosine-vs-exact-TBPTT
   check into a new `tests/check_trace_bptt.py` (`--vs-bptt`; the name §4
   action 1 already uses). Expected: per-layer relative error grows toward
   layer 0;
   gives us a ruler number for the trace bias. **Diagnostic only** — see the
   #23 caveat in §1: cosine vs BPTT does not predict training outcome.
   **DONE 2026-10-04** — `tests/check_trace_bptt.py`, free/CPU/seconds. Verdict:
   **no bias** (cosine 1.000000, rel 3.7e-07 equal-footing, both decay regimes).
   Instead it found a real defect: the loss is `/B` so every gradient carries
   `1/B`, but the decay gradient `(dlds * decaytrace).sum(0)` does not, so the
   8,192 decay coordinates receive a gradient **exactly B× larger** than every
   other parameter group. Harmless under AdamW (per-parameter, scale-invariant) —
   which is why 889,897 steps ran clean and nobody noticed. **Not** harmless under
   SGD; see P4.
2. **Hybrid window + trace trainer** (from SimonVitzethum's design / #17):
   **HELD — do not build on current evidence.** §4 action 2 made this conditional
   on "if the gate shows the bias"; it does not. Kept here because the design is
   sound and the hypothesis may be revived by a *different* measurement (e.g. an
   SNR readout on a trained checkpoint, where σ is heterogeneous per coordinate
   rather than uniform at init). Any revival must state what new evidence
   motivates it. The design, for the record:
   - forward: batched windows B×T (e.g. 16×128) — GPU-efficient, reuse of the
     existing compiled ring credit;
   - gradients: exact TBPTT inside the window **for all weights**;
   - across boundaries: exact traces for decay/embed/gate only (cheap because
     the recurrence is diagonal); `seqlen=1` reduces exactly to the current
     trace rule (a knob, not a fork).
   - Independence argument: keeps the no-max-window philosophy for per-channel
     params while the rest trains exactly; hybrid cosine = 0.97–0.99 vs BPTT
     (measured in #17 on a fork). Caveat: those are the gate/decay rows only
     (0.86–0.96 elsewhere, per #23's independent check), and #23 warns the
     cosine does not predict outcome — treat as a tracer, not a bar.
3. **Gate**: matched from-scratch arms (seed 0, byte-matched; judge at **equal
   wall-clock**, not just equal steps — the #23 A/B style) with the ruler;
   success = SNR measurable improvement **and** a decay histogram with a real
   long tail **and** no CE/needle regression vs the trace trainer. Cosine
   alone is not a pass (§1). If SNR moves but CE/needle do not, fire the
   pre-committed branch in §1 instead of adding more credit work.

### P1 — SSM design iteration (T-RTU; the video's cornerstone 1, now unblocked)
Run these only after P0 shows gradients can support long memory:
1. **Input-dependent gates** (write/erase/read) — the "Mamba selectivity"
   feature to steal: `A` depends on the current input, so the cell decides what
   to write/keep/forget. Sources: #25 (7 params/dim; the issue's "0.07% of
   4.5M" is for a *single* 512-dim layer — at TMT's 16×512 that is 57,344
   params ≈ 1.28 % of 4,481,793, still cheap but not free),
   `ARCHITECTURE_IDEAS2.md` §3.2/§11–14, georgygarbaly "not content-
   dependent" critique, Mamba S6.
2. **Multi-slot RTU + read routing** — content tags so a slot can address a
   *kind* of memory; the associative-memory component from #7 (delta-rule,
   fixed-size 128×128) as a *component*, not an RTU replacement.
3. **Log-spaced half-life init** (from #3's fix, retracted as a standalone):
   `1–500` byte half-lives across dims. Two corrections from the #3 thread:
   mar-i0's one-epoch TinyStories check showed exact gradients **undo** a
   1–500 spread on real text (both inits converge to the same model), so the
   "valid paired with exact gradients" claim holds on the synthetic probe
   only; and reyvvon closed the issue with a 1–32 spread + LR cooldown on the
   *original* trace code — that cheaper combination is the first test here.
4. **Bounded/gated recurrence** (`s = a·s + (1−a)·x` with a = sigmoid(decay +
   gate·x)) — targets the unbounded state growth (state RMS 0.40→0.46 while
   memory saturates, `RESULTS.md`). Prior art to cite: the identical
   scale-preserving recurrence was already tested in #3 and measured a null
   for CE/collapse (1.995 vs 1.978 bits/byte) — this item is about *state
   RMS containment*, so include a matched CE control to avoid relitigating it.

### P2 — Memory ruler cells (make the goal measurable)
1. **Distance curve** (Karpathy Fig. 5): probability of closing byte as a
   function of distance to the matching open byte (quotes/brackets/parens) —
   goes from binary pass/fail to a curve with a 2015 external ceiling to beat.
2. **Needle probe** as a permanent eval cell (already designed in #23; make it
   reproducible with fixed seeds). Record its scope honestly: detection needs
   a long literal cue (a 6-byte cue → nil even at d=64), and obfuscated /
   parallel-tracking retrieval is untested. Add the **k-needle variant**: #23
   argues the single recurrent state has no mechanism to keep k items
   separately addressable — that is the direct instrument for P1.2 (multi-slot
   + associative #7) and the sharpest falsifiable test the roadmap has.
3. **Error-attribution oracles** (Karpathy §4.4): n-gram → dynamic-memory →
   rare-word → after-space/quote/newline → punctuation → boost. This tells us
   *which* future lever matters (capacity vs external memory vs byte-BPE vs
   word-boundary handling).
4. **Gate-saturation statistics** (Karpathy §4.2 analog): forget-gate satura-
   tion fractions over held-out text; the diagnostic that would have caught #3.
5. Existing: half-life histogram (eval.py), hot-dim count, state-RMS trend
   (telemetry).
6. **Build**: word-length drift — mean bytes/word and premature-line-end rate
   vs a CE control; the direct signature of the MI bias. Designed in
   `RESULTS.md` (2026-10-03) but not implemented yet; cheap, CPU.

### P3 — Distillation steps 1–4 (parallel track, unblocked by P0)

**REVISED 2026-10-04 — this section was stale and said "continue unchanged" for a
plan whose step 3 had already been closed and then found to be closed on a
defective target.** State as of 2026-10-04:

- Steps 1–2: **withdrawn** (2026-10-03). Step 3's KD arm: **closed 2026-10-03,
  closure reopened 2026-10-04**. Step 4: not started.
- Why reopened: the Marginalize-It target was built with an off-by-one (it
  conditioned on the byte it was predicting), then "fixed" by deleting all depths
  >= 1 — 23.1% coverage instead of the paper's 89%. Corrected target, corrected
  gradient, and a half-CE control arm added. See `RESULTS.md`, "The MI target was
  wrong at every depth >= 1".
- Current measured state: coverage 64.4%, target entropy 0.54 nats, mean depth
  1.26, top-1 == gold 71.1%. Re-run A/B (2,000 steps, B=64, seed 0) shows **no
  divergence** and KD **neutral** vs its control (−0.035 nats). The old closure is
  not reproduced; the experiment is not yet a science arm.
- Still true from the original text: teacher quality measured (0.49 nats/byte
  better), gate 0 cleared, MI at k=32, α=0.5, non-CE loss terms frozen so this
  does not conflict with P0–P2.

Next, in order: (1) the word-length-drift ruler on generated text, which is the
only instrument that can see the depth-1 not-ending bias (top-1 == gold at just
44.9% of depth-1 positions); (2) a gradient-SNR readout for the KD arm, since
§0's premise is SNR and neither old A/B measured it; (3) only then a longer arm
with an α sweep. Do **not** re-run the 2,000-step CE/KD pair without arm C.

**ADDED 2026-10-04 — predict the α sweep's outcome before running it.** The
objective is `(1-α)·CE + α·KL`, so under AdamW only the mixture ratio `α/(1-α)`
matters, and coverage/fidelity do not depend on α at all. Both the CE discount and
the KD recovery are therefore near-linear in that ratio. Measured at ratio 1:1:
the discount costs **+0.070 nats** (arm C − A) and KD recovers **0.035** of it
(arm C − B) — **half**. So net(α) is negative at every α > 0 and **α = 0 is the
CE-optimal setting; no α wins.** A sweep run expecting a winner is three runs
confirming a negative. The one thing that could overturn this is that the result
is CE-only: MI's not-ending bias predicts *longer* generated words, a signature
CE cannot see, so **(1) must come first** — if word-length drift shows a real KD
benefit, the sweep is justified; if it shows none, close the track on fidelity
grounds (larger k, or EoT's `<eot>` channel) rather than on α. This is a
prediction, not a result: one seed, one α, B=64 against a frozen B=1024.

### P4 — Optimizer as a factorial row, not a decision
SGD / new-muon / SOAP: add SGD+momentum+WD as one arm in the role-specific-LR
factorial. Notes: Karpathy used **RMSProp** (batch 100, unroll 100, lr 2e-3,
decay 0.95); the "SGD > Adam" folklore is contested; muon/SOAP are built for
much larger runs and add machinery that is mostly noise at 4.5M. **Do not
block P0 on this.** If SNR doesn't move with the estimator, revisit.

**PRECONDITION on any SGD arm (added 2026-10-04).** The P0 action-1 gate found
that the decay rows' gradient is **exactly B× larger** than every other parameter
group (the loss is `/B`; `(dlds * decaytrace).sum(0)` is not). AdamW normalises
this away per parameter. SGD does not: under SGD the decay rows would train at an
effective learning rate **B× larger** than the rest — 1,024x at the frozen B. Any
SGD arm must first divide the decay gradient by B, or it measures a learning-rate
blowup on 8,192 coordinates rather than an optimizer effect.

**UPDATE 2026-10-04 — this axis is now ANSWERED; do not spend another arm on it.**
Run in the isolated study `../scratch-optimizer-study/` (repo untouched; the SGD
optimizer is derived from `gpu_port.py` and gated **bit-identical in `adamw`
mode**). A step-matched arm at lr=5e-4 — 1.00x AdamW's median step, decay
gradient corrected by 1/B, 30,000 steps from the same t=889,897 as the reused
AdamW control — gave held-out CE **1.2279 vs 1.2094 (+0.0185)** and cross-domain
**2.4745 vs 2.4176 (+0.0569)**. **SNR did not move** (0.065 trunk, 0.065 decay).

Two facts from it are worth more than the verdict:

1. **The precondition is satisfiable, and it is what makes the arm fair** — the
   same 1/B fix is bit-exact under AdamW, so the AdamW control needs no re-run.
2. **The decay role is optimizer-dependent.** AdamW's scale-invariance has been
   training the 8,192 decay coordinates at full `lr` for 889,897 steps regardless
   of their true gradient magnitude; a magnitude-proportional optimizer gives
   them ~1024x less and **freezes them outright** (hot dims 104->172 under AdamW,
   104->104 under corrected SGD; sigma moves 1e-5 over 30k steps). The memory
   timescale's learned history is a property of AdamW's normalization, not of the
   gradient's magnitude. That is an **architecture question** and it supersedes
   this row as the thing worth thinking about.

Independent convergence worth noting: swapping the optimizer and auditing the
gradient estimator both leave SNR ~ 0.05 exactly where it was. See `RESULTS.md`,
"the optimizer axis is answered".

### P5 — Throughput: Mamba parallel scan / ParaRNN (later)
After P0–P1 prove the memory direction works. Parallel-over-time training is
the only axis your optimization history has never touched (all current work is
per-step). ParaRNN (Taylor-truncated nonlinear RNNs) is the exotic end; a
Mamba-style chunked associative scan over TMT's linear-in-state recurrence is
the natural middle. Both preserve streaming forward pass.

### P6 — Community (the video's asks, re-ranked)
1. **CLI** for train/chat (fixes #21-style onboarding friction — reproducing
   the data and running the trainer; #9 is a feature, not an onboarding gap.
   Cheap, high value).
2. **AMD port** (ROCm) — nice-to-have; any contributor willing may own it.
3. **Public HF release** of a banked checkpoint + demo (comments asked
   repeatedly; the hub is currently private). Best community ROI.
4. Grants / Modal free credits (compute accretion).
5. **P2P training (#8): parked, not promised** — conflicts with seed-0
   byte-matched arm discipline and sha256 checkpoint integrity; parameter
   merging across divergent streams is unattributable and a security surface.
   Revisit only as an opt-in toy with "no science claims".

## 3. Explicitly rejected (with reasons)

| idea | why |
|---|---|
| Switch to BPE now | contradicts byte thesis; distill's Marginalize-It already pulls token-level semantics in without any tokenizer in TMT; keep byte-BPE (#28) as a *later* test if word-boundary oracles demand it |
| Drop the MSE/embed loss | deliberately kept: variance hinge + embed-MSE are the JEPA/memory insurance (distill plan §3 "unchanged") |
| Holographic/phazor byte encoding | too exotic now; revisit only if P1 gates plateau |
| Full transformer-style attention | memory is bounded by design; use the fixed-size associative component (#7) instead |
| MCTS/test-time compute | SNR is the blocker, not expressivity — premature at 4.5M |
| P2P training | see P6.5 |
| "2000 epochs" framing of Karpathy | it's 50 epochs in the paper; don't cite the blog number as the paper's |

## 4. First actions (this week, in order)

1. ~~`tests/check_trace_bptt.py` — cosine-vs-exact-TBPTT gate with `--layers 4`
   (reproduce #2's 0.36 / 0.934 numbers on the numpy port).~~
   **DONE 2026-10-04.** Built; run on the **torch** port, not numpy: numpy has no
   autograd, so an exact-BPTT reference there would mean hand-writing a second
   forward, which is the bug factory. Verdict: **the reproduction target should
   fail** — cosine 1.000000, not 0.934, and the 0.36 relative error is the `/B`
   asymmetry. Recording the failed reproduction deliberately: it is the evidence
   that #2's bias claim is retracted rather than merely unexamined. The gate runs
   on real line-bucketed corpus bytes — with random bytes the states are iid and
   every lag weighting collapses to the same direction, giving a confident
   cosine 1.0000 for free.
2. ~~If the gate shows the bias:~~ prototype hybrid window+trace trainer on CPU at
   small scale (dim 64, L4), gate it with `--vs-bptt`, measure SNR at matched
   steps **and matched wall-clock** against the trace trainer (the #23 A/B
   style: equal time is the honest comparison).
3. Add the distance-curve cell to `tools/eval.py`.
4. Start distill steps 1–4 (T4) in parallel — unchanged plan.
5. CLI + HF release drafted whenever compute legs are idle.

## 5. Sources map

- **YouTube comments** → gates (LSDev_1, georgygarbaly), eval baselines
  (Markov order-2), HF publishing (multiple), drop the MSE (data-m6l6f,
  upthera92), SGD row (upthera92), "tested against forgetting" (TeddyM,
  4cryingoutloud), ParaRNN (universeum8434), content-dependent memory
  (georgygarbaly), grants/Modal (comments), overclaiming critique (many).
- **Issues** → trace bias (#2), decay collapse (#3), associative memory (#7),
  multi-timescale+complex recurrence (#10), data-limited (#23), needle probe
  (#23), gates 7-param (#25), yardsticks/measurement discipline (#24), KG
  memory (#15), byte-BPE test (#28), word-structure (#30).
- **Paper 1506.02078v2** → 230-char memory with BPTT-100 (achievable ceiling),
  distance curve (Fig. 5), error oracles (§4.4), gate-saturation statistics
  (§4.2; Fig. 3's left/right saturation plot — the "two-register eval" that
  earlier drafts attributed to the paper does not appear in it), RMSProp,
  "scaling buys n-gram errors" (architecture-not-scale).
- **Video decisions (2026-10-03)** → SSM-first cornerstone, no-BPE (keep
  control), optimizer investigation, CLI, AMD, humility on claims.

## 6. What success looks like

A checkpoint where (a) gradient **SNR** improves over the trace trainer at
matched wall-clock **without a CE/needle regression**, (b) the distance curve
beats the 2015 LSTM baseline beyond ~60 bytes, (c) the decay histogram has a
real tail (>8 B dims: meaningfully more than 7), and (d) a byte-matched KD arm
that **beats its own half-CE control** (`--distill-null-target`) on the two-ruler
gate.

Note on (a): it previously read "the `--vs-bptt` gate shows the trace bias mapped
to a measurable SNR gain after the hybrid trainer," which was **unsatisfiable** —
the 2026-10-04 gate measured no trace bias to map (cosine 1.000000, rel 3.7e-07).
It is restated above as an SNR target with the mechanism left open, and the hybrid
trainer of P0 item 2 is demoted from the means to one candidate means. The gate
name has no `--vs-bptt` flag; it is the gate's only mode, and the flag name is
kept here to avoid churning references.

Note on (d): it previously read "distill steps 1–4 produced a byte-matched KD arm
that does not regress on the two-ruler gate", which was written as a success
criterion for an arm `RESULTS.md` had already falsified. Comparing a KD arm to a
plain CE baseline does not isolate distillation, because `α·KL + (1−α)·CE`
changes the CE weight too; the control arm is the comparison that means something.

**Also a defined non-success:** if (a)'s SNR moves but the rulers do not, P0
is falsified (the shape #23 observed), the branch in §1 fires, and the roadmap
proceeds to P1 without further credit-assignment work.