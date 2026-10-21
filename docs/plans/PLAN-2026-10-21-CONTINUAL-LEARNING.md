# PLAN — continual learning: the one plan (2026-10-21)

**Status:** the single active plan. It supersedes the *ordering* of
`PLAN-2026-10-04-ROADMAP.md` v3, `PLAN-2026-10-06-MEMORY-BENCHMARKS.md` and
`PLAN-2026-10-11-T-AXIS.md`; it retains their evidence and their closed axes.
`PLAN-2026-10-13-COLAB-HF-RUNS.md` survives as the *run protocol* (how a leg is
executed), not as a plan. `SPEED-INVESTIGATION.md` is closed.

**One sentence:** build the ruler that measures continual learning, then make
the model pass it — learn a new fact from a stream, recall it, keep the old
ones, and not lose the general language ability.

**The goal, stated once:** a model that *keeps learning*. The README makes two
claims — long-range memory and continual/test-time learning — and the second
one has never been measured in any generation of the code.

**Why this plan exists:** 14 plans have been written. 13 produced measurable
verdicts. **0 produced an end-to-end win.** The reason is structural, and it
is now understood (§7): every plan substituted a proxy for the goal, because
the goal was never a criterion and was not even expressible in the engine.

---

## 0. What is already true (evidence, not plans)

All numbers are in `RESULTS.md` (append-only) with their ruler settings. This
section exists so a new reader does not re-derive them.

**Assets that exist and work.**
- A bit-identical CPU/CUDA engine with a parity gate (`numpy_port.py` ≡
  `gpu_port.py`, `tests/check_parity.py`).
- 25 executable gates in `tests/`.
- A training-time credit apparatus (ring / tail / trace) with **zero inference
  parameters** — it is plumbing, not weights.
- Colab + private-HF durability that survives runtime recycling, with
  content-verified transport (`checkpoint_hub.py`).
- A ruler (`tools/eval.py`, `tools/ruler.py`) and a synthetic memory-benchmark
  suite (`tools/membench/`).
- A speed result: the fused `nvrtc` ring credit is **1.26× end to end, bit
  identical** (loss `7.58001`, `max|Δ| = 0`; 26.72 vs 33.68 ms/step).
- A capacity result: **P4 + P6 give 15× the incumbent's retention** — 37.5 /
  17.1 / 5.3 / 0.78 % of the control floor at 8 / 32 / 128 / 512 B against
  the incumbent's 4.8 / 1.7 / 0.36 / 0.0007 %, with a half-life of 95.4 B
  instead of 1.05 B. The eroder is weight decay on the decay logits
  (92.30 → 39.07 B without P6), not training.

**Axes closed, with their causes. Do not reopen them.**
| Axis | Verdict | Cause |
|---|---|---|
| gradient SNR / trace quality | **closed** | the gradient was already exact for every parameter with a cross-time path; the programme was solving a problem this architecture does not have |
| per-role learning rate | **closed** | moves CE, not the memory; "+29 % hot tail for zero gain" |
| distillation (Marginalize-It) | **closed as specified** | KD measured CE-neutral (B−C = −0.035) — a confound, not a science arm; "plain CE is the optimum of this family" |
| global `--lr` / optimizer | **closed** | six sweeps; it moves CE, not σ |
| FSP (FlashLM) auxiliary | **null, with a measured cause** | the head scored F1 0.685 (0.792 at top-22) against a **global byte prior of 0.817** — the objective was satisfied without the state; and τ=64 is beyond the state's 8–32 B usable horizon |
| T axis at this scale | **T=1 wins, does not rule out T** | T=1 1.1272 ± 0.0225 vs T=4 1.2210, T=16 1.1867, incumbent 1.2094 ± 0.0143 at matched bytes; matched updates flips it (1.1377 / 1.1350 / 1.1867). The cause is **update count**, not credit, and the model is underfitting — so this *locates the regime*, it does not settle the axis |
| the speed hypothesis | **refuted, then replaced** | "the wall is the bytes, not the atomics"; the real prizes are `masked_fill` (5.1 ms/step) and 1,072 kernel launches (10.1 ms CPU) |
| the B-reduction bet | **refuted** | B=64 is 3.7× worse; take T at B=1024 and keep the ring |

**What is NOT established.**
- That the state can be read out. The claim *"the readout never beats its own
  control"* — which the previous plan pivoted on — rests on a **broken
  instrument** (§6, defect D1).
- The n-gram deficit. `0.81 nats/byte` (order-8 byte n-gram) vs `1.2094`
  (TMT) is quoted in three documents, but the script that produced it
  (`lag5.py`) was never committed and is gone (§6, defect D2). It is also not
  byte-matched to the ruler's held-out set.
- Anything about continual learning. `grep -c forgetting RESULTS.md` → **0**.

---

## 1. The criterion

The goal becomes three numbers, measured on one ruler, with a control.

> **CL-RULER.** For a model M and a stream of new facts F:
> 1. **ACQUIRE** — after the stream, M recalls the new facts
>    (exact-match accuracy on a held-out set of the same shape).
> 2. **RETAIN** — M still recalls the facts it knew before the stream
>    (the same metric on the pre-stream set, and the pre/post gap).
> 3. **NO REGRESSION** — M's held-out CE on the general corpus has not
>    degraded (ΔCE ≤ 0.02 nats/B).

**Why these three and not one.** They are the minimum that makes "continual
learning" a claim rather than a hope, and they map exactly onto the three
halves of the design principle already written down in the project's own
notes: *learning* (ACQUIRE), *retention* (RETAIN), *validation* (NO
REGRESSION). A model that acquires and forgets is not learning; a model that
acquires and regresses is trading away what it had.

**The old success criteria are spent and are replaced by CL-RULER.** All four
of `PLAN-2026-10-04-ROADMAP.md §6` are dead or saturated: (a) SNR — axis
closed; (b) "beats the 2015 LSTM baseline beyond ~60 B" — that anchor is
Karpathy/Johnson/Fei-Fei on *War and Peace + Linux kernel source*, a
**depth counter**, not associative retrieval, on a different corpus (this
project's own `STRATEGIC-REVIEW §0b`); (c) "decay histogram has a real tail"
— already satisfied by the init, so it measures capacity, not use; (d) KD —
closed.

**Pre-registered thresholds.** Chosen before any run, and fixed:
- **ACQUIRE** passes at ≥ 0.50 exact recall of a 4-byte payload at d=32
  (chance 2⁻³²) on a *trained* model; a probe on a frozen checkpoint is
  diagnostic, not a pass.
- **RETAIN** passes at ≥ 0.90 of the pre-stream recall, measured on a
  **replayed** set of pre-stream facts that were *not* in the stream.
- **NO REGRESSION** passes at ΔCE ≤ 0.02 nats/B on the general ruler.
- A **veto control** accompanies every ACQUIRE cell: the same readout on
  *permuted* labels. If the readout cannot beat its own veto, the cell is
  UNMEASURED, never PASS. (The repo already has this discipline in
  `tools/p0_verdict.py`: "UNMEASURED is never PASS".)

---

## 2. The architecture, as it actually is

This section replaces the intuition in everyone's head with the code.

**The recurrent state is a per-coordinate leaky integrator of the byte
embedding.** `gpu_port.py:247`, `numpy_port.py:142`:

```python
state = states * a + enc          # a = sigmoid(decay)   (or sigmoid(decay + gate*x_in) in the P3 arm)
h, cache = _layer_norm(state, self.g, self.b)
z = h @ self.W.t()
return x_in + y, state, a, ...
```

and the caller `gpu_port.py:1701-1707`:

```python
enc = self.embed[c_arr]           # ONE embedding
x = enc
for l in self.layers: x, state, decay, cache = l.forward(enc, x, l.states)
```

Consequences, all verified:

1. **The residual stream never enters the state.** The only inputs to the
   memory are the byte embedding and the decay.
2. **There is no cross-layer state flow.** Layer L's memory is not a function
   of layer L−1's memory.
3. **The 16 layers are 16 nonlinear readouts of ONE 512-dimensional
   multi-scale EMA bank**, accumulated into a residual stream. They are not a
   feature hierarchy.
4. **The 8,192 "memory channels" are 16 permutations of the same 512
   time-scales.** The P4 init (`gpu_port.py:1017-1021`) builds
   `np.geomspace(hmin, hmax, 512)` *inside* the layer loop and reshuffles it
   per layer. So the model's distinct memory time-scales number **512, not
   8,192** — and under the *default* init every half-life is ≈1 byte, i.e.
   there is no multi-scale at all (which is why the default model retains
   4.8 % and P4 retains 37.5 %).
5. **The whole multi-timescale memory of this model lives in 8,192 logits =
   0.18 % of the weights.**

**Parameter budget** (`numpy_port.Model(dim=512, layers=16)` → `theta.size =
4,481,793`, verified live):

| group | params | % |
|---|---|---|
| per-layer `W` (512×512) × 16 | 4,194,304 | **93.585** |
| per-layer `g`+`b` (1024) × 16 | 16,384 | 0.366 |
| per-layer **`decay` (512) × 16** | **8,192** | **0.183** |
| `embed` (256×512) | 131,072 | 2.925 |
| decoder `Wd`+`bd` | 131,328 | 2.931 |
| `Ws`+`bs` | 513 | 0.011 |
| **total** | **4,481,793** | 100 |

**The structural limit.** `state_j(t) = Σ_k a_j^k · emb(byte_{t−k})_j`: a
*weighted average* of one embedding coordinate over time. It can represent
"there has been a lot of `e` recently", not "the value that followed this
key". A linear readout over a sum of EMAs cannot isolate one term without a
**key-conditioned read**. This is the plan's own §7.3 item 8 and
`STRATEGIC-REVIEW §3(iii)`, and it is the reason the read path fails —
*not* a shortage of capacity, and not a shortage of layers.

---

## 3. The size decision: 16 layers and 512 dim

**Do we need them? No — and for continual learning they are mostly cost.**

- **Capacity is not the binding constraint.** Data-limited, not
  capacity-limited (14 epochs beats 24: 1.63 vs 1.82 BPB). The CUDA fork's
  **34M-parameter** run still shows *no retrieval past 64 B*, and its traces
  are 0.5 % *worse* than plain TBPTT. Bigger did not help there either.
- **94.1 % of the parameters sit in the readout stack** — precisely the
  surface a test-time update overwrites. That is the catastrophic-forgetting
  surface, and it is 4.19 M parameters wide.
- **The memory's entire control surface is 0.18 %** (the decay logits).

**So the decision is not "16 vs 12 vs 8 layers". It is: keep the core small
and stable, and put new capacity where new knowledge lives.**

| change | params | note |
|---|---|---|
| 16 → 8 layers | 2,372,353 (−47 %) | for a cheaper core |
| dim 512 → 256 | 1,192,449 (−73 %) | `W²` dominates |
| **decay-only expert** | **8,192 (0.18 %)** | a distinct lifetime profile — the natural expert here |
| **LoRA r=4 on all 16 layers' `W`** | **65,536 (1.46 %)** | `1024r` per layer |
| a full depth copy per expert | 4,218,880 | **never** — this is why the T-RTU idea was deferred ("at K=16 the write projection alone is 4.2 M params — nearly doubling the model") |

**The rule this plan adopts:** *the core is trained once and then frozen; new
knowledge lives in experts, and an expert never copies the depth.* A frozen
core is what makes the core's knowledge survive a stream by construction, and
it is what makes the "promotion" step in §1 meaningful (promotion promotes an
*expert*, never the core).

---

## 4. Two candidate mechanisms for "learn a fact, keep it, recall it"

CL-2 and CL-3 exist to choose between these. They are not interchangeable
and the plan does not pretend otherwise.

**M1 — make the in-state write addressable.** The write becomes a function of
context (the residual stream) and/or of a key, and the read is conditioned on
a key. This is the "hierarchical feature-state wiring" the previous plan
imported from the CUDA fork. It keeps everything inside the weights, which is
the philosophically right answer for a recurrent model. It is also the
expensive one: it is a change to how the state is written and read, it is the
one thing the CUDA fork *also* fails to demonstrate (no retrieval past 64 B in
its own runs), and it must be re-derived, not copied.

**M2 — an explicit keyed store alongside the state.** A small associative
memory (key → value) that the model learns to write to and read from, with the
recurrent state unchanged. Far cheaper, far more likely to work, and it is
where "short-term vs long-term memory" becomes trivially expressible (entry
TTL, promotion, quarantine). Its cost is honest: the knowledge is not in the
weights.

**The plan tests M2 first and keeps M1 as the follow-up**, because M2 is
cheap and decisive about the *question* (can this model learn and keep a fact
at all?), and because the M2 result is exactly what an M1 design would need
to justify itself.

**The four project ideas map onto these mechanisms:**

| idea | where it lives | first test |
|---|---|---|
| **1. Experts for new domains** | a decay set (8,192) + optional rank-r readout LoRA (1,024r/layer) per expert, selected by a learned router. No router exists in the code today (`grep expert\|router\|topk` → 0 hits) | CL-3 arm (c) |
| **2. Short-term / long-term memory** | **already ~80 % built**: the per-channel decay logit *is* a lifetime; P4's log-spaced init creates the spectrum; P6 stops training eroding it; the P3 gate makes retention input-dependent. Missing: a **promotion/demotion policy** and a ruler that measures lifetime-vs-usefulness | CL-3 arm (b) |
| **3. Prevent harmful self-learning** | the risk is currently **dormant**: `numpy_port.Runtime.ingest` (866-912) does update weights on user bytes, but generated bytes are **never re-ingested** (only upstream's `legacy/main.py:415` trains on its own output). The three missing safeguards are all missing: **no replay buffer or knowledge store exists**; `grep -c forgetting RESULTS.md` → **0**; and `verify_saved` (`train_gpu.py:517`) checks *transport* identity, never quality — **no checkpoint is ever accepted or rejected on quality** anywhere. The 25 gates validate *code*, not *capabilities* | CL-3 arm (d) + the CL-RULER |
| **4. Cross-depth self-distillation** | architecturally available today (depth = 16 stacked nonlinear readouts of one bank, so a later stage is a function of an earlier one), and **untested**. It depends on depth producing *better* representations, which is what an addressable write (M1) would enable | CL-4, after the ruler exists |

---

## 5. The steps

Each step is independently gated, states its falsifier in advance, and ends
in a written verdict in `RESULTS.md`.

### CL-0 — Build the ruler. (1 h, CPU)
Build the three-number CL-RULER on the existing harness (`tools/ruler.py` +
`tools/membench/generators.py` + `train_gpu.py`'s corpus machinery): a fact
stream generator, a replay set, a recall cell with a permuted-label veto, and
a CE-regression cell. **Nothing below CL-0 counts as a result.**
*Falsifier:* if the ruler cannot reproduce the model's own CE on a known
checkpoint, it is broken — fix it before use.

### CL-1 — Re-measure the read path with a working instrument. (1 day, CPU)
Fix the two defects in §6, then re-run the copy/retrieval cells on the
existing P4+P6 checkpoint with (a) the corrected ridge readout, (b) all four
payload bytes scored with the right chance level, and (c) a **key-identity vs
payload-identity** split.
*Why the split:* the state is a *mixture*; if the **key** is decodable but
the **payload** is not, the write has address but no content (read-path
framing is right). If **neither** is decodable, the write is order-losing and
the priority is the write, not the readout. One hour of CPU decides which
half of the previous plan's pivot was real.
*Falsifier:* if the corrected readout beats its veto at d=32, the previous
plan's central claim is refuted and the priority moves to the write — which
is M1, and it is a bigger job than planned.

**CL-1b — the probe must be POWERED before its verdict means anything.**
A second, deeper limit on the copy probe was found while fixing D1: the
readout is a **256-way one-hot ridge**, so it can only ever *emit* a byte
value it saw in its training half. The copy task's payloads are **random**,
so with `n` samples the training half contains only ≈ `1 − exp(−n/256)` of
the 256 byte values — with the ruler's `sample_count=128` that is **≈ 22 %**
of the alphabet. The probe's per-byte accuracy is therefore capped near
22 % *whatever the state contains*, and the 4-byte exact rate is capped near
`0.22^4 ≈ 0.2 %`, against a stated chance of 2⁻³². The published metric and
its chance level are not on the same scale, and a probe that is
coverage-limited cannot distinguish "the state is empty" from "the probe
cannot name the answer".

So CL-1 must report, for every cell: the **class coverage** (the fraction of
test-half payload byte *values* that occur anywhere in the training half, at
the byte level and at the 4-byte level) next to the accuracy. The probe is
properly powered only when coverage ≈ 1, which needs `n` of a few thousand
samples, or a **matched alphabet** (a fixed small set of payload byte
values, with the alphabet size reported). A verdict from a coverage-limited
probe is `UNMEASURED`, not `PASS` and not a null.
*Falsifier:* if a fully powered probe still sits at its veto, the read-path
null stands (on a sound instrument) and the pivot to the write is correct.

### CL-2 — The first trained continual-learning run. (2 days, T4)
Train on a synthetic fact stream with the fact **in the loss** — the
"Selective-copy training run — not started" item from
`RESPONSE-2026-10-06-EXTERNAL-CRITIQUE.md`. Plus a replay set of pre-stream
facts to measure forgetting. This is the decisive experiment: it makes memory
a **loss** instead of a probe.
*Falsifier:* if after the agreed budget the trained model does not reach
≥ 0.50 exact recall at d=32 (chance 2⁻³²), the recurrent state cannot be
trained to hold a fact, and the in-state route (M1) is expensive for nothing.

### CL-3 — The four ideas, on the ruler. (1 week, T4)
Four arms, same seed / corpus / byte budget, judged **only** on the three
CL-RULER numbers:
 (a) plain CE control, (b) + per-channel lifetime promotion/demotion
 (idea 2), (c) + experts as decay-sets/LoRAs with a learned router (idea 1),
 (d) + quarantine and verified-signal promotion (idea 3).
*Falsifier:* an arm that does not beat the control on ACQUIRE and does not
lose it on RETAIN and NO REGRESSION is a null, and is closed.

### CL-4 — Cross-depth self-distillation. (later)
Only after the ruler exists. Add a loss pulling an earlier stage toward a
later one, keeping the CE objective intact, and judge it on the same three
numbers plus compute cost.

### CL-5 — M1 (addressable write), if and only if CL-2 and CL-3 justify it.

---

## 6. Two defects to fix before anything is measured (found 2026-10-21)

**D1 — the read-path instrument is broken.** `tools/membench/model_backend.py:130`,
`_ridge_score`:

```python
if acc > best_acc:
    best_acc, best = lam, acc     # ← SWAPPED
```

`best_acc` receives the *lambda* and `best` receives the *accuracy*, so:
λ=1e3 and λ=1e5 are never tried (after the first iteration `best_acc` is
1.0 and `acc > 1.0` is never true), and `chosen = best` is a **training
accuracy**, so the final fit solves `G + 0·I` — unregularized, on a
rank-deficient 64×64 Gram over 8,192 features. Verified by calling the
project's own function: it returns `lam = 0.0`. The ruler's own stored output
is the fingerprint — `runs/taxis/ruler_A_p4_wd0_c.txt` prints `lam=0`, which
the intended code cannot produce.

Second defect in the same cell: `y = targets.astype(np.int64).ravel()`
flattens the `(n, 4)` payload targets, and the split then indexes only
`idx ∈ [0, n)`, so **only payload byte 0 of each sample is ever decoded** —
while the metric is published as *"fraction of exact 4-byte payload matches"*
against a chance level of 2⁻³² (the true chance for byte 0 is 1/256).

⇒ **"The readout never beats its own control" → "the read path does not
work" → the previous plan's pivot is NOT established.** It is an artifact of
a broken probe. This is the same failure mode the project has already caught
three times (the harness that "never ran", the contention measured 256× too
high, the FSP arm that "was unmeasurable") — this time in the instrument
that produced the headline verdict.

**D2 — the defining number is unreproducible.** `0.81 nats/byte` (order-8
byte n-gram) vs `1.2094` (TMT) appears in `RESULTS.md`, in
`STRATEGIC-REVIEW §1` and in `PLAN §0` finding 2. The script that produced
it, `lag5.py`, was never committed and no longer exists. The corpus it needs
is present. Two further facts to record when it is rebuilt: the original
comparison was **not byte-matched** (the n-gram was scored on a 300 kB head
of `tinystories_valid.txt`; the model is scored on 6 × 20,001-byte slices
spread across **both** files in `data/eval/`), and the n-gram's training file
(`data/eval/wiki_tinystories_005.txt`) is also one of the model's scored
eval files.

**D3 — the copy probe is coverage-limited (found while fixing D1).** See
CL-1b. The readout can only emit byte values it saw in its training half;
with random payloads and `sample_count=128` that is ≈ 22 % of the alphabet,
so the probe's ceiling is far below the chance level it is quoted against.
This is a third instance of the same failure mode — an instrument that was
never checked against its own null.

---

## 7. Why the last 14 plans did not converge, and what changes

The programme ran a **proxy-substitution loop**:

1. the goal was not expressible in the engine and was not a criterion in any
   plan, so each plan substituted a measurable proxy;
2. each proxy got a hard, well-instrumented experiment, and most were
   **nulls** — good science, no movement;
3. each null got a documented cause, the cause generated the next plan, and
   the programme read as advancing because verdicts kept landing.

**What changes, concretely:**
1. **The goal is a criterion.** CL-RULER is three numbers with thresholds
   fixed in advance, and every future experiment is judged on them.
2. **The engine can express the goal.** The ~60 lines that carry generation,
   a no-grad forward and a chat/stream loop from `numpy_port.py` to
   `gpu_port.py` (`Model.sample`, `Model.frozen_step`, `Runtime.chat`),
   parity-gated by `tests/check_parity.py`. The unmerged `web-ui` branch is a
   *working numpy-based demonstration of exactly this goal* (SSE byte
   streaming, `remember()` = one `step_batch` per byte at B=1) and needs only
   a three-method `numpy_port` delta (`observe_stats`, `observe`,
   `sample_stats`); it must be **cherry-picked, not merged** (it lacks all of
   `t-axis`: +3,046 / −52,617).
3. **One current plan.** `docs/INDEX.md` currently marks 9 documents "current
   — read before new experiments", omits 5 live ones, and points at 2 closed
   axes. From now on: exactly one plan is current (this file), everything
   else is `superseded` or `closed` and reachable by name.
4. **A whole-programme kill criterion**, which the repo has never had (§8).

---

## 8. The kill criterion

**The programme is closed if, after CL-2 and CL-3:**
- the trained model does not reach ≥ 0.50 exact recall at d=32 (chance
  2⁻³²) on the fact task, **and**
- the M2 (external keyed store) arm also fails to beat 0.90 recall at d=32
  while keeping ΔCE ≤ 0.02 nats/B, **and**
- the T-axis re-test at the corrected 1.26× speed still favours T=1 by
  > 2σ at matched bytes.

In that case the honest conclusion is: *this architecture, at this scale,
with these objectives, does not do continual learning* — and the programme
switches to (a) packaging the findings for upstream and the community, and
(b) scale and data on the upstream configuration, which is where the evidence
says the remaining headroom is.

**Intermediate kill (per arm):** any arm that fails its pre-registered
threshold is closed and is not re-run with different knobs. That rule is
what the T-axis A/B needed and did not have.

---

## 9. Governance

- `RESULTS.md` stays **append-only**. Corrections are new sections that
  supersede older ones; rows are never edited. A CE without its SE is not a
  result.
- Exactly **one** plan may be marked current. When this one closes, the next
  is written before another starts.
- Every experiment states, *before* it runs: the hypothesis, the arms, the
  ruler setting, the threshold, and the verdict template.
- A null with a measured cause is a result. A null with an instrument bug is
  a bug. `UNMEASURED` is never `PASS`.
- The measurement apparatus is treated as code with gates: **any number that
  decides a decision needs a test** (this is how D1 will be prevented from
  recurring).
