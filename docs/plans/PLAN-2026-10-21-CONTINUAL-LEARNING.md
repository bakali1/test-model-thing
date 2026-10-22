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

> ## THE CURRENT STATE (2026-10-22, after one night of experiments)
>
> **Read this box before anything else in this plan. It is the summary; the
> sections below are the reasoning, and several of them are older than the
> results and have been amended in place.**
>
> **The goal is unchanged:** continual learning, judged by three numbers
> (§1). **The instrument exists and works** (`tools/cl_ruler.py`; 35
> gates). **The first trained continual-learning arm has been run, and it
> is a FAIL on two of the three numbers** (`RESULTS.md` §20):
>
> | | result | threshold | verdict |
> |---|---|---|---|
> | **ACQUIRE** | 0.8906 exact-4 on the stream's facts | ≥ 0.50 | **PASS** |
> | **RETAIN** | ratio 0.0022 | ≥ 0.90 | **FAIL** |
> | **NO REGRESSION** | 3.0695 → 5.2037 nats/B | ≤ +0.02 | **FAIL** |
>
> **And the programme now knows WHY** — which is the night's real product
> (`RESULTS.md` §17–§23, in four steps):
>
> 1. **The recurrent state carries the ADDRESS; the weights carry the
>    CONTENT** (§22). A fresh readout on the state, split over *board
>    facts* rather than rows, scores **exact-4 = 0.0000 in every cell**.
> 2. **The cause is the TRAINING SIGNAL, not the architecture** (§23).
>    With the head's table removed (`--head-frozen`) the memory loss **does
>    not fall at all** over 19 steps, where the trainable-head arm fell
>    4.5 nats in 25. The state was never asked.
> 3. **P3 — write selectivity in the recurrent state — is CLOSED as a
>    lever** (§23, M1/M6). The gate engages but is not selective (its
>    between-sample spread is **2.4e-05** of its within-sample spread), and
>    a selective *state* write could not have fixed a failure that was a
>    *table* being overwritten.
> 4. **The plan's own §4/M2 — an explicit keyed store alongside the state
>    — is the answer**, and a D5-era update talked the programme out of it
>    and into P3. **§4 has been restored.**
>
> **The next three things** (`docs/NIGHT-LOG-2026-10-21.md`): a longer H1
> (500 steps, ~45 min) to make the negative conclusive; a **D5-style copy
> circuit trained explicitly** inside the CL-2 arm (D5 proves the state
> *can* carry a payload, so it is **untrained for the job, not
> incapable**); and the **keyed store** (M2).

## 0. What is already true (evidence, not plans)

All numbers are in `RESULTS.md` (append-only) with their ruler settings. This
section exists so a new reader does not re-derive them.

**Assets that exist and work.**
- A bit-identical CPU/CUDA engine with a parity gate (`numpy_port.py` ≡
  `gpu_port.py`, `tests/check_parity.py`).
- **35** executable gates in `tests/` (25 at the time this section was
  written; ten more were added tonight, six of them for the tools below).
- A training-time credit apparatus (ring / tail / trace) with **zero inference
  parameters** — it is plumbing, not weights.
- Colab + private-HF durability that survives runtime recycling, with
  content-verified transport (`checkpoint_hub.py`).
- A ruler (`tools/eval.py`, `tools/ruler.py`) and a synthetic memory-benchmark
  suite (`tools/membench/`).
- A speed result: the fused `nvrtc` ring credit is **1.26× end to end, bit
  identical** (loss `7.58001`, `max|Δ| = 0`; 26.72 vs 33.68 ms/step).
  **AMENDED 2026-10-22: this 1.26× was NEVER REALISED by any default run
  until tonight.** The `nvrtc` default was **inert** — `GpuModel.__init__`
  eagerly rewrote `"auto"` to `"compiled"`, so the branch that selects the
  fused path was dead code (the eager rewrite dated from 2026-09-24, when
  `compiled` *was* the fastest, so it was correct then and **silently became
  wrong** when `nvrtc` shipped). The default is now resolved lazily
  (`tests/check_ring_credit_default.py`). **The speedup that is actually
  bit-identical and in force is the fused AdamW: 1.09× on the whole step**
  (151.26 → 138.83 ms/step, `max|Δ| = 0` on CPU and MX150).
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
- ~~That the state can be read out.~~ **SETTLED 2026-10-21, on a fixed
  instrument.** The state **can** be read out, and **further than anyone
  thought**: with the state read at the cue, a matched 16-symbol alphabet,
  n=2048 and a matched null, the payload decodes **4.6× over its own null
  at 8–12 B of lag**, 3.3× at 32 B and 1.5× at 128 B, while the **cue key
  decodes 65–165× over null and holds ~27 % at 128 B**. The claims that
  "the readout never beats its own control" and that the horizon is "~12
  bytes" were both artifacts of broken instruments (D1, D4). The Δh
  capacity curve and the decodable-payload curve **are the same curve**, so
  there is no capacity-vs-usable-content paradox (D5).
  **AMENDED 2026-10-22 (§19, §22, §23).** D5 **survives its own copy
  control** (its nuisance collapses the decode to 0.56 %/0.32 % against a
  0.39 % chance), so **this model's state really does hold a decodable
  payload** — the measurement is sound. **But it is a DIFFERENT MODEL**: D5
  is a delayed-copy model whose objective asked the state to carry the
  payload, and the CL-2 arm's objective never did (§23). Two consequences,
  and the second one has been routinely misread:
  * **D5 says the state is CAPABLE of holding a payload.** So the CL-2
    failure is **untrained, not incapable** — which is why the next arm is
    a *D5-style copy circuit trained explicitly*, not a bigger state.
  * **D5's "the address persists, the content decays" is NOT evidence that
    write selectivity is the lever** (§23). The address costs ~1 dim per
    fact and the payload 4–16 dims per byte, so under capacity pressure
    the *cheap* thing survives and the *expensive* thing is dropped —
    "the address persists and the content decays" is what a
    **capacity-limited linear store** looks like
    (`tools/state_capacity_budget.py`: the state is 1.06×–4.06× short for
    ONE board of 512 facts). **P3 was aimed at the wrong object.**
- ~~The n-gram deficit.~~ **SETTLED 2026-10-21, reproducibly and
  byte-matched** (D2): the order-8 byte n-gram scores **0.6512** nats/byte
  on the ruler's own held-out bytes against the model's **1.2094** — the
  model is **0.56 nats/byte worse than a zero-parameter table lookup**. The
  deficit is real, larger than the 0.40 the old (unmatched) comparison
  suggested, and now has a script.
- ~~Anything about continual learning. `grep -c forgetting RESULTS.md` → **0**.~~
  **SUPERSEDED 2026-10-22 — this line was the most misleading in the
  plan.** It is now the opposite: `RESULTS.md` has 23 sections, §17–§23
  are entirely about continual learning, and there is a **full verdict
  from a trained arm** (§20): **ACQUIRE PASS / RETAIN FAIL / NO REGRESSION
  FAIL**. See the CURRENT STATE box at the top.

---

## 1. The criterion

The goal becomes three numbers, measured on one ruler, with a control.

> **CL-RULER.** For a model M and a stream of new facts F:
> 1. **ACQUIRE** — after the stream, M recalls the new facts: **the facts
>    in the stream, on FRESH streams of them** (new layouts, new
>    distractors, new gap content — never the training sequences).
> 2. **RETAIN** — M still recalls the facts it knew before the stream
>    (the same metric on the pre-stream set, and the pre/post gap).
> 3. **NO REGRESSION** — M's held-out CE on the general corpus has not
>    degraded (ΔCE ≤ 0.02 nats/B), **measured against a TEXT-ONLY arm
>    with the same model, corpus and budget** — not against the same
>    model's own earlier checkpoint.
>
> **AMENDED 2026-10-22 (§20, §22).** Three of these were ambiguous, and
> the ambiguity was resolved in the wrong direction twice:
> * **ACQUIRE's board.** The original wording said "a held-out set of the
>   same shape", which reads as *facts the model was never taught*. The
>   ruler's headline was therefore wired to board C, and that **turned a
>   PASS into a FAIL**: the head scores 0.8906 exact-4 on the stream's
>   facts (board B) and 0.0000 on board C. **The new facts ARE board B** —
>   that is what a stream is. **Board C is a separate and stronger claim**
>   ("recall a fact you were never taught"), and it is reported **beside**
>   the criterion, never as it. Both cells must be stated, and neither may
>   be mistaken for the other.
> * **RETAIN's readout.** A refitted probe and the model's own parameter
>   give **0.857 and 0.0022** for the same checkpoint. **The criterion uses
>   the model's own readout** (the trained head), because a refitted probe
>   measures the state's *copyability*, not what the model learned. This
>   must be stated in the criterion, not discovered later.
> * **NO REGRESSION's baseline.** The pre-phase checkpoint is not a valid
>   baseline: the model was still improving steeply when it was taken
>   (5.99 → 3.11 → 2.99 over 200 steps), so "held-out CE went 2.99 → 5.20"
>   compares a model at step 200 with a model at step 2200 and leaves the
>   counterfactual **unknown**. The baseline must be the text-only arm.

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
  (**chance 16⁻⁴ = 1.53e-05** at the matched alphabet — *AMENDED: the plan
  previously wrote "chance 2⁻³²", which is the full-byte figure and is
  wrong for a 16-symbol alphabet; the same error survives in §8*) on a
  *trained* model; a probe on a frozen checkpoint is diagnostic, not a
  pass.
  **ALPHABET CAVEAT (2026-10-21, measured):** this threshold is only
  *measurable* with a **matched alphabet**, and the plan must say so. At the
  full 256-symbol alphabet the D5 measurement gives a payload per-byte
  accuracy of 12.11 % at 8 B and 2.25 % at 32 B, so the exact-4-byte rate is
  `0.0225⁴ ≈ 2.6e-7` — a 0.50 exact rate is unreachable at 32 B by
  arithmetic, whatever the model does. With a **16-symbol** alphabet the
  same state gives 22.09 %/byte at 32 B, an exact-4 rate of 0.49 %, and
  16⁻⁴ = 1.5e-5 as the chance. **So ACQUIRE is scored on a matched alphabet
  (16 or 4 symbols, reported), and the alphabet size is part of the ruler
  setting** — exactly as the SE and the sample count are. The full-256 run
  is kept as the *upper* bound on the signal (it gave 19.2× over null, the
  strongest ratio measured), and the matched-alphabet run is the one the
  threshold applies to.
- **RETAIN** passes at ≥ 0.90 of the pre-stream recall, measured on a
  **replayed** set of pre-stream facts that were *not* in the stream.
- **NO REGRESSION** passes at ΔCE ≤ 0.02 nats/B on the general ruler.
- A **veto control** accompanies every ACQUIRE cell: the same readout on
  *permuted* labels. If the readout cannot beat its own veto, the cell is
  UNMEASURED, never PASS. (The repo already has this discipline in
  `tools/p0_verdict.py`: "UNMEASURED is never PASS".)

**THE MANDATORY CONTROLS (added 2026-10-22, §17, §19, §22).** Tonight the
programme produced four wrong conclusions in a row, each because a
control was *absent* and its absence silently became the interpretation.
**A cell without these is not a result.**

1. **THE PERMUTED-LABEL VETO** (above) — unchanged.
2. **THE COPY CONTROL.** Replace the payload bytes **in the input** with
   fresh random bytes, leaving the cue and the target unchanged. A readout
   that still answers has learned the *key*, not the *payload*, and is
   reading a table rather than the state. **This is the control that
   separated "the state holds a copy" from "the head holds a table"** — and
   it changed the programme's conclusion.
3. **THE FACT-LEVEL SPLIT, for every FITTED readout.** A readout fitted on
   rows can memorise a **table over the board**, because the rows share
   board facts. **Split over the things the readout could memorise** — here
   over the board facts, not over the rows. A cell fitted on a row split is
   UNMEASURED until it is re-run.
4. **THE NUISANCE CONTROL** (the target's key AND payload both replaced) —
   the readout must collapse to its null. It does, in every cell.

**A REFIT PROBE IS NOT A RETENTION MEASURE.** A readout that is refitted at
each measurement measures the state's *copyability*. Retention is measured
with the model's **own parameters**.

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

**UPDATE 2026-10-21 (D5):** the D5 measurement does not change M1-vs-M2 —
it changes what is *inside* M1. The state already holds an addressable,
persistent address and a decaying-but-real content, and the readout is not
key-blind. So M1's "hierarchical feature-state wiring" is **not** the next
month: the lever is the write's **selectivity** (the P3 gate), because the
address survives where the content decays, which is the signature of an
order-losing weighted average.

> ### UPDATE 2026-10-22 (§22, §23): THE D5 UPDATE WAS WRONG, AND M2 WAS RIGHT.
>
> **The paragraph above is superseded. It is left in place because its
> error is instructive, not because it is true.**
>
> **What the D5 update got right:** the state holds an addressable,
> persistent address; the readout is not key-blind. That survives, and
> §22 confirms it more sharply: the state carries the **address** (the
> key), durably and generically, even for facts the model was never
> taught.
>
> **What it got wrong, in two ways.**
> 1. **"The lever is the write's selectivity" is falsified.** The gate
>    *engages* (it reaches 0.110 max) but it is **not selective**: its
>    between-sample spread is **2.4e-05** of its within-sample spread
>    (M1) — it is a global per-dimension re-scaling. And M6 shows the
>    conflict cost is not reduced specifically by the gate. **Worse, a
>    selective *state* write could not have fixed the failure at all**,
>    because the failure was **a table in the weights being overwritten**,
>    not a state being overwritten. **P3 was aimed at the wrong object.**
> 2. **"The content decays" was read as a write problem; it is a
>    CAPACITY result.** The address costs ~1 dim per fact and the payload
>    4–16 dims per byte, so under capacity pressure the cheap thing
>    survives and the expensive thing is dropped. "The address persists and
>    the content decays" is what a **capacity-limited linear store** looks
>    like: the state is **1.06×–4.06× short** for ONE board of 512 facts
>    (`tools/state_capacity_budget.py`).
>
> **And the D5 update talked the programme OUT of M2, which §23 lands back
> on.** §23's conclusion is that the goal needs a store that is **neither a
> copy nor a table** — one that is written once at a key and read many
> times without evicting its neighbours. **That is M2.** The plan had it
> at the top of this section all along, and it was correct; the D5 update
> was not. **M2 is now the primary mechanism and M1 is closed as a lever**
> — not because an addressable write is impossible, but because (a) the
> state is **untrained** for it (D5 proves the state *can* carry a payload
> when its objective asks it to), and (b) the failure that is actually
> costing the criterion was a **table** being overwritten, which a state
> write cannot reach.

**The four project ideas map onto these mechanisms:**

| idea | where it lives | first test |
|---|---|---|
| **1. Experts for new domains** | a decay set (8,192) + optional rank-r readout LoRA (1,024r/layer) per expert, selected by a learned router. No router exists in the code today (`grep expert\|router\|topk` → 0 hits) | **deferred** — a router is a way to make M2's keyed store bigger, and the store is the prerequisite |
| **2. Short-term / long-term memory** | **already ~80 % built**: the per-channel decay logit *is* a lifetime; P4's log-spaced init creates the spectrum; P6 stops training eroding it. **AMENDED 2026-10-22: "the P3 gate makes retention input-dependent" is FALSIFIED** — the gate's between-sample spread is 2.4e-05 (§23, M1). Missing: a **promotion/demotion policy** and a ruler that measures lifetime-vs-usefulness | superseded by the M2 keyed store |
| **3. Prevent harmful self-learning** | the risk is currently **dormant**: `numpy_port.Runtime.ingest` (866-912) does update weights on user bytes, but generated bytes are **never re-ingested** (only upstream's `legacy/main.py:415` trains on its own output). The three missing safeguards are all missing: **no replay buffer or knowledge store exists**; `grep -c forgetting RESULTS.md` → **0** *(AMENDED 2026-10-22: now 1, and §17–§23 are all about forgetting — this cell is obsolete)*; and `verify_saved` (`train_gpu.py:517`) checks *transport* identity, never quality — **no checkpoint is ever accepted or rejected on quality** anywhere. The 35 gates validate *code*, not *capabilities* | superseded by the M2 keyed store + the CL-RULER |
| **4. Cross-depth self-distillation** | architecturally available today (depth = 16 stacked nonlinear readouts of one bank, so a later stage is a function of an earlier one), and **untested**. It depends on depth producing *better* representations, which is what an addressable write (M1) would enable | CL-4, after the ruler exists |

---

## 5. The steps

Each step is independently gated, states its falsifier in advance, and ends
in a written verdict in `RESULTS.md`.

### CL-0 — Build the ruler. (1 h, CPU) — **DONE 2026-10-21**
Build the three-number CL-RULER on the existing harness (`tools/ruler.py` +
`tools/membench/generators.py` + `train_gpu.py`'s corpus machinery): a fact
stream generator, a replay set, a recall cell with a permuted-label veto, and
a CE-regression cell. **Nothing below CL-0 counts as a result.**
*Falsifier:* if the ruler cannot reproduce the model's own CE on a known
checkpoint, it is broken — fix it before use.

### CL-1 — Re-measure the read path with a working instrument. (1 day, CPU) — **DONE 2026-10-21**
**DONE 2026-10-21. The verdict changed, and it changed in the programme's
favour.** The `λ` swap, the test-half leak and the payload ravel are fixed
(`tools/membench/model_backend.py`), the cell is gated by
`tests/check_ridge_readout.py` (which fails on the pre-fix code), and the
re-measurement on `runs/taxis/A_p4_wd0_c.npz` gave:

* the instrument **has power** — the same ridge over the same states decodes
  the **trailing key** at **41.7 %/byte** (d=8) and **21.4 %/byte** (d=32)
  against a ~0.5 % veto;
* class coverage is **not** the binding constraint (100 % coverage at
  n=2048, still only 2.3 %/byte for the payload);
* the payload **is** decodable, but only for **~12 bytes**: 12.4 %/byte at
  12 B of lag (23× its null), 1.69 % at 36 B, nothing at 132 B or beyond,
  while the trailing key stays at 25–41 % throughout.

So **"the read path does not work" was false** — it works over a horizon of
about a dozen bytes — and **"the write path works" was too strong**, because
the Δh capacity (37.5 % at 8 B, 17 % at 32 B) does **not** show up as
decodable payload beyond ~12 B.

**CL-1 is therefore not a step to run — it is the measurement that reframes
the programme**, and it leaves one sharp question (§6, D5) that everything
downstream depends on.

**The instrument's own requirements, now fixed as gates** (three defects,
one per requirement):
1. read the state **at the cue**, not after the batch's zero padding
   (D4 — `model_copy_readout` padded to the longest sample and read
   `final_states` at `t == L-1`, so every d≥32 cell was read after 24–504
   padding bytes);
2. report **per-byte**, not exact-4-bytes, at 256 symbols — `per_byte^4 ≈
   3e-7`, so the exact metric cannot report a trace this weak at any
   feasible n;
3. use a **matched alphabet** when an `exact` rate is wanted (16 symbols →
   21.3 %/byte, 4 symbols → 44.9 %/byte, both with their own nulls).

### CL-2 — The first trained continual-learning run. (2 days, T4) — **RUN 2026-10-22; the arm TRAINS**

**Status (verified from the MX150 logs, 2026-10-22; the raw files are in
`docs/results/cl2/armA_ce.jsonl` and `docs/results/cl2/armA.log`).** After
the detached-leaf / `requires_grad` fixes (see `RESULTS.md` §16), the arm
trains for the first time:

| phase | step | held-out CE | ce_text | ce_behav | head per-byte / exact-4 | behaviour per-byte / exact-4 |
|---|---:|---:|---:|---:|---:|---:|
| pre | 0 | 5.9863 | – | – | – | – |
| pre | 100 | 3.1128 | 2.6062 | 0.0000 | 0.000 / 0.000 | 0.000 / 0.000 |
| pre | 200 | 2.9856 | 2.4332 | 0.0000 | 0.000 / 0.000 | 0.000 / 0.000 |
| f1 | 100 | 8.4117 | 2.2806 | 1.5517 | 0.760 / 0.385 | 0.486 / 0.043 |
| f1 | 400 | 7.7234 | 1.8907 | 0.2247 | 0.902 / 0.709 | 0.947 / 0.816 |
| f1 | 700 | 6.0694 | 1.8195 | 0.0761 | 0.945 / 0.809 | 0.985 / 0.947 |
| f1 | 1000 | 4.9274 | 1.7640 | 0.0453 | 0.971 / 0.896 | 0.988 / 0.957 |
| f2 | 100 | 5.7354 | 1.8434 | 0.5084 | 0.727 / 0.336 | 0.836 / 0.525 |
| f2 | 500 | 4.1937 | 1.7256 | 0.0543 | 0.919 / 0.760 | 0.990 / 0.965 |
| f2 | 700 | 4.0011 | 1.6384 | 0.0405 | 0.944 / 0.818 | 0.992 / 0.967 |
| f2 | 1000 | 4.5938 | 1.5886 | 0.0181 | 0.975 / 0.910 | 0.999 / 0.994 |

**The read-out is a memory measurement; the behaviour is probably a copy.**
The memory head is a linear readout of the **state at the cue**
(`state_cue @ wm` in `tools/cl2_train.py:train_step`), so it structurally
cannot read anything outside the state; it reaches **0.975 per-byte and
0.910 exact-4** on the training board, against chance 1.53e-05 and a byte
prior of 0.1659. The model's own logits reach 0.999/0.994 at the payload
slots, but the payload sits in the **input** 32 B earlier, so that number
is most likely a **copy** and must be treated as the confound, not the
result.

**The new finding, and the next problem.** The general held-out CE
**regressed** — and the shape of the regression is the interesting part. In
f2 the text CE **rises** 4.1937 (step 500) -> 4.5938 (step 1000) while
`ce_behav` **falls** 0.0543 -> 0.0181 and the fact accuracies climb to
0.999/0.994: **as the fact behaviour gets better, the text gets worse.**
In f1 the same trade shows in reverse — the text recovers 8.4117 -> 4.9274
while `ce_behav` falls 1.5517 -> 0.0453. That is an interference
signature, and the mechanism candidate is specific: **the behavioural loss
is computed on the SAME logits as the text CE** (both come from `flat` in
`train_step`), so the model's one output path must model natural text *and*
emit an arbitrary 4-symbol payload. The behavioural CE opened at **8.78
nats** when the fact phase began.

**And the NO REGRESSION cell is NOT WELL-POSED as measured.** The `pre`
baseline of 2.9856 is **not a converged text model** — the text CE was
still falling steeply (5.99 -> 3.11 -> 2.99 over 200 steps, ~0.015
nats/step and still steep), so "held_ce went 2.99 -> 4.59" compares a
model at step 200 with a model at step 2200 and the **text-only
counterfactual is unknown**. It might have reached 1.5 (a huge regression)
or 4.0 (a small one). **NO REGRESSION is therefore UNMEASURED until a
text-only arm of the SAME budget exists.** That arm is the missing
control, and it is also the **matched null** for the regression cell.

**The next experiment is now CL-2b** (pre-registered, in
`docs/design/CL2-INTERFERENCE-EXPERIMENT.md`): separate interference from
fact-distribution-shift, with a matched control per candidate cause —
`behaviour-weight` 1.0 vs 0.1, a fact-diluted batch, a **decoupled head**
(the payload read by a separate readout instead of the shared logits), the
**text-only counterfactual** (the null for the regression cell), and
`--memory-alpha 0` (which answers a DIFFERENT question — does the *memory
loss* cost text, not does the *fact training* cost text). The pre-registered
test is a **shape**, not just an end point: the text CE should be
flat-to-falling across f2 under the decoupled head, instead of rising 0.4
nats.

*Falsifier:* if the decoupled-head arm does **not** bring the held-out CE
back within 0.05 nats/B of the matched control, then the regression is not
output-head interference and the cause lies elsewhere (e.g. the facts'
distribution, or the state itself being overwritten); the next lever moves
to the write.
*Original falsifier, now partly retired:* "if the trained model does not
reach >= 0.50 exact recall at d=32 the recurrent state cannot be trained to
hold a fact" — the head's 0.975 is at the **read-out**, not yet at the
**behaviour**, and the ruler's verdict governs.

### CL-2b — THE TRAINING SIGNAL. (2026-10-22; rewritten three times; **the reasoning that CL-6 executes**)

> **CL-2b is the RECORD of what was measured and what it means. It is not
> the current step — CL-6 is.** CL-2b was rewritten three times in one
> night, and each rewrite is a correction of the one before: it was
> "the store", then "the training signal", then the measured result. The
> section is kept whole because the sequence of corrections is the
> argument.

**What CL-2 measured** (`RESULTS.md` §20, §22; the ruler's raw output is
`docs/results/cl2/armA_ruler.json` and the fact-capacity output is
`docs/results/cl2/factcap.log`):

| | board B (the stream's facts) | board A (pre-stream, no replay) | board C (never trained) |
|---|---:|---:|---:|
| the model's own head, per-byte / exact-4 | **0.9697 / 0.8906** | **0.1978 / 0.0020** | 0.1011 / 0.0 |

* **ACQUIRE: PASS** (0.8906 exact-4, null 0.0020, threshold 0.50).
* **RETAIN: FAIL** — pre 0.9644/0.8691 → post 0.1978/0.0020, **ratio
  0.0022** against a threshold of 0.90. The pre-stream facts are gone.
* **NO REGRESSION: FAIL** — held-out CE 3.0695 (base) → 5.2037 (post), and
  the post model is **within 0.34 nats of the unigram baseline 5.5452**.

**THE PICTURE, AFTER THE AUDIT THAT CORRECTED IT TWICE (§17, §19, §21,
§22).**

> **The recurrent state carries the ADDRESS — the key — durably and
> generically, even for facts the model was never taught. The CONTENT —
> the payload — lives only in the WEIGHTS, in a table, and only for the
> facts it was trained on.** Board B's 512 entries overwrote board A's,
> which is why RETAIN is 0.0022.

The state does **not** hold a copy of the payload: a fresh ridge readout
on the state, fitted on board facts 0-255 and tested on facts 256-511,
scores **exact-4 = 0.0000 in every cell**, on board B and on board C
(`tools/state_fact_capacity.py`). The ruler's apparent 0.1953 exact-4 on
board C was a **table over the board**: its split is over rows, and rows
share board facts, so the state supplied the *index* and the readout's
weights supplied the *content*.

**AND D5 IS EXPLAINED BY THE LOSS, NOT THE ARCHITECTURE.** D5's model —
trained on a delayed-copy task — demonstrably **does** hold a decodable
payload in its state (19.2x over null at 8 B, with its copy control
collapsing to chance). The CL-2 model does **not**. Same architecture,
same state size. **D5's objective asked the state to carry the payload;
CL-2's objective never did** — the memory loss `state_cue @ wm` is far
cheaper to satisfy by fitting the *head's weights* to a table, and the
gradient took that path. **So the CL-2 failure is a TRAINING-SIGNAL
failure: the state was never asked.**

**ARM H1 HAS NOW BEEN RUN, AND IT ANSWERED THE QUESTION (§23).** With
`--head-frozen` (the head held at its random init, so it cannot accumulate
a table), the memory loss **does not fall at all** over 19 steps — it sits
at ~5.9, above the uniform level — while armA's falls 4.5 nats in 25
(5.9234 -> 1.4057 -> 1.0041). **The entire drop in armA was the head's
table being fitted.** *Caveat: 19 steps is a strong signal, not a proof;
the conclusive version is 500 steps, ~45 minutes.* And **P3 is closed as a
lever**: the gate engages (it reaches 0.110 max) but its between-sample
spread is **2.4e-05** of its within-sample spread — a global per-dimension
re-scaling, not selectivity — and a selective *state* write could not have
fixed the failure anyway, because the failure was a **table being
overwritten**.

**THE DECISIVE EXPERIMENT, AND IT IS CHEAP. ARM H1 — REMOVE THE HEAD'S
TABLE FROM THE EQUATION.** Train the same arm with the memory loss
computed so the head's weights **cannot** carry the content (the head is
re-initialised every step, or replaced by a fixed random projection, or
frozen at a random init). Then the only way to reduce the memory loss is
to **write the payload into the state**. Everything else identical to
armA. **Pre-registered:**

* the memory loss must fall **below armA's 0.0200-0.0453**;
* with the head **zeroed at eval**, the head's exact-4 on board B must
  stay **>= 0.50** (the ACQUIRE threshold);
* the head's exact-4 on board A (post) must rise by **>= 0.05** over
  0.0020 (RETAIN);
* the held-out CE must stay within **0.05 nats/B** of armA's 5.2037.

**A FAILURE IS A RESULT AND IT IS WORTH HAVING.** If the loss cannot fall
with the table removed, the state cannot carry the payload under this
objective, the D5-style copy circuit must be trained explicitly, and the
programme's next architecture is a **content-addressed store with learned
addressing** — a third kind of store, neither a copy nor a table.

**ARM H2 = H1 with `--board-size 64`.** The capacity budget
(`tools/state_capacity_budget.py`) says the state (8192 scalars) is
**1.06x-4.06x short** for one board of 512 facts in a linear-readable
store, and that 64 facts per board (2 176 dims for two boards) fits
comfortably. **If only one of H1/H2 can be run, run H2** — it is the arm
most likely to succeed and the most informative.

**THE OTHER ARMS, AND THEIR PLACE NOW.** The **capacity** arm (a bigger
state, dim >= 1088 optimal / >= 4160 one-hot) and the **store-kind** arms
(a content-addressed slot store, a protected table, **replay**) are still
needed, but they are now **downstream of the training-signal question**:
if H1/H2 succeed, the programme has a working memory and the question
becomes how to make it scale; if they fail, the store kind becomes the
priority. **REPLAY remains the missing reference point** — until a replay
baseline is measured on the same three numbers, "continual learning is
hard" is a claim without a control.

**WHAT THE AUDITS TAUGHT, AND IT IS THE PROGRAMME'S REAL METHOD NOW.**
Four times tonight a measurement's meaning was set by a control that was
absent: a missing copy control (§17), a missing nuisance control (§19), a
missing criterion wiring (§20), and a **split that let a readout memorise
the board instead of generalising** (§22). The rule that comes out of it:
**when a readout is fitted, the split must be over the things the readout
could memorise** — here, over the board facts, not over the rows — and
**every fitted readout in the ruler must be re-run with the fact-level
split before it is believed.**

**TWO FIXES TO THE INSTRUMENT, both the same class of bug (a tool default
that does not match the question).**
* **The ruler's ACQUIRE headline is wired to board C** (never trained),
  which asks a different question from the criterion's "recall the NEW
  facts" (board B) and **turns a PASS into a FAIL**. Both cells must be
  reported; the criterion's ACQUIRE is board B, with board C beside it as
  the separate, stronger claim.
* **The ruler's `--facts` default (64) does not match the trainer's (4).**
  The checkpoint stores `cl2_facts`/`cl2_gap`/`cl2_alphabet`, so the ruler
  should read the arm's regime **from the checkpoint**.

**AND THE NO REGRESSION FAILURE IS NO LONGER BLOCKED ON ITS COUNTERFACTUAL.**
The text-only arm (A5) would have to reach 5.20 nats/B for the regression
to be zero, and the base was 3.07 while the model was still improving
steeply — so the counterfactual can only make the regression **larger**.
The failure is certain; A5 quantifies it. The text regression and the
RETAIN failure are **two different failures with two different causes**
(the behavioural loss rides on the same logits as the text CE; the head's
table is overwritten) and the design must keep them apart.


### CL-3 — The four ideas, on the ruler. (1 week, T4) — **SUPERSEDED 2026-10-22; do not run**

> **Why it is superseded.** CL-3 was four arms of *architectural* ideas
> judged on the three numbers. §22–§23 showed that the thing standing
> between the programme and a PASS was **not an architecture the four ideas
> could reach**: the state carries the address, the weights carry a table,
> and the table is overwritten. The four ideas (experts, lifetime
> promotion, quarantine, distillation) all act on the **state or the
> decay**, and the state is not where the content is. **The replacement is
> CL-6: the keyed store (M2), plus the training-signal arms.**
>
> The text below is kept for the record. **Do not run it.**

Four arms, same seed / corpus / byte budget, judged **only** on the three
CL-RULER numbers:
 (a) plain CE control, (b) + per-channel lifetime promotion/demotion
 (idea 2), (c) + experts as decay-sets/LoRAs with a learned router (idea 1),
 (d) + quarantine and verified-signal promotion (idea 3).
*Falsifier:* an arm that does not beat the control on ACQUIRE and does not
lose it on RETAIN and NO REGRESSION is a null, and is closed.

### CL-4 — Cross-depth self-distillation. (later) — **DEFERRED**: §22–§23 show the blocker is the store, not the representation; a better representation of a state that holds only the address does not help.
Only after the ruler exists. Add a loss pulling an earlier stage toward a
later one, keeping the CE objective intact, and judge it on the same three
numbers plus compute cost.

### CL-5 — M1 (addressable write), if and only if CL-2 and CL-3 justify it. — **CLOSED as a lever (§23)**: the gate engages but is not selective (between/within 2.4e-05), and a selective *state* write cannot fix a table that was overwritten. **The addressable write survives only as the D5-style copy circuit, which is a training-signal question, not a write-selectivity one.**

---

### CL-6 — THE STACK: a training-signal arm, then a keyed store. (the current step)

> **This replaces CL-3/CL-4/CL-5.** Three arms, in this order, because each
> one is cheap and each one changes what the next one means. The full
> reasoning and the pre-registered thresholds are in §CL-2b and in
> `docs/plans/PLAN-2026-10-21-CONTINUAL-LEARNING.md` §CL-2b's arm list.

**CL-6a — THE LONGER H1 (500 steps, ~45 min).** Make tonight's negative
conclusive: with the head frozen, does the memory loss fall *at all* over
500 steps rather than 19? §23's 19-step probe showed **no downward trend
whatsoever** where the trainable-head arm fell 4.5 nats in 25, but 19 steps
is a strong signal, not a proof. **If the loss falls, the state CAN be
taught to carry the payload and the whole programme is a training-signal
problem.** If it does not, the state cannot be taught to carry the payload
under this objective and the keyed store is the only way forward.

**CL-6b — THE D5-STYLE COPY CIRCUIT, TRAINED EXPLICITLY (~2 h).** Add a
loss that asks the *state* to carry the payload, with the head's table
removed (`--head-frozen`) so the table cannot take the job. **D5 proves
the state CAN carry a payload when its objective asks it to** (its copy
control collapses to chance, so it is not a copy of the input), so the
state is **untrained for the job, not incapable.** This is the arm that
distinguishes "the state cannot" from "the state was not asked".

**CL-6c — THE KEYED STORE (M2).** A small associative memory (key → value)
alongside the state, which the model learns to write to and read from,
with the recurrent state unchanged. **This is the architecture the goal
needs** (§23): a fact is written **once** at a key and read **many** times
without evicting its neighbours. Neither a copy (overwritten by the next
write) nor a table (overwritten by the next stream) has that property,
and that is why ACQUIRE passes and RETAIN is 0.0022 in the same run.
**Its cost is honest and must be reported: the knowledge is not in the
weights.** The ruler already has the readout to score it.

**ALSO OUTSTANDING, AND CHEAP.** The **text-only counterfactual (A5)** for
NO REGRESSION's magnitude (the failure itself is certain, §20); the
`--memory-alpha 0` control arm, **stopped at f2 step 100 and resumable**
from `~/tmt-work/runs/cl2/armB_f2_t000100.npz`; and the **replay
baseline**, which the programme still does not have and without which
"continual learning is hard" is a claim without a reference point.

---

## 6. Two defects to fix before anything is measured (found 2026-10-21) — **BOTH FIXED 2026-10-22**

> **Status: both fixed and gated.** D1 (the ridge readout's swapped lambda
> selection, and the multi-byte payload read once instead of per byte) and
> D4 (the copy probe reading `final_states` after zero-padding rather than
> at the cue) are fixed, with `tests/check_ridge_readout.py` failing on the
> old code. **Three more defects were found tonight, and they are the same
> KIND of defect — an instrument that silently measures something else:**
>
> | defect | what it did | gate |
> |---|---|---|
> | **the `nvrtc` default was inert** | `GpuModel.__init__` eagerly rewrote `"auto"` → `"compiled"`, so the branch that selects the fused path was **dead code** and the 1.26× speedup was **never realised** by any default run | `tests/check_ring_credit_default.py` |
> | **the ruler's ACQUIRE headline was wired to the wrong board** | board C (never trained) instead of board B (the stream's facts) — **turning a PASS into a FAIL** | `tests/check_ruler_criterion.py` |
> | **the ruler's ridge split was over rows** | the rows share board facts, so a fitted readout could **memorise a table over the board** and report a "generalisation" that was not one — the single most consequential error of the night | `tests/check_ruler_criterion.py` |
>
> **The lesson, which is now a governance rule (§9): a measurement's
> meaning is set by its controls, and a control that is ABSENT silently
> becomes the interpretation.**

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
  *(AMENDED 2026-10-22: 2⁻³² is the FULL-BYTE figure and is wrong for
  the matched 16-symbol alphabet, where the chance for a 4-byte payload is
  16⁻⁴ = 1.53e-05. The same error survives in §8 and has been fixed there;
  it is left visible here because it is the same class of error as D1 — a
  number that is right for one setting and quoted for another.)*

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

**D2 — RESOLVED 2026-10-21.** `tools/ngram_ce.py` is the rebuild, with an
exact (aliasing-free) keying and a `--selftest` against a pure-Python dict.
**Byte-matched on the ruler's own held-out bytes, the order-8 n-gram scores
0.6512 nats/byte (0.7458 at r=0.7; 0.90–0.65 across the r sweep) against
the model's 1.2094 — 0.56 nats/byte better, ×1.74 in perplexity.** The
original 0.81 is consistent with that range, so the claim survives and is
now reproducible. The rebuild supports orders 1–8 (the original was
MAXORD=12), so it is a **weaker** baseline than the original — which makes
the conclusion stronger, not weaker. Full table and caveats in `RESULTS.md`
§6.

**Consequence for the programme:** the CE ruler cannot be the headline. A
zero-parameter, 8-byte-context lookup table beats the 4.5M-parameter trained
recurrent model by half a nat per byte on the same bytes. **The bar the
model has to clear before any memory claim is worth making is the n-gram's
CE**, and the plan's own criterion (§1) should be read with that in mind:
ACQUIRE and RETAIN are the memory claims; NO REGRESSION is the *floor*, and
the floor is currently **0.56 nats/byte above a table lookup**.

**D3 — the copy probe was coverage-limited (found while fixing D1; then
REFUTED as the binding constraint).** See CL-1b below.

**D4 — the copy cell read the state AFTER the batch's zero padding.**
`model_copy_readout` pads the variable-length batch to the longest sample
and reads `final_states`, which `_frozen_forward` takes at `t == L-1`; so
at d≥32 every short-gap sample was read after 24–504 padding bytes instead
of at the cue. Only the d=8 cell was padding-free.

**D5 — the measurement paradox, which is now the programme's central
question.** The P4+P6 init gives a **95.4 B half-life** and the Δh retention
curve reports **37.5 % at 8 B, 17 % at 32 B, 5.3 % at 128 B**; but the
*decodable* payload horizon was measured at **~12 B** (12.4 %/byte at 12 B
of lag, 23× its null; 1.69 % at 36 B; nothing at 132 B). Meanwhile the
**cue key** decoded at 25–41 % per byte even at 516 B of lag.

**D5 — RESOLVED 2026-10-21 (`tools/d5_probe.py`).** The paradox was an
instrument artifact, like D1 and D4 before it. With the state read **at the
cue**, a **matched 16-symbol alphabet**, n=2048 (full class coverage) and a
**matched null** per cell, the payload decodes:

| gap | payload | null | ×null | cue key | null | ×null |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 28.03 % | 6.05 % | **4.63×** | 47.95 % | 0.29 % | **165×** |
| 12 | 29.49 % | 6.42 % | 4.59× | 43.21 % | 0.44 % | 98× |
| 32 | 22.09 % | 6.64 % | 3.33× | 28.64 % | 0.32 % | 90× |
| 64 | 15.89 % | 5.66 % | 2.81× | 27.12 % | 0.42 % | 65× |
| 128 | 10.16 % | 6.67 % | 1.52× | 26.88 % | 0.32 % | 84× |

and the Δh curve and the decodable-payload curve **are the same curve**
(37.5 % vs 21.98 % at 8 B; 17 % vs 15.45 % at 32 B; 5.3 % vs 3.49 % at
128 B).

**The answer is (a) and (c) together, and NOT (b).**
- The **address is strongly present and persistent**: the cue key decodes
  65–165× over its own null and holds ~27 % at 128 B of lag.
- The **content is present and decaying**: the payload decodes 4.6× over
  null at 8–12 B, down to 1.5× at 128 B. What decays is the payload's
  *separability*, not its presence.
- The readout is **not key-blind** (a global readout already recovers the
  payload at 4.6× null), so a per-key readout is **not** the missing piece.

**And this changes the priority order.** The old framing was "the write
path works, the read path does not, so try the hierarchical feature-state
wiring." The measurement says something different: the state holds an
addressable, persistent address and a decaying-but-real content, and the
address survives *where the content decays* — which is exactly what an
**order-losing weighted average** does, because the most recent key
dominates the state and the older payload is averaged into a mixture.
**So the lever is the write's SELECTIVITY (the P3 gate), not the read's
wiring (M1).** The hierarchical wiring stays on the table, but it is no
longer the next month, and the per-layer FSP variant (16× the head's
flops) is now clearly the wrong next arm.

**AND THE P3 GATE IS NOT BLOCKED BY THE REASON WE GAVE**
(`docs/design/P3-GATE-ANALYSIS.md`, 2026-10-22, 632 lines). The plan and
the code both said the ring credit, the tail and the decaytrace "all assume
a per-dim `(dim,)` decay vector". The *shape* part is true and trivial; the
*substance* — "the ring's stored value and the credit's decay no longer
agree" — **does not survive a numerical check**: the ring's per-step
`w *= a` **is** the exact survival product for a time-varying, per-sample
retention (measured max rel dev **0.000e+00**).

The **real** blocker is the **truncation bound**: the ring drops the credit
beyond lag K because `σ_d^K ≤ 1e-4` for every dim, and a two-sided
per-sample gate can push the effective retention `a` for *any* dim
arbitrarily close to 1, so the bound evaporates (at `a = 0.99`, K = 14, the
ring silently drops **87 %** of the credit mass for that (stream, dim)).
The hot set, which protects the hot dims, is computed from the **ungated**
`σ_d` — per-dim and input-independent — which is exactly the invariant a
two-sided gate destroys.

**And the repo already has an exact credit path with no such bound:** the
**dense trace** (`trace_mode="dense"`), which is bit-identical to
ring+tail (`0.00e+00`) and has **no `ring_k`, no hot set, no mask, no
rebase and no truncation** — so it is exact for *any* per-sample `a`, with
a one-line broadcast change. The gate gradient needs **no change** at all
(`gate_trace_update` already handles `(B, dim)` and is pinned against
autograd and finite differences).

**So the next step is concrete and cheap** (the design's §5.3): ~12 lines
in `gpu_port.py`, ~35 in `numpy_port.py` (the reference has no dense path),
~15 to flip `check_gated_recurrence.py`'s refusal assertion into a run
assertion, ~5 for a `--gated` flag, ~5 for the probe — and, above all, the
gate that does **not** exist today: `tests/check_gated_credit.py` pinning
the gated step's **full** gradient (`dembed`, `ddecay`, `dgate`) against
torch autograd. `check_gate_grad.py` pins the *recurrence*'s gradient and
never touches the *credit* — and the arm's whole reason for refusing is
the risk of a plausible-but-wrong credit. **That gate is the deliverable.**
Memory is the dense bill (`B·256·dim·4` per layer) — the **same** bill the
P4 ring already pays, since `ring_k` saturates at 256 under the P4 init.

---

## 7. Why the last 14 plans did not converge, and what changes

> **AMENDED 2026-10-22 — the deeper answer.** The plan's original answer
> was *proxies*: the programme optimised stand-ins for the goal instead of
> the goal, so 14 plans produced 13 measurable verdicts and 0 end-to-end
> wins. **Tonight added a second, harder answer: the INSTRUMENT was wrong
> more often than the hypothesis was.** Four times in one night a
> measurement's meaning was set by a control that was absent —
> a missing copy control (§17), a missing nuisance control (§19), a
> missing criterion wiring (§20), and a **split that let a readout
> memorise the board** (§22) — and twice that produced a *wrong
> conclusion that had to be corrected in the append-only log* (§16, §22).
>
> **So the programme has two failure modes, and they compound:** a proxy
> goal lets you succeed at the wrong thing, and an un-controlled
> measurement lets you believe it. **The governance change in §9 is the
> response to the second.**

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

> **REWRITTEN 2026-10-22.** The old criterion had three problems, all of
> them now fixed: it was **conditioned on CL-3**, which §22–§23 superseded;
> it used **"chance 2⁻³²"**, which is the full-byte figure and is wrong
> for the matched 16-symbol alphabet (the chance is **16⁻⁴ = 1.53e-05**);
> and it made the programme's closure depend on **the T-axis speed
> question**, which is a different claim from "this architecture does not
> do continual learning". A kill criterion that can be triggered by an
> unrelated result is not a kill criterion.

**The programme is closed if, after CL-6a, CL-6b and CL-6c:**
- the trained model does not reach **≥ 0.50 exact recall at d=32** (chance
  **16⁻⁴ = 1.53e-05**) on the fact task, **and**
- the **keyed store (M2)** arm also fails to beat **0.90** of the
  pre-stream recall while keeping **ΔCE ≤ 0.02 nats/B** against the
  text-only baseline, **and**
- the **text-only counterfactual** shows the regression is not recoverable
  by any training-signal change — i.e. the NO REGRESSION failure survives
  the decoupled head.

**AND the programme is closed EARLIER, by a different route, if the
training-signal question comes out negative:**

- **CL-6a's H1 shows the memory loss does not fall over 500 steps with the
  head's table removed, AND CL-6b's explicit copy circuit also fails to
  make the state carry the payload.** Then the state **cannot** be made to
  carry the payload under any objective this project has tried, the
  content **must** live outside the recurrent state, and the only
  remaining route is a **store that is neither a copy nor a table** — and
  if CL-6c's keyed store has already failed, there is nowhere left to put
  it that is not the weights, and **the honest conclusion is that this
  architecture, at this scale, with these objectives, does not do
  continual learning.**

In either case the programme switches to (a) packaging the findings for
upstream and the community, and (b) scale and data on the upstream
configuration, which is where the evidence says the remaining headroom is.

**AND THE T-AXIS IS NOT A KILL CRITERION.** It is a speed question about
a different axis of the design. It may be reported alongside, and it may
inform what to build, but **it must not be able to close a
continual-learning programme, and it must not be able to keep one open.**

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
  recurring). **A gate that has never failed is not known to test
  anything** — every new gate is run against the pre-change code and the
  failure is recorded.

> ### THE CONTROLS RULE (added 2026-10-22 — the night's main product)
>
> **A measurement's meaning is set by its controls, and a control that is
> ABSENT silently becomes the interpretation.** Tonight produced four wrong
> conclusions in a row for exactly this reason, and twice had to be
> corrected in the append-only log. **So:**
>
> - **Every recall cell carries a MATCHED NULL** (permuted labels, the
>   same fixed readout) and a **NUISANCE CONTROL** (the target's key AND
>   payload both replaced) and a **COPY CONTROL** (the payload in the
>   input replaced, the key kept). A cell without them is **UNMEASURED**,
>   never PASS.
> - **Every FITTED readout is split over the things it could MEMORISE** —
>   over the board facts, not over the rows. A row-level split on a finite
>   board is **not** safe, and it is how a table came to be mistaken for a
>   mechanism.
> - **A REFIT PROBE IS NOT A RETENTION MEASURE.** Retention is measured
>   with the model's own parameters; a refitted probe measures copyability.
> - **The instrument's defaults must match the question.** A default that
>   silently does not (the `--facts` 64 vs 4, the ACQUIRE board, the
>   inert `nvrtc` branch) is a defect, and it is gated.
> - **The verdict names the readout.** "ACQUIRE = 0.89" means nothing
>   without saying whether the 0.89 came from a trained parameter or a
>   probe fitted for the occasion.

- **The programme's record is append-only, INCLUDING its own errors.** A
  correction is a new section that supersedes the old one, and it must say
  **which** claim it supersedes — §16 withdrew §14, and §22 withdrew part
  of §20 and §21. **A reader who stops at the last section would be
  misled by the earlier ones, so every superseding section carries a
  pointer back.**
