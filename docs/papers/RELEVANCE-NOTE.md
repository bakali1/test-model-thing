# RELEVANCE NOTE — "FlashLM: CPU-Native Language Models Trained From Scratch on Free-Tier Hardware"

**Paper:** Cheng Chang, *FlashLM: CPU-Native Language Models Trained From Scratch on
Free-Tier Hardware*, May 2026. 8 pages. Self-reported preprint, no venue, MIT,
`github.com/changcheng967/FlashLM`.
**Read from:** `/home/baku/Downloads/output.pdf` (text extraction kept beside this
note as `flashlm.paper.txt`, per the convention in `docs/ideas/papers/`).
**Companion:** `docs/papers/PAPER-NOTE.md` is a full summary/transcription of the same
paper, written in parallel by another pass. **This note is the project-relevance
half**: the audit of the engine, the cost table, the experiment and the verdict.
A runbook for the same arm already exists in parallel — `docs/runbooks/fsp.sh`; this
note agrees with its recipe and **corrects its `pos_weight` calibration** (§2.2 D4)
and adds the blocker it does not mention (**D1: the ruler cannot read an FSP
checkpoint**).
**Date of this note:** 2026-10-20.
**What I did:** read the paper (full text + figures none), read `docs/INDEX.md`,
`docs/plans/PLAN-2026-10-11-T-AXIS.md` (§§0–7) and `RESULTS.md` (the 2026-10-12
memory ruler, the 2026-10-13 T-axis A/B, the 2026-10-13 P4 usage arms), audited the
engine, and ran **CPU-only** micro-experiments (20–40 steps at dim 32) to check the
claims I make below.
**What I did NOT do:** modify engine code, run a GPU job, touch the network.

**Ledger status (read `docs/ideas/papers/INDEX.md` first):** that ledger's rule is
"if an item is not in this table with a `VERIFIED` status, do not cite it". This
paper is **not** in the ledger. My recommendation is a new ledger row with status
**VERIFIED (document) / CLAIMS UNREPLICATED** and four `CLAIM-NOT-SUPPORTED`
annotations (§4.3). Until then, nothing in this note may be cited as evidence for a
TMT decision; it may only be cited as *a hypothesis to be tested*.

---

## 0. The paper in one paragraph, with its own numbers

The claim: **the training objective matters more than the architecture**, and the
lever is an auxiliary loss.

> "Across 30+ experiments on the TinyStories dataset, we find that the training
> objective matters more than the architecture: adding Future Sentence Prediction
> (FSP) as an auxiliary loss improved perplexity by 2.5× (25.08 to 10.24) with only
> 1.7% parameter overhead, while 21 architecture-only changes failed to match this
> gain." (p. 1, Abstract)

FSP, exactly as defined (p. 2–3, §3.1):

> "At every r-th position (we use r = 16), the model predicts which tokens will appear
> in the next τ positions (we use τ = 64)" (p. 3)

> `L_FSP = 1/|P| Σ_{p∈P} BCE( f(h_p), BoW(y_{p+1:p+τ}) )` (p. 2, Eq. 1)

> "where P = {0, r, 2r, . . .} is the set of prediction positions, h_p is the hidden
> state at position p, f is a linear projection into vocabulary space, and BoW
> creates a binary multi-hot vector indicating which tokens appear in the future
> window. Binary cross-entropy uses a positive weight of 50 to handle the extreme
> class imbalance." (p. 2)

> `L = L_CE + α · L_FSP,  α = 0.1` (p. 3, Eq. 2)

> "The FSP head consists of one d × d linear projection (65K parameters for d = 256),
> sharing the output embedding matrix. This is 1.7% of the total 3.74M parameter
> model." (p. 3)

The ablation that carries the claim (p. 6, §6.2): "v10 base (Attention, no FSP)"
→ PPL **25.08**; "v10 FSP" → PPL **10.24**; identical 4-layer transformer, differing
only in the auxiliary loss.

The architecture arm (p. 3, §3.2.2) is **CumMix**: compress each token to a code
dimension k (`c_t = ReLU(W_c · Norm(x_t))`), **cumulative-sum** it, mix in code space
(`m_t = W_m · Norm(g_t)`), expand back with a residual. "Each position sees the sum
of all previous compressed representations, creating a running summary of the
sequence. The compression bottleneck (d → k) forces the model to learn what to
aggregate." (p. 3)

The negative result that matters most to us (p. 7, §7.1):

> "21 architecture-only experiments—spanning attention, linear attention, recurrence,
> convolution, delta rules, and their combinations—never achieved more than marginal
> improvements over the baseline."

And the capacity caveat (p. 7, §7.2): "Below ∼10M parameters, models lack the
capacity for this state, regardless of architecture or training objective."

**The evaluation, stated plainly, because it bounds everything below.** Perplexity
on held-out TinyStories V2 with a **4,096-token BPE vocabulary**, sequences of
length 256, batch 4 × grad-accum 8, lr 5e-4, 200-step warmup, wd 0.1, dropout 0.1,
clip 1.0 (p. 5, §5.1–5.3); PPL is "computed on 20 random batches from the held-out
validation split" and coherence is judged on "three fixed prompts" with nucleus
sampling (p = 0.9, temperature = 0.8) into a subjective "Coherent? Partial" column
(p. 5–6). **Checked fact: the paper contains zero occurrences of `recall`,
`retrieval`, `associative`, `induction`, `needle`, `copy`, or `horizon`.** The only
occurrence of "long-range" is one sentence in §7.1 ("no global planning incentive").
There is **no long-range capability benchmark of any kind** in the paper.

---

## 1. Does the mechanism bear on the WRITE/READ path? Five candidates, judged.

### (i) Make the state hold features instead of raw bytes — **the paper is a paper AGAINST this**

The closest thing the paper has to a write-path mechanism is CumMix's compression
bottleneck: "The compression bottleneck (d → k) forces the model to learn what to
aggregate" (p. 3) — i.e. write a *compressed, mixed* representation into the state
instead of the raw token. That is our candidate (i), restated.

But the paper's own framing is explicitly a rejection of recurrent state
compression (p. 2, §2.3):

> "RWKV [Peng et al., 2023] and Mamba [Gu and Dao, 2023] replace attention entirely by
> recurrence. Gated DeltaNet [Yang et al., 2025] combines the delta rule with gating
> for efficient sequence mixing. **Our CumMix mechanism takes a different approach:
> rather than recurrent state compression, we use cumulative sum over compressed token
> representations**, leveraging the vectorized cumsum operation that is highly
> optimized in Intel MKL."

And the quality of that architecture is **NOT STATED**: Table 2 (p. 6) has rows for
v5, v4-Large, v7.4, v10-FSP, v5.2, v4, v10-base — **there is no v11 / CumMix row**.
CumMix is only ever costed: "CumMix costs 136 µs per layer versus 2,062 µs for
attention—a 15× reduction" (p. 3) and "cumsum over T=256 | 11 | Nearly free"
(p. 6, Table 3). So the paper's own token-mixing mechanism has **no reported
perplexity**. Anything we might want to quote from this paper about "the state should
hold features" would be quoting an *unreported* number.

**Verdict: does not bear.** The paper gives a *cost* argument for abandoning
recurrent state compression and gives no *quality* argument either way. It cannot
support or refute the CUDA fork's hierarchical wiring (state fed by the previous
layer's residual output), which is our candidate (i).

### (ii) Input-dependent / gated retention factor (the write side) — **the paper is a weak negative prior, and no more**

Our candidate (ii) is P3 (`a = sigmoid(decay + gate·x_in)`), currently refused because
its gradient is documented wrong by 4.59–22.4×. The paper's only data point on
gating is one row of Table 2 (p. 6):

> "v7.4 | Gated DeltaNet + SWA | 6.6M | 2h | 2.33 | 2,000 | No"

— the best small-scale PPL in the paper (2.33) and, in the paper's own coherence
column, **not coherent**; §7.2 adds "Our best-perplexity model at small scale, v7.4
(PPL 2.33), generates repetitive text." Gated DeltaNet is *cited* (§2.3, §8) and
*run once*, in a 2-hour CPU budget, at a different tokenization, with **no recall
benchmark**. And §7.1 lists "delta rules" among the 21 architecture experiments that
"never achieved more than marginal improvements over the baseline."

**Verdict: does not bear, mildly against.** The paper is the first external voice
saying "the write side is not where the win is" — but it is one PPL number in an
undertrained regime with no memory metric. It says nothing about whether gating fixes
*addressing*. Note also that v7.4 is a *delta rule + SWA* hybrid, not our
elementwise-gated EMA; the paper is not evidence about our P3 arm either way.

### (iii) Make the state a nonlinear function of the history so it is not order-losing — **the paper is evidence AGAINST, by construction**

This is the sharpest thing the paper says, and it says it by *being* the thing. CumMix's
state is a **cumulative sum** — literally an order-losing bag of the past
("Each position sees the sum of all previous compressed representations", p. 3) — and
the paper presents that as the winning design. Its read path is linear in that sum
(`m_t = W_m · Norm(g_t)`).

Our `RESULTS.md` (2026-10-13) names exactly this property as the reason the model
"cannot answer 'what did I see 128 bytes ago'": "a state that is a linear,
order-losing EMA of raw byte embeddings can hold a signal's *magnitude* out to 128 B,
but a linear readout of an order-losing sum cannot answer 'what did I see 128 bytes
ago'". The paper's architecture is that exact object, at 136 µs/layer, and it is
presented as the answer.

**Verdict: the paper is not evidence for (iii); it is (unmeasured) evidence against
the *importance* of order.** To be fair to it: the paper never *tests* a recall task,
so it cannot be cited as "order-losingness does not matter". It can only be cited as
"an order-losing architecture is compatible with a 15× latency win and with
competitive perplexity" — and the second half of that is **NOT STATED** (no CumMix
PPL row).

### (iv) Multi-token prediction heads — **the paper cites MTP, never runs it**

p. 2, §2.2: "Multi-token prediction [Gloeckle et al., 2024] trains models to predict
multiple future tokens simultaneously, improving sample efficiency and downstream
performance. Mahajan et al. [2025] extended this idea to sentence-level predictions.
**Our FSP loss adapts the bag-of-words prediction framework as a lightweight auxiliary
objective.**" MTP itself is never run; Gloeckle et al. (arXiv 2404.19737) and Mahajan
et al. (arXiv 2510.14751) are neither in `docs/ideas/papers/INDEX.md` nor shelved in
`docs/ideas/papers/`, so per the ledger rule we may not cite them yet.

What the paper *does* contribute here is the closest thing to an evaluated
multi-token-ish auxiliary: FSP itself, with a measured 2.5× PPL gain. So (iv) and
"the FSP loss" are the same lever in this paper.

**Verdict: bears, but only through FSP** — and FSP is a *bag* target, i.e. it is
**order-insensitive by construction** (BoW of the future window). A bag target cannot
teach *addressing* ("what was at offset −128"), only *content class* ("the next
window will contain letters e, t, a…"). Our failure has two halves — the state holds
a signal it cannot *use*, and the ruler's copy tasks need exact content at an exact
offset — and FSP can only speak to the first half.

### (v) Anything about a window / T axis — **NOT STATED**

The paper has no notion of an optimizer window, a credit horizon, or a
tokens-per-update axis. The only sequence-length facts: "Sequences of length 256 are
sampled randomly" (p. 5) and the FSP lookahead τ = 64 fits inside that 256-token
sequence. Its training loop is "batch size 4 with gradient accumulation 8 (effective
batch 32)" (p. 5) — a *token* batch, not a window.

The one indirect relevance: **FSP is an inherently windowed objective** (it needs the
next 64 tokens at every 16th position). In our engine the same objective needs a
64-byte lookahead in the data pipeline, and it can be had either at `--seqlen 1` (a
pipeline change: the target must be built from bytes ahead of the current position)
or at `--seqlen ≥ 64` (which our own A/B measured as a *quality* cost at this scale:
"T=1 wins, and by more than noise … the effect is the update count, not the credit",
`RESULTS.md` 2026-10-13). Note the paper's regime is the opposite of ours: its FSP
arm also paid for its lookahead in optimizer updates (batch 4 × accum 8 over
256-token sequences) and still won — that is the only transferable data point here,
and it is weak.

**Verdict: NOT STATED, with one weak indirect data point.**

---

## 2. Cost: what the paper's ideas cost in THIS codebase

### 2.1 The paper's ideas, ranked by our implementation cost

| paper idea | where it would land in our code | our cost | evidence in the paper |
|---|---|---|---|
| **FSP aux loss** | new head on the recurrent state + a BCE term + a lookahead target | **low, and ALREADY 90% written** (see §2.2) | **the only positive result**: 25.08 → 10.24 |
| TACE (freq-weighted CE + learned temperature) | a 256-vector of token frequencies + 1 scalar in `Trainer`; no state-path change | **lowest** (≈10 lines) | **NOT STATED** — no v11 row in Table 2 |
| PowerNorm (learnable Lp) | `gpu_port._layer_norm` + `numpy_port.Layer.forward`; +2 params/layer | low–medium (touches the read path; parity gate) | **NOT STATED** — no v11 row |
| CumStepPos (random-walk positions) | new positional signal into the state | medium (touches the write) | **NOT STATED** — no v11 row |
| HarmonicFFN | replaces the FFN | medium | **NOT STATED** — no v11 row |
| DualMomAdam | replaces the flat AdamW in both ports | medium–high (two momentum buffers, per-coordinate wd interplay) | **NOT STATED** — no v11 row |
| CumMix / hierarchical wiring (candidate (i)) | the state's write term `enc` → features | **HIGH** — see below | **NOT STATED** for quality |
| Gating (candidate (ii)) | already implemented (P3), refused | high (known-wrong gradient) | weak negative (v7.4) |

**Why (i) is expensive here, concretely.** Our state update is
`state = states * a + enc` (`gpu_port.py:242`; `numpy_port.py:126`), where
`enc = self.embed[c_arr]` is the raw byte embedding, and the read path is
`_layer_norm(state, g, b)` → `z = h @ W.T` → `y = z*sigmoid(z)` → residual
(`gpu_port.py:243–249`). Making the state carry *features* (the previous layer's
residual output) instead of bytes changes what the **ring credit** is for: today
every byte's credit lands in the embedding only — `self._ring_dest = dembed`
(`gpu_port.py:1223`), "every byte's credit lands here" — and the per-lag survival
weights `w` are applied to `dlds` (the per-layer state gradient). If the state carries
features, the surviving lag carries a *feature vector*, and the gradient that must be
pushed back along it is `dW`/`dg`/`db` of the layer that produced it, not a byte id:
the ring's `(B, K, dim)` weight buffer would have to be applied to per-layer weight
gradients, per layer. That is a rewrite of `_scatter_ring_credit` / `_ring_credit_compiled`
/ the NVRTC kernel / the tail — not a one-line change, and not in this month.

**Verdict on cost: FSP is the cheapest *evidenced* idea, and it is already written.**

### 2.2 The FSP path is ALREADY in the working tree — uncommitted, and it has five defects

`git status` shows `M gpu_port.py`, `M train_gpu.py` (232 insertions) with a complete
FSP implementation: `--fsp-alpha`, `--fsp-r` (16), `--fsp-tau` (64),
`--fsp-pos-weight` (50), `--fsp-layer` (-1 = last); the head is `fsp.W` (dim → 256)
+ `fsp.b`, created in `GpuModel._build` **only when `fsp_alpha > 0`** (so the default
theta layout is untouched, the same rule the P3 gate uses); the loss is
BCE-with-logits on a binary multi-hot; the head reads the **last layer's state**
through that layer's own LayerNorm; and `Trainer._fsp_args` builds the target from
`mats[t0+1 : t0+1+tau]` with a scatter into a (B, 256) buffer.

**I audited it (CPU only, dim 32 / layers 2, 20–40 steps). Findings:**

**D1 — the arm is UNMEASURABLE with our own instrument. This is the blocker.**
`tools/ruler.py` runs on `numpy_port` (`tools/membench/model_backend.load_frozen` →
`numpy_port.Model.load`), and `numpy_port.py` has **not** been given the FSP keys
(re-verified against the working tree at 11:45 today: `grep -n fsp numpy_port.py` →
no hits; the numpy_port changes in the tree are for the P3 gate, not for FSP).
The FSP head lives in `theta`, so the checkpoint's `m`/`v` are longer than a
numpy_port model, and the length check at `numpy_port.py` (the `raise ValueError`
inside `load`) fires. Reproduced verbatim:

```
$ TMT_RULER_CKPT=/tmp/fsptest/fsp.npz .venv/bin/python tools/ruler.py
...
  File ".../tools/membench/model_backend.py", line 65, in load_frozen
    m.load(ckpt)
  File ".../numpy_port.py", line 555, in load
    raise ValueError(...)
ValueError: checkpoint /tmp/fsptest/fsp.npz holds 27,361 parameters but this model
was built with 18,913 (dim=32, layers=2); pass matching --dim/--layers
```

So the arm can be **trained** and cannot be **read**. Until `numpy_port.Model` learns
`fsp.W`/`fsp.b` (add them to `init`/`self.shapes` when the checkpoint carries them —
the same "only in the armed arm" rule) and `model_backend.load_frozen` builds the
model accordingly, every ruler cell — retention curve, selective copy, associative
retrieval, long-range language CE, held-out CE — is unavailable for this arm.
*This is the first thing to fix, and it is a two-file, ~10-line change.*

**D2 — the FSP loss reaches exactly one of sixteen layers' memories.** The gradient
path is correct as written (the state gradient is added to `lgrads[fsp_layer]`, which
is what the ring credits to the embedding and what the decaytrace turns into the
decay logits' gradient), but it is added **after** the backward loop, so it does not
enter `dx`: layers 0–14 receive **no** FSP gradient at all. Measured (20 steps,
dim 32, layers 2, FSP on vs off):

| key | max abs difference (FSP vs no-FSP) |
|---|---|
| `layer0.W` | **0.000e+00** (bit-identical) |
| `layer0.decay` | **0.000e+00** (bit-identical) |
| `embed` | 2.384e-07 (the FSP gradient does reach the embedding credit) |
| `layer1.decay` (the FSP layer) | 7.451e-08 (it reaches that layer's decay logits) |
| `layer1.g` (the FSP layer's LN gain) | 4.292e-06 |

So FSP as implemented trains **the last layer's memory and nobody else's**. If the
hypothesis is "give the long-lived channels a job", one of sixteen channels getting
the job is a thin test. A `--fsp-layer` sweep (last vs every 4th) or a per-layer FSP
head would be the honest version; **NOT STATED** in the paper, and the paper's model
is a single-stack transformer where "the state" is one thing.

**D3 — the FSP loss is not a pure auxiliary: it moves the last layer's LayerNorm
gain, and therefore the local read-out.** `lg[fsp_layer] += dg_fsp` and
`lb[fsp_layer] += db_fsp` put the FSP gradient into the *shared* LN parameters of
the layer the head reads. Measured: `layer1.g` moves by 4.3e-6 and `Wd` moves by
1.5e-8 as a result. So an FSP arm's held-out CE can move for a reason that has
nothing to do with memory (the LN gain of the last layer is a local-model knob).
Cheap fix: give the FSP head its **own** normalisation parameters (`fsp.g` dim,
`fsp.b0` 1) instead of borrowing the layer's; that removes the confound and adds
513 parameters. **NOT STATED** in the paper.

**D4 — `--fsp-pos-weight 50` is the paper's number and it is mis-calibrated for
bytes.** The paper's 50 is tuned to a BoW target over a 4,096-token vocabulary: at
τ = 64 tokens the target is at most 64/4096 = 1.56 % dense, so positives outnumber
negatives ~1:63 and a 50× positive weight makes the positive class carry ≈44 % of
the loss.

**Measured on our corpus** (`data/wikipedia_clean/wikistories/wiki_tinystories_001.txt`,
two independent estimators, 250k positions, τ = 64): the byte-bag holds **21.9
distinct byte values of 256 on average** (a 1000-window random sample: mean 21.87,
min 15, max 30) — i.e. the target is **8.55 % dense**, not the 22 % asserted in the
parallel runbook `docs/runbooks/fsp.sh` (which claims "≈57 distinct values = 22 %
positives"; that is wrong by ~2.6×, and it is the difference between a calibrated
knob and a mis-set one). Consequences at d = 0.0855:

| calibration | value | positive mass |
|---|---:|---:|
| the paper's literal 50 | 50 | **82 %** — strongly all-positive; the head learns "everything is coming" |
| exactly balanced, w = (1−d)/d | **10.7** | 50 % |
| matching the paper's own effective balance (44.25 %) | **8.5** | 44 % |

So use **`--fsp-pos-weight 8.5`** (paper-matched) or **10.7** (balanced); **not 50**,
and not the runbook's 3.5 (which comes from the wrong density — at d = 0.0855 it
under-weights positives 2.4× relative to balance). The knob is monotone in the
head's operating point, so this only sets where the head sits, not what it learns —
but at 50 the FSP term is roughly 6× over-weighted relative to the paper's own
balance and it dominates the loss early.

**D5 — the FSP loss is never logged.** `GpuModel.last_fsp` is computed
(`gpu_port.py:1520`) and nothing prints it: `grep -n "last_fsp" train_gpu.py` → no
hits. Without it, an arm cannot distinguish "FSP learned the target" from "FSP did
nothing", which is the first thing one wants to know. The KD term sets the precedent:
"the KD term has to be readable directly" (`gpu_port.py`, the KD block).

**What is correct** (so the implementation is not thrown away): the target
construction is right (the row-major `(τ, B)` → batch index `i % B` mapping is
correct; repeated bytes are clamped to 1; the FSP positions are `{0, r, 2r, …}` in
*byte* position, which is what the paper's P is); the head is created only when
`fsp_alpha > 0`; and — checked — **the default path is untouched**: 20 steps at the
same seed with the committed engine vs the working-tree engine give **23/23
checkpoint keys bit-identical** (`max|diff| = 0.000e+00` on `theta` components, `m`,
`v`, `state.*`, `ring_k`). That matters right now: the P4 long run in flight on Colab
embeds the engine, and this change cannot have perturbed it.

### 2.3 What the FSP path *should* look like, at file/function level

Nothing below has been implemented — this is the specification.

| file · function | change | new buffer |
|---|---|---|
| `numpy_port.py` · `Model.__init__` / `_build` (L144–215) | accept `fsp_alpha`; add `init["fsp.W"] = (dim,256)`, `init["fsp.b"] = (256,)` **only when armed**, mirroring the `gated` precedent | theta +131,328 floats; `m`/`v` +131,328 each |
| `numpy_port.py` · `Model.load` (L526) | nothing to add if the keys are in `self.shapes` — the length check then matches | — |
| `numpy_port.py` · `Model.step_batch` (L324) | the FSP read-out + BCE, for parity with `gpu_port` (`tests/check_parity.py` gates the two engines bit-exactly) | — |
| `gpu_port.py` · `GpuModel._build` (L618) | already done (D2/D3 caveats) | as above |
| `gpu_port.py` · `GpuModel.step_batch` (L1004) | already done | — |
| `train_gpu.py` · `Trainer._fsp_args` (L373) | already done; needs the **log** of the FSP term | per-step (B, 256) float32 multi-hot (1 MB at B=1024) + (B, 64) int64 gather (512 KB) |
| `train_gpu.py` · `Trainer.train` (L340) | log `m.last_fsp` and `m.fsp_positions` every `--log-every` | — |
| `train_gpu.py` · `main` (L586) | `--fsp-pos-weight` default → 8.0 (D4) | — |
| `tools/membench/model_backend.py` · `load_frozen` (L56) | build the numpy model with the FSP head when the checkpoint carries it | — |
| `tools/membench/eval_api.py` | **new cell**: `fsp_bag_accuracy(model, tau=64)` — the frozen model's own FSP head scored against the future bag, reported as F1 against the two baselines I measured in §B (global prior 0.7505, local 4-gram context 0.7690) | — |
| `tests/check_fsp.py` | **new gate** (four claims, below) | — |

**The four gates the arm must pass before it touches a GPU** (the repo's convention:
every feature has a gate; see `docs/plans/PLAN-2026-10-11-T-AXIS.md` §2 P0):

* **G1 (default path untouched):** `--fsp-alpha 0` over ≥120 steps is bit-identical
  to the current engine — *already verified on CPU over 20 steps (23/23 keys, 0.000e+00);
  the gate should extend this to 120 steps and to CUDA.*
* **G2 (parity):** `numpy_port` and `gpu_port` agree bit-exactly **with** the FSP
  arm on (`tests/check_parity.py`) — *currently impossible: numpy_port has no FSP.*
* **G3 (target correctness):** the multi-hot for position p equals the set of byte
  values in bytes[p+1 : p+1+τ], against a numpy reference, for T=1 and T>1.
* **G4 (the gradient reaches the memory):** assert that `dL/d(decay)` of the FSP
  layer is non-zero when the FSP term is on and exactly zero when it is off, and
  that it is *zero* for the other layers (this pins D2 as a measured property rather
  than an accident).

**Parameter overhead, measured against the paper's:** `fsp.W` (512×256) + `fsp.b`
(256) = **131,328 parameters = 2.93 %** of 4,481,793, against the paper's "1.7 %
parameter overhead" for its 65K-parameter head on a 3.74M model. Note the paper's own
description is internally inconsistent: Eq. (1) says `f` is "a linear projection into
vocabulary space" (which for a 4,096-token vocab and d = 256 would be ~1.05 M
parameters, not 65 K), while §3.1 says "one d × d linear projection (65K parameters
for d = 256), sharing the output embedding matrix" (65K matches the stated 1.7 %).
Which of the two the authors implemented is **NOT STATED**; for us the direct
d → 256 head is the cheaper reading and is what the working tree does.

---

## 3. The decisive experiment

### 3.1 What is being tested

> **H1 (the paper's claim, at byte level):** adding an auxiliary long-horizon
> objective to the training signal changes what the *state* is for, and therefore
> changes the memory measurements — retention up, and the copy/retrieval cells above
> their vetoes.

> **H0 (the alternative):** the auxiliary loss is a local-frequency prior wearing a
> long-horizon costume; it changes CE a little and changes nothing the ruler can see.

The paper cannot distinguish H1 from H0 (it has no memory metric). **We can**, and the
ruler already exists. That is the whole value of running this arm.

### 3.2 The arms — one new arm, two controls already in flight

Everything is matched to the **P4 long run in flight** (`docs/runbooks/p4-long.sh`):
from scratch, seed 0, B = 1024, `--seqlen 1`, `--ring-k 18`, `TMT_TAIL=0`, the same
4-file corpus glob, the same 1-epoch byte budget (161,333 steps × 1024 B = 165 MB),
the same `--warmup 20 --no-tagged --device cuda`.

| arm | init | `--wd-decay` | FSP | status |
|---|---|---|---|---|
| **`A_p4_wd0_fsp`** | P4 (`--half-min 2 --half-max 512`) | 0.0 | `--fsp-alpha 0.1 --fsp-r 16 --fsp-tau 64 --fsp-pos-weight 8.5` | **the new arm** (one epoch); the same arm as `docs/runbooks/fsp.sh` |
| `A_p4_wd0` | P4 | 0.0 | off | **the control — already in flight** (identical recipe, identical seed, identical bytes) |
| `B_default_wd001` | default | 0.01 | off | **the second control — already in flight** (the memory-free, CE-cheap arm) |
| `C_p4_wd001` | P4 | 0.01 | off | in flight; tells us the erosion arm's CE, for reference |

Because `A_p4_wd0` is *already running with the same seed and the same byte budget*,
**the FSP arm costs one extra arm and no extra control** — the control is not
rebuilt, it is reused. That is the cheapest possible A/B of this idea in this project.

**Do not add a separate "FSP with shuffled targets" arm unless the primary readout
is positive.** If the primary readout is ambiguous, the cheapest disambiguator is a
*shuffled-target* arm (same loss magnitude, targets permuted across streams): it
separates "any extra loss term regularises" from "the long-range target specifically
matters". Cost: one more arm. (This is the repo's standing practice — "the control
arm that was missing", `RESULTS.md` 2026-10-04.)

### 3.3 The byte budget and the commands

Byte budget: **161,333 steps × B=1024 × T=1 = 165,199,232 B ≈ 165 MB = 1 epoch over
the 4-file corpus** — the same budget as the three in-flight arms, so all four CEs
are comparable.

```bash
# --- the new arm (D), on the Colab T4, same recipe as docs/runbooks/p4-long.sh ---
cd /content
export TMT_TAIL=0
export OMP_NUM_THREADS=2
mkdir -p runs/taxis

python3 train_gpu.py runs/taxis/A_p4_wd0_fsp.npz train \
  --dataset "data/wikipedia_clean/wikistories/wiki_tinystories_*.txt" \
  --batch 1024 --seqlen 1 --steps 161333 --warmup 20 --no-tagged \
  --ring-k 18 --half-min 2 --half-max 512 --wd-decay 0.0 \
  --fsp-alpha 0.1 --fsp-r 16 --fsp-tau 64 --fsp-pos-weight 8.5 \
  --device cuda --seed 0 > runs/taxis/A_p4_wd0_fsp.log 2>&1

# --- the ruler, on all four arms (TMT_RULER_CKPT selects the checkpoint) ---
export TMT_RULER_RAW=/content/data/eval/tinystories_valid.txt
for name in A_p4_wd0_fsp A_p4_wd0 B_default_wd001 C_p4_wd001; do
  TMT_RULER_CKPT=/content/runs/taxis/$name.npz python3 tools/ruler.py \
      > runs/taxis/ruler_$name.txt 2>&1
done
```

**Two operational notes, both verified today.**

1. **`TMT_RULER_RAW` is dead in the current ruler.** `tools/ruler.py:36` reads it
   (`RAW = os.environ.get("TMT_RULER_RAW", ...)`) and **never uses it**: the
   held-out CE cell evaluates `sorted(glob.glob("data/eval/**/*", recursive=True))`
   at 3 slices × 20,000 B, and the cross-domain cell globs `data/*.txt` excluding
   `sample.txt`. Setting the variable changes nothing. Keep setting it (the recipe
   does) but do not believe it is doing anything.
2. **The D arm cannot be read by the ruler until D1 is fixed** (§2.2). The training
   command above is fine; the ruler command will raise `ValueError: checkpoint …
   holds N parameters but this model was built with M` until `numpy_port` learns the
   FSP keys. **Fix D1 before launching**, or the arm will be trained and
   unmeasurable — which is exactly how this project has lost time before ("Four
   bugs the ruler exposed, none caught by the gates", `RESULTS.md` 2026-10-12).

**GPU contention:** the three in-flight arms run concurrently on one T4. A fourth arm
adds roughly the same footprint as one of them (ring buffer (B=1024, K=18, 512)
fp32 × 16 layers ≈ 604 MB, plus states and the optimizer's 3 × 4.48 M fp32 ≈ 54 MB,
tail off). Check `nvidia-smi` before adding it; if free memory is under ~2 GB, run D
after A finishes rather than beside it.

### 3.4 The instrument, and exactly what each cell decides

`tools/ruler.py`, unchanged (the same instrument as the incumbent's 2026-10-12
numbers, which is the only reason the numbers are comparable):

| ruler cell | what it measures | what a move means |
|---|---|---|
| §1 retention curve, d = 8/32/128/512, 64 samples | matched-pair Δh ÷ control floor | **the write side**: are the long-lived channels now *carrying* something the model uses, or just present? (control: 5.5 % at d=128 for `A_p4_wd0`) |
| §2 selective copy, d = 8/32/128/512, 128 samples | readout accuracy vs its permuted-label veto | **the read side**: the only cell that can show addressing |
| §3 associative retrieval, same grid | key→payload accuracy vs veto | the read side, harder |
| §4 long-range language CE, L = 512 | nats/byte on a marker task (chance 5.5248) | "planning": the paper's own claim |
| §5 held-out CE | 3 slices × 20,000 B over `data/eval/**` | the local objective; the paper's own metric |
| §5 cross-domain CE | 9 novels × 8,000 B | generalisation |
| **NEW (proposed)** `fsp_bag_accuracy(τ=64)` | the frozen model's own FSP head vs the future bag, F1 | whether the FSP head learned anything beyond the prior (§B) |

**The decision rule, fixed in advance (this is the point of writing it down now):**

* **H1 supported** iff (a) the D arm's held-out CE is **not worse** than `A_p4_wd0`'s
  by more than 0.05 nats (the paper's claim is a *gain*, so a loss is already a
  partial refutation), **and** (b) at least one of: the d=128 retention ratio rises
  above 5.5 %, or a copy cell's readout beats its permuted-label veto (which no
  model in this log has ever done).
* **H0 supported** iff the D arm's held-out CE is ≥ A's, and the ruler cells are
  unchanged within noise. **Then FSP is closed** and the month's priority stays
  exactly where the plan put it (write path / readout).
* **Ambiguous** (CE moves, ruler does not): the paper's objective helped the *local*
  model only, which is consistent with the paper's own finding ("the CE ruler is
  known … to be structurally insensitive to long-range memory" — our ruler's §8
  note) and tells us nothing about memory. Do not promote FSP; do not denounce it.
* **The pre-flight gate (G5, §3.5) can refuse before any GPU time is spent.**

### 3.5 The pre-flight gate that should run first (5 minutes, CPU, no GPU)

Before spending a T4 hour on the D arm, run the §B measurement on our own corpus:
**how much of the 64-byte future bag is predictable without any long-range
information?** I ran it (script in §B):

* the bag is **8.55 % dense** (21.9 distinct byte values of 256; two independent
  estimators agree: 21.87 over 1,000 random 64-byte windows, 21.88 over 250k
  sliding positions — the parallel runbook's "≈57 distinct = 22 %" is wrong by 2.6×);
* a **context-free global prior** already achieves **F1 = 0.7505**;
* conditioning on the **current byte only** gives **F1 = 0.7541** (+0.36 points);
* conditioning on a **4-gram local context** gives **F1 = 0.7690** (+1.85 points over
  the prior).

**Interpretation (the skeptical core of this note):** at byte level, ~75 % of the FSP
target is explained by *which letters are common in English*, and the current byte
alone buys 0.36 points on top of that. The residual that a *long-range* mechanism
could exploit is the ~1.9 F1 points between the local-context predictor and the
prior — and the head reads a state that is dominated by the current byte (the
residual stream passes `enc` straight through: `return x_in + y, …`,
`gpu_port.py:249`). So the FSP head can get most of what is achievable **without any
long-range information at all**, and the gradient that reaches the long-lived state
dimensions is correspondingly small.

That is not a reason to skip the arm. It is a reason to (a) run it as a **falsifiable
test with a pre-written decision rule**, not as a promising direction; (b) fix the
pos-weight calibration (D4) so the loss is not dominated by the prior;
(c) **log** the FSP term (D5) so we can see the head's loss fall — if it falls to
the prior's level and stops, that is the measurement, and it is decisive.

**If the pre-flight says the target is >90 % prior-explainable (it is 97 % of the
locally-achievable F1), the honest recommendation is: run the D arm anyway, but
spend the parallel GPU slot on the write-path arm instead** — the one our own
measurements point at (`RESULTS.md` 2026-10-13: "the remaining deficit is
architectural, not a batching, credit or optimization problem"), and the one this
paper explicitly declines to support (§7.1).

### 3.6 Cost

One arm, one epoch, ~165 MB. At the in-flight arms' rate that is one more T4 slot.
The instrument fixes (D1, D3, D4, D5, G1–G4) are a few hours of CPU work, not GPU.

---

## 4. Where the paper is already contradicted by our measurements, and where it would have explained something we could not explain

### 4.1 Contradicted

* **"Below ∼10M parameters, models lack the capacity for this state, regardless of
  architecture or training objective" (§7.2)** — contradicted as stated. Our
  **4.48 M**-parameter model demonstrably holds a signal **512 bytes out**: the
  trained P4 arm reads **0.91 %** of the control floor at d = 512 and the fresh
  P4 model 1.05 % (`RESULTS.md` 2026-10-13), against 0.0007 % for the incumbent.
  The paper's claim is about a "narrative state" (characters, events, causal
  relations), which is a different object from "a retained magnitude", so this is a
  partial contradiction of the *capacity* claim and no contradiction of the
  coherence claim.
* **"the training objective matters more than the architecture" (§7.1), read as
  "architecture changes do nothing"** — our log is *consistent* with it (nothing we
  did on the architecture side ever bought the CE win), but the same log shows
  architecture/init changes moving the **memory** metric by 15× (the P4 init:
  0.36 % → 5.5 % of the control floor at d = 128). The paper's claim is about
  *perplexity*; it says nothing about memory metrics, and it must not be quoted as
  if it did. On the memory axis, our own measurements say the opposite of the
  paper's ranking.
* **"a 65K-parameter head (1.7 % of total) that adds negligible training overhead"
  (p. 6)** — in our engine the FSP overhead is *not* negligible in the way the
  paper means: it needs a **lookahead** in the data pipeline (the target is the
  *future* 64 bytes), which the paper gets for free from its 256-token sequences
  and which our `--seqlen 1` recipe does not have. The paper reports 2,750 tok/s
  for both v10-base and v10-FSP (p. 6, Table 2) — an exact equality that is
  implausible to us and that we could not reproduce at all; treat it as
  **NOT STATED** in any transferable form.
* **The paper's own architecture has no perplexity.** Table 2 (p. 6) has no
  v11/CumMix row. Any claim of the form "CumMix is better than X" is
  **CLAIM-NOT-SUPPORTED** by this paper.

### 4.2 Where the paper *would* have explained something we could not explain

This is the paper's real value to us, and it is worth more than its experiment.

* **Our P4 arms' CE penalty.** "the P4 arms read 1.8834 / 1.6907 nats/byte against
  the default-init control's 1.4085 at the same byte budget" (`RESULTS.md`
  2026-10-13) — i.e. giving the model long-lived channels made the local objective
  **worse**, and nobody could say why. The paper says why: "This local objective
  gives the model no incentive to maintain representations useful for long-range
  coherence" (p. 2, §3.1.1). Under CE-only, the long channels are dead weight that
  the objective penalises, because nothing in the loss ever asks the state to be
  useful. **This is a mechanism-level explanation of a result we had**, and it
  predicts a testable consequence: an auxiliary long-range term should *shrink* that
  0.47-nat gap, not widen it. That is precisely the primary readout of §3.4.
* **Our long-range language CE being worse than chance.** "Long-range language CE
  10.4767 nats/B against a chance level of 5.5248 … this one is actively
  miscalibrated at that range" (`RESULTS.md` 2026-10-12). Under a local-only
  objective the optimal policy is to *ignore* the marker — a model that has learned
  that the marker never helps will suppress its influence, which is what a CE worse
  than chance looks like. The paper's framing ("no incentive to … maintain
  representations useful for long-range") predicts exactly this sign, and it is a
  reading we had not written down.
* **The one thing the paper does *not* explain** is the read side: "the model now
  holds a long-memory signal and still cannot use it". FSP puts a read-out on the
  state, so it speaks to *holding-and-shaping*; but its target is a **bag**, so it
  cannot teach addressing. Our copy cells need exact content at an exact offset. If
  the D arm moves the copy cells, the paper will have taught us something nobody
  else has; if it moves only the CE, it has not.

### 4.3 Four CLAIM-NOT-SUPPORTED annotations for the ledger (recommended)

1. "CumMix improves quality" — Table 2 has no CumMix/v11 perplexity row.
2. "Auxiliary losses improve long-range capability" — no recall/retrieval/associative
   benchmark exists in the paper (0 occurrences of recall/retrieval/copy/needle).
3. "Gated DeltaNet is the best small-scale architecture" — v7.4 is the best *PPL* in
   Table 2 and is "No" in the paper's own coherence column; no other gated variant
   was run.
4. "FSP costs 1.7 % parameters and negligible overhead" — the head's description is
   internally inconsistent (§3.1's "d × d linear projection (65K parameters)" vs
   Eq. (1)'s "projection into vocabulary space", which for a 4,096-token vocab would
   be ~1.05 M parameters); the 2,750 tok/s equality in Table 2 is unexplained.

---

## 5. Verdict

> **The paper does NOT change the project's next month. It adds ONE arm — a
> CPU-gated, instrument-fixed FSP arm (`A_p4_wd0_fsp`) — to run in parallel with the
> P4 long run now in flight, and it changes nothing else.**

**The ONE experiment to add in parallel with the P4 long run on Colab** (the
fourth arm, same recipe, same seed, same 165 MB budget, one extra epoch):

```
python3 train_gpu.py runs/taxis/A_p4_wd0_fsp.npz train \
  --dataset "data/wikipedia_clean/wikistories/wiki_tinystories_*.txt" \
  --batch 1024 --seqlen 1 --steps 161333 --warmup 20 --no-tagged \
  --ring-k 18 --half-min 2 --half-max 512 --wd-decay 0.0 \
  --fsp-alpha 0.1 --fsp-r 16 --fsp-tau 64 --fsp-pos-weight 8.5 \
  --device cuda --seed 0
```

against the **already-running** `A_p4_wd0` (same bytes, same seed, no FSP) as its
control, and the already-running `B_default_wd001` as the memory-free control;
instrument `tools/ruler.py` with `TMT_RULER_CKPT`, once D1 (numpy_port learns the
FSP keys) is fixed; decision rule fixed in §3.4 **before** the run starts.

**Why it is worth one arm and not more:**

* it is the **cheapest** of the paper's ideas in this codebase and it is **already
  written** (§2.2) — the marginal cost is one GPU arm plus ~10 lines of instrument
  fix;
* it is the **only** external claim that aligns with our own null record ("the
  objective, not the architecture") and it gives a *mechanism* for the one result we
  could not explain (the CE penalty of the long channels, §4.2);
* it is **falsifiable here** in a way it was not falsifiable in the paper — the
  ruler has copy cells with vetoes that no model in this log has ever beaten, so
  "FSP helps" and "FSP is a prior in a long-horizon costume" are separable;
* and it is cheap enough that being wrong is cheap. **The paper's evidence does not
  support promoting FSP to a priority** — its own architecture has no reported
  perplexity, its only positive result is a PPL number in a 2-hour CPU budget with a
  subjective coherence judgement on three prompts, and it contains no long-range
  capability benchmark at all.

**What the paper does *not* change:** the priority order from
`PLAN-2026-10-11-T-AXIS.md` §7.3 — the write path and the readout stay ahead of the
init, the T axis and the ring. If anything the paper is a *reason to be careful*
about spending the month on architecture-side write-path work: "21
architecture-only experiments—spanning attention, linear attention, recurrence,
convolution, delta rules, and their combinations—never achieved more than marginal
improvements" (§7.1) is the closest thing we have to an external prior that says
"the P3 gate and the hierarchical wiring are not where the month goes". But it is a
PPL-only result in an undertrained regime, and we have measured (in a regime the
paper would recognise as its own) that architecture changes *do* move our memory
metric by 15×. So: use the paper as a *hedge* on the write path, not as a veto.

**One caveat to carry into the arm's launch:** if the D arm's primary readout is
"CE improved, ruler unchanged", the correct conclusion is **not** "FSP works". Our own
ruler says why (its §8): "the CE ruler is known (from the n-gram anchor) to be
structurally insensitive to long-range memory, so the CE numbers here tell you about
the ruler, not about memory". A CE win with a flat ruler is the paper's result
reproduced, and the paper's result is not about memory.

---

## Appendix A — quote ledger (page = the PDF's own page)

| # | quote | page |
|---|---|---|
| A1 | "adding Future Sentence Prediction (FSP) as an auxiliary loss improved perplexity by 2.5× (25.08 to 10.24) with only 1.7% parameter overhead, while 21 architecture-only changes failed to match this gain" | 1 |
| A2 | "Every experiment used token-level cross-entropy as the sole training objective—predict the next token, nothing else. This local objective gives the model no incentive to maintain representations useful for long-range coherence." | 2 |
| A3 | "f is a linear projection into vocabulary space, and BoW creates a binary multi-hot vector indicating which tokens appear in the future window. Binary cross-entropy uses a positive weight of 50 to handle the extreme class imbalance." | 2 |
| A4 | "At every r-th position (we use r = 16), the model predicts which tokens will appear in the next τ positions (we use τ = 64)" | 3 |
| A5 | "L = L_CE + α · L_FSP, α = 0.1" | 3 |
| A6 | "The FSP head consists of one d × d linear projection (65K parameters for d = 256), sharing the output embedding matrix. This is 1.7% of the total 3.74M parameter model." | 3 |
| A7 | "Each position sees the sum of all previous compressed representations, creating a running summary of the sequence. The compression bottleneck (d → k) forces the model to learn what to aggregate." | 3 |
| A8 | "Our CumMix mechanism takes a different approach: rather than recurrent state compression, we use cumulative sum over compressed token representations" | 2 |
| A9 | "Multi-token prediction [Gloeckle et al., 2024] … Mahajan et al. [2025] extended this idea to sentence-level predictions. Our FSP loss adapts the bag-of-words prediction framework as a lightweight auxiliary objective." | 2 |
| A10 | "21 architecture-only experiments—spanning attention, linear attention, recurrence, convolution, delta rules, and their combinations—never achieved more than marginal improvements over the baseline." | 7 |
| A11 | "We hypothesize that coherence requires sufficient model capacity to maintain a 'narrative state'… Below ∼10M parameters, models lack the capacity for this state, regardless of architecture or training objective." | 7 |
| A12 | "v10 base \| Attention (no FSP) \| 3.74M \| 2h \| 25.08" / "v10 FSP \| Attention + FSP \| 3.74M \| 2h \| 10.24" | 6 |
| A13 | "v7.4 \| Gated DeltaNet + SWA \| 6.6M \| 2h \| 2.33 \| 2,000 \| No" | 6 |
| A14 | "Sequences of length 256 are sampled randomly" / "batch size 4 with gradient accumulation 8 (effective batch 32), learning rate 5×10⁻⁴ with cosine decay and 200-step warmup, weight decay 0.1, gradient clipping 1.0, dropout 0.1" | 5 |
| A15 | "Validation perplexity is computed on 20 random batches from the held-out validation split. Generation samples use nucleus sampling (p = 0.9, temperature = 0.8) with three fixed prompts." | 5 |

**NOT STATED** (used as absence-of-evidence, flagged as such in the text): CumMix's
perplexity; any recall/retrieval/associative/induction benchmark; an isolation of
PowerNorm, CumStepPos, HarmonicFFN, TACE or DualMomAdam; a byte-level calibration of
the positive weight; an optimizer-window / credit-assignment result of any kind; the
cost of the FSP lookahead in tok/s terms.

## Appendix B — the byte-level measurement of the FSP target (my own; CPU only)

This is the measurement that decides how much the FSP arm can possibly be worth.
Reproduce with the script below (no GPU, no network, ~20 s):

```python
import numpy as np
data = open('data/wikipedia_clean/wikistories/wiki_tinystories_001.txt','rb').read(4_000_000)
b = np.frombuffer(data, np.uint8)
tau, r = 64, 16
W = np.lib.stride_tricks.sliding_window_view(b, tau)
idx = np.arange(0, (len(b)-tau)//r*r, r)
bags = W[idx]; cur = b[idx]
pres = (bags[:,:,None] == np.arange(256)[None,None,:]).any(axis=1).astype(np.float32)
n = len(idx)
def prf(pred, true):
    tp=(pred*true).sum(1); fp=(pred*(1-true)).sum(1); fn=((1-pred)*true).sum(1)
    p=tp/np.maximum(tp+fp,1e-9); rr=tp/np.maximum(tp+fn,1e-9)
    return p.mean(), rr.mean(), (2*p*rr/np.maximum(p+rr,1e-9)).mean()
glob = pres.mean(0)
print("density", pres.mean(), "distinct", pres.sum(1).mean())
print("global prior   ", prf(np.tile(glob,(n,1)), pres))
cond = np.stack([pres[cur==c].mean(0) if (cur==c).sum() else glob for c in range(256)])
print("cond on current byte", prf(cond[cur], pres))
```

**Measured (250k positions, τ = 64):**

| predictor | precision | recall | F1 |
|---|---:|---:|---:|
| context-free global prior | 0.7504 | 0.7546 | **0.7505** |
| conditioned on the current byte | 0.7540 | 0.7581 | **0.7541** |
| conditioned on a 4-gram local context | 0.7692 | 0.7726 | **0.7690** |
| (target density) | | **0.0855** (21.9 of 256 symbols) | |

**Reading:** at byte level the future bag is ~75 % "which letters are common in
English", the current byte adds **+0.36 F1 points**, and a local 4-gram context adds
**+1.85**. The head reads a state dominated by the current byte, so most of what it
can score is available *without any long-range information*. This does not make the
arm pointless — it makes it a **test of a mechanism that may be mostly prior** — and
it sets the bar for reading the result.

**pos_weight calibration** (the paper's 50 is for a 4,096-token vocab; ours is a
256-symbol alphabet): the paper's effective positive fraction is
`50·0.015625 / (50·0.015625 + 0.984375) = 0.4425`; matching that at our measured
density 0.0855 gives **w = 8.5**; exact balance gives **w = (1−d)/d = 10.7**;
the paper's literal 50 gives **82 %** positive mass at our density (all-positive
bias). → **use `--fsp-pos-weight 8.5`** (or 10.7), not 50.

## Appendix C — how to reproduce the §2.2 audit (CPU only, ~2 minutes)

```bash
# 1. the FSP path runs
.venv/bin/python train_gpu.py /tmp/fsptest/fsp.npz train \
  --dataset "data/wikipedia_clean/wikistories/wiki_tinystories_001.txt" \
  --batch 8 --dim 32 --layers 2 --steps 40 --warmup 2 --no-tagged --ring-k 8 \
  --fsp-alpha 0.1 --fsp-r 16 --fsp-tau 64 --fsp-pos-weight 50 \
  --device cpu --seed 0 --log-every 10

# 2. the ruler CANNOT read it (D1)
TMT_RULER_CKPT=/tmp/fsptest/fsp.npz .venv/bin/python tools/ruler.py
#   ValueError: checkpoint ... holds 27,361 parameters but this model was built with 18,913

# 3. the default path is untouched (G1): committed engine vs working-tree engine,
#    same seed, 20 steps -> all 23 checkpoint keys bit-identical (max|diff| = 0.000e+00)
git show HEAD:gpu_port.py > /tmp/oldrepo/gpu_port.py   # and numpy_port.py, train_gpu.py, ...
```

## Appendix D — what this note did not do

* No engine code was modified (the FSP implementation in the working tree is
  **someone else's uncommitted work**; this note audits it and specifies the
  missing pieces, it does not change it).
* No GPU job was run; every number produced here is CPU-only and is labelled as
  such.
* The network was not touched; `docs/ideas/papers/INDEX.md` (the ledger) was not
  modified — the recommended ledger row is described in §4.3 for someone else to
  add.
* The P4 long run in flight on Colab was not disturbed; the audit in §2.2 confirms
  the uncommitted engine change is bit-identical on the default path.
