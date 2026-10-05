# Verification ledger — the external LLM paper list

Every item an external model recommended for TMT, with an explicit status.
**Built 2026-10-05. Nothing on this page was taken on trust from the recommender.**

- The recommendation arrived with **9 arXiv IDs** and **9 items with no ID at all**,
  from two model answers, with one ID (`arXiv:2610.01599`) encoding a month
  *after* the earlier answer could have been written.
- **Every ID was fetched and checked against its abstract page; every found paper
  was checked for whether it actually says what was attributed to it.**
- **16 of 18 items are shelved. 1 is refused as NOT-FOUND. 1 is shelved with a
  material correction.** Four claims attributed to real papers are recorded as
  **CLAIM-NOT-SUPPORTED**.

**Reading rule for anyone downstream:** if an item is not in this table with a
`VERIFIED` status, do not cite it. If it is `VERIFIED` but its note carries a
"does NOT support" section, that section wins over the summary line.

---

## Status legend

| status | meaning |
|---|---|
| **VERIFIED** | fetched; title + authors match; the attributed claim is in the paper |
| **VERIFIED (corrected)** | paper is real, but a claim attributed to it is wrong or the venue/ID is wrong — see the note |
| **PARTIALLY VERIFIED** | paper is real; some attributed claims are in it, at least one is **not** |
| **CLAIM-NOT-SUPPORTED** | a specific attributed claim was searched for in the full text and is **absent** (attached to the item it belongs to) |
| **NOT-FOUND** | **not shelved.** Could not be located by any route. No `.IDEA.md`, no source text. |

---

## Group A — arXiv IDs given; verify the ID resolves to the claimed paper

| # | citation | ID | status | one-line relevance to TMT |
|---|---|---|---|---|
| A1 | McCandlish, Kaplan, Amodei et al., *An Empirical Model of Large-Batch Training* (2018) | [`1812.06162v1`](https://arxiv.org/abs/1812.06162) | **VERIFIED** | GNS `B_simple` predicts `B_crit`; **noise scale is LR-dependent** → the instrument that turns "SNR 0.05 vs 0.633" into a sweepable quantity. [note](1812.06162v1.IDEA.md) |
| A2 | Smith, Kindermans, Ying, Le, *Don't Decay the Learning Rate, Increase the Batch Size* (ICLR 2018) | [`1711.00489v2`](https://arxiv.org/abs/1711.00489) | **VERIFIED** (Adam evidence is one small figure — see note) | `B ∝ ε` converts an LR schedule into a batch schedule with **no tuning**; makes TMT's never-swept `--lr` and the proposed accumulation sweep the same 1-D sweep. [note](1711.00489v2.IDEA.md) |
| A3 | Zhuang, Tang, Ding et al., *AdaBelief Optimizer* (NeurIPS 2020) | [`2010.07468v5`](https://arxiv.org/abs/2010.07468) | **VERIFIED** | denominator is the EMA of `(g−m)²` — a **prediction-error / consistency** measure, not a scale → the cheapest discriminator for "is Adam's constant denominator throwing signal away?". [note](2010.07468v5.IDEA.md) |
| A4 | Cutkosky & Orabona, *Momentum-Based Variance Reduction in Non-Convex SGD* (STORM) | [`1905.10018v3`](https://arxiv.org/abs/1905.10018) | **VERIFIED** | variance reduction **at fixed batch**, via a stale second evaluation point → the one paper that directly attacks "SNR too low at this batch size". [note](1905.10018v3.IDEA.md) |
| A5 | "Convergence Analysis of STORM Under Different Geometries" | [`2610.01599v1`](https://arxiv.org/abs/2610.01599) | **VERIFIED** (submitted **2026-10-01**, four days before this ledger) | STORM's `O(T^{-1/3})` needs average smoothness; with standard smoothness only it drops to **`O(T^{-1/4})` = SGD's rate**. Scoping, not a reason to adopt. [note](2610.01599v1.IDEA.md) |
| A6 | Wilson, Roelofs, Stern, Srebro, Recht, *The Marginal Value of Adaptive Gradient Methods* (ICML 2017) | [`1705.08292v2`](https://arxiv.org/abs/1705.08292) | **VERIFIED** | adaptive methods reach **equal-or-lower train loss with worse test error** → makes TMT's "AdamW ≡ SGD here, constant normalizer" claim testable against plain SGD. [note](1705.08292v2.IDEA.md) |
| A7 | Mandt, Hoffman, Blei, *SGD as Approximate Bayesian Inference* (JMLR 2017) | [`1704.04289v2`](https://arxiv.org/abs/1704.04289) | **VERIFIED** | stationary covariance ∝ **η ÷ curvature** → the `η`-dependence that predicts a 12.6× frozen-vs-moving SNR gap, and makes it falsifiable by an LR sweep. [note](1704.04289v2.IDEA.md) |
| A8 | Gu, Gupta, Goel, Ré, *On the Parameterization and Initialization of Diagonal State Space Models* (S4D) | [`2206.11893v2`](https://arxiv.org/abs/2206.11893) | **VERIFIED** | "the initialization is **critical**"; random `A` "does not perform well"; sensitive to low-rank perturbations → the architecture-side explanation for TMT's 8–32 B horizon. [note](2206.11893v2.IDEA.md) |
| A9 | Karpathy, Johnson, Fei-Fei, *Visualizing and Understanding Recurrent Networks* | [`1506.02078v2`](https://arxiv.org/abs/1506.02078) | **VERIFIED** — every repo-cited number checks out (batch 100, RMSProp, lr 2e-3, unroll 100, ~1.3 M params, **~230 chars** quote cell, **50** epochs) | the repo's existing external anchor and ruler source. **Was cited in `docs/plans/PLAN-2026-10-04-ROADMAP.md` but was NOT shelved** — now it is. [note](1506.02078v2.IDEA.md) |

**Group A: 9 / 9 IDs resolve to exactly the claimed paper.** No ID was wrong, no ID
was a redirect to a different paper, no ID was invented.

## Group B — no ID given; find them or report them unfound

| # | claimed item | resolved to | status | one-line relevance |
|---|---|---|---|---|
| B1 | "Adaptive Methods through the Lens of SDEs", ICLR 2025 | [`2411.15958v2`](https://arxiv.org/abs/2411.15958) — Compagnoni, Liu, Islamov, Proske, Orvieto, Lucchi. **Comments field confirms "Accepted at ICLR 2025 (Poster)".** | **VERIFIED (corrected)** — venue right; the "Adam ≡ SignSGD when noise-dominated" gloss is **wrong for AdamW** | SDEs for SignSGD/RMSprop/Adam(W); SignSGD's asymptotic loss is **linear in σ** where SGD's is quadratic; "mimics SignSGD" holds for **vanilla** Adam, and **decoupled weight decay changes it**. [note](2411.15958v2.IDEA.md) |
| B2 | Chen & Wang — SignSGD convergence under symmetric noise, SNR-weighted stationarity, critical batch size `B_noise`, annealed Gaussian dithering | [`2604.25550v1`](https://arxiv.org/abs/2604.25550) — Haoran Chen, Wentao Wang, 2026-04-28 | **PARTIALLY VERIFIED** — SNR-weighted stationarity ✓, pre-sign annealed dithering ✓, **`B_noise` / critical batch size = CLAIM-NOT-SUPPORTED** | `Φ_k = Σ|g_i|·min(1, SNR_i)` separates "small because converged" from "small because noise-swamped" → the only instrument here that can answer the blocker question. [note](2604.25550v1.IDEA.md) |
| B3 | "MaxVA" — per-coordinate β₂ maximizing observed gradient variance | [`2006.11918v4`](https://arxiv.org/abs/2006.11918) — Zhu, Cheng, Gan, Huang, Liu, Goldstein | **VERIFIED** — mechanism confirmed verbatim | a *conservative* second-moment weighting, not a faithful noise measure → a directly testable alternative to TMT's measured constant denominator. [note](2006.11918v4.IDEA.md) |
| B4 | "Revealing Modular Gradient Noise Imbalance" (MoLS) | [`2605.05794v1`](https://arxiv.org/abs/2605.05794) — Wen, Liu, Wang, Luo, Shen, Li, Sun | **VERIFIED** | module-level SNR rescaling of Adam — **but its precondition (Adam update ∝ per-parameter SNR) is measured false in TMT**; best use is as a falsifiable prediction. [note](2605.05794v1.IDEA.md) |
| B5 | UAI 2025 paper with a "periodical moving average" (PMA) | *Periodical Moving Average Accelerates Gradient Accumulation for Post-Training*, Liu, Li, Li, Yu, Wang, **PMLR 286:2748–2768 (UAI 2025)**. **No arXiv preprint exists.** [note](pmlr-v286-liu25e.IDEA.md) | **VERIFIED** — PMLR page and full PDF read | **argues *against* the recommended accumulation experiment**: PMA replaces EMA with a uniform moving average *inside fixed periods* to get accumulation's smoothing without accumulation's wall clock, ~2× faster on SFT/DPO. Post-training, minimal parameter drift — **not** TMT's from-scratch regime. [note](pmlr-v286-liu25e.IDEA.md) |
| B6 | "SARAH-AD" — SARAH variance reduction + adaptive learning rates | — | **NOT-FOUND — REFUSED, NOT SHELVED** | — |
| B7 | the GNS-PyTorch library | PyPI **`gns-pytorch`** v0.1.5 (2025-03-17), repo `elyxlz/gns-pytorch` | **VERIFIED (corrected)** — the capitalised repo `zhouxian/GNS-PyTorch` is a **graph-network simulator**, unrelated | implements `B_simple = tr(Σ)/\|G\|²`; its README warns that EMA-of-the-ratio is **50–70 % biased** and that `\|G\|²` can go **negative** at low SNR → an audit of TMT's own SNR estimator before trusting the 0.05/0.633 gap. [note](gns-pytorch.IDEA.md) |
| B8 | Reddi, Kale, Kumar, *On the Convergence of Adam and Beyond* (AMSGrad) | [`1904.09237v1`](https://arxiv.org/abs/1904.09237), comments "Appeared in ICLR 2018" | **VERIFIED — ID found, and the obvious ID is a trap** | `arXiv:1804.04235` is **Adafactor**, not Reddi. AMSGrad's running max is a coarse fix for Adam's `β₂` EMA → **the cheapest untested TMT lever is a `β₂` sweep**. [note](1904.09237v1.IDEA.md) |
| B9 | Sutskever, Martens, Dahl, Hinton, *On the Importance of Initialization and Momentum in Deep Learning* | **PMLR 28(3):1139–1147, 2013** — no arXiv preprint | **VERIFIED (corrected)** — it is **ICML 2013**, *not* an ICLR 2013 workshop | tuned momentum + a **slowly increasing momentum schedule** suffices for deep/recurrent nets on long-term dependencies → introduces a third blocker hypothesis for TMT: *under-tuned schedule*, not floor and not over-large-step. [note](pmlr-v28-sutskever13.IDEA.md) |

**Group B: 8 shelved, 1 refused.**

---

## Refused

| item | why it is not on this shelf |
|---|---|
| **"SARAH-AD"** | Searched and not found. arXiv API `all:"SARAH-AD"` → **0 entries**; arXiv web search → 0 results; loose queries surfaced only unrelated work (Cox-model inference arXiv:2602.07477, 5G-NR error correction arXiv:2604.04277); DuckDuckGo returned only social-media profiles; existing SARAH variants found are **SARAH-M**, *Accelerating mini-batch SARAH by step size rules*, error-feedback SignSGD — **none named SARAH-AD**; the UAI 2025 volume contains no "SARAH". One engine (Google Scholar) bot-blocked automated access, so this is *absence of evidence on four independent routes*, not proof of non-existence. **Not shelved.** The nearest real things it may be a garbled memory of are [`1905.10018v3`](https://arxiv.org/abs/1905.10018) (STORM — variance reduction **with** adaptive learning rates, and the actual "SARAH-like without mega-batches" method) and Reddi et al. (adaptive rates **without** variance reduction). |

## Claims that do not survive contact with the text

The most valuable output of this pass. In each case the paper is real and the ID is
right, but the attributed claim is absent or wrong.

| attributed claim | the paper actually says | status |
|---|---|---|
| Chen & Wang's SignSGD paper defines **"a critical batch size `B_noise`"** | `B_noise`, `Bnoise`, "critical batch size" and "critical batch" **do not occur anywhere in the paper**. What it has is a *non-vanishing* noise floor `σ̃₁/√n` that **decreases** with batch and is never thresholded. `B_noise` belongs to **McCandlish et al. 1812.06162** — the two papers are unrelated and were conflated. | **CLAIM-NOT-SUPPORTED** — see [`2604.25550v1.IDEA.md`](2604.25550v1.IDEA.md) §2 |
| ICLR-2025 SDE paper shows "**Adam becomes indistinguishable from SignSGD** when the second moment is noise-dominated" | True of **vanilla** Adam/RMSprop *asymptotics on convex quadratics*, and the paper says the **opposite for AdamW/RMSpropW**: decoupled weight decay stabilises at high noise. TMT runs **AdamW**. The abstract's own summary is "the role of noise is **much more complex**." | **CLAIM MISATTRIBUTED** — see [`2411.15958v2.IDEA.md`](2411.15958v2.IDEA.md) §2 |
| Sutskever et al. is an "**ICLR 2013 workshop**" paper | PMLR v28(3):1139–1147 is the **30th ICML**, 2013. The PMLR part of the citation was right; the venue was not. | **VENUE CORRECTED** |
| "GNS-PyTorch" is a gradient-noise-scale library | The PyPI package is lowercase **`gns-pytorch`** (`elyxlz`). The repo literally named `GNS-PyTorch` (`zhouxian`) is a **Graph Network Simulator** — different acronym. Following the capitalised name gets the wrong project. | **NAME CORRECTED** |
| "MaxVA chooses β₂ to recover gradient-variance information" | MaxVA chooses the weighting to **maximise** a variance *estimate*, producing a **conservative step size for stability**. It does not claim the maximised quantity is a faithful per-coordinate noise measure. | **CLAIM OVERSTATED** — see [`2006.11918v4.IDEA.md`](2006.11918v4.IDEA.md) §3 |
| MoLS is "module-level SNR scaling for LLMs" | Correct, but the gloss omits that MoLS **requires** Adam's update magnitude to track per-parameter SNR (Orvieto & Gower 2025) — **which TMT has measured to be flat** (`\|update\|/\|m\|` spread 1.0020×). The precondition is locally false. | **CLAIM INCOMPLETE** — see [`2605.05794v1.IDEA.md`](2605.05794v1.IDEA.md) §3 |
| PMA is a variance-reduction / noise method | It is a **cheap substitute for gradient accumulation**: EMA → uniform moving average inside fixed periods. Its whole setting is memory-constrained **post-training with minimal parameter drift**; TMT is from-scratch pre-training with maximal drift. | **CLAIM MISLEADING** — see [`pmlr-v286-liu25e.IDEA.md`](pmlr-v286-liu25e.IDEA.md) §2 |

**Also worth recording:** `arXiv:2610.01599` was the ID I expected to be fabricated and
it is **real**, submitted four days before this ledger. The correct lesson is that a
future-dated ID is evidence of recency, not of invention — and that the *un*
dated-looking items (`B_noise`, `SARAH-AD`, "GNS-PyTorch") were where the errors
actually were.

---

## Cross-references into this repo

- **`docs/ideas/ARCHITECTURE_IDEAS2.md` references NONE of these.** Checked: no
  arXiv IDs, no author names, no paper titles, no `cs.*` identifiers anywhere in its
  1 548 lines. Whatever it argues, it argues without this shelf — so if any of these
  papers changes its reasoning, the connection has to be made by hand.
- **`docs/ideas/ARCHITECTURE_IDEAS.md`**: also no references to any of these.
- **`docs/plans/PLAN-2026-10-04-ROADMAP.md`**: references **Karpathy only**, at
  lines 32, 103–104, 127, 198, 279, 289, 293, 344, 412, 446 — including the
  `~230 chars` / BPTT-100 ruler and the already-recorded "it's 50 epochs, not 2000"
  correction (both confirmed correct against arXiv:1506.02078v2).
- **`RESULTS.md`**: the blocker facts these notes are written against — SNR ≈ 0.05
  training / 0.633 frozen θ (12.6×, diagnosed as over-large step); memory horizon 8 B
  with bracket [8, 32] B; 8,192 `.decay` coordinates of 4,481,793 params (0.183 %);
  Adam's second moment measured non-discriminating (`|update|/|m|` spread 1.0020×,
  `|m|` spread 763×, AdamW ≡ SGD); global `--lr` **never swept**.

---

## What is *not* established by anything on this shelf

Read this before quoting the directory as support for a decision.

- **Nothing here measures gradient SNR, noise scale, or `B_crit` in TMT.** Every SNR
  statement is borrowed from McCandlish et al. or from the Chen & Wang `Φ_k` measure.
  The SNR audit in [`gns-pytorch.IDEA.md`](gns-pytorch.IDEA.md) §2 says that may be a
  prerequisite, not a detail.
- **No paper on this shelf tests a byte-level recurrent model.** Not one.
- **No paper on this shelf supports or refutes TMT's JEPA latent objective or the
  O(1)-memory claim.** Neither direction.
- **Every architecture claim (S4D, Karpathy, Sutskever) is from a non-byte, non-TMT
  model**, and every optimiser claim (Wilson, MaxVA, Reddi, PMA) is from a CNN, an
  RNN, or a Transformer far larger than 4.48 M parameters.
- **The shelf's convergent signal is narrow and it is all optimiser-side:** three
  independent designs (AdaBelief's prediction-error denominator, MaxVA's
  variance-maximising `β_t`, PMA's uniform-in-period average) all conclude that
  **Adam's second-moment EMA is the wrong object**. TMT's own measurement that its
  second moment is non-discriminating is consistent with all three. That is a
  hypothesis with three independent convergences behind it — still a hypothesis.
- **Per this repo's standard** (`../../INDEX.md`, `../../../RESULTS.md`): a test that
  passes is a hypothesis, not a proof, and `RESULTS.md` wins over anything on this
  page.