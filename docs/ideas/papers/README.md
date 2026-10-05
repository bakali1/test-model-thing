# Papers

External papers kept **next to the notes taken from them**, so any idea can be
checked against the source without a network round-trip.

Convention, matching the rest of `docs/`:

- `NNNN.NNNNNvN.IDEA.md` — the notes: what the paper claims, what it measured,
  what it admits it did not, and (for this repo) which claims are testable here.
  These are **ideas, not plans and not results**. `../../INDEX.md` and
  `../../../RESULTS.md` win over anything written here.
- `NNNN.NNNNNvN.paper.txt` — the paper's own extracted text, unedited apart from
  HTML-to-text conversion. Figures are not in the text extract; the plots carry
  the quantitative detail, so read them in the original.
- `NNNN.NNNNNvN.paper.html.gz` — the raw fetched HTML, for fidelity.

Items with no arXiv preprint (PMLR-only papers) or which are software rather than
papers use a descriptive stem instead — `pmlr-v28-sutskever13.*`,
`pmlr-v286-liu25e.*`, `gns-pytorch.*`.

**Every note carries an explicit verification status.** Where a paper was found but
does *not* say what an external source attributed to it, that is recorded in the
note's own "what this paper does NOT support" section, and summarised in
[`INDEX.md`](INDEX.md). Do not cite anything here without reading its status line
and its "does NOT support" section.

## Verification ledger

[`INDEX.md`](INDEX.md) — every item recommended by the external LLM answers, with an
explicit **VERIFIED / VERIFIED (corrected) / PARTIALLY VERIFIED / NOT-FOUND /
CLAIM-NOT-SUPPORTED** status, a one-line relevance note, the list of refused
citations and why, and the cross-references into this repo. Built 2026-10-05; every
arXiv ID was fetched and checked against its abstract page, and every found paper was
checked for the claim attributed to it.

## The shelf

| paper | notes | status | why it is here |
|---|---|---|---|
| [arXiv:2609.12303v1](https://arxiv.org/html/2609.12303v1) — *Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models* (UW + Meta FAIR) | [`2609.12303v1.IDEA.md`](2609.12303v1.IDEA.md) | VERIFIED | External evidence for the byte-level thesis, plus two concrete untested levers for TMT (an inserted boundary symbol, and distillation — which is cheap precisely because TMT's vocabulary is already 256 bytes). Also the source of the §6.6 ruler warning: **lower BPB is not a better model**. |
| [arXiv:1812.06162v1](https://arxiv.org/abs/1812.06162) — McCandlish et al., *An Empirical Model of Large-Batch Training* | [`1812.06162v1.IDEA.md`](1812.06162v1.IDEA.md) | VERIFIED | Gradient noise scale `B_simple` predicts the critical batch size — and is **LR-dependent**, which is what makes TMT's 0.05-vs-0.633 SNR gap a sweepable quantity rather than a mystery. |
| [arXiv:1711.00489v2](https://arxiv.org/abs/1711.00489) — Smith et al., *Don't Decay the Learning Rate, Increase the Batch Size* (ICLR 2018) | [`1711.00489v2.IDEA.md`](1711.00489v2.IDEA.md) | VERIFIED | `B ∝ ε` turns an LR schedule into a batch schedule with no tuning — collapses TMT's never-swept `--lr` and the proposed accumulation sweep into one 1-D sweep. |
| [arXiv:2010.07468v5](https://arxiv.org/abs/2010.07468) — Zhuang et al., *AdaBelief Optimizer* (NeurIPS 2020) | [`2010.07468v5.IDEA.md`](2010.07468v5.IDEA.md) | VERIFIED | Denominator is the EMA of `(g−m)²` — a **consistency** measure. The cheapest test of whether TMT's constant-normalizer Adam is discarding per-coordinate signal. |
| [arXiv:1905.10018v3](https://arxiv.org/abs/1905.10018) — Cutkosky & Orabona, *Momentum-Based Variance Reduction in Non-Convex SGD* (STORM) | [`1905.10018v3.IDEA.md`](1905.10018v3.IDEA.md) | VERIFIED | Variance reduction **without mega-batches** — the only item here aimed squarely at "SNR too low at this batch size". |
| [arXiv:2610.01599v1](https://arxiv.org/abs/2610.01599) — Jiang et al., *Convergence Analysis of STORM Under Different Geometries* (2026-10-01) | [`2610.01599v1.IDEA.md`](2610.01599v1.IDEA.md) | VERIFIED (4 days old) | The ID that looked fabricated and is not. Its value is **scoping**: without average smoothness, STORM falls to `O(T^{-1/4})` = SGD's rate. |
| [arXiv:1705.08292v2](https://arxiv.org/abs/1705.08292) — Wilson et al., *The Marginal Value of Adaptive Gradient Methods in Machine Learning* (ICML 2017) | [`1705.08292v2.IDEA.md`](1705.08292v2.IDEA.md) | VERIFIED | Adaptive methods hit equal-or-lower **train** loss with worse **test** error — makes "AdamW ≡ SGD here" a claim that must be tested against plain SGD. |
| [arXiv:1704.04289v2](https://arxiv.org/abs/1704.04289) — Mandt, Hoffman & Blei, *SGD as Approximate Bayesian Inference* (JMLR 2017) | [`1704.04289v2.IDEA.md`](1704.04289v2.IDEA.md) | VERIFIED | Stationary covariance ∝ **η ÷ curvature** — the mechanism behind an over-large-step SNR reading, and a falsifiable prediction. |
| [arXiv:2206.11893v2](https://arxiv.org/abs/2206.11893) — Gu, Gupta, Goel & Ré, *On the Parameterization and Initialization of Diagonal State Space Models* (S4D) | [`2206.11893v2.IDEA.md`](2206.11893v2.IDEA.md) | VERIFIED | "The initialization is critical"; random `A` "does not perform well" — the architecture-side candidate explanation for TMT's 8–32 B memory horizon, and a reason to fix the decay spectrum **before** adding gates. |
| [arXiv:1506.02078v2](https://arxiv.org/abs/1506.02078) — Karpathy, Johnson & Fei-Fei, *Visualizing and Understanding Recurrent Networks* | [`1506.02078v2.IDEA.md`](1506.02078v2.IDEA.md) | VERIFIED | The repo's existing anchor and ruler source (already cited in `docs/plans/PLAN-2026-10-04-ROADMAP.md`), **previously cited but not shelved**. All repo-quoted numbers re-checked: batch 100, RMSProp, lr 2e-3, unroll 100, ~1.3 M params, ~230-char quote cell, **50** epochs. |
| [arXiv:2411.15958v2](https://arxiv.org/abs/2411.15958) — Compagnoni et al., *Adaptive Methods through the Lens of SDEs* (ICLR 2025) | [`2411.15958v2.IDEA.md`](2411.15958v2.IDEA.md) | VERIFIED, claim **corrected** | SDEs for SignSGD/RMSprop/Adam(W). The "Adam ≡ SignSGD when noise-dominated" gloss is true of **vanilla** Adam and **false for AdamW**, which is what TMT runs. |
| [arXiv:2604.25550v1](https://arxiv.org/abs/2604.25550) — Chen & Wang, *Enhancing SignSGD* | [`2604.25550v1.IDEA.md`](2604.25550v1.IDEA.md) | PARTIALLY VERIFIED; **`B_noise` claim NOT SUPPORTED** | The SNR-weighted stationarity measure `Φ_k` is the only instrument here that can separate "converged" from "noise-swamped". The claimed **critical batch size `B_noise` is not in the paper** — it belongs to McCandlish et al. |
| [arXiv:2006.11918v4](https://arxiv.org/abs/2006.11918) — Zhu et al., *MaxVA* | [`2006.11918v4.IDEA.md`](2006.11918v4.IDEA.md) | VERIFIED | A deliberately **conservative** reweighting of the second moment — a testable alternative to TMT's measured constant denominator. Not a noise *measurement*. |
| [arXiv:2605.05794v1](https://arxiv.org/abs/2605.05794) — Wen et al., *Revealing Modular Gradient Noise Imbalance* (MoLS) | [`2605.05794v1.IDEA.md`](2605.05794v1.IDEA.md) | VERIFIED, **precondition false in TMT** | Module-level SNR rescaling. Requires Adam's update to track per-parameter SNR — which TMT has measured flat. Best use is as a falsifiable prediction, not a method. |
| [arXiv:1904.09237v1](https://arxiv.org/abs/1904.09237) — Reddi, Kale & Kumar, *On the Convergence of Adam and Beyond* (ICLR 2018) | [`1904.09237v1.IDEA.md`](1904.09237v1.IDEA.md) | VERIFIED — **ID found; `1804.04235` is Adafactor** | Adam's `β₂` EMA is itself the source of non-convergence. Makes a **`β₂` sweep** the cheapest untested lever on the shelf. |
| [PMLR 28(3):1139–1147](https://proceedings.mlr.press/v28/sutskever13.html) — Sutskever, Martens, Dahl & Hinton, *On the Importance of Initialization and Momentum in Deep Learning* (**ICML 2013**) | [`pmlr-v28-sutskever13.IDEA.md`](pmlr-v28-sutskever13.IDEA.md) | VERIFIED, **venue corrected** | Tuned momentum with a slowly increasing schedule suffices for deep/recurrent nets with long-term dependencies → a third blocker hypothesis for TMT: under-tuned schedule. |
| [PMLR 286:2748–2768](https://proceedings.mlr.press/v286/liu25e.html) — Liu et al., *Periodical Moving Average Accelerates Gradient Accumulation for Post-Training* (UAI 2025) | [`pmlr-v286-liu25e.IDEA.md`](pmlr-v286-liu25e.IDEA.md) | VERIFIED — no arXiv preprint | **Argues against the recommended accumulation experiment**: gets accumulation's smoothing without accumulation's cost — but in a post-training, minimal-drift regime that is not TMT's. |
| [`gns-pytorch`](https://pypi.org/project/gns-pytorch/) v0.1.5 (`elyxlz`) — gradient noise scale in PyTorch | [`gns-pytorch.IDEA.md`](gns-pytorch.IDEA.md) | VERIFIED, **name corrected** | Its README warns that EMA-of-the-ratio is **50–70 % biased** and that `\|G\|²` can go **negative** at low SNR → audit TMT's own SNR estimator *before* attributing the 0.05/0.633 gap. (The repo capitalised `GNS-PyTorch` is a graph simulator.) |

**Not shelved:** `SARAH-AD` — searched on four independent routes and **not found**.
See [`INDEX.md`](INDEX.md) for the evidence and for the near-misses.