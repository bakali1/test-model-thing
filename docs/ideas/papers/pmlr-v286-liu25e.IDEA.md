# Paper notes — "Periodical Moving Average Accelerates Gradient Accumulation for Post-Training" / PMA (PMLR v286)

> **Found, but the external gloss was wrong in an interesting way.** The list
> attributed to "a UAI 2025 paper proposing momentum with a 'periodical moving
> average' (PMA)" — and that is a real paper. But the list did not say what it is
> *for*, and the answer is **gradient accumulation**: PMA is a *cheap substitute for
> gradient accumulation*, not a variance-reduction or noise-measurement method. That
> difference matters for TMT, because it makes the paper an argument *against* the
> accumulation experiment the same list was recommending.

**Paper:** Yumou Liu, An Li, Chaojie Li, Fei Yu, Benyou Wang — *Periodical Moving
Average Accelerates Gradient Accumulation for Post-Training*, **Proceedings of the
Forty-first Conference on Uncertainty in Artificial Intelligence (UAI 2025)**, PMLR
**286:2748–2768**, 2025 (eds. Silvia Chiappa & Sara Magliacane). CUHK-Shenzhen,
School of Data Science.

**Source text:** `pmlr-v286-liu25e.paper.txt` and `.paper.pdf`, extracted from the PMLR
PDF. **There is no arXiv preprint** — arXiv searches for "periodical moving average"
and "AdamW-PMA" return nothing relevant (the only near-match, arXiv:2604.13689, is an
unrelated statistics paper on cyclostationary time series). Hence the non-arXiv
filename.

**Code:** `https://github.com/liuyumou/periodical-moving-average`

**Verification status: VERIFIED.** PMLR v286 landing page and the full PDF fetched and
read 2026-10-05; title, all five authors, volume, pages and year confirmed on the
PMLR page; mechanism confirmed in §§1, 3, 4 of the PDF.

---

## 1. What the paper actually claims

The problem it attacks is stated in its own abstract: "**High gradient variance**
presents a significant obstacle to efficient post-training of LLMs on
memory-constrained devices. Existing practical strategies — such as reducing batch
sizes or adopting gradient accumulation (GA) — suffer from an inherent trade-off:
smaller batches exacerbate convergence issues due to increased gradient noise, while
GA substantially prolongs training time owing to its sequential processing."

The mechanism (§1): in AdamW, "since historical gradients receive exponentially
decaying weights," `v_t` is an EMA. "Our method maintains uniform weighting for
recent gradients" — training is structured into **fixed periods**, and within each
period the accumulator is a **uniform** moving average rather than an EMA.
"Periodical" refers to the periodic reset/structure, and the effect is to make the
*momentum update itself* behave like an accumulated large-batch gradient.

Instantiated as **AdamW-PMA** and **Lion-PMA**. Theory: AdamW-PMA "matches the
convergence guarantees of standard Adam" (i.e. no guarantee lost). Results:
**~2× faster training than gradient accumulation** on SFT and DPO, with consistently
better downstream evaluation.

## 2. What this means for TMT — including the part that argues against us

**For:** PMA and Smith et al. ([`1711.00489v2.IDEA.md`](1711.00489v2.IDEA.md)) are
independent evidence that **the batch schedule and the moment schedule are the same
lever**. TMT's recommended "accumulation experiment" can be run either way; PMA is
the wall-clock-cheap version of it. If PMA changes TMT's held-out CE at fixed wall
clock where a true accumulation arm does not, the useful quantity is the *batch-like
smoothing of the moment*, not the extra samples per step — which would be direct
evidence that TMT's gradient is **variance-dominated**, i.e. evidence for the
noise-floor side of the blocker.

**Against (and this is the important part):** the entire premise is
**memory-constrained LLM post-training with minimal parameter drift**. PMA's stated
motive is that EMA "exponentially discounts historical gradients, thereby limiting
their effectiveness in stabilizing parameter updates, **especially during
post-training when parameter drift is minimal**." TMT is the opposite regime:
**from-scratch byte-level pre-training** with maximal parameter drift, 4.48 M
parameters, 890 k steps, and a problem that `RESULTS.md` characterises as an
**over-large-step** regime rather than a high-variance one. Do not expect PMA's
~2× post-training speedup to transfer; the paper makes no from-scratch claim.

Second mismatch: PMA's baseline is **gradient accumulation across micro-batches on
one device**, i.e. it assumes accumulation is *expensive* (sequential forward/backward
passes). TMT at 4.48 M parameters has no such constraint — accumulation is nearly
free here. The paper's advantage is a solution to a problem TMT does not have.

## 3. What this paper does NOT support

- **It does not support "TMT should do gradient accumulation."** It says the
  opposite: if you want the smoothing, there is a cheaper way. Read alongside
  Smith et al., the accumulation recommendation and this paper are competing
  proposals for the same effect, and PMA's is the one that keeps wall clock low.
- **It does not say variance reduction or noise measurement.** There is no SNR
  statistic, no noise scale, no `B_noise`/`B_crit`. "High gradient variance" is the
  motivating observation, not a measured quantity. Nothing here connects to TMT's
  SNR ≈ 0.05 / 0.633 numbers except by analogy.
- **It does not cover from-scratch pre-training**, small models, recurrent nets, or
  byte-level models. SFT and DPO only.
- **It does not validate the `β₂` question.** PMA changes *how the moment is
  weighted*, not `β₂` itself — a distinct intervention from Reddi et al.'s AMSGrad
  (`B ∝ 1/(1−m)`-style running max; see [`1904.09237v1.IDEA.md`](1904.09237v1.IDEA.md))
  and from MaxVA's variance-maximising `β_t`
  ([`2006.11918v4.IDEA.md`](2006.11918v4.IDEA.md)). All four are "reweight the
  second moment" proposals with different rules; do not conflate them.
- **"Matches the convergence guarantees of standard Adam" is a non-result** — it means
  PMA does not break anything, not that it improves the guarantee.
- **~2× is a wall-clock speedup on their tasks**, not a sample-efficiency or quality
  claim. The quality claim is "consistently better performance on downstream
  evaluations," unquantified in the abstract.

## 4. Bearing on the three open questions

- **Noise floor vs over-large step?** This paper is the strongest item on the shelf
  for the **noise-floor / variance** side — it takes high gradient variance as the
  obstacle and shows a smoothing intervention changes outcomes. But the transfer to
  TMT's from-scratch, drifting regime is unestablished, so it is a *hypothesis
  generator*, not evidence.
- **Estimator vs optimiser?** Optimiser (moment weighting), and it is a **third**
  distinct answer alongside AdaBelief's prediction-error denominator and
  MaxVA's variance-maximising `β_t`. Three independent designs all concluding "Adam's
  second-moment EMA is the wrong object" is the strongest convergent signal in this
  whole shelf — for TMT, whose second moment has been *measured* to be
  non-discriminating.
- **Input-dependent gating?** No bearing.

## 5. Reproducing the source

```bash
curl -sSL https://proceedings.mlr.press/v286/liu25e.html -o liu25e.html
curl -sSL https://raw.githubusercontent.com/mlresearch/v286/main/assets/liu25e/liu25e.pdf -o liu25e.pdf
pdftotext -layout liu25e.pdf liu25e.txt
```