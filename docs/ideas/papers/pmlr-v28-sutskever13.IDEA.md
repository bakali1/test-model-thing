# Paper notes — "On the Importance of Initialization and Momentum in Deep Learning" (PMLR v28)

**Paper:** Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton — *On the
importance of initialization and momentum in deep learning*, **Proceedings of the
30th International Conference on Machine Learning (ICML 2013)**, PMLR
**28(3):1139–1147**, 2013.

**Source text:** `pmlr-v28-sutskever13.paper.txt` (and `.paper.pdf`) — extracted from
the PMLR PDF; there is no arXiv preprint, hence the non-arXiv filename.

**Verification status: FOUND — with a VENUE CORRECTION.**

The external answer cited it as "*ICLR 2013 workshop, PMLR v28*". **The PMLR part is
right; the venue is wrong.** The PMLR volume-28 landing page states: *"Proceedings
of the 30th International Conference on Machine Learning, PMLR 28(3):1139–1147,
2013"* — that is **ICML 2013**, not an ICLR workshop. Quote it as ICML 2013. (ICLR
2013's own workshop proceedings are a separate, non-PMLR publication.)

**The attributed claim is otherwise fully supported** — verified against the
published abstract, quoted verbatim:

> "Deep and recurrent neural networks (DNNs and RNNs respectively) are powerful
> models that were considered to be almost impossible to train using stochastic
> gradient descent with momentum. In this paper, we show that when SGD with momentum
> uses a well-designed random initialization and a **particular type of slowly
> increasing schedule for the momentum parameter**, it can train both DNNs and RNNs
> (on **datasets with long-term dependencies**) to levels of performance that were
> previously achievable only with Hessian-Free optimization. We find that **both the
> initialization and the momentum are crucial** since poorly initialized networks
> cannot be trained with momentum and well-initialized networks perform markedly
> worse when the momentum is absent or poorly tuned."

---

## 1. What the claim is, precisely

Momentum is not a generic accelerator here; it is the thing that makes a deep/recurrent
objective trainable **at all**, and the paper attributes the historical failure to
*poor initialization schemes* rather than to first-order methods being inadequate:
"carefully tuned momentum methods suffice for dealing with the curvature issues in
deep and recurrent network training objectives without the need for sophisticated
second-order methods." The mechanism named is a **slowly increasing momentum
schedule** (`μ` ramped, not fixed at 0.9 from step 0) — which is *the opposite* of
the schedule in [`1711.00489v2.IDEA.md`](1711.00489v2.IDEA.md) §1's `B ∝ 1/(1−m)`
rule (which *raises* `m`). Both are consistent — one ramps momentum up, the other
scales the batch as momentum grows — but a TMT arm must not do both naively.

## 2. What this means for TMT

The relevance is to the **recurrent memory component**, exactly as the external gloss
said, and it lands on a specific untested thing.

TMT is a 16-layer byte-level recurrent model whose stated blockers are gradient SNR
(≈ 0.05 moving / 0.633 frozen) and an 8–32 B memory horizon. Sutskever et al. say
that for deep and recurrent nets, a **fixed** momentum coefficient is a mis-set
parameter and that a ramp is part of what makes optimisation work at all. TMT's
repo has never swept the global `--lr` (`RESULTS.md`) and, as far as the shelf can
tell, has no documented momentum ramp.

Two concrete uses, both cheap:

- **Separate "the optimiser is under-tuned" from "the architecture has a memory
  wall."** If a slowly-ramped momentum (and a swept `--lr`) moves held-out CE
  materially at unchanged architecture, the 8–32 B horizon is a *consequence* of
  under-training, not a ceiling. That directly attacks the repo's open question of
  whether the SNR blocker is a noise floor or an over-large-step regime, from a
  third angle: neither — it is a **schedule** problem.
- **It corroborates Karpathy.** Karpathy's largest model used **RMSProp**, not
  momentum-SGD, and reached a ~230-char quote cell with BPTT-100
  ([`1506.02078v2.IDEA.md`](1506.02078v2.IDEA.md)). Sutskever shows tuned momentum is
  *sufficient* for long-range RNNs. Neither is evidence about a byte-level 4.48 M
  model with an 8–32 B horizon, but together they establish that **"recurrent memory
  didn't form" is not by itself evidence that a gate or an SSM is required.**

Named test: an arm with momentum ramped from ~0 to the production value over the
first N steps, at fixed everything else, judged on held-out CE and the state-probe
horizon. If the horizon moves, the memory wall was an optimisation artefact.

## 3. What this paper does NOT support

- **It is not about byte-level or character-level language modelling, and not about
  any modern architecture.** The models are 2013-era deep belief nets / deep
  transition networks on MNIST and CIFAR-10, plus RNNs on synthetic
  long-dependency tasks (adding a problem, Hochreiter & Schmidhuber 1997). The RNN
  experiments use **Hessian-free optimisation** as the comparison point, so the
  headline "momentum suffices" is about matching Hessian-free on those specific
  datasets — not about competing with today's byte-level models.
- **It says nothing about Adam, adaptive denominators, SNR, or noise scale.** The
  optimiser is plain SGD with momentum. Do not use it to argue anything about
  TMT's constant-normalizer finding.
- **It does not establish that a momentum ramp is necessary in 2016+ settings.**
  Adam's `β₁ = 0.9` *is* a form of momentum warm-up (effective momentum starts at
  0.1), and AdamW dominates in the architectures TMT cares about. The honest
  statement is that the *degree* of ramp is a real hyperparameter in a plain-momentum
  regime, not that any TMT arm must add one on top of Adam.
- **"Initialization is crucial" is about 2013 random-init schemes** (careful
  orthogonal/data-dependent init), not about modern initialisation practice. Do not
  carry it into a claim that TMT's initialization is a problem.
- **Nothing about memory horizons, retrieval, decay parameters, or the
  8,192-of-4,481,793 slice.**
- PMLR 28(3) is a 9-page conference paper. The published text is what is quoted
  here; there is no arXiv version to check against.

## 4. Bearing on the three open questions

- **Noise floor vs over-large step?** Introduces a **third** hypothesis the repo has
  not named: *under-tuned schedule* (momentum ramp + never-swept LR). If it holds, the
  SNR reading is a symptom, not the cause.
- **Estimator vs optimiser?** Optimiser, again, and never the estimator.
- **Input-dependent gating?** The most directly anti-gating item on the shelf:
  "carefully tuned momentum methods suffice for dealing with the curvature issues in
  deep and recurrent network training objectives." Fix the optimiser before adding
  input-dependent architectural machinery.

## 5. Reproducing the source

```bash
curl -sSL https://proceedings.mlr.press/v28/sutskever13.pdf -o sutskever13.pdf
pdftotext -layout sutskever13.pdf sutskever13.txt
```