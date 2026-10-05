# Library notes — `gns-pytorch` (gradient noise scale in PyTorch)

> **Not a paper.** This is a small third-party PyPI package. It is shelved here
> because the external answer recommended "the GNS-PyTorch library (gradient noise
> scale in PyTorch)" as a tool, and because a tool recommendation in a research list
> deserves the same verification as a citation. It gets a shorter note and **no
> `.paper.txt`**, because there is no paper.

**Package:** `gns-pytorch` on PyPI, **v0.1.5**, uploaded **2025-03-17**, author
`elyx <elio@pascarelli.com>`. Repo: `github.com/elyxlz/gns-pytorch`.
**No arXiv ID, no DOI, no peer review.** Unrelated to anything else in this
directory.

**Source text:** `gns-pytorch.README.txt` (PyPI long description) and
`gns-pytorch.GITHUB-README.md` (repo README, master branch) — both fetched and kept
beside this note.

**Verification status: FOUND — but the *name* in the external answer points at a
different project. See §3.**

---

## 1. What it actually does

It implements the McCandlish et al. statistic (see
[`1812.06162v1.IDEA.md`](1812.06162v1.IDEA.md)) directly from per-example losses:

```
B_simple = tr(Σ) / |G|²
```

via three entry points: `gns_per_example(...)` → a `GnsStats` object holding unbiased
estimates of `tr(Σ)` and `|G|²` plus the batch sizes they came from;
`GnsEma(decay)`, which EMAs the two quantities **separately**; and
`compute_gns(...)`, a float-returning compatibility wrapper. `use_vmap=True`
replaces the `n` backward passes with one vmapped pass (incompatible with flex
attention and `torch.compile`); squared norms are accumulated in fp32 regardless of
gradient dtype; `param_percentage` samples a parameter subset.

## 2. The one thing in it worth quoting to the repo

The README's warning is more useful than the estimator:

> "Both `tr(Σ)` and `|G|²` are unbiased, but each single-batch `|G|²` estimate is very
> noisy — its relative standard deviation is roughly `(B_simple / n)·√(2/d_eff)` for
> a batch of `n` — and **often negative when the batch is far below the noise scale**.
> The ratio of two such numbers is biased and heavy-tailed, so **an EMA of per-batch
> GNS values converges to the wrong number (50–70 % off in the synthetic test)**.
> Accumulate `tr(Σ)` and `|G|²` separately and divide once."

This is a **directly actionable warning for TMT's own SNR estimator**, which is a
ratio of gradient quantities over a 4,481,793-parameter byte model
(`tools/state_probe.py`, per `RESULTS.md`). Three implications:

1. If TMT's SNR is computed per-step and then smoothed, **the smoothing is on the
   wrong side of the ratio.** Accumulate numerator and denominator, divide once.
2. `|G|²` **can go negative** at low SNR. A "SNR" that is occasionally negative is
   not a broken model — it is the estimator telling you the batch is far below the
   noise scale. Worth checking whether TMT's ~0.05 ever dips below zero and whether
   anything downstream clips it.
3. A systematic **50–70 % bias** in the naive formulation is the same order as
   TMT's reported 12.6× frozen-vs-moving gap, so the estimator's construction must be
   ruled out *before* the gap is attributed to the iterate moving.

Its unit test (`tests/test_gns.py`) uses a linear model with `tr(Σ)` and `|G|²` known
in closed form and asserts agreement within 10 % — a good template for validating
whatever TMT uses.

## 3. Why this entry is flagged rather than just shelved

The external answer named the tool "**GNS-PyTorch**". There are two different things
with that name, and picking the wrong one silently would have wasted a day:

| what was asked for | what exists | verdict |
|---|---|---|
| a PyTorch **gradient noise scale** library | `elyxlz/gns-pytorch` (PyPI `gns-pytorch`, lowercase) | **this is the real one** |
| a GitHub repo literally named `GNS-PyTorch` | `zhouxian/GNS-PyTorch` — a reimplementation of DeepMind's **Graph Network-based Simulators** (particle dynamics), last push 2022-02 | **different acronym, unrelated to gradients** |

A naive `pip install GNS-PyTorch` or a GitHub search for "GNS-PyTorch" surfaces the
graph-simulator project first. That is almost certainly how the external answer got
the name.

## 4. What this package does NOT support

- **It is not a citation.** v0.1.5, six releases in about two weeks, one author, no
  review, no tests published to a paper. Cite McCandlish et al. for the statistic;
  cite this only for the implementation and the ratio-bias warning.
- **It is not verified against TMT's code.** Nothing here has been run in this repo.
  Per repo policy this note is an idea, not a result.
- **It does not implement `B_crit`.** It computes `B_simple` only; the critical batch
  size still requires the Pareto-front fits of McCandlish et al. §3.
- **The README's own caveats are load-bearing** and are repeated here because they
  are easy to skip: naive EMA-of-the-ratio is biased by 50–70 %, and `use_vmap=True`
  does not compose with `torch.compile` or flex attention — both of which a TMT-side
  integration would plausibly want.
- **Cost.** Per-example gradients are `n` backward passes (or one vmapped pass). At
  TMT's batch sizes that is a periodic cost, which is why the README says to call it
  every ~100 steps.
- It says nothing about recurrent models, byte-level data, or any of the three TMT
  open questions; it is a measurement tool.

## 5. Bearing on the three open questions

- **Noise floor vs over-large step?** It does not answer the question, but §2 gives
  TMT the means to make sure its SNR number is trustworthy before the question is
  asked. **Do the estimator audit before the LR sweep** — otherwise the sweep is
  explaining a bias.
- **Estimator vs optimiser?** It is an **estimator** tool.
- **Input-dependent gating?** No bearing.

## 6. Reproducing the source

```bash
curl -sSL https://pypi.org/pypi/gns-pytorch/json          # metadata + description
curl -sSL https://raw.githubusercontent.com/elyxlz/gns-pytorch/master/README.md
```