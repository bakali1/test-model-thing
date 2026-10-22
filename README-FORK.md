# README-FORK.md — TMT (fork)

## What this is
- **Upstream:** [jrz97619761/test-model-thing](https://github.com/jrz97619761/test-model-thing) (MIT, © 2026 jrz97619761). A proof-of-concept **byte-level recurrent LM** (JEPA-style latent prediction, internal state + RTUs, byte I/O, continual/test-time training). 
- **Scale:** **4.5M parameters** (default `dim=512`, `layers=16`). The model has **never been trained above 4.5M params** in this repo.
- **Engines:** Original MLX (Apple Silicon/Linux CPU possible) and a **bit-identical** torch port (`gpu_port.py` ≡ `numpy_port.py`, gated by `tests/check_parity.py`). Default GPU path uses fused `nvrtc` ring credit (**1.26× end-to-end, bit-identical**; 26.72 vs 33.68 ms/step).

## What this fork adds
- **Bit-identical numpy↔torch engine** with a parity gate.
- **35 executable gates** (`tests/`) enforcing correctness (parity, ridge readout, LR roles, etc.). **Every new gate is run against the pre-change code and the failure is recorded** — a gate that has never failed is not known to test anything.
- **Colab/HF durability layer** (`checkpoint_hub.py`): content-verified transport that survives runtime recycling (push every ~2000 steps, re-pull/resume).
- **1.26× faster ring credit** via fused `nvrtc` (default on CUDA), with measured-and-losing candidates removed. **This speedup was inert until 2026-10-22** — the constructor eagerly rewrote the `auto` default to the slower path, so the branch was dead code; it is now resolved lazily and gated (`tests/check_ring_credit_default.py`).
- **Capacity/result work (P4+P6):** 15× incumbent retention with **95.4 B half-life** (37.5%/17.1%/5.3%/0.78% at 8/32/128/512 B vs incumbent 4.8%/1.7%/0.36%/0.0007%); eroder is **weight decay on the decay logits** (92.30 → 39.07 B without P6).
- **Append-only measurement log:** all results recorded in `RESULTS.md` (corrections appended, never history-edited). Instruments were fixed when found broken (e.g. ridge readout λ swap/byte-0 ravel/padding).

## How to run it
**Hardware:** MLX on Apple Silicon or Linux CPU (`mlx[cpu]`); torch/CUDA for GPU runs (T4/A100 etc.). GPU leg protocol in [`docs/runbooks/MORNING-RUNBOOK.md`](docs/runbooks/MORNING-RUNBOOK.md).

**Setup (upstream-style):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install 'mlx[cpu]'  # or torch/cuda deps as appropriate
```

**Entry points:**
- `main.py` (MLX) — prompts for mode or use env vars: `TMT_MODE=0|1`, `TMT_STEPS=N`, `TMT_LOG=1`. Saves/loads weights to `model.safetensors` by default; falls back to `data/*.txt` or built-in sample if `data/wikipedia_clean/` missing.
- `train_gpu.py` / `gpu_port.py` (torch) — batched GPU training; checkpoints via `checkpoint_hub.py`. Ruler: `python tools/eval.py <checkpoint.npz> [--slices 3] [--bytes 20000]`.

**Data:** Corpora are **gitignored and local-only**. Training/eval data fetched via `fetch_corpus.py`/`fetch_eval_data.py` into `data/`, `corpus/`, etc. `eval.py` refuses to run if any eval file is reachable by the training glob. See [`docs/INDEX.md`](docs/INDEX.md) for layout.

## Measured numbers (from `RESULTS.md`)
| Quantity | Value | Source |
| --- | --- | --- |
| Clean baseline CE (held-out `tinystories_valid.txt`) | **1.1741 ± 0.0415** nats/byte (t=64747) | `RESULTS.md` (2026-09-23) |
| Banked incumbent (roadmap ref) | **1.1741** per roadmap; checkpoint `models/tmt_t094747.npz` | `RESULTS.md`, `docs/INDEX.md` |
| Order-8 byte n-gram (0 params, byte-matched to ruler) | **0.6512** nats/byte | `RESULTS.md` §6 (D2), `tools/ngram_ce.py` |
| **Gap vs n-gram** | **Model is 0.56 nats/byte worse** (1.2094 vs 0.6512; ×1.74 perplexity) | `RESULTS.md` §6 |
| P4+P6 retention (vs incumbent 4.8/1.7/0.36/0.0007%) | **37.5% / 17.1% / 5.3% / 0.78%** at **8 / 32 / 128 / 512 B** | `RESULTS.md` §0/plan |
| P4+P6 half-life | **95.4 B** (vs 1.05 B incumbent) | `RESULTS.md` §0, `docs/plans/PLAN-2026-10-21-CONTINUAL-LEARNING.md` §0 |
| Ring credit speedup | **1.26× end-to-end, bit-identical** (26.72 vs 33.68 ms/step) | `RESULTS.md`, `docs/optimization/...` |
| State readout (fixed instrument, D5) — **on the D5 delayed-copy model** | Payload **4.6× over null at 8–12 B**, **1.5× at 128 B**; **cue key 65–165× over null, holds ~27% at 128 B**. Works over **~12–128 B** depending on metric. | `RESULTS.md` §7 (D5), `tools/d5_probe.py` |
| **Continual learning, first trained arm** | **ACQUIRE PASS** (0.8906 exact-4 on the stream's facts, null 0.0020) · **RETAIN FAIL** (ratio 0.0022, threshold 0.90) · **NO REGRESSION FAIL** (3.0695 → 5.2037 nats/B; unigram baseline 5.5452) | `RESULTS.md` §20, `tools/cl_ruler.py` |
| Why RETAIN fails | The **state carries the address, the weights carry the content**: a fresh readout on the state, split over board facts, scores **exact-4 = 0.0000 in every cell**. The content lives in a **table** in the head's weights, and the new stream overwrites it. | `RESULTS.md` §22–§23 |

## Honest caveats
- **Worse than a trivial baseline:** The model is **0.56 nats/byte WORSE than a zero-parameter order-8 byte n-gram on the same bytes** (0.6512 vs 1.2094). The CE ruler is not the headline; approaching the n-gram is the bar.
- **Readout is limited, and it depends on the model:** on the **D5 delayed-copy model** the state holds an addressable persistent **address** and a decaying **content**; decodable payload is real but short-horizon (~12 B strong, fading by 128 B). Earlier "never beats control / ~12 B horizon" claims were **artifacts of broken instruments** (λ swap, byte-0 ravel, post-padding read). **On the CL-2 continual-learning arm the state holds only the address** — the content moved into the weights (§22) — so "the state holds a payload" is **not** a general property of this architecture, it is a property of a model whose objective asked for it.
- **Data not public:** The corpora are **not in git** and are **local-only**. Eval sets live under `data/eval/` (moved out of training glob).
- **Scale:** **4.5M params max tested**; larger runs (e.g. 34M CUDA fork) showed no retrieval past 64 B in those tests.
- **The continual-learning result is a FAIL, and it is the honest headline.** The programme's goal is continual learning, judged by three pre-registered numbers, and the first **trained** arm scores **ACQUIRE PASS / RETAIN FAIL / NO REGRESSION FAIL** (`RESULTS.md` §20): the model recalls the stream's facts exactly, **forgets every fact it knew before**, and its language modelling degrades from 3.07 to 5.20 nats/byte — within 0.34 nats of predicting a constant byte. **The cause is known** (§22–§23): the memory ended up in a **table in the weights** rather than in the recurrent state, and a new stream overwrites that table. The next arms are a training-signal arm (make the state carry the payload) and an **explicit keyed store**. The repo follows its own rules: **"a CE without its SE is not a result"**, **"a null with a measured cause is a result"**, and **"a measurement's meaning is set by its controls"**.
- **Instruments matter — this is the programme's main methodological result:** many earlier nulls were **measurement bugs**, and four of them in one night produced four wrong conclusions. The controls are now part of the criterion itself: a **matched null**, a **nuisance control**, a **copy control** (replace the payload in the input, keep the key), and — for anything fitted — a **split over the things the readout could memorise** (over the board facts, not over the rows). Gates exist to prevent regressions (`tests/check_ridge_readout.py`, `tests/check_ruler_criterion.py`, `tests/check_state_capacity.py`, parity gates).
