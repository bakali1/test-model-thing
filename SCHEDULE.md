# SCHEDULE — GPU quota ledger (PLAN-2026-09-23-GPU §2, §5)

One row per session. The Kaggle weekly GPU cap is the hard constraint; this
ledger exists so it is never blown mid-experiment. Append; never rewrite a row.
Platform split rule: Colab = dev loop (<20 min answers), Kaggle = long haul.

## Quotas (verify on signup; these drift)

| platform | quota | session cap | notes |
|---|---|---|---|
| Kaggle GPU | 30 h / week | 12 h | T4 ×1–2 or P100; hard stop at cap |
| Kaggle TPU | 20 h / week | 9–12 h | reserve, not runway (PLAN §4.1) |
| Kaggle CPU | 30 h / week | 12 h | eval / ruler / staging, separate pool |
| Colab (free) | usage-capped, evictible | ~12 h | dev loop; idle disconnect |
| Colab PAYG/Pro+ | credits | 12 h | A100/L4 for B>4096 arms |
| rented (runpod/vast) | ~$0.3–1/h | — | fallback, same `train_gpu.py` |

## Running this week

Week of: _(fill on first Kaggle session)_  ·  GPU hours used: **0.0 / 30 Kaggle** ·
Colab validation runs: **5 short runs** (see below)

| date | platform | purpose | run_id | hours | outcome |
|---|---|---|---|---|---|
| night session | Colab T4 | GPU gate #1 (parity + bench) | gate-1 | ~0.1 | speed bug found (per-step ring→CPU sync, 2.6 kB/s); correctness PASS |
| night session | Colab T4 | GPU gate #2 after fix | gate-2 | ~0.1 | **PASS**: 32.4–34.3 kB/s, CUDA loss parity 1.1e-07, roundtrip exact |
| night session | Colab T4 | GPU gate #3: full CUDA state parity | gate-3a/b | ~0.2 | first two runs FAILED (metric floors), third **PASS**: worst tensor 3.4e-05 of RMS, ring ids exact |
| — | Kaggle | *first long leg: NOT RUN yet* | — | 0.0 | morning decision after the live-checkpoint Colab leg |

Notes: no Kaggle quota spent. The three Colab runs were all short final tests:
two for the original gate, three total attempts for the extended state-parity
gate (the failures were measurement-definition bugs the new gate exposed, each
caught before any training time). Still to run on Colab: the
live-checkpoint 10k-step leg; then Kaggle. The ledger gains a row for each.

## Log

| date | platform | session start–end | hours | ckpt pulled (`t`) | ckpt pushed (`t`) | purpose | notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Rules

- Save + push every ~10 min (`--save-every`), so Kaggle's hard 12 h stop costs
  at most minutes of work.
- On boot: pull max-`t` checkpoint; **skip if pulled `t ≤ local t`** (a stale
  mirror must never win).
- Never judge an A/B arm across a CPU-leg and a GPU-leg; keep both legs on the
  same platform and `--data-seed`.
- Record hours here the moment a session ends, not at week's end.
