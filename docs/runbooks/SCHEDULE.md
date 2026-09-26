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

Week of: 2026-09-25  ·  GPU hours used: **11.36 / 30 Kaggle** ·
Colab validation runs: **5 short runs** (see below)

| date | platform | purpose | run_id | hours | outcome |
|---|---|---|---|---|---|
| night session | Colab T4 | GPU gate #1 (parity + bench) | gate-1 | ~0.1 | speed bug found (per-step ring→CPU sync, 2.6 kB/s); correctness PASS |
| night session | Colab T4 | GPU gate #2 after fix | gate-2 | ~0.1 | **PASS**: 32.4–34.3 kB/s, CUDA loss parity 1.1e-07, roundtrip exact |
| night session | Colab T4 | GPU gate #3: full CUDA state parity | gate-3a/b | ~0.2 | first two runs FAILED (metric floors), third **PASS**: worst tensor 3.4e-05 of RMS, ring ids exact |
| 2026-09-24 08:24 | Colab T4 | **live 30k-step leg** (t 64,747 -> 94,747) | live-leg-1 | 0.4 | all infra checks PASS; throughput 22.1 kB/s vs a 25k floor taken from the synthetic bench; **tags lost when the VM was released** — leg now pushes the final tag |
| 2026-09-25 | Kaggle T4x2 (1 GPU used) | 10k mixed A/B/C calibration, K=18 | tmt-mixed-2026-09-25-cal10k-v3 | ~0.09 h | **PASS**: 0→10,000, 10,240,000 bytes, finite loss/CE, final HF tag + SHA-256 verified, clean reload t/B/K=10000/1024/18; hot=0 at 10k |
| 2026-09-25 | Kaggle T4x2 (1 GPU used) | long mixed A/B/C leg, K=18 | tmt-mixed-2026-09-25-long889897 | 8.59 h | **COMPLETE**: t=889,897, 911,254,528 bytes, wrapper rc=0, clean reload PASS, final HF tag + SHA-256 verified; 104 hot dims, ~111 MB tail at final checkpoint; 400k/600k/final checkpoints downloaded |

Notes: no Kaggle quota spent. All Colab time so far is short validation
(~1 h total). The live leg proved the full path — hub pull with digest
verification, in-session corpus fetch, 30k steps, tagged saves, reload — and
corrected the real throughput to 22.1 kB/s (the synthetic bench's 32.4 kB/s
was optimistic), which is the number corpus sizing now uses.

## Log

| date | platform | session start–end | hours | ckpt pulled (`t`) | ckpt pushed (`t`) | purpose | notes |
|---|---|---|---|---|---|---|---|
| 2026-09-25 | Colab T4 | cal10k v1 | ~0.08 | — | 10,000 | first mixed 10k attempt | **INVALID**: telemetry reported `ring_k=14`, no hot tail, one corpus part >8 MB. Speed 34.6–36.1 kB/s must not be used for planning |
| 2026-09-25 | Kaggle T4x2 | cal10k v2 | 0.00 | — | — | second attempt | aborted before an accepted artifact; no quota spent |
| 2026-09-25 | Kaggle T4x2 | cal10k v3 | 0.09 | — | 10,000 | 10k K=18 calibration | PASS; digest + clean reload verified; produced the 889,897-step budget |
| 2026-09-25 | Kaggle T4x2 | long leg | 8.59 | 10,000 | 889,897 | 8h mixed A/B/C leg | COMPLETE, rc=0; 911,254,528 bytes; 444 telemetry rows, 0 non-finite |
| 2026-09-25 | local CPU venv | ruler @400k | 0.19 | 400,000 | — | held-out CE 1.6999 | 11m23s |
| 2026-09-25 | local CPU venv | ruler @600k | 0.36 | 600,000 | — | held-out CE 1.7133 | 21m50s |
| 2026-09-26 | local CPU venv | ruler @889,897 | 0.20 | 889,897 | — | held-out CE 1.7438 | 12m02s; digest `5291ef86ba4d…` verified |
| 2026-09-25/26 | local CPU venv | chat inspection ×3 | ~0.05 | 400k/600k/890k | — | qualitative generation | 600k never terminated (2000-token cap); 890k clean. **Unrepeated single observation — not a result until fixed-prompt protocol confirms it** |

## Rules

- Save + push every ~10 min (`--save-every`), so Kaggle's hard 12 h stop costs
  at most minutes of work.
- On boot: pull max-`t` checkpoint; **skip if pulled `t ≤ local t`** (a stale
  mirror must never win).
- Never judge an A/B arm across a CPU-leg and a GPU-leg; keep both legs on the
  same platform and `--data-seed`.
- Record hours here the moment a session ends, not at week's end.
