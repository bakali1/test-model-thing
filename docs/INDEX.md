# Documentation index

Start with **`../README.md`** (what this is, how to run it) and
**`../RESULTS.md`** (every measured number, including the ruler and today's
from-scratch experiment).

## Where things are

| doc | what it is | status |
|---|---|---|
| `../README.md` | project entry point, how to run | **current** |
| `../RESULTS.md` | the ruler and all measured results | **current — append-only** |
| `plans/PLAN-2026-09-23-GPU.md` | the GPU port plan: engine, hub, correctness gate, week-1 order | **current** |
| `plans/PLAN-2026-09-25.md` | next-day execution plan | **current** (written by the other model) |
| `plans/PLAN-2026-09-22.md` | the original phase plan | superseded by the GPU plan |
| `optimization/OPTIMIZATION_TEST_STATUS.md` | what was measured, what was **not**, what to try next | **current — read before optimizing** |
| `optimization/GPU_OPTIMIZATION_PLAN.md` | the optimization plan and its measured outcome (§0) | historical + measured |
| `optimization/GPU_OPTIMIZATION_PLAN-REVIEW.md` | review of that plan | review, closed |
| `reviews/PLAN-2026-09-23-GPU-IMPLEMENTATION-REVIEW.md` | implementation review + responses | review, closed |
| `runbooks/MORNING-RUNBOOK.md` | how to run the GPU legs | **current** |
| `runbooks/SCHEDULE.md` | quotas and weekly schedule | **current** |
| `ideas/ARCHITECTURE_IDEAS.md`, `ideas/ARCHITECTURE_IDEAS2.md` | speculative directions | ideas, not plans |

## Reading order

1. **`../README.md`** — orientation.
2. **`../RESULTS.md`** — what is actually true, with numbers.
3. **`optimization/OPTIMIZATION_TEST_STATUS.md`** — what has *not* been tested.
   Read this before proposing an optimization; it names the candidates that
   were already measured and lost.
4. **`plans/PLAN-2026-09-23-GPU.md` §7** — the live status and next steps.

## Conventions that matter

- **`RESULTS.md` is append-only.** Corrections are added as new sections that
  explicitly supersede the earlier claim, never by editing history. There is a
  worked example: the "layered mix is dominated" verdict was written, then
  overturned by its own dilution control, and both are recorded.
- **Every number needs its ruler setting.** Sample size, seed and bar travel
  with the number. A CE without its SE is not a result.
- **`OPTIMIZATION_TEST_STATUS.md` has a "NOT tested" section.** Listing only
  successes is how a plan ends up shipping predictions that measurement
  already refuted.

## Local layout

```
gpu_port.py  train_gpu.py  numpy_port.py      engine (fp32, CUDA/CPU parity)
fetch_corpus.py  checkpoint_hub.py            in-session data + checkpoint transport
eval.py  bench_variants.py  chat_models.py    ruler, benchmark board, sampling
fs_verdict.py  make_*_bundle.py               from-scratch experiment + Colab bundles
tests/                                        free CPU gates; run these first
models/                                       banked + historical checkpoints (gitignored)
```

### Where checkpoints live

| path | what it is | why there |
|---|---|---|
| `tmt.npz` (repo root) | the **rolling working checkpoint** | `checkpoint_hub.py --dest` and `make_leg_bundle.py` both default to this exact path, and the save/keep-k pruning writes its tags beside it. Moving it breaks the tooling. |
| `models/tmt_t094747.npz` | the banked incumbent (t=94,747) | versioned, so it belongs with the other named models |
| `models/nlt0.1.npz`, `models/nlt-fix-0.1.npz` | earlier prototypes (2026-09-20/21) | historical; no current code references them |
| `fs_*.npz` | from-scratch arm checkpoints | **not kept locally.** All verified present on the private hub; re-pull with `fs_verdict.py --pull --at-step N`. Their cached eval transcripts (`*.evaltxt.*`, also gitignored) let the verdict tables be re-rendered without re-pulling. |

Generated Colab bundles (`colab_*.py`) are gitignored: they embed the HF
token, and they are rebuilt with `make_*_bundle.py`.
