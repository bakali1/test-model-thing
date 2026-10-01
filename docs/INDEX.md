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

## Repo layout

The **engine stays flat at the root** on purpose: 24 files import it and the
Colab bundles embed it by flat module name, so a package layout would buy
tidiness at the cost of a fragile refactor across the whole transport layer.

```
# engine (root) — imported by everything, embedded into the Colab bundles
gpu_port.py  numpy_port.py  tail_credit.py       model + tail credit
train_gpu.py                                       the GPU trainer (CPU or CUDA)
checkpoint_hub.py  fetch_corpus.py  fetch_eval_data.py   transport + in-session data
serve.py                                           the chat + memory page (flask or stdlib)
test_tail.py  tmt.npz                              the rolling working checkpoint

tools/     eval.py  eval_npz.py  bench_variants.py        the ruler + benchmark board
           chat_models.py  fs_verdict.py  memory_report.py  sampling + experiment analysis
bundles/   make_*.py                                       Colab bundle generators
           colab_*.py                                      GENERATED (gitignored)
legacy/    main.py  main_torch.py  benchmark*.py          the original per-byte line
           smaller-4.5m.safetensors  tinystories_bench.txt
tests/     check_*.py                                      free CPU gates; run these first
           check_web.py                                   the page: memory math, both servers, the save path
docs/      INDEX.md  plans/  optimization/  reviews/  runbooks/  ideas/
cache/     cached eval transcripts (gitignored)            lets tables re-render free
models/    banked + historical checkpoints (gitignored)
kaggle/    tmt_gpu_train.py                               the Kaggle mirror
data/  corpus/                                              corpora, staged in-session
```

**Every tool runs from any directory.** Each one inserts the repo root on
`sys.path`, so `python tools/eval.py` works from `/tmp`.

### Rebuilding the Colab bundles

```bash
.venv/bin/python bundles/make_fs_bundle.py       # -> bundles/colab_fs_*.py
colab run --gpu T4 --timeout 4200 bundles/colab_fs_w.py
```

The generators read the engine from the repo root and write into `bundles/`.
Each generated script embeds the engine as gzip+base64 and unpacks it on the
VM, because `colab run` uploads only the one script. They are gitignored: they
embed the HF token.

### Where checkpoints live

| path | what it is | why there |
|---|---|---|
| `tmt.npz` (root) | the **rolling working checkpoint** | `checkpoint_hub.py --dest` and `make_leg_bundle.py` default to this exact path, and save/keep-k pruning writes its tags beside it |
| `models/tmt_t094747.npz` (+ `.sha256`) | the banked incumbent (t=94,747) | versioned, so it belongs with the other named models |
| `models/nlt0.1.npz`, `models/nlt-fix-0.1.npz` | earlier prototypes | historical; no current code references them |
| `fs_*.npz` | from-scratch arm checkpoints | **not kept locally.** All verified on the private hub; re-pull with `fs_verdict.py --pull --at-step N`. Cached transcripts in `cache/` let the tables re-render without re-pulling. |
