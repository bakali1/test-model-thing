# RESULTS — the trustworthy ruler (PLAN-2026-09-22 Phase 0)

Run id / data / seed / bytes / per-slice CE / histogram / chat verdict.
One row per checkpoint that a decision was based on. Never edit a row; append.

## Ruler

    .venv/bin/python eval.py <checkpoint.npz> [--slices 3] [--bytes 20000]

Prints: held-out CE per disjoint slice + mean ± SE, unigram/bigram baselines
from the training corpus, cross-domain CE on the Gutenberg novels, decay /
half-life histogram, hot-dim count, and stop-head AUC + recall/precision.

`eval.py` refuses to run if any eval file is reachable by the training glob.
Held-out = `data/eval/` (physically outside `data/wikipedia_clean/`).
The held-out set for in-domain is `wiki_tinystories_005.txt`.

## Phase 0 changes

  - **0.1 split.** `wiki_tinystories_005.txt` moved to `data/eval/`; the
    training glob `data/wikipedia_clean/**/wiki_*` can no longer reach it
    (4 files, 165 MB remain). `eval.py` has a hard overlap guard.
  - **0.2 `--data-seed`.** The training file shuffle is seeded (default 0) and
    its order is logged at startup, so two A/B arms share one byte stream.
  - **0.3 tagged saves.** `--save-every` writes `tmt_t{step:06d}.npz` and keeps
    the last `--keep-k` (default 8); `tmt.npz` is a hard link to the newest tag.
    A result's checkpoint is never silently overwritten.
  - **0.4 `eval.py`.** Unified ruler (above). `eval_npz.py` is kept for its
    stop-head detail and now also finds file 005 under `data/eval/`.

## Contamination caveat (important)

The live checkpoint `tmt` **trained on all five wikistories files**, including
005, before the split existed. Its held-out CE below is therefore optimistic
and NOT a clean generalization number — it is recorded only as the pre-split
reference. Every checkpoint trained from now on (with 005 in `data/eval/`) has
a genuinely held-out in-domain number. The **cross-domain** (novels) number
was never in the wiki training glob and is honest for all checkpoints.

## Log

| date | ckpt | t | bytes | held-out CE (mean ± SE) | cross-domain CE | hot dims | half-life mean | note |
|---|---|---|---|---|---|---|---|---|
| 2026-09-23 | tmt.npz | 64,747 | 57,602,080 | 1.2235 ± 0.0103 (3×20k, file 005) | 3.3785 ± 0.2933 (12 novels × 8k) | 57 | 1.12 B | pre-split ckpt: 005 was trained on, so in-domain CE is optimistic; ruler baseline only |

Stop head @ tmt.npz: AUC 0.9710; thr 0.35 recall 11.2% / precision 51.0%.
Memory histogram (half-life): <1 B: 2131 | 1–2 B: 5926 | 2–4 B: 124 |
4–8 B: 9 | 8–16 B: 1 | 16–32 B: 0 | >32 B: 1.
