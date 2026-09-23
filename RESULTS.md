# RESULTS — the trustworthy ruler (PLAN-2026-09-22 Phase 0)

Run id / data / seed / bytes / per-slice CE / histogram / chat verdict.
One row per checkpoint that a decision was based on. Never edit a row; append.

## Ruler

    .venv/bin/python eval.py <checkpoint.npz> [--slices 3] [--bytes 20000]

Prints: held-out CE per disjoint slice + mean ± SE, unigram/bigram baselines
from the training corpus, a same-task distribution-shift CE, cross-domain CE on
the Gutenberg novels, decay / half-life histogram, hot-dim count, and stop-head
AUC + recall/precision.

`eval.py` refuses to run if any eval file is reachable by the training glob.
The held-out corpora live physically outside `data/wikipedia_clean/`:

  * `data/eval/` — in-domain held-out. `wiki_tinystories_005.txt` (the file
    moved out at the Phase 0 split) + `tinystories_valid.txt` (see below).
  * `data/eval_shift/` — `tinystories_v2gpt4_valid.txt`, TinyStoriesV2 generated
    by GPT-4: same task, independent generation ⇒ a distribution-shift number.

### Held-out data provenance (`fetch_eval_data.py`, reproducible)

    .venv/bin/python fetch_eval_data.py

Downloads the canonical TinyStories validation splits from
`huggingface.co/datasets/roneneldan/TinyStories`, strips the `<|endoftext|>`
markers (the training corpus has none), and drops stories that are
near-duplicates of a training story (TinyStories is heavily templated, so the
official split still shares ~3% of stories with the train set). The result is
byte-identical on every run (sha256-stable).

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
005, before the split existed. Its CE on 005 is optimistic and NOT a
generalization number. The clean baseline is therefore measured on
`data/eval/tinystories_valid.txt` (never in the training glob) and on the
novels (cross-domain; the wiki run never read them).

## Log

Baseline command (hold fixed for future rows unless noted):

    .venv/bin/python eval.py tmt.npz --eval-glob 'data/eval/tinystories_valid.txt' --xd-files 4

| date | ckpt | t | bytes | in-domain held-out CE | shift CE | cross-domain CE | hot dims | half-life mean | note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-23 | tmt.npz | 64,747 | 57,602,080 | *(1.2235 ± 0.0103 on file 005 — CONTAMINATED, reference only)* | – | 3.3785 ± 0.2933 (12 novels) | 57 | 1.12 B | pre-split ckpt: 005 was trained on |
| **2026-09-23** | **tmt.npz** | **64,747** | **57,602,080** | **1.1741 ± 0.0415** (3×20k, `tinystories_valid`) | **1.1425 ± 0.0165** (3×20k, v2gpt4) | **3.1532 ± 0.4275** (4 novels × 8k) | 57 | 1.12 B | **clean baseline** — held-out bytes never trained on |

Stop head @ clean baseline: AUC 0.9547; thr 0.35 recall 9.6% / precision 40.4%.
Memory histogram (half-life): <1 B: 2131 | 1–2 B: 5926 | 2–4 B: 124 |
4–8 B: 9 | 8–16 B: 1 | 16–32 B: 0 | >32 B: 1.

The in-domain SE (0.04) is currently the coarse number: one held-out file, 3
slices. A win inside ~1 SE is not a win. Raise `--slices` before trusting a
sub-0.1 CE difference, or add a second held-out file.
