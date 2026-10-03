"""decontaminate.py — drop generated stories that overlap TRAINING **or EVAL** bytes.

Why this file exists
--------------------
The distillation corpus (PLAN-2026-10-03-DISTILL.md step 1) is written by a
teacher model in the *ruler's own register*. That is what makes it useful, and
it is also what makes it dangerous: the teacher was trained on web text that
very likely CONTAINS TinyStories, so it can regurgitate held-out eval stories
verbatim — poisoning the very ruler we score with.

`fetch_eval_data.train_lines()` builds its membership set from the TRAINING
glob only. That is correct for its job (de-duplicating an official split
against the train set) and *wrong* here: a generated story copied from
`data/eval/tinystories_valid.txt` is invisible to it. So the membership set is
widened here to train ∪ eval ∪ shift.

What the filter is and is not
-----------------------------
It is `fetch_eval_data.prepare()`'s existing, already-blessed test: a story is
dropped when >= DUP_THRESH of its >=40-char lines appear VERBATIM in the
membership set. That catches copying. It does **not** catch near-paraphrase:
TinyStories is heavily templated, so a teacher can produce a story with the
same beats and different wording and sail through. Every report produced from a
decontaminated corpus must carry that caveat (the plan states it), because a
clean filter result is not a proof of independence.
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import fetch_eval_data as fed  # noqa: E402

# Train + every eval surface the ruler can score. `data/eval` and
# `data/eval_shift` are separate globs from the training one by construction
# (Phase 0.1 moved wiki_tinystories_005.txt out of the training path).
DEFAULT_GLOBS = (
    "data/wikipedia_clean/**/wiki_*",     # what TMT trains on today
    "data/eval/*",                        # the in-domain ruler
    "data/eval_shift/*",                  # the distribution-shift ruler
    "data/*.txt",                         # cross-domain novels, if ever trained on
)


def membership_lines(globs=DEFAULT_GLOBS, verbose=True):
    """set of lines that a generated story must not reproduce verbatim."""
    lines = set()
    for pattern in globs:
        for path in sorted(glob.glob(pattern, recursive=True)):
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as fh:
                lines.update(fh.read().split(b"\n"))
            if verbose:
                print(f"  membership: {pattern} -> {os.path.basename(path)}")
    if verbose:
        print(f"[decontaminate] membership set: {len(lines):,} distinct lines")
    return lines


def filter_corpus(raw, lines, thresh=fed.DUP_THRESH, verbose=True):
    """Return (kept_bytes, dropped_stories, dropped_bytes).

    Thin wrapper over fetch_eval_data.prepare() so there is exactly ONE
    implementation of the drop rule in the repo.
    """
    kept, dropped, dbytes = fed.prepare(raw, lines, thresh)
    if verbose and dropped:
        print(f"[decontaminate] dropped {dropped:,} story/stories "
              f"({dbytes:,} B) at threshold {thresh}")
    return kept, dropped, dbytes


def filter_file(path, lines, out_path=None, thresh=fed.DUP_THRESH, verbose=True):
    raw = open(path, "rb").read()
    kept, dropped, dbytes = filter_corpus(raw, lines, thresh, verbose)
    if out_path:
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(kept)
        os.replace(tmp, out_path)
    return {"in_bytes": len(raw), "kept_bytes": len(kept),
            "dropped_stories": dropped, "dropped_bytes": dbytes,
            "kept_fraction": len(kept) / max(1, len(raw)), "out": out_path}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", nargs="+", help="generated corpus file(s) to filter")
    ap.add_argument("--out-dir", default=None, help="write filtered copies here")
    ap.add_argument("--thresh", type=float, default=fed.DUP_THRESH)
    ap.add_argument("--train-glob", default=None,
                    help="override DEFAULT_GLOBS with a single glob (the old "
                         "training-only behaviour, for comparison)")
    args = ap.parse_args()

    globs = (args.train_glob,) if args.train_glob else DEFAULT_GLOBS
    lines = membership_lines(globs)
    total = {"in_bytes": 0, "kept_bytes": 0, "dropped_stories": 0, "dropped_bytes": 0}
    for p in args.corpus:
        out = os.path.join(args.out_dir, os.path.basename(p)) if args.out_dir else None
        r = filter_file(p, lines, out, args.thresh)
        print(f"[decontaminate] {p}: kept {r['kept_fraction']:.4f} "
              f"({r['kept_bytes']:,}/{r['in_bytes']:,} B), "
              f"dropped {r['dropped_stories']:,} stories -> {out or '(dry run)'}")
        for k in total:
            total[k] += r[k]
    print(f"[decontaminate] TOTAL kept {total['kept_bytes'] / max(1, total['in_bytes']):.4f}, "
          f"dropped {total['dropped_stories']:,} stories, "
          f"{total['dropped_bytes']:,} B")