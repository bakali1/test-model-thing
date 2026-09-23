#!/usr/bin/env python3
"""Fetch and prepare the held-out eval corpora (PLAN Phase 0.1).

Downloads the canonical TinyStories validation splits, strips the
`<|endoftext|>` markers (the training corpus has none -- 0 occurrences in the
local wikistories files), and removes stories that are near-duplicates of a
training story. The latter matters: TinyStories is heavily templated, so the
official split still shares ~3% of stories with a 1.9 GB train set, and
scoring those measures memorization.

    .venv/bin/python fetch_eval_data.py

Outputs (not tracked by git -- data/ is untracked; run this on each platform):

    data/eval/tinystories_valid.txt          TinyStories v1, in-domain held-out
    data/eval_shift/tinystories_v2gpt4_valid.txt  TinyStoriesV2-GPT4, shift

The near-duplicate test is story-level: drop the story if >= DUP_THRESH of its
lines of >= 40 bytes already appear in the training corpus.
"""
import glob
import os
import sys
import urllib.request

HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_GLOB = "data/wikipedia_clean/**/wiki_*"
DUP_THRESH = 0.6
JOBS = [
    ("TinyStories-valid.txt", "data/eval/tinystories_valid.txt"),
    ("TinyStoriesV2-GPT4-valid.txt", "data/eval_shift/tinystories_v2gpt4_valid.txt"),
]


def train_lines():
    files = sorted(glob.glob(TRAIN_GLOB, recursive=True))
    if not files:
        print(f"[warn] no training files match {TRAIN_GLOB!r}; cannot dedup",
              file=sys.stderr)
        return set()
    blob = b"".join(open(f, "rb").read() for f in files)
    lines = set(blob.split(b"\n"))
    print(f"training: {len(files)} file(s), {len(blob):,} B, "
          f"{len(lines):,} distinct lines")
    return lines


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tmt-eval-fetch/1"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return os.path.getsize(dest)


def prepare(raw, tlines, thresh=DUP_THRESH):
    raw = raw.replace(b"<|endoftext|>", b"")      # match the training format
    kept, dropped, dbytes = [], 0, 0
    for seg in raw.split(b"\n\n"):
        core = seg.strip(b"\n")
        long = [l for l in core.split(b"\n") if len(l) >= 40]
        if long and tlines:
            hit = sum(1 for l in long if l in tlines)
            if hit / len(long) >= thresh:
                dropped += 1
                dbytes += len(seg) + 2
                continue
        kept.append(seg)
    return b"\n\n".join(kept), dropped, dbytes


def main():
    tlines = train_lines()
    for name, dest in JOBS:
        tmp = "/tmp/" + name
        n = download(HF + name, tmp)
        out, dropped, dbytes = prepare(open(tmp, "rb").read(), tlines)
        open(dest, "wb").write(out)
        print(f"{dest}\n  {n:,} B downloaded -> {len(out):,} B "
              f"(dropped {dropped} near-duplicate stories, {dbytes:,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
