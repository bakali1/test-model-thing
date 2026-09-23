#!/usr/bin/env python3
"""Unified held-out ruler for numpy_port checkpoints (PLAN Phase 0.4).

One command prints everything a result needs:

  * held-out CE, per slice and as a mean +/- SE across disjoint slices
    (multiple slices kill snippet luck; a win inside one SE is not a win)
  * unigram/bigram baselines built from the training corpus
  * cross-domain CE on the Gutenberg novels (out-of-domain generalization)
  * recurrent health: decay spread, half-life histogram, hot-dim count
  * stop-head calibration: AUC vs line-ends, recall/precision at thresholds

Hard guard (Phase 0.1): if any eval file is also reachable by the training
glob, the script refuses to run -- an eval on trained bytes is worse than no
eval.

    .venv/bin/python eval.py tmt.npz
    .venv/bin/python eval.py tmt.npz --slices 3 --bytes 20000
    .venv/bin/python eval.py tmt.npz --train-glob 'data/wikipedia_clean/**/wiki_*'
"""
import argparse
import glob
import os
import sys

import numpy as np

import numpy_port as N
from eval_npz import (auc_score, cross_entropy, infer_config, line_end_labels)

TRAIN_GLOB_DEFAULT = "data/wikipedia_clean/**/wiki_*"
EVAL_GLOB_DEFAULT = "data/eval/**/*"
SHIFT_GLOB_DEFAULT = "data/eval_shift/**/*"
NOVEL_GLOB_DEFAULT = "data/*.txt"


def _files(pattern):
    return sorted(f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f))


def _check_no_overlap(train_files, eval_files):
    tr = {os.path.realpath(f) for f in train_files}
    ov = sorted(t for t in (os.path.realpath(f) for f in eval_files) if t in tr)
    if ov:
        print("[FATAL] eval files are reachable by the training glob:", file=sys.stderr)
        for f in ov:
            print(f"    {f}", file=sys.stderr)
        print("reserving bytes you also train on measures memorization, not "
              "generalization. Move them out of the training glob.", file=sys.stderr)
        return False
    return True


def _slices(raw, n_slices, n_bytes):
    """n_slices disjoint windows spread across `raw`."""
    if len(raw) <= n_bytes + 1:
        return [raw[:len(raw) - 1]] if len(raw) > 1 else []
    stride = (len(raw) - n_bytes) // n_slices
    out = []
    for i in range(n_slices):
        start = i * stride
        out.append(raw[start:start + n_bytes + 1])
    return out


def _baselines(train_files, per_file=2_000_000):
    """Unigram/bigram counts from the training corpus (never eval bytes)."""
    uni = np.ones(256)
    bg = np.ones((256, 256))
    for f in train_files:
        with open(f, "rb") as fh:
            data = np.frombuffer(fh.read(per_file), np.uint8)
        uni += np.bincount(data, minlength=256)
        np.add.at(bg, (data[:-1], data[1:]), 1.0)
    uni /= uni.sum()
    bg /= bg.sum(axis=1, keepdims=True)
    return uni, bg


def _eval_corpus(m, files, n_slices, n_bytes):
    """Score `n_slices` disjoint windows of each file. Returns
    (per_slice_ce, per_slice_src, total_bytes, stops, labels)."""
    ces, srcs, stops, labs, total = [], [], [], [], 0
    for f in files:
        raw = np.frombuffer(open(f, "rb").read(), np.uint8)
        for seq in _slices(raw, n_slices, n_bytes):
            ce, st, _ = cross_entropy(m, seq, return_per_byte=True)
            ces.append(ce)
            srcs.append(f)
            stops.append(st)
            labs.append(line_end_labels(seq))
            total += len(seq) - 1
    ces = np.array(ces)
    stops = np.concatenate(stops) if stops else np.array([])
    labs = np.concatenate(labs) if labs else np.array([])
    return ces, srcs, total, stops, labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--train-glob", default=TRAIN_GLOB_DEFAULT)
    ap.add_argument("--eval-glob", default=EVAL_GLOB_DEFAULT)
    ap.add_argument("--shift-glob", default=SHIFT_GLOB_DEFAULT)
    ap.add_argument("--novel-glob", default=NOVEL_GLOB_DEFAULT)
    ap.add_argument("--slices", type=int, default=3,
                    help="disjoint held-out slices per file (SE comes from these)")
    ap.add_argument("--bytes", type=int, default=20000, help="bytes per slice")
    ap.add_argument("--xd-bytes", type=int, default=8000,
                    help="bytes per cross-domain (novel) slice; 0 disables")
    ap.add_argument("--xd-files", type=int, default=0,
                    help="limit novels scored (0 = all)")
    ap.add_argument("--threshold", type=float, default=0.35)
    args = ap.parse_args()

    train_files = _files(args.train_glob)
    eval_files = _files(args.eval_glob)
    if not eval_files:
        print(f"[FATAL] no held-out files match {args.eval_glob!r}", file=sys.stderr)
        return 1
    if not _check_no_overlap(train_files, eval_files):
        return 1

    d = np.load(args.ckpt)
    dim, layers = infer_config(d)
    m = N.Model(dim=dim, layers=layers, temp=0.75, lr=5e-4)
    m.new_states(1)
    m.load(args.ckpt)          # honors the checkpoint's ring_k
    K = m.ring_k
    m.new_states(1)            # zero memory: score the weights, not one stream

    print(f"checkpoint : {args.ckpt}")
    print(f"params     : {m.count():,}  (dim={dim}, layers={layers})")
    print(f"ring       : K={K} lags (read from checkpoint)")
    if "bytes" in d:
        print(f"steps (t)  : {int(d['t']):,}   bytes seen {int(d['bytes']):,} (exact)")
    else:
        print(f"steps (t)  : {int(d['t']):,}   bytes seen ~= {int(d['t']) * int(d['B']):,}")

    print(f"\ntrain glob : {args.train_glob}  ({len(train_files)} file(s), "
          f"{sum(os.path.getsize(f) for f in train_files) / 1e6:.0f} MB)")
    print(f"eval files : {len(eval_files)} held-out file(s), no overlap")

    uni, bg = _baselines(train_files)

    # ---- held-out (in-domain) ----
    ces, srcs, n, stops, labs = _eval_corpus(
        m, eval_files, args.slices, args.bytes)
    if len(ces) == 0:
        print("\n[FATAL] held-out files too small to slice", file=sys.stderr)
        return 1
    ce = float(ces.mean())
    se = float(ces.std(ddof=1) / np.sqrt(len(ces))) if len(ces) > 1 else float("nan")
    print(f"\nheld-out CE (nats/byte; {len(ces)} slices, {n:,} B):")
    for f, c in zip(srcs, ces):
        print(f"  {os.path.basename(f):<32} {c:.4f}")
    print(f"  {'mean':<32} {ce:.4f} +/- {se:.4f}  (SE across {len(ces)} slices)")
    print(f"  perplexity {np.exp(ce):.2f}")

    # baselines on the first held-out slice for a same-bytes comparison
    raw0 = np.frombuffer(open(eval_files[0], "rb").read(), np.uint8)
    seq0 = _slices(raw0, args.slices, args.bytes)[0]
    ce_uni = float(np.mean([-np.log(uni[b]) for b in seq0[1:]]))
    ce_bg = float(np.mean([-np.log(bg[a, b]) for a, b in zip(seq0[:-1], seq0[1:])]))
    print(f"  unigram {ce_uni:.4f} | bigram {ce_bg:.4f} | model {ce_bg - ce:.4f} vs bigram")

    # ---- near-domain shift (same task, independent generation) ----
    shift_files = _files(args.shift_glob)
    if shift_files:
        sces, _, _, _, _ = _eval_corpus(m, shift_files, args.slices, args.bytes)
        sce = float(sces.mean())
        sse = (float(sces.std(ddof=1) / np.sqrt(len(sces)))
               if len(sces) > 1 else float("nan"))
        print(f"\nshift-distribution CE ({len(shift_files)} file(s), "
              f"{len(sces)} slices):")
        print(f"  mean {sce:.4f} +/- {sse:.4f}  (in-domain held-out {ce:.4f}; "
              f"gap {sce - ce:+.4f})")

    # ---- cross-domain (novels) ----
    if args.xd_bytes > 0:
        novel_files = _files(args.novel_glob)
        if args.xd_files:
            novel_files = novel_files[:args.xd_files]
        ntr = {os.path.realpath(f) for f in train_files}
        seen = [f for f in novel_files if os.path.realpath(f) in ntr]
        if seen:
            print(f"\n[warn] {len(seen)} novel file(s) are in the training glob; "
                  f"cross-domain is NOT held out")
        xce_s, _, _, _, _ = _eval_corpus(m, novel_files, 1, args.xd_bytes)
        xce = float(xce_s.mean()) if len(xce_s) else float("nan")
        xse = (float(xce_s.std(ddof=1) / np.sqrt(len(xce_s)))
               if len(xce_s) > 1 else float("nan"))
        print(f"\ncross-domain CE ({len(novel_files)} novel(s), ~{args.xd_bytes} B each):")
        print(f"  mean {xce:.4f} +/- {xse:.4f}  (in-domain held-out {ce:.4f})")
        print(f"  gap  {xce - ce:+.4f} nats/byte (out-of-domain cost)")

    # ---- recurrent memory health ----
    sig = 1.0 / (1.0 + np.exp(-np.stack([d[f"layer{i}.decay"] for i in range(layers)])))
    flat = sig.ravel()
    hl = np.log(0.5) / np.log(flat)
    hot = int((flat ** K > 0.01).sum())
    print(f"\ndecay sigma: min={flat.min():.4f} mean={flat.mean():.4f} "
          f"max={flat.max():.4f} std={flat.std():.4f}")
    B_ref = int(d["B"]) if "B" in d else 1024
    print(f"half-life   mean={hl.mean():.2f} B | hot dims (sigma^K>1%): {hot} "
          f"| tail ~{hot * B_ref * 256 * 4 / 1e6:.1f} MB at B={B_ref}")
    bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, float("inf"))]
    hist = " | ".join(f"{lo}-{hi if hi != float('inf') else 'inf'}: "
                      f"{int(((hl >= lo) & (hl < hi)).sum())}" for lo, hi in bins)
    print(f"half-life hist (<1 B ... >32 B): {hist}")

    # ---- stop-head calibration ----
    if len(labs):
        lab = labs
        pos, neg = stops[lab], stops[~lab]
        print(f"\nstop head ({lab.sum()} line-ends / {len(lab)} positions, "
              f"{100 * lab.mean():.2f}% positive):")
        if len(pos):
            print(f"  mean at line ends {pos.mean():.4f} | mid-line {neg.mean():.4f} "
                  f"| AUC {auc_score(stops, lab):.4f}")
        for thr in (0.1, 0.2, 0.35, 0.5):
            rec = float((pos > thr).mean()) if len(pos) else float("nan")
            sel = stops > thr
            pre = float(lab[sel].mean()) if sel.any() else float("nan")
            print(f"  thr {thr:<4} recall {100 * rec:5.1f}%  precision {100 * pre:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
