#!/usr/bin/env python3
"""Unified held-out ruler for numpy_port checkpoints (PLAN Phase 0.4).

One command prints everything a result needs:

  * held-out CE, per slice and as a mean +/- SE across disjoint slices
    (multiple slices kill snippet luck; a win inside one SE is not a win)
  * unigram/bigram baselines built from the training corpus
  * cross-domain CE on the Gutenberg novels (out-of-domain generalization)
  * recurrent health: decay spread, half-life histogram, hot-dim count
  * stop-head calibration: AUC vs line-ends, recall/precision at thresholds
  * distance curve (Karpathy 2015 Fig. 5): P(correctly closing a byte) as a
    function of the distance back to its matching open byte. Opt-in
    (--dc-bytes N); reads the SAME held-out files and the SAME forward-pass
    convention as the CE cells above.

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

# The engine lives at the repo root; this file lives in tools/, so make the
# root importable no matter which directory it is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import numpy_port as N
from eval_npz import (auc_score, cross_entropy, infer_config, line_end_labels)

TRAIN_GLOB_DEFAULT = "data/wikipedia_clean/**/wiki_*"
EVAL_GLOB_DEFAULT = "data/eval/**/*"
SHIFT_GLOB_DEFAULT = "data/eval_shift/**/*"
NOVEL_GLOB_DEFAULT = "data/*.txt"

# --- distance-curve cell (Karpathy 2015 S4.2, Fig. 5) --------------------
# Delimiter classes as (class name, open byte, close byte). Kept explicit
# because the brace half of the 2015 anchor has 0-5 occurrences per 18-41 MB
# file in this corpus: that is reported as "no events", which is a different
# statement from a measured null and must not read like one.
DC_CLASSES = [("quote", 0x22, 0x22), ("apos", 0x27, 0x27),
              ("paren", 0x28, 0x29), ("brack", 0x5B, 0x5D),
              ("brace", 0x7B, 0x7D)]
# Distance bins in BYTES (the model's unit). Fixed edges so two runs are
# comparable without re-deriving bins from the data.
DC_BINS = [(1, 8), (8, 32), (32, 64), (64, 128), (128, 256), (256, 512),
           (512, 1024), (1024, float("inf"))]
DC_SCOPE_NOTE = (
    "SCOPE - read before reading a null here: the existing needle probe "
    "(#23) tested literal-cue induction only. Obfuscated and parallel-"
    "tracking retrieval were never tested. So this cell measures one thing, "
    'and its null does not generalize to "no memory."')
DC_PARITY_NOTE = (
    "CONFOUND, stated with the curve because it changes what a HIGH lift "
    "means: same-byte classes (quote, apostrophe) strictly alternate in this "
    "corpus, so a model that has only counted opens and closes since the last "
    "byte can score well here without holding any content. This cell measures "
    "parity tracking at minimum, which is the cheapest form of memory there "
    "is. The two-byte classes (paren/bracket/brace), where alternation gives "
    "nothing, are the ones that would test content -- and they have ~0 events "
    "in this corpus.")
DC_ANCHOR = (
    "anchor, Karpathy 2015 S4.2 / Fig. 5 (arXiv:1506.02078v2): a 1M-param "
    "LSTM trained with truncated BPTT-100 + RMSProp tracks ~230-char quotes "
    "and boosts closing-brace accuracy to ~60 chars. The honest null for "
    "this cell is BELOW that 2015 LSTM, not at it.")


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


def _dc_match(raw):
    """Matched (open, close) index pairs, per delimiter class.

    Same-byte classes (quote, apostrophe) are matched by alternation; a
    two-byte class (paren/bracket/brace) is matched with a stack, nearest
    open first. Closes with no open are COUNTED rather than dropped: an
    unmatched close means the pairing assumption failed on this text, which is
    a fact about the corpus the cell has to report, not an error to hide.
    """
    out = {}
    for name, ob, cb in DC_CLASSES:
        if ob == cb:
            ev = [(int(i), k % 2) for k, i in enumerate(np.flatnonzero(raw == ob))]
        else:
            ev = [(int(i), 0) for i in np.flatnonzero(raw == ob)]
            ev += [(int(i), 1) for i in np.flatnonzero(raw == cb)]
            ev.sort()
        stack, pairs, unmatched = [], [], 0
        for pos, kind in ev:
            if kind == 0:
                stack.append(pos)
            elif stack:
                pairs.append((stack.pop(), pos))
            else:
                unmatched += 1
        out[name] = (pairs, unmatched, len(ev))
    return out


def _dc_curve(m, files, n_bytes, batch):
    """P(correctly closing a byte | distance back to its matching open byte).

    Teacher-forced on held-out bytes: at each position the model emits a 256-way
    byte distribution, and we read off the probability of the byte that really
    came next. A pair (open at o, close at c) contributes P(byte c | bytes
    [0..c-1]) at distance d = c - o. The window is the FIRST n_bytes of each
    held-out file, concatenated -- fixed rather than sampled, because two arms
    of an A/B must be scored on the SAME pairs and the pairs are what the bin
    means are taken over.

    Streams are contiguous chunks, not strided, because a pair's open and
    close must be fed through the SAME recurrent state.

    Returns a dict of raw per-pair probabilities (plus the floor), so a caller
    can do a PAIRED arm-vs-arm difference on identical pairs instead of
    comparing two independent bin means.
    """
    parts, srcs = [], []
    for f in files:
        with open(f, "rb") as fh:
            parts.append(np.frombuffer(fh.read(n_bytes), np.uint8))
        srcs.append(os.path.basename(f))
    blob = np.concatenate(parts)
    keep = (len(blob) // batch) * batch
    truncated = len(blob) - keep
    blob = blob[:keep]
    L = keep // batch

    matched = _dc_match(blob)
    # step -> [(stream, pair index, closing byte)]: the step whose OUTPUT
    # distribution has to contain the closing byte. Keyed by step alone, so the
    # hot loop is one dict lookup per step instead of one per stream.
    want = {}
    skipped = 0
    cls_list, dist, slots = [], [], []
    for name, _ob, cb in DC_CLASSES:
        pairs, _unmatched, _events = matched[name]
        for o, c in pairs:
            stream, loc = divmod(c, L)
            if loc == 0:
                skipped += 1      # close is byte 0 of its stream: no prior state
                continue
            pidx = len(dist)
            dist.append(c - o)
            cls_list.append(name)
            slots.append(pidx)
            want.setdefault(loc - 1, []).append((stream, pidx, cb))
    vals = np.full(len(slots), np.nan)

    m.new_states(batch)
    marg = {name: 0.0 for name, _ob, _cb in DC_CLASSES}
    ce_sum = 0.0
    ce_n = 0
    n_pos = 0
    seq = blob.reshape(batch, L)              # (batch, L) time-minor
    for t in range(L):
        enc = m.embed[seq[:, t]]
        x = enc
        for l in m.layers:
            x, st, _d, _c = l.forward(enc, x, l.states)
            l.states = st
        logits = x @ m.Wd.T + m.bd
        mx = logits.max(axis=1)
        lse = mx + np.log(np.exp(logits - mx[:, None]).sum(axis=1))
        p = np.exp(logits - lse[:, None])
        for name, _ob, cb in DC_CLASSES:
            marg[name] += float(p[:, cb].sum())
        if t + 1 < L:
            # CE is logsumexp MINUS the logit of the byte that actually came
            # next. Reporting the logsumexp alone would be reporting the
            # partition function (~14.7 nats on this checkpoint) and calling
            # it a loss, which is the kind of definitional slip this repo
            # already paid for once with coverage.
            ce_sum += float((lse - logits[np.arange(batch), seq[:, t + 1]]).sum())
            ce_n += batch
        n_pos += batch
        for stream, pidx, byte in want.get(t, ()):
            vals[pidx] = p[stream, byte]
    m.new_states(1)
    return {
        "classes": sorted(set(cls_list)),
        "cls": np.array(cls_list),
        "dist": np.array(dist, float),
        "p": vals,
        "floor": {k: (v / max(1, n_pos)) for k, v in marg.items()},
        "n_positions": n_pos,
        "window_bytes": keep,
        "truncated_bytes": truncated,
        "stream_len": L,
        "batch": batch,
        "files": srcs,
        "skipped_boundary": skipped,
        "unmatched": {k: v[1] for k, v in matched.items() if v[2]},
        "events": {k: v[2] for k, v in matched.items()},
        "mean_nats_per_byte": ce_sum / max(1, ce_n),
    }


def _dc_bins(dc, cls):
    """(n, mean p_close, SE of the mean) per distance bin, for one class.

    SE is across PAIRS, not across positions: two pairs in one bin are two
    independent samples of "the model closed correctly at this distance", and
    a bin mean over ~10 pairs needs that error bar printed next to it or the
    curve invites reading noise as a decay.
    """
    msk = dc["cls"] == cls
    d, p = dc["dist"][msk], dc["p"][msk]
    rows = []
    for lo, hi in DC_BINS:
        sel = (d >= lo) & (d < hi)
        n = int(sel.sum())
        if n == 0:
            rows.append((lo, hi, 0, None, None))
            continue
        v = p[sel]
        mean = float(v.mean())
        se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else None
        rows.append((lo, hi, n, mean, se))
    return rows


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
    ap.add_argument("--dc-bytes", type=int, default=0,
                    help="distance-curve cell: held-out bytes per eval file "
                         "(0 = off). Costs real CPU time on this engine "
                         "(~370 B/s batched at --dc-batch 64), so it is "
                         "opt-in and its budget is printed with the curve.")
    ap.add_argument("--dc-batch", type=int, default=64,
                    help="independent contiguous streams for the "
                         "distance-curve cell; must divide dc-bytes*files")
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
    # An FSP arm's checkpoint carries the auxiliary head's 132,352
    # parameters, so the reference model must have the same layout --
    # otherwise the load fails on a parameter-count mismatch (which is
    # how the FSP arm's CE cell died: in the loader, not the measurement).
    _has_fsp = "fsp.W" in d.files
    m = N.Model(dim=dim, layers=layers, temp=0.75, lr=5e-4, fsp=_has_fsp)
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

    # ---- distance curve (Karpathy 2015 Fig. 5) ----
    # P(close the byte) against the distance back to its matching open byte.
    # The definition and the floor are printed with the numbers on purpose: a
    # bare probability here is a frequency, not a memory measurement.
    if args.dc_bytes > 0:
        print(f"\n{'=' * 74}")
        print(f"distance curve (Karpathy 2015 Fig. 5)  "
              f"[opt-in, --dc-bytes {args.dc_bytes:,}/file]")
        print(f"  DEFINITION: for every matched (open, close) delimiter pair in")
        print(f"  the held-out bytes, take P(byte c | bytes[0..c-1]) from the")
        print(f"  model's 256-way output at position c-1 -- i.e. the probability")
        print(f"  the model put on the byte that actually closed the pair --")
        print(f"  and bin it by d = c - o, the distance back to the opener.")
        print(f"  Teacher-forced on real text; no sampling, no generation.")
        print(f"  FLOOR: the model's MEAN probability of that same closing byte")
        print(f"  over every position scored. That is the no-cue level for THIS")
        print(f"  model on THIS text. A bin at its floor is not memory.")
        print(f"  {DC_SCOPE_NOTE}")
        print(f"  {DC_PARITY_NOTE}")
        print(f"  {DC_ANCHOR}")
        dc = _dc_curve(m, eval_files, args.dc_bytes, max(1, args.dc_batch))
        _drop = (f" ({dc['truncated_bytes']:,} B dropped to fit the batch)"
                 if dc["truncated_bytes"] else "")
        print(f"  window: {dc['window_bytes']:,} B = {args.dc_bytes:,} B x "
              f"{len(dc['files'])} file(s) [{', '.join(dc['files'])}]{_drop}")
        print(f"  streams: {dc['batch']} contiguous x {dc['stream_len']:,} B"
              f" | {dc['n_positions']:,} scored positions"
              f" | CE on this window {dc['mean_nats_per_byte']:.4f} nats/B")
        ev = ", ".join(f"{k}={dc['events'][k]}" for k, _a, _b in DC_CLASSES)
        print(f"  delimiter events found: {ev}")
        print(f"  unmatched closes (pairing assumption failed on this text): "
              f"{dc['unmatched'] or 'none'}; "
              f"pairs dropped at a stream boundary: {dc['skipped_boundary']}")
        for cls in dc["classes"]:
            fl = dc["floor"][cls]
            print(f"\n  --- class {cls} | no-memory floor (mean P of its closing "
                  f"byte over {dc['n_positions']:,} positions) = {fl:.6f}")
            print(f"      {'distance (B)':>14} {'n':>6} {'mean p_close':>14} "
                  f"{'SE':>9} {'floor':>9} {'lift':>8} {'>=1.96SE above floor'}")
            rows = _dc_bins(dc, cls)
            for lo, hi, n, mean, se in rows:
                if n == 0:
                    # Printed with full field padding, not "no events": the
                    # paired arm-vs-arm reader zips the two arms' bin lists by
                    # position, and a dropped row would silently shift every
                    # bin after it by one and compare the wrong distances.
                    print(f"      {f'[{lo},{hi if hi != float('inf') else 'inf'})':>14} "
                          f"{0:>6} {'(no events)':>14} {'n/a':>9} {'n/a':>9} "
                          f"{'n/a':>9}  n/a")
                    continue
                lift = mean / fl if fl > 0 else float("inf")
                sig = "yes" if (se is not None and mean - fl > 1.96 * se) else (
                    "no SE (n=1)" if se is None else "no")
                print(f"      {f'[{lo},{hi if hi != float('inf') else 'inf'})':>14} "
                      f"{n:>6} {mean:>14.6f} "
                      f"{'n/a' if se is None else f'{se:.6f}':>9} "
                      f"{fl:>9.6f} {lift:>7.1f}x  {sig}")
            # Crossover: the furthest REACHED bin, contiguously from the
            # shortest distance, whose mean is still above the floor by more
            # than its own SE. Stated as a rule because "where the curve falls
            # off" is otherwise a judgement call wearing a number. A bin with
            # fewer than 3 pairs cannot qualify: its SE is the measurement.
            MIN_N = 3
            xover, xn, stopped = 0, 0, None
            for lo, hi, n, mean, se in rows:
                if n == 0:
                    continue                     # absent measurement, not a null
                if n >= MIN_N and se is not None and mean - fl > 1.96 * se:
                    xover, xn = hi, n
                else:
                    stopped = (lo, hi, n, mean, se)
                    break
            stop_txt = ("" if stopped is None else
                        f"; stopped at [{stopped[0]},{stopped[1]}) n={stopped[2]}"
                        f"{'' if stopped[2] >= MIN_N else f' (<{MIN_N} pairs: no SE)'}")
            print(f"      DC-XOVER {cls}: furthest contiguously-reached bin with "
                  f"n>={MIN_N} whose mean is >=1.96 SE above the no-memory floor"
                  f" -> up to d < {xover}"
                  f"{' (NO bin qualifies)' if not xover else f' (n={xn} there)'}"
                  f"{stop_txt}")
            print(f"      DC-PAIRS {cls}: {int(np.sum([r[2] for r in rows]))}")
        for name, _ob, _cb in DC_CLASSES:
            if name not in dc["classes"]:
                print(f"\n  --- class {name}: NO EVENTS in the scored window "
                      f"(0 of the {dc['window_bytes']:,} bytes). The Karpathy "
                      f"brace/bracket half of the anchor is NOT MEASURED here; "
                      f"that is an absent measurement, not a null.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
