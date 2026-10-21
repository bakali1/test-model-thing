#!/usr/bin/env python3
"""ngram_ce.py -- the byte n-gram baseline, rebuilt and made byte-matched.

The project has quoted "an order-8 byte n-gram (0 parameters, 8 MB of
training) scores 0.81 nats/byte on TMT's held-out set, against TMT's
1.2094" since 2026-10-05 (STRATEGIC-REVIEW-2026-10-05.md section 1; its
section 7 says the numbers came from `lag5.py`).  `lag5.py` was never
committed and no longer exists, so the number could not be reproduced or
checked -- and the comparison was not byte-matched to the ruler's held-out
set either.

This is the rebuild.  It fixes two things:

  1. it is REPRODUCIBLE: the script is in the repo and the protocol is
     documented in --help and in the docstring;
  2. it is BYTE-MATCHED: `--held-out ruler` scores exactly the bytes the
     model ruler scores (tools/eval.py), so the two numbers can be compared
     without a corpus argument.

PROTOCOL (the "ruler" mode, the one that matters)
  train     the first 8,000,000 bytes of data/eval/wiki_tinystories_005.txt
  held-out  for EACH file matched by glob data/eval/**/* (two files):
              3 disjoint windows of 20,001 bytes, stride (len(raw)-20000)//3
              (tools/eval.py:_slices, n_slices=3, n_bytes=20000)
            each window is scored as the mean of -log P(byte[t+1] | byte[t])
            over its 20,000 positions
            the file's score is the UNWEIGHTED MEAN of its 3 window scores
            the final score is the UNWEIGHTED MEAN of the per-file scores
            (the ruler prints the same unweighted mean: tools/eval.py:358)
  NOTE: the n-gram's own training file is also one of the model's scored
  eval files, which the original protocol also had.

`--held-out legacy` reproduces the ORIGINAL protocol, for comparison with
the 0.81 figure: the first 300,000 bytes of data/eval/tinystories_valid.txt
as one segment (STRATEGIC-REVIEW section 7).

THE MODEL: Jelinek-Mercer interpolation with a geometrically falling
order-dependent weight, `lambda_k = 1 - r**(k-1)`, so lambda_1 = 0 (pure
unigram) and higher orders take over gradually:

    P_k(w | c) = (1 - lambda_k) * P_{k-1}(w | c') + lambda_k * count(c,w)/count(c)

where c' is the (k-1)-gram context of c.  If the pair (c,w) never
occurred the first term survives alone; if the context c itself never
occurred, the model drops an order.  Absolute discounting is available as
`--method dd`.

COUNTING IS EXACT: an n-gram is keyed by its BYTES, packed big-endian into
a uint64 (an order-8 gram is exactly 8 bytes, so the key is lossless; for
orders < 8 the gram is left-aligned with zeros, which is still injective).
There is no rolling hash and no aliasing.  `--selftest` checks the packing
against a pure-Python dict of byte tuples.

ORDERS: 1..8 are supported exactly and fast.  The original script is
documented as MAXORD=12 (STRATEGIC-REVIEW section 7); orders 9..12 are NOT
reproduced here, because an order-9..12 gram does not fit in a uint64 and
the exact keying would need a slower path.  The consequence is stated
plainly in the report: an order-8 reconstruction is a WEAKER baseline than
the original order-12 model, so if it still beats the model, the original
claim survives; if it does not, the original protocol has to be recovered
before the claim can be refuted.

Run:
    .venv/bin/python tools/ngram_ce.py
    .venv/bin/python tools/ngram_ce.py --held-out ruler --max-order 8
    .venv/bin/python tools/ngram_ce.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN = os.path.join(REPO, "data/eval/wiki_tinystories_005.txt")
DEFAULT_EVAL_DIR = os.path.join(REPO, "data/eval")
TRAIN_BYTES = 8_000_000
LEGACY_HELD_OUT = 300_000
VOCAB = 256
MAX_EXACT_ORDER = 8


# --------------------------------------------------------------------------
# exact n-gram keying
# --------------------------------------------------------------------------
def gram_key(buf: np.ndarray, order: int) -> np.ndarray:
    """uint64 key per position: the order-gram ending at that position,
    left-aligned big-endian, zeros below.  Injective for order <= 8."""
    n = len(buf)
    key = np.zeros(n, dtype=np.uint64)
    ar = np.arange(n)
    for i in range(order):                 # i=0 is the OLDEST byte of the gram
        pos = ar - (order - 1) + i
        ok = pos >= 0
        vals = np.zeros(n, dtype=np.uint64)
        vals[ok] = buf[pos[ok]].astype(np.uint64)
        key |= vals << np.uint64(8 * (7 - i))
    return key


def pair_key(context_key: np.ndarray, sym: np.ndarray, order: int) -> np.ndarray:
    """the key of the (context, symbol) gram of the given order."""
    return context_key | (sym.astype(np.uint64) << np.uint64(64 - 8 * order))


def table_of(keys: np.ndarray):
    """(sorted unique keys, counts) over the valid positions."""
    uniq, counts = np.unique(keys, return_counts=True)
    return uniq, counts.astype(np.float64)


def count_of(uniq: np.ndarray, counts: np.ndarray, key: np.ndarray) -> np.ndarray:
    """count of each key (0 if absent)."""
    pos = np.searchsorted(uniq, key)
    posc = np.clip(pos, 0, len(uniq) - 1)
    ok = (pos < len(uniq)) & (uniq[posc] == key)
    return np.where(ok, counts[posc], 0.0)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score_sequence(tables, buf: np.ndarray, r: float, method: str,
                   discount: float) -> float:
    """mean -log P(buf[t+1] | buf[:t+1]) over t = 0 .. len(buf)-2."""
    n = len(buf)
    if n < 2:
        return float("nan")
    K = len(tables) - 1                     # tables[0] is a placeholder
    cnt_all = np.bincount(buf, minlength=VOCAB).astype(np.float64)
    uni = (cnt_all + 1.0) / (cnt_all.sum() + VOCAB)     # add-one smoothing

    ek = [None] + [gram_key(buf, j) for j in range(1, K + 1)]
    target = buf[1:]
    nt = len(target)

    # f[j] is the modelled PROBABILITY of the target byte, for every
    # position t (index t of the array means "the target after buf[t]").
    # The interpolation is in probability space; the log is taken at the end.
    f = {1: uni[buf[1:]]}
    for j in range(2, K + 1):
        uniq_j, cnt_j = tables[j]
        uniq_p, cnt_p = tables[j - 1]
        lam = (1.0 - r ** (j - 1)) if method == "jm" else 1.0
        # the context of the j-gram at position t is the (j-1)-gram at t
        c_key = ek[j - 1][:-1]                           # at t = 0..n-2
        ctx_cnt = count_of(uniq_p, cnt_p, c_key)
        pk = pair_key(c_key, target, j)
        pair = count_of(uniq_j, cnt_j, pk)
        back = np.empty(nt)
        back[0] = f[j - 1][0]                            # f_{j-1} at t-1
        back[1:] = f[j - 1][:-1]
        if method == "jm":
            term = np.where(ctx_cnt > 0, pair / np.maximum(ctx_cnt, 1), 0.0)
            val = (1.0 - lam) * back + lam * term
        else:                                            # absolute discounting
            d = discount
            disc = np.maximum(pair - d, 0.0)
            alpha = d / np.maximum(ctx_cnt, 1)
            val = disc / np.maximum(ctx_cnt, 1) + alpha * back
        val = np.where(ctx_cnt > 0, val, f[j - 1])       # context unseen -> drop
        f[j] = val

    # the highest order whose gram was actually seen at that position
    best = np.ones(nt, dtype=np.int64)
    for j in range(2, K + 1):
        uniq_j, cnt_j = tables[j]
        seen = count_of(uniq_j, cnt_j, ek[j][:-1]) > 0
        best = np.where(seen, j, best)
    out = np.empty(nt)
    for j in range(1, K + 1):
        sel = best == j
        if sel.any():
            out[sel] = f[j][sel]
    return float(-np.log(np.maximum(out, 1e-300)).mean())


def ruler_held_out():
    """the exact slices tools/eval.py scores, as (name, bytes) pairs."""
    out = []
    files = sorted(glob.glob(os.path.join(DEFAULT_EVAL_DIR, "**", "*"),
                             recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    n_slices, n_bytes = 3, 20000
    for f in files:
        raw = np.frombuffer(open(f, "rb").read(), np.uint8)
        if len(raw) <= n_bytes + 1:
            sl = [raw[:len(raw) - 1]]
        else:
            stride = (len(raw) - n_bytes) // n_slices
            sl = [raw[i * stride: i * stride + n_bytes + 1] for i in range(n_slices)]
        for i, s in enumerate(sl):
            out.append((f"{os.path.basename(f)}[{i}]", s))
    return out


def build_tables(train: np.ndarray, max_order: int):
    tables = [None]
    for j in range(1, max_order + 1):
        keys = gram_key(train, j)[j - 1:]               # valid positions only
        tables.append(table_of(keys))
    return tables


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--held-out", choices=("ruler", "legacy", "both"),
                    default="both")
    ap.add_argument("--max-order", type=int, default=8)
    ap.add_argument("--r", type=float, default=0.7)
    ap.add_argument("--method", choices=("jm", "dd"), default="jm")
    ap.add_argument("--discount", type=float, default=0.75)
    ap.add_argument("--train-bytes", type=int, default=TRAIN_BYTES)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.max_order > MAX_EXACT_ORDER:
        print(f"WARNING: only orders 1..{MAX_EXACT_ORDER} are exact and fast; "
              f"--max-order {args.max_order} will be clamped.", file=sys.stderr)
        args.max_order = MAX_EXACT_ORDER

    t0 = time.time()
    train = np.frombuffer(open(args.train, "rb").read(args.train_bytes), np.uint8)
    print(f"[train] {args.train}  {len(train):,} bytes  ({time.time()-t0:.1f}s)",
          flush=True)

    sets = []
    if args.held_out in ("ruler", "both"):
        sets += ruler_held_out()
    if args.held_out in ("legacy", "both"):
        raw = np.frombuffer(
            open(os.path.join(DEFAULT_EVAL_DIR, "tinystories_valid.txt"),
                 "rb").read(LEGACY_HELD_OUT), np.uint8)
        sets.append(("tinystories_valid[0:300k]  (LEGACY)", raw))

    tables = build_tables(train, args.max_order)
    print(f"[build] tables for orders 1..{args.max_order}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    for method in (args.method,):
        print(f"\n=== method={method}  r={args.r}  discount={args.discount}"
              f"  max_order={args.max_order} ===", flush=True)
        ruler_ces, legacy_ces = [], []
        ruler_files = {}
        for name, buf in sets:
            t1 = time.time()
            ce = score_sequence(tables, buf, args.r, method, args.discount)
            print(f"  {name:<40} CE {ce:.4f} nats/byte   "
                  f"({len(buf)-1:,} positions, {time.time()-t1:.1f}s)", flush=True)
            if "LEGACY" in name:
                legacy_ces.append(ce)
            else:
                ruler_ces.append((name.split("[")[0], ce))
        if ruler_ces:
            # the ruler's score is the unweighted mean of the per-FILE means,
            # and each file's mean is the unweighted mean of its 3 slices
            by_file = {}
            for f, ce in ruler_ces:
                by_file.setdefault(f, []).append(ce)
            per_file = {f: float(np.mean(v)) for f, v in by_file.items()}
            agg = float(np.mean(list(per_file.values())))
            print(f"  {'RULER AGGREGATE (byte-matched)':<40} CE {agg:.4f} "
                  f"nats/byte   ({len(ruler_ces)} slices over "
                  f"{len(per_file)} files)", flush=True)
            for f, ce in sorted(per_file.items()):
                print(f"      {f:<36} {ce:.4f}")
            print(f"  {'TMT model, same ruler held-out':<40} CE 1.2094 nats/byte"
                  f"   -> the n-gram is {1.2094 - agg:.4f} nats/byte BETTER",
                  flush=True)
        if legacy_ces:
            print(f"  {'LEGACY AGGREGATE (300k head)':<40} CE "
                  f"{float(np.mean(legacy_ces)):.4f} nats/byte", flush=True)
    print(f"\n[total {time.time()-t0:.1f}s]")
    return 0


def selftest():
    """the uint64 keying must agree exactly with a dict of byte tuples."""
    rng = np.random.default_rng(0)
    buf = rng.integers(0, 256, size=200_000, dtype=np.uint8)
    for order in range(1, MAX_EXACT_ORDER + 1):
        key = gram_key(buf, order)[order - 1:]
        uniq, inv = np.unique(key, return_inverse=True)
        np_counts = np.bincount(inv, minlength=len(uniq))
        py = {}
        for p in range(order - 1, len(buf)):
            t = tuple(buf[p - order + 1: p + 1].tolist())
            py[t] = py.get(t, 0) + 1
        assert len(py) == len(uniq), (order, len(py), len(uniq))
        # unpack every key and check the count agrees
        bad = 0
        for i, k in enumerate(uniq):
            b = int(k).to_bytes(8, "big")
            t = tuple(b[:order])
            if py.get(t, 0) != np_counts[i]:
                bad += 1
        assert bad == 0, f"order {order}: {bad} disagreements"
        print(f"[selftest] order {order}: {len(uniq):,} distinct, counts agree "
              f"exactly with the pure-Python dict")
    print("[selftest] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
