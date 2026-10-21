#!/usr/bin/env python3
"""Held-out evaluation for a numpy_port checkpoint.

Reports cross-entropy against unigram/bigram baselines so training progress is
a number instead of a vibe, plus the recurrent-health stats that matter for
this architecture (decay spread, stop-head calibration).

    .venv/bin/python eval_npz.py nlt-0.1.npz
    .venv/bin/python eval_npz.py nlt-0.1.npz --bytes 4000 --file 005
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

HELDOUT_START = 5_000_000     # every train run reads files from byte 0
BASELINE_SLICE = (10_000_000, 12_000_000)


def infer_config(d):
    dim = d["Wd"].shape[1]
    layers = sum(1 for k in d.files if k.startswith("layer") and k.endswith(".W"))
    return dim, layers


def forward_logits(m, c):
    """One frozen forward step; returns (logits(256,), stop)."""
    enc = m.embed[c]
    x = enc[None, :]
    for l in m.layers:
        x, st, _, _ = l.forward(enc[None, :], x, l.states[0:1])
        l.states[0:1] = st
    return x[0] @ m.Wd.T + m.bd, float(N.sigmoid(x[0] @ m.Ws + m.bs)[0])


def cross_entropy(m, seq, return_per_byte=False):
    """Returns (mean CE, stops) or (mean CE, stops, per-byte CE array)."""
    m.new_states(1)
    tot, stops, per, n = 0.0, [], [], 0
    for c, nxt in zip(seq[:-1], seq[1:]):
        lg, s = forward_logits(m, int(c))
        v = lg.max() + np.log(np.exp(lg - lg.max()).sum()) - lg[nxt]
        tot += v
        per.append(v)
        stops.append(s)
        n += 1
    out = (tot / n, np.array(stops))
    if return_per_byte:
        out += (np.array(per),)
    return out


def line_end_labels(seq):
    """stop-head target for each position: True when seq[t+1] finishes a line.

    Matches the training target from Runtime.make_streams (the last byte-pair of
    each line), so this measures exactly what the head was asked to learn.
    """
    lab = np.zeros(len(seq) - 1, bool)
    lab[:-1] = seq[1:-1] == 0x0A        # next byte is a newline
    lab[-1] = True                      # final line may lack a trailing newline
    return lab


def auc_score(scores, labels):
    """P(score of a positive > score of a negative), ties counted as 0.5."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    r = np.arange(1, len(scores) + 1, dtype=float)
    s_sorted = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks[order] = r
    npos = int(labels.sum())
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--bytes", type=int, default=20000, help="held-out bytes to score")
    ap.add_argument("--file", default="005", help="wikistories file suffix for the held-out slice")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="chat stop threshold (Runtime default 0.35)")
    args = ap.parse_args()

    d = np.load(args.ckpt)
    dim, layers = infer_config(d)
    # build the eval model with the checkpoint's own ring K so the tail's
    # hot-set accounting during load matches what the trainer actually did
    if "ring_k" in d:
        N.RING_TOL = 0.5115 ** int(d["ring_k"])
    # An FSP arm's checkpoint carries the auxiliary head's 132,352
    # parameters, so the reference model must have the same layout --
    # otherwise the load fails on a parameter-count mismatch (which is
    # how the FSP arm's CE cell died: in the loader, not the measurement).
    _has_fsp = "fsp.W" in d.files
    m = N.Model(dim=dim, layers=layers, temp=0.75, lr=5e-4, fsp=_has_fsp)
    m.new_states(1)
    m.load(args.ckpt)
    m.new_states(1)          # zero memory: score the weights, not one stream's state

    print(f"checkpoint : {args.ckpt}")
    print(f"params     : {m.count():,}  (dim={dim}, layers={layers})")
    if "ring_k" in d:
        print(f"ring       : K={int(d['ring_k'])} lags (sized from trainer's TMT_RING_TOL)")
    if "bytes" in d:
        bytes_line = f"bytes seen : {int(d['bytes']):,}  (exact, from checkpoint)"
    else:
        bytes_line = (f"bytes seen ~= {int(d['t']) * int(d['B']):,}"
                      f"  (t*{int(d['B'])}; approximate if the batch size changed)")
    print(f"steps (t)  : {int(d['t']):,}   {bytes_line}")

    path = glob.glob(f"data/eval/**/*{args.file}*.txt", recursive=True)
    if not path:
        path = glob.glob(f"data/wikipedia_clean/**/wiki_*{args.file}.txt",
                         recursive=True)
    if not path:
        print("no wikistories file found; pass --file", file=sys.stderr)
        return 1
    raw = np.frombuffer(open(path[0], "rb").read(), np.uint8)
    held = raw[HELDOUT_START:HELDOUT_START + args.bytes + 1]
    bsrc = raw[BASELINE_SLICE[0]:BASELINE_SLICE[1]]

    uni = np.bincount(bsrc, minlength=256) + 1.0
    uni /= uni.sum()
    bg = np.ones((256, 256))
    for c, n in zip(bsrc[:-1], bsrc[1:]):
        bg[c, n] += 1
    bg /= bg.sum(axis=1, keepdims=True)

    ce_lm, stops, per = cross_entropy(m, held, return_per_byte=True)
    ce_uni = float(np.mean([-np.log(uni[n]) for n in held[1:]]))
    ce_bg = float(np.mean([-np.log(bg[c, n]) for c, n in zip(held[:-1], held[1:])]))
    se = float(per.std() / np.sqrt(len(per)))
    print(f"\nheld-out CE (nats/byte, {len(per)} bytes of file {args.file} @{HELDOUT_START:,}):")
    print(f"  unigram        {ce_uni:.4f}")
    print(f"  bigram         {ce_bg:.4f}")
    print(f"  model          {ce_lm:.4f} ± {se:.4f}   ({ce_bg - ce_lm:+.4f} vs bigram)")
    print(f"  perplexity     {np.exp(ce_lm):.1f}")
    if len(per) < 10000:
        print(f"  [note: ±{se:.3f} SE on this slice -- differences under ~2 SE are noise;")
        print(f"         use --bytes 20000 or more when comparing checkpoints]")

    sig = lambda x: 1 / (1 + np.exp(-x))
    allv = np.concatenate([sig(d[f"layer{i}.decay"]) for i in range(layers)])
    print(f"\ndecay = sigmoid(decay): min={allv.min():.4f} max={allv.max():.4f} "
          f"mean={allv.mean():.4f} std={allv.std():.4f}")
    print(f"  half-life    {np.log(0.5) / np.log(allv.mean()):.2f} bytes (mean)")
    print(f"stop head on held-out: mean={stops.mean():.4f} max={stops.max():.4f} "
          f"share>{args.threshold}={100 * (stops > args.threshold).mean():.2f}%")

    lab = line_end_labels(held)
    pos, neg = stops[lab], stops[~lab]
    print(f"\nstop head calibration ({lab.sum()} line-ends vs {len(lab) - lab.sum()} "
          f"mid-line, {100 * lab.mean():.2f}% positive):")
    if lab.sum() < 50:
        print(f"  [warning: only {lab.sum()} line-ends -- recall moves in "
              f"{100 / lab.sum():.1f}% steps; use --bytes 20000 for ~150]")
    print(f"  mean stop at line ends  {pos.mean():.4f}   (max {pos.max():.4f})")
    print(f"  mean stop mid-line      {neg.mean():.4f}   (max {neg.max():.4f})")
    print(f"  separation (AUC)        {auc_score(stops, lab):.4f}   (0.5 = no signal)")
    for thr in (0.1, 0.2, 0.35, 0.5):
        rec = float((pos > thr).mean()) if len(pos) else float("nan")
        pre = float(lab[stops > thr].mean()) if (stops > thr).any() else float("nan")
        print(f"  thr {thr:<4} recall {100 * rec:5.1f}%   precision {100 * pre:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
