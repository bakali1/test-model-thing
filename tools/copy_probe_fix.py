#!/usr/bin/env python3
"""copy_probe_fix.py - CORRECTED readout control for the copy-task Stage 1.

The original copy_probe.py readout control had a confound: the two conditions
differed in BOTH the distractor stream AND the second key, so KL>0 proved only
"different recent input produces different output".

This fixes it: both conditions share the SAME distractor stream and the SAME
second key, and differ ONLY in the first 4 bytes (KEY vs DIFFERENT_KEY).

  cond A: [KEY]      + d shared distractors + KEY
  cond B: [DIFF_KEY] + d shared distractors + KEY

If the recurrent state retains any trace of the first key through d distractors,
the output distribution after the (identical!) second key will differ between
conds -> KL > 0. With sigma ~ 0.53 (measured), the trace should die by d ~ 16.

Also runs a permutation control: shuffle the A/B labels 200 times and report
the empirical KL null distribution, so "positive" means "above the null", not
just "> 0".

Only the readout leg. No copy leg (that one needs training and was never the
measurement). Runtime: ~4 min for 5 distances, batch 384.
"""
import os, sys, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import numpy_port as N
from eval_npz import infer_config

KEY = np.array([0x41, 0x42, 0x43, 0x44], dtype=np.uint8)          # "ABCD"
DIFFERENT_KEY = np.array([0x00, 0x01, 0x02, 0x03], dtype=np.uint8)
DISTANCES = [8, 32, 128, 512, 2048]
VOCAB = 256
RNG_REP = 100


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def kl_smoothed(p, q):
    """KL(p||q) with floor-smoothing, normalized."""
    eps = 1e-10
    p = np.clip(p, eps, 1.0); p = p / p.sum()
    q = np.clip(q, eps, 1.0); q = q / q.sum()
    return float((p * np.log(p / q)).sum())


def feed_seq(m, seq):
    """Feed a (B, L) uint8 batch through the model; return final logits (B, V)."""
    B, L = seq.shape
    m.new_states(B)
    enc = m.embed[seq[:, 0]]
    x = enc
    for l in m.layers:
        x, st, _d, _c = l.forward(enc, x, l.states)
        l.states = st
    for t in range(1, L):
        enc = m.embed[seq[:, t]]
        x = enc
        for l in m.layers:
            x, st, _d, _c = l.forward(enc, x, l.states)
            l.states = st
    return x @ m.Wd.T + m.bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distractors", choices=["random", "real"], default="random")
    ap.add_argument("--trials", type=int, default=384)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs/lrroles/arm_lrdecay1x_t919897.npz"))
    ap.add_argument("--distractor-file", default=os.path.join(HERE, "data/eval/tinystories_valid.txt"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.ckpt}")
    d = np.load(args.ckpt)
    dim, layers = infer_config(d)
    m = N.Model(dim=dim, layers=layers, temp=0.75, lr=5e-4)
    m.new_states(1); m.load(args.ckpt); m.new_states(1)
    print(f"dim={dim} layers={layers}")

    if args.distractors == "real":
        raw = np.frombuffer(open(args.distractor_file, "rb").read(), dtype=np.uint8)
    else:
        raw = None

    nA = args.trials // 2
    nB = args.trials - nA
    B = nA + nB
    print(f"trials={B} (A={nA} KEY-first, B={nB} DIFF-first), rep={RNG_REP} permutations")
    print(f"{'d':>6} {'KL_pos0':>10} {'KL_mean(4)':>11} {'null95':>10} {'null99':>10} {'above95':>8} {'above99':>8} {'t(s)':>7}")

    for ddist in DISTANCES:
        t0 = time.time()
        L = 4 + ddist + 4
        # ONE shared distractor stream, SAME for both conditions
        if args.distractors == "random":
            shared = rng.integers(0, VOCAB, size=(B, ddist), dtype=np.uint8)
        else:
            shared = np.zeros((B, ddist), dtype=np.uint8)
            n_avail = max(len(raw) - ddist, 1)
            for i in range(B):
                off = rng.integers(0, n_avail)
                shared[i] = raw[off:off + ddist]

        seq = np.zeros((B, L), dtype=np.uint8)
        firstkeys = np.vstack([np.tile(KEY, (nA, 1)), np.tile(DIFFERENT_KEY, (nB, 1))])
        seq[:, :4] = firstkeys
        seq[:, 4:4 + ddist] = shared          # SAME distractors for A and B
        seq[:, 4 + ddist:] = KEY              # SAME second key for A and B

        # Sample from the readout distribution at the SAME positions
        # (immediately after the second key; no probe-byte feeding needed for
        #  the pure retention question, but feed KEY bytes like the original
        #  to be comparable -- the first 3 don't affect relative KL much).
        logits = feed_seq(m, seq)             # logits right after second key
        p0 = softmax(logits)                  # (B, V) at probe position 0
        pA0 = p0[:nA]; pB0 = p0[nA:]
        kl0 = kl_smoothed(pA0.mean(axis=0), pB0.mean(axis=0))

        # permutation null on position-0 KL: shuffle A/B labels
        perm_kl = np.zeros(RNG_REP)
        for r in range(RNG_REP):
            perm = rng.permutation(B)
            pA = p0[perm[:nA]]; pB = p0[perm[nA:]]
            perm_kl[r] = kl_smoothed(pA.mean(axis=0), pB.mean(axis=0))
        null95 = float(np.quantile(perm_kl, 0.95))
        null99 = float(np.quantile(perm_kl, 0.99))
        above95 = float(kl0 > null95)
        above99 = float(kl0 > null99)

        print(f"{ddist:>6} {kl0:>10.4f} {'-':>11} {null95:>10.4f} {null99:>10.4f} "
              f"{'YES' if above95 else 'no':>8} {'YES' if above99 else 'no':>8} {time.time()-t0:>7.1f}",
              flush=True)

    print("DONE")


if __name__ == "__main__":
    main()