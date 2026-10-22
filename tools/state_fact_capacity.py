#!/usr/bin/env python3
"""HOW MANY FACTS DOES THE STATE HOLD AT ONCE?

The programme's synthesis (RESULTS.md 21) says the recurrent state holds a
COPY of the input's payload -- generic to unseen facts, but overwritten by
the next write -- while the trained head holds a TABLE in the weights that
board B overwrote.  The cheapest decisive arm (CL-2b) is a SMALLER BOARD
(board_size 64 instead of 512), and that arm PRE-SUPPOSES the state can
hold ~64 facts at once.  This measures whether it can, and how many.

THE MEASUREMENT.  One state capture, at the CUE (the readout point), and
FOUR different targets: the payload of fact 1, 2, 3 and 4.  For each
target k, a ridge readout is fitted from the state at the cue to fact k's
payload, FITTED ON FACTS 0-255 and TESTED ON FACTS 256-511 (disjoint
FACTS, not disjoint streams), on a board the model was never trained on.
The readout is the ruler's own (`tools/d5_probe.py:make_cell`, which also
gives the permuted-label null, the class coverage and the verdict rule).

  * if only the NEWEST fact's payload is decodable, the state holds about
    ONE fact, the capacity story is confirmed, and the board-64 arm would
    NOT help -- the programme must go straight to a content-addressed
    slot store;
  * if several are decodable, the state holds several, and the
    board-64 arm is worth running.

THE COPY CONTROL, per target.  Replace ONLY fact k's payload bytes in the
INPUT with fresh random bytes, leaving every other byte identical, and
re-measure target k.  A readout that still recovers the payload is NOT
reading fact k's payload out of the state -- it is reading something else
(the key, or another fact's copy).  A readout that collapses is reading a
copy of fact k's payload that the state really is carrying.

Everything here is a MEASUREMENT: the readout is REFITTED on each cell, so
these cells measure the state's COPYABILITY, never the model's retention.
The model's own head (not refitted) is the honest measure of what was
LEARNED; see RESULTS.md 20.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.cl_ruler as R  # noqa: E402
import tools.d5_probe as d5  # noqa: E402


def _apply_copy_control(seq, k, seed, alpha):
    """The COPY CONTROL: replace ONLY fact k's payload bytes IN THE INPUT
    with fresh random bytes, leaving EVERY other byte of the sequence
    identical (the key, the other facts, the gap, the cue).

    The payload of fact k sits at `k*FACT_LEN + KEY_LEN` .. `+PAYLOAD_LEN`,
    because the facts are concatenated as [KEY(4)][PAYLOAD(4)] and the
    delayed-copy cell is the LAST fact.  A control that changed anything
    else would not be a single-variable control, so the gate checks that
    every other byte is bit-identical.
    """
    out = seq.copy()
    off = k * R.FACT_LEN + R.KEY_LEN
    pool = (np.arange(256, dtype=np.uint8) if alpha is None
            else np.array(alpha, dtype=np.uint8))
    rng = np.random.default_rng(seed)
    out[:, off:off + R.PAYLOAD_LEN] = rng.choice(
        pool, size=(out.shape[0], R.PAYLOAD_LEN)).astype(out.dtype)
    return out


def _ridge_predict(Xf, Yf, Xt, seed, lams=None):
    """The repo's ridge discipline, returning PREDICTIONS so that the null
    can be the SAME fixed readout against permuted labels.

    Mirrors tools/membench/model_backend.py:_ridge_score: standardize on the
    training half, choose the lambda on an INNER hold-out carved out of the
    training half, refit at the chosen lambda on the full training half, and
    predict the test rows.  A k-byte payload gets k INDEPENDENT readouts, one
    per byte position (as _ridge_score does).
    """
    from tools.d5_probe import LAMS
    lams = lams or LAMS
    rng = np.random.default_rng(seed)
    n = Xf.shape[0]
    idx = rng.permutation(n)
    n_tr = max(4, n // 2)
    tr, te = idx[:n_tr], idx[n_tr:]

    def fit_predict(Xa, Ya, Xb, lam):
        mu = Xa.mean(axis=0, keepdims=True)
        sd = Xa.std(axis=0, keepdims=True) + 1e-8
        Za = (Xa - mu) / sd
        Zb = (Xb - mu) / sd
        k = Ya.shape[1]
        preds = np.zeros((Xb.shape[0], k), dtype=np.int64)
        for j in range(k):
            Y = np.zeros((len(Za), 256), np.float64)
            Y[np.arange(len(Za)), Ya[:, j]] = 1.0
            G = Za @ Za.T
            alpha = np.linalg.solve(G + lam * np.eye(len(Za)), Y)
            P = Zb @ Za.T @ alpha
            preds[:, j] = P.argmax(axis=1)
        return preds

    # choose the lambda on the inner hold-out
    best_lam, best_acc = None, -1.0
    for lam in lams:
        p_in = fit_predict(Xf[tr], Yf[tr], Xf[te], lam)
        acc = float((p_in == Yf[te]).mean())
        if acc > best_acc:
            best_acc, best_lam = acc, lam
    preds = fit_predict(Xf, Yf, Xt, best_lam)
    return preds, float(best_lam), best_acc


def _cell_factsplit(name, X, Y, fit, tst, seed, alpha_size, note=None):
    """A cell whose FIT/TEST split is over ROWS CHOSEN BY THE BOARD FACT of
    slot k -- so the readout is fitted on one set of board facts and tested
    on facts it has never been fitted on.  A row split would let the same
    board fact appear on both sides, and the readout would then be a table
    over the board rather than a mechanism.

    The null is the ruler's own: the SAME fixed readout (the fit depends
    only on the training half, so it is bit-identical) evaluated against
    PERMUTED test labels.  The coverage is the class coverage of the test
    rows.
    """
    import math
    from tools.d5_probe import _se_per_byte, coverage_report, verdict
    Y = np.asarray(Y, dtype=np.int64)
    Xf, Yf = X[fit], Y[fit]
    Xt, Yt = X[tst], Y[tst]
    preds, lam, inner_acc = _ridge_predict(Xf, Yf, Xt, seed)
    pb = float((preds == Yt).mean())
    e4 = float((preds == Yt).all(axis=1).mean())
    perm = np.random.default_rng(seed + 7919).permutation(len(tst))
    null_pb = float((preds == Yt[perm]).mean())
    null_e4 = float((preds == Yt[perm]).all(axis=1).mean())
    se = float(math.sqrt(max(e4 * (1 - e4), 0.0) / max(len(tst), 1)))
    cov = coverage_report(Y, fit, tst, alpha_size)
    out = {"cell": name, "per_byte": pb, "exact4": e4,
           "null_per_byte": null_pb, "null_exact4": null_e4, "se": se,
           "se_per_byte": _se_per_byte(pb, len(tst)),
           "coverage": cov, "label_alphabet_size": int(alpha_size),
           "chance_per_byte": 1.0 / alpha_size,
           "chance_exact4": (1.0 / alpha_size) ** Y.shape[1],
           "lam": float(lam), "inner_holdout_acc": inner_acc,
           "n_fit": int(len(fit)), "n_test": int(len(tst)), "note": note}
    out["verdict"] = verdict(out, cov, alpha_size, null_pb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--facts", type=int, default=4)
    ap.add_argument("--gap", type=int, default=32)
    ap.add_argument("--alphabet", default="16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boards", default="C,B")
    ap.add_argument("--text-frac", type=float, default=0.5)
    ap.add_argument("--split", type=int, default=256,
                    help="facts 0..split-1 are the FIT set, the rest TEST")
    args = ap.parse_args()

    from tools.cl2_train import alphabet_symbols, make_board
    from tools.membench import model_backend as mb
    alpha = alphabet_symbols(args.alphabet)
    alpha_size = 256 if alpha is None else len(alpha)

    m, _info = mb.load_frozen(args.ckpt)
    models = {"post": m}
    print(f"[loaded] {args.ckpt}")
    print(f"[setting] n={args.n} facts={args.facts} gap={args.gap} "
          f"alphabet={args.alphabet} fit-facts=0..{args.split-1} "
          f"test-facts={args.split}..")

    rng = np.random.default_rng(args.seed)

    for bname in args.boards.split(","):
        board = make_board({"A": 101, "B": 202, "C": 303}[bname], 512, alpha)
        # the stream, WITH the board index of every fact, so the fit/test
        # split can be over FACTS (not over rows): a row split would let the
        # same board fact appear in both, and the readout would then be a
        # table over the board rather than a mechanism.
        nb = board["keys"].shape[0]
        idx = rng.integers(0, nb, size=(args.n, args.facts))
        keys = board["keys"][idx]
        pays = board["payloads"][idx]
        stream = {"keys": keys, "payloads": pays, "idx": idx,
                  "facts": np.concatenate([keys, pays], axis=2)}
        # the gap: real text for a fraction of the rows, uniform random for
        # the rest -- the trainer's own rule.  The text pool is optional;
        # without it the gap is all random bytes (a distribution shift, so
        # the numbers are comparable within this run and to the ruler's
        # random-gap cells, but not to the trainer's own eval).
        gap = None
        try:
            import glob as _glob
            tf = sorted(_glob.glob("data/*.txt"))
            if tf:
                from tools.cl2_train import TextPool
                pool = TextPool(tf, seed=args.seed)
                gap = np.empty((args.n, args.gap), np.uint8)
                use_text = np.random.default_rng(args.seed + 5).random(
                    args.n) < args.text_frac
                nt = int(use_text.sum())
                if nt:
                    gap[use_text] = pool.windows(nt, args.gap,
                                                 np.random.default_rng(args.seed + 6))
                nr = args.n - nt
                if nr:
                    gap[~use_text] = np.random.default_rng(args.seed + 7).integers(
                        0, 256, size=(nr, args.gap)).astype(np.uint8)
        except Exception as e:  # pragma: no cover
            print(f"[gap] no text pool ({e}); using uniform random bytes")
            gap = np.random.default_rng(args.seed + 1).integers(
                0, 256, size=(args.n, args.gap)).astype(np.uint8)
        seq = R.build_sequences("stream", args.gap, stream, rng, alpha, gap=gap)
        X = d5.states_at_cue(models["post"], seq)

        # the FACT INDEX of each sample's stream: we need to know which
        # board fact each row used, so the fit/test split is over FACTS.
        print(f"\n===== BOARD {bname} =====")
        for k in range(args.facts):
            Y = stream["payloads"][:, k].astype(np.int64)
            # --- the FIT/TEST split is over the BOARD FACT of slot k
            kidx = stream["idx"][:, k]
            fit = np.where(kidx < args.split)[0]
            tst = np.where(kidx >= args.split)[0]
            if len(fit) < 32 or len(tst) < 32:
                print(f"  fact {k}: too few rows in the split "
                      f"({len(fit)} fit / {len(tst)} test); skipping")
                continue
            c = _cell_factsplit(f"fact{k}", X, Y, fit, tst, args.seed,
                                alpha_size,
                                note=f"the payload of fact {k} read from the "
                                     f"state at the cue; FIT on the board "
                                     f"facts 0..{args.split-1}, TEST on "
                                     f"{args.split}..")
            # --- the copy control: ONLY fact k's payload replaced in the input
            seqc = _apply_copy_control(seq, k, args.seed + 31 + k, alpha)
            Xc = d5.states_at_cue(models["post"], seqc)
            cc = _cell_factsplit(f"fact{k}_copy", Xc, Y, fit, tst, args.seed,
                                 alpha_size,
                                 note="ONLY fact k's payload replaced in the "
                                      "INPUT; everything else identical")
            print(f"  fact {k}: signal pb={c['per_byte']:.4f} e4={c['exact4']:.4f} "
                  f"null={c['null_per_byte']:.4f}/{c['null_exact4']:.5f} "
                  f"{c['verdict']:9} | copy-control "
                  f"pb={cc['per_byte']:.4f} e4={cc['exact4']:.4f} "
                  f"{cc['verdict']}")
        print(f"  (chance per-byte {1.0/alpha_size:.4f}, "
              f"chance exact-4 {(1.0/alpha_size)**R.PAYLOAD_LEN:.3e})")


if __name__ == "__main__":
    main()
