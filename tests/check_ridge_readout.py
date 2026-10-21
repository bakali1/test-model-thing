#!/usr/bin/env python3
"""check_ridge_readout.py -- the gate for the copy/retrieval readout.

This gate exists because of defect D1 (2026-10-21), found while auditing the
read path: `tools/membench/model_backend.py:_ridge_score` had a swapped tuple
assignment in its lambda selection

    if acc > best_acc:
        best_acc, best = lam, acc     # <- SWAPPED

so `best_acc` held a LAMBDA, only lam=1.0 was ever eligible, and
`chosen = best` was a TRAINING ACCURACY.  The final fit therefore solved
`G + 0*I` -- an unregularized solve of a rank-deficient Gram matrix over
8192 features.  The ruler's stored output (`lam=0`) is the fingerprint of the
bug, and the "the readout never beats its own control" verdict that the
previous plan pivoted on was an artifact of it.

The same cell had a second defect: the (n, 4) payload targets were raveled
before the split, so only payload byte 0 of each sample was ever decoded,
while the metric was published as an "exact 4-byte payload" rate against a
chance of 2**-32 (the true chance for byte 0 is 1/256).

The gate checks the properties that matter:
  1. the returned lambda is one of the lambdas offered (not an accuracy);
  2. the lambda is chosen on the TRAINING half, not the test half;
  3. a decodable payload IS recovered (the readout has power -- otherwise
     every downstream "at chance" is uninformative);
  4. a payload with no signal is NOT recovered, and the permuted-label veto
     sits at chance.

Run:  .venv/bin/python tests/check_ridge_readout.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
from membench.model_backend import _ridge_score  # noqa: E402

LAM_OFFERED = (1.0, 1e3, 1e5)


def _decodable(n=512, d=64, k=4, seed=0):
    """Features X (n, d) and a k-byte payload y (n, k) that a LINEAR readout
    can recover: byte j is a clean threshold on feature j."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.empty((n, k), dtype=np.int64)
    for j in range(k):
        y[:, j] = (X[:, j] > 0.0).astype(np.int64)
    return X, y


def _noise(n=512, d=64, k=4, seed=0):
    """Features and a payload with NO relationship between them."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 256, size=(n, k))
    return X, y


def _te_indices(n, seed=0):
    """The test-half row indices _ridge_score will use for these args.

    It must mirror the implementation's split, otherwise the test cannot
    perturb the test half in isolation.
    """
    idx = np.random.default_rng(seed).permutation(n)
    n_train = max(4, n // 2)
    return idx[n_train:]


def main():
    fails = []

    # -- 1. the returned lambda is a lambda, not an accuracy --
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 60))
    y1 = rng.integers(0, 256, size=40)
    lam = _ridge_score(X, y1, X, y1, lams=LAM_OFFERED, seed=0)[3]
    if lam not in LAM_OFFERED:
        fails.append(f"[1] returned lam={lam!r} is not one of {LAM_OFFERED} "
                     f"-- the lambda selection is broken again")
    else:
        print(f"[1] OK  lambda selection returns a lambda ({lam:g})")

    # -- 2. the lambda is chosen on the TRAINING half --
    # Two datasets with IDENTICAL training halves and different test halves
    # must yield the same lambda.  (The old code chose on the test half.)
    n = 512
    Xa, ya = _decodable(n=n, seed=1)
    Xb, yb = _decodable(n=n, seed=2)
    te = _te_indices(n, seed=0)
    Xb2, yb2 = Xa.copy(), ya.copy()
    Xb2[te] = Xb[te]
    yb2[te] = yb[te]
    la = _ridge_score(Xa, ya, Xa, ya, lams=LAM_OFFERED, seed=0)[3]
    lb = _ridge_score(Xb2, yb2, Xb2, yb2, lams=LAM_OFFERED, seed=0)[3]
    if la != lb:
        fails.append(f"[2] lambda depends on the TEST half ({la:g} vs {lb:g}) "
                     f"-- it must be chosen on the training half")
    else:
        print(f"[2] OK  lambda is chosen on the training half ({la:g})")

    # -- 3. a decodable k-byte payload IS recovered --
    # The bar is deliberately modest: this is a 256-way ONE-HOT ridge, so a
    # readout can only ever emit classes it saw in the training half, and
    # the synthetic below populates few classes.  The point of the check is
    # POWER (readout >> null), not a ceiling.  Note for the real experiment
    # (see PLAN-2026-10-21-CONTINUAL-LEARNING.md, CL-1): with RANDOM
    # payloads the training half covers only ~1 - exp(-n_train/256) of the
    # 256 byte values, which CAPS the probe's per-byte accuracy far below
    # what the state could contain.  The real re-measurement must use enough
    # samples for class coverage to be complete, or a matched alphabet.
    Xs, ys = _decodable(n=512, d=64, k=4, seed=3)
    exact_s, per_s, se_s, lam_s = _ridge_score(Xs, ys, Xs, ys,
                                                lams=LAM_OFFERED, seed=0)
    if per_s < 0.60 or exact_s < 0.10:
        fails.append(f"[3] a linearly decodable 4-byte payload was recovered "
                     f"at only {per_s:.3f} per byte (exact {exact_s:.3f}); the "
                     f"readout has no power, so any 'at chance' result "
                     f"downstream is uninformative")
    else:
        print(f"[3] OK  decodable payload recovered (per-byte {per_s:.3f}, "
              f"exact {exact_s:.3f}, lam {lam_s:g})")

    # -- 4. no signal -> no recovery, and the veto sits at chance --
    Xn, yn = _noise(n=512, d=64, k=4, seed=4)
    exact_n, per_n, se_n, _ = _ridge_score(Xn, yn, Xn, yn,
                                            lams=LAM_OFFERED, seed=0)
    chance_exact = 256.0 ** -4
    if per_n > 0.02:
        fails.append(f"[4] a payload with NO signal was 'recovered' at "
                     f"{per_n:.4f} per byte -- the readout is inventing signal")
    else:
        print(f"[4] OK  no-signal payload not recovered (per-byte "
              f"{per_n:.4f}, exact {exact_n:.4f}, chance {chance_exact:.2e})")

    perm = np.random.default_rng(9).permutation(len(yn))
    exact_v, per_v, se_v, _ = _ridge_score(Xn, yn[perm], Xn, yn[perm],
                                            lams=LAM_OFFERED, seed=1)
    if per_v > 0.02:
        fails.append(f"[4] the permuted-label veto did not collapse "
                     f"({per_v:.4f} per byte)")
    else:
        print(f"[4] OK  permuted-label veto collapses ({per_v:.4f} per byte)")

    if fails:
        print("\nFAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS  the readout selects a real lambda, chooses it on the "
          "training half,\n      recovers a decodable payload, and invents no "
          "signal when there is none.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
