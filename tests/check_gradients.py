#!/usr/bin/env python3
"""Finite-difference gradient check for numpy_port.py.

Runs the port in float64 and compares its analytic gradients against central
differences of its own reported loss. The byte sequence uses disjoint input
bytes (0..127) and target bytes (128..255), so the stop-gradient target — which
the analytic pass intentionally omits — can never contaminate the rows that are
scored.

    .venv/bin/python tests/check_gradients.py
    .venv/bin/python tests/check_gradients.py --steps 6 --dim 8 --layers 3

Exits non-zero if any parameter group exceeds --tol.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy_port as N


def check(dim, layers, steps, seed, tol):
    N.F = np.float64                      # the port reads the module-level F

    rng = np.random.default_rng(seed)
    cs = list(rng.integers(0, 128, size=steps))
    ns = list(rng.integers(128, 256, size=steps))
    ends = [0.0] * steps
    ends[-1] = 1.0

    def fresh():
        m = N.Model(dim=dim, layers=layers, temp=0.75, lr=5e-4)
        m.new_states(1)
        m.lr = 0.0                        # freeze weights: loss is a pure f(theta)
        m.wd = 0.0
        return m

    def loss_and_grad(theta):
        m = fresh()
        m.theta[:] = theta
        loss = None
        for k in range(steps):
            loss = m.step_batch([cs[k]], [ns[k]], [ends[k]])
        return loss, m.grads.copy()

    base = fresh()
    theta0 = base.theta.copy()
    n = theta0.size
    loss0, ana = loss_and_grad(theta0)

    eps = 1e-6
    num = np.zeros(n)
    for i in range(n):
        tp = theta0.copy(); tp[i] += eps
        tm = theta0.copy(); tm[i] -= eps
        num[i] = (loss_and_grad(tp)[0] - loss_and_grad(tm)[0]) / (2.0 * eps)

    off = 0
    results = []
    for name, shape in base.shapes:
        size = int(np.prod(shape))
        if name == "embed":
            # Score only input rows: target rows carry the (intentionally
            # omitted) stop-gradient target path and are not comparable.
            rows = sorted(set(cs))
            av = ana[:size].reshape(shape)[rows].ravel()
            rv = num[:size].reshape(shape)[rows].ravel()
        else:
            av = ana[off:off + size]
            rv = num[off:off + size]
        denom = max(float(np.abs(rv).max()), 1e-12)
        results.append((name, float(np.abs(av - rv).max() / denom)))
        off += size

    print(f"steps={steps} dim={dim} layers={layers} seed={seed}  loss={loss0:.6f}")
    worst, wn = 0.0, ""
    for name, rel in results:
        print(f"  {'ok  ' if rel < tol else 'FAIL'} {name:16} rel.err {rel:.3e}")
        if rel > worst:
            worst, wn = rel, name
    print(f"worst {worst:.3e} ({wn}); tol {tol:.1e}")
    return worst < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    ok = check(args.dim, args.layers, args.steps, args.seed, args.tol)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
