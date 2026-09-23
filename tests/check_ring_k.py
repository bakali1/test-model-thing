#!/usr/bin/env python3
"""Regression test: Model.load() must honor the checkpoint's stored ring_k.

ring_k (the number of retained trace lags) is sized from the decay range of the
run that wrote the checkpoint. If a load re-derives it from this process's
TMT_RING_TOL instead, the live run's K=18 silently becomes 14 at the default
tolerance and the embed credit for the hottest dims is truncated.

This test saves a model with a non-default ring_k, loads it into a model built
with different settings, and asserts K, params and a real training step all
survive the round-trip.

    .venv/bin/python tests/check_ring_k.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy_port as N


def main():
    # Source model: a decay_init that yields a ring_k far from the default.
    src = N.Model(dim=32, layers=2, decay_init=2.0)
    src.new_states(4)
    # Perturb params so a bit-identity check is meaningful.
    src.theta += 0.01

    # Destination: default settings => a different ring_k, so the stored value
    # must win. Assert the premise first.
    dst = N.Model(dim=32, layers=2)
    if src.ring_k == dst.ring_k:
        print(f"FAIL: test is vacuous, both ring_k={src.ring_k}; "
              f"pick a decay_init that differs from the default")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ck.npz")
        src.save(path)
        dst.new_states(4)
        dst.load(path)

    ok = True
    if dst.ring_k != src.ring_k:
        print(f"FAIL: ring_k not restored: checkpoint {src.ring_k}, "
              f"loaded {dst.ring_k}")
        ok = False
    if not np.array_equal(src.theta, dst.theta):
        print("FAIL: params changed by the ring_k resize")
        ok = False

    # And the resized buffers must actually work in a training step.
    try:
        dst.step_batch([65, 66, 67, 68], [66, 67, 68, 69], [False] * 4)
    except Exception as e:                       # pragma: no cover
        print(f"FAIL: step after load raised {type(e).__name__}: {e}")
        ok = False

    print(f"src ring_k={src.ring_k} dst ring_k={dst.ring_k} "
          f"step ok; {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
