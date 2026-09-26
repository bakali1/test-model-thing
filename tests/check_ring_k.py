#!/usr/bin/env python3
"""Regression test: Model.load() must honor the checkpoint's stored ring_k.

ring_k (the number of retained trace lags) is sized from the decay range of the
run that wrote the checkpoint. If a load re-derives it from this process's
TMT_RING_TOL instead, the live run's K=18 silently becomes 14 at the default
tolerance and the embed credit for the hottest dims is truncated.

This test covers both contracts: an inferred (non-explicit) load restores the
checkpoint's stored K, while an explicit K=18 construction is authoritative
and rejects a checkpoint with a different stored K. It also checks that the
resized buffers still work in a real training step.

    .venv/bin/python tests/check_ring_k.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy_port as N
import gpu_port


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

    # Explicit construction must be authoritative, independent of RING_TOL
    # inference. A scratch model requested with K=18 reports K=18.
    explicit = gpu_port.GpuModel(dim=32, layers=2, device="cpu", ring_k=18)
    if explicit.ring_k != 18:
        print(f"FAIL: explicit from-scratch ring_k={explicit.ring_k}, expected 18")
        return 1

    # Use a GPU-native checkpoint for the mismatch check so numpy compatibility
    # keys cannot mask the requested ring_k assertion.
    with tempfile.TemporaryDirectory() as tmp:
        gpu_path = os.path.join(tmp, "gpu_k18.npz")
        explicit.new_states(4)
        explicit.save(gpu_path)
        mismatch = gpu_port.GpuModel(dim=32, layers=2, device="cpu", ring_k=14)
        try:
            mismatch.load(gpu_path)
            print("FAIL: explicit ring_k mismatch was silently accepted")
            mismatch_rejected = False
        except ValueError as e:
            print(f"  explicit K mismatch rejected: {str(e)[:75]}...")
            mismatch_rejected = True

    ok = mismatch_rejected
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
