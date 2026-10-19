#!/usr/bin/env python3
"""CPU correctness gate for the surviving optimization paths.

This gate used to test the `cold` / `kloop` / `segmented` ring-credit candidates
and the factored ring/tail variants. All of those were measured and deleted from
gpu_port.py (see RESULTS.md), so the gate died with `ValueError: unknown
ring_credit='cold'` instead of testing anything -- a broken gate is worse than no
gate, because it reads as coverage.

It now tests the two things that are actually true and actually unguarded:

  1. `grad_assembly="loop"` vs `"cat"` -- documented in gpu_port.py as a pure
     copy reorganization (same self.shapes order, one kernel instead of 69), so
     the two must be bit-identical. Nothing else covered this axis.
  2. The deleted candidates must be REJECTED with a clear ValueError rather than
     silently falling back to a reference path. A silent fallback would let a
     caller believe it is running a candidate that no longer exists.

`ring_credit` itself is deliberately not compared here: on CPU every mode
resolves to `full` (`_ring_mode`), so a comparison would be vacuous. The CUDA
paths (`compiled`/`triton`/`nvrtc`) need a GPU and are covered by the T4/Colab
board runs recorded in RESULTS.md.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gpu_port


def make(dim=32, layers=2, B=8, seed=123, **kw):
    torch.manual_seed(seed)
    m = gpu_port.GpuModel(dim=dim, layers=layers, device="cpu", **kw)
    # Force one hot dimension so the tail-credit path is genuinely exercised.
    for layer in m.layers:
        layer.decay[0] = 4.0
    m.new_states(B)
    return m


def drive(m, steps=25, seed=9):
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        c = torch.randint(1, 200, (m.B,), generator=g)
        n = torch.randint(1, 200, (m.B,), generator=g)
        e = (torch.rand(m.B, generator=g) < .1).float()
        losses.append(m.step_batch(c, n, e))
    return losses


def nrel(a, b):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return float((a - b).abs().max() / (b.square().mean().sqrt() + 1e-30))


REMOVED = [
    ("ring_credit", "cold"),
    ("ring_credit", "kloop"),
    ("ring_credit", "segmented"),
    ("ring_credit", "compiled-bogus"),
]


def main():
    base = make()
    base_losses = drive(base)

    # 1. grad_assembly: loop must reproduce cat exactly.
    m = make(grad_assembly="loop")
    losses = drive(m)
    theta = nrel(base.theta, m.theta)
    assert theta == 0.0, f"grad_assembly=loop theta {theta:.3e}, expected bit-exact"
    for i, (a, b) in enumerate(zip(losses, base_losses)):
        assert abs(a - b) < 1e-6, f"grad_assembly=loop loss drift at step {i}: {a} vs {b}"
    print(f"grad_assembly=loop vs cat: bit-exact over {len(losses)} steps: PASS")

    # 2. ring_credit="full" is the portable reference and the CPU resolution of
    #    "auto", so it must match the default exactly.
    m = make(ring_credit="full")
    losses = drive(m)
    theta = nrel(base.theta, m.theta)
    assert theta == 0.0, f"ring_credit=full theta {theta:.3e}, expected bit-exact"
    for i, (a, b) in enumerate(zip(losses, base_losses)):
        assert abs(a - b) < 1e-6, f"ring_credit=full loss drift at step {i}: {a} vs {b}"
    assert m._ring_mode() == "full", "CPU must resolve ring_credit to 'full'"
    print("ring_credit=full == auto on CPU: PASS")

    # 3. The deleted candidates must be refused, not silently downgraded.
    for key, value in REMOVED:
        try:
            make(**{key: value})
        except ValueError as exc:
            assert value in str(exc), f"error should name {value!r}, got: {exc}"
        else:
            raise AssertionError(
                f"{key}={value!r} was accepted; this candidate was measured and "
                f"removed from gpu_port.py -- a silent fallback would let a "
                f"caller believe it is running code that no longer exists")
    print(f"removed candidates refused ({len(REMOVED)}): "
          f"{', '.join(v for _, v in REMOVED)}: PASS")

    # 4. A checkpoint carrying the removed factored keys must still be refused on
    #    load -- that guard is in gpu_port.load and has no other coverage.
    import numpy as np
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "tmt-opt-stale.npz")
    base.save(p)
    with np.load(p) as d:
        data = {k: d[k] for k in d.files}
    data["algorithm_version"] = np.array(1, np.int64)
    np.savez(p, **data)
    try:
        make().load(p)
    except ValueError as exc:
        assert "factored" in str(exc) or "algorithm_version" in str(exc), str(exc)
    else:
        raise AssertionError("stale factored checkpoint loaded silently")
    print("stale factored-candidate checkpoint refused: PASS")

    print("OPTIMIZATION PATHS: PASS")


if __name__ == "__main__":
    main()
