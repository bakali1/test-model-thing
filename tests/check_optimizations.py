#!/usr/bin/env python3
"""CPU correctness gate for the optimization candidates."""
import copy
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gpu_port


def make(dim=32, layers=2, B=8, seed=123, **kw):
    torch.manual_seed(seed)
    m = gpu_port.GpuModel(dim=dim, layers=layers, device="cpu", **kw)
    # Force one hot dimension so A1 genuinely exercises the cold-column path
    # and tail credit is active in every candidate.
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


def main():
    base = make()
    base_losses = drive(base)
    checks = []
    for name, kw, tol in [
        ("a1-cold", {"ring_credit": "cold"}, 1e-6),
        ("a2-kloop", {"ring_credit": "kloop"}, 1e-6),
        ("c-segmented", {"ring_credit": "segmented"}, 1e-6),
        ("b1-ring", {"ring_algorithm": "factored"}, 2e-3),
        ("b2-tail", {"tail_algorithm": "factored"}, 2e-3),
        ("b1-b2", {"ring_algorithm": "factored", "tail_algorithm": "factored"}, 2e-3),
    ]:
        m = make(**kw)
        losses = drive(m)
        theta = nrel(base.theta, m.theta)
        checks.append((name, theta, tol, losses[-1], base_losses[-1]))
    for row in checks:
        name, err, tol, loss, expected = row
        assert err < tol, f"{name} theta state/RMS {err:.3e} >= {tol:.3e}"
        assert abs(loss - expected) < 1e-4, f"{name} loss drift {loss} vs {expected}"
        print(f"{name}: theta state/RMS={err:.3e}, loss delta={loss-expected:+.3e}")

    # Exercise the periodic factored-tail renormalization (the first 100-step
    # boundary is otherwise easy to leave untested).
    renorm = make(tail_algorithm="factored")
    drive(renorm, 101)
    assert all(torch.isfinite(f).all() for f in renorm.tail.factor.values())
    assert all(float(f.min()) > 0 for f in renorm.tail.factor.values())
    print("factored tail renormalization: PASS")

    # Factored state carries an algorithm version and tail factors. A factored
    # load of a legacy file is allowed only through the explicit migration path.
    p = os.path.join(tempfile.gettempdir(), "tmt-opt-version.npz")
    base.save(p)
    strict = make(ring_algorithm="factored")
    try:
        strict.load(p)
    except ValueError:
        pass
    else:
        raise AssertionError("factored model silently loaded a legacy checkpoint")
    migrated = make(ring_algorithm="factored", allow_algo_migration=True)
    migrated.load(p)
    assert migrated.algorithm_version == 1
    migrated.step_batch(*(torch.ones(8, dtype=torch.long) for _ in range(2)),
                        torch.zeros(8))
    migrated.save(p)
    resumed = make(ring_algorithm="factored", allow_algo_migration=True)
    resumed.load(p)
    assert "algorithm_version" in __import__("numpy").load(p)
    print("checkpoint algorithm version/migration: PASS")
    print("OPTIMIZATION CANDIDATES: PASS")


if __name__ == "__main__":
    main()
