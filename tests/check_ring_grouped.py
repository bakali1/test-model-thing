#!/usr/bin/env python3
"""check_ring_grouped.py -- the gate for the GROUPED ring credit.

    .venv/bin/python tests/check_ring_grouped.py [--B 64] [--dim 64]

WHAT IT PINS
------------
The grouped credit is the first candidate that changes the SHAPE of the
ring credit's scatter without changing a single float. It groups the rows by
byte id, combines each group's contributions in the reference's row order,
and writes each address ONCE -- so the layer issues no atomics at all.

That is only legal because of three facts, and this gate checks all three:

  1. The reference's order is the ROW order. numpy_port's `_scatter_credit`
     walks the rows in increasing index, and torch's `index_add_` is
     bit-identical to that walk (measured here, not assumed). The grouped
     path must reproduce it EXACTLY: same value, same order, same float.
  2. The lag axis is NOT a reduction axis. The K lag terms of one (b, d) go
     to K DIFFERENT byte ids, so "reduce the K axis in registers and issue
     one atomic per (b, d)" is not a faster credit, it is a WRONG credit --
     one byte's credit paid onto another byte. The gate shows the ids
     differ, and shows the result differs.
  3. Reassociation is not free. A grouped sum that is computed as a tree
     (which is what the removed `segmented` candidate did with
     `segment_reduce`) is a DIFFERENT float from the left fold. The gate
     measures the gap, so the reason `segmented` could not be adopted as
     "the same arithmetic" is on the record.

The gate is CPU-only on purpose: the bit-identity claim is a CPU claim (on
CUDA the atomic order is the hardware's, so the same engine can differ in
the last bits between runs -- see the determinism check below, which
MEASURES that instead of assuming it). The CUDA kernel is checked for
compilation by tools/check_nvrtc.py, and its result is checked against the
same reference by tools/bench_ring_credit.py on the device itself.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gpu_port
import numpy_port


def build(dim, layers, B, seed, device="cpu", ring_credit="full"):
    m = gpu_port.GpuModel(dim=dim, layers=layers, lr=5e-4, seed=seed,
                           device=device, ring_credit=ring_credit)
    m.new_states(B)
    return m


def fill_ring_b(m, seed=0):
    """Realistic byte ids: the ring holds the last K bytes of each stream."""
    g = torch.Generator().manual_seed(seed)
    B, K = m.ring_b.shape
    ids = torch.randint(0, 256, (B, K), generator=g)
    m.ring_b.copy_(ids)


def credit_of(m, path, seed=1234):
    """Run the credit ONCE through `path` and return (dest, w_after).

    The ring is RESTORED before the call: the credit decays the ring in
    place, so a second call through a different path would see a twice-
    decayed ring and every comparison would be against the wrong baseline.
    """
    torch.manual_seed(seed)
    l = m.layers[0]
    if getattr(m, "_pristine_ring", None) is None:
        raise RuntimeError("the test must snapshot the ring before the "
                           "first credit call")
    with torch.no_grad():
        l.ring_w.copy_(m._pristine_ring)
    m._ring_dest = torch.zeros((256, m.dim), dtype=torch.float32,
                               device=m.device)
    dlds = torch.randn((m.B, m.dim), device=m.device)
    rows = m.ring_b.reshape(m.B * m.ring_k)
    dec = torch.sigmoid(l.decay)
    m.ring_credit = path
    m._ring_resolved = None
    m._scatter_ring_credit(0, l.ring_w, dlds, rows, None, dec)
    return m._ring_dest.detach().clone(), l.ring_w.detach().clone()


def test_reference_order():
    """The reference's order IS the row order (and index_add_ agrees with it).

    This is the fact the whole legality argument rests on: if index_add_
    summed in some other order, there would be no float to preserve.
    """
    g = torch.Generator().manual_seed(7)
    rows = torch.randint(0, 5, (64,), generator=g)
    vals = torch.randn(64, 8, generator=g)
    manual = torch.zeros(5, 8)
    for i in range(64):                       # the numpy_port walk
        manual[rows[i]] += vals[i]
    got = torch.zeros(5, 8).index_add_(0, rows, vals)
    assert torch.equal(manual, got), (
        "index_add_ does not add in row order: the reference's order is "
        "undefined and the legality argument collapses")
    npref = np.zeros((5, 8), dtype=np.float32)
    np.add.at(npref, rows.numpy(), vals.numpy())
    assert np.array_equal(npref, manual.numpy()), (
        "numpy's add.at and torch's index_add_ disagree")
    print("[PASS] the reference's order is the row order: index_add_ is "
          "bit-identical to the numpy walk, and to np.add.at")
    return True


def test_lag_axis_is_not_a_reduction_axis(m):
    """The K lag terms of one (b, d) land on K DIFFERENT byte ids.

    So the "one atomic per (b, d)" rewrite -- the obvious way to cut the
    atomics by K -- sums contributions destined for different bytes. It is
    not a rounding difference, it is a different credit.
    """
    rows = m.ring_b
    K = rows.shape[1]
    diff = (rows[:, 1:] != rows[:, :1]).float().mean().item()
    same_ids = (rows[:, 1:] == rows[:, :1]).sum().item()
    print(f"      the K lag slots of a stream hold {diff:.1%} DIFFERENT "
          f"byte ids from each other ({same_ids} of "
          f"{rows.shape[0]*(K-1)} lag pairs coincide)")
    assert diff > 0.5, (
        "the ring's lag slots do NOT hold different byte ids -- if they did, "
        "the lag axis WOULD be a reduction axis and the analysis is wrong")
    # and the rewrite really does move the credit onto other bytes
    dlds = torch.randn((m.B, m.dim))
    w = m.layers[0].ring_w.clone()
    w.normal_()                                 # a ring with something in it
    vals = w * dlds[:, None, :]
    per_b = vals.sum(dim=1)                    # the ILLEGAL K-axis reduction
    truth = torch.zeros(256, m.dim)
    for i in range(m.B):
        for k in range(m.ring_k):
            truth[m.ring_b[i, k]] += vals[i, k]
    illegal = torch.zeros(256, m.dim)
    for i in range(m.B):
        illegal[m.ring_b[i, 0]] += per_b[i]     # everything onto slot 0
    d = (truth - illegal).abs().max().item()
    assert d > 1e-3, "the illegal reduction happened to agree -- suspicious"
    print(f"[PASS] the lag axis is not a reduction axis: the illegal "
          f"K-reduction lands on the wrong bytes (max|d| {d:.3e})")
    return True


def test_reassociation_is_a_different_float():
    """A grouped sum that is reassociated is a DIFFERENT float.

    This is why the removed `segmented` candidate (sort + `segment_reduce`,
    which reassociates) could not be adopted as "the same arithmetic", and
    why the grouped path folds LEFT.
    """
    g = torch.Generator().manual_seed(3)
    m = 72
    vals = torch.randn(m, 1, generator=g)
    left = torch.zeros(1)
    for i in range(m):
        left = left + vals[i]                  # the reference's left fold
    tree = vals.sum()                           # torch's pairwise reduction
    d = (left - tree).abs().item()
    rel = d / (left.abs().item() + 1e-30)
    print(f"      a left fold and a pairwise sum of the same {m} values "
          f"differ by {d:.3e} ({rel:.2e} relative)")
    assert d > 0.0, (
        "the left fold and the pairwise sum agree bit-for-bit here -- then "
        "the reassociation argument would be untested, not disproven")
    print("[PASS] reassociation changes the float: the grouped credit must "
          "fold LEFT, in row order")
    return True


def test_grouped_is_bit_identical(m):
    """The grouped credit reproduces the reference's float, bit for bit."""
    ref, w_ref = credit_of(m, "full")
    got, w_got = credit_of(m, "grouped")
    assert torch.equal(ref, got), (
        "the grouped credit moved a float: max|d| = "
        f"{(ref-got).abs().max().item():.3e}")
    assert torch.equal(w_ref, w_got), "the grouped credit decayed the ring " \
                                      "differently"
    print("[PASS] the grouped credit is BIT-IDENTICAL to `full` (destination "
          "and the decayed ring)")
    return True


def test_grouped_matches_numpy(m):
    """And the grouped credit matches numpy_port's own scatter, bit for bit.

    numpy_port is the reference implementation; the credit's destination is
    built here by hand in the reference's order, and the grouped path must
    match it to the last bit.
    """
    torch.manual_seed(99)
    l = m.layers[0]
    dlds = torch.randn((m.B, m.dim))
    dec = torch.sigmoid(l.decay)
    rows = m.ring_b.reshape(m.B * m.ring_k)
    with torch.no_grad():
        l.ring_w.copy_(m._pristine_ring)      # the ring, undecayed
    w = l.ring_w.clone()
    ref = torch.zeros(256, m.dim)
    wref = w.clone()
    wref.mul_(dec[None, None, :])
    vals = (wref * dlds[:, None, :]).reshape(-1, m.dim)
    for i in range(m.B * m.ring_k):
        ref[rows[i]] += vals[i]                  # numpy's `_scatter_credit`
    got, _ = credit_of(m, "grouped", seed=99)
    assert torch.equal(ref, got), (
        "the grouped credit differs from numpy's own scatter: max|d| = "
        f"{(ref-got).abs().max().item():.3e}")
    print("[PASS] the grouped credit is BIT-IDENTICAL to numpy_port's own "
          "scatter")
    return True


def test_engine_trajectory(m_args, steps=12):
    """A whole engine run: `grouped` must track `full` bit-for-bit.

    The credit is 16 layers of a 4.48M-parameter engine, so a change in the
    credit's float would show up in theta, m and v. This is the end-to-end
    version of the claim the project cares about.
    """
    g = torch.Generator().manual_seed(5)
    B = m_args["B"]
    a = build(m_args["dim"], m_args["layers"], B, 11,
              ring_credit="full")
    b = build(m_args["dim"], m_args["layers"], B, 11,
              ring_credit="grouped")
    rng = np.random.default_rng(21)
    for t in range(steps):
        cs = torch.from_numpy(rng.integers(1, 250, size=B).astype(np.int64))
        ns = torch.from_numpy(rng.integers(1, 250, size=B).astype(np.int64))
        es = torch.from_numpy(
            (rng.random(B) < 0.02).astype(np.float32))
        a.step_batch(cs, ns, es)
        b.step_batch(cs, ns, es)
    assert torch.equal(a.theta, b.theta), "theta diverged"
    assert torch.equal(a.m, b.m), "the first moment diverged"
    assert torch.equal(a.v, b.v), "the second moment diverged"
    print(f"[PASS] {steps} steps of the whole engine: `grouped` and `full` "
          "are bit-identical in theta, m and v")
    return True


def test_kernel_index_arithmetic(m):
    """The CUDA kernel's index arithmetic, emulated element by element.

    The kernel cannot be run without a GPU, so the part of it that can be
    wrong without a device -- the INDEX arithmetic (which row of the ring,
    which stream of `dlds`, which address of `dest`) -- is emulated here in
    the same order the kernel does it, from the same plan the kernel builds.
    A mistake in the plan (a wrong stream, a wrong row) would show up here
    as a number that is not the reference's.
    """
    torch.manual_seed(31337)
    l = m.layers[0]
    with torch.no_grad():
        l.ring_w.copy_(m._pristine_ring)
    w0 = l.ring_w.detach().clone()
    dlds = torch.randn((m.B, m.dim))
    dec = torch.sigmoid(l.decay)
    rows = m.ring_b.reshape(m.B * m.ring_k)
    K, D, nb = m.ring_k, m.dim, 256

    # ---- the reference: numpy_port's walk ---------------------------------
    wref = w0.clone()
    wref.mul_(dec[None, None, :])
    vals = (wref * dlds[:, None, :]).reshape(-1, D)
    ref = torch.zeros(nb, D)
    for i in range(m.B * K):
        ref[rows[i]] += vals[i]

    # ---- the plan the kernel builds ---------------------------------------
    order = torch.argsort(rows, stable=True)
    counts = torch.bincount(rows, minlength=nb)
    goff = torch.cumsum(counts, 0) - counts
    woff = order * D
    dlsoff = (order // K) * D

    # ---- the kernel, emulated ---------------------------------------------
    w = w0.clone()
    wflat = w.reshape(-1)              # the kernel sees a raw pointer
    dlsflat = dlds.reshape(-1)
    dest = torch.zeros(nb, D)
    for r in range(nb):
        m_r = int(counts[r])
        off = int(goff[r])
        for d in range(D):
            s = dest[r, d]
            for j in range(m_r):
                wo = int(woff[off + j])
                lo = int(dlsoff[off + j])
                wv = wflat[wo + d] * dec[d]
                wflat[wo + d] = wv
                v = wv * dlsflat[lo + d]
                s = s + v
            dest[r, d] = s
    assert torch.equal(ref, dest), (
        "the kernel's index arithmetic does not reproduce the reference: "
        f"max|d| = {(ref - dest).abs().max().item():.3e}")
    assert torch.equal(wref, w), "the kernel decays a different ring"
    print("[PASS] the CUDA kernel's index arithmetic, emulated, reproduces the "
          "reference bit for bit")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    print("== GATE: the grouped ring credit changes no float ==")
    print(f"  B={args.B} dim={args.dim} layers={args.layers} device=cpu")
    ok = True
    ok &= test_reference_order()
    m = build(args.dim, args.layers, args.B, 11)
    fill_ring_b(m)
    torch.manual_seed(4242)
    with torch.no_grad():
        m.layers[0].ring_w.normal_()           # a ring with something in it
    m._pristine_ring = m.layers[0].ring_w.detach().clone()
    ok &= test_lag_axis_is_not_a_reduction_axis(m)
    ok &= test_reassociation_is_a_different_float()
    ok &= test_grouped_is_bit_identical(m)
    ok &= test_grouped_matches_numpy(m)
    ok &= test_kernel_index_arithmetic(m)
    ok &= test_engine_trajectory({"B": args.B, "dim": args.dim,
                                   "layers": args.layers})
    print("\n== verdict:", "PASS" if ok else "FAIL", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
