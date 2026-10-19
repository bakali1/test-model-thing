#!/usr/bin/env python3
"""check_dense_trace.py -- P2 gate: does the dense trace reproduce ring+tail?

The ring is a K-lag approximation of one object: an exact per-byte-id embedding
eligibility trace. tail_credit.py already materialises that object for the dims
where the truncation costs more than 1% of the credit mass ("hot" dims). The
dense trace is therefore the tail with EVERY dim hot -- not a new mechanism, the
approximation removed.

That claim is testable, and it is the plan's own P2 gate: where both are
computable, the dense trace must reproduce ring+tail credit. If it does, the
ring was an approximation of something we can now afford exactly, and deleting
it is safe.

Three regimes, because "reproduce" means different things in each:

  (a) DENSE == TAIL exactly, when the tail covers every dim. The tail IS the
      dense trace restricted to a column subset, so with no restriction the two
      must agree to the last bit. This is the strongest and cheapest check.

  (b) DENSE vs RING where the ring is exact, i.e. at lags < K. Beyond K the ring
      truncates by construction, so agreement there is NOT expected -- what is
      expected is that the ring's error is bounded by the HOT_FRAC=1% mass it
      was designed around.

  (c) The dense trace's cost is the honest reason the ring existed: B*256*dim*4
      bytes per layer. Priced here so the trade is stated, not assumed.

Also pins that the default engine is untouched: trace_mode defaults to "ring",
so check_parity's numpy==torch identity and every existing arm are unaffected.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_port
import numpy_port

# DEVICE CONVENTION. These gates compare two implementations of the same
# thing (dense trace vs ring+tail, gated vs ungated, half-life init vs the
# default build), so they pin the device to "cpu" on purpose: the claim they
# make is device-independent, and letting the engine pick the device silently
# turned them into CUDA gates on a GPU box, where their CPU-built input
# tensors die with "Expected all tensors to be on the same device" -- they
# were not testing anything there, they were crashing. The device-parameterised
# gate is tests/check_seqlen_credit.py --device, which runs the T-axis claim on
# both.
DEV = "cpu"


DIM = 16          # small B and D: this is a correctness gate, not a speed test
LAYERS = 2
B = 4
T = 6
SEED = 0
# The ring's stated design tolerance: it truncates lags >= K, and tail_credit
# adopts a dim once sigma**ring_k > HOT_FRAC would drop more than 1% of mass.
RING_TOL = 0.01

PASS = True


def check(name, ok, detail=""):
    global PASS
    if not ok:
        PASS = False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))


def batch(seed, B, T):
    rng = np.random.default_rng(seed)
    cs = torch.from_numpy(rng.integers(1, 250, (B, T)).astype(np.int64))
    ns = torch.from_numpy(rng.integers(1, 250, (B, T)).astype(np.int64))
    ends = torch.zeros((B, T), dtype=torch.float32)
    return cs, ns, ends


def embed_grad(m, cs, ns, ends):
    """Run one step and return the embedding gradient (256, dim) as numpy.

    The engine accumulates dembed internally and folds it into self.grads; the
    cleanest way to read it is through the flat gradient's 'embed' slice, which
    is exactly what the optimizer would apply.
    """
    keep = (m.theta.detach().clone(), m.m.detach().clone(),
            m.v.detach().clone(), m.t)
    m.new_states(B)
    m.step_batch(cs, ns, ends)
    off = 0
    for n, shape in m.shapes:
        size = int(np.prod(shape))
        if n == "embed":
            g = m.grads.detach().cpu().numpy().astype(np.float64)
            out = g[off:off + size].reshape(shape)
            break
        off += size
    else:
        raise KeyError("no 'embed' in shapes")
    m.theta.copy_(keep[0])
    m.m.copy_(keep[1])
    m.v.copy_(keep[2])
    m.t = keep[3]
    return out


def main():
    print("P2 GATE -- dense trace vs ring+tail")
    print(f"  dim={DIM} layers={LAYERS} B={B} T={T} seed={SEED}\n")

    cs, ns, ends = batch(SEED, B, T)

    # ---- (a) dense == tail exactly, when the tail covers every dim --------
    print("(a) dense trace == TailCredit with all dims hot (exact)")
    # Force every dim hot by shrinking ring_k to 1: sigma**1 > 0.01 for any
    # sigma above 0.01, which holds for the seeded init. Then the tail covers
    # all D dims and the comparison is dense-vs-dense.
    m_ring = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED, ring_k=1)
    m_dense = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                                trace_mode="dense")
    n_hot = m_ring._ensure_tail_count() if hasattr(
        m_ring, "_ensure_tail_count") else None
    # Drive the tail into existence the way the engine does.
    m_ring.new_states(B)
    try:
        m_ring._ensure_tail()
    except Exception:
        pass
    n_hot = sum(len(c) for c in getattr(m_ring.tail, "cols", {}).values()) \
        if getattr(m_ring, "tail", None) is not None else 0
    print(f"      tail hot dims: {n_hot} of {DIM * LAYERS} "
          f"(layer*dim slots)")
    if n_hot >= DIM * LAYERS:
        gr = embed_grad(m_ring, cs, ns, ends)
        gd = embed_grad(m_dense, cs, ns, ends)
        denom = np.abs(gr).max()
        rel = (np.abs(gr - gd).max() / denom) if denom > 0 else 0.0
        check("dense == ring+tail exactly when the tail covers every dim",
              rel <= 1e-6, f"max rel dev {rel:.2e}")
    else:
        check("dense == ring+tail exactly when the tail covers every dim",
              False, f"tail only covered {n_hot}/{DIM * LAYERS} slots; "
                     f"the comparison regime was not set up")

    # ---- (b) vs the ring alone, where the ring is exact ----------------
    print("\n(b) dense vs ring, at lags the ring actually covers")
    # ring_k large enough that no dim is hot, so this is the pure ring arm --
    # the approximation whose cost P2 removes.
    m_ring2 = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED, ring_k=64)
    m_ring2.new_states(B)
    gr2 = embed_grad(m_ring2, cs, ns, ends)
    gd2 = embed_grad(m_dense, cs, ns, ends)
    denom = np.abs(gd2).max()
    rel2 = (np.abs(gr2 - gd2).max() / denom) if denom > 0 else 0.0
    # With T=6 < ring_k=64 the ring loses NO lag inside the window, so the only
    # divergence is the credit carried in from before the window -- which starts
    # at zero here because new_states zeroes the ring. They should therefore be
    # very close, and the gap is the ring's truncation error.
    check("dense vs ring with T < ring_k and a zeroed ring", rel2 <= RING_TOL,
          f"max rel dev {rel2:.2e} (ring's stated tolerance {RING_TOL:.0%})")

    # ---- (c) the cost, stated -------------------------------------------
    print("\n(c) the cost the ring existed to avoid")
    per_layer = B * 256 * DIM * 4
    print(f"      dense trace: {per_layer / 1e6:.1f} MB/layer at B={B}, "
          f"dim={DIM} -> {per_layer * LAYERS / 1e6:.1f} MB total")
    for bb, dd, ll in ((16, 512, 16), (64, 512, 16), (256, 512, 16)):
        gb = bb * 256 * dd * 4 * ll / 1e9
        print(f"      B={bb:>4} dim={dd} layers={ll:>2}: "
              f"{gb:.2f} GB  ({'fits' if gb < 8 else 'tight'} on an 8 GB box)")
    check("the trade is stated, not assumed", True)

    # ---- (d) the default engine is untouched ----------------------------
    print("\n(d) the default engine is untouched")
    d = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED)
    check("trace_mode defaults to 'ring'", d.trace_mode == "ring")
    check("no dense_E allocated in the default engine", d.dense_E is None)
    try:
        gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, trace_mode="bogus")
        check("an unknown trace_mode is refused", False, "accepted")
    except ValueError:
        check("an unknown trace_mode is refused", True)

    # ---- (e) dense save/load round-trips, and cross-mode is refused ---
    print("\n(e) dense save/load round-trips; cross-mode loads refuse")
    import tempfile
    md = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                           trace_mode="dense")
    md.new_states(B)
    md.step_batch(cs, ns, ends)
    p = os.path.join(tempfile.mkdtemp(), "d.npz")
    md.save(p)
    md2 = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                            trace_mode="dense")
    md2.new_states(B)
    md2.load(p)
    check("the dense trace round-trips bit-exactly",
          all(torch.equal(md.dense_E[i], md2.dense_E[i])
              for i in range(LAYERS)))
    check("and the trace really is non-zero after a step",
          float(sum(x.abs().sum() for x in md.dense_E)) > 0)
    # A ring checkpoint must not load into a dense engine: trace_mode is absent
    # from every legacy checkpoint, so 'absent' has to mean ring. Checking only
    # for the dense case let that through and silently zeroed the trace.
    mr = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED)
    mr.new_states(B)
    mr.step_batch(cs, ns, ends)
    pr = os.path.join(tempfile.mkdtemp(), "r.npz")
    mr.save(pr)
    for label, engine, path, want in (
            ("ring engine <- dense ckpt",
             gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED), p, "refuse"),
            ("dense engine <- ring ckpt",
             gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                               trace_mode="dense"), pr, "refuse")):
        engine.new_states(B)
        try:
            engine.load(path)
            got = "accepted"
        except ValueError:
            got = "refuse"
        check(f"{label} is refused", got == want, f"{got}, wanted {want}")
    mb = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                           trace_mode="dense")
    mb.new_states(2)
    mb.load(p)
    check("a smaller B adapts rather than crashing",
          tuple(mb.dense_E[0].shape) == (2, 256, DIM))
    os.remove(p)
    os.remove(pr)

    print("\n" + ("P2 GATE: PASS" if PASS else "P2 GATE: FAIL"))
    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
