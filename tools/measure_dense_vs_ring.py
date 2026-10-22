#!/usr/bin/env python3
"""measure_dense_vs_ring.py -- the dense-vs-ring(+tail) pin, at a size where
the two summation orders actually differ.

tests/check_dense_trace.py pins "dense == ring+tail exactly when the tail
covers every dim" at B=4, dim=16 -- a size where the reformulated contraction
and the einsum are BIT-IDENTICAL, so the pin reads 0.00e+00 before and after
the change.  That is the right gate size (it is a correctness gate, not a
speed test), but it does not say what the pin means at the sizes the engine
actually runs at.  This script measures it: the same comparison at B=64,
dim=512, where the two summation orders differ in the last bits.

The comparison is the one check_dense_trace.py makes: the ring arm with
ring_k=1 (so the tail covers EVERY dim, i.e. the ring+tail arm IS the dense
trace restricted to nothing) against the dense arm, on the embedding
gradient.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gpu_port

DEV = os.environ.get("TMT_GATE_DEV", "cpu")
DIM = int(os.environ.get("DIM", "512"))
LAYERS = int(os.environ.get("LAYERS", "2"))
B = int(os.environ.get("B", "64"))
T = 4
SEED = 0


def embed_grad(m, cs, ns, ends):
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
    m.theta.copy_(keep[0])
    m.m.copy_(keep[1])
    m.v.copy_(keep[2])
    m.t = keep[3]
    return out


def main():
    rng = np.random.default_rng(SEED)
    cs = torch.from_numpy(rng.integers(1, 250, (B, T)).astype(np.int64)).to(DEV)
    ns = torch.from_numpy(rng.integers(1, 250, (B, T)).astype(np.int64)).to(DEV)
    ends = torch.zeros((B, T), dtype=torch.float32, device=DEV)

    m_ring = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                               ring_k=1)
    m_ring.new_states(B)
    try:
        m_ring._ensure_tail()
    except Exception:
        pass
    n_hot = sum(len(c) for c in getattr(m_ring.tail, "cols", {}).values()) \
        if getattr(m_ring, "tail", None) is not None else 0
    m_dense = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                                trace_mode="dense")
    gr = embed_grad(m_ring, cs, ns, ends)
    gd = embed_grad(m_dense, cs, ns, ends)
    denom = max(float(np.abs(gr).max()), 1e-30)
    rel = float(np.abs(gr - gd).max() / denom)
    print(f"device={DEV} dim={DIM} layers={LAYERS} B={B} T={T} "
          f"tail hot dims={n_hot}/{DIM * LAYERS}")
    print(f"  dense vs ring+tail: max rel dev {rel:.3e}  "
          f"(max abs {float(np.abs(gr - gd).max()):.3e}, "
          f"scale {denom:.3e})")


if __name__ == "__main__":
    main()
