"""Tests for the tail-credit fix (exact long-lag embed credit).

stage0  bit-identity regression: fresh model, no hot dims -> TMT_TAIL=1 must
        produce grads bit-identical to TMT_TAIL=0 (the patch must not disturb
        the normal ring path).
stage1  unit: TailCredit's decay->credit->add EMA matches a brute-force dense
        reference computed from the byte history (hot sigma included).
stage2  differential: three fresh identical models, one step, a prefilled
        ring holding a 30-byte history:
          (a) ring K=18 + tail (injected deep lags 19..30)
          (b) ring K=240 (near-dense: covers all 30 lags)
          (c) ring K=18, no tail (status-quo truncation)
        Hot-column embed grads: (a) must match (b); (c) must be clearly off.
        Cold columns must agree across all three.

Run:  TMT_RING_TOL=1e-5 .venv/bin/python test_tail.py
"""
import os
import sys
import numpy as np

F = np.float32


def rel(a, b):
    """norm-relative distance: max|a-b| scaled by the max signal in the slice."""
    den = max(float(np.abs(b).max()), 1e-8)
    return float(np.abs(a - b).max()) / den


def stage0():
    import numpy_port as np_
    os.environ["TMT_RING_TOL"] = "1e-4"
    os.environ["TMT_TAIL"] = "1"
    np_.rng = np.random.default_rng(3)
    m1 = np_.Model(dim=32, layers=1)
    m1.new_states(4)
    os.environ["TMT_TAIL"] = "0"
    np_.rng = np.random.default_rng(3)
    m2 = np_.Model(dim=32, layers=1)
    m2.new_states(4)
    cs = [10, 11, 12, 13]
    ns = [11, 12, 13, 14]
    ends = [0, 0, 0, 0]
    r1 = m1.step_batch(cs, ns, ends)
    r2 = m2.step_batch(cs, ns, ends)
    d = float(np.abs(m1.grads - m2.grads).max())
    print(f"stage0  cold-model bit-identity: max|dgrad| = {d}")
    assert d == 0.0, "tail must be a no-op when no dim is hot"
    assert abs(r1 - r2) == 0.0


def stage1():
    from tail_credit import TailCredit
    rng = np.random.default_rng(0)
    B, V, T, H = 3, 256, 400, 5
    sigma = rng.uniform(0.3, 0.7, H).astype(F)
    sigma[0] = 0.97                       # one runaway dim
    bs = rng.integers(0, V, size=(T, B))
    tail = TailCredit(B, ring_k=18)
    sigs = sigma[None, :]
    ring_w = [np.zeros((B, 18, H), F)]
    ring_ids = np.zeros((B, 18), np.int64)
    tail.sync(sigs, ring_w, ring_ids)
    assert tail.nhot(0) == 1, "sigma=0.97 at K=18 must be hot"
    for t in range(T):
        tail.begin_step(0, sigma)
        tail.add_byte(0, bs[t])

    ref = np.zeros((B, V, H))
    for t in range(T):
        ref *= sigma[None, None, :]
        for h in range(H):
            ref[np.arange(B), bs[t], h] += 1.0
    # only the hot column (dim 0) is tracked by the tail; cold columns are
    # the ring's job and legitimately differ (they're zero here)
    err = float(np.abs(ref[..., 0] - tail.E[0][:, :, 0]).max())
    print(f"stage1  EMA vs brute force: max|diff| = {err:.3e}  (values ~{ref.max():.0f})")
    assert err < 5e-3, "EMA recurrence diverged from the exact geometric series"

    g = rng.normal(size=(B, H)).astype(F)
    out = tail.credit(0, g)
    want = np.einsum("bvh,bh->vh", tail.E[0], g[:, 0:1])   # hot column only
    cerr = float(np.abs(out - want).max())
    cerr = float(np.abs(out - want).max())
    print(f"stage1  credit vs einsum:  max|diff| = {cerr:.2e}")
    assert cerr < 1e-4


def stage2():
    import numpy_port as np_
    rng = np.random.default_rng(11)
    L_HIST = 30
    y = np.arange(1, 1 + L_HIST, dtype=np.int64)     # 30 distinct history bytes
    B = 4
    cs = list(map(int, rng.integers(200, 256, B)))
    ns = list(map(int, rng.integers(200, 256, B)))
    ends = [False] * B

    def build(tol, tail):
        np_.RING_TOL = float(tol)                 # module global, read by __init__
        os.environ["TMT_TAIL"] = "1" if tail else "0"
        np_.rng = np.random.default_rng(7)
        m = np_.Model(dim=48, layers=2)
        m.layers[0].decay[0] = 6.0                # sigma ~0.9975 (deep hot dim)
        m.layers[1].decay[5] = 3.4                # sigma ~0.9678 (realistic hot dim)
        m.new_states(B)
        K = m.ring_k
        sigs = [np_.sigmoid(l.decay).astype(F) for l in m.layers]
        # ring: lag L sits at slot (L-1)%K with weight sigma^{L-1}
        for j in range(min(L_HIST, K)):
            L = j + 1
            slot = (L - 1) % K
            m.ring_b[:, slot] = y[L - 1]
            m.layers[0].ring_w[:, slot] = (sigs[0] ** (L - 1))[None, :]
            m.layers[1].ring_w[:, slot] = (sigs[1] ** (L - 1))[None, :]
        if tail:
            m._ensure_tail()
            # lags K+1..30 live beyond the ring; give them to the tail EMA at
            # their pre-decay weights (sigma^{L-1}, pattern of the ring seed)
            for li, hd in ((0, 0), (1, 5)):
                cols = m.tail.cols[li]
                hh = int(np.searchsorted(cols, hd))
                sig = sigs[li][hd]
                E = m.tail.E[li]
                for L in range(K + 1, L_HIST + 1):
                    E[:, y[L - 1], hh] += (sig ** (L - 1)).astype(F)
        return m

    ma = build("1e-4",  True)      # exact: ring K=15 + tail deep finish 16..30
    mb = build("1e-65", False)     # dense reference: ring K=240 covers all 30
    mc = build("1e-4",  False)     # status quo: ring K=15, lags 16..30 dropped
    for m in (ma, mb, mc):
        m.step_batch(cs, ns, ends)
    ga, gb, gc = (m.gviews["embed"].copy() for m in (ma, mb, mc))

    for hotdim, layer in ((0, 0), (5, 1)):
        rab = rel(ga[:, hotdim], gb[:, hotdim])
        rcb = rel(gc[:, hotdim], gb[:, hotdim])
        print(f"stage2  hot dim {hotdim} (layer{layer}) "
              f"tail-vs-dense rel={rab:.2e}  ring-vs-dense rel={rcb:.3f}")
        assert rab < 5e-3, f"tail credit diverged from the dense reference: {rab}"
        assert rcb > 1e-3, "status-quo ring should clearly truncate the hot dim"

    # cold columns: all three formulations must agree
    cold = [d for d in range(ma.dim) if d not in (0, 5)]
    ra = rel(ga[:, cold], gb[:, cold])
    rc = rel(gc[:, cold], gb[:, cold])
    print(f"stage2  cold cols: tail-vs-dense rel={ra:.2e}  ring-vs-dense rel={rc:.2e}")
    assert ra < 1e-2 and rc < 1e-2


if __name__ == "__main__":
    for name, fn in (("stage0", stage0), ("stage1", stage1), ("stage2", stage2)):
        print(f"== {name} ==")
        fn()
    print("\nALL STAGES PASSED")