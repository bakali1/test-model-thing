"""tools/bench_ring_credit.py -- time the ring credit ALONE, per path.

WHY THIS EXISTS. The engine's optimizer step is dominated by one thing: the
ring credit's atomic scatter. At B=1024, K=18, dim=512, 16 layers the credit
computes `vals = w * dlds` over the whole (B,K,D) tensor and then does
`dest.index_add_(0, rows, vals)` -- so every element of that tensor is an
fp32 atomic add: 9.44M per layer, 151M per step, into a (256, dim) = 131K
address destination, i.e. ~72 contending atomics per address.

The engine measures ~83 ms per step on a Colab T4 and ~300 ms on an MX150.
The traffic is only ~604 MB/step (1.9 ms at 320 GB/s) and the model's FLOPs
are trivial, so the hypothesis is that CONTENDED ATOMICS are the wall -- and
the two machines' similar-but-not-equal step times (a 3.7x gap for a 4.3x
FLOP gap and a 20x SM gap) are what an atomic limit looks like.

This harness measures the credit on its own, so the step time is not diluted
by everything else, and it prints the atomic arithmetic next to the
measurement -- the number the hypothesis is actually about.

WHAT THIS HARNESS FIXES (the version that shipped with the task was wrong in
a way that would have mis-measured the contention):

  * The ring's byte ids were left at their INITIAL value -- all zeros -- so
    every one of the B*K rows addressed byte id 0. That is 18,432 contending
    atomics on ONE address instead of the real ~72, i.e. a 256x overstatement
    of the contention, and it would have made any "the atomics are the wall"
    conclusion an artefact of the harness. The byte ids are now drawn from a
    realistic (Zipf-like) byte-frequency distribution, because real byte ids
    are NOT uniform: the commonest byte of a text corpus takes ~12% of the
    rows, and that -- not the mean -- is what a contended atomic feels.
  * `dlds` was all zeros, so the credit added 0.0 to every address. The
    timing is the same, but the VALUES were not, which made the harness
    useless for the one thing it must also do: check that the paths agree.
  * The paths are now compared against `full` for BIT-IDENTITY of the result,
    which is the legality claim of docs/plans/SPEED-INVESTIGATION.md, made
    measurable on the device the measurement runs on.

Usage:
    python tools/bench_ring_credit.py --batch 1024 --dim 512 --layers 16 \
        --ring-k 18 --iters 50
    python tools/bench_ring_credit.py --ckpt runs/taxis/A_p4_wd0.npz

It reports "unavailable" for a path the GPU cannot run (Triton/Inductor need
compute capability >= 7.0) rather than crashing, because the pre-Volta
fallback is part of the story.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gpu_port  # noqa: E402

PATHS = ("full", "compiled", "triton", "nvrtc", "grouped")

# How many fp32 atomic element-adds each path issues per layer, as a
# function of (B, K, dim, nbytes). `nbytes` is the destination's row count
# (256 byte ids), `dim` the row width.
#
#   full/compiled/triton/nvrtc -- every one of the B*K*dim values is added
#       to a destination address with an ATOMIC add. The count is B*K*dim per
#       layer, and it does not depend on how the bytes are distributed: the
#       number of atomic instructions is the same, only their contention
#       differs.
#   grouped -- the contributions to one address are combined in the kernel
#       FIRST (in the order the reference adds them), so the address is
#       written ONCE, by one thread, with a plain store: ZERO atomics, and
#       256*dim read-modify-writes per layer instead.
def atomic_count(path, B, K, dim, layers, nbytes=256):
    """(atomic element-adds per step, plain RMW writes per step)."""
    per_layer_atomic = {
        "full": B * K * dim,
        "compiled": B * K * dim,
        "triton": B * K * dim,
        "nvrtc": B * K * dim,
        "grouped": 0,
    }.get(path, B * K * dim)
    per_layer_plain = (nbytes * dim) if path == "grouped" else 0
    return per_layer_atomic * layers, per_layer_plain * layers


def _zipf_rows(n, nbytes, seed=0):
    """A realistic byte-id draw: p(id) ~ 1/(rank+1), i.e. the top byte id
    takes ~12% of the rows, as the commonest byte of a text corpus does."""
    rng = np.random.default_rng(seed)
    p = 1.0 / (np.arange(nbytes, dtype=np.float64) + 1.0)
    p /= p.sum()
    return rng.choice(nbytes, size=n, p=p)


def fill_ring_b(m, mode="zipf", seed=0):
    """Give the ring realistic byte ids. `ring_b` is (B, K): the byte id of
    each stream's last K bytes. The credit's `rows` is that flattened, so
    the byte-id histogram of the credit IS the corpus's byte histogram."""
    B, K = m.ring_b.shape
    n = B * K
    if mode == "uniform":
        rng = np.random.default_rng(seed)
        ids = rng.integers(0, 256, size=n)
    elif mode == "single":
        ids = np.zeros(n, dtype=np.int64)      # the pathological case
    else:
        ids = _zipf_rows(n, 256, seed=seed)
    m.ring_b.copy_(torch.from_numpy(ids.reshape(B, K)).to(m.ring_b.device))


def _ensure_dest(m):
    """The credit's destination.

    `_ring_dest` is normally allocated by `step_batch` (it is the embed
    gradient buffer, (256, dim)). The harness times the credit OUTSIDE a
    step, so it has to make the destination itself -- the version that
    shipped with the task never did, and every path failed with
    "no attribute '_ring_dest'".
    """
    if getattr(m, "_ring_dest", None) is None:
        m._ring_dest = torch.zeros((256, m.dim), dtype=torch.float32,
                                   device=m.device)
    return m._ring_dest


def bench_path(m, path, iters, warmup=5):
    """Time one credit path on the live ring buffers.

    Returns (ms, ok, note). `ok` is False for a path this device cannot run:
    the note then says why, and the caller prints "unavailable".
    """
    i = 0
    l = m.layers[i]
    w = l.ring_w
    _ensure_dest(m)
    dlds = torch.randn((m.B, m.dim), device=m.device)
    rows = m.ring_b.reshape(m.B * m.ring_k)
    dec = torch.sigmoid(l.decay)
    m.ring_credit = path
    m._ring_resolved = None          # force a re-resolve for this path
    dev = torch.device(m.device)
    try:
        modes = getattr(gpu_port, "RING_CREDIT_MODES",
                        ("auto", "full", "compiled", "triton", "nvrtc"))
        if path not in modes:
            return None, False, "unavailable: this engine has no such path"
        eff = m._ring_mode()
        if eff != path and path in ("compiled", "triton", "nvrtc", "grouped"):
            return None, False, f"unavailable on this device (resolved to {eff})"
        for _ in range(warmup):
            m._scatter_ring_credit(i, w, dlds, rows, None, dec)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            m._scatter_ring_credit(i, w, dlds, rows, None, dec)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    except Exception as exc:                          # noqa: BLE001
        return None, False, f"failed: {type(exc).__name__}: {str(exc)[:80]}"
    return dt * 1e3, True, ""


def credit_result(m, path, seed=0):
    """Run the credit ONCE through `path` and return the destination it
    produced, for the bit-identity check against `full`."""
    torch.manual_seed(seed)
    l = m.layers[0]
    w = l.ring_w
    _ensure_dest(m)
    dlds = torch.randn((m.B, m.dim), device=m.device)
    rows = m.ring_b.reshape(m.B * m.ring_k)
    dec = torch.sigmoid(l.decay)
    m.ring_credit = path
    m._ring_resolved = None
    m._scatter_ring_credit(0, w, dlds, rows, None, dec)
    return m._ring_dest.detach().clone()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="a checkpoint to load")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--ring-k", type=int, default=18)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--rows", default="zipf", choices=("zipf", "uniform", "single"),
                    help="the byte-id distribution the ring carries")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    print("=" * 78)
    print("RING CREDIT, TIMED ALONE  (the engine's dominant cost?)")
    print("=" * 78)
    if dev.type == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"device: {torch.cuda.get_device_name()}  compute capability "
              f"{cap[0]}.{cap[1]}")
    else:
        print("device: CPU (the numbers below are NOT the GPU's)")
    print(f"shape: B={args.batch} K={args.ring_k} dim={args.dim} "
          f"layers={args.layers}   iters={args.iters}")
    print()

    if args.ckpt:
        d = np.load(args.ckpt)
        dim = int(d["Wd"].shape[1])
        layers = sum(1 for k in d.files if k.startswith("layer") and k.endswith(".W"))
        m = gpu_port.GpuModel(device=str(dev), dim=dim, layers=layers, seed=0)
        m.new_states(args.batch)
        m.load(args.ckpt)
        m.new_states(args.batch)
        print(f"loaded {args.ckpt} (dim={dim}, layers={layers})")
    else:
        m = gpu_port.GpuModel(device=str(dev), dim=args.dim, layers=args.layers,
                              seed=0, ring_k=args.ring_k)
        m.new_states(args.batch)

    K = m.ring_k
    fill_ring_b(m, args.rows)
    ids, counts = torch.unique(m.ring_b.reshape(-1), return_counts=True)
    top = int(counts.max())
    per_addr = args.batch * K * m.dim / (256 * m.dim)
    print(f"ring byte ids: {args.rows} distribution, {len(ids)} distinct ids, "
          f"the busiest address takes {top} of the {args.batch*K} rows "
          f"({top/(args.batch*K):.1%}) -> the MEAN contention is "
          f"{per_addr:.0f} atomics per address")
    print(f"the credit is called ONCE PER LAYER per step, so a path's "
          f"ms/call x {m.layercount} = its share of the step")
    print()

    hdr = (f"{'path':>10} | {'ms/call':>9} | {'ms/step':>9} | "
           f"{'atomic adds/step':>17} | {'G adds/s':>9} | {'adds/addr':>9} | "
           f"{'vs full':>8} | note")
    print(hdr)
    print("-" * len(hdr))
    rows_out = []
    for path in PATHS:
        atoms, plains = atomic_count(path, args.batch, K, m.dim, m.layercount)
        ms, ok, note = bench_path(m, path, args.iters)
        if not ok:
            print(f"{path:>10} | {'-':>9} | {'-':>9} | {atoms:>17,} | "
                  f"{'-':>9} | {per_addr:>9.0f} | {'-':>8} | {note}")
            continue
        rate = (atoms / (ms / 1e3) / 1e9) if atoms else 0.0
        print(f"{path:>10} | {ms:>9.3f} | {ms*m.layercount:>9.2f} | "
              f"{atoms:>17,} | {rate:>9.2f} | {per_addr:>9.0f} | "
              f"{'':>8} |")
        rows_out.append((path, ms, atoms, rate))

    # ---- the legality check: do the paths agree, bit for bit? --------------
    print()
    print("bit-identity of the credit's RESULT against `full` "
          "(the legality claim; on CUDA the atomic order is not defined, so a")
    print("last-bit difference there is the ORDER, not a bug -- and the "
          "grouped path is the one that matches numpy's order exactly):")
    try:
        ref = credit_result(m, "full")
        for path in PATHS:
            if path == "full":
                continue
            got = credit_result(m, path)
            same = torch.equal(ref, got)
            d = (ref - got).abs().max().item()
            rms = float(((ref - got) ** 2).mean().sqrt())
            print(f"  {path:>10}: {'BIT-IDENTICAL' if same else 'differs'} "
                  f"(max|d| {d:.3e}, rms {rms:.3e})")
    except Exception as exc:                          # noqa: BLE001
        print(f"  (skipped: {type(exc).__name__}: {str(exc)[:70]})")

    print()
    if rows_out:
        best = min(rows_out, key=lambda r: r[1])
        print(f"fastest path: {best[0]} at {best[1]:.3f} ms/call "
              f"= {best[1]*m.layercount:.2f} ms/step "
              f"({best[3]:.2f} G atomic adds/s)")
        # the traffic floor: the ring is (B,K,D) fp32, and the decay writes
        # it back, so a fused kernel must read AND write it.
        one = args.batch * K * m.dim * 4
        traffic = one * m.layercount
        bw = 320e9 if dev.type == "cuda" else 50e9
        print(f"the traffic floor for the same work: the ring is "
              f"{one/1e6:.1f} MB per layer, {traffic/1e6:.0f} MB per step "
              f"READ -> {traffic/bw*1e3:.2f} ms at {bw/1e9:.0f} GB/s")
        print(f"  and the decay writes the ring back, so a fused kernel moves "
              f"{2*traffic/1e6:.0f} MB/step -> {2*traffic/bw*1e3:.2f} ms")
        print("so the ceiling for a contention-free kernel is the floor above,")
        print("not the measured time: the gap is the atomic contention.")
    print()
    print("BATCH-SCALING NOTE: the credit's cost is B*K*dim per layer, so it")
    print("scales with B. The optimizer step's OTHER costs (the AdamW update")
    print("over 4.48M params, the launches) do not -- so a bigger batch buys")
    print("bytes per step only if the credit is not the wall.")


if __name__ == "__main__":
    main()
