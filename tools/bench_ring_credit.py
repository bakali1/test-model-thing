"""tools/bench_ring_credit.py -- time the ring credit ALONE, per path.

WHY THIS EXISTS. The engine's optimizer step is dominated by one thing: the
ring credit's atomic scatter. At B=1024, K=18, dim=512, 16 layers the credit
computes `vals = w * dlds` over the whole (B, K, dim) tensor and then does
`dest.index_add_(0, rows, vals)` -- so every element of that tensor is an
fp32 atomic add: 9.44M per layer, 151M per step, into a (256, dim) = 131K
address destination, i.e. ~72 contending atomics per address.

The engine measures ~83 ms per step on a Colab T4 and ~300 ms on an MX150.
The traffic is only ~604 MB/step (1.9 ms at the T4's 320 GB/s) and the
model's FLOPs are trivial, so the hypothesis is that CONTENDED ATOMICS are
the wall -- and the two machines' similar-but-not-equal step times (a 3.7x
gap for a 4.3x FLOP gap and a 20x SM gap) are what an atomic limit looks
like.

This harness measures the credit on its own, so the step time is not diluted
by everything else, and it prints the atomic arithmetic next to the
measurement -- the number the hypothesis is actually about.

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

PATHS = ("full", "compiled", "triton", "nvrtc")


def atomic_count(path, B, K, dim, layers):
    """fp32 atomic element-adds per optimizer step for a credit path.

    Every existing path scatters the full (B, K, dim) tensor, so the count is
    B*K*dim per layer. A path that reduced the K axis in registers would
    issue B*dim per layer instead -- but see the note: the K lag terms have
    DIFFERENT byte ids, so that reduction is not legal without changing the
    arithmetic.
    """
    per_layer = B * K * dim
    return per_layer * layers


def _is_cuda(m):
    """The engine stores its device as a STRING, not a torch.device."""
    return str(m.device).startswith("cuda")


def bench_path(m, path, iters, warmup=5):
    """Time one credit path on the live ring buffers. Returns (ms, ok, note)."""
    i = 0
    l = m.layers[i]
    w = l.ring_w
    dlds = torch.zeros((m.B, m.dim), device=m.device)
    rows = m.ring_b.reshape(m.B * m.ring_k)
    dec = torch.sigmoid(l.decay)
    m.ring_credit = path
    m._ring_resolved = None          # force a re-resolve for this path
    # The engine sets the scatter destination per step (it aliases the
    # embedding's gradient); outside a step it does not exist, and the NVRTC
    # kernel needs the pointer. A zero buffer of the right shape is enough
    # for a timing run.
    if not hasattr(m, "_ring_dest") or m._ring_dest is None:
        m._ring_dest = torch.zeros((256, m.dim), device=m.device,
                                   dtype=torch.float32)
    try:
        eff = m._ring_mode()
        if eff != path and path in ("compiled", "triton"):
            return None, False, f"unavailable on this device (resolved to {eff})"
        for _ in range(warmup):
            m._scatter_ring_credit(i, w, dlds, rows, None, dec)
        if _is_cuda(m):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            m._scatter_ring_credit(i, w, dlds, rows, None, dec)
        if _is_cuda(m):
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    except Exception as exc:                          # noqa: BLE001
        return None, False, f"failed: {type(exc).__name__}: {str(exc)[:80]}"
    return dt * 1e3, True, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="a checkpoint to load")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--ring-k", type=int, default=18)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    print("=" * 78)
    print("RING CREDIT, TIMED ALONE  (the engine's dominant cost)")
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
    print(f"\n{'path':>10} | {'ms/call':>9} | {'atomic adds/step':>17} | "
          f"{'G adds/s':>9} | note")
    print("-" * 78)
    rows = []
    for path in PATHS:
        ms, ok, note = bench_path(m, path, args.iters)
        if not ok:
            print(f"{path:>10} | {'-':>9} | {atomic_count(path, args.batch, K, m.dim, m.layercount):>17,} |"
                  f" {'-':>9} | {note}")
            continue
        atoms = atomic_count(path, args.batch, K, m.dim, m.layercount)
        rate = atoms / (ms / 1e3) / 1e9
        print(f"{path:>10} | {ms:>9.3f} | {atoms:>17,} | {rate:>9.2f} |")
        rows.append((path, ms, atoms, rate))

    print()
    if rows:
        best = min(rows, key=lambda r: r[1])
        print(f"fastest path: {best[0]} at {best[1]:.3f} ms/call "
              f"({best[3]:.2f} G atomic adds/s)")
        # what a fused path COULD reach if the contention were removed: the
        # traffic floor. The ring read is B*K*dim*4 bytes per layer.
        traffic = args.batch * K * m.dim * 4 * m.layercount
        bw = 320e9 if dev.type == "cuda" else 50e9
        print(f"the traffic floor for the same work: {traffic/1e6:.0f} MB/step "
              f"-> {traffic/bw*1e3:.2f} ms at {bw/1e9:.0f} GB/s")
        print("so the ceiling for a contention-free kernel is the floor above,")
        print("not the measured time: the gap is the atomic contention.")
    print()
    print("BATCH-SCALING NOTE: the credit's cost is B*K*dim per layer, so it")
    print("scales with B. The optimizer step's OTHER costs (the AdamW update")
    print("over 4.48M params, the launches) do not -- so a bigger batch buys")
    print("bytes per step only if the credit is not the wall.")


if __name__ == "__main__":
    main()
