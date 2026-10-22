#!/usr/bin/env python3
"""measure_p3.py -- the P3 dense credit's THROUGHPUT and MEMORY, measured.

The design (docs/design/P3-GATE-ANALYSIS.md 5.3) left exactly ONE number
unmeasured: the dense credit's cost per step, and its memory.  Everything
else about the recommendation was measured.  This script measures it, on the
card that is available, at the batch sizes the card can hold.

WHAT IS COMPARED
  * the RING path (the incumbent): a K-lag ring buffer + the tail's exact EMA
    for the hot dims;
  * the DENSE path (the P2 arm, and the P3 gated arm's credit path): the
    exact (B, 256, dim) eligibility trace per layer.

at the SAME batch, at two inits:
  * the DEFAULT init (every half-life ~1 B): ring_k ~ 15, so the ring is
    cheap and the dense trace is ~17x its buffer;
  * the P4 init (half-lives 8..512 B): ring_k saturates at 256, so the ring
    pays the SAME B*256*dim*4 bill as the dense trace -- this is the D5
    configuration, and the regime in which the dense trace is the cheaper of
    the two (the ring ALSO carries the tail's accumulator on top).

Every number is a mean over `--steps` timed steps after `--warmup` untimed
ones, with a cuda synchronize around the timed region, and the peak is
torch.cuda.max_memory_allocated (so it is the engine's own buffers, not the
CUDA context).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_port


def measure(tag, dim, layers, B, mode, half_life=None, gated=False,
            steps=20, warmup=5, seed=0, device="cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    kw = dict(device=device, dim=dim, layers=layers, seed=seed, gated=gated)
    if mode is not None:
        kw["trace_mode"] = mode
    if half_life is not None:
        kw["half_life_init"] = half_life
    m = gpu_port.GpuModel(**kw)
    m.new_states(B)
    rng = np.random.default_rng(0)
    cs = torch.from_numpy(rng.integers(1, 250, (B, 1)).astype(np.int64)).to(device)
    ns = torch.from_numpy(rng.integers(1, 250, (B, 1)).astype(np.int64)).to(device)
    ends = torch.zeros((B, 1), device=device)
    t0 = None
    try:
        for i in range(warmup + steps):
            if i == warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
            m.step_batch(cs, ns, ends)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        print(f"{tag}: OOM ({str(e).splitlines()[0][:70]})")
        del m
        torch.cuda.empty_cache()
        return None
    dt = (time.perf_counter() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"{tag:<52} {dt * 1000:8.2f} ms/step   peak {peak:6.3f} GB"
          f"   ring_k={m.ring_k}")
    sys.stdout.flush()
    del m
    torch.cuda.empty_cache()
    return dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--batch", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--p4", type=float, nargs=2, default=[8.0, 512.0],
                    help="the P4 half-life range (bytes)")
    args = ap.parse_args()
    d, L = args.dim, args.layers
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"dim={d} layers={L}")
    print(f"dense bill per layer: B*256*dim*4 = "
          f"{args.batch[0] * 256 * d * 4 / 1e6:.0f} MB at B={args.batch[0]}")
    for B in args.batch:
        print(f"\n--- B = {B} ---")
        measure("ring, default init (ring_k ~ 15)", d, L, B, "ring",
                steps=args.steps, warmup=args.warmup)
        measure("dense, default init (the P3 credit path)", d, L, B, "dense",
                steps=args.steps, warmup=args.warmup)
        measure("gated arm (dense credit + the gate)", d, L, B, "dense",
                gated=True, steps=args.steps, warmup=args.warmup)
        measure("ring, P4 init (ring_k = 256) + tail", d, L, B, "ring",
                half_life=tuple(args.p4), steps=args.steps, warmup=args.warmup)
        measure("dense, P4 init", d, L, B, "dense", half_life=tuple(args.p4),
                steps=args.steps, warmup=args.warmup)


if __name__ == "__main__":
    main()
