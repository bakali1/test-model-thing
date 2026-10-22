#!/usr/bin/env python3
"""measure_p3_credit.py -- where the dense credit's time actually goes.

The end-to-end measurement (tools/measure_p3.py) says the dense credit path
costs ~2.0 s/step at B=256, dim=512, layers=16 on the MX150, against the
ring's 44 ms.  That is 45x, and the design's projection was 2-4 ms at T4
bandwidth -- so either the projection was wrong or the cost is not where the
projection assumed.  This script separates the three operations the dense
credit performs, per layer:

    E.mul_(dec[:, None, :])            the decay  (a full pass over E)
    einsum("bvd,bd->vd", E, dlds)      the credit (a contraction over b)
    E[ar, c_arr, :] += 1.0             the write  (B scattered rows)

and, for reference, the shape-reformulated credit
    (dlds @ E.reshape(B, 256*dim)).reshape(256, dim)
which is the same contraction with a different (GEMM) shape.  It is NOT
bit-identical (a different summation order), so it is a MEASUREMENT here, not
a change.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

B = int(os.environ.get("B", "256"))
DIM = int(os.environ.get("DIM", "512"))
LAYERS = int(os.environ.get("LAYERS", "16"))
REPS = int(os.environ.get("REPS", "10"))
F = torch.float32
dev = "cuda"


def bench(name, fn, reps=REPS):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    print(f"  {name:<46} {dt * 1000:9.3f} ms")
    return dt


def main():
    print(f"B={B} dim={DIM} layers={LAYERS}  (per-layer numbers, then the "
          f"x{LAYERS} total)")
    E = torch.zeros((B, 256, DIM), dtype=F, device=dev)
    E.normal_()
    dec = torch.rand((B, DIM), dtype=F, device=dev) * 0.5 + 0.4
    dlds = torch.randn((B, DIM), dtype=F, device=dev)
    ar = torch.arange(B, device=dev)
    c_arr = torch.randint(0, 256, (B,), device=dev)
    out = torch.zeros((256, DIM), dtype=F, device=dev)
    Ebytes = E.numel() * 4

    print(f"  E is {Ebytes / 1e6:.1f} MB")
    t_mul = bench("E.mul_(dec[:, None, :])", lambda: E.mul_(dec[:, None, :]))
    t_ein = bench('einsum("bvd,bd->vd", E, dlds)',
                  lambda: torch.einsum("bvd,bd->vd", E, dlds))
    t_add = bench("E[ar, c_arr, :] += 1.0", lambda: E.__setitem__(
        (ar, c_arr, slice(None)), E[ar, c_arr, :] + 1.0))
    t_gemm = bench("(E * dlds[:, None, :]).sum(0)   [ALT: same contraction,",
                   lambda: (E * dlds[:, None, :]).sum(0))
    t_gemm2 = bench("  ... a different summation order!]",
                    lambda: (E * dlds[:, None, :]).sum(0))
    print(f"\n  per layer, as shipped : {(t_mul + t_ein + t_add) * 1000:9.3f} ms"
          f"   -> x{LAYERS} = {(t_mul + t_ein + t_add) * LAYERS * 1000:.1f} ms")
    print(f"  per layer, ALT einsum : {(t_mul + t_gemm + t_add) * 1000:9.3f} ms"
          f"   -> x{LAYERS} = {(t_mul + t_gemm + t_add) * LAYERS * 1000:.1f} ms")
    print(f"\n  the decay alone moves {Ebytes * 2 / 1e9:.3f} GB per layer "
          f"({Ebytes * 2 * LAYERS / 1e9:.2f} GB per step)")
    print(f"  the einsum reads {Ebytes / 1e9:.3f} GB per layer "
          f"({Ebytes * LAYERS / 1e9:.2f} GB per step)")


if __name__ == "__main__":
    main()
