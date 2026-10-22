#!/usr/bin/env python3
"""bench_credit_shapes.py -- every way of writing the dense credit's
contraction, timed and scored against the float64 truth.

THE QUESTION
    The dense credit's contraction is

        out[v,d] = sum_b E[b,v,d] * dlds[b,d]        (E: (B, 256, dim))

    and torch.einsum("bvd,bd->vd", E, dlds) lowers it to 131 072 batched
    1x1x256 GEMVs: 116.10 ms per layer on the MX150 at B=256, dim=512,
    against 6.45 ms for the trace's decay.  The elementwise form
    (E * dlds[:,None,:]).sum(0) does the same contraction in 9.52 ms.

    The brief asks for a SECOND option: keep the einsum but reduce its shape
    cost.  This script times every candidate -- and scores each one against
    the float64 truth, so "fastest" and "at the float32 floor" are answered
    together, never separately.

THE CANDIDATES
    einsum      torch.einsum("bvd,bd->vd", E, dlds)        the incumbent
    alt         (E * dlds[:, None, :]).sum(0)               multiply-then-sum
    bmm_d       E.permute(2,1,0).contiguous() -> bmm       the GEMM shape,
                (dlds.t() as a (dim,B,1) operand)           WITH its copy
    bmm_d_nc    the same, the copy hoisted out of the timer  (the best case
                for that formulation, to price the copy itself)
    einsum_v    E.permute(1,0,2).contiguous() -> einsum     the einsum on a
                ("vbd,bd->vd")                             contiguous view,
                                                           WITH its copy
    baddbmm     baddbmm(0, Ed, w)                          the fused-add GEMM
                                                           WITH its copy

Every variant's timing includes EVERY copy it needs: a formulation that has
to transpose a 134 MB trace to be fast has to pay for the transpose, and
the honest number is the one that includes it.

The float32 floor: every float32 product E[b,v,d]*dlds[b,d] is exact in
float64, so a float64 sum of them is the EXACT sum of the float32 products.
The deviation of a float32 result from that is the SUMMATION-ORDER error
alone -- the quantity the reassociation can move.  The 0.5*eps*|truth| term
is the float32 rounding floor of the RESULT itself: no float32
implementation of this contraction can be closer than that to the truth.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

F = torch.float32


def truth64(E, dlds):
    En = E.detach().cpu().numpy().astype(np.float64)
    dn = dlds.detach().cpu().numpy().astype(np.float64)
    return np.einsum("bvd,bd->vd", En, dn)


def exact32(E, dlds):
    En = E.detach().cpu().numpy().astype(np.float32).astype(np.float64)
    dn = dlds.detach().cpu().numpy().astype(np.float32).astype(np.float64)
    return (En * dn[:, None, :]).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    B, D = args.B, args.dim
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    E = (torch.rand((B, 256, D), generator=g, dtype=F) * 0.9).to(dev)
    E[:, 0, :] += 1.0
    dlds = torch.randn((B, D), generator=g, dtype=F).to(dev)

    Ed = E.permute(2, 1, 0).contiguous()          # (D, 256, B)
    Ev = E.permute(1, 0, 2).contiguous()          # (256, B, D)
    w = dlds.t().contiguous().unsqueeze(2)        # (D, B, 1)
    zero = torch.zeros((D, 256, 1), dtype=F, device=dev)

    variants = {
        "einsum  torch.einsum('bvd,bd->vd')":
            lambda: torch.einsum("bvd,bd->vd", E, dlds),
        "alt     (E*dlds[:,None,:]).sum(0)":
            lambda: (E * dlds[:, None, :]).sum(0),
        "bmm_d   permute+contiguous -> bmm (copy INCLUDED)":
            lambda: torch.bmm(E.permute(2, 1, 0).contiguous(), w).squeeze(2).t(),
        "bmm_d_nc  bmm, the copy hoisted (best case)":
            lambda: torch.bmm(Ed, w).squeeze(2).t(),
        "einsum_v  permute+contiguous -> einsum (copy INCLUDED)":
            lambda: torch.einsum("vbd,bd->vd",
                                  E.permute(1, 0, 2).contiguous(), dlds),
        "baddbmm  permute+contiguous -> baddbmm (copy INCLUDED)":
            lambda: torch.baddbmm(zero, E.permute(2, 1, 0).contiguous(),
                                   w).squeeze(2).t(),
    }

    print(f"device={dev}  B={B} dim={D}  E is {E.numel() * 4 / 1e6:.1f} MB"
          f"  reps={args.reps} (interleaved, so a busy GPU hits every "
          f"variant equally)")
    t = truth64(E, dlds)
    ex = exact32(E, dlds)
    scale = max(float(np.abs(t).max()), 1e-30)
    floor_abs = 0.5 * float(np.finfo(np.float32).eps) * scale
    print(f"  the float32 rounding floor of the RESULT: {floor_abs:.3e} abs "
          f"({floor_abs / scale:.3e} rel) -- the best any float32 "
          f"implementation can do")

    times = {k: [] for k in variants}
    outs = {}
    for r in range(args.reps + 2):
        for k, fn in variants.items():
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if r >= 2:
                times[k].append(dt)
            outs[k] = out.detach().cpu().numpy().astype(np.float64)
    print(f"\n  {'variant':<52} {'ms':>9} {'vs truth':>11} {'vs truth':>11} "
          f"{'order-only':>11} {'x floor':>8}")
    print(f"  {'':<52} {'':>9} {'abs':>11} {'rel':>11} {'':>11} {'':>8}")
    for k in variants:
        o = outs[k]
        d = float(np.abs(o - t).max())
        rel = d / scale
        dord = float(np.abs(o - ex).max())
        ms = float(np.median(times[k]))
        print(f"  {k:<52} {ms:9.3f} {d:11.3e} {rel:11.3e} {dord:11.3e} "
              f"{d / floor_abs:8.1f}")
    print("\n  'order-only' = the deviation from the EXACT sum of the float32 "
          "products (the summation order alone).")
    print("  'x floor' = how many float32 rounding floors of the result the "
          "deviation is.")


if __name__ == "__main__":
    main()
