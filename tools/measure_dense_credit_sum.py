#!/usr/bin/env python3
"""measure_dense_credit_sum.py -- the dense credit's contraction, REFORMULATED.

THE PROBLEM
-----------
The dense credit's contraction is

    torch.einsum("bvd,bd->vd", E, dlds)        # E: (B, 256, dim), dlds: (B, dim)

which is `out[v,d] = sum_b E[b,v,d] * dlds[b,d]`.  torch turns it into
131 072 batched 1x1x256 GEMVs and pays ~18x the cost of the memory traffic it
implies (116.10 ms/layer against 6.45 ms for the decay and 9.52 ms for the
elementwise-multiply-then-sum on the MX150, B=256, dim=512 -- see
docs/design/P3-GATE-IMPLEMENTATION.md section 3).

The reformulation

    (E * dlds[:, None, :]).sum(0)

computes the SAME contraction 12x faster.  It is NOT bit-identical: the two
sum the same B products in a DIFFERENT ORDER, and float32 addition is not
associative, so they differ in the last bits.

WHAT THIS SCRIPT MEASURES
-------------------------
Not "is it bit-identical" (it is not, and it cannot be) but:

  * how far each formulation is from the FLOAT64 TRUTH, and
  * how far each formulation is from the EXACT SUM OF THE FLOAT32 PRODUCTS
    (the "accumulation-only" error, which isolates the summation order from
    the input rounding -- every float32 product E[b,v,d]*dlds[b,d] is exactly
    representable in float64, so a float64 sum of them IS the exact sum of
    the float32 products).

The second number is the one that answers the question the change poses: the
reformulation changes ONLY the order of the additions, so the two numbers it
must be compared on are the two accumulation errors.  If the new one is at or
below the old one, the change costs nothing in accuracy.

Both the torch (GPU/CPU) and the numpy (CPU reference) paths are measured.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

F = torch.float32


def ref_float64(E, dlds):
    """The truth, in float64: sum_b E[b,v,d]*dlds[b,d], float64 inputs."""
    E64 = E.detach().cpu().numpy().astype(np.float64)
    d64 = dlds.detach().cpu().numpy().astype(np.float64)
    return np.einsum("bvd,bd->vd", E64, d64)


def ref_exact_of_float32_products(E, dlds):
    """The EXACT sum of the float32 products (float64 accumulation of the
    float32 terms): the reference that isolates the SUMMATION ORDER."""
    En = E.detach().cpu().numpy().astype(np.float32)
    dn = dlds.detach().cpu().numpy().astype(np.float32)
    # the products are exact in float64 (24-bit x 24-bit mantissa -> 48 bits)
    return (En.astype(np.float64) * dn.astype(np.float64)[:, None, :]).sum(0)


def dev(a, b):
    """(max abs, max rel) of a against b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    scale = max(float(np.abs(b).max()), 1e-30)
    return float(d.max()), float(d.max() / scale)


def bench(fn, reps, dev_):
    for _ in range(2):
        fn()
    if dev_ == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if dev_ == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shapes", default="64x512,256x512,256x64,16x512,3x8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", action="store_true",
                    help="also time the two formulations")
    args = ap.parse_args()
    dev_ = args.device
    print(f"device: {dev_}   torch {torch.__version__}")
    if dev_ == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}   "
              f"allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
    print(f"  dtypes: einsum out={F}, alt out="
          f"{(torch.zeros(1) * torch.zeros(1)).dtype}\n")
    print(f"{'B':>5} {'dim':>5} | {'einsum vs f64':>26} | "
          f"{'alt vs f64':>26} | {'accum-only err':>24}")
    print(f"{'':>5} {'':>5} | {'maxabs':>12} {'maxrel':>12} | "
          f"{'maxabs':>12} {'maxrel':>12} | {'einsum':>11} {'alt':>11}")
    rows = []
    for shape in args.shapes.split(","):
        B, D = (int(x) for x in shape.split("x"))
        g = torch.Generator(device="cpu").manual_seed(args.seed + B * 1000 + D)
        # a realistic trace: a decayed EMA in [0,1] with a byte spike
        E = (torch.rand((B, 256, D), generator=g, dtype=F) * 0.9).to(dev_)
        E[:, 0, :] += 1.0
        dlds = torch.randn((B, D), generator=g, dtype=F).to(dev_)
        truth = ref_float64(E, dlds)
        exact = ref_exact_of_float32_products(E, dlds)

        ein = torch.einsum("bvd,bd->vd", E, dlds)
        alt = (E * dlds[:, None, :]).sum(0)
        assert ein.dtype == alt.dtype == F, (ein.dtype, alt.dtype)

        e_abs, e_rel = dev(ein.cpu().numpy(), truth)
        a_abs, a_rel = dev(alt.cpu().numpy(), truth)
        e_acc, _ = dev(ein.cpu().numpy(), exact)
        a_acc, _ = dev(alt.cpu().numpy(), exact)
        e_vs_a, e_vs_a_rel = dev(ein.cpu().numpy(), alt.cpu().numpy())
        print(f"{B:>5} {D:>5} | {e_abs:12.3e} {e_rel:12.3e} | "
              f"{a_abs:12.3e} {a_rel:12.3e} | {e_acc:11.3e} {a_acc:11.3e}")
        rows.append((B, D, e_abs, e_rel, a_abs, a_rel, e_acc, a_acc,
                     e_vs_a, e_vs_a_rel))
        if args.time:
            t_e = bench(lambda: torch.einsum("bvd,bd->vd", E, dlds),
                        args.reps, dev_)
            t_a = bench(lambda: (E * dlds[:, None, :]).sum(0), args.reps, dev_)
            # the same contraction as a batched GEMV over d (out[:,d] =
            # E[:,:,d].T @ dlds[:,d]); a materialising copy, so it is a
            # reference point, not a proposal
            Ed = E.permute(2, 1, 0).contiguous()      # (D, 256, B)
            w = dlds.t().contiguous().unsqueeze(2)    # (D, B, 1)
            t_g = bench(lambda: torch.bmm(Ed, w).squeeze(2).t(),
                        args.reps, dev_)
            g_out = torch.bmm(Ed, w).squeeze(2).t()
            g_abs, g_rel = dev(g_out.cpu().numpy(), truth)
            g_acc, _ = dev(g_out.cpu().numpy(), exact)
            print(f"      time: einsum {t_e * 1000:8.3f} ms | "
                  f"alt {t_a * 1000:8.3f} ms | "
                  f"bmm-over-d {t_g * 1000:8.3f} ms")
            print(f"      bmm-over-d: vs f64 maxrel {g_rel:.3e}, "
                  f"accum-only {g_acc:.3e}")

    # ---- the numpy reference path -------------------------------------
    print("\nnumpy reference (CPU):")
    for shape in args.shapes.split(","):
        B, D = (int(x) for x in shape.split("x"))
        g = np.random.default_rng(args.seed + B * 1000 + D)
        E = (g.random((B, 256, D)).astype(np.float32) * 0.9)
        E[:, 0, :] += 1.0
        dlds = g.standard_normal((B, D)).astype(np.float32)
        truth = np.einsum("bvd,bd->vd", E.astype(np.float64),
                           dlds.astype(np.float64))
        exact = (E.astype(np.float64) * dlds.astype(np.float64)[:, None, :]
                 ).sum(0)
        n_ein = np.einsum("bvd,bd->vd", E, dlds)
        n_alt = (E * dlds[:, None, :]).sum(0)
        e_abs, e_rel = dev(n_ein, truth)
        a_abs, a_rel = dev(n_alt, truth)
        e_acc, _ = dev(n_ein, exact)
        a_acc, _ = dev(n_alt, exact)
        print(f"{B:>5} {D:>5} | {e_abs:12.3e} {e_rel:12.3e} | "
              f"{a_abs:12.3e} {a_rel:12.3e} | {e_acc:11.3e} {a_acc:11.3e}")
        if args.time:
            t_e = bench(lambda: np.einsum("bvd,bd->vd", E, dlds), args.reps, "cpu")
            t_a = bench(lambda: (E * dlds[:, None, :]).sum(0), args.reps, "cpu")
            print(f"      time: np.einsum {t_e * 1000:8.3f} ms | "
                  f"np alt {t_a * 1000:8.3f} ms")

    print("\nVERDICT (accumulation-only error: lower is better; the change is "
          "free only if alt <= einsum)")
    worse = [r for r in rows if r[7] > r[6] * 1.05]
    for B, D, e_abs, e_rel, a_abs, a_rel, e_acc, a_acc, eva, evar in rows:
        flag = "WORSE" if a_acc > e_acc * 1.05 else (
            "better" if a_acc < e_acc * 0.95 else "same")
        print(f"  B={B:<5} dim={D:<5} einsum {e_acc:.3e}  alt {a_acc:.3e}  "
              f"-> {flag}   (einsum vs alt: {eva:.3e} abs, {evar:.3e} rel)")
    print(f"\n{'FORMULATION IS WORSE' if worse else 'FORMULATION IS NOT WORSE'}"
          f" in {len(worse)}/{len(rows)} settings")


if __name__ == "__main__":
    main()
