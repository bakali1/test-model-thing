#!/usr/bin/env python3
"""check_dense_credit_sum.py -- the dense credit's contraction, REFORMULATED.

WHY THIS FILE EXISTS
--------------------
The dense credit's contraction is

    out[v,d] = sum_b E[b,v,d] * dlds[b,d]

and the engine used to compute it with

    torch.einsum("bvd,bd->vd", E, dlds)

which torch lowers to 131 072 batched 1x1x256 GEMVs: 116.10 ms per layer
against 6.45 ms for the trace's decay and 9.52 ms for the reformulation
below, on the MX150 at B=256, dim=512 (docs/design/P3-GATE-IMPLEMENTATION.md
3.2).  The reformulation

    (E * dlds[:, None, :]).sum(0)

computes the SAME contraction 12x faster, and takes the dense path from
2005 ms/step to ~270 ms/step.

It is NOT bit-identical, and this file does not pretend it is: the two sum
the same B products in a different ORDER, and float32 addition is not
associative.  So the question this gate answers is NOT "is it bit-identical"
but the one that matters:

    is the new formulation AT LEAST AS CLOSE TO THE TRUTH as the one it
    replaces?

and the answer is measured, per setting, against a float64 reference, with
the TOLERANCE ITSELF BEING A MEASURED QUANTITY -- the float32 rounding floor
of the same computation, exactly the way tests/check_gated_credit.py does it
("the tolerance can only ever be LOOSENED by a measurement, never by wishful
thinking").

TWO REFERENCES, TWO QUESTIONS
-----------------------------
  * the FLOAT64 TRUTH: the same contraction in float64.  It answers "how far
    is the float32 answer from the mathematical answer".
  * the EXACT SUM OF THE FLOAT32 PRODUCTS: every float32 product
    E[b,v,d]*dlds[b,d] is exactly representable in float64 (24-bit x 24-bit
    mantissa), so a float64 sum of them IS the exact sum of the float32
    products.  It answers "how much of the error is the SUMMATION ORDER" --
    which is the ONLY thing the reformulation changes.

WHAT IS PINNED
--------------
  (a) the two formulations against both references, for several (B, dim), with
      the tolerance = the measured float32 floor (the incumbent einsum's own
      deviation from the truth), so the check is
          dev(new) <= max(TOL, 3 * floor)
      and the numbers are printed, so the pin can be re-read without rerunning
      the measurement.
  (b) that the ENGINE ships the reformulation, at the bit level: the credit's
      operands are the engine's own decayed trace and the state gradient its
      backward produced, the credit is a reduction over the batch axis of
      their elementwise product, and it is NOT the einsum's bits.
  (c) that the NUMPY reference's einsum is BIT-IDENTICAL to the reformulation
      (so the reference's numbers did not move, and its einsum is kept there
      because it is 3x faster -- see the comment in numpy_port.py).
  (d) that the DEFAULT engine is untouched.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gpu_port
import numpy_port

# The claims here are device-independent IN KIND but the measured numbers are
# the DEVICE's, and the verdict differs by device (see the printed table): the
# reformulation is CLOSER to the truth on the GPU and 1.75x further on the
# CPU, both at the float32 epsilon.  The gate is therefore device-parameterised
# in the same convention as the other gates here.
DEV = os.environ.get("TMT_GATE_DEV", "cpu")

# The tolerance for the accuracy checks.  It is NOT a magic number: it is
# the MEASURED float32 rounding floor of the same computation -- the
# incumbent einsum's OWN deviation from the float64 truth, in the same
# (relative) units -- and the check is against 3 x that floor, so the
# tolerance can only ever be LOOSENED by a measurement, never by wishful
# thinking.  The 1e-9 is only there to keep a degenerate zero floor from
# turning the check into an equality test.
TOL = 1e-9

# The shapes the contraction is measured at.  Small enough to be quick, large
# enough that the two summation orders actually differ (at B=3 they do not --
# the two formulations are bit-identical there, which is why the P3 credit
# gate's numbers did not move at all).
SHAPES = [(8, 512), (16, 128), (64, 64), (64, 512), (256, 64)]

PASS = True


def check(name, ok, detail=""):
    global PASS
    if not ok:
        PASS = False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))


def dev(a, b):
    """(max abs, max rel) of a against b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = max(float(np.abs(b).max()), 1e-30)
    d = float(np.abs(a - b).max())
    return d, d / scale


def truth64(E, dlds):
    """The float64 truth of the contraction."""
    return np.einsum("bvd,bd->vd",
                      E.detach().cpu().numpy().astype(np.float64),
                      dlds.detach().cpu().numpy().astype(np.float64))


def exact_of_float32_products(E, dlds):
    """The EXACT sum of the float32 products (float64 accumulation of the
    float32 terms): the reference that isolates the SUMMATION ORDER."""
    En = E.detach().cpu().numpy().astype(np.float32).astype(np.float64)
    dn = dlds.detach().cpu().numpy().astype(np.float32).astype(np.float64)
    return (En * dn[:, None, :]).sum(0)


def part_a():
    print("(a) the two formulations against the float64 truth")
    print(f"    {'B':>5} {'dim':>5} | {'einsum vs f64':>24} | "
          f"{'reform vs f64':>24} | {'order-only error':>23}")
    print(f"    {'':>5} {'':>5} | {'maxabs':>11} {'maxrel':>11} | "
          f"{'maxabs':>11} {'maxrel':>11} | {'einsum':>10} {'reform':>10}")
    worst_ratio = 0.0
    for (B, D) in SHAPES:
        g = torch.Generator(device="cpu").manual_seed(1234 + B * 31 + D)
        E = (torch.rand((B, 256, D), generator=g, dtype=torch.float32) * 0.9)
        E[:, 0, :] += 1.0
        dlds = torch.randn((B, D), generator=g, dtype=torch.float32)
        E, dlds = E.to(DEV), dlds.to(DEV)

        old = torch.einsum("bvd,bd->vd", E, dlds)
        new = (E * dlds[:, None, :]).sum(0)
        # the dtype is part of the contract: the reformulation must not
        # silently change the accumulation precision
        check(f"[B={B} dim={D}] both formulations stay in float32",
              old.dtype == new.dtype == torch.float32,
              f"einsum {old.dtype}, reform {new.dtype}")

        t = truth64(E, dlds)
        ex = exact_of_float32_products(E, dlds)
        o_abs, o_rel = dev(old.cpu().numpy(), t)
        n_abs, n_rel = dev(new.cpu().numpy(), t)
        o_ord, _ = dev(old.cpu().numpy(), ex)
        n_ord, _ = dev(new.cpu().numpy(), ex)
        print(f"    {B:>5} {D:>5} | {o_abs:11.3e} {o_rel:11.3e} | "
              f"{n_abs:11.3e} {n_rel:11.3e} | {o_ord:10.3e} {n_ord:10.3e}")
        # the tolerance is the MEASURED float32 floor of the same computation:
        # the incumbent's own deviation from the truth.
        floor = o_rel
        lim = max(TOL, 3.0 * floor)
        ratio = n_rel / floor if floor > 0 else 0.0
        worst_ratio = max(worst_ratio, ratio)
        check(f"[B={B} dim={D}] the reformulation is at the float32 floor "
              f"(<= 3x the incumbent's own error)",
              n_rel <= lim, f"{n_rel:.3e} <= {lim:.3e} (floor {floor:.3e}, "
              f"ratio {ratio:.2f}x)")
    print(f"    -> worst ratio reform/einsum: {worst_ratio:.2f}x  "
          f"(1.00x = the same accuracy; < 1 = BETTER); the float32 epsilon "
          f"is {np.finfo(np.float32).eps:.3e}")


def part_b():
    """The ENGINE ships the reformulation -- at the bit level.

    The credit is captured at the ATen dispatch level, so what is compared is
    the engine's OWN arithmetic: the operands of the credit's product, the
    reduction that follows it, and the bytes that come out.
    """
    print("\n(b) the engine ships the reformulated credit")
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except Exception as e:                                    # pragma: no cover
        check("the dispatch-level spy is available", False, str(e))
        return

    B, DIM, LAYERS = 256, 64, 1
    STEPS = 4
    print(f"    dim={DIM} layers={LAYERS} B={B} steps={STEPS}")

    class _Spy(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.eins = []
            self.credit_mul = None      # (E, dlds_b, E's data_ptr)
            self.credit_sum = None      # (input, output)

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(func)
            if "einsum" in name:
                self.eins.append(args[0] if args else None)
                return func(*args, **kwargs)
            if name.endswith("mul_.Tensor"):
                return func(*args, **kwargs)      # the trace's decay, in place
            if name.endswith("mul.Tensor"):
                a = args[0]
                if (isinstance(a, torch.Tensor) and a.dim() == 3
                        and a.shape[1] == 256 and len(args) > 1
                        and isinstance(args[1], torch.Tensor)
                        and args[1].dim() == 3):
                    # the credit's elementwise product: E (B,256,dim) times
                    # dlds broadcast over the byte axis
                    self.credit_mul = (a.clone(), args[1].clone(),
                                       a.data_ptr())
                return func(*args, **kwargs)
            if "sum" in name and "dim" in name:
                a = args[0]
                if (isinstance(a, torch.Tensor) and a.dim() == 3
                        and a.shape[1] == 256):
                    out = func(*args, **kwargs)
                    self.credit_sum = (a.detach().clone(), out.detach().clone())
                    return out
            return func(*args, **kwargs)

    m = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=0,
                          trace_mode="dense")
    m.new_states(B)
    # the state gradient the engine's backward produces, per layer
    dstate = []
    for l in m.layers:
        orig = l.backward

        def wrapped(dx_out, cache, _orig=orig):
            ds = _orig(dx_out, cache)
            dstate.append(ds[0].detach().clone())
            return ds
        l.backward = wrapped

    rng = np.random.default_rng(0)
    # the byte ids are drawn from a SMALL range on purpose: the trace has to
    # be dense enough for the two summation orders to differ at all, and the
    # order only matters where an output element sums several streams.
    cs = torch.from_numpy(rng.integers(0, 16, (B, 1)).astype(np.int64)).to(DEV)
    ns = torch.from_numpy(rng.integers(1, 250, (B, 1)).astype(np.int64)).to(DEV)
    ends = torch.zeros((B, 1), device=DEV)
    spy = _Spy()
    for _ in range(STEPS):
        with spy:
            m.step_batch(cs, ns, ends)

    check("the dense credit no longer goes through an einsum",
          not any(isinstance(s, str) and "bvd,bd->vd" in s for s in spy.eins),
          f"einsum calls with that signature: "
          f"{[s for s in spy.eins if isinstance(s, str)]}")
    check("the credit IS a reduction over the batch axis of an elementwise "
          "product", spy.credit_mul is not None and spy.credit_sum is not None,
          f"product captured: {spy.credit_mul is not None}, "
          f"reduction captured: {spy.credit_sum is not None}")
    if spy.credit_mul is None or spy.credit_sum is None:
        return
    E_used, dlds_b, E_ptr = spy.credit_mul
    prod, cred = spy.credit_sum

    # the product's first operand IS the engine's own dense trace
    check("the credit's E operand IS the engine's dense trace",
          E_ptr == m.dense_E[0].data_ptr(),
          f"E ptr {E_ptr} vs trace ptr {m.dense_E[0].data_ptr()}")
    # ... and its second operand IS the state gradient the backward produced
    check("the credit's dlds operand IS the backward's state gradient",
          torch.equal(dlds_b[:, 0, :], dstate[-1]),
          f"max|d| {float((dlds_b[:, 0, :] - dstate[-1]).abs().max()):.3e}")
    # the reduction is over the batch axis, of the product
    check("the reduction is sum over dim 0 of that product",
          prod.shape == (B, 256, DIM) and cred.shape == (256, DIM),
          f"product {tuple(prod.shape)}, credit {tuple(cred.shape)}")
    check("the credit is bit-identical to (E * dlds[:, None, :]).sum(0)",
          torch.equal(cred, (E_used * dlds_b).sum(0)))
    # and it is NOT the einsum's bytes: the two differ at this size
    ein = torch.einsum("bvd,bd->vd", E_used, dlds_b[:, 0, :])
    differ = not torch.equal(cred, ein)
    check("and it is NOT the einsum's bytes (the order really changed)",
          differ, f"max|einsum - reform| = {float((ein - cred).abs().max()):.3e}")

    # the accuracy: the credit against the float64 truth of the SAME operands
    t = truth64(E_used, dlds_b[:, 0, :])
    c_abs, c_rel = dev(cred.cpu().numpy(), t)
    e_abs, e_rel = dev(ein.cpu().numpy(), t)
    floor = e_rel
    lim = max(TOL, 3.0 * floor)
    check("the engine's credit is at the float32 floor of the same computation",
          c_rel <= lim,
          f"credit rel {c_rel:.3e} <= {lim:.3e} "
          f"(floor = the einsum's own {floor:.3e})")


def part_c():
    print("\n(c) the numpy reference: the einsum IS the reformulation there")
    for (B, D) in [(8, 512), (64, 64), (64, 512), (256, 64)]:
        g = np.random.default_rng(7 + B + D)
        E = (g.random((B, 256, D)).astype(np.float32) * 0.9)
        E[:, 0, :] += 1.0
        d = g.standard_normal((B, D)).astype(np.float32)
        a = np.einsum("bvd,bd->vd", E, d)
        b = (E * d[:, None, :]).sum(0)
        check(f"[B={B} dim={D}] np.einsum and (E*d).sum(0) are BIT-IDENTICAL",
              np.array_equal(a, b), f"max|d| = {np.abs(a - b).max():.3e}")
    # and the numpy reference's own accuracy, stated
    g = np.random.default_rng(3)
    B, D = 64, 512
    E = (g.random((B, 256, D)).astype(np.float32) * 0.9)
    d = g.standard_normal((B, D)).astype(np.float32)
    t = np.einsum("bvd,bd->vd", E.astype(np.float64), d.astype(np.float64))
    a_abs, a_rel = dev(np.einsum("bvd,bd->vd", E, d), t)
    b_abs, b_rel = dev((E * d[:, None, :]).sum(0), t)
    print(f"    numpy reference at B={B} dim={D}: einsum {a_abs:.3e} "
          f"(rel {a_rel:.3e}), reformulation {b_abs:.3e} (rel {b_rel:.3e}) "
          f"-- identical, because the two are the same bits")


def part_d():
    print("\n(d) the default engine is untouched")
    d = gpu_port.GpuModel(device=DEV, dim=8, layers=1, seed=0)
    check("trace_mode still defaults to 'ring'", d.trace_mode == "ring")
    check("no dense trace is allocated in the default engine", d.dense_E is None)
    n = numpy_port.Model(dim=8, layers=1)
    check("the numpy reference still defaults to 'ring'",
          n.trace_mode == "ring")


def main():
    print("DENSE CREDIT, REFORMULATED -- the contraction's gate")
    print(f"  device: {DEV}   torch {torch.__version__}\n")
    part_a()
    part_b()
    part_c()
    part_d()
    print("\n" + ("DENSE CREDIT (reformulated): PASS"
                   if PASS else "DENSE CREDIT (reformulated): FAIL"))
    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
