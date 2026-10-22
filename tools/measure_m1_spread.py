#!/usr/bin/env python3
"""measure_m1_spread.py -- M1: the per-sample spread of `a`, the P3 gate's
whole reason to exist.

M1 (docs/design/P3-GATE-ANALYSIS.md section 2) asks: on a TRAINED model,
what is the per-dim and the per-sample spread of

    a_{b,t,d} = sigma(decay_d + gate_d * x_{b,t,d}) ?

Because the gate is a per-DIM vector and x is per-SAMPLE, a non-zero gate
makes the RETENTION per-sample.  If the spread of `a` across the batch is
~0, the gated arm is a no-op: the per-sample retention the design wants
cannot happen, whatever the gate does to `a`.  M1 is therefore the cheapest
and the most decisive number in the whole P3 programme -- and it is the
number that decides whether a 12x speedup of the dense credit is worth
anything at all.

What this script measures, on REAL corpus bytes and a TRAINED checkpoint:

  1. the per-sample spread of `a` (max over dims of a.max(0)-a.min(0), and
     the same relative to a's own mean), for a ladder of gate scales;
  2. the per-sample spread of the STATE s_t -- the precondition: a
     per-sample retention can only matter if the states differ across the
     batch in the first place;
  3. the per-sample spread of x_in, the quantity the gate multiplies;
  4. at gate = 0 exactly, as the arm is initialised: the spread is
     identically 0, which is why the arm starts bit-identical to the
     incumbent and whether it EVER leaves that is the open question.

It needs no GPU and no training: it is a forward-pass measurement.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gpu_port
import numpy_port

F = torch.float32


def spread(a):
    """(max over dims of the across-batch range, the same relative)."""
    a = a.detach()
    rng = (a.max(dim=0).values - a.min(dim=0).values)
    mean = a.mean(dim=0).abs().clamp_min(1e-30)
    return float(rng.max()), float(rng.mean()), float((rng / mean).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/mx2048/tmt.npz")
    ap.add_argument("--corpus", default="data/moby_dick.txt")
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--gates", default="0,0.1,0.3,1.0,3.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-grad", action="store_true",
                    help="also measure how fast the GATE ITSELF moves on this "
                         "trained checkpoint: the gated arm's dgate per step, "
                         "and the number of steps a consistent gradient would "
                         "need to reach the gate magnitude that makes M1 "
                         "non-trivial")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gate-T", type=int, default=4,
                    help="the window length for the gate-gradient measurement "
                         "(>= 2: the first byte has no previous state)")
    args = ap.parse_args()

    z = np.load(args.ckpt, allow_pickle=False)
    keys = list(z.keys())
    dim = int(z["embed"].shape[1])
    layers = 1 + max(int(k.split(".")[0][5:]) for k in keys
                     if k.startswith("layer") and k.count(".") == 1)
    print(f"checkpoint: {args.ckpt}   dim={dim} layers={layers}")
    print(f"corpus: {args.corpus}   B={args.B} T={args.T}")

    m = gpu_port.GpuModel(device="cpu", dim=dim, layers=layers, seed=args.seed)
    m.load(args.ckpt)
    m.new_states(args.B)

    data = open(args.corpus, "rb").read()
    streams, _ends = numpy_port.Runtime.make_streams(data, args.B)
    byte_ids = [np.frombuffer(s.encode() if isinstance(s, str) else s,
                              dtype=np.uint8).astype(np.int64)
                for s in streams]

    g = torch.Generator().manual_seed(args.seed)
    for scale in [float(x) for x in args.gates.split(",")]:
        for l in m.layers:
            if scale == 0.0:
                l.gate = torch.zeros(dim, dtype=F)
            else:
                v = torch.randn(dim, generator=g, dtype=F)
                l.gate = (v / v.pow(2).mean().sqrt() * scale).contiguous()
            l.gated = True
        m.new_states(args.B)
        a_spread = np.zeros(layers)
        a_spread_rel = np.zeros(layers)
        a_spread_mean = np.zeros(layers)
        s_spread = np.zeros(layers)
        x_rms = np.zeros(layers)
        s_rel = np.zeros(layers)
        with torch.no_grad():
            for t in range(args.T):
                c = torch.from_numpy(
                    np.stack([b[t] for b in byte_ids])).long()
                enc = m.embed[c]
                x = enc
                for i, l in enumerate(m.layers):
                    x_in = x
                    x, state, a, _cache = l.forward(enc, x, l.states)
                    l.states = state
                    mx, mn, rel = spread(a)
                    a_spread[i] = max(a_spread[i], mx)
                    a_spread_mean[i] = max(a_spread_mean[i], mn)
                    a_spread_rel[i] = max(a_spread_rel[i], rel)
                    sx, smn, srel = spread(state)
                    s_spread[i] = max(s_spread[i], sx)
                    s_rel[i] = max(s_rel[i], srel)
                    x_rms[i] = max(x_rms[i], float(x_in.pow(2).mean().sqrt()))
        print(f"\n  gate RMS = {scale:g}  (a = sigma(decay + gate*x_in))")
        print(f"    {'layer':>5} | {'spread(a) max':>14} {'mean':>10} "
              f"{'relative':>10} | {'spread(state)':>14} {'rel':>10} | "
              f"{'RMS(x_in)':>10}")
        for i in range(layers):
            print(f"    {i:>5} | {a_spread[i]:14.3e} {a_spread_mean[i]:10.3e} "
                  f"{a_spread_rel[i]:10.3e} | {s_spread[i]:14.3e} "
                  f"{s_rel[i]:10.3e} | {x_rms[i]:10.3e}")
        print(f"    -> the per-sample spread of `a` at the deepest layer: "
              f"{a_spread[-1]:.3e} absolute, {a_spread_rel[-1]:.3e} relative")
        print(f"    -> the per-sample spread of the STATE (the precondition): "
              f"{s_spread[-1]:.3e} absolute, {s_rel[-1]:.3e} relative")

    if not args.gate_grad:
        return

    # ---- how fast does the gate itself move? ---------------------------
    print("\n  the GATE's own rate of movement on this trained checkpoint")
    # The gated arm starts from the incumbent's parameters with gate = 0
    # exactly, so the trained checkpoint's parameters ARE the gated arm's
    # starting point. (The loader refuses a ring checkpoint into a dense
    # engine on purpose -- a dense trace must not start cold -- so the
    # parameters are copied by name instead, and the gate is left at 0.)
    mg = gpu_port.GpuModel(device="cpu", dim=dim, layers=layers, seed=args.seed,
                           gated=True)
    with torch.no_grad():
        for name, arr in z.items():
            if name in mg.views and arr.dtype == np.float32:
                mg.views[name].copy_(torch.from_numpy(arr))
    mg.new_states(args.B)
    # the gate's gradient needs a CROSS-TIME chain: at the first byte the
    # previous state is 0, so dgate is exactly 0 there. A window of T >= 2
    # bytes is what makes it non-zero.
    c = torch.from_numpy(np.stack(
        [np.stack([b[t] for t in range(args.gate_T)]) for b in byte_ids]
    )).long()
    n = torch.from_numpy(np.stack(
        [np.stack([b[t + 1] for t in range(args.gate_T)]) for b in byte_ids]
    )).long()
    ends = torch.zeros((args.B, args.gate_T), dtype=F)
    mg.step_batch(c, n, ends)
    off = 0
    grads = {}
    gflat = mg.grads.detach().numpy().astype(np.float64)
    for name, shape in mg.shapes:
        size = int(np.prod(shape))
        grads[name] = gflat[off:off + size].reshape(shape)
        off += size
    print(f"    {'layer':>5} | {'RMS(dgate)':>12} {'max|dgate|':>12} | "
          f"{'RMS(dgate)*lr':>14} | {'steps to RMS 0.1':>17}")
    for i in range(layers):
        d = grads[f"layer{i}.gate"]
        rms = float(np.sqrt((d ** 2).mean()))
        mx = float(np.abs(d).max())
        per_step = rms * args.lr
        steps = 0.1 / per_step if per_step > 0 else float("inf")
        print(f"    {i:>5} | {rms:12.3e} {mx:12.3e} | {per_step:14.3e} | "
              f"{steps:17.0f}")
    print("    (a per-dim gradient with mixed signs does a RANDOM WALK: the "
          "gate's RMS then grows like sqrt(steps) x the per-step size, so "
          "the numbers above are an UPPER bound on the rate.)")


if __name__ == "__main__":
    main()
