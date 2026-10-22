#!/usr/bin/env python3
"""measure_m1_gate.py -- M1: DOES THE GATE ENGAGE AT ALL?

M1 is the P3 design's cheapest and most decisive question: if the per-sample
spread of the gated retention

    a_{b,t,d} = sigma(decay_d + gate_d * x_{b,t,d})

is ~0, the gate is INERT -- the arm is a no-op with a measured cause, and
that is a more important result than any speedup.

THE METRIC
    For every layer and every byte position, `a` is a (B, dim) matrix: one
    retention per (stream, dim).  The question is whether the gate makes
    DIFFERENT streams behave differently.  So:

        per-sample mean   mu_b  = mean_d a[b,d]        (what stream b gets)
        per-sample std    sd_b  = std_d  a[b,d]        (how varied b's own
                                                       retentions are)
        BETWEEN-sample    = std_b(mu_b)                (do streams differ?)
        WITHIN-sample     = mean_b(sd_b)               (the baseline spread)
        RATIO             = BETWEEN / WITHIN          -> 0 means INERT

    and the per-dim version, against the TEMPORAL spread (which is the
    spread the engine's own memory lives in):

        RATIO_dim = mean_d( std_b a[b,d] ) / mean_{b,d}( std_t a[b,t,d] )

WHAT IS MEASURED
    1. the trained CL-2 checkpoint (armA), read IN PLACE, on the CL-2
       board's own byte distribution -- the distribution the arm trained on.
       Its gate is at its INIT value (the CL-2 arm is ungated: the
       checkpoint has no layer{i}.gate key), so this is the baseline.
    2. the SAME measurement on the UNTRAINED P4-init model.
    3. the SENSITIVITY: the same ratios as a function of the gate's
       magnitude.  This is the number that says how big the gate has to
       become before the arm can do anything -- i.e. whether the gate's
       gradient has a chance of mattering.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gpu_port
import cl2_train as cl2

F = torch.float32


def build(args):
    m = gpu_port.GpuModel(dim=args.dim, layers=args.layers, lr=args.lr,
                          seed=args.seed, device=args.device,
                          half_life_init=(args.half_min, args.half_max),
                          ring_k=8, ring_credit="full", gated=True)
    return m


def load_params(m, path):
    """Copy the checkpoint's parameters into the model, BY NAME.

    A ring (ungated) checkpoint has no layer{i}.gate key, so the gate stays
    at its init (exactly 0) -- which is the point of the measurement: it is
    the state every checkpoint in existence is in.
    """
    d = np.load(path, allow_pickle=False)
    n = 0
    with torch.no_grad():
        for name, view in m.views.items():
            if name in d.files and d[name].dtype == np.float32:
                view.copy_(torch.from_numpy(d[name]).to(view.device))
                n += 1
    meta = {k[4:]: d[k] for k in d.files if k.startswith("cl2_")}
    return n, meta


def measure(m, seq, gate_scale, seed=0):
    """One forward over the sequence, collecting `a` per layer."""
    g = torch.Generator().manual_seed(seed)
    for l in m.layers:
        if gate_scale == 0.0:
            l.gate = torch.zeros(m.dim, dtype=F, device=m.device)
        else:
            v = torch.randn(m.dim, generator=g, dtype=F)
            l.gate = (v / v.pow(2).mean().sqrt() * gate_scale).to(m.device)
        l.gated = True
    B, L = seq.shape
    seq_f = torch.from_numpy(seq.astype(np.int64)).to(m.device)
    layers = m.layers
    states = [torch.zeros((B, m.dim), dtype=F, device=m.device)
              for _ in layers]
    A = [[] for _ in layers]
    with torch.no_grad():
        for t in range(L):
            enc = m.embed[seq_f[:, t]]
            x = enc
            for i, l in enumerate(layers):
                x, st, a, _c = l.forward(enc, x, states[i])
                states[i] = st
                A[i].append(a.detach().cpu().numpy().astype(np.float64))
    out = []
    for i in range(len(layers)):
        # (T, B, dim) in float64: numpy's float32 std has a 1-ulp artefact
        # (E[x^2]-E[x]^2 in float32), which would show up as a "spread" of
        # exactly-identical values.
        At = np.stack(A[i], axis=0).astype(np.float64)
        mu = At.mean(axis=2)                 # (T, B)  per-sample mean
        sd = At.std(axis=2)                  # (T, B)  per-sample std
        between = float(mu.mean(axis=0).std())      # across samples
        within = float(sd.mean(axis=0).mean())      # within a sample
        ratio_bw = between / within if within > 0 else 0.0
        # the per-dim version: across-sample vs across-TIME
        sd_b = At.std(axis=1)                # (T, B, dim) across samples
        sd_t = At.std(axis=0)                # (T, B, dim) across time
        ratio_d = (float(sd_b.mean())
                   / float(sd_t.mean()) if sd_t.mean() > 0 else 0.0)
        out.append({
            "layer": i,
            "a_mean": float(At.mean()),
            "between_samples": between,
            "within_sample": within,
            "ratio_between_within": ratio_bw,
            "ratio_across_samples_to_across_time": ratio_d,
            "gate_abs_max": float(np.abs(
                m.layers[i].gate.detach().cpu().numpy()).max()),
            "gate_rms": float(np.sqrt(np.mean(
                m.layers[i].gate.detach().cpu().numpy() ** 2))),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--half-min", type=float, default=2.0)
    ap.add_argument("--half-max", type=float, default=512.0)
    ap.add_argument("--facts", type=int, default=4)
    ap.add_argument("--alphabet", default="16")
    ap.add_argument("--gap", type=int, default=32)
    ap.add_argument("--board-seed-a", type=int, default=101)
    ap.add_argument("--board-size", type=int, default=512)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--gates", default="0,0.01,0.03,0.1,0.3,1.0")
    ap.add_argument("--text-frac", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"M1 -- does the gate engage?  device={args.device} "
          f"B={args.B} dim={args.dim} layers={args.layers}")
    print(f"  checkpoint: {args.ckpt or 'NONE (the UNTRAINED P4 init)'}")

    m = build(args)
    if args.ckpt:
        n, meta = load_params(m, args.ckpt)
        print(f"  loaded {n} parameter tensors; the checkpoint's meta: "
              f"{ {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in meta.items() if k in ('phase','step','facts','alphabet','board_seed_a','gap')} }")
        if "layer0.gate" in np.load(args.ckpt, allow_pickle=False).files:
            print("  NOTE: this checkpoint CARRIES a gate -- it is a GATED "
                  "arm's checkpoint")
        else:
            print("  NOTE: this checkpoint has NO layer{i}.gate key -> the "
                  "gate is at its INIT (exactly 0)")

    # the CL-2 board's own byte distribution (the training distribution)
    alpha = cl2.alphabet_symbols(args.alphabet)
    board = cl2.make_board(args.board_seed_a, args.board_size, alpha)
    import glob
    files = sorted(glob.glob(os.path.join(ROOT, "data", "*.txt")))
    pool = cl2.TextPool(files, seed=args.seed) if files else None
    rng = np.random.default_rng(args.seed)
    batch = cl2.build_fact_batch(args.B, board, alpha, pool, args.text_frac,
                                  args.dim, rng, n_facts=args.facts)
    seq = batch["seq"]
    print(f"  the CL-2 board's own bytes: seq {seq.shape}, "
          f"alphabet {args.alphabet}, {args.facts} facts/stream, "
          f"gap {args.gap}")

    rows = {}
    for scale in [float(x) for x in args.gates.split(",")]:
        res = measure(m, seq, scale, seed=args.seed)
        rows[scale] = res
        last = res[-1]
        print(f"\n  gate RMS = {scale:g}")
        print(f"    {'layer':>5} | {'a mean':>9} {'between':>11} "
              f"{'within':>11} {'B/W ratio':>10} | {'across-samp/':>13} "
              f"{'across-time':>12}")
        for r in res:
            print(f"    {r['layer']:>5} | {r['a_mean']:9.5f} "
                  f"{r['between_samples']:11.3e} {r['within_sample']:11.3e} "
                  f"{r['ratio_between_within']:10.3e} | "
                  f"{r['ratio_across_samples_to_across_time']:13.3e}")
        print(f"    -> deepest layer: BETWEEN/WITHIN = "
              f"{last['ratio_between_within']:.3e}  (0 = INERT), "
              f"across-samples/across-time = "
              f"{last['ratio_across_samples_to_across_time']:.3e}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=1)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
