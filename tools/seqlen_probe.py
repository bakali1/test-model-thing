#!/usr/bin/env python3
"""seqlen_probe.py — the (B,T) throughput measurement PLAN-2026-10-11 P1 exists
to make.

    .venv/bin/python tools/seqlen_probe.py                     # the default grid
    .venv/bin/python tools/seqlen_probe.py --cells 1024:1,512:2,256:4,64:16,16:64
    .venv/bin/python tools/seqlen_probe.py --device cuda --dim 512 --layers 16

WHY THIS SCRIPT AND NOT A FLAG ON bench_variants.py
---------------------------------------------------
`bench_variants.py` is a *variant board*: it loads one immutable checkpoint and
compares ring-credit / grad-assembly candidates against each other at a fixed
B, so every arm must share a checkpoint and a B. The T axis is the opposite
experiment -- B and T trade against each other at a fixed B*T -- and it needs a
fresh model per cell, no checkpoint, and a matched-work normalisation the board
has no concept of. Keeping them separate also keeps the board's contract (same
checkpoint, same bytes, state-error reporting) untouched.

WHAT IT MEASURES
----------------
bytes/s at MATCHED B*T. That normalisation is the whole point: the ring costs
`B * ring_k * D * 4` bytes of memory and `B * ring_k * D` elements of work per
layer, so a cell with 16x more bytes per stream has 16x less ring work per byte.
Reporting raw steps/s would hide exactly the effect being hunted.

`--warmup` windows are run and DISCARDED before the clock starts, because on
CPU the first calls pay allocator growth and (on CUDA) Inductor compilation:
`train_gpu.py --warmup` exists for the same reason and measures 1,081 B/s
against a true 27.5 kB/s without it.

THE P2 FALSIFIER (PLAN-2026-10-11 §5)
-------------------------------------
    "if throughput per byte at B=64, T=16 is more than 2x worse than today's
     B=1024, T=1 on the same device, the ring stays and P2 is abandoned."

The last line of the table states that verdict explicitly against the measured
numbers, so the decision is made by the printout rather than by whoever reads
it. Do not tune T to rescue the number.

THE MEMORY CLAIM
----------------
The engine runs with requires_grad=False everywhere, so a T-loop does NOT build
a T-deep autograd graph and each byte's forward cache is released when its
iteration ends. Activation memory is therefore O(B*D) per layer, not O(B*T*D)
-- PLAN-2026-10-11 §2 P1 predicts the latter and is wrong for this engine.
Peak RSS per cell is printed so that claim is falsifiable from this same table:
at matched B*T it must fall as B falls, not stay flat.
"""
import argparse
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import gpu_port
from gpu_port import GpuModel

DEFAULT_CELLS = "1024:1,256:4,64:16,16:64"


def peak_rss_gb():
    """Peak resident set size in GB. Linux reports ru_maxrss in KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def make_windows(B, T, n, seed, device=None):
    """n contiguous (B,T) byte windows, built the way train_gpu will build them.

    The trainer holds a per-file `(L, B)` byte matrix and steps a window out of
    it with `mats[t0:t0+T].t()`. Building the inputs that way here means the
    probe pays the same transpose the real run pays, instead of measuring a
    hypothetically contiguous layout it will never see.

    `device` moves the windows to the engine's device. This is NOT optional:
    the engine keeps theta, state, ring and tail on that device, and
    `step_batch` mixes them with the caller's inputs, so a CPU `ends` against
    a CUDA `stop` dies with "Expected all tensors to be on the same device".
    The trainer has always moved its windows to the device; the probe did not,
    which is why the T-axis measurement was never taken on a GPU (this file
    was written and only ever run on a CPU box).
    """
    rng = np.random.default_rng(seed)
    L = n * T + 1
    flat = rng.integers(1, 250, size=(L, B)).astype(np.int64)      # (L, B)
    cs = np.empty((n, B, T), np.int64)
    ns = np.empty((n, B, T), np.int64)
    es = np.empty((n, B, T), np.float32)
    for w in range(n):
        t0 = w * T
        cs[w] = np.ascontiguousarray(flat[t0:t0 + T].T)
        ns[w] = np.ascontiguousarray(flat[t0 + 1:t0 + 1 + T].T)
        es[w] = (rng.random((B, T)) < 0.02)
    return (torch.from_numpy(cs).to(device), torch.from_numpy(ns).to(device),
            torch.from_numpy(es).to(device))


def run_cell(B, T, dim, layers, device, ring_credit, grad_assembly, steps,
             warmup, seed, tail=True):
    """One (B,T) cell. Returns a dict of measurements.

    A FRESH model per cell, always: B is not a cosmetic knob here (state,
    ring and tail are all B-shaped), so carrying a model across cells would
    measure a resize instead of a run.
    """
    prev = os.environ.get("TMT_TAIL")
    os.environ["TMT_TAIL"] = "1" if tail else "0"
    try:
        m = GpuModel(dim=dim, layers=layers, device=str(device),
                     ring_credit=ring_credit, grad_assembly=grad_assembly,
                     seed=seed)
    finally:
        if prev is None:
            os.environ.pop("TMT_TAIL", None)
        else:
            os.environ["TMT_TAIL"] = prev
    m.new_states(B)
    n_win = warmup + steps
    cs, ns, es = make_windows(B, T, n_win, seed=seed, device=device)

    # ---- warmup: allocator growth, and Inductor compile on CUDA ------------
    # These are deliberately outside the clock. A short calibration run that
    # does not discard them reports compile time as if it were steady state.
    for i in range(warmup):
        m.step_batch(cs[i], ns[i], es[i])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = peak_rss_gb()

    # ---- timed region: step_batch ONLY -------------------------------------
    # The inputs are pre-built, so nothing but the engine is inside the clock.
    # sync_metrics stays on (the default) because a real training loop reads the
    # loss every step; --async-metrics is an orthogonal candidate, not a T-axis
    # effect, and mixing it in here would confound the comparison.
    losses = []
    t0 = time.perf_counter()
    for i in range(warmup, n_win):
        losses.append(float(m.step_batch(cs[i], ns[i], es[i])))
    if device.type == "cuda":
        torch.cuda.synchronize(device)          # else the timing is meaningless
    dt = (time.perf_counter() - t0) / steps

    hot = (sum(m.tail.nhot(i) for i in range(m.layercount))
           if m.tail is not None else 0)
    peak_cuda = (torch.cuda.max_memory_allocated(device) / 1e9
                 if device.type == "cuda" else 0.0)
    return {
        "B": B, "T": T, "BT": B * T, "ms": dt * 1e3, "Bps": B * T / dt,
        "ms_per_byte": dt * 1e3 / (B * T), "rss": peak_rss_gb(),
        "rss_delta": peak_rss_gb() - rss_before, "peak_cuda": peak_cuda,
        "ring_k": m.ring_k, "hot": hot, "t": m.t,
        "bytes_seen": m.bytes_seen, "loss": losses[-1],
        "params": int(m.theta.numel()),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=DEFAULT_CELLS,
                    help="comma-separated B:T pairs; they are run in the order "
                         "given and every cell gets a FRESH model")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20,
                    help="timed windows per cell (>=1)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="windows run and DISCARDED before the clock starts; "
                         "excludes allocator growth and Inductor compile")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--ring-credit",
                    choices=("auto", "full", "compiled", "triton", "nvrtc"),
                    default="auto")
    ap.add_argument("--grad-assembly", choices=("cat", "loop"), default="cat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tail", action="store_true",
                    help="TMT_TAIL=0: ring-only embed credit, no tail memory")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads; CPU throughput is otherwise "
                         "whatever the machine default is, which makes the "
                         "number unreproducible")
    args = ap.parse_args()

    if args.steps < 1:
        ap.error("--steps must be >= 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable")
    if args.threads:
        torch.set_num_threads(args.threads)

    cells = []
    for tok in args.cells.split(","):
        tok = tok.strip()
        if not tok:
            continue
        b, _, t = tok.partition(":")
        cells.append((int(b), int(t) if t else 1))

    rc = args.ring_credit
    if device.type != "cuda" and rc in ("compiled", "triton", "nvrtc"):
        rc = "full"          # the engine's own rule: those paths are CUDA-only
    print(f"device={device} dim={args.dim} layers={args.layers} "
          f"ring_credit={rc} grad_assembly={args.grad_assembly} "
          f"tail={'off' if args.no_tail else 'on'} "
          f"threads={torch.get_num_threads()} torch={torch.__version__}")
    print(f"{len(cells)} cells, {args.warmup} warmup + {args.steps} timed "
          f"windows each (warmup excluded from every number below)\n")

    rows = []
    for B, T in cells:
        r = run_cell(B, T, args.dim, args.layers, device, rc,
                     args.grad_assembly, args.steps, args.warmup, args.seed,
                     tail=not args.no_tail)
        rows.append(r)
        print(f"  B={r['B']:5d} T={r['T']:3d} (B*T={r['BT']:5d}): "
              f"{r['ms']:8.2f} ms/window = {r['Bps']:10.0f} B/s "
              f"({r['ms_per_byte']*1e3:6.2f} us/byte) | params {r['params']:,} "
              f"ring_k {r['ring_k']} hot {r['hot']:3d} | peak RSS "
              f"{r['rss']:.2f} GB (cell +{r['rss_delta']:.2f}) "
              f"{'peak CUDA %.2f GB' % r['peak_cuda'] if device.type == 'cuda' else ''}")

    print(f"\n[table] matched B*T — the T axis against the incumbent B=1024,T=1")
    print(f"  {'B':>6} {'T':>4} {'B*T':>6} {'ms/win':>9} {'B/s':>10} "
          f"{'us/byte':>9} {'vs (1024,1)':>12} {'peak RSS':>10}")
    base = next((r for r in rows if (r["B"], r["T"]) == (1024, 1)), rows[0])
    for r in rows:
        print(f"  {r['B']:>6} {r['T']:>4} {r['BT']:>6} {r['ms']:>9.2f} "
              f"{r['Bps']:>10.0f} {r['ms_per_byte']*1e3:>9.2f} "
              f"{r['Bps']/base['Bps']:>11.2f}x {r['rss']:>9.2f} GB")

    # ---- the P2 falsifier, stated against the measured numbers -------------
    p2 = next((r for r in rows if (r["B"], r["T"]) == (64, 16)), None)
    print()
    if p2 is None or base["Bps"] == 0:
        print("  [P2] no (64,16) cell in the grid; re-run with it to get the "
              "verdict")
    else:
        ratio = p2["Bps"] / base["Bps"]
        verdict = "P2 VIABLE" if ratio > 0.5 else "P2 BLOCKED — the ring stays"
        print(f"  [P2] (64,16) does {p2['Bps']:.0f} B/s vs (1024,1) "
              f"{base['Bps']:.0f} B/s = {ratio:.2f}x "
              f"({1/ratio:.2f}x worse per byte)")
        print(f"       falsifier is 2x worse -> {verdict}")

    # ---- the memory claim, from the same table -----------------------------
    if len({r["BT"] for r in rows}) == 1 and len(rows) > 1:
        print("\n[memory] all cells at matched B*T, so peak RSS must track B")
        print("          (O(B*D) per layer) and NOT B*T:")
        for r in rows:
            print(f"          B={r['B']:5d} T={r['T']:3d}: peak RSS "
                  f"{r['rss']:.2f} GB")
        print("          -> if RSS is flat across B, activation retention is "
              "scaling with T and the O(B*T*D) note in PLAN-2026-10-11 §2 P1 "
              "is right; if it falls with B, that note is wrong for this "
              "engine and the reason is the missing requires_grad.")
    print("\nNOTE: B and T are BOTH part of the run contract. State, ring and")
    print("tail are B-shaped and one AdamW update covers T bytes, so an A/B arm")
    print("must state --batch AND --seqlen together; with the gradient summed")
    print("over T, decoupled weight decay is relatively T times weaker.")


if __name__ == "__main__":
    main()
