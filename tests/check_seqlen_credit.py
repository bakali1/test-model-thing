#!/usr/bin/env python3
"""check_seqlen_credit.py — GATE 3 of PLAN-2026-10-11 P1: no credit is dropped.

    .venv/bin/python tests/check_seqlen_credit.py [--B 64] [--T 2,4,8,16,20]
                                                  [--dim 64] [--layers 4]

WHAT THIS IS THE DIRECT TEST OF
-------------------------------
`gpu_port.step_batch` addresses the ring with ONE slot per byte:

    p = (self.t - 1) % K          # gpu_port.py, the ring write pointer
    w[:, p] = 1.0                 # this byte = lag 1
    self.ring_b[:, p] = c_arr     # the byte id that lag-1 credit lands on

`p` is derived from the optimizer-step counter, which advances ONCE per window.
Derive it from `self.t` alone and all T bytes of a `(B,T)` window write the SAME
slot: the last byte wins, and T-1 bytes' worth of embedding credit disappears.
Nothing raises. The loss still falls. Every T=1 gate still passes, because at
T=1 there is only one byte and the bug is invisible.

The same trap exists for `tail.begin_step` / `tail.add_byte` (per-timestep
operations), for the `rows = self.ring_b.reshape(B*K)` snapshot (taken once,
before the layer loop, while the ring is rewritten T times), and for
`l.decaytrace` / `l.states` (state carried per byte). Each one is a plausible
wrong gradient, which is why this gate exists as a separate executable instead
of a paragraph in the plan.

THE CLAIM
---------
With the parameters FROZEN (lr=0, so both arms see bit-identical states and the
only difference is how the gradient is accumulated):

    sum over the T individual T=1 steps of (embedding-gradient mass)
  ==
    the single (B,T) window's embedding-gradient mass

within fp32 tolerance. `embed` is the load-bearing parameter here: it is the one
whose gradient arrives through the ring/tail per-lag credit, so a dropped slot
shows up as missing mass on specific byte ids rather than as a uniform scale
error. `.decay` is checked too, because it is the other cross-time parameter.

The structural assertions come FIRST and are exact rather than tolerance-based:
after the window, `ring_b` must hold the T window bytes in T DISTINCT slots, and
`ring_w` / `states` / `decaytrace` / `tail.E` must equal what the T=1 sequence
leaves behind. Those catch a dropped write even when the mass happens to
reconcile.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gpu_port
from gpu_port import GpuModel


def build(B, dim, layers, seed, n_hot, ce_only=False, device="cpu"):
    """Seeded engine, hot dims forced so the TAIL path is exercised as well as
    the ring. Hot dims are forced BEFORE new_states because new_states drops the
    tail and the next _ensure_tail rebuilds it from the current decays."""
    m = GpuModel(dim=dim, layers=layers, lr=5e-4, seed=seed, ce_only=ce_only,
                 device=device)
    with torch.no_grad():
        for l in m.layers:
            for j in range(n_hot):
                l.decay[j] = 4.0          # pre-sigmoid +4 => sigma ~0.982
    m.new_states(B)
    m.lr = 0.0                            # freeze: isolate the GRADIENT only
    return m


def streams(T, B, seed):
    rng = np.random.default_rng(seed)
    flat = rng.integers(1, 250, size=(T + 2) * B).astype(np.int64)
    cs = flat.reshape(T + 2, B)
    return (torch.from_numpy(cs), torch.from_numpy(np.roll(cs, -1, axis=0)),
            torch.from_numpy((rng.random((T + 2, B)) < 0.02).astype(np.float32)))


def live_state(m):
    out = {"ring_b": m.ring_b}
    for i, l in enumerate(m.layers):
        out[f"states.{i}"] = l.states
        out[f"decaytrace.{i}"] = l.decaytrace
        out[f"ring_w.{i}"] = l.ring_w
        if m.tail is not None and i in m.tail.E:
            out[f"tail_E.{i}"] = m.tail.E[i]
    return out


def compare(T, B, dim, layers, seed, n_hot, ce_only=False, verbose=True,
            device="cpu"):
    """Returns (ok, summary dict)."""
    dev = torch.device(device)
    cs, ns, es = streams(T, B, seed)
    # The engine keeps theta, state, ring and tail on its device, and
    # step_batch mixes them with the caller's inputs, so the windows have to
    # live on the SAME device. On CPU this was invisible; on CUDA a CPU
    # `ends` against a CUDA `stop` dies with "Expected all tensors to be on
    # the same device" -- which is how this gate was found to have never run
    # on a GPU at all.
    #
    # The CPU copies are kept: the (B,T) window is built from them below (a
    # device tensor cannot be handed to numpy), and the per-byte steps use
    # the device copies.
    cs_d, ns_d, es_d = cs.to(dev), ns.to(dev), es.to(dev)
    a = build(B, dim, layers, seed, n_hot, ce_only, device=device)  # T separate T=1 steps
    b = build(B, dim, layers, seed, n_hot, ce_only, device=device)  # one (B,T) window
    K = a.ring_k

    # ---- arm A: T individual (B,) steps ------------------------------------
    # Neither dembed nor the decay row is persisted between steps (`grads` is
    # overwritten), so both are read out of gviews after every call and summed
    # here -- that sum is exactly "the T individual T=1 steps' mass".
    a = build(B, dim, layers, seed, n_hot, ce_only, device=device)
    gA_embed = None
    decA = [None] * layers
    for t in range(T):
        a.step_batch(cs_d[t], ns_d[t], es_d[t])
        g = a.gviews["embed"].detach()
        gA_embed = g.clone() if gA_embed is None else gA_embed + g
        for i in range(layers):
            d = a.gviews[f"layer{i}.decay"].detach()
            decA[i] = d.clone() if decA[i] is None else decA[i] + d

    # ---- arm B: one (B,T) window -------------------------------------------
    cw = torch.from_numpy(np.ascontiguousarray(cs[:T].T)).to(dev)
    nw = torch.from_numpy(np.ascontiguousarray(ns[:T].T)).to(dev)
    ew = torch.from_numpy(np.ascontiguousarray(es[:T].T)).to(dev)
    b.step_batch(cw, nw, ew)
    gB_embed = b.gviews["embed"].detach().clone()
    decB = [b.gviews[f"layer{i}.decay"].detach().clone() for i in range(layers)]

    ok = True
    notes = []

    # ---- the run contract: t is OPTIMIZER steps, bytes_seen is BYTES ----
    # `self.t` drives AdamW's bias correction and the LR schedule, so it must
    # advance by exactly 1 per window; `bytes_seen` counts bytes, so it must
    # advance by B*T. Getting either wrong is invisible at T=1 (where the two
    # coincide) and changes the schedule silently at T>1, which is why it is
    # asserted here rather than left to the mass check.
    if (b.t, b.bytes_seen) != (1, B * T):
        ok = False
        notes.append(f"counters after one (B,T) window: t={b.t} (want 1), "
                     f"bytes_seen={b.bytes_seen} (want {B * T}) -- t is the "
                     f"optimizer-step counter, bytes_seen counts bytes")
    if a.t != T:
        ok = False
        notes.append(f"arm A ran {T} steps but t={a.t}")

    # ---- structural: the ring pointer advanced T times ---------------------
    # The pointer must visit min(T, K) DISTINCT slots -- one per byte -- and the
    # most recent min(T, K) of them must hold the window bytes in write order.
    # With the single-slot bug (p = (self.t-1) % K, no +tt) every byte of the
    # window writes the SAME slot: only one slot ever changes, the rest keep
    # whatever the previous window left, and `slots == want` fails on the second
    # entry. That is the assertion that names the trap, so it is made exact
    # rather than tolerance-based.
    # `b.t` is the WINDOW's step counter (1), so the last byte's slot is
    # (b.t - 1 + T - 1) % K; arm A's counter is irrelevant here.
    t_last = (b.t - 1 + T - 1) % K
    n_slot = min(T, K)
    visited = len({(t_last - k) % K for k in range(T)})
    slots = [int(b.ring_b[0, (t_last - k) % K]) for k in range(n_slot)]
    want = [int(cs[T - 1 - k, 0]) for k in range(n_slot)]
    if not (visited == n_slot and slots == want):
        ok = False
        notes.append(f"ring pointer visited {visited}/{n_slot} slots and the "
                     f"last {n_slot} hold {slots}, expected {want} -- the write "
                     f"pointer did not advance once per byte")

    # ---- structural: everything else equals the T=1 sequence ---------------
    sa, sb = live_state(a), live_state(b)
    for k in sa:
        if k not in sb:
            continue
        if not torch.equal(sa[k], sb[k]):
            # ring_b is int64, so scale by its magnitude rather than .abs().mean()
            ref = sa[k].to(torch.float64)
            d = (ref - sb[k].to(torch.float64)).abs().max().item()
            scale = float(ref.abs().mean()) + 1e-30
            ok = False
            notes.append(f"{k}: max|d| {d:.3e} vs mean|ref| {scale:.3e}")

    # ---- the mass claim -----------------------------------------------------
    massA = float(gA_embed.abs().sum())
    massB = float(gB_embed.abs().sum())
    rowA = gA_embed.abs().sum(dim=1)
    rowB = gB_embed.abs().sum(dim=1)
    row_relden = float((rowA - rowB).abs().max() / (rowA.max() + 1e-30))
    rms = float(gA_embed.pow(2).mean().sqrt()) + 1e-30
    elem = float((gA_embed - gB_embed).abs().max() / rms)
    if abs(massA - massB) / (massA + 1e-30) >= 1e-5:
        ok = False
        notes.append(f"embed L1 mass {massA:.6e} != {massB:.6e}")
    if row_relden >= 1e-5:
        ok = False
        notes.append(f"per-byte-id mass max rel dev {row_relden:.3e}")
    if elem >= 1e-4:
        ok = False
        notes.append(f"elementwise max rel dev {elem:.3e}")

    # ---- the other cross-time parameter ------------------------------------
    dw = 0.0
    for i in range(layers):
        d = decA[i] - decB[i]
        dw = max(dw, float(d.abs().max() / (decA[i].abs().max() + 1e-30)))
    if dw >= 1e-4:
        ok = False
        notes.append(f"decay gradient max rel dev {dw:.3e}")

    if verbose:
        tag = "ce_only" if ce_only else "default"
        print(f"  T={T:3d} K={K:3d} B={B:4d} [{tag:7s}] "
              f"embed L1 mass {massA:.6e} vs {massB:.6e} "
              f"(rel {abs(massA-massB)/(massA+1e-30):.2e}) | "
              f"per-id max rel {row_relden:.2e} | elementwise {elem:.2e} | "
              f"decay {dw:.2e} | ring pointer {visited}/{n_slot} slots "
              f"-> {'ok' if ok else 'FAIL'}")
        for n in notes:
            print(f"        !! {n}")
    return ok, {"T": T, "K": K, "massA": massA, "massB": massB,
                "row_relden": row_relden, "elem": elem, "decay": dw,
                "visited": visited, "n_slot": n_slot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-hot", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--T", default="1,2,4,8,16,20",
                    help="comma-separated window lengths; T=1 is the control")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the engine and the windows. 'cpu' "
                         "is the reference; 'cuda' runs the SAME claim on the "
                         "GPU, which is where the arms will actually train, "
                         "and that is not redundant: this gate compares two "
                         "implementations of the same gradient, and the GPU's "
                         "atomics and reduction orders are exactly where such "
                         "a claim could differ.")
    args = ap.parse_args()
    Ts = [int(x) for x in args.T.split(",") if x.strip()]

    m0 = build(args.B, args.dim, args.layers, args.seed, args.n_hot,
               device=args.device)
    print(f"== GATE 3 (P1): no credit dropped at T>1 ==")
    print(f"  B={args.B} dim={args.dim} layers={args.layers} "
          f"ring_k={m0.ring_k} hot dims/layer={args.n_hot} "
          f"device={args.device} "
          f"(lr frozen at 0 so both arms see identical states)")
    ok = True
    worst_elem, worst_mass = 0.0, 0.0
    for T in Ts:
        for ce in (False, True):
            good, s = compare(T, args.B, args.dim, args.layers, args.seed,
                              args.n_hot, ce_only=ce, device=args.device)
            ok &= good
            worst_elem = max(worst_elem, s["elem"])
            worst_mass = max(worst_mass,
                             abs(s["massA"] - s["massB"]) / (s["massA"] + 1e-30))
    # ---- and the ring-only path (TMT_TAIL=0), because the tail's per-byte
    # begin_step / add_byte are exactly the operations a T loop can under-fire
    print("  -- ring-only (TMT_TAIL=0): the tail path is disabled --")
    prev = os.environ.get("TMT_TAIL")
    os.environ["TMT_TAIL"] = "0"
    try:
        for T in [t for t in Ts if t > 1]:
            good, s = compare(T, args.B, args.dim, args.layers, args.seed,
                              args.n_hot)
            ok &= good
            worst_elem = max(worst_elem, s["elem"])
            worst_mass = max(worst_mass,
                             abs(s["massA"] - s["massB"]) / (s["massA"] + 1e-30))
    finally:
        if prev is None:
            os.environ.pop("TMT_TAIL", None)
        else:
            os.environ["TMT_TAIL"] = prev

    print(f"\n  worst embed L1-mass relative deviation: {worst_mass:.3e}")
    print(f"  worst elementwise relative deviation:   {worst_elem:.3e}")
    print(f"  [{'PASS' if ok else 'FAIL'}] the sum of the T individual T=1 "
          f"steps' embedding-gradient mass equals the single (B,T) window's "
          f"mass, and ring_w / states / decaytrace / tail.E are identical "
          f"afterwards")
    print("  -> this is the direct test of the single-slot-write trap: with "
          "p = (self.t-1) % K and no +tt, every byte of the window writes the "
          "same slot, the last byte wins, and the per-byte-id mass collapses "
          "onto T-1 fewer byte ids. Nothing would raise.")
    print("\n== verdict:", "PASS" if ok else "FAIL", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
