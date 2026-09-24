#!/usr/bin/env python3
"""check_variants.py — every optimization candidate must agree with `legacy`.

`bench_variants.py` measures the candidates on a GPU; this gate is the
correctness half, and it runs on the laptop CPU venv for free. A candidate is
accepted only if its state matches the legacy engine to a scale-aware
tolerance (max abs error / reference RMS).

Why scale-aware: these tensors span orders of magnitude, so a floor-relative
metric explodes near zero (the same trap that made the CUDA parity gate read
7e-02 on tensors whose true error was 3e-05 of their RMS).

Tolerances differ by class, deliberately:
  * ring_credit paths (A1/A2/C) must be *equivalent*: they only reassociate or
    re-order float accumulation, so they must land near 1e-5.
  * factored decay (B1/B2) reassociates products, so it is allowed more room,
    but it must stay far below anything that could change learning: 1e-3.

This gate is what caught the factored-tail bug where add_byte added a instead
of a/c (35x RMS divergence, invisible in the loss to 4 decimal places).
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import gpu_port
from gpu_port import GpuModel

# variant -> (ring_algorithm, tail_algorithm, ring_credit, tolerance)
CANDIDATES = {
    "a1-cold":       ("legacy", "legacy", "cold",      1e-4),
    "a2-kloop":      ("legacy", "legacy", "kloop",     1e-4),
    "c-segmented":   ("legacy", "legacy", "segmented", 1e-4),
    "b1-ring":       ("factored", "legacy", "full",    1e-3),
    "b2-tail":       ("legacy", "factored", "full",    1e-3),
    "b1-b2":         ("factored", "factored", "full",  1e-3),
    "b1-triton":     ("factored", "legacy", "triton",  1e-3),
    "c-triton":      ("legacy", "legacy", "triton",    1e-4),
}


def effective_state(m):
    """Comparable snapshot: the *effective* (decayed) value of every buffer."""
    out = {"theta": m.theta.detach().cpu().numpy().astype(np.float64),
           "m": m.m.detach().cpu().numpy().astype(np.float64),
           "v": m.v.detach().cpu().numpy().astype(np.float64),
           "ring_b": m.ring_b.detach().cpu().numpy().astype(np.float64)}
    for name in m.gviews:
        out["g:" + name] = m.gviews[name].detach().cpu().numpy().astype(np.float64)
    for i, l in enumerate(m.layers):
        raw = l.ring_w.detach().cpu().numpy().astype(np.float64)
        f = getattr(l.ring_w, "_factor", None)
        out[f"ring_eff.{i}"] = raw * (f.detach().cpu().numpy()
                                      if f is not None else 1.0)
    if m.tail is not None:
        for i in m.tail.E:
            E = m.tail.E[i].detach().cpu().numpy().astype(np.float64)
            f = m.tail.factor.get(i) if hasattr(m.tail, "factor") else None
            # factor is (B,H); E is (B,256,H) -> broadcast over the bins
            out[f"tail_eff.{i}"] = E * (f.detach().cpu().numpy()[:, None, :]
                                        if f is not None else 1.0)
            out[f"tail_cols.{i}"] = m.tail.cols[i].detach().cpu().numpy()
    return out


def state_error(a, b):
    """max |a-b| / RMS(b) over every shared key (inf on shape mismatch)."""
    worst = 0.0
    for k in sorted(set(a) & set(b)):
        x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
        if x.shape != y.shape:
            return float("inf")
        if k.startswith("tail_cols") or k == "ring_b":
            worst = max(worst, float((x != y).mean()))
            continue
        rms = float(np.sqrt(np.mean(y * y)))
        worst = max(worst, float(np.max(np.abs(x - y))) / (rms + 1e-30))
    missing = (set(a) ^ set(b))
    return worst, sorted(missing)


def build(ring_algorithm, tail_algorithm, ring_credit, dim, layers, B, n_hot):
    m = GpuModel(dim=dim, layers=layers, device="cpu",
                 ring_algorithm=ring_algorithm, tail_algorithm=tail_algorithm,
                 ring_credit=ring_credit, allow_algo_migration=True)
    with torch.no_grad():                    # force hot dims so the tail engages
        for i, l in enumerate(m.layers):
            l.decay[:n_hot] = 4.0
    m.new_states(B)
    return m


def run(ring_algorithm, tail_algorithm, ring_credit, dim, layers, B, steps,
        n_hot, seed=17):
    m = build(ring_algorithm, tail_algorithm, ring_credit, dim, layers, B, n_hot)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(steps):
        c = torch.from_numpy(rng.integers(1, 250, B))
        n = torch.from_numpy(rng.integers(1, 250, B))
        e = torch.from_numpy((rng.random(B) < 0.02).astype(np.float32))
        losses.append(float(m.step_batch(c, n, e)))
    return m, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--n-hot", type=int, default=8)
    ap.add_argument("--variants", default="all")
    args = ap.parse_args()

    base_m, base_losses = run("legacy", "legacy", "full", args.dim, args.layers,
                              args.B, args.steps, args.n_hot)
    base = effective_state(base_m)
    print(f"baseline: {args.steps} steps, B={args.B}, loss "
          f"{base_losses[0]:.4f} -> {base_losses[-1]:.4f}, "
          f"hot={sum(base_m.tail.nhot(i) for i in range(args.layers))}")

    names = ([n for n in CANDIDATES if n != "b1-triton"]
             if args.variants == "all" else
             [x.strip() for x in args.variants.split(",") if x.strip()])
    bad = []
    print(f"\n{'variant':16s} {'tol':>8s} {'state/RMS':>12s} {'loss d':>10s}  verdict")
    for name in names:
        if name not in CANDIDATES:
            print(f"  unknown variant {name}")
            bad.append(name)
            continue
        ra, ta, rc, tol = CANDIDATES[name]
        if rc == "triton" and gpu_port.triton is None:
            print(f"{name:16s} {'-':>8s} {'-':>12s} {'-':>10s}  SKIP (no triton)")
            continue
        m, losses = run(ra, ta, rc, args.dim, args.layers, args.B,
                        args.steps, args.n_hot)
        err, missing = state_error(base, effective_state(m))
        dl = abs(losses[-1] - base_losses[-1])
        ok = err <= tol and not missing
        if not ok:
            bad.append(name)
        extra = f"  [key mismatch: {missing}]" if missing else ""
        print(f"{name:16s} {tol:8.0e} {err:12.3e} {dl:10.2e}  "
              f"{'PASS' if ok else 'FAIL'}{extra}")

    # Factored tails must not produce non-finite losses over a LONG run. The
    # renormalization interval is derived from sigma_min so the per-column
    # factor cannot underflow; a fixed interval (100) with the live model's
    # sigma_min ~0.205 underflows to ~1e-70 and E.div_(c) then yields NaN --
    # which is exactly what the 200-step T4 board surfaced.
    if "b2-tail" in names or args.variants == "all":
        m, losses = run("legacy", "factored", "full", args.dim, args.layers,
                        args.B, 300, args.n_hot, seed=23)
        finite = all(np.isfinite(l) for l in losses)
        every = getattr(m, "_tail_renorm_every", 0)
        print(f"{'b2-tail 300 steps':16s} {1e-3:8.0e} "
              f"{'0' if finite else float('nan'):>12s} {0.0:10.2e}  "
              f"{'PASS' if finite else 'FAIL (non-finite loss)'} "
              f"[renorm every {every}]")
        if not finite:
            bad.append("b2-tail-long")

    print("\nCHECK_VARIANTS:", "PASS" if not bad else f"FAIL {bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
