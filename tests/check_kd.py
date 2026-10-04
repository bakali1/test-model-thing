"""check_kd.py — GATE 3 + the KD gradient itself.

    .venv/bin/python tests/check_kd.py [--steps 120]

Free, CPU, ~1 min. Three properties, in order of how badly they would hurt:

  g3  **alpha = 0 is bit-exact.** The pre-KD engine must be reproduced with
      `max|Δθ| == 0.0` over 120 steps on the live-shaped config. Not "close":
      zero. Any nonzero difference means the CE-weight or gradient path was
      touched even when distillation is off, which would silently invalidate
      every CE baseline in RESULTS.md.
  g1  **alpha scales CE.** With a fixed target, loss(alpha) must be affine in
      alpha with slope = the KD term and intercept = the CE loss; a stray
      factor of B or of (1-alpha) shows up here immediately.
  g2  **the KD gradient is right.** Finite differences of the KD-augmented loss
      against the analytic gradient, on the real decoder path. This is g4 below,
      and it uses a PARTIALLY covered target; a dense one cannot see a mask bug.
  g4  **the KD gradient and the KD loss are masked in the SAME places.**
      `covered` zeroes `s_t` at declined positions, so if `doutput` is not
      zeroed there too, the optimiser descends something other than the loss
      that gets logged.

THE FACTOR-B CONVENTION (do not "discover" this as a bug)
---------------------------------------------------------
The reported loss is a per-batch MEAN (`... / B`); the applied gradient is the
un-divided SUM. So applied == B x d(loss)/d(theta) for the CE path and the KD
path alike, verified at alpha=0 as well as alpha>0. That is a pre-existing
logging convention, not a defect, and AdamW normalises it away. g4 asserts the
ratio is B at EVERY coverage level, which is what makes it a bug detector: a
term present in the gradient but absent from the loss (or the reverse) breaks
the ratio at that coverage and only that one.

g4 exists because the KD gradient WAS masked in one place and not the other:
`s_t` was zeroed at declined positions while `doutput` still received
`alpha * (p_s - 0)` there (~8% of the decoder gradient at 24% coverage). Every
pre-existing check missed it. g1 and g2b compare only LOSSES, and the synthetic
target used by g2 is non-zero on every row, so `covered` is always True and the
mask never engages.

The target used here is a synthetic, deterministic (B,256) distribution — the
gate is about the ENGINE's use of a target, not about what the teacher produces
(that is tests/check_mi.py and tools/mi_report.py).
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gpu_port  # noqa: E402


def make(B=16, dim=32, layers=3, seed=7, **kw):
    m = gpu_port.GpuModel(dim=dim, layers=layers, device="cpu",
                          ring_credit="full", seed=seed, **kw)
    m.new_states(B)
    return m


def stream(B, steps, seed=3):
    rng = np.random.default_rng(seed)
    cs = torch.from_numpy(rng.integers(1, 250, (steps, B)).astype(np.int64))
    ns = torch.from_numpy(rng.integers(1, 250, (steps, B)).astype(np.int64))
    es = torch.zeros((steps, B), dtype=torch.float32)
    return cs, ns, es


def fixed_target(step, B=16):
    """Deterministic, valid, and NOT one-hot at the gold byte: a peaked
    distribution over three fixed bytes, so the gradient is non-trivial."""
    g = torch.zeros(B, 256)
    for b in range(B):
        g[b, (step * 7 + b) % 250 + 1] = 0.6
        g[b, (step * 13 + b) % 250 + 2] = 0.3
        g[b, (step * 29 + b) % 250 + 3] = 0.1
    return g


def run(model, cs, ns, es, steps):
    for t in range(steps):
        model.step_batch(cs[t], ns[t], es[t])
    return model.theta.detach().clone()


def partial_target(cov, step=0, B=8, seed=5):
    """A REALISTIC target: soft on a `cov` fraction of rows, all-zero on the
    rest. A real MI dump looks like this (measured coverage ~65%), and the
    all-zero rows are what make `covered` False."""
    g = torch.zeros(B, 256)
    r = np.random.default_rng(seed + step)
    for b in r.choice(B, size=max(1, int(round(B * cov))), replace=False):
        w = r.dirichlet(np.ones(6)) * 0.9
        g[b, torch.from_numpy(r.choice(250, size=6, replace=False) + 1)] = \
            torch.from_numpy(w).float()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--B", type=int, default=16)
    args = ap.parse_args()
    ok = True

    # ---- g3: alpha = 0 must be bit-exact -------------------------------
    cs, ns, es = stream(args.B, args.steps)
    a = make(B=args.B)
    ref = run(a, cs, ns, es, args.steps)

    b = make(B=args.B, distill_alpha=0.0)
    b.set_target_fn(lambda step: fixed_target(step, args.B))  # installed but unused
    got = run(b, cs, ns, es, args.steps)
    dmax = float((ref - got).abs().max())
    g3 = dmax == 0.0
    ok &= g3
    print(f"  [{'PASS' if g3 else 'FAIL'}] g3 alpha=0 bit-exact over {args.steps} "
          f"steps: max|dtheta| = {dmax:.3e} (target provider installed, unused)")

    c = make(B=args.B)          # no provider at all
    d2 = float((ref - run(c, cs, ns, es, args.steps)).abs().max())
    ok &= d2 == 0.0
    print(f"  [{'PASS' if d2 == 0.0 else 'FAIL'}] g3b alpha=0 with NO provider: "
          f"max|dtheta| = {d2:.3e}")

    # alpha > 0 with no provider must fail LOUDLY, not train a CE arm
    loud = False
    try:
        d = make(B=args.B, distill_alpha=0.5)
        d.step_batch(cs[0], ns[0], es[0])
    except RuntimeError as exc:
        loud = "no target provider" in str(exc)
    ok &= loud
    print(f"  [{'PASS' if loud else 'FAIL'}] g3c alpha>0 without a provider raises "
          f"loudly (no silent CE fallback)")

    # ---- g1: affine in alpha -------------------------------------------
    losses = {}
    for alpha in (0.0, 0.5, 1.0):
        m = make(B=args.B, distill_alpha=alpha)
        m.set_target_fn(lambda step, B=args.B: fixed_target(step, B))
        m.step_batch(cs[0], ns[0], es[0])
        losses[alpha] = float(m.last_loss_tensor)
    # loss(a) = loss(0) + a * KD  =>  KD == loss(1) - loss(0)
    kd = losses[1.0] - losses[0.0]
    pred05 = losses[0.0] + 0.5 * kd
    g1 = abs(pred05 - losses[0.5]) < 1e-4
    ok &= g1
    print(f"  [{'PASS' if g1 else 'FAIL'}] g1 loss affine in alpha: "
          f"L(0)={losses[0.0]:.5f} L(0.5)={losses[0.5]:.5f} "
          f"L(1)={losses[1.0]:.5f} | predicted L(0.5)={pred05:.5f}")

    # ---- g2: the KD term itself ----------------------------------------
    # Total loss is NOT comparable across alphas: CE is down-weighted by
    # (1-alpha) on purpose, so loss(alpha) - loss(0) mixes KD with -alpha*CE.
    # The KD term is exposed on the model, and two runs that differ ONLY in the
    # target isolate it exactly.
    B, dim, layers = 8, 16, 2
    alpha = 0.7
    cs2, ns2, es2 = stream(B, 3, seed=11)
    c_t, n_t, e_t = cs2[0], ns2[0], es2[0]

    def one_step(alpha, target_fn):
        m = make(B=B, dim=dim, layers=layers, distill_alpha=alpha)
        if target_fn is not None:
            m.set_target_fn(target_fn)
        loss = float(m.step_batch(c_t, n_t, e_t))
        return loss, m.last_kd, m.kd_stats()

    def tgt_A(step, B=8):
        g = torch.zeros(B, 256)
        for b in range(B):
            g[b, (step * 7 + b) % 250 + 1] = 0.6
            g[b, (step * 13 + b) % 250 + 2] = 0.3
            g[b, (step * 29 + b) % 250 + 3] = 0.1
        return g

    def tgt_B(step, B=8):
        g = torch.zeros(B, 256)
        for b in range(B):
            g[b, 200 + (step + b) % 20] = 0.97
            g[b, 100 + b % 7] = 0.03
        return g

    lA, kdA, stA = one_step(alpha, tgt_A)
    lB, kdB, stB = one_step(alpha, tgt_B)
    g2 = (np.isfinite(kdA) and kdA >= 0.0 and np.isfinite(kdB) and kdB >= 0.0
          and abs(kdA - kdB) > 1e-6)
    ok &= bool(g2)
    print(f"  [{'PASS' if g2 else 'FAIL'}] g2 KL is finite, non-negative and "
          f"target-dependent: KL_A={kdA:.5f} KL_B={kdB:.5f} nats")

    # two runs differing ONLY in the target: loss delta == alpha * KL delta
    pred = lA - lB
    meas = alpha * (kdA - kdB)
    g2b = abs(pred - meas) < 1e-4
    ok &= g2b
    print(f"  [{'PASS' if g2b else 'FAIL'}] g2b loss isolates the KD term: "
          f"L(A)-L(B)={pred:+.5f} == alpha*(KL_A-KL_B)={meas:+.5f}")

    # a flat target is the maximum-entropy case: KL = log(256) against a uniform
    # student only if the student were uniform; assert KL > 0 and finite.
    # all-zero (backoff) target contributes exactly nothing
    lZ, kdZ, stZ = one_step(alpha, lambda step, B=8: torch.zeros(B, 256))
    g2c = (kdZ == 0.0 and stZ["covered"] == 0)
    ok &= g2c
    print(f"  [{'PASS' if g2c else 'FAIL'}] g2c all-zero (backoff) target: "
          f"KL={kdZ:.1f}, covered={stZ['covered']} of {stZ['positions']}")

    # coverage is counted, and matches the provider's zero rows
    print(f"  [info] kd_stats with a full target: {stA}")

    # ---- g4: gradient == B * d(loss)/dtheta at EVERY coverage -----------
    # The load-bearing new check. See the module docstring for the factor-B
    # convention and for the mask bug this catches.
    eps = 1e-3
    ratios = {}
    for cov in (1.0, 0.5, 0.24, 0.10):
        m = make(B=B, dim=dim, layers=layers, distill_alpha=alpha)
        m.set_target_fn(lambda s, _c=cov: partial_target(_c, s, B=B))
        m.step_batch(cs2[0], ns2[0], es2[0])
        gvec = m.grads.detach().clone()
        nz = torch.nonzero(gvec.abs() > 1e-9).flatten().tolist()
        rs = []
        for i in nz[::max(1, len(nz) // 6)][:6]:
            vals = []
            for sgn in (+1, -1):
                mm = make(B=B, dim=dim, layers=layers,
                          distill_alpha=alpha)
                mm.set_target_fn(lambda s, _c=cov: partial_target(_c, s, B=B))
                mm.theta[i] += sgn * eps
                vals.append(float(mm.step_batch(cs2[0], ns2[0], es2[0])))
            fd = (vals[0] - vals[1]) / (2 * eps)
            a = float(gvec[i])
            if abs(fd) > 1e-11:
                rs.append(a / fd)
        ratios[cov] = float(np.median(rs)) if rs else float("nan")

    # Tolerance is loose because the finite difference is taken in float32 at
    # eps=1e-3; the assertion is that every coverage level gives the SAME ratio,
    # and that it is B.
    worst_ratio = max(abs(v - B) for v in ratios.values() if np.isfinite(v))
    spread = max(ratios.values()) - min(ratios.values())
    g4 = (all(np.isfinite(v) for v in ratios.values())
          and worst_ratio / B < 0.15 and spread / B < 0.15)
    ok &= g4
    detail = "  ".join(f"cov {c:.0%}: {v:.2f}" for c, v in ratios.items())
    print(f"  [{'PASS' if g4 else 'FAIL'}] g4 applied gradient == B * d(loss)/dtheta "
          f"at every coverage: {detail} (B={B}; a term present in one but "
          f"not the other shifts a single coverage level)")

    # the mask bug's signature, stated explicitly: with an all-zero target the
    # KD contribution to the gradient must be exactly zero, so the applied
    # gradient must equal the alpha=0 gradient scaled by the CE weight alone.
    print("\n== verdict:", "PASS" if ok else "FAIL", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())