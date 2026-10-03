#!/usr/bin/env python3
"""check_lr_roles.py — GATE for the per-role learning rate (`--lr-decay`).

    .venv/bin/python tests/check_lr_roles.py [--steps 120]

Free (CPU, ~1 min). The knob scales the learning rate of ONLY the 16 per-layer
`.decay` slices (512 coords each, 8,192 total = 0.183% of the parameters). It
is the role-specific-LR axis of the planned factorial. Three claims have to
hold, and only the first is automatic:

  L1  `lr_decay == lr` reproduces the historical scalar-LR engine BIT-EXACTLY.
      The optimizer multiplies by a per-coordinate vector now; that is only
      allowed if every coordinate holds exactly float32(lr). A CE "close
      enough" is not the bar -- this repo gates float reassociation at
      bit-equality, because a silently different trajectory invalidates every
      A/B arm compared against a stored checkpoint.
  L2  `lr_decay != lr` actually MOVES the parameters (otherwise the flag is a
      no-op that would silently produce four identical arms on the GPU).
  L3  the difference is confined to the decay coordinates. If any trunk
      coordinate moved differently, the index arithmetic in `_role_lr_vec` is
      wrong and the arm is measuring the wrong thing.

Exit 0 = PASS.
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np
import torch

import gpu_port
import telemetry
from gpu_port import GpuModel


def _decay_index(model):
    """The flat indices the role LR is allowed to touch."""
    idx, off = [], 0
    for name, shape in model.shapes:
        k = int(np.prod(shape)) if shape else 1
        if name.endswith(".decay"):
            idx.append(np.arange(off, off + k))
        off += k
    return np.concatenate(idx) if idx else np.zeros(0, int)


def run(lr, lr_decay, steps, dim, layers, B):
    torch.manual_seed(0)
    m = GpuModel(dim=dim, layers=layers, device="cpu", ring_credit="full",
                 seed=0, lr=lr, lr_decay=lr_decay)
    m.new_states(B)
    g = torch.Generator().manual_seed(11)
    cs = torch.randint(1, 250, (steps + 1, B), generator=g)
    ns = torch.randint(1, 250, (steps + 1, B), generator=g)
    es = (torch.rand(steps, B, generator=g) < 0.02).float()
    losses = []
    for t in range(steps):
        losses.append(float(m.step_batch(cs[t], ns[t], es[t])))
    return m, np.array(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()
    ok = True

    def t(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'PASS' if good else 'FAIL'}] {name}" +
              (f" — {detail}" if detail else ""))
        return good

    n_decay = 0
    probe = GpuModel(dim=args.dim, layers=args.layers, device="cpu", seed=0)
    n_decay = len(_decay_index(probe))
    n_tot = probe.theta.numel()
    print(f"== setup ==\n  dim={args.dim} layers={args.layers} B={args.B} "
          f"steps={args.steps}")
    print(f"  parameters {n_tot:,} | decay-role coordinates {n_decay:,} "
          f"({n_decay/n_tot*100:.3f}%)\n")
    print("== L1: lr_decay == lr must be BIT-EXACT vs the scalar-LR engine ==")
    m_scalar, l_scalar = run(args.lr, None, args.steps, args.dim, args.layers, args.B)
    m_equal, l_equal = run(args.lr, args.lr, args.steps, args.dim, args.layers, args.B)
    same_theta = torch.equal(m_scalar.theta, m_equal.theta)
    same_mom = (torch.equal(m_scalar.m, m_equal.m)
                and torch.equal(m_scalar.v, m_equal.v))
    maxdl = float(np.abs(l_scalar - l_equal).max())
    t("theta bit-identical after %d steps" % args.steps, same_theta)
    t("AdamW moments bit-identical", same_mom)
    t("per-step loss bit-identical", maxdl == 0.0, f"max|dloss| = {maxdl:.3e}")

    print("\n== L2/L3: lr_decay != lr must move ONLY the decay coordinates ==")
    m_up, _ = run(args.lr, args.lr * 16, args.steps, args.dim, args.layers, args.B)
    d = (m_up.theta - m_equal.theta).abs()
    t("parameters actually moved", float(d.max()) > 0.0,
      f"max|dtheta| = {float(d.max()):.3e}")
    idx = _decay_index(m_equal)
    moved = np.zeros(n_tot, bool)
    moved[idx] = d.detach().numpy()[idx] > 0
    t("every decay coordinate moved", bool(moved[idx].all()),
      f"{int(moved[idx].sum())}/{len(idx)}")
    trunk = np.setdiff1d(np.arange(n_tot), idx)
    t("NO trunk coordinate moved", not bool(moved[trunk].any()),
      f"{int(moved[trunk].sum())} of {len(trunk)} trunk coords changed "
      f"(max {float(d[trunk].max()):.3e})" if moved[trunk].any()
      else f"all {len(trunk)} trunk coords bit-identical")
    rms = float(m_equal.theta.pow(2).mean().sqrt())
    t("change is measurable against parameter scale",
      float(d[idx].max()) / rms > 1e-9,
      f"max|dtheta|/rms on the decay role = {float(d[idx].max())/rms:.3e}")

    # the role LR must also scale weight decay on that role (documented
    # param-group semantics), so state that we are not silently ignoring it
    lv = m_up._role_lr_vec()
    t("lr vector is `lr` everywhere except the decay role",
      torch.equal(lv[idx], torch.full_like(lv[idx], args.lr * 16))
      and torch.equal(lv[torch.from_numpy(trunk)],
                      torch.full((len(trunk),), args.lr, dtype=lv.dtype)),
      f"trunk={float(lv[torch.from_numpy(trunk)].unique()[0]):.2e} "
      f"decay={float(lv[idx].unique()[0]):.2e}")

    print("\n== control: the historical path is unchanged when the flag is absent ==")
    m_none, _ = run(args.lr, None, 3, args.dim, args.layers, args.B)
    lv0 = m_none._role_lr_vec()
    t("lr_decay unset -> uniform vector equal to float32(lr)",
      bool((lv0 == float(np.float32(args.lr))).all()),
      f"unique = {float(lv0.unique()[0]):.10g} vs float32(lr) = "
      f"{float(np.float32(args.lr)):.10g}")

    print("\nCHECK_LR_ROLES:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())