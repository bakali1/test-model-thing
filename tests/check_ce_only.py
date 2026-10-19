#!/usr/bin/env python3
"""GATE: the CE-only switch must be a no-op when off and an exact zero when on.

`gpu_port.py:404` refers to this file, but it did not exist -- the switch was
implemented (uncommitted) with its gate missing. PLAN-2026-10-06 priority 3
needs CE-only for the synthetic memory benchmark, and the switch is only
trustworthy if these four things are proven:

  1. **Off is bit-identical.** With `ce_only=False`, `aux_w` must be exactly
     1.0, so every auxiliary term keeps its historical value. Proven by
     `last_aux_masked == last_aux_raw` bit-exactly (an fp32 `x * 1.0 == x`
     check that catches any accidental reordering), plus run-to-run
     determinism at a fixed seed.
  2. **On is exactly zero.** With `ce_only=True`, the variance hinge, the
     embed-MSE/JEPA term and the end-of-text term must contribute EXACTLY 0.0
     to the loss and to the gradient -- not "small". The RAW values are
     asserted non-zero in the same breath, which is what makes the check mean
     anything: a zeroed zero proves nothing.
  3. **The stop head gets no gradient.** Its weights then move only by weight
     decay, so `dWs`/`dbs` must be exactly 0.0.
  4. **KD and CE-only are mutually exclusive**, refused at construction rather
     than silently disabled.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gpu_port


def make(dim=32, layers=2, B=8, seed=123, **kw):
    torch.manual_seed(seed)
    m = gpu_port.GpuModel(dim=dim, layers=layers, device="cpu", **kw)
    for layer in m.layers:
        layer.decay[0] = 4.0          # force a hot dim so the tail path is live
    m.new_states(B)
    return m


def drive(m, steps=60, seed=9):
    """Run `steps` steps, returning the loss trajectory and the per-step
    (masked, raw) auxiliary triples."""
    g = torch.Generator().manual_seed(seed)
    losses, triples = [], []
    for _ in range(steps):
        c = torch.randint(1, 200, (m.B,), generator=g)
        n = torch.randint(1, 200, (m.B,), generator=g)
        e = (torch.rand(m.B, generator=g) < .1).float()
        losses.append(m.step_batch(c, n, e))
        triples.append((tuple(float(x) for x in m.last_aux_masked),
                        tuple(float(x) for x in m.last_aux_raw)))
    return losses, triples


def nrel(a, b):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return float((a - b).abs().max() / (b.square().mean().sqrt() + 1e-30))


def main():
    # ---- 1. off: bit-identical, and determinism at a fixed seed -------------
    a = make()
    la, ta = drive(a)
    b = make()
    lb, _ = drive(b)
    assert la == lb, "ce_only=False is not deterministic at a fixed seed"
    assert nrel(a.theta, b.theta) == 0.0, "ce_only=False theta is not deterministic"

    raw_nonzero = 0
    for masked, raw in ta:
        for mv, rv in zip(masked, raw):
            # x * 1.0 == x in fp32, exactly. Any other value of aux_w shows here.
            assert mv == rv, f"ce_only=False perturbed an auxiliary term: {mv!r} vs {rv!r}"
            if rv != 0.0:
                raw_nonzero += 1
    assert raw_nonzero > 0, (
        "every auxiliary term was identically zero, so this gate proved nothing "
        "about masking -- raise the step count or the dim so the terms are live")
    print(f"ce_only=False: masked == raw bit-exactly over {len(ta)} steps "
          f"({raw_nonzero} non-zero term-steps), deterministic: PASS")

    # ---- 2 + 3. on: exactly zero, with the raw values proven non-zero -------
    m = make(ce_only=True)
    losses, triples = drive(m)
    for step, (masked, raw) in enumerate(triples):
        for i, (mv, rv) in enumerate(zip(masked, raw)):
            assert mv == 0.0, f"step {step} term {i} masked to {mv!r}, expected 0.0"
            assert rv != 0.0, (
                f"step {step} term {i} was already zero before masking; the gate "
                f"cannot distinguish masking from absence")
    # step_batch returns a python float when sync_metrics is on (the default).
    assert all(math.isfinite(l) for l in losses), "CE-only produced non-finite loss"

    # The stop head must receive exactly no gradient under CE-only.
    assert float(m.gviews["Ws"].abs().sum()) == 0.0, (
        "ce_only=True left a gradient on Ws; the stop-head gradient must be "
        "exactly zero so its weights move only by weight decay")
    assert float(m.gviews["bs"].abs().sum()) == 0.0, "ce_only=True left a gradient on bs"
    print(f"ce_only=True: 3 aux terms exactly 0.0 over {len(triples)} steps, "
          f"raw values non-zero, stop-head gradient exactly 0: PASS")

    # "CE-only still trains" cannot be asserted as a falling loss at this scale:
    # at dim=32/layers=2 both arms are flat over 200 steps (measured: full 6.65 ->
    # 6.64, CE-only 5.62 -> 5.62). Assert the two things that hold at any scale
    # instead: the weights actually move, and the total is strictly lower, since
    # CE-only removes three non-negative terms.
    theta0 = m.theta.clone()
    drive(m, steps=40)
    assert nrel(theta0, m.theta) > 0.0, "CE-only did not update any weight"

    full = make()
    drive(full, steps=40)
    assert full.last_loss_tensor > m.last_loss_tensor, (
        f"CE-only total {m.last_loss_tensor:.6f} should be below the full loss "
        f"{full.last_loss_tensor:.6f}: three non-negative terms were removed")
    print(f"CE-only trains (theta moves) and its total is below the full loss "
          f"({m.last_loss_tensor:.4f} < {full.last_loss_tensor:.4f}): PASS")

    # ---- 4. KD and CE-only are mutually exclusive ---------------------------
    for alpha in (0.1, 0.5, 1.0):
        try:
            make(ce_only=True, distill_alpha=alpha)
        except ValueError as exc:
            assert "ce_only" in str(exc), f"unhelpful error: {exc}"
        else:
            raise AssertionError(
                f"ce_only=True accepted distill_alpha={alpha}; a CE-only run "
                f"strips KD by design and must refuse loudly")
    # alpha=0 is the one legal combination and must construct cleanly.
    make(ce_only=True, distill_alpha=0.0)
    print("ce_only refuses KD at construction (alpha>0), accepts alpha=0: PASS")

    print("CE-ONLY: PASS")


if __name__ == "__main__":
    main()
