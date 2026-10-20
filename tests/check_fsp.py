#!/usr/bin/env python3
"""check_fsp.py -- the FSP auxiliary loss, pinned against finite differences.

WHAT THIS PINS. The Future State Prediction auxiliary (FlashLM, Chang 2026,
§3.1) adds a read-out on the RECURRENT STATE: every r bytes, the model
predicts which bytes appear in the next tau bytes, as a binary multi-hot, with
BCE (pos_weight 50); L = CE + alpha * L_FSP.

Three things can be wrong in such an addition, and each is checked here
against a central finite difference of the loss, which needs no reference
implementation and no trust in the engine's algebra:

  1. THE HEAD'S OWN GRADIENT. If the BCE's derivative, the pos_weight or the
     alpha/1/B scaling is wrong, the head learns the wrong thing.
  2. THE GRADIENT'S REACH. The whole point is that the auxiliary reaches the
     MEMORY: the head reads the state, and the engine's existing credit
     machinery (the ring for the embedding, the decaytrace for the decay
     logits) must carry the FSP gradient to the decay logits. If it does not,
     FSP is a read-out on a frozen state and the idea is dead on arrival.
  3. THE DEFAULT PATH. With alpha = 0 the engine must be bit-identical to the
     no-FSP engine -- the head's parameters do not exist, the layout is
     unchanged, and the loss is the same number.

The finite differences are taken with lr = 0 and the state reset before every
evaluation, so a run measures the function, not a trajectory.

Run:  python tests/check_fsp.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gpu_port  # noqa: E402

DEV = "cpu"
DIM = 8
LAYERS = 2
B = 3
T = 2
ALPHA = 0.1
POSW = 50.0
# The finite-difference step. It has to be LARGE: the loss is float32, so
# its ULP near 13 is ~1e-6, and a central difference of eps multiplies that
# quantum by 1/(2*eps) -- at eps=1e-3 the finite difference of a 1e-3
# gradient is pure cancellation noise, and two different coordinates come
# out EXACTLY equal, which is the tell. At eps=0.05 the quantum is ~1e-5 and
# the smallest gradient checked here (~1e-3) is still resolved to ~1%.
EPS = 0.05

_fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def build(alpha=ALPHA, seed=0):
    torch.manual_seed(seed)
    return gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=seed,
                              fsp_alpha=alpha, fsp_r=1, fsp_tau=4,
                              fsp_pos_weight=POSW, fsp_layer=-1)


def data():
    g = torch.Generator().manual_seed(7)
    cs = torch.randint(0, 256, (B, T), generator=g)
    ns = torch.randint(0, 256, (B, T), generator=g)
    ends = torch.zeros(B, T)
    tgt = torch.zeros(B, T, 256)
    tgt[:, 0, 17] = 1.0
    tgt[:, 0, 200] = 1.0
    tgt[:, 1, 99] = 1.0
    return cs, ns, ends, tgt


def run(model, cs, ns, ends, tgt, mask=(1, 1)):
    """One loss evaluation from a RESET state, with no parameter movement."""
    saved = model.theta.detach().clone()
    model.new_states(B)
    model.lr = 0.0
    try:
        loss = model.step_batch(cs, ns, ends,
                                fsp=(mask, tgt) if tgt is not None else None)
        out = float(loss)
    finally:
        with torch.no_grad():
            model.theta.copy_(saved)
    return out


def analytic(model, cs, ns, ends, tgt):
    """The engine's own gradient for one window."""
    model.new_states(B)
    model.lr = 0.0
    saved = model.theta.detach().clone()
    model.step_batch(cs, ns, ends, fsp=((1, 1), tgt))
    grads = model.grads.detach().clone()
    with torch.no_grad():
        model.theta.copy_(saved)
    return grads


def coord_index(shapes, name, coord):
    """The flat index of `coord` inside the view called `name`."""
    off = 0
    for n, s in shapes:
        size = int(np.prod(s))
        if n == name:
            return off + coord
        off += size
    raise KeyError(name)


def fd_grad(model, shapes, name, coord, cs, ns, ends, tgt):
    """Central difference of the REPORTED loss wrt one flat coordinate.

    The engine's gradients are the gradient of the batch SUM, while the
    reported loss is the batch MEAN (the `/B` in the loss expression is for
    reporting -- the repo's P0 gate documents the same asymmetry for the
    decay rows). So the finite difference of the reported loss has to be
    scaled by B to be comparable with the engine's gradient. Getting this
    wrong makes a correct engine look wrong by exactly B, which is what a
    first reading of this check did.
    """
    idx = coord_index(shapes, name, coord)
    base = model.theta.detach().clone()
    with torch.no_grad():
        model.theta[idx] += EPS
    lp = run(model, cs, ns, ends, tgt)
    with torch.no_grad():
        model.theta.copy_(base)
        model.theta[idx] -= EPS
    lm = run(model, cs, ns, ends, tgt)
    with torch.no_grad():
        model.theta.copy_(base)
    return (lp - lm) / (2.0 * EPS) * B


def main():
    print("=" * 78)
    print("FSP (Future State Prediction): the auxiliary loss, pinned by finite differences")
    print("=" * 78)
    cs, ns, ends, tgt = data()

    # ---- (1) the head's own gradient -----------------------------------------
    print("(1) the head's own gradient vs a central finite difference")
    m = build()
    grads = analytic(m, cs, ns, ends, tgt)
    for name, coord in (("fsp.W", 0), ("fsp.W", 37), ("fsp.b", 3),
                         ("fsp.b", 200)):
        got = float(grads[coord_index(m.shapes, name, coord)])
        want = fd_grad(m, m.shapes, name, coord, cs, ns, ends, tgt)
        rel = abs(got - want) / max(1e-9, abs(want))
        check(f"d/d{name}[{coord}] engine {got:+.6e} vs finite diff {want:+.6e}",
              rel < 0.05, f"rel dev {rel:.2e}")

    # ---- (2) the gradient REACHES the memory ------------------------------
    print("(2) the auxiliary reaches the DECAY logits (the memory), not just the head")
    for layer in range(LAYERS):
        name = f"layer{layer}.decay"
        got = float(grads[coord_index(m.shapes, name, 2)])
        want = fd_grad(m, m.shapes, name, 2, cs, ns, ends, tgt)
        rel = abs(got - want) / max(1e-9, abs(want))
        check(f"d/d{name}[2] engine {got:+.6e} vs finite diff {want:+.6e}",
              rel < 0.02, f"rel dev {rel:.2e}")
    # and the FSP term must be a NON-NEGLIGIBLE part of that gradient: a
    # gradient that matches the finite difference but is dominated by the CE
    # would mean the auxiliary is decoration.
    m_nofsp = build(alpha=0.0)
    g_nofsp = analytic(m_nofsp, cs, ns, ends, None)
    # the two models have different theta LAYOUTS (the FSP arm has two extra
    # views), so the comparison is over the SHARED views only.
    shared = ["embed", "Wd", "bd", "Ws", "bs"] + [
        f"layer{i}.{k}" for i in range(LAYERS) for k in ("W", "g", "b", "decay")]
    d_fsp = 0.0
    d_ce = 0.0
    for nm in shared:
        a = grads[coord_index(m.shapes, nm, 0):
                  coord_index(m.shapes, nm, 0) + int(np.prod(
                      dict(m.shapes)[nm]))]
        b = g_nofsp[coord_index(m_nofsp.shapes, nm, 0):
                    coord_index(m_nofsp.shapes, nm, 0) + int(np.prod(
                        dict(m_nofsp.shapes)[nm]))]
        d_fsp = max(d_fsp, float((a - b).abs().max()))
        d_ce = max(d_ce, float(b.abs().max()))
    check("the FSP term changes the gradient", d_fsp > 1e-6,
          f"max |grad_fsp - grad_ce| = {d_fsp:.3e} (CE alone {d_ce:.3e})")

    # ---- (3) the default path is untouched --------------------------------
    print("(3) alpha=0 leaves the default engine bit-identical")
    a = build(alpha=0.0)
    b = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=0)
    check("the theta layout is the same", a.shapes == b.shapes,
          f"{[n for n, _ in a.shapes]}")
    la = run(a, cs, ns, ends, None)
    lb = run(b, cs, ns, ends, None)
    check("the loss is the same number", la == lb, f"{la!r} vs {lb!r}")
    ga = analytic(a, cs, ns, ends, None)
    gb = analytic(b, cs, ns, ends, None)
    check("the gradient is bit-identical",
          bool(torch.equal(ga, gb)),
          f"max |diff| = {float((ga - gb).abs().max()):.3e}")

    # ---- (4) the loss term is the BCE it claims to be ---------------------
    print("(4) the reported FSP term equals the BCE it claims to be")
    m2 = build()
    m2.new_states(B)
    m2.lr = 0.0
    loss_fsp = float(m2.step_batch(cs, ns, ends, fsp=((1, 1), tgt)))
    last_fsp = m2.last_fsp
    n_pos = m2.fsp_positions
    m2.new_states(B)
    m2.lr = 0.0
    loss_plain = float(m2.step_batch(cs, ns, ends, fsp=((0, 0), tgt)))
    # the loss carries alpha * (the per-position batch MEAN); last_fsp is the
    # mean over the positions of the same quantity.
    expect = alpha_total = ALPHA * last_fsp * n_pos
    check("loss_fsp - loss_ce == alpha * FSP (per position)",
          abs((loss_fsp - loss_plain) - expect) < 1e-4,
          f"implied {(loss_fsp - loss_plain):.6f} vs expected {expect:.6f} "
          f"({n_pos} positions)")

    # ---- (5) the arm is MEASURABLE ---------------------------------------
    # The ruler runs on the numpy reference, and the FSP head's parameters
    # live in theta -- so a checkpoint the ruler cannot LOAD makes the whole
    # arm unmeasurable. This check writes a checkpoint and reads it back
    # through the ruler's own loading path.
    print("(5) the arm is measurable: the ruler's loader reads the checkpoint")
    import tempfile
    import tools.membench.model_backend as mb
    m3 = build()
    m3.new_states(B)
    m3.step_batch(cs, ns, ends, fsp=((1, 1), tgt))
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        path = fh.name
    m3.save(path)
    try:
        mm, info = mb.load_frozen(path)
        ok = ("fsp.W" in mm.views)
        check("the numpy reference loads the FSP checkpoint",
              ok and int(mm.theta.size) == int(m3.theta.numel()),
              f"numpy theta {mm.theta.size} vs torch {m3.theta.numel()}")
        same = bool(np.array_equal(np.asarray(mm.views["fsp.W"]),
                                   m3.views["fsp.W"].detach().numpy()))
        check("the head's parameters line up with the checkpoint", same)
    except Exception as exc:                      # noqa: BLE001
        check("the numpy reference loads the FSP checkpoint", False,
              f"{type(exc).__name__}: {exc}")
    finally:
        os.unlink(path)

    print()
    if _fails:
        print(f"FAILED: {len(_fails)} check(s): {_fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
