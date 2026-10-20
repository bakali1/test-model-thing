#!/usr/bin/env python3
"""check_gated_recurrence.py -- P3 gate, PART 1: what is verified.

P3 adds a per-dim gate on the recurrence: a = sigmoid(decay + gate * x),
s = a*s + x.

STATUS: the gated arm is implemented but REFUSES at step_batch, because gating
makes the retention factor per-sample (B, dim) while the ring-credit, tail and
decaytrace machinery all assume a per-dim (dim,) vector, and ring_k is inferred
from sigma_max which then is not a scalar. Until that rework lands the arm
cannot run, so this gate deliberately pins ONLY the claims that hold without
it, and asserts the refusal.

WHAT IS VERIFIED HERE
  (a) gate=0 is bit-identical to the incumbent -- parameters, and the recurrence
      reduces EXACTLY to sigmoid(decay). This is the property that makes the A/B
      clean and lets an incumbent checkpoint be resumed as its own control.
  (b) the gate parameter exists only in the gated arm, is exactly zero at init,
      and is elementwise (layers*dim, not a matmul). Its absence from the
      default engine is what keeps theta's layout -- and check_parity's
      numpy==torch identity -- intact.
  (c) the gate has real authority over the loss, and the arm refuses BEFORE any
      state moves.

THE GATE GRADIENT: KNOWN NOW, AND PINNED ELSEWHERE
  This file used to say the gate gradient was UNKNOWN and that the missing
  pieces were (i) the cross-time chain in ds_t/da_t and (ii) the state
  gradient passing back through the layer norm and the z*sigmoid(z) gated
  linear PER LAYER. Both were real, and both are now settled:

    * the CHAIN: the gate enters a_t at time t only, but a_t keeps
      multiplying s_{t-1} into every later state, so the gate's derivative
      carries the same cross-time chain the decay's `decaytrace` already
      carries -- plus the x_t factor, because the gate reaches a_t through
      gate * x_t:

          gatetrace_t = a_t*gatetrace_{t-1} + a_t(1-a_t)*x_t*s_{t-1}
          dL/dgate    = sum_t dL/ds_t * gatetrace_t

      with the READOUT-ONLY dL/ds_t -- the one the engine's per-byte backward
      produces, because the states are stop_gradient there. The trace is what
      rebuilds the chain the engine does not have. (The four-way truth table
      -- plain/trace x readout-only/total dL/ds_t -- is measured in
      tools/gate_grad_probe.py: the plain form on the readout-only gradient
      is wrong, the trace on it is exact, the plain form on the TOTAL gradient
      is exact too but the engine does not have the total, and the trace on
      the total double counts the chain.)
    * the engine's own line was ALSO summing the wrong axis: `dec[None, :]`
      on a (B, dim) dec broadcasts to (1, B, dim), so `.sum(dim=0)` summed an
      axis of size 1 -- nothing was summed over the batch, and the result
      kept a batch axis a (dim,) gate parameter cannot have.

  The gradient is therefore no longer "unknown": it is implemented
  (gpu_port.gate_gradient / numpy_port.gate_gradient) and pinned against
  torch autograd AND against a central finite difference of the loss by
  tests/check_gate_grad.py. That file is the one to read for the gradient;
  this one keeps the claims that are about the ARM, and the refusal.

  What is still NOT claimed here: that the gated arm can RUN. It cannot, and
  the blocker is the per-sample CREDIT machinery (the ring credit, the tail
  and ring_k's inference), not the gradient. That is a separate, larger piece
  of work -- PLAN-2026-10-11-T-AXIS.md P3.

The plan wrote the update as s = a*s + (1-a)*x. That form cannot satisfy (a):
(1-a) is not 1.0, so gate=0 would still not reproduce the incumbent and the
"resumable A/B" property the same paragraph asks for is impossible. We implement
s = a*s + x and pin that choice here.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_port

# DEVICE CONVENTION. These gates compare two implementations of the same
# thing (dense trace vs ring+tail, gated vs ungated, half-life init vs the
# default build), so they pin the device to "cpu" on purpose: the claim they
# make is device-independent, and letting the engine pick the device silently
# turned them into CUDA gates on a GPU box, where their CPU-built input
# tensors die with "Expected all tensors to be on the same device" -- they
# were not testing anything there, they were crashing. The device-parameterised
# gate is tests/check_seqlen_credit.py --device, which runs the T-axis claim on
# both.
DEV = "cpu"


DIM = 8
LAYERS = 2
B = 3
T = 4
SEED = 0

PASS = True


def check(name, ok, detail=""):
    global PASS
    if not ok:
        PASS = False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))


def batch(seed):
    # step_batch takes (B, T) -- the T-axis window shape -- NOT (T, B).
    rng = np.random.default_rng(seed)
    cs = torch.from_numpy(rng.integers(1, 250, (B, T)).astype(np.int64))
    ns = torch.from_numpy(rng.integers(1, 250, (B, T)).astype(np.int64))
    return cs, ns, torch.zeros((B, T), dtype=torch.float32)


def _ln(s, g, b):
    mu = s.mean()
    v = s.var(unbiased=False) + 1e-4
    return (s - mu) / torch.sqrt(v) * g + b


def _forward(m, cs, ns, ends, force_ungated=False):
    """The gated recurrence over cloned leaf tensors, differentiable.

    CE only: the engine's aux terms (variance hinge, embed MSE, end-of-text) are
    orthogonal to the gate and omitting them keeps this comparable to itself,
    which is what the gate=0 test needs. l.decay/l.gate are views into theta and
    so are not leaves; cloning gives autograd something to hang a graph on.
    """
    dec = [l.decay.detach().clone().requires_grad_(True) for l in m.layers]
    # The ungated engine has no gate parameter, so substitute a zero leaf: the
    # recurrence then reduces to sigmoid(decay) on IDENTICAL code, which is what
    # makes the gate=0 comparison meaningful rather than tautological.
    gate = [((l.gate.detach().clone() if l.gate is not None
              else torch.zeros(m.dim, dtype=torch.float32))
             .requires_grad_(True)) for l in m.layers]
    gs = [l.g.detach().clone().requires_grad_(True) for l in m.layers]
    bs = [l.b.detach().clone().requires_grad_(True) for l in m.layers]
    Ws = [l.W.detach().clone().requires_grad_(True) for l in m.layers]
    emb = m.embed.detach().clone().requires_grad_(True)
    Wd = m.Wd.detach().clone().requires_grad_(True)
    bd = m.bd.detach().clone().requires_grad_(True)

    loss = torch.zeros((), dtype=torch.float32)
    for b in range(B):
        states = [torch.zeros(m.dim, dtype=torch.float32) for _ in m.layers]
        for t in range(T):
            enc = emb[cs[b, t]]
            xin = enc
            for i in range(m.layercount):
                a = (torch.sigmoid(dec[i]) if force_ungated
                     else torch.sigmoid(dec[i] + gate[i] * xin))
                s = a * states[i] + enc
                h = _ln(s, gs[i], bs[i])
                z = h @ Ws[i].t()
                y = z * torch.sigmoid(z)
                states[i] = s
                xin = xin + y
            out = xin @ Wd.t() + bd
            loss = loss + torch.logsumexp(out, 0) - out[ns[b, t]]
    return loss / (B * T), dec, gate


def _loss(m, cs, ns, ends, force_ungated=False):
    loss, _d, _g = _forward(m, cs, ns, ends, force_ungated)
    return float(loss.detach())


def main():
    print("P3 GATE (part 1: the ARM's claims -- the gate GRADIENT is pinned by "
          "tests/check_gate_grad.py, see the module docstring)")
    print(f"  dim={DIM} layers={LAYERS} B={B} T={T} seed={SEED}\n")

    cs, ns, ends = batch(SEED)
    a = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED, gated=True)
    b = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED, gated=False)

    # ---- (a) gate=0 reproduces the incumbent exactly ---------------------
    print("(a) gate=0 reproduces the incumbent exactly")
    a_th = torch.cat([a.views[n].reshape(-1) for n, _ in a.shapes
                      if not n.endswith(".gate")])
    check("non-gate parameters are bit-identical at init",
          torch.equal(a_th, b.theta),
          f"{int((a_th != b.theta).sum())} differing elements")
    check("the gate itself starts at exactly zero",
          all(float(l.gate.abs().max()) == 0.0 for l in a.layers))
    la = _loss(a, cs, ns, ends)
    lb = _loss(b, cs, ns, ends, force_ungated=True)
    check("the recurrence reduces EXACTLY to sigmoid(decay) at gate=0",
          la == lb, f"gated {la!r} vs ungated {lb!r}")

    # ---- (b) the parameter exists only in the gated arm -----------------
    print("\n(b) the gate parameter exists only in the gated arm")
    check("no layer{i}.gate key in the default shapes",
          not any(n.endswith(".gate") for n, _ in b.shapes),
          "its absence is what keeps check_parity's theta layout valid")
    check("gated arm adds exactly one gate vector per layer",
          sum(1 for n, _ in a.shapes if n.endswith(".gate")) == LAYERS)
    n_gate = sum(int(np.prod(s)) for n, s in a.shapes if n.endswith(".gate"))
    check("gate is elementwise, not a matmul", n_gate == LAYERS * DIM,
          f"{n_gate} params == {LAYERS}*{DIM}")

    # ---- (c) the arm refuses, before any state moves --------------------
    print("\n(c) the gated ARM refuses rather than emitting a wrong credit")
    snap = (a.theta.detach().clone(), a.m.detach().clone(), a.t)
    try:
        a.new_states(B)
        a.step_batch(cs, ns, ends)
        check("gated step_batch refuses", False, "it ran")
    except NotImplementedError as e:
        check("gated step_batch refuses", True,
              str(e).split(":", 1)[1].strip()[:58] + "...")
    check("the refusal fires before theta, moments or t move",
          torch.equal(snap[0], a.theta.detach())
          and torch.equal(snap[1], a.m.detach()) and a.t == snap[2])

    # ---- (d) the gate has real authority over the loss ------------------
    print("\n(d) the gate has real authority over the loss")
    with torch.no_grad():
        for l in a.layers:
            l.gate.add_(0.5)
    moved = _loss(a, cs, ns, ends)
    check("a nonzero gate changes the loss", moved != la,
          f"{la:.6f} -> {moved:.6f}")

    # ---- (e) the ungated engine still runs normally ---------------------
    print("\n(e) the ungated engine is unaffected")
    before = b.theta.detach().clone()
    b.new_states(B)
    b.step_batch(cs, ns, ends)
    check("the ungated engine still steps", b.t == 1)
    check("and it actually updated theta",
          not torch.equal(before, b.theta.detach()))

    print("\n" + ("P3 GATE (part 1): PASS" if PASS else "P3 GATE (part 1): FAIL"))
    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
