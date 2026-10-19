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

WHAT IS DELIBERATELY NOT CLAIMED
  The gate gradient. It is implemented in gpu_port.step_batch but is UNVERIFIED,
  and two independent references agree that it is wrong as written: torch
  autograd on the same recurrence, and a central finite difference of the loss,
  both disagree with the engine's formula. Measured at dim=8 layers=2 B=3 T=4
  with the gate perturbed to linspace(-0.7, 0.9):

      dL/dgate   engine vs autograd   max rel dev 4.59 (layer0), 8.19 (layer1)
      dL/ddecay  engine vs autograd   max rel dev 2.99 (layer0), 4.74 (layer1)
      dL/dgate   engine vs finite-diff max rel dev 11.7 (layer0), 22.4 (layer1)

  The missing pieces, for whoever picks this up: ds_t/da_t must carry the
  cross-time chain (what `decaytrace` already does for the ungated engine, and
  what a naive per-byte transcription drops), and the state gradient has to pass
  back through the layer-norm and the z*sigmoid(z) gated linear per layer, not
  through a single decoder-level dx. This is the same class of bug the T-axis
  plan is about: a plausible gradient that is quietly wrong.

  A gate that fails on an unverified gradient is the correct outcome; one that
  passes by testing only the easy half is not.

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
    print("P3 GATE (part 1: verified claims only -- the gate gradient is NOT "
          "claimed, see the module docstring)")
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
    print("\n(c) the gated ARM refuses rather than emitting a wrong gradient")
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
