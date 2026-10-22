#!/usr/bin/env python3
"""check_gate_grad.py -- the P3 gate GRADIENT, pinned against two references.

WHAT THIS GATE DOES, PLAINLY
----------------------------
The engine's gated `step_batch` RUNS (on the dense trace -- see
docs/design/P3-GATE-ANALYSIS.md 5.1). This gate still does NOT run a full
training step: it tests the GRADIENT FORMULA on the recurrence directly --
and it does so through the engine's OWN function, `gpu_port.gate_gradient`,
the one `step_batch` calls, so what is pinned is the shipped code and not a
re-derivation of it. The last section asserts that the arm runs, so the file
cannot be read as claiming more than it checks.

The CREDIT -- the part the gated step_batch's embedding gradient depends on,
and the part that used to be unpinned -- is a SEPARATE gate:
tests/check_gated_credit.py compares the gated step_batch's full gradient
against torch autograd, and the dense trace's exactness under a per-sample
retention.

THE MECHANISM (now known, and measured -- tools/gate_grad_probe.py)
--------------------------------------------------------------------
The gate makes the retention input-dependent:

    a_t = sigmoid(decay + gate * x_t)      (per SAMPLE: B x dim)
    s_t = a_t * s_{t-1} + e_t
    h_t = LayerNorm(s_t) -> W -> z_t

The gate enters a_t at time t ONLY -- but a_t does not stop at s_t: it
multiplies s_{t-1} into s_t, and s_t then keeps propagating, multiplied by
every later a. So the gate's derivative carries the SAME cross-time chain the
decay's `decaytrace` carries, plus the x_t factor (the gate reaches a_t
through gate * x_t):

    gatetrace_t = a_t * gatetrace_{t-1} + a_t(1-a_t) * x_t * s_{t-1}
    dL/dgate    = sum_t dL/ds_t * gatetrace_t

`dL/ds_t` here is the READOUT-ONLY state gradient -- the one the engine's
per-byte backward produces, because the states are stop_gradient there -- and
the trace is what rebuilds the chain the engine does not have. The four-way
truth table (plain/trace x readout-only/total dL/ds_t) is in
tools/gate_grad_probe.py: trace + readout-only is the only pair that is both
computable in the engine and exact. Feeding the trace the TOTAL state
gradient double counts the chain, because the trace's chain and the total
gradient's chain are the same chain.

The engine's ORIGINAL gate line was wrong in two ways at once: it dropped
the cross-time chain, and it summed the wrong axis (`dec[None, :]` on a
(B, dim) dec broadcasts to (1, B, dim), so `.sum(dim=0)` summed an axis of
size 1 -- nothing was summed over the batch and the result kept a batch axis
it must not have). Both are reproduced below as documentation of what was
fixed.

WHAT IS PINNED HERE
  (a) the engine's gate gradient == torch autograd on the same recurrence;
  (b) the engine's gate gradient == a central finite difference of the loss;
  (c) numpy_port's copy of the formula == gpu_port's, element by element
      (numpy_port is the reference implementation of the engine, and
      check_parity.py compares the two engines element by element);
  (d) the per-dim and per-sample retention both work in the trace;
  (e) the engine's gated step_batch RUNS (the refusal is lifted; the credit
      it needed is pinned by tests/check_gated_credit.py).
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gpu_port
import numpy_port

# DEVICE CONVENTION (like the other gates here): the claim is
# device-independent, and the inputs are built on the CPU on purpose.
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


def rel_dev(got, ref):
    """max|got - ref| against the scale of ref (the signed reference)."""
    scale = float(ref.abs().max())
    scale = scale if scale > 1e-30 else 1e-30
    return float((got - ref).abs().max() / scale)


# ---------------------------------------------------------------------------
# the window, over the ENGINE's own layer code
# ---------------------------------------------------------------------------
class _Swapped:
    """Point the engine's layers at leaf tensors for the duration of a run.

    TorchLayer.forward reads self.decay/self.gate/self.g/self.b/self.W, so
    swapping them in is what makes this test drive the engine's own layer
    code -- its layer norm, its silu, its state update -- instead of a
    re-implementation of it that could drift from the shipped one.
    """

    def __init__(self, m, pl):
        self.m = m
        self.pl = pl
        self.saved = []

    def __enter__(self):
        for l, p in zip(self.m.layers, self.pl):
            self.saved.append((l, l.decay, l.gate, l.g, l.b, l.W))
            l.decay = p["decay"]
            l.gate = p["gate"]
            l.g = p["g"]
            l.b = p["b"]
            l.W = p["W"]
        return self

    def __exit__(self, *exc):
        for l, dec, gate, g, b, W in self.saved:
            l.decay, l.gate, l.g, l.b, l.W = dec, gate, g, b, W
        self.saved = []


def leaves(m, dtype):
    """The model's parameters as leaves of the given dtype."""
    P = {"embed": m.embed.detach().clone().to(dtype).requires_grad_(True),
         "Wd": m.Wd.detach().clone().to(dtype).requires_grad_(True),
         "bd": m.bd.detach().clone().to(dtype).requires_grad_(True)}
    PL = []
    for l in m.layers:
        PL.append({
            "decay": l.decay.detach().clone().to(dtype).requires_grad_(True),
            "gate": l.gate.detach().clone().to(dtype).requires_grad_(True),
            "g": l.g.detach().clone().to(dtype).requires_grad_(True),
            "b": l.b.detach().clone().to(dtype).requires_grad_(True),
            "W": l.W.detach().clone().to(dtype).requires_grad_(True),
        })
    return P, PL


def forward(m, P, PL, cs, ns, detach_states):
    """One window of the gated recurrence, over the engine's own layers.

    `detach_states=True` cuts the cross-time chain EXACTLY the way the
    engine's stop_gradient does, and no other way: the state a timestep
    reads (s_{t-1}) is detached before the layer's forward, and the state it
    produces is detached before the next timestep reads it. The state's own
    value is untouched, and the gradient that reaches the state is therefore
    its READOUT-ONLY part -- the quantity the engine's per-byte backward
    produces, and the one the gate trace is built to be combined with.
    """
    dtype = P["embed"].dtype
    states = [torch.zeros(B, m.dim, dtype=dtype) for _ in m.layers]
    loss = torch.zeros((), dtype=dtype)
    # Per TIMESTEP, then per LAYER: q["a"][t][i] is layer i's effective
    # retention at timestep t. (The layer axis is the inner one because the
    # engine's trace block walks layers inside the timestep loop.)
    q = {"a": [], "x_in": [], "s_prev": [], "state": []}
    ar = torch.arange(B)
    for t in range(T):
        # cs/ns are (B, T): column t is this timestep's byte for every stream
        enc = P["embed"][cs[:, t]]
        x = enc
        ta, tx, ts, tstate = [], [], [], []
        for i, l in enumerate(m.layers):
            s_prev = states[i].detach() if detach_states else states[i]
            x, state, a, cache = l.forward(enc, x, s_prev)
            ta.append(a)
            tx.append(cache[4])
            ts.append(s_prev)
            tstate.append(state)
            states[i] = state.detach() if detach_states else state
        q["a"].append(ta)
        q["x_in"].append(tx)
        q["s_prev"].append(ts)
        q["state"].append(tstate)
        out = x @ P["Wd"].t() + P["bd"]
        loss = loss + (out.logsumexp(dim=1) - out[ar, ns[:, t]]).sum()
    return loss, q


def engine_gate_grad(m, q, dlds, gate_fn):
    """The engine's gate gradient, one layer at a time, through `gate_fn`.

    `gate_fn` is gpu_port.gate_gradient (the shipped one) or the numpy
    reference's twin; the loop around it is the same in both cases, so a
    difference between the two engines can only come from the formula.
    """
    out = []
    for i in range(len(m.layers)):
        gt = torch.zeros(B, m.dim, dtype=q["a"][0][0].dtype)
        g = torch.zeros(m.dim, dtype=gt.dtype)
        for t in range(T):
            contrib, gt = gate_fn(dlds[t][i], gt, q["a"][t][i],
                                  q["x_in"][t][i], q["s_prev"][t][i])
            g = g + contrib
        out.append(g)
    return out


def numpy_gate_grad(m, q, dlds):
    """The same loop, with numpy_port's copy of the formula."""
    out = []
    for i in range(len(m.layers)):
        gt = np.zeros((B, m.dim), np.float64)
        g = np.zeros(m.dim, np.float64)
        for t in range(T):
            contrib, gt = numpy_port.gate_gradient(
                dlds[t][i].detach().cpu().numpy().astype(np.float64),
                gt,
                q["a"][t][i].detach().cpu().numpy().astype(np.float64),
                q["x_in"][t][i].detach().cpu().numpy().astype(np.float64),
                q["s_prev"][t][i].detach().cpu().numpy().astype(np.float64))
            g = g + contrib
        out.append(g)
    return out


def old_form(m, q, dlds):
    """The engine's ORIGINAL gate line, reproduced for the record.

    Two bugs in one expression: the cross-time chain is missing (it is the
    LOCAL effect of a_t on s_t), and the axis is wrong -- `dec[None, :]` on
    a (B, dim) dec broadcasts to (1, B, dim), so `.sum(dim=0)` sums an axis
    of size 1 and the result keeps a BATCH axis the (dim,) gate parameter
    cannot have.
    """
    out = []
    for i in range(len(m.layers)):
        g = torch.zeros(B, m.dim, dtype=q["a"][0][0].dtype)
        for t in range(T):
            a = q["a"][t][i]
            aa = a[None, :]                       # the bug: (B, dim) -> (1,B,dim)
            g = g + (dlds[t][i] * q["s_prev"][t][i] * aa
                     * (1.0 - aa) * q["x_in"][t][i]).sum(dim=0)
        out.append(g)
    return out


def run_case(m, cs, ns, dtype):
    """One dtype's worth of the comparison. Returns a dict of gradients."""
    P, PL = leaves(m, dtype)
    with _Swapped(m, PL):
        # (a) the true gradient: torch autograd on the same recurrence
        loss, _ = forward(m, P, PL, cs, ns, detach_states=False)
        g_auto = torch.autograd.grad(
            loss, [PL[i]["gate"] for i in range(len(m.layers))])

        # the readout-only dL/ds_t: the states are DETACHED leaves
        loss_r, q = forward(m, P, PL, cs, ns, detach_states=True)
        flat = [s for t in range(T) for s in q["state"][t]]
        dlds_flat = torch.autograd.grad(loss_r, flat)
        dlds = [[dlds_flat[t * len(m.layers) + i] for i in range(len(m.layers))]
                for t in range(T)]

        # the engine's form, through the engine's own function
        g_engine = engine_gate_grad(m, q, dlds, gpu_port.gate_gradient)
        # the numpy reference's form
        g_numpy = numpy_gate_grad(m, q, dlds)
        # the original (broken) form
        g_old = old_form(m, q, dlds)

    return {
        "loss": float(loss.detach()),
        "auto": g_auto,
        "engine": g_engine,
        "numpy": g_numpy,
        "old": g_old,
    }


def finite_difference(m, cs, ns, eps=1e-6):
    """A central finite difference of the loss, in float64.

    float64 ON PURPOSE: the difference's own noise floor is the loss's
    rounding divided by the step, |L|*1e-7/(2*eps) in float32, which for
    this loss is LARGER than the difference being measured. The float64
    difference is the meaningful one, and the float32 engine is compared
    against it (the two losses differ by the float32 rounding of the
    arithmetic, i.e. ~1e-7 relative).
    """
    P, PL = leaves(m, torch.float64)
    g_fd = []
    with _Swapped(m, PL):
        for i in range(len(m.layers)):
            row = torch.zeros(m.dim, dtype=torch.float64)
            base = PL[i]["gate"].detach().clone()
            for d in range(m.dim):
                with torch.no_grad():
                    PL[i]["gate"].data.copy_(base)
                    PL[i]["gate"].data[d] = base[d] + eps
                    lp, _ = forward(m, P, PL, cs, ns, detach_states=False)
                    PL[i]["gate"].data.copy_(base)
                    PL[i]["gate"].data[d] = base[d] - eps
                    lm, _ = forward(m, P, PL, cs, ns, detach_states=False)
                    PL[i]["gate"].data.copy_(base)
                row[d] = (float(lp) - float(lm)) / (2.0 * eps)
            g_fd.append(row)
    return g_fd


def main():
    print("P3 GATE: the gate GRADIENT (formula on the recurrence -- the gated")
    print("step_batch still refuses; see the last section)")
    print(f"  dim={DIM} layers={LAYERS} B={B} T={T} seed={SEED}\n")

    m = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                          gated=True)
    rng = np.random.default_rng(SEED)
    cs = torch.from_numpy(rng.integers(1, 250, (B, T)).astype(np.int64))
    ns = torch.from_numpy(rng.integers(1, 250, (B, T)).astype(np.int64))

    # the finite difference of the loss (float64 -- see finite_difference)
    g_fd = finite_difference(m, cs, ns)

    for dtype, tol_auto, tol_fd, tag in (
            (torch.float64, 1e-9, 1e-5, "float64"),
            (torch.float32, 2e-4, 1e-4, "float32 (the engine's dtype)")):
        r = run_case(m, cs, ns, dtype)
        print(f"({tag})  loss {r['loss']:.6f}")
        for i in range(LAYERS):
            d_auto = rel_dev(r["engine"][i], r["auto"][i])
            d_fd = rel_dev(r["engine"][i], g_fd[i])
            d_np = max(float(np.abs(r["numpy"][i]
                                    - r["engine"][i].detach().cpu().numpy()).max()),
                       0.0)
            d_old = rel_dev(r["old"][i].sum(0), r["auto"][i])
            print(f"  layer{i}: engine vs autograd  max rel dev {d_auto:.3e}")
            print(f"          engine vs finite diff max rel dev {d_fd:.3e}")
            print(f"          engine vs numpy      max abs dev {d_np:.3e}")
            print(f"          ORIGINAL form vs autograd max rel dev {d_old:.3e}"
                  "  (the bug this gate exists for)")
            check(f"[{tag}] layer{i}: gate gradient == autograd",
                  d_auto < tol_auto, f"{d_auto:.3e} < {tol_auto:.0e}")
            check(f"[{tag}] layer{i}: gate gradient == central finite difference",
                  d_fd < tol_fd, f"{d_fd:.3e} < {tol_fd:.0e}")
            check(f"[{tag}] layer{i}: numpy_port's copy == gpu_port's, "
                  "element by element",
                  d_np < 1e-12 if dtype == torch.float64 else d_np < 1e-5,
                  f"max abs dev {d_np:.3e}")
        print()

    # ---- the ORIGINAL form's shape, which is the second half of the bug ---
    print("the ORIGINAL engine line, and why it was wrong twice")
    r = run_case(m, cs, ns, torch.float64)
    old_shape = tuple(r["old"][0].shape)
    check("the original line returns a (B, dim) result for a (dim,) parameter",
          old_shape == (B, DIM),
          f"shape {old_shape} -- `dec[None, :]` on a (B, dim) dec broadcasts "
          f"to (1, B, dim), so .sum(dim=0) summed an axis of size 1")
    d_old = max(rel_dev(r["old"][i].sum(0), r["auto"][i])
                for i in range(LAYERS))
    check("and it is numerically wrong even after the batch is summed",
          d_old > 1e-3, f"max rel dev {d_old:.3e} (the missing cross-time chain)")

    # ---- the trace works for a per-dim retention too ---------------------
    print("\nthe gate trace works for BOTH retention shapes")
    a_perdim = torch.full((DIM,), 0.6, dtype=torch.float64)
    a_persample = a_perdim[None, :].expand(B, DIM).contiguous()
    gt0 = torch.zeros(B, DIM, dtype=torch.float64)
    xin = torch.randn(B, DIM, dtype=torch.float64)
    sprev = torch.randn(B, DIM, dtype=torch.float64)
    g1 = gpu_port.gate_trace_update(gt0, a_perdim, xin, sprev)
    g2 = gpu_port.gate_trace_update(gt0, a_persample, xin, sprev)
    check("a per-dim (dim,) retention broadcasts over the batch",
          float((g1 - g2).abs().max()) == 0.0,
          f"max abs diff {float((g1 - g2).abs().max()):.3e}")
    g3 = numpy_port.gate_trace_update(
        np.zeros((B, DIM)), a_perdim.numpy(), xin.numpy(), sprev.numpy())
    check("numpy_port's trace == gpu_port's, element by element",
          float(np.abs(g3 - g1.numpy()).max()) == 0.0,
          f"max abs diff {float(np.abs(g3 - g1.numpy()).max()):.3e}")

    # ---- the arm runs, on the dense trace -------------------------------
    print("\nthe gated ARM runs (the credit machinery, not the gradient, was "
          "the blocker)")
    snap = (m.theta.detach().clone(), m.m.detach().clone(), m.t)
    m.new_states(B)
    m.step_batch(cs, ns, torch.zeros((B, T), dtype=torch.float32))
    check("gated step_batch runs", m.t == 1,
          f"trace_mode={m.trace_mode!r}")
    check("and it moved theta, moments and the step counter",
          not torch.equal(snap[0], m.theta.detach())
          and not torch.equal(snap[1], m.m.detach()) and m.t == snap[2] + 1)

    print("\n" + ("P3 GATE (gradient): PASS" if PASS
                  else "P3 GATE (gradient): FAIL"))
    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
