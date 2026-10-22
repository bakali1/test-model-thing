#!/usr/bin/env python3
"""check_gated_credit.py -- the P3 gate's CREDIT, pinned against autograd.

WHY THIS FILE EXISTS
--------------------
The P3 gated arm used to REFUSE at `step_batch`. The refusal's own words
listed four blockers, and the design's analysis
(docs/design/P3-GATE-ANALYSIS.md) showed that three of them were BROADCAST
changes and the fourth -- the ring's truncation bound -- is dissolved by the
DENSE trace, which has no `ring_k`, no hot set, no mask, no rebase and no
truncation, and is therefore exact for ANY per-sample retention.

So the arm now RUNS, on the dense trace. But the refusal's reason was not
"the code is missing", it was "a plausible-looking gradient that is quietly
wrong". `tests/check_gate_grad.py` pins the gate's RECURRENCE gradient -- the
traced form, against autograd and against a central finite difference -- and
it never touches the credit at all. So the credit had NEVER been pinned.

This file is the gate that was missing. It pins, in order:

  (a) gate = 0 is BIT-IDENTICAL to the incumbent. The forward, the loss and
      the states, and (against the SAME trace mode) every gradient slice
      except the gate's own.
  (b) the gated `step_batch`'s FULL gradient -- dembed, ddecay, dgate --
      against TORCH AUTOGRAD on the same window, with the deviation REPORTED
      (and compared against the float32 rounding floor of the same
      computation, because a PASS without a number is not a result).
  (c) the dense credit's exactness under a PER-SAMPLE, time-varying
      retention: the design's section-2 measurement, now a permanent gate
      (ring vs dense vs the explicitly enumerated survival products), plus
      the engine's own dense credit under a per-sample dec.
  (d) the numpy reference's gated arm against the torch engine's (the
      reference grew a dense path for exactly this).
  (e) the default engine is untouched: no `layer{i}.gate` key, and the
      ungated arm still steps.

THE CONVENTION THAT MATTERS
---------------------------
The engine's gradients are the gradient of the batch SUM; the reported loss
is the batch MEAN (see gpu_port.step_batch: "NO 1/B HERE, and that is not an
oversight"). So the autograd reference below sums the loss over the batch and
over the window, and the reported loss is compared as sum/B. Getting this
backwards is a factor-of-B error that would look like a plausible gradient.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gpu_port
import numpy_port

# DEVICE CONVENTION (like the other gates here): the claims are
# device-independent, and the inputs are built on the CPU on purpose. The
# device can be overridden for a device-specific measurement with
# TMT_GATE_DEV=cuda -- the claims are the same, but the numbers (the float32
# rounding floor in particular) are the DEVICE's.
DEV = os.environ.get("TMT_GATE_DEV", "cpu")

DIM = 8
LAYERS = 2
B = 3
T = 4
SEED = 0

# The tolerance for the gradient comparison. It is NOT a magic number: it is
# the float32 rounding floor of the same computation, measured below and
# reported. The engine accumulates the credit over the window in float32 and
# sums it with an einsum, while autograd sums the same terms in a different
# order; the two are mathematically identical and numerically different at the
# float32 epsilon. The floor is measured (|ref32 - ref64|) and the check is
# against max(TOL, 3 * floor), so the tolerance can only ever be LOOSENED by
# the measurement, never by wishful thinking.
TOL = 1e-4

PASS = True


def check(name, ok, detail=""):
    global PASS
    if not ok:
        PASS = False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))


def rel_dev(a, b):
    """max|a - b| against the scale of b (the reference)."""
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    scale = float(np.abs(b).max())
    scale = scale if scale > 1e-30 else 1e-30
    return float(np.abs(a - b).max() / scale)


def batch(seed, B, T):
    rng = np.random.default_rng(seed)
    cs = torch.from_numpy(rng.integers(1, 250, (B, T)).astype(np.int64))
    ns = torch.from_numpy(rng.integers(1, 250, (B, T)).astype(np.int64))
    ends = torch.zeros((B, T), dtype=torch.float32)
    return cs, ns, ends


def grad_slices(m):
    """The engine's flat gradient, sliced by parameter name."""
    out = {}
    off = 0
    g = m.grads.detach().cpu().numpy().astype(np.float64)
    for name, shape in m.shapes:
        size = int(np.prod(shape))
        out[name] = g[off:off + size].reshape(shape)
        off += size
    return out


# ---------------------------------------------------------------------------
# the autograd reference, over the ENGINE's own layer code
# ---------------------------------------------------------------------------
class _Swapped:
    """Point the engine's layers at leaf tensors for the duration of a run.

    TorchLayer.forward reads self.decay/self.gate/self.g/self.b/self.W, so
    swapping them in is what makes this drive the engine's OWN layer code --
    its layer norm, its silu, its state update -- instead of a
    re-implementation of it that could drift from the shipped one.
    """

    def __init__(self, m, pl):
        self.m, self.pl, self.saved = m, pl, []

    def __enter__(self):
        for l, p in zip(self.m.layers, self.pl):
            self.saved.append((l, l.decay, l.gate, l.g, l.b, l.W))
            l.decay, l.gate, l.g, l.b, l.W = (p["decay"], p["gate"], p["g"],
                                               p["b"], p["W"])
        return self

    def __exit__(self, *exc):
        for l, dec, gate, g, b, W in self.saved:
            l.decay, l.gate, l.g, l.b, l.W = dec, gate, g, b, W
        self.saved = []


def leaves(m, dtype):
    """The model's parameters as leaves of the given dtype."""
    P = {"embed": m.embed.detach().clone().to(dtype).requires_grad_(True),
         "Wd": m.Wd.detach().clone().to(dtype).requires_grad_(True),
         "bd": m.bd.detach().clone().to(dtype).requires_grad_(True),
         "Ws": m.Ws.detach().clone().to(dtype).requires_grad_(True),
         "bs": m.bs.detach().clone().to(dtype).requires_grad_(True)}
    PL = []
    for l in m.layers:
        PL.append({
            "decay": l.decay.detach().clone().to(dtype).requires_grad_(True),
            "gate": (l.gate.detach().clone() if l.gate is not None
                     else torch.zeros(m.dim, dtype=dtype)).to(dtype)
                     .requires_grad_(True),
            "g": l.g.detach().clone().to(dtype).requires_grad_(True),
            "b": l.b.detach().clone().to(dtype).requires_grad_(True),
            "W": l.W.detach().clone().to(dtype).requires_grad_(True),
        })
    return P, PL


def ref_grads(m, P, PL, cs, ns, ends):
    """The window's gradient, by autograd, over the engine's own layers.

    The FULL chain (the states are NOT detached): the engine's credit and its
    two traces RECONSTRUCT the cross-time chain the engine's per-byte
    backward does not have, so the reference must carry the chain and the
    engine must be shown to reproduce it. (Feeding the trace the engine's
    readout-only state gradient is what makes the reconstruction exact; the
    four-way truth table is in tools/gate_grad_probe.py.)

    The loss is the SUM over the batch and over the window -- the engine's
    gradient convention. The reported loss is that sum / B.
    """
    dtype = P["embed"].dtype
    B_, T_ = cs.shape
    ar = torch.arange(B_)
    states = [torch.zeros(B_, m.dim, dtype=dtype) for _ in m.layers]
    loss = torch.zeros((), dtype=dtype)
    for t in range(T_):
        enc = P["embed"][cs[:, t]]
        x = enc
        for i, l in enumerate(m.layers):
            x, state, _a, _cache = l.forward(enc, x, states[i])
            states[i] = state
        out = x @ P["Wd"].t() + P["bd"]
        stop = torch.sigmoid(x @ P["Ws"] + P["bs"][0])
        # the engine's own loss terms, term for term. `tgt` is DETACHED in the
        # engine (the target byte's embedding row gets no gradient), so it is
        # detached here too -- otherwise the reference would carry a gradient
        # path the engine does not have.
        tgt = P["embed"][ns[:, t]].detach()
        xv = x.var(dim=1, unbiased=False) + 1e-4
        var_active = (1.0 - torch.sqrt(xv)) > 0
        ce = out.logsumexp(dim=1) - out[ar, ns[:, t]]
        var_t = torch.where(var_active, 1.0 - torch.sqrt(xv),
                            torch.zeros_like(xv)).sum()
        mse_t = ((x - tgt) ** 2).sum() / m.dim
        end_t = ((stop - ends[:, t]) ** 2).sum()
        ce_t = ce.sum()
        loss = loss + (var_t + mse_t + ce_t + end_t)
    wanted = [P["embed"]]
    for i in range(len(m.layers)):
        wanted += [PL[i]["decay"], PL[i]["gate"]]
    g = torch.autograd.grad(loss, wanted)
    names = ["embed"]
    for i in range(len(m.layers)):
        names += [f"layer{i}.decay", f"layer{i}.gate"]
    return float(loss.detach()), dict(zip(names, [x.detach() for x in g]))


# ---------------------------------------------------------------------------
# (c) the design's section-2 measurement, as a permanent gate
# ---------------------------------------------------------------------------
def dense_vs_ring_under_per_sample_dec(B_=3, K=16, D=5, T_=12, seed=0):
    """Ring credit vs dense credit vs the EXPLICITLY ENUMERATED survival
    products, under a per-SAMPLE, time-varying retention a[b, t, d].

    This is the measurement that made the recommendation possible: with no
    truncation the ring is bit-exact too, and the ONLY disagreement is the
    ring's lag-K truncation -- which is exactly the bound a two-sided gate
    destroys. The dense trace has no such bound, so it is exact in every
    regime.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros((B_, K, D), np.float32)
    ring_b = np.zeros((B_, K), np.int64)
    E = np.zeros((B_, 256, D), np.float32)
    A = rng.uniform(0.3, 0.99, (B_, T_, D)).astype(np.float32)
    cs = rng.integers(0, 256, (B_, T_))
    dlds = rng.standard_normal((B_, D)).astype(np.float32)
    dr = np.zeros((256, D))
    dd = np.zeros((256, D))
    for t in range(T_):
        a = A[:, t, :]
        w = w * a[:, None, :]
        vals = w * dlds[:, None, :]
        np.add.at(dr, ring_b.ravel(), vals.reshape(B_ * K, D))
        p = t % K
        w[:, p, :] = 1.0
        ring_b[:, p] = cs[:, t]
        E = E * a[:, None, :]
        dd += np.einsum("bvd,bd->vd", E, dlds)
        E[np.arange(B_), cs[:, t], :] += 1.0
    # the ground truth: the survival product, enumerated byte by byte
    dt = np.zeros((256, D))
    for b in range(B_):
        for t in range(T_):
            for s in range(t):
                prod = np.ones(D, np.float32)
                for j in range(s + 1, t + 1):
                    prod = prod * A[b, j, :]
                dt[cs[b, s]] += prod * dlds[b]
    rel = lambda x, y: float(np.abs(x - y).max() / (np.abs(y).max() + 1e-12))
    return {"ring_vs_truth": rel(dr, dt),
            "dense_vs_truth": rel(dd, dt),
            "ring_vs_dense": rel(dr, dd)}


def main():
    print("P3 GATE: the CREDIT (the gated step_batch's full gradient vs "
          "autograd)")
    print(f"  dim={DIM} layers={LAYERS} B={B} T={T} seed={SEED}\n")

    cs, ns, ends = batch(SEED, B, T)

    # ---------------------------------------------------------------- (a)
    print("(a) gate = 0 is BIT-IDENTICAL to the incumbent")
    m_g = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                            gated=True)
    m_d = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                            trace_mode="dense")
    m_r = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                            trace_mode="ring")
    check("the gated arm runs on the dense trace", m_g.trace_mode == "dense",
          f"trace_mode={m_g.trace_mode!r}")
    for m in (m_g, m_d, m_r):
        m.new_states(B)
    l_g = m_g.step_batch(cs, ns, ends)
    l_d = m_d.step_batch(cs, ns, ends)
    l_r = m_r.step_batch(cs, ns, ends)
    check("the loss is bit-identical to the ungated engine",
          l_g == l_d == l_r, f"{l_g!r} / {l_d!r} / {l_r!r}")
    check("the CE is bit-identical too", m_g.last_ce == m_d.last_ce)
    same_state = all(
        torch.equal(m_g.layers[i].states, m_d.layers[i].states)
        and torch.equal(m_g.layers[i].states, m_r.layers[i].states)
        for i in range(LAYERS))
    check("the recurrent states are bit-identical", same_state)
    gg, gd = grad_slices(m_g), grad_slices(m_d)
    names = [n for n, _ in m_g.shapes if not n.endswith(".gate")]
    worst, worst_name = 0.0, ""
    for n in names:
        d = rel_dev(gg[n], gd[n])
        if d > worst:
            worst, worst_name = d, n
    check("every non-gate gradient slice is bit-identical to the ungated "
          "engine (same trace mode)",
          all(np.array_equal(gg[n], gd[n]) for n in names),
          f"max rel dev {worst:.3e} ({worst_name})")
    # the gate's own gradient is NOT zero at gate=0 -- by design: the gate's
    # EFFECT is zero (a == sigmoid(decay) bit-for-bit) but its DERIVATIVE is
    # not (da/dgate = a(1-a) * x). That is what makes the arm able to move.
    check("the gate's gradient is non-zero at gate=0 (the derivative, not the "
          "effect)", float(np.abs(gg["layer0.gate"]).max()) > 0.0,
          f"max |dgate| = {float(np.abs(gg['layer0.gate']).max()):.3e}")
    gr = grad_slices(m_r)
    d_ring = rel_dev(gg["embed"], gr["embed"])
    check("and the credit agrees with the ring incumbent to the ring's own "
          "truncation", d_ring <= 0.01,
          f"dembed rel dev {d_ring:.3e} (the ring's stated tolerance 1e-2; "
          f"the dense trace is the exact one)")

    # ---------------------------------------------------------------- (b)
    print("\n(b) the gated step_batch's FULL gradient vs torch autograd")
    m = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                          gated=True)
    m.new_states(B)
    P64, PL64 = leaves(m, torch.float64)
    P32, PL32 = leaves(m, torch.float32)
    with _Swapped(m, PL64):
        loss64, g64 = ref_grads(m, P64, PL64, cs, ns, ends)
    with _Swapped(m, PL32):
        loss32, g32 = ref_grads(m, P32, PL32, cs, ns, ends)
    m.step_batch(cs, ns, ends)
    ge = grad_slices(m)
    check("the engine's reported loss is the reference sum / B",
          abs(float(m.last_loss_tensor) - loss64 / B) <= 1e-4 * abs(loss64 / B),
          f"engine {float(m.last_loss_tensor):.6f} vs reference "
          f"{loss64 / B:.6f}")
    for n in ["embed", "layer0.decay", "layer0.gate", "layer1.decay",
              "layer1.gate"]:
        r64 = g64[n].detach().cpu().numpy().astype(np.float64)
        r32 = g32[n].detach().cpu().numpy().astype(np.float64)
        e = ge[n]
        d_like = rel_dev(e, r32)      # like-for-like: two fp32 orders
        d_exact = rel_dev(e, r64)     # against the float64 truth
        floor = rel_dev(r32, r64)     # the fp32 rounding floor itself
        lim = max(TOL, 3.0 * floor)
        print(f"  {n:<14} vs autograd: fp32-vs-fp32 {d_like:.3e}   "
              f"vs float64 {d_exact:.3e}   (fp32 floor {floor:.3e}, "
              f"tol {lim:.1e})")
        check(f"[{n}] the engine's gradient matches autograd", d_like <= lim,
              f"{d_like:.3e} <= {lim:.1e}")
    # The deviation is reported, not just passed: print the summary line the
    # repo's rule asks for ("a CE without its SE is not a result").
    devs = [rel_dev(ge[n], g32[n].detach().cpu().numpy().astype(np.float64))
            for n in ["embed", "layer0.decay", "layer0.gate", "layer1.decay",
                      "layer1.gate"]]
    print(f"  -> max |engine - autograd| over the pinned slices: "
          f"{max(devs):.3e} (relative)")

    # ---------------------------------------------------------------- (c)
    print("\n(c) the dense credit is exact under a PER-SAMPLE retention")
    r = dense_vs_ring_under_per_sample_dec(T_=12, K=16)
    check("with no truncation, ring == dense == the enumerated truth",
          r["ring_vs_truth"] == 0.0 and r["dense_vs_truth"] == 0.0,
          f"ring {r['ring_vs_truth']:.3e}, dense {r['dense_vs_truth']:.3e}")
    r2 = dense_vs_ring_under_per_sample_dec(T_=20, K=16)
    check("with truncation, the RING drops credit and the DENSE trace does "
          "not", r2["dense_vs_truth"] < 1e-6 and r2["ring_vs_dense"] > 1e-4,
          f"dense vs truth {r2['dense_vs_truth']:.3e}, ring vs dense "
          f"{r2['ring_vs_dense']:.3e} -- this is the bound a two-sided gate "
          f"destroys, and the reason the gated arm uses the dense trace")
    # and the ENGINE's own dense path under a per-sample dec: the gate makes
    # a = sigmoid(decay + gate * x) a (B, dim) tensor, so the engine's decay
    # broadcast is the per-sample one. Check it is really per-sample: with a
    # non-zero gate the per-stream retentions must differ.
    mg2 = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                            gated=True)
    with torch.no_grad():
        for l in mg2.layers:
            l.gate.add_(0.7)
    mg2.new_states(B)
    with torch.no_grad():
        enc = mg2.embed[cs[:, 0]]
        x = enc
        a = None
        for l in mg2.layers:
            x, _s, a, _c = l.forward(enc, x, l.states)
    check("the gate makes the retention genuinely per-sample",
          a.shape == (B, DIM) and float(a.std(dim=0).max()) > 0.0,
          f"a shape {tuple(a.shape)}, per-sample spread "
          f"{float(a.std(dim=0).max()):.3e}")

    # ---------------------------------------------------------------- (d)
    print("\n(d) the numpy reference's gated arm vs the torch engine's")
    numpy_port.rng = np.random.default_rng(SEED)
    np_model = numpy_port.Model(dim=DIM, layers=LAYERS, lr=5e-4, gated=True)
    np_model.new_states(B)
    gp = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, lr=5e-4,
                           gated=True)
    gp.from_numpy(np_model)
    c1 = cs[:, 0].tolist()
    n1 = ns[:, 0].tolist()
    e1 = ends[:, 0].tolist()
    l_np = np_model.step_batch(c1, n1, e1)
    l_gp = gp.step_batch(cs[:, :1], ns[:, :1], ends[:, :1])
    d_loss = abs(l_np - l_gp) / max(1.0, abs(l_np))
    check("the gated losses agree", d_loss < 1e-3,
          f"|{l_np:.6f} - {l_gp:.6f}| / max(1,|L|) = {d_loss:.3e}")
    gn = {k: v for k, v in np_model.gviews.items()}
    gt = {k: v.detach().cpu().numpy() for k, v in gp.gviews.items()}
    worst, worst_name = 0.0, ""
    for k in gn:
        if k not in gt:
            continue
        r = rel_dev(gn[k], gt[k])
        if r > worst:
            worst, worst_name = r, k
    check("the gated gradients agree (parity tolerances: 2e-2 per "
          "parameter)", worst < 2e-2,
          f"max rel dev {worst:.3e} ({worst_name})")

    # ---------------------------------------------------------------- (e)
    print("\n(e) the default engine is untouched")
    d = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED)
    check("trace_mode still defaults to 'ring'", d.trace_mode == "ring")
    check("the default engine has no gate key",
          not any(n.endswith(".gate") for n, _ in d.shapes),
          "its absence is what keeps check_parity's theta layout valid")
    d.new_states(B)
    d.step_batch(cs, ns, ends)
    check("the default engine still steps", d.t == 1)

    print("\n" + ("P3 GATE (credit): PASS" if PASS else "P3 GATE (credit): FAIL"))
    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
