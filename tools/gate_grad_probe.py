"""tools/gate_grad_probe.py -- why the P3 gate gradient was wrong, in numbers.

The engine's P3 gate makes the retention factor input-dependent:

    a_t     = sigmoid(decay + gate * x_t)        (per SAMPLE: B x dim)
    s_t     = a_t * s_{t-1} + e_t
    h_t     = LayerNorm(s_t) -> W -> z_t         (the layer's read-out)

and the engine formed dL/dgate as a plain sum over the window:

    dL/dgate = sum_t  dL/ds_t * s_{t-1} * a_t(1-a_t) * x_t

which is the gradient of the effect of a_t on s_t ONLY. But a_t does not stop
at s_t: a_t multiplies s_{t-1} INTO s_t, and s_t then propagates, multiplied
by every later a. So the true derivative carries a cross-time chain:

    ds_k/da_t = (prod_{j=t+1..k} a_j) * s_{t-1}          (k > t)
    dL/dgate_t = a_t(1-a_t) x_t s_{t-1} * sum_{k>=t} (prod_{j=t+1..k} a_j) dL/ds_k

and the chain factor is exactly what the DECAY gradient already carries in its
`decaytrace` recurrence. The same structure works for the gate: define the
gate trace

    GT_t = a_t * GT_{t-1} + a_t(1-a_t) * x_t * s_{t-1}

then the older source j is multiplied by prod_{k=j+1..t} a_k, and

    dL/dgate = sum_t dL/ds_t * GT_t

is the true gradient.

WHICH dL/ds_t?
--------------
This is the part that is easy to get wrong, and the part this probe exists to
settle. There are two different quantities called "dL/ds_t":

  * the READOUT-ONLY one: the gradient of the loss with respect to the state
    when every state is a DETACHED leaf. The engine produces exactly this
    (its per-byte backward stops at the state, because the states are
    stop_gradient), and it is the one the trace reconstruction is built for;
  * the TOTAL one: what torch autograd gives for the state, which INCLUDES
    the chain through the recurrence (dL/ds_t picks up the later states).

The two gate formulas each become exact with a different dL/ds_t, and mixing
them is how you get a number that looks plausible and is not:

    plain form  + readout-only dlds   -> WRONG   (the local effect only)
    gatetrace   + readout-only dlds   -> CORRECT (the trace carries the chain)
    plain form  + TOTAL dlds          -> CORRECT (but the engine has no total)
    gatetrace   + TOTAL dlds          -> WRONG   (double counts the chain)

The truth table below measures all four against torch autograd on the SAME
recurrence. It is standalone on purpose -- it does not import the engine, so it
cannot inherit the engine's bug.

Run:  python tools/gate_grad_probe.py
"""
import argparse

import torch


def _layer_norm(s, g, b, eps=1e-5):
    mu = s.mean(dim=-1, keepdim=True)
    xc = s - mu
    var = (xc * xc).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(var + eps)
    xh = xc * inv
    return xh * g + b


def run(dim=6, batch=3, window=6, seed=0, a_near_one=False):
    """One run of the recurrence.

    Returns a dict with the autograd gate gradient, BOTH dL/ds_t flavours and
    the recurrence's own quantities (a_t, s_{t-1}), so that the four
    combinations below can be formed from the same data.
    """
    gen = torch.Generator().manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    # a_near_one makes the retention long-lived, which is the regime the
    # log-spaced init creates and the regime where the chain factor is large.
    decay = torch.zeros(dim, dtype=torch.float64)
    if a_near_one:
        decay = decay + 2.5          # sigmoid(2.5) = 0.924 -> half-life ~9 B
    gate = rn(dim) * 0.5
    W = rn(dim, dim) * 0.3
    g_ = torch.ones(dim, dtype=torch.float64)
    b_ = torch.zeros(dim, dtype=torch.float64)

    xs = [rn(batch, dim) for _ in range(window)]
    # `es` carries the graph for the TOTAL-dL/ds_t run: without a tensor that
    # requires grad, autograd has no root to differentiate at and the "total"
    # state gradient cannot even be asked for.
    es = [rn(batch, dim).requires_grad_(True) for _ in range(window)]
    tgts = [rn(batch, dim) for _ in range(window)]

    def forward(gate_leaf, detach_states):
        """The recurrence.

        gate_leaf is used as given. detach_states=True turns every state into
        a fresh leaf, which CUTS the cross-time chain -- that is the engine's
        view (the states are stop_gradient there). detach_states=False keeps
        the graph, and the gradient of the loss wrt the state INTERMEDIATES is
        then the TOTAL dL/ds_t (autograd's own answer).
        """
        prev = torch.zeros(batch, dim, dtype=torch.float64)
        loss = torch.zeros((), dtype=torch.float64)
        leaves, keep = [], []
        for t in range(window):
            a = torch.sigmoid(decay + gate_leaf * xs[t])
            s = a * prev + es[t]
            if detach_states:
                s = s.detach().requires_grad_(True)
                leaves.append(s)
            else:
                keep.append(s)
            h = _layer_norm(s, g_, b_)
            z = h @ W.t()
            loss = loss + (z * tgts[t]).sum()
            prev = s
        return loss, leaves, keep

    # ---- (0) autograd: the true dL/dgate --------------------------------
    gate_ag = gate.clone().requires_grad_(True)
    loss, _, _ = forward(gate_ag, detach_states=False)
    grad_autograd = torch.autograd.grad(loss, gate_ag)[0].clone()

    # ---- the READOUT-ONLY dL/ds_t: the states are DETACHED leaves ---------
    loss_r, leaves, _ = forward(gate.detach(), detach_states=True)
    dlds_readout = torch.autograd.grad(loss_r, leaves)

    # ---- the TOTAL dL/ds_t: autograd through the whole recurrence --------
    loss_t, _, keep = forward(gate.detach(), detach_states=False)
    dlds_total = torch.autograd.grad(loss_t, keep)

    # ---- the recurrence's own quantities: a_t and s_{t-1} ----------------
    a_seq, sprev_seq = [], []
    prev = torch.zeros(batch, dim, dtype=torch.float64)
    for t in range(window):
        a = torch.sigmoid(decay + gate * xs[t])
        a_seq.append(a)
        sprev_seq.append(prev)
        prev = a * prev + es[t].detach()

    def plain(dlds):
        """The engine's ORIGINAL form: the LOCAL effect of a_t on s_t."""
        out = torch.zeros(dim, dtype=torch.float64)
        for t in range(window):
            a = a_seq[t]
            out = out + (dlds[t] * sprev_seq[t] * a * (1.0 - a) * xs[t]).sum(0)
        return out

    def trace(dlds):
        """The FIXED form: the local effect carried forward by the chain."""
        out = torch.zeros(dim, dtype=torch.float64)
        gt = torch.zeros(batch, dim, dtype=torch.float64)
        for t in range(window):
            a = a_seq[t]
            gt = a * gt + a * (1.0 - a) * xs[t] * sprev_seq[t]
            out = out + (dlds[t] * gt).sum(0)
        return out

    return {
        "autograd": grad_autograd,
        "dlds_readout": dlds_readout,
        "dlds_total": dlds_total,
        "plain_readout": plain(dlds_readout),
        "trace_readout": trace(dlds_readout),
        "plain_total": plain(dlds_total),
        "trace_total": trace(dlds_total),
    }


def rel_dev(got, ref):
    """Max |got - ref| against the scale of ref.

    The SIGN of ref matters: an earlier version of this probe compared
    `got` against `|ref|` (it clamped the reference in place and then
    subtracted the clamped tensor), so a gradient that matched the
    reference elementwise was reported as a ~1.0 deviation. The reference
    is kept signed here and only the DENOMINATOR uses its magnitude.
    """
    scale = float(ref.abs().max())
    scale = scale if scale > 1e-12 else 1e-12
    return float((got - ref).abs().max() / scale)


def truth_table(args, label, **kw):
    """The four combinations, measured against torch autograd."""
    r = run(dim=args.dim, batch=args.batch, window=args.window,
            seed=args.seed, **kw)
    rows = [
        ("plain  + readout-only dlds", "engine's ORIGINAL form", r["plain_readout"]),
        ("trace  + readout-only dlds", "the FIX", r["trace_readout"]),
        ("plain  + TOTAL dlds", "exact, but the engine has no total", r["plain_total"]),
        ("trace  + TOTAL dlds", "double counts the chain", r["trace_total"]),
    ]
    print(f"  {label}")
    print(f"    {'combination':<28} | {'rel dev vs autograd':>20} | verdict")
    print("    " + "-" * 72)
    devs = []
    for name, note, got in rows:
        d = rel_dev(got, r["autograd"])
        devs.append(d)
        v = "CORRECT" if d < 1e-9 else "WRONG"
        print(f"    {name:<28} | {d:>18.4f}x | {v}  ({note})")
    return devs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print("THE P3 GATE GRADIENT: four combinations, measured against autograd")
    print("=" * 78)
    print(f"dim={args.dim} B={args.batch} window={args.window} "
          f"seed={args.seed} float64\n")

    devs = []
    for label, near in (("short-lived a~0.5", False),
                        ("long-lived a~0.92", True)):
        devs += truth_table(args, label, a_near_one=near)
        print()

    print("READ IT AS:")
    print("  * the readout-only dL/ds_t is what the ENGINE has (its per-byte")
    print("    backward stops at the state, because the states are")
    print("    stop_gradient). With it, the plain sum is the LOCAL effect of")
    print("    a_t on s_t only, and the gate trace is the exact gradient:")
    print("    the trace carries the cross-time chain")
    print("        ds_k/da_t = (prod_{j=t+1..k} a_j) * s_{t-1}")
    print("    the same chain the decay's decaytrace already carries.")
    print("  * the TOTAL dL/ds_t is what torch autograd gives for the state.")
    print("    With it the plain sum is already exact -- and the gate trace")
    print("    on top of it DOUBLE COUNTS the chain, because the trace's chain")
    print("    and the total gradient's chain are the same chain.")
    print("  * so the engine cannot use the total (it does not have it) and")
    print("    must use the trace: trace + readout-only is the only pair")
    print("    that is both computable in the engine and exact.")
    print()
    print("The chain factor grows as the retention approaches 1 -- which is")
    print("exactly the regime the log-spaced (P4) init creates and exactly the")
    print("regime where the engine's original form is most wrong. A gate that")
    print("is wrong by a factor that scales with the memory it is supposed to")
    print("create is worse than no gate at all.")
    print()

    # The verdict is about the pair the ENGINE can actually form: the gate
    # trace on the readout-only gradient. The other two rows are the
    # cross-checks that pin the argument down (the plain form is wrong on the
    # engine's own gradient, and the trace on the total gradient double
    # counts), and the plain form on the total gradient is the independent
    # confirmation that the trace's chain is the right one.
    ok_fix = truth_table(args, "engine's own gradient (the pair the engine can form)",
                          a_near_one=False)
    ok = (ok_fix[1] < 1e-9 and ok_fix[0] > 1e-3 and ok_fix[2] < 1e-9
          and ok_fix[3] > 1e-3)
    print(f"VERDICT: {'the gate trace on the readout-only gradient is exact' if ok else 'STILL WRONG'}"
          f" (rel dev {ok_fix[1]:.2e}; the original form is off by "
          f"{ok_fix[0]:.2f}x)")


if __name__ == "__main__":
    main()
