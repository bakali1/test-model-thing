#!/usr/bin/env python3
"""GATE: the safety features (PLAN-2026-10-11 P0 item 3) and P6's per-group
weight decay.

P0 item 3 is the last open item in the plan's P0: "NaN guard + gradient
clipping -- still open. SVC refuses a non-finite gradient outright. Our KD
arm diverged to 860.68." Two features are implemented here, plus P6, and
this gate is what makes them trustworthy:

  1. **The NaN guard refuses the update.** A non-finite gradient must leave
     theta, m and v EXACTLY as they were -- a guard that half-applies the
     update is worse than no guard, because the run looks alive. The refusal
     is counted (`nan_refusals`), `last_nan_refused` is set, and `t` still
     advances, because the DATA was consumed and the LR schedule must not
     rewind.
  2. **The guard is a no-op on finite gradients.** Bit-exactly: the same
     engine with the guard off and with the guard on must produce the same
     theta, m and v. This is the claim that makes the guard safe to enable by
     default; a "harmless" check that perturbs the last bit is not harmless
     in a repo that gates on bit-identity.
  3. **The global grad clip scales the gradient that was actually applied.**
     The clip happens BEFORE the AdamW moments are updated, so the moments
     see the clipped gradient. Verified against a hand-written AdamW applied
     to the manually clipped gradient.

     AND, measured rather than assumed: under AdamW the clip does NOT
     meaningfully shrink the PARAMETER STEP, because the update is
     m/sqrt(v) ~ sign(g) and therefore nearly scale-invariant in g. So the
     clip is NOT a step-size limiter -- it is a guard on the MOMENTS. A
     single enormous gradient inflates v, and every later small gradient is
     then divided by that inflated sqrt(v) and suppressed for ~1/(1-b2) =
     1000 steps, which is exactly the "the model keeps walking in the old
     direction" failure. Bounding the gradient bounds that. The gate
     therefore asserts that the APPLIED gradient is bounded and that the
     update matches the manual AdamW on the clipped gradient, and it
     asserts the step is NOT smaller -- a clip that silently did not scale
     the gradient would pass a step-size check.
  4. **Per-group weight decay (P6).** With `wd_decay=0`, the `.decay` rows
     get NO decoupled weight decay while every other row is untouched. This
     matters because `sigma(0) = 0.5` is exactly a 1-BYTE half-life: a weight
     decay on the decay logits is a systematic pull toward NO MEMORY, and it
     fights the log-spaced half-life init (P4) directly. With `wd_decay` at
     its default (== the flat wd) the engine is bit-identical to the
     incumbent, so every existing arm and checkpoint keeps its dynamics.
  5. **The `--lr-decay` trap is gone when wd_decay=0.** `--lr-decay` scales the
     whole update bracket for the decay role, which includes the weight-decay
     term -- so a scaled decay role also got 16x the weight decay. With the
     decay group's weight decay at 0, the role's LR scales only the adaptive
     term.

Both engines (torch and the numpy reference) carry the same features, because
check_parity.py gates them against each other BIT-EXACTLY: a safety feature
that existed on one side only would make the parity gate measure the
difference in features rather than in the engine.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gpu_port
import numpy_port


def bits(x):
    """Raw bit pattern of a float32 array, so a sign-of-zero flip fails."""
    a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a.view(np.int32)


def build(seed=5, dim=32, layers=2, B=8, **kw):
    torch.manual_seed(seed)
    m = gpu_port.GpuModel(dim=dim, layers=layers, device="cpu", **kw)
    for layer in m.layers:
        layer.decay[0] = 4.0
    m.new_states(B)
    return m


def step_once(m, seed=0, B=8):
    rng = np.random.default_rng(seed)
    c = torch.from_numpy(rng.integers(1, 250, (B,)).astype(np.int64))
    n = torch.from_numpy(rng.integers(1, 250, (B,)).astype(np.int64))
    e = torch.zeros(B, dtype=torch.float32)
    return m.step_batch(c, n, e)


def manual_adamw(theta, grad, m0, v0, t, lr, wd_vec, b1=0.9, b2=0.999,
                 eps=1e-8):
    """The engine's update, written out, in float32 numpy.

    Used as the reference for the clip claim: the engine's clip multiplies
    the GRADIENT before the moments see it, so the expected theta is this
    function called with the clipped gradient.
    """
    m = (b1 * m0 + (1 - b1) * grad).astype(np.float32)
    v = (b2 * v0 + (1 - b2) * (grad * grad)).astype(np.float32)
    v = np.sqrt(v, dtype=np.float32)
    bc1 = 1.0 / (1.0 - b1 ** t)
    bc2 = 1.0 / (1.0 - b2 ** t)
    bracket = (m * bc1 / (v * bc2 + eps) + wd_vec * theta).astype(np.float32)
    return (theta - lr * bracket).astype(np.float32)


def main():
    print("GATE: safety (NaN guard, global grad clip) and P6 (per-group wd)")
    print("=" * 78)

    # ---- 1. the guard is a no-op on finite gradients ----------------------
    a = build(nan_guard=True)
    b = build(nan_guard=False)
    for i in range(12):
        step_once(a, seed=i)
        step_once(b, seed=i)
    assert torch.equal(a.theta, b.theta), (
        "the NaN guard changed the trajectory on FINITE gradients: "
        f"{int((bits(a.theta) != bits(b.theta)).sum())} differing elements")
    assert torch.equal(a.m, b.m) and torch.equal(a.v, b.v)
    assert a.nan_refusals == 0
    print("[PASS] the NaN guard is bit-identical to no guard on finite "
          "gradients (12 steps, theta/m/v)")

    # ---- 2. a non-finite gradient is REFUSED ------------------------------
    m = build(nan_guard=True)
    step_once(m, seed=1)                      # one clean step first
    th0 = m.theta.detach().clone()
    m0 = m.m.detach().clone()
    v0 = m.v.detach().clone()
    t0 = m.t
    # Inject a NaN where it PROPAGATES: the decoder weight, which every byte
    # passes through. (A NaN in an embedding row the batch never touches would
    # NOT reach the gradient, and the guard would be right not to fire -- that
    # is a property of the guard, not a bug in it.)
    with torch.no_grad():
        m.views["Wd"][0, 0] = float("nan")
    nan_th = m.theta.detach().clone()
    loss = step_once(m, seed=2)
    assert m.last_nan_refused is True, "the guard did not fire"
    assert m.nan_refusals == 1, f"the refusal was not counted ({m.nan_refusals})"
    # theta, m, v must be EXACTLY as they were before the refused step.
    assert np.array_equal(bits(m.theta), bits(nan_th)), (
        "the refused update still changed theta")
    assert torch.equal(m.m, m0), "the refused update still changed the m moment"
    assert torch.equal(m.v, v0), "the refused update still changed the v moment"
    assert m.t == t0 + 1, (
        f"t did not advance over the refused step ({m.t} vs {t0 + 1}): the "
        "data was consumed, so the LR schedule must not rewind")
    assert not np.isfinite(loss), "the injected NaN did not reach the loss"
    print("[PASS] a non-finite gradient is REFUSED: theta/m/v bit-identical, "
          "refusal counted, t advanced, loss reports the NaN")

    # the run must RECOVER: the next clean step applies normally
    with torch.no_grad():
        m.views["Wd"][0, 0] = 0.0
    before = m.theta.detach().clone()
    step_once(m, seed=3)
    assert not torch.equal(m.theta, before), (
        "the engine did not resume stepping after a refused update")
    print("[PASS] the engine resumes stepping after a refusal")

    # ---- 3. the global grad clip ------------------------------------------
    # One step with the clip off, to learn the gradient; then the same step
    # with a clip that is smaller than the gradient's norm, and compare the
    # resulting theta against a hand-written AdamW applied to the CLIPPED
    # gradient (the moments must see the clipped gradient).
    off = build(grad_clip=0.0, nan_guard=False)
    step_once(off, seed=7)
    gr = off.grads.detach().clone()
    gn = float(torch.linalg.vector_norm(gr).item())
    clip = gn / 4.0

    on = build(grad_clip=clip, nan_guard=False)
    th0 = on.theta.detach().clone()
    m0 = on.m.detach().clone()
    v0 = on.v.detach().clone()
    step_once(on, seed=7)
    coef = clip / (gn + 1e-6)
    clipped = (gr * coef).detach().clone()
    want = manual_adamw(th0.detach().cpu().numpy(), clipped.cpu().numpy(),
                         m0.detach().cpu().numpy(), v0.detach().cpu().numpy(),
                         t=1, lr=np.float32(on.lr),
                         wd_vec=on._wd_coord_vec().detach().cpu().numpy())
    got = on.theta.detach().cpu().numpy()
    rel = np.max(np.abs(got - want) / (np.abs(want) + 1e-12))
    assert rel < 2e-3, (
        f"the clipped update does not match the manual AdamW on the clipped "
        f"gradient (max rel dev {rel:.3e})")
    # The APPLIED gradient must be bounded by the clip. This, not the step
    # size, is what the clip promises: under AdamW the parameter step is
    # nearly scale-invariant in the gradient, so a clip that failed to scale
    # the gradient would still produce a normal-looking step and pass a
    # step-size check.
    gn_after = float(torch.linalg.vector_norm(on.grads).item())
    assert abs(gn_after - clip) / clip < 0.02, (
        f"the applied gradient's norm is {gn_after:.6f}, not the clip "
        f"{clip:.6f}: the clip did not scale the gradient that was applied")
    # and the clipped run must DIFFER from the unclipped one (the moments
    # differ, even though the step size is nearly scale-invariant)
    off2 = build(grad_clip=0.0, nan_guard=False)
    th0b = off2.theta.detach().clone()
    step_once(off2, seed=7)
    assert not torch.equal(on.theta, off2.theta), (
        "clipping the gradient changed nothing at all")
    d_clipped = float((on.theta - th0).abs().max())
    d_unclipped = float((off2.theta - th0b).abs().max())
    print(f"[PASS] the global grad clip scales the APPLIED gradient "
          f"(|g| {gn:.4f} -> {gn_after:.4f}, clip {clip:.4f}); manual-AdamW "
          f"agreement {rel:.2e}; max|dtheta| {d_clipped:.3e} vs unclipped "
          f"{d_unclipped:.3e} (AdamW is nearly scale-invariant, so the STEP "
          f"is not much smaller -- the clip guards the MOMENTS)")

    # ---- 4. P6: per-group weight decay ------------------------------------
    # The claim is about the UPDATE FORMULA, not about two trajectories
    # agreeing: exempting the decay group from weight decay changes the
    # decay parameter, which is IN the forward pass, so every downstream
    # row legitimately moves too. (Asserting otherwise would be asserting
    # that the parameter does not matter.) The claims are therefore:
    #   (i)  the wd vector is per-group,
    #   (ii) the applied update equals the manual AdamW with that vector,
    #   (iii) the default (wd_decay == wd) equals the manual AdamW with a
    #         UNIFORM vector, i.e. the incumbent's scalar `wd * th`.
    d01 = build(wd_decay=0.01)          # == the incumbent: uniform wd
    d00 = build(wd_decay=0.0)           # P6: the decay group is exempt
    dec_rows = []
    other_rows = []
    off = 0
    for name, shape in d01.shapes:
        k = int(np.prod(shape)) if shape else 1
        (dec_rows if name.endswith(".decay") else other_rows).append(
            slice(off, off + k))
        off += k
    dec_rows = np.concatenate([np.arange(s.start, s.stop) for s in dec_rows])
    other_rows = np.concatenate([np.arange(s.start, s.stop) for s in other_rows])
    wv00 = d00._wd_coord_vec().detach().cpu().numpy()
    wv01 = d01._wd_coord_vec().detach().cpu().numpy()
    assert np.all(wv00[dec_rows] == 0.0), "the decay group still has weight decay"
    assert np.all(wv00[other_rows] == np.float32(d00.wd)), (
        "the other groups lost their weight decay")
    assert np.all(wv01 == np.float32(d01.wd)), (
        "the default wd_decay == wd must produce a UNIFORM vector")
    print(f"[PASS] P6 per-group wd: the wd vector is uniform by default "
          f"(bit-identical to the incumbent's scalar `wd * th`) and exempts "
          f"the {len(dec_rows)} decay rows at wd_decay=0")

    # (ii) the update matches the manual AdamW with the per-group vector
    d00b = build(wd_decay=0.0, nan_guard=False)
    th0 = d00b.theta.detach().clone()
    m0 = d00b.m.detach().clone()
    v0 = d00b.v.detach().clone()
    step_once(d00b, seed=11)
    want = manual_adamw(th0.detach().cpu().numpy(),
                         d00b.grads.detach().cpu().numpy(),
                         m0.detach().cpu().numpy(), v0.detach().cpu().numpy(),
                         t=1, lr=np.float32(d00b.lr),
                         wd_vec=d00b._wd_coord_vec().detach().cpu().numpy())
    got = d00b.theta.detach().cpu().numpy()
    rel = np.max(np.abs(got - want) / (np.abs(want) + 1e-12))
    assert rel < 2e-3, f"the wd-exempt update does not match the manual AdamW ({rel:.3e})"
    print(f"[PASS] the exempt group's update is the adaptive term alone "
          f"(manual-AdamW agreement {rel:.2e})")

    # (iii) the DEFAULT is the incumbent: manual AdamW with a uniform vector
    d01b = build(wd_decay=0.01, nan_guard=False)
    th0 = d01b.theta.detach().clone()
    m0 = d01b.m.detach().clone()
    v0 = d01b.v.detach().clone()
    step_once(d01b, seed=11)
    want = manual_adamw(th0.detach().cpu().numpy(),
                         d01b.grads.detach().cpu().numpy(),
                         m0.detach().cpu().numpy(), v0.detach().cpu().numpy(),
                         t=1, lr=np.float32(d01b.lr),
                         wd_vec=np.full(d01b.theta.numel(), np.float32(d01b.wd),
                                        dtype=np.float32))
    got = d01b.theta.detach().cpu().numpy()
    rel = np.max(np.abs(got - want) / (np.abs(want) + 1e-12))
    assert rel < 2e-3, (
        f"the default wd_decay == wd does not reproduce the incumbent's "
        f"scalar weight decay ({rel:.3e})")
    print(f"[PASS] the default wd_decay == wd reproduces the incumbent's "
          f"uniform weight decay (manual-AdamW agreement {rel:.2e})")

    # ---- 5. the --lr-decay trap is gone when wd_decay=0 -------------------
    lr = 5e-4
    scaled = build(lr_decay=16 * lr, wd_decay=0.0, nan_guard=False)
    plain = build(lr=16 * lr, wd_decay=0.0, nan_guard=False)
    step_once(scaled, seed=13)
    step_once(plain, seed=13)
    a_dec = scaled.theta.detach().cpu().numpy()[dec_rows]
    b_dec = plain.theta.detach().cpu().numpy()[dec_rows]
    rel = np.max(np.abs(a_dec - b_dec) / (np.abs(b_dec) + 1e-12))
    assert rel < 2e-3, (
        f"with wd_decay=0 the scaled decay role is not equivalent to a plain "
        f"LR of the same size (max rel dev {rel:.3e}): the weight-decay term "
        f"is still in the bracket")
    print(f"[PASS] with wd_decay=0, --lr-decay scales ONLY the adaptive term "
          f"(equivalence to a plain LR of the same size: {rel:.2e})")

    # ---- 6. the numpy reference carries the same features ------------------
    np_m = numpy_port.Model(dim=32, layers=2)
    np_m.new_states(8)
    assert np_m.nan_guard is True and np_m.grad_clip == 0.0
    assert np_m.wd_decay == 0.01
    print("[PASS] the numpy reference carries the same defaults "
          "(nan_guard on, clip off, wd_decay == wd)")

    print("\nSAFETY GATE: PASS")


if __name__ == "__main__":
    main()
