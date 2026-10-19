"""check_parity.py — numpy_port ≡ gpu_port parity gate (the "100% sure" test).

Given the same init (numpy model seeded, hot dims forced so the tail is
engaged) and the same byte stream, both implementations must stay in lockstep:
per-step loss / last_ce within ~1e-3 relative, per-parameter gradients within
~2e-2 relative (fp32 matmul accumulation order is the only tolerated
difference; elementwise/buffer ops are bit-identical by construction).

Also verifies the checkpoint round trip (numpy -> torch -> npz -> numpy) and a
mid-run continuation through the .npz bridge.

Usage:  .venv/bin/python tests/check_parity.py [--steps 150] [--B 32]
Exit code 0 = PASS.
"""
import argparse
import os
import struct
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)         # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/, for
#                                                             # _engine_pre_t
import numpy_port
import gpu_port


def build_pair(dim, layers, B, seed, n_hot):
    """Seeded numpy model; forced hot dims per layer; torch copy via from_numpy."""
    numpy_port.rng = np.random.default_rng(seed)
    np_model = numpy_port.Model(dim=dim, layers=layers, lr=5e-4)
    # force hot dims so the tail engages: pre-sigmoid +4 => sigma ~0.982
    for i, l in enumerate(np_model.layers):
        for j in range(n_hot):
            l.decay[j] = 4.0
    np_model.new_states(B)
    gpu_model = gpu_port.GpuModel(dim=dim, layers=layers, lr=5e-4)
    gpu_model.from_numpy(np_model)
    return np_model, gpu_model


def byte_streams(T, B, seed):
    rng = np.random.default_rng(seed)
    flat = (rng.integers(1, 250, size=T * B)).astype(np.int64)
    cs = flat.reshape(T, B)
    ns = np.roll(cs, -1, axis=0)
    ends = (rng.random((T, B)) < 0.02).astype(np.float32)
    return cs, ns, ends


def rel(a, b, floor=1e-4):
    return float(np.max(np.abs(a - b) / (np.abs(b) + floor)))


def run_steps(np_model, gpu_model, cs, ns, ends, n_steps, warm="", verbose=True):
    B = cs.shape[1]
    bad_loss, bad_ce = 0.0, 0.0
    dev = gpu_model.device
    worst_grad = ("", 0.0)
    t0 = time.time()
    for t in range(1, n_steps + 1):
        c = cs[t].tolist(); n = ns[t].tolist(); e = ends[t].tolist()
        l_np = np_model.step_batch(c, n, e)
        l_gp = gpu_model.step_batch(
            torch.from_numpy(cs[t]).to(dev), torch.from_numpy(ns[t]).to(dev),
            torch.from_numpy(ends[t]).to(dev))
        bad_loss = max(bad_loss, abs(l_np - l_gp) / max(1.0, abs(l_np)))
        bad_ce = max(bad_ce, abs(np_model.last_ce - gpu_model.last_ce)
                     / max(1.0, abs(np_model.last_ce)))
        if t % 50 == 0 and verbose:
            print(f"  [{warm}] step {t}: loss {l_np:.4f}/{l_gp:.4f} "
                  f"maxrel_loss {bad_loss:.2e} ce {bad_ce:.2e}", flush=True)
    # gradients + moments after the run
    for name in np_model.views:
        a = np_model.gviews[name]
        b = gpu_model.gviews[name].detach().cpu().numpy()
        r = rel(a, b)
        if r > worst_grad[1]:
            worst_grad = (name, r)
    rm = rel(np_model.m, gpu_model.m.detach().cpu().numpy())
    rv = rel(np_model.v, gpu_model.v.detach().cpu().numpy())
    # accumulator parity: ring ids, ring weights, tail cols, tail E
    acc = {"ring_b": 0.0, "ring_w": 0.0, "tail_cols": 0.0, "tail_E": 0.0}
    g_rb = gpu_model.ring_b.detach().cpu().numpy()
    acc["ring_b"] = float((np_model.ring_b != g_rb).mean())     # ids: exact
    for i, l in enumerate(np_model.layers):
        acc["ring_w"] = max(acc["ring_w"],
                            rel(l.ring_w, gpu_model.layers[i].ring_w
                                .detach().cpu().numpy(), floor=1e-6))
        nt = np_model.tail.nhot(i) if np_model.tail else 0
        gt = gpu_model.tail.nhot(i) if gpu_model.tail else 0
        if nt != gt:
            acc["tail_cols"] = 1.0
        elif nt:
            e_np = np_model.tail.E[i]
            e_gp = gpu_model.tail.E[i].detach().cpu().numpy()
            acc["tail_E"] = max(acc["tail_E"],
                                rel(e_np, e_gp, floor=1e-3))
    # params after the run (theta must track too, not just grads/moments)
    pth = rel(np_model.theta, gpu_model.theta.detach().cpu().numpy(),
              floor=1e-6)
    return (bad_loss, bad_ce, worst_grad, rm, rv, acc, pth,
            time.time() - t0)


def ring_warmup_test(np_writer, ckpt_path, cs, ns, ends, n_steps):
    """Review #7: ring_b/ring_w are NOT persisted, so a resumed model starts
    with an empty ring: cold-dim embed credit is missing from step 1. Unlike
    the tail (E is saved, so hot dims are exact immediately), this difference
    does NOT wash out — a perturbed embed changes later states, so the two
    trajectories separate and stay separated. That is the accepted cost of the
    numpy-compatible checkpoint size, and the reason A/B arms must not compare
    the first ring_k steps after a resume as if they were a learning effect.

    Asserts the bounded, documented facts rather than convergence:
      * saved tail E is restored EXACTLY at resume;
      * the first-step embed perturbation is finite and small in absolute
        terms relative to the embed scale;
      * the loss stays in the same band as the writer's (no blow-up).
    Returns (first_embed_rel, last_embed_rel, abs_rms, loss_ratio, ring_k).
    """
    import copy
    # writer must be the SAME engine that wrote the file: torch->torch is
    # bit-exact for saved state; numpy->torch is only parity-level (~1e-6).
    writer = copy.deepcopy(np_writer)            # torch, still has its live ring
    resumed = gpu_port.GpuModel(dim=writer.dim, layers=writer.layercount,
                                lr=writer.lr)
    resumed.new_states(cs.shape[1])
    resumed.load(ckpt_path)                      # ring starts empty, tail E saved
    dev = resumed.device
    # (a) the persisted tail must come back exactly
    e_tail_exact = True
    for i in range(writer.layercount):
        w = writer.tail.E[i].detach().cpu().numpy()
        g = resumed.tail.E[i].detach().cpu().numpy()
        if w.shape != g.shape or not np.array_equal(w, g):
            e_tail_exact = False
    first_rel, last_rel, losses = 0.0, 0.0, []
    w_scale = float(torch.sqrt(torch.mean(writer.embed ** 2))) + 1e-12
    for t in range(1, min(n_steps, cs.shape[0] - 1) + 1):
        c, n_, e_ = cs[t], ns[t], ends[t]
        lw = writer.step_batch(torch.from_numpy(c).to(dev),
                               torch.from_numpy(n_).to(dev),
                               torch.from_numpy(e_).to(dev))
        lr_ = resumed.step_batch(torch.from_numpy(c).to(dev),
                                 torch.from_numpy(n_).to(dev),
                                 torch.from_numpy(e_).to(dev))
        losses.append((float(lw), float(lr_)))
        e = rel(writer.embed.detach().cpu().numpy().ravel(),
                resumed.embed.detach().cpu().numpy().ravel(), floor=1e-6)
        if t == 1:
            first_rel = e
        last_rel = e
    abs_rms = float(np.sqrt(np.mean(
        (writer.embed.detach().cpu().numpy().ravel()
         - resumed.embed.detach().cpu().numpy().ravel()) ** 2)))
    loss_ratio = max((abs(r) / max(1e-6, abs(w)) for w, r in losses[-5:]),
                     default=1.0)
    return e_tail_exact, first_rel, last_rel, abs_rms / w_scale, loss_ratio, \
        resumed.ring_k


def membership_cadence_test(dim, layers, B, seed, n_hot):
    """Review #8: numpy re-evaluates hot membership every step; gpu every
    REBASE_EVERY (500). A dim that crosses the threshold mid-interval is
    adopted by numpy immediately and by gpu at the next rebase. Pins that
    documented behavior: lag exists inside the interval, and an explicit
    rebase reconciles the two engines exactly."""
    np_m, gp_m = build_pair(dim, layers, B, seed, n_hot)
    t0, T = 0, 40
    cs, ns, ends = byte_streams(T + 2, B, seed=seed + 1)
    dev = gp_m.device
    for t in range(1, t0 + 1):
        np_m.step_batch(cs[t].tolist(), ns[t].tolist(), ends[t].tolist())
        gp_m.step_batch(torch.from_numpy(cs[t]).to(dev),
                        torch.from_numpy(ns[t]).to(dev),
                        torch.from_numpy(ends[t]).to(dev))
    np_m.layers[0].decay[5] = 4.0               # cross into hot (both engines)
    gp_m.layers[0].decay[5] = 4.0               # same truth; not re-evaluated
    for t in range(t0 + 1, t0 + 6):
        np_m.step_batch(cs[t].tolist(), ns[t].tolist(), ends[t].tolist())
        gp_m.step_batch(torch.from_numpy(cs[t]).to(dev),
                        torch.from_numpy(ns[t]).to(dev),
                        torch.from_numpy(ends[t]).to(dev))
    np_hot = np_m.tail.nhot(0)
    gp_hot = gp_m.tail.nhot(0)                  # stale: rebase not due
    gp_m._tail_sync()                           # force the documented rebase
    gp_hot_after = gp_m.tail.nhot(0)
    return np_hot, gp_hot, gp_hot_after


# ---------------------------------------------------------------------------
# GATE 1 of PLAN-2026-10-11 P1: the T axis must not move a single bit at T=1.
#
# check_parity.py's job is numpy ≡ torch, and numpy is pinned at T=1 only
# (~1e-7, fp32 matmul reassociation). The T-axis gate needs a STRONGER claim
# than that: a `(B,)` call, a `(B,1)` call and the engine that existed BEFORE
# step_batch grew a t-loop must agree BIT-EXACTLY over a live multi-step run --
# theta, m, v, ring_b, ring_w, the per-layer states and decaytraces, the tail
# accumulators, and the per-step loss/ce. `==` tolerances cannot express that,
# so every tensor is compared as raw int32 bit patterns (which also catches a
# sign-of-zero flip) and every scalar through its IEEE-754 bits.
#
# The pre-T-axis engine is a frozen whole-file copy at
# tests/_engine_pre_t.py -- see its header for why a copy and not a cleverer
# reference. Same seed => same init, so the twins are built, not bridged.
# ---------------------------------------------------------------------------
def _twin(dim, layers, B, seed, n_hot, mod, ce_only=False, device=None):
    """Same-seed engine twin in `mod`, hot dims forced so the tail engages."""
    kw = {} if mod is not gpu_port else {"ce_only": ce_only}
    if device is not None:
        kw["device"] = device
    m = mod.GpuModel(dim=dim, layers=layers, lr=5e-4, seed=seed, **kw)
    with torch.no_grad():
        for l in m.layers:
            for j in range(n_hot):
                l.decay[j] = 4.0          # pre-sigmoid +4 => sigma ~0.982
    m.new_states(B)
    return m


def _bitsnap(m):
    """Every piece of live state the T axis can perturb, as raw bit patterns.

    Compared as int32 views rather than with `==` so that a sign-of-zero flip
    (the one thing `0.0 + x` can change, and the one thing AdamW provably
    cannot observe) would still fail the gate instead of hiding in it.
    """
    out = {"theta": m.theta, "m": m.m, "v": m.v, "grads": m.grads,
           "ring_b": m.ring_b}
    for i, l in enumerate(m.layers):
        out[f"states.{i}"] = l.states
        out[f"decaytrace.{i}"] = l.decaytrace
        out[f"ring_w.{i}"] = l.ring_w
        if m.tail is not None and i in m.tail.E:
            out[f"tail_E.{i}"] = m.tail.E[i]
    return {k: v.detach().contiguous().view(torch.int32).clone()
            for k, v in out.items()}


def _fbits(x):
    return struct.pack("<d", float(x))


def t_axis_bitidentity_test(dim, layers, B, seed, n_hot, steps):
    """(B,) ≡ (B,1) ≡ the pre-T-axis engine, bit for bit, over a live run.

    Five twins on one byte stream:
      pre  -- tests/_engine_pre_t.py, called the historical 1-D way
      1d   -- the current engine, called 1-D        (the historical path)
      w1   -- the current engine, called (B,1)      (the same thing, windowed)
      ce1d -- current engine, ce_only=True, 1-D
      cew1 -- current engine, ce_only=True, (B,1)
    and the TMT_TAIL=0 (ring-only) path is run too, because the tail's
    per-byte begin_step / credit / add_byte are exactly the operations a T loop
    can silently under-fire.

    Returns (n_compared, worst_diff_name, worst_ndiff, ok).
    """
    import _engine_pre_t
    cs, ns, ends = byte_streams(steps + 2, B, seed=seed + 5)
    # The engines in this test are pinned to CPU, and so are the inputs.
    #
    # The claim here is BIT identity ("T=1 must be bit-identical to today's
    # engine"), and bit identity is a CPU claim: on CUDA the ring credit is
    # accumulated with atomic adds, so two twins stepping the same bytes can
    # differ in the last bits by reduction ORDER alone, with no bug involved.
    # Making this test run on a GPU would therefore either fail for a reason
    # that is not a defect, or -- worse -- be "fixed" by loosening the bar
    # until it stops testing anything.
    #
    # The CUDA version of the same underlying question is tolerance-based and
    # lives in tests/check_trace_bptt.py --device cuda, which compares the
    # windowed gradient against an exact CPU reference.
    dev = "cpu"
    worst = ["", 0]                      # [what, how many elements differ]
    n_cmp = [0]

    def note(what, ndiff):
        n_cmp[0] += 1
        if ndiff > worst[1]:
            worst[0], worst[1] = what, ndiff

    for tag, tail_env in (("tail on", "1"), ("TMT_TAIL=0", "0")):
        # tail_enabled is read from the environment at CONSTRUCTION, so the
        # ring-only arm is reached by setting it for the duration of the arm and
        # restoring it after -- not by mutating a live model.
        prev = os.environ.get("TMT_TAIL")
        os.environ["TMT_TAIL"] = tail_env
        try:
            pre = _twin(dim, layers, B, seed, n_hot, _engine_pre_t, device=dev)
            m1d = _twin(dim, layers, B, seed, n_hot, gpu_port, device=dev)
            mw1 = _twin(dim, layers, B, seed, n_hot, gpu_port, device=dev)
            c1d = _twin(dim, layers, B, seed, n_hot, gpu_port, ce_only=True,
                        device=dev)
            cw1 = _twin(dim, layers, B, seed, n_hot, gpu_port, ce_only=True,
                        device=dev)
            # (name, windowed model, its 1-D reference). Each model is
            # stepped exactly ONCE per byte and may serve as the reference for
            # more than one pair -- stepping it per pair would advance it twice
            # and compare two different runs.
            pairs = [("1d-vs-pre", m1d, pre),
                     ("(B,1)-vs-1d", mw1, m1d),
                     ("ce_only (B,1)-vs-1d", cw1, c1d)]
            for t in range(1, steps + 1):
                c = torch.from_numpy(cs[t]).to(dev)
                n = torch.from_numpy(ns[t]).to(dev)
                e = torch.from_numpy(ends[t]).to(dev)
                cw, nw, ew = c.unsqueeze(1), n.unsqueeze(1), e.unsqueeze(1)
                # one call per model: 1-D for the pre engine and every 1-D twin,
                # (B,1) for every windowed twin
                for m, args in ((pre, (c, n, e)), (m1d, (c, n, e)),
                                (mw1, (cw, nw, ew)), (c1d, (c, n, e)),
                                (cw1, (cw, nw, ew))):
                    m.step_batch(*args)
                for name, win, ref in pairs:
                    if _fbits(win.last_loss_tensor) != _fbits(
                            ref.last_loss_tensor):
                        note(name + " loss", 1)
                    if _fbits(win.last_ce) != _fbits(ref.last_ce):
                        note(name + " last_ce", 1)
                    if _fbits(win.last_ce_sum) != _fbits(ref.last_ce_sum):
                        note(name + " last_ce_sum", 1)
                    for i, (x, y) in enumerate(zip(win.last_aux_masked,
                                                   ref.last_aux_masked)):
                        if _fbits(x) != _fbits(y):
                            note(name + f" last_aux_masked[{i}]", 1)
                    for i, (x, y) in enumerate(zip(win.last_aux_raw,
                                                   ref.last_aux_raw)):
                        if _fbits(x) != _fbits(y):
                            note(name + f" last_aux_raw[{i}]", 1)
            # ---- final live state, bit for bit -------------------------------
            for name, win, ref in pairs:
                if (win.t, win.bytes_seen) != (ref.t, ref.bytes_seen):
                    note(name + " counters t/bytes_seen", 1)
                sa, sb = _bitsnap(win), _bitsnap(ref)
                for k in sa:
                    if k not in sb:
                        continue
                    note(f"{name} {k}", int((sa[k] != sb[k]).sum()))
            print(f"  [T=1 bit-identity, {tag}] {steps} steps x {len(pairs)} "
                  f"pairs: worst differing element {worst[1]} "
                  f"({worst[0] or 'none'})", flush=True)
        finally:
            if prev is None:
                os.environ.pop("TMT_TAIL", None)
            else:
                os.environ["TMT_TAIL"] = prev
    return n_cmp[0], worst[0], worst[1], worst[1] == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-hot", type=int, default=3, help="forced hot dims/layer")
    args = ap.parse_args()

    T = args.steps + 2
    cs, ns, ends = byte_streams(T, args.B, seed=7)
    np_model, gpu_model = build_pair(args.dim, args.layers, args.B, 11,
                                     args.n_hot)
    print(f"np: {np_model.count():,} params | ring_k {np_model.ring_k} | "
          f"B {args.B} | dim {args.dim} layers {args.layers} | "
          f"tail hot: {[(i, np_model.tail.nhot(i)) for i in range(args.layers) if np_model.tail and np_model.tail.nhot(i)]}")

    bl, bce, wg, rm, rv, acc, pth, dt = run_steps(np_model, gpu_model, cs,
                                                   ns, ends, args.steps, "live")
    hot_total = (sum(np_model.tail.nhot(i) for i in range(args.layers))
                 if np_model.tail else 0)
    print(f"\n  live {args.steps} steps in {dt:.1f}s: maxrel loss {bl:.2e} "
          f"ce {bce:.2e} | tail engaged on {hot_total} dims")
    print(f"  worst grad rel {wg[0]} = {wg[1]:.2e} | m {rm:.2e} v {rv:.2e} "
          f"| theta {pth:.2e}")
    print(f"  accumulators: ring_b mismatch {acc['ring_b']:.0%} | ring_w rel "
          f"{acc['ring_w']:.2e} | tail_cols mismatch {acc['tail_cols']:.0%} | "
          f"tail_E rel {acc['tail_E']:.2e}")

    # ---- continuation: save npz from torch, reload into BOTH, keep going ----
    gpu_model.save("/tmp/_parity_a.npz")
    np2 = numpy_port.Model(dim=args.dim, layers=args.layers, lr=5e-4)
    numpy_port.rng = np.random.default_rng(99)   # any; load overrides below
    np2.new_states(args.B)
    np2.load("/tmp/_parity_a.npz")
    gp2 = gpu_port.GpuModel(dim=args.dim, layers=args.layers, lr=5e-4)
    gp2.new_states(args.B)
    gp2.load("/tmp/_parity_a.npz")
    assert np2.t == gp2.t == np_model.t, (np2.t, gp2.t, np_model.t)
    assert np2.ring_k == gp2.ring_k == np_model.ring_k
    bl2, bce2, wg2, rm2, rv2, acc2, pth2, dt2 = run_steps(
        np2, gp2, cs, ns, ends, 40, "continue", verbose=False)
    print(f"  continue 40 steps through npz bridge: maxrel loss {bl2:.2e} "
          f"ce {bce2:.2e} worst grad {wg2[1]:.2e} m {rm2:.2e} v {rv2:.2e} "
          f"theta {pth2:.2e}")
    print(f"  acc ring_b {acc2['ring_b']:.0%} ring_w {acc2['ring_w']:.2e} "
          f"tail_cols {acc2['tail_cols']:.0%} tail_E {acc2['tail_E']:.2e}")

    # ---- round-trip bit-exactness: torch save -> numpy save -> compare ----
    gpu_model.save("/tmp/_parity_b.npz")
    np3 = numpy_port.Model(dim=args.dim, layers=args.layers)
    np3.new_states(args.B)
    np3.load("/tmp/_parity_b.npz")
    d1 = np.load("/tmp/_parity_a.npz")
    d2 = np.load("/tmp/_parity_b.npz")
    for k in d1.files:
        if k in ("t", "bytes", "B", "ring_k"):
            assert int(d1[k]) == int(d2[k]), k
        else:
            assert np.array_equal(d1[k], d2[k]), f"round-trip mismatch: {k}"
    print("  saved-state npz round-trip bit-exact: PASS "
          "(saved state only; ring_b/ring_w are NOT persisted — see #7)")

    # ---- review #7: ring warmup after a resume ----
    e_exact, first_rel, last_rel, abs_rel, loss_ratio, ring_k = ring_warmup_test(
        gpu_model, "/tmp/_parity_a.npz", cs, ns, ends, 2 * args.layers + 20)
    print(f"  ring warmup: tail E restored exactly={e_exact}; embed rel err "
          f"first step {first_rel:.2e} -> last {last_rel:.2e} "
          f"(abs/RMS {abs_rel:.2e}); loss ratio {loss_ratio:.2f}; K={ring_k}")
    # ---- review #8: membership rebase cadence ----
    np_hot, gp_hot, gp_after = membership_cadence_test(
        args.dim, args.layers, args.B, 11, args.n_hot)
    print(f"  membership cadence: numpy hot {np_hot}, gpu before rebase "
          f"{gp_hot} (lagging by design), after forced rebase {gp_after}")

    # Tolerances: loss/ce are ~1e-7 (fp32 matmul reassociation only). Grads
    # and the tail EMA use a relative floor because the embed gradient and the
    # eligibility EMA span several orders of magnitude; the threshold is set
    # ~1.5x above the measured worst (3.6e-3 / 1e-3) so a real divergence
    # (orders of magnitude) fails loudly. ring_b must match EXACTLY (ids).
    # Post-resume moments/params/accumulators are gated too (review #5: rm2
    # and rv2 used to be printed but not asserted).
    ok = (bl < 1e-5 and bce < 1e-5 and wg[1] < 5e-3 and rm < 5e-3
          and rv < 1e-4 and pth < 5e-3
          and acc["ring_b"] == 0.0 and acc["tail_cols"] == 0.0
          and acc["ring_w"] < 5e-3 and acc["tail_E"] < 5e-3
          and bl2 < 1e-5 and bce2 < 1e-5 and wg2[1] < 5e-3
          and rm2 < 5e-3 and rv2 < 1e-4 and pth2 < 5e-3
          and acc2["ring_b"] == 0.0 and acc2["tail_cols"] == 0.0
          and acc2["ring_w"] < 5e-3 and acc2["tail_E"] < 5e-3)
    # #7: the ring omission is real and does not wash out — assert the
    # documented shape of it (tail exact, bounded perturbation, no blow-up)
    # rather than pretending two trajectories re-converge.
    ok &= e_exact and first_rel > 0 and abs_rel < 0.5 and loss_ratio < 2.0
    # #8: gpu lags inside the cadence window, then matches on rebase
    ok &= (gp_hot == np_hot - 1) and (gp_after == np_hot)
    # ---- GATE 1 (PLAN-2026-10-11 P1): T=1 must be bit-identical ----------
    n_cmp, wname, wdiff, t_ok = t_axis_bitidentity_test(
        args.dim, args.layers, args.B, 11, args.n_hot, 60)
    print(f"  T=1 bit-identity: {n_cmp} comparisons, worst differing element "
          f"{wdiff} ({wname or 'none'}) -> "
          f"{'PASS' if t_ok else 'FAIL'}")
    ok &= t_ok
    print("\nPARITY:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())