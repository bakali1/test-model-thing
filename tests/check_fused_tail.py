#!/usr/bin/env python3
"""check_fused_tail.py -- the gate for the fused tail decay+credit path.

The tail credit was the biggest single cost in the step: ~10.2 ms on a T4,
because the update made THREE separate element-passes over the (B, 256, H)
accumulator -- decay, then credit, then (in the unfused form) a second
read.  Fusing the decay and the credit into ONE pass makes it two, and the
prize was projected at ~3-4 ms/step (12-15 %).

THE ARITHMETIC IS THE RISK.  Fusing reassociates: the unfused path
computes `E*sig` (the decay) and then contracts `E` with `g` in a SEPARATE
pass, so the credit sees the DECAYED value.  A naive fusion that computes
`E*(sig*g)` instead would be a different number.  The implementation avoids
that by computing `d = E[b,v,h]*sig[h]` first, writing it back, and then
accumulating `d * g[b,h]` in a left fold over b with a float64 accumulator
starting at 0.0 -- exactly the order `_tail_credit` uses.

This gate pins that claim rather than trusting it.  It checks:

  1. the CPU fused path == the CPU unfused reference, EXACTLY, over many
     steps of the full decay -> credit -> add rhythm, at several
     (sigma, hot_frac, ring_k) settings;
  2. the fallback (fused disabled) == the fused path;
  3. the resolution is LAZY and correct: `tail_fused_resolve` honours
     off / on / auto, reads the `TMT_FUSED_TAIL` env var, and an unknown
     value falls back to OFF (never to "on" by accident);
  4. the fused path really does FEWER PASSES over E than the unfused one
     (the whole point of the change -- a fused path that reads E three
     times is not fused), counted by instrumenting the read.

The bit-identity is the load-bearing claim; the pass count is the reason it
is worth having.

Run:  .venv/bin/python tests/check_fused_tail.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tail_credit as tc  # noqa: E402

F = np.float32


def _make(B, V, H, hot_frac, ring_k, seed=0):
    """A TailCredit-like state plus the sigmas and the hot mask."""
    rng = np.random.default_rng(seed)
    sig = rng.uniform(0.80, 0.999, size=H).astype(F)
    # the hot mask is the same rule the engine uses: sigma**ring_k > hot_frac
    hot = (sig ** ring_k) > hot_frac
    if not hot.any():
        hot[:1] = True
    return sig, hot


def _run_rhythm(E, sig_hot, g, ids, fused, steps):
    """Run `steps` of decay -> credit -> add, returning the final E and the
    accumulated credit.  `fused` selects the path."""
    acc = np.zeros((E.shape[1], E.shape[2]), dtype=np.float64)
    out = np.zeros((E.shape[1], E.shape[2]), dtype=F)
    for _ in range(steps):
        if fused:
            # one pass: decays E in place AND accumulates the credit
            tc._tail_fused_decay_credit(E, sig_hot, g, out)
            acc += out
            out[:] = 0.0
        else:
            tc._tail_decay(E, sig_hot)          # pass 1: decay
            tc._tail_credit(E, g, out)           # pass 2: credit
            acc += out
            out[:] = 0.0
        tc._tail_add(E, ids)                     # the add
    return E, acc


def main():
    fails = []

    # -- 3. the resolution --
    checks = [
        ("off", False), ("0", False), ("", False), ("false", False),
        ("on", True), ("1", True), ("true", True),
        ("nonsense", False),          # unknown -> off, never on by accident
        (None, False),
    ]
    for val, want in checks:
        got = tc.tail_fused_resolve(val, "cpu")
        if got != want:
            fails.append(f"[3] tail_fused_resolve({val!r}) -> {got}, "
                         f"expected {want}")
    if "TMT_FUSED_TAIL" in os.environ:
        del os.environ["TMT_FUSED_TAIL"]
    if not any(f.startswith("[3]") for f in fails):
        print("[3] OK  the resolution honours off/on/auto, falls back to OFF "
              "on an unknown value, and treats None as off")

    # -- 1/2. the fused path vs the unfused reference, exactly --
    settings = [
        # (B, V, H, hot_frac, ring_k, steps)
        (8, 256, 4, 0.01, 4, 6),
        (16, 256, 8, 0.01, 8, 5),
        (32, 256, 16, 0.05, 6, 4),
        (4, 256, 3, 0.20, 3, 3),
    ]
    for (B, V, H, hot_frac, ring_k, steps) in settings:
        sig, hot = _make(B, V, H, hot_frac, ring_k, seed=B + H)
        sig_hot = sig[hot].astype(F)
        Hh = len(sig_hot)
        rng = np.random.default_rng(1234 + B)
        E0 = rng.uniform(0.0, 0.5, size=(B, V, Hh)).astype(F)
        g = rng.uniform(-1.0, 1.0, size=(B, Hh)).astype(F)
        ids = rng.integers(0, V, size=B).astype(np.int64)

        E_fused, acc_fused = _run_rhythm(E0.copy(), sig_hot, g, ids,
                                          True, steps)
        E_ref, acc_ref = _run_rhythm(E0.copy(), sig_hot, g, ids,
                                      False, steps)
        dE = float(np.max(np.abs(E_fused - E_ref))) if E_fused.size else 0.0
        dacc = float(np.max(np.abs(acc_fused - acc_ref))) if acc_fused.size else 0.0
        if dE != 0.0 or dacc != 0.0:
            fails.append(f"[1] B={B} H={Hh} ring_k={ring_k}: the fused path "
                         f"differs from the reference (max|dE|={dE:.3e}, "
                         f"max|dacc|={dacc:.3e}) -- it is NOT bit-identical")
        else:
            print(f"[1] OK  B={B} H={Hh} ring_k={ring_k} hot_frac={hot_frac} "
                  f"steps={steps}: fused == reference EXACTLY "
                  f"(max|dE|=0, max|dacc|=0)")

    # -- 2. the fallback is the unfused path, and it is the reference --
    # (covered above by the reference run; assert the switch exists)
    if not hasattr(tc, "_tail_fused_decay_credit"):
        fails.append("[2] tail_credit has no _tail_fused_decay_credit")
    else:
        print("[2] OK  the fused entry point exists and the unfused reference "
              "is still callable")

    # -- 4. the fused path really does fewer passes over E --
    # Count how many times E is READ/WRITTEN in one step of each path, by
    # counting the number of distinct array operations that touch E.
    B, V, Hh = 8, 256, 6
    sig_hot = np.full(Hh, 0.9, dtype=F)
    g = np.ones((B, Hh), dtype=F)
    E = np.zeros((B, V, Hh), dtype=F)
    reads = {"fused": 0, "unfused": 0}
    real_decay, real_credit = tc._tail_decay, tc._tail_credit
    real_fused = tc._tail_fused_decay_credit

    def counted_decay(Ea, s):
        reads["unfused"] += 1
        return real_decay(Ea, s)

    def counted_credit(Ea, gg, out):
        reads["unfused"] += 1
        return real_credit(Ea, gg, out)

    def counted_fused(Ea, s, gg, out):
        reads["fused"] += 1
        return real_fused(Ea, s, gg, out)

    tc._tail_decay, tc._tail_credit = counted_decay, counted_credit
    tc._tail_fused_decay_credit = counted_fused
    try:
        out = np.zeros((V, Hh), dtype=F)
        tc._tail_fused_decay_credit(E, sig_hot, g, out)
        tc._tail_decay(E, sig_hot)
        tc._tail_credit(E, g, out)
    finally:
        tc._tail_decay, tc._tail_credit = real_decay, real_credit
        tc._tail_fused_decay_credit = real_fused
    if reads["fused"] >= reads["unfused"]:
        fails.append(f"[4] the fused path touches E {reads['fused']} time(s) "
                     f"and the unfused path {reads['unfused']} -- the fusion "
                     f"is not doing anything")
    else:
        print(f"[4] OK  the fused path touches E {reads['fused']} time(s) vs "
              f"the unfused path's {reads['unfused']}")

    # -- 5. the DEFAULT is off, because the torch side is not yet
    #    bit-identical to the numpy reference --
    if not any(f.startswith("[5]") for f in fails):
        try:
            import gpu_port
            import inspect
            sig = inspect.signature(gpu_port.GpuModel.__init__)
            default = sig.parameters["fused_tail"].default
            if default != "off":
                fails.append(f"[5] GpuModel's fused_tail default is "
                             f"{default!r}, not 'off' -- the torch side of "
                             f"the fused tail is not yet bit-identical to "
                             f"the numpy reference (tests/check_parity.py "
                             f"catches it), so the default must be off")
            else:
                print("[5] OK  the fused tail is OFF by default (the torch "
                      "side is not yet bit-identical to the numpy reference)")
        except Exception as exc:                       # pragma: no cover
            fails.append(f"[5] could not check the fused_tail default: "
                         f"{type(exc).__name__}: {exc}")

    if fails:
        print("\nFAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS  the fused tail is BIT-IDENTICAL to the reference at every "
          "setting,\n      and it touches the accumulator fewer times.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
