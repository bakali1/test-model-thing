#!/usr/bin/env python3
"""check_halflife_init.py -- P4 gate: the log-spaced half-life init.

P4 gives the model decay dims spanning half-lives [half_min, half_max]
geometrically. The incumbent uniform init cannot express this at all:
|logit| <= 1/sqrt(dim) = 0.044 puts every sigma within 1% of 0.5, i.e. every
half-life within a byte of 1 B. The trained checkpoint confirms it stayed
there -- sigma_max 0.9847, longest dim ~45 B, 8 of 512 dims above 8 B -- so
the incumbent has no dim capable of holding 512 B.

Three things are pinned here, because each is a way P4 could silently not be
what it claims:

  (a) the requested half-lives are what the init actually produces, exactly,
      in every layer;
  (b) the SET is geometric while the ASSIGNMENT to dims is shuffled, so dim
      index is not coupled to a timescale;
  (c) with the flags absent the engine is BIT-IDENTICAL to the pre-P4 build,
      which is what keeps check_parity.py and every existing A/B arm valid.

(c) is the one that makes P4 an additive flag rather than a behaviour change.
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


DIM = 64          # small: this is an init gate, not a throughput gate
LAYERS = 3
SEED = 0
HMIN, HMAX = 2.0, 512.0
# 1e-3 relative, NOT 1e-5. The logits are float32, and the round trip
# sigma -> half-life amplifies that at the long end: for h=512, sigma=0.99865
# so ln(sigma) is ~-1.35e-3, and h = ln(0.5)/ln(sigma) divides by that small
# number. Measured round-trip deviation is 3.9e-5, which is float32
# conditioning, not wrong spacing -- a real spacing bug is percent-level.
RTOL = 1e-3

PASS = True


def check(name, ok, detail=""):
    global PASS
    if not ok:
        PASS = False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))


def decay_of(m, layer):
    """Return sigma (not logits) for one layer's decay vector."""
    return gpu_port._sig(m.layers[layer].decay.detach()).cpu().numpy().astype(
        np.float64)


def half_lives_from_sigma(sig):
    # half-life h satisfies sigma = 0.5**(1/h)  =>  h = ln(0.5)/ln(sigma)
    return np.log(0.5) / np.log(sig)


def main():
    print("P4 GATE -- log-spaced half-life init")
    print(f"  dim={DIM} layers={LAYERS} seed={SEED} "
          f"half_life_init=({HMIN}, {HMAX})\n")

    # ---- (a) the init produces the requested half-lives, exactly ----------
    print("(a) requested half-lives are what the init produces")
    m = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                          half_life_init=(HMIN, HMAX))
    want = np.sort(np.geomspace(HMIN, HMAX, DIM))
    for layer in range(LAYERS):
        got = np.sort(half_lives_from_sigma(decay_of(m, layer)))
        rel = np.max(np.abs(got - want) / want)
        check(f"layer{layer} half-life set == geomspace({HMIN}, {HMAX})",
              rel <= RTOL,
              f"max rel dev {rel:.2e} (bar {RTOL:.0e})")
    # the endpoints must be present, not merely approached
    sig0 = decay_of(m, 0)
    hl0 = half_lives_from_sigma(sig0)
    check(f"a dim really does hold {HMAX:.0f} B",
          hl0.max() >= HMAX * (1 - RTOL), f"longest {hl0.max():.4f} B")
    check(f"a dim really does forget in {HMIN:.0f} B",
          hl0.min() <= HMIN * (1 + RTOL), f"shortest {hl0.min():.4f} B")

    # ---- (b) geometric set, shuffled assignment --------------------------
    print("\n(b) the set is geometric; the assignment to dims is shuffled")
    # If the dims were left in geometric ORDER, half-life would be monotone in
    # dim index. Shuffling keeps the set while breaking that coupling.
    monotone = np.all(np.diff(hl0) > 0) or np.all(np.diff(hl0) < 0)
    check("dim index is not ordered by half-life (set is shuffled)",
          not monotone,
          "half-lives are monotone in dim index -- the shuffle did not happen"
          if monotone else f"e.g. first 8 half-lives {np.round(hl0[:8], 2)}")
    # and the shuffle must be seeded/reproducible, or A/B arms are not resumable
    m2 = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                           half_life_init=(HMIN, HMAX))
    same = all(
        np.array_equal(decay_of(m, i), decay_of(m2, i)) for i in range(LAYERS))
    check("same seed => same shuffled assignment (reproducible A/B)", same)
    m3 = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED + 1,
                           half_life_init=(HMIN, HMAX))
    differs = not np.array_equal(decay_of(m, 0), decay_of(m3, 0))
    check("different seed => different assignment (shuffle is live)", differs)

    # ---- (c) default build is bit-identical to pre-P4 --------------------
    print("\n(c) with the flags absent the engine is unchanged")
    base = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED)
    # The incumbent bound is |logit| <= 1/sqrt(dim), so sigma lies in
    # (0.5-ish, ~0.511) -- it is RANDOM within that bound, not equal to a
    # constant, so the meaningful assertion is the bound itself.
    lim = 1.0 / np.sqrt(DIM)
    lg = base.layers[0].decay.detach().cpu().numpy().astype(np.float64)
    check("default decay logits stay inside the incumbent bound",
          bool(np.all(np.abs(lg) <= lim * (1 + 1e-6))),
          f"|logit| in [{np.abs(lg).min():.5f}, {np.abs(lg).max():.5f}], "
          f"bound {lim:.5f} -> every half-life ~1 B")
    hl_base = half_lives_from_sigma(gpu_port._sig(
        torch.from_numpy(lg)).numpy().astype(np.float64))
    check("incumbent half-lives are all ~1 B (the deficit P4 addresses)",
          bool(np.all((hl_base > 0.9) & (hl_base < 1.1))),
          f"half-lives [{hl_base.min():.3f}, {hl_base.max():.3f}] B")
    # The decisive one: half_life_init=None must not perturb ANY parameter.
    ref = gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                            half_life_init=None)
    check("half_life_init=None is bit-identical to the default build",
          torch.equal(base.theta, ref.theta)
          and all(np.array_equal(decay_of(base, i), decay_of(ref, i))
                  for i in range(LAYERS)))
    check("P4 arm differs from the default (the flag does something)",
          not torch.equal(m.theta, base.theta))

    # ---- ring_k must cover the longest-lived dim -------------------------
    print("\n(d) ring_k inference survives a per-dim decay array")
    # This crashed before P4: dmax was computed with .item() on what is now a
    # per-dim quantity, and a scalar max would have under-sized the ring.
    want_k = int(min(256, np.ceil(
        np.log(gpu_port.RING_TOL) / np.log(0.5 ** (1.0 / HMAX)))))
    check("inferred ring_k covers the longest half-life",
          m.ring_k == want_k,
          f"ring_k={m.ring_k} expected {want_k} for a {HMAX:.0f} B dim")

    # ---- argument validation --------------------------------------------
    print("\n(e) bad ranges are refused, not silently clamped")
    for bad in [(0.0, 512.0), (-2.0, 512.0), (512.0, 2.0)]:
        try:
            gpu_port.GpuModel(device=DEV, dim=DIM, layers=LAYERS, seed=SEED,
                              half_life_init=bad)
            check(f"half_life_init={bad} refused", False, "accepted")
        except ValueError:
            check(f"half_life_init={bad} refused", True)

    print("\n" + ("P4 GATE: PASS" if PASS else "P4 GATE: FAIL"))
    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
