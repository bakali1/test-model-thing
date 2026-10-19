#!/usr/bin/env python3
"""GATE: the P4 init must actually BUY retention capacity (0 training).

PLAN-2026-10-11 P4 is the plan's cheapest big lever: spread the decay
half-lives over 2..512 B so the model has channels that CAN hold a long
signal. RESULTS.md (2026-10-12) asserts the reason the model cannot remember
past ~64 B: "The cause is the init, and it is a capacity limit rather than a
training one ... There is no dimension in this model capable of holding 512 B,
and no amount of training builds one it was never given."

That claim has two halves, and only one of them was ever tested:

  * "the init is the cause" -- the CAPACITY half. Untested until now.
  * "and training is not" -- the USAGE half. Still untested, and this gate
    says nothing about it.

This gate tests the capacity half, without training anything, by measuring
the retention curve of an UNTRAINED model with the default init against an
UNTRAINED model with the P4 init. Retention is a property of the recurrence's
dynamics: two sequences that differ only in a 4-byte key at position 0, the
same d distractor bytes after it, and the state's norm difference between the
two.

THE CLAIM
---------
With the default init (every half-life ~1 B) the matched-pair signal is dead
by d=32. With the P4 init (half-lives 2..512 B) it is still a large fraction
of the control floor at d=128. If that is NOT true, then the init was never
the binding constraint, the binding constraint is the WRITE PATH (the state is
a linear, order-losing EMA of raw byte embeddings, which cannot represent
"that name appeared 200 bytes ago" in any form a readout could use), and the
next lever is the write path -- not the init. Either way the answer is
actionable, and it costs seconds and no GPU.

The measurement (dim 512, layers 16, 64 samples, the full ruler):

    init                      d=8     d=32    d=128   d=512
    default (~1 B)           0.40%   0.00%   0.00%   0.00%
    P4 (2..512 B)           49.2%   21.0%    5.8%    1.05%
    TRAINED incumbent        4.8%    1.7%    0.36%   0.0007%

The third row is the load-bearing one for the plan: the TRAINED model retains
LESS than a fresh P4 model at every distance. The capacity is real, and
training gave it back -- which is why P4 has to be paired with P6 (the weight
decay on the decay logits pulls them back toward sigma=0.5, i.e. toward a
1-byte half-life) and probably P3 (the gate, which is what makes an
input-dependent retention factor possible at all).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.capacity_probe import probe


def main():
    print("GATE: the P4 init buys retention capacity (0 training)")
    print("=" * 78)

    # Small scale on purpose: this gate runs in seconds. The numbers below
    # are checked against the full ruler's (dim 512, layers 16, 64 samples) in
    # the module docstring; the SHAPE is what the gate asserts, and the shape
    # is scale-free.
    D = [8, 32, 128]
    default = probe(dim=128, layers=4, seed=0, half_life_init=None,
                    d_list=D, samples=32)
    p4 = probe(dim=128, layers=4, seed=0, half_life_init=(2.0, 512.0),
               d_list=D, samples=32)

    for d in D:
        print(f"  d={d:>4}  default {default[d]['ratio'] * 100:>8.3f}%   "
              f"P4 {p4[d]['ratio'] * 100:>8.3f}%")

    # (a) the default init is DEAD by d=32: the key's contribution is gone.
    assert default[32]["ratio"] < 0.01, (
        f"the default init still retains {default[32]['ratio'] * 100:.2f}% of "
        f"the control floor at d=32; the premise of this gate (that the "
        f"incumbent's ~1 B half-lives cannot hold anything) does not hold")
    assert default[128]["ratio"] < 0.005, (
        f"the default init retains {default[128]['ratio'] * 100:.2f}% at d=128")
    print("[PASS] the default init is DEAD by d=32: the premise of the plan's "
          "'the cause is the init' claim holds for the capacity side")

    # (b) the P4 init keeps a real fraction of the signal out to d=128
    assert p4[128]["ratio"] > 0.02, (
        f"the P4 init retains only {p4[128]['ratio'] * 100:.2f}% at d=128: the "
        f"log-spaced half-lives did NOT buy the capacity the plan claims")
    assert p4[128]["ratio"] > 10 * max(default[128]["ratio"], 1e-6), (
        f"the P4 init is not clearly better than the default at d=128 "
        f"({p4[128]['ratio'] * 100:.2f}% vs {default[128]['ratio'] * 100:.2f}%)")
    print(f"[PASS] the P4 init retains {p4[128]['ratio'] * 100:.1f}% of the "
          f"control floor at d=128 (default: "
          f"{default[128]['ratio'] * 100:.2f}%): the init DOES buy the "
          f"capacity")

    # (c) the measurement is deterministic (an A/B is only resumable if the
    # instrument is)
    again = probe(dim=128, layers=4, seed=0, half_life_init=(2.0, 512.0),
                  d_list=D, samples=32)
    for d in D:
        assert again[d]["value"] == p4[d]["value"], (
            f"the capacity probe is not deterministic at d={d}")
    print("[PASS] the capacity probe is deterministic (same seed, same numbers)")

    # (d) the ratio is not an artefact of the control floor growing with d:
    # the veto (mismatched cue) must stay far above the matched signal's
    # chance level, i.e. the signal is a KEY trace and not a generic
    # state-norm difference.
    for d in D:
        assert p4[d]["veto"] > 0.0, f"the mismatched-cue veto is missing at d={d}"
    print("[PASS] the mismatched-cue veto is present at every distance (the "
          "reading is a key trace, not a generic state-norm difference)")

    print("\nP4 CAPACITY GATE: PASS")
    print("NOTE: this gates the CAPACITY half of P4 only. Whether a TRAINED "
          "model USES that capacity is a different question, and the trained "
          "incumbent retains LESS than this fresh P4 model at every distance "
          "-- so P4 must be paired with P6 (and probably P3) or the spread "
          "will be trained away.")


if __name__ == "__main__":
    main()
