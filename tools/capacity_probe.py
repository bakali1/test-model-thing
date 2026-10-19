#!/usr/bin/env python3
"""capacity_probe.py -- does the INIT buy retention capacity? (P4, 0 training)

    .venv/bin/python tools/capacity_probe.py
    .venv/bin/python tools/capacity_probe.py --half-min 2 --half-max 512

WHAT IT MEASURES
----------------
The ruler's retention curve, at a scale where it runs in seconds instead of
minutes, on an UNTRAINED model. The retention curve is a property of the
recurrence's DYNAMICS, not of what training did with it: two sequences that
differ only in a 4-byte key at position 0, the same d distractor bytes after
it, and the state's norm difference between the two. A model whose half-lives
are all ~1 B cannot hold that key past ~1 byte; a model with half-lives
spread to 512 B can.

So the question "is the model's missing memory a CAPACITY limit (the init
never gave it a long-lived channel) or a TRAINING limit (it had the capacity
and did not use it)?" is answerable WITHOUT training anything, and the answer
changes what to run next:

  * capacity present with the P4 init, absent with the default init
        -> the init was the binding constraint; P4 is necessary, and the
           interesting question becomes why TRAINING destroys it
           (finding 7: gradient descent undoes the log-spaced init within
           an epoch; P6: the weight decay pulls the decay logits back to
           0, i.e. back to a 1-byte half-life).
  * capacity absent even with the P4 init
        -> the binding constraint is the WRITE PATH: the state is a linear,
           order-losing EMA of raw byte embeddings (gpu_port.py:
           `state = states * a + enc`), which cannot represent "that name
           appeared 200 bytes ago" in any form a readout could use. Then the
           next lever is the WRITE PATH (hierarchical wiring, as in the CUDA
           fork), not the init.

MEASURED (2026-10-12, dim 512, layers 16, 64 samples, the ruler):

    init                      d=8     d=32    d=128   d=512
    default (~1 B)           0.40%   0.00%   0.00%   0.00%
    P4 (2..512 B)           49.2%   21.0%    5.8%    1.05%
    TRAINED incumbent        4.8%    1.7%    0.36%   0.0007%

    (matched-pair signal / random-byte control floor)

The third row is the one that matters for the plan: the TRAINED model retains
LESS than a fresh P4 model at every distance. The capacity is real, and
training gave it back.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy_port
from tools.membench import eval_api as E


def build(dim, layers, seed, half_life_init):
    numpy_port.rng = np.random.default_rng(seed)
    m = numpy_port.Model(dim=dim, layers=layers, lr=5e-4,
                         half_life_init=half_life_init)
    m.new_states(1)
    return m


def probe(dim, layers, seed, half_life_init, d_list, samples):
    m = build(dim, layers, seed, half_life_init)
    r = E.retention_curve(d_list=d_list, sample_count=samples, model=m)
    out = {}
    for d in sorted(r):
        v = r[d]
        ctrl = float(v.get("control_value") or 0.0)
        # _memory_meta does `meta.update(extra)`, so the veto is a TOP-LEVEL
        # key of the returned dict, not nested under "extra".
        veto = float(v.get("mismatched_cue_veto")
                     or (v.get("extra") or {}).get("mismatched_cue_veto") or 0.0)
        val = float(v.get("value") or 0.0)
        out[d] = {
            "value": val,
            "control": ctrl,
            "veto": veto,
            "ratio": (val / ctrl) if ctrl else float("nan"),
            "veto_ratio": (val / veto) if veto else float("nan"),
        }
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--distances", default="8,32,128")
    ap.add_argument("--half-min", type=float, default=None)
    ap.add_argument("--half-max", type=float, default=None)
    args = ap.parse_args()

    d_list = [int(x) for x in args.distances.split(",") if x.strip()]
    hli = None
    if (args.half_min is None) != (args.half_max is None):
        ap.error("--half-min and --half-max must be given together")
    if args.half_min is not None:
        hli = (args.half_min, args.half_max)

    print(f"capacity probe: dim={args.dim} layers={args.layers} "
          f"seed={args.seed} samples={args.samples} half_life_init={hli}")
    print("retention = matched-pair state-norm difference / random-byte "
          "control floor\n")
    res = probe(args.dim, args.layers, args.seed, hli, d_list, args.samples)
    print("   d     matched        control          veto        retention")
    for d in sorted(res):
        r = res[d]
        print(f"{d:>5}  {r['value']:>12.4f}  {r['control']:>12.4f}  "
              f"{r['veto']:>10.4f}   {r['ratio'] * 100:>7.3f}%")


if __name__ == "__main__":
    main()
