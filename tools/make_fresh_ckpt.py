#!/usr/bin/env python3
"""make_fresh_ckpt.py -- write a checkpoint for an UNTRAINED model.

    .venv/bin/python tools/make_fresh_ckpt.py out.npz
    .venv/bin/python tools/make_fresh_ckpt.py out.npz --half-min 2 --half-max 512

WHY THIS EXISTS
---------------
RESULTS.md (2026-10-12) concludes from the ruler:

    "The cause is the init, and it is a capacity limit rather than a training
     one ... There is no dimension in this model capable of holding 512 B, and
     no amount of training builds one it was never given."

That is a claim about CAPACITY, and it is testable without training anything:
the retention curve is a property of the recurrence's dynamics, so a model
with a long half-life available must retain a signal further out than a model
whose every half-life is ~1 B -- IF the capacity story is the true one.

The honest form of the experiment is therefore three rulers, no training:

    (a) the TRAINED incumbent   -- what we have (RESULTS.md, 2026-10-12)
    (b) a FRESH default-init model  -- every half-life ~1 B, random weights
    (c) a FRESH P4-init model    -- half-lives spread 2..512 B, random weights

(b) vs (c) isolates the INIT with no training, no data and no optimisation in
the way: same random weights, same seed, same ruler, only the half-life
schedule differs. If the retention curve of (c) extends past ~64 B while (b)
does not, the capacity story holds and P4 is worth running. If (c) dies at the
same distance as (b), then the init was never the binding constraint and the
binding constraint is the WRITE PATH -- the state is a linear, order-losing
EMA of raw byte embeddings (gpu_port.py: `state = states * a + enc`), which
cannot represent "a name appeared 200 bytes ago" in any form a readout could
use. That would move hierarchical wiring (the CUDA fork's answer) up the
queue, and it would be worth more than P4.

This script only writes the checkpoint. The ruler is the instrument:
    TMT_RULER_CKPT=out.npz .venv/bin/python tools/ruler.py
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gpu_port import GpuModel


def half_life_report(m):
    sig = np.concatenate([1.0 / (1.0 + np.exp(-m.layers[i].decay.detach()
                                                .cpu().numpy()))
                          for i in range(m.layercount)])
    hl = np.log(0.5) / np.log(np.clip(sig, 1e-12, 1 - 1e-12))
    return {
        "sigma_mean": float(sig.mean()),
        "sigma_max": float(sig.max()),
        "half_life_min_B": float(hl.min()),
        "half_life_max_B": float(hl.max()),
        "half_life_mean_B": float(hl.mean()),
        "n_gt_8B": int((hl > 8).sum()),
        "n_gt_16B": int((hl > 16).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1,
                    help="B for new_states; the ruler resets state per "
                         "sample, so this only affects the saved state shape")
    ap.add_argument("--half-min", type=float, default=None,
                    help="P4: log-spaced half-life init, lower end in BYTES. "
                         "Needs --half-max.")
    ap.add_argument("--half-max", type=float, default=None,
                    help="P4: log-spaced half-life init, upper end in BYTES. "
                         "Needs --half-min.")
    args = ap.parse_args()

    hli = None
    if (args.half_min is None) != (args.half_max is None):
        ap.error("--half-min and --half-max must be given together")
    if args.half_min is not None:
        hli = (args.half_min, args.half_max)

    m = GpuModel(dim=args.dim, layers=args.layers, seed=args.seed,
                 half_life_init=hli)
    m.new_states(args.batch)
    rep = half_life_report(m)
    m.save(args.out)

    print(f"wrote {args.out}")
    print(f"  dim={args.dim} layers={args.layers} seed={args.seed} "
          f"half_life_init={hli}")
    for k, v in rep.items():
        print(f"  {k:>18}: {v}")
    print("  this is an UNTRAINED model: the ruler on it measures CAPACITY, "
          "not what a trained model does with that capacity.")


if __name__ == "__main__":
    main()
