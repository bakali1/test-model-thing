#!/usr/bin/env python3
"""inspect_halflives.py -- the half-life distribution of a checkpoint's weights.

    .venv/bin/python tools/inspect_halflives.py runs/lrroles/arm_lrdecay1x_t919897.npz
    .venv/bin/python tools/inspect_halflives.py armP4.npz --per-layer

WHY THIS EXISTS
---------------
The P4 question has two halves and the second one is the one nobody has
measured. The first: does the log-spaced init BUY the capacity to hold a
long signal? Yes -- `tools/capacity_probe.py` measures it, 0 training, and a
fresh P4-init model retains 49%/21%/5.8% of the control floor at d=8/32/128
against 0.4%/0%/0% for the default init.

The second: does that capacity SURVIVE TRAINING? This is the one that
decides whether P4 is worth anything, because the spread is not protected
from the optimizer:

  * `sigma(0) = 0.5` is exactly a 1-byte half-life, so the decoupled weight
    decay on the decay logits (P6) is a systematic pull toward NO MEMORY;
  * finding 7 says gradient descent undoes the log-spaced init within one
    epoch on real text.

And the answer is visible in the WEIGHTS, with no ruler and no training
history: if the spread survived, the checkpoint still has dims with long
half-lives; if it collapsed, every dim is back to ~1 B.

The trap this avoids: the trained incumbent (t=919,897) has a 1.16 B mean
half-life -- but it was INITIALISED with every half-life at ~1 B, so it
never had a spread to lose. It is not evidence that training destroys a P4
spread. Nobody has ever trained a P4-init model, which is why this tool
exists.

What to look for
----------------
    mean half-life        the headline: ~1 B means collapsed, ~92 B means alive
    dims > 8 B / > 16 B   how much of the spread is left
    sigma_max             the single longest-lived channel
    per-layer             whether the collapse is uniform or a few layers went

And the comparison that matters is against the FRESH P4 model, not against
the incumbent: `tools/make_fresh_ckpt.py --half-min 2 --half-max 512` writes
one, and it has mean half-life 92.29 B with 6144 dims > 8 B.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def half_lives(decay_logits):
    """Half-life in BYTES from the pre-sigmoid decay logits.

    sigma = sigmoid(logit) and half-life = log(0.5)/log(sigma). The naive
    form overflows for a saturated channel (log(sigma) -> 0), so log(sigma) is
    computed as log_sigmoid(logit) = -softplus(-logit) with
    np.logaddexp, which is stable for every finite logit.

    Returns (half_lives, sigma).
    """
    x = np.asarray(decay_logits, dtype=np.float64)
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))
    log_sigma = -np.logaddexp(0.0, -x)
    hl = np.log(0.5) / log_sigma
    return hl, sig


def report(name, logits):
    hl, sig = half_lives(logits)
    print(f"  {name:<12} n={len(hl):>6}  mean half-life {hl.mean():>9.2f} B  "
          f"median {np.median(hl):>8.2f} B  max {hl.max():>9.2f} B  "
          f"min {hl.min():>7.2f} B")
    print(f"  {'':<12} dims >8 B: {int((hl > 8).sum()):>6}   "
          f">16 B: {int((hl > 16).sum()):>6}   >64 B: {int((hl > 64).sum()):>6}"
          f"   sigma_max {sig.max():.6f}  sigma_mean {sig.mean():.4f}")
    return hl


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--per-layer", action="store_true",
                    help="print the per-layer distribution as well")
    args = ap.parse_args()

    d = np.load(args.checkpoint)
    names = sorted(k for k in d.files if k.startswith("layer") and
                   k.endswith(".decay"))
    if not names:
        print(f"{args.checkpoint}: no layer*.decay keys found")
        return 1
    per_layer = {}
    allhl = []
    for k in names:
        logits = d[k].ravel()
        per_layer[k] = logits
        allhl.append(half_lives(logits)[0])
    allhl = np.concatenate(allhl)
    print(f"checkpoint: {args.checkpoint}")
    print(f"  layers: {len(names)}   dims/layer: {len(per_layer[names[0]])}")
    report("ALL", np.concatenate([d[k].ravel() for k in names]))
    if args.per_layer:
        print("  per layer:")
        for k in names:
            report(k, per_layer[k])
    return 0

if __name__ == "__main__":
    sys.exit(main())
