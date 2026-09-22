#!/usr/bin/env python3
"""Report how many dims have short / medium / long memory.

Each dim's decay parameter sets how long its channel remembers:
    half-life (bytes) = ln(2) / -ln(sigmoid(decay))
A dim at the init (sigmoid ~0.5) remembers ~1 byte. This script bins every
dim of a checkpoint into fixed half-life categories, so running it
periodically shows mass migrating from short to long memory.

Also reports what the current ring truncation (K lags) costs: the share of
each dim's embed-credit mass that falls past lag K (equals sigma^K per dim).

usage: python memory_report.py CKPT.npz [CKPT2.npz ...]
       (multiple checkpoints print side-by-side columns for comparison)
"""
import os
import sys

import numpy as np

# fixed half-life edges (bytes) so runs are comparable over time
CATEGORIES = [
    ("very short   <1B", 0.0, 1.0),
    ("short       1-2B", 1.0, 2.0),
    ("medium      2-8B", 2.0, 8.0),
    ("long       8-32B", 8.0, 32.0),
    ("very long   >32B", 32.0, float("inf")),
]


def layer_count(d):
    return 1 + max(int(k.split(".")[0][5:]) for k in d.files
                   if k.startswith("layer") and k.split(".")[1] == "decay")


def tail_cols(d, L):
    """Hot dims covered by the tail accumulator (tail_credit.py), per layer."""
    cols = {}
    for i in range(L):
        k = f"tail_cols.{i}"
        if k in d:
            cols[i] = np.asarray(d[k], int)
    return cols


def load(path):
    d = np.load(path)
    L = layer_count(d)
    sig = np.concatenate([1 / (1 + np.exp(-d[f"layer{i}.decay"])) for i in range(L)])
    hl = np.log(2) / -np.log(np.clip(sig, 1e-12, 1 - 1e-12))
    t = int(d["t"]) if "t" in d else -1
    # ring length is a parameter fixed at Model.__init__ (trainer's TMT_RING_TOL);
    # newer checkpoints record it. Older ones: fall back to the env guess.
    K = int(d["ring_k"]) if "ring_k" in d else ring_k()
    return path, t, L, sig, hl, tail_cols(d, L), K


def ring_k():
    """The K the port would use right now (init decay sizes the ring)."""
    tol = float(os.environ.get("TMT_RING_TOL", "1e-4"))
    return int(min(256, np.ceil(np.log(tol) / np.log(0.5115))))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    reports = [load(p) for p in sys.argv[1:]]

    names = [f"{r[0].split('/')[-1]} (t={r[1]:,})" for r in reports]
    print("memory report")
    for path, t, L, sig, hl, tcols, K in reports:
        print(f"  {path}: t={t:,}  {sig.size} dims  {L} layers")
    K0 = reports[0][-1]
    print(f"\n  categories (by half-life; ring K read from each checkpoint):")
    w = max(len(n) for n in names) + 2
    hdr = f"  {'category':<18}" + "".join(f"{n:>{w}}" for n in names)
    print(hdr)
    counts = {}
    for label, lo, hi in CATEGORIES:
        row = f"  {label:<18}"
        for _, _, _, _, hl, _, _ in reports:
            n = int(((hl >= lo) & (hl < hi)).sum())
            counts.setdefault(label, []).append(n)
            row += f"{n:>{w}}"
        print(row)

    # bars for the last checkpoint (usually the current one)
    print(f"\n  distribution of the last checkpoint:")
    last = [ns[-1] for ns in counts.values()]
    mx = max(last) or 1
    for (label, _, _), n in zip(CATEGORIES, last):
        print(f"  {label:<18} {n:>7,}  {'#' * max(1, round(48 * n / mx)) if n else ''}")

    print(f"\n  longest-memory dims (last checkpoint):")
    path, t, L, sig, hl, _, _ = reports[-1]
    order = np.argsort(hl)[::-1][:5]
    for i in order:
        layer = i // (sig.size // L)
        print(f"    layer{layer:<3d} sigma={sig[i]:.4f}  half-life {hl[i]:6.2f} bytes")

    print(f"\n  ring truncation cost (share of embed-credit mass dropped, = sigma^K):")
    print("    (dims covered by the tail accumulator are exact; cost shown is cold-only)")
    for path, t, L, sig, hl, tcols, K in reports:
        dim = sig.size // L
        dropped = sig ** K
        if tcols:
            mask = np.zeros(sig.size, bool)
            for i, c in tcols.items():
                if c.size:
                    mask[i * dim + c] = True
            covered = int(mask.sum())
            dropped = np.where(mask, 0.0, dropped)
            extra = f" tail-covered dims: {covered:>3}"
        else:
            extra = (" would-adopt: %3d" % int((dropped > 0.01).sum())
                     + "  (pre-tail checkpoint)")
        hot = int((dropped > 0.01).sum())
        print(f"    {path.split('/')[-1]:<22} K={K:>3}  total {dropped.mean():7.4%}   "
              f"dims dropping >1%: {hot:>3}   worst dim: {dropped.max():.2%}"
              f"{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
