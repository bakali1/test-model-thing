#!/usr/bin/env python3
"""measure_m6_selectivity.py -- M6: IS THE WRITE SELECTIVE?

The P3 design's most diagnostic test.  The D5 finding is that a write
OVERWRITES or dilutes what is already in the state: the ADDRESS persists, the
CONTENT decays.  If the gate makes the write SELECTIVE -- i.e. if it
suppresses a write to a channel that already holds a different fact -- then
the OLD fact's content should survive a CONFLICT better than a RANDOM
write, and the gain over the ungated arm should be CONCENTRATED in the
conflicting case.

THE THREE CONDITIONS (the geometry is the trainer's own: 4 facts, the
trainer's gap, the cue, the behavioural target -- only the cue/target point
at the OLDEST fact, and only the NEWEST fact's key changes):

  no-conflict   the 4th (most recent) fact carries a RANDOM key: nothing
                contests the target's channel.
  random-key    the same, with the 4th fact's key drawn at random -- the
                control for "a recent write happened at all".
  confusable    the 4th fact carries the TARGET'S OWN KEY with a different
                payload: the maximally confusable distractor (same cue, same
                key, different payload).

The quantity that decides the hypothesis is the CONFLICT COST

    conflict cost = recall(no-conflict) - recall(confusable)

and the hypothesis is that the GATED arm's conflict cost is LOWER (it
suppresses the contested write) -- i.e. that the gain over the ungated arm
is concentrated in the confusable condition.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gpu_port
import cl2_train as cl2
import cl_ruler as ruler

F = torch.float32


def build_model(ckpt, device):
    d = np.load(ckpt, allow_pickle=False)
    dim = int(d["Wd"].shape[1])
    layers = sum(1 for k in d.files if k.startswith("layer") and k.endswith(".W"))
    gated = any(k.endswith(".gate") for k in d.files)
    m = gpu_port.GpuModel(dim=dim, layers=layers, lr=5e-4, seed=0,
                          device=device, ring_k=8, ring_credit="full",
                          gated=gated)
    with torch.no_grad():
        for name, view in m.views.items():
            if name in d.files and d[name].dtype == np.float32:
                view.copy_(torch.from_numpy(d[name]).to(view.device))
    Wm = torch.from_numpy(d["Wm"].astype(np.float32)).to(device)
    bm = torch.from_numpy(d["bm"].astype(np.float32)).to(device)
    return m, Wm, bm, gated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphabet", default="16")
    ap.add_argument("--facts", type=int, default=4)
    ap.add_argument("--gap", type=int, default=32)
    ap.add_argument("--board-seed-a", type=int, default=101)
    ap.add_argument("--board-size", type=int, default=512)
    ap.add_argument("--text-frac", type=float, default=0.5)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    label = args.label or os.path.basename(args.ckpt)

    m, Wm, bm, gated = build_model(args.ckpt, args.device)
    print(f"M6 -- is the write SELECTIVE?  {label}  ({args.ckpt})")
    print(f"  gated={gated}  dim={m.dim} layers={len(m.layers)} B={args.B} "
          f"gap={args.gap} facts={args.facts}")

    alpha = cl2.alphabet_symbols(args.alphabet)
    board = cl2.make_board(args.board_seed_a, args.board_size, alpha)
    import glob
    data_dir = args.data_dir or os.path.join(ROOT, "data")
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    pool = cl2.TextPool(files, seed=args.seed) if files else None
    lay = cl2.sample_layout(args.facts, args.gap)
    rng = np.random.default_rng(args.seed)

    # the stream: 4 facts from the board, the trainer's gap, the cue and the
    # behavioural target -- but the cue/target point at the OLDEST fact.
    B = args.B
    idx = rng.integers(0, board["keys"].shape[0], size=(B, args.facts))
    keys = board["keys"][idx].copy()
    pays = board["payloads"][idx].copy()
    seq = np.zeros((B, lay["length"]), np.uint8)
    facts = np.concatenate([keys, pays], axis=2)
    seq[:, lay["facts0"]:lay["gap0"]] = facts.reshape(B, -1)
    gap = np.empty((B, args.gap), np.uint8)
    if pool is not None:
        gap[:] = pool.windows(B, args.gap, rng)
    else:
        gap[:] = rng.integers(0, 256, size=(B, args.gap)).astype(np.uint8)
    seq[:, lay["gap0"]:lay["gap1"]] = gap
    target = pays[:, 0].astype(np.int64)          # the OLDEST fact's payload
    cue_key = keys[:, 0].copy()                    # the OLDEST fact's key

    out = {}
    for cond in ("no-conflict", "random-key", "confusable"):
        s = seq.copy()
        k = keys.copy()
        if cond == "confusable":
            # the most recent fact carries the TARGET'S KEY and a different
            # payload: the maximally confusable distractor.
            k[:, -1] = cue_key
            alt = pays[:, -1].copy()
            alt[:, 0] = (alt[:, 0] + 1) % 16
            p = np.concatenate([k, alt], axis=2)
            s[:, lay["facts0"]:lay["gap0"]] = p.reshape(B, -1)
        elif cond == "random-key":
            k[:, -1] = rng.integers(0, 16, size=B)
            p = np.concatenate([k, pays], axis=2)
            s[:, lay["facts0"]:lay["gap0"]] = p.reshape(B, -1)
        s[:, lay["cue0"]:lay["cue1"]] = cue_key
        with torch.no_grad():
            _logits, state_cue = cl2.engine_forward(m, s, lay["cue_last"])
        X = state_cue.detach().cpu().numpy().astype(np.float64)
        W = Wm.detach().cpu().numpy().astype(np.float64)
        bb = bm.detach().cpu().numpy().astype(np.float64)
        cell = ruler.make_head_cell(cond, X, target, W, bb, args.seed,
                                    16 if alpha is None else len(alpha))
        out[cond] = cell
        print(f"  {cond:<12} per_byte {cell['per_byte']:.4f}  exact4 "
              f"{cell['exact4']:.4f}  null_exact4 {cell['null_exact4']:.4f}  "
              f"verdict {cell['verdict']}")
    nc = out["no-conflict"]["exact4"]
    cf = out["confusable"]["exact4"]
    print(f"  -> the CONFLICT COST (exact4): no-conflict {nc:.4f} - "
          f"confusable {cf:.4f} = {nc - cf:+.4f}")
    print(f"  -> per byte: {out['no-conflict']['per_byte']:.4f} - "
          f"{out['confusable']['per_byte']:.4f} = "
          f"{out['no-conflict']['per_byte'] - out['confusable']['per_byte']:+.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": label, "ckpt": args.ckpt, "gated": gated,
                       "cells": {k: {kk: vv for kk, vv in v.items()
                                     if kk in ("per_byte", "exact4",
                                               "null_exact4", "verdict")}
                                 for k, v in out.items()}}, f, indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
