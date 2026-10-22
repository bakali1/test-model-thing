#!/usr/bin/env python3
"""measure_m6_selectivity.py -- M6: IS THE WRITE SELECTIVE, AND WHAT DOES
THE STATE HOLD?

The programme's synthesis (RESULTS.md 21) says two different things are
being measured and they must never be mixed:

  * the TRAINED HEAD is the model's OWN parameter: it is what the model
    LEARNED.  The programme has shown it is a TABLE (key -> payload in the
    weights): replacing the payload bytes IN THE INPUT leaves its answer
    intact.  It is the honest measure of what was learned -- and it is NOT a
    measure of the state.
  * a FRESH RIDGE readout, fitted on the state, measures the state's
    COPYABILITY: whether the state still holds a linear-readable copy of a
    payload.  It is REFITTED, so it must never be reported as retention.

M6 asks the question that separates the two stores and tests the gate's
mechanism: when a second write contests the first one's channel, what
happens?

  conditions (the trainer's own geometry: 4 facts, the trainer's gap, the
  cue, the behavioural target; only the cue/target point at the OLDEST fact
  and only the 4th fact's key changes):

    no-conflict  the 4th (most recent) fact carries a random key: nothing
                 contests the target's channel.
    random-key   the same, with the 4th fact's key drawn at random: the
                 control for "a recent write happened at all".
    confusable   the 4th fact carries the TARGET'S OWN KEY with a different
                 payload: the maximally confusable distractor (same cue, same
                 key, different payload).

  and beside every condition, the COPY CONTROL: the target's PAYLOAD bytes
  in the INPUT are replaced (the key is kept).  If the head's answer is
  unchanged, the head read the KEY and looked the payload up in the weights
  (a table); if the ridge's answer follows the replacement, the state holds
  a COPY of the input.

THE DECISION THIS TOOL FEEDS
    Does the write SELECT (the target survives a conflict) or does every
    write cost the same (a global scaling)?  The number is the CONFLICT COST
    = recall(no-conflict) - recall(confusable), and the gate's version of it
    must be LOWER than the ungated arm's.
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
import membench.model_backend as mb

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
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    label = args.label or os.path.basename(args.ckpt)
    alpha = cl2.alphabet_symbols(args.alphabet)
    alpha_size = 256 if alpha is None else len(alpha)

    m, Wm, bm, gated = build_model(args.ckpt, args.device)
    print(f"M6 -- is the write SELECTIVE?  {label}")
    print(f"  ckpt {args.ckpt}  gated={gated}  dim={m.dim} "
          f"layers={len(m.layers)}  B={args.B}  gap={args.gap}")

    board = cl2.make_board(args.board_seed_a, args.board_size, alpha)
    import glob
    data_dir = args.data_dir or os.path.join(ROOT, "data")
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    pool = cl2.TextPool(files, seed=args.seed) if files else None
    lay = cl2.sample_layout(args.facts, args.gap)
    rng = np.random.default_rng(args.seed)
    B = args.B
    idx = rng.integers(0, board["keys"].shape[0], size=(B, args.facts))
    keys = board["keys"][idx].copy()
    pays = board["payloads"][idx].copy()
    base = np.zeros((B, lay["length"]), np.uint8)
    base[:, lay["facts0"]:lay["gap0"]] = np.concatenate(
        [keys, pays], axis=2).reshape(B, -1)
    if pool is not None:
        base[:, lay["gap0"]:lay["gap1"]] = pool.windows(B, args.gap, rng)
    else:
        base[:, lay["gap0"]:lay["gap1"]] = rng.integers(
            0, 256, size=(B, args.gap)).astype(np.uint8)
    target = pays[:, 0].astype(np.int64)          # the OLDEST fact's payload
    cue_key = keys[:, 0].copy()                    # the OLDEST fact's key

    def build(cond, copy_control):
        s = base.copy()
        k = keys.copy()
        p = pays.copy()
        if cond == "confusable":
            k[:, -1] = cue_key                      # the SAME cue, new payload
            p[:, -1, 0] = (p[:, -1, 0] + 1) % alpha_size
        elif cond == "random-key":
            k[:, -1] = rng.integers(0, alpha_size,
                                    size=(B, 4)).astype(keys.dtype)
        if copy_control:
            # the COPY CONTROL: the target's PAYLOAD bytes in the input are
            # replaced, the key is kept.
            p[:, 0, :] = (p[:, 0, :] + 1) % alpha_size
        s[:, lay["facts0"]:lay["gap0"]] = np.concatenate(
            [k, p], axis=2).reshape(B, -1)
        s[:, lay["cue0"]:lay["cue1"]] = cue_key
        return s, p

    out = {}
    for cond in ("no-conflict", "random-key", "confusable"):
        for cp in (False, True):
            s, p = build(cond, cp)
            with torch.no_grad():
                logits, state_cue = cl2.engine_forward(m, s, lay["cue_last"])
            # 1) the model's OWN behaviour: the next-byte prediction at the
            #    four slots after the cue, against the TARGET's payload.
            b0, b1 = lay["cue1"] - 1, lay["tgt1"] - 1
            pred = logits[:, b0:b1].argmax(-1).cpu().numpy()
            behav_pb = float((pred == target).mean())
            behav_ex = float((pred == target).all(axis=1).mean())
            # 2) the TRAINED HEAD on the state at the cue (the model's own
            #    parameter; what it LEARNED).
            X = state_cue.detach().cpu().numpy().astype(np.float64)
            W = Wm.detach().cpu().numpy().astype(np.float64)
            bb = bm.detach().cpu().numpy().astype(np.float64)
            head = ruler.make_head_cell("head", X, target, W, bb, args.seed,
                                         alpha_size)
            # 3) a FRESH RIDGE on the state: the state's COPYABILITY.  Fitted
            #    on half the rows, tested on the other half.  Reported for
            #    BOTH the target's payload and the 4th fact's (the most
            #    recent write): which payload does the state hold?
            tr, te = slice(0, B // 2), slice(B // 2, B)
            y_t = target.astype(np.int64)
            y_d = p[:, -1].astype(np.int64)
            ridge_t = mb._ridge_score(X[tr], y_t[tr], X[te], y_t[te],
                                      seed=args.seed)
            ridge_d = mb._ridge_score(X[tr], y_d[tr], X[te], y_d[te],
                                      seed=args.seed)
            key = f"{cond}{'/copy-control' if cp else ''}"
            out[key] = {
                "behav_per_byte": behav_pb, "behav_exact4": behav_ex,
                "head_per_byte": head["per_byte"],
                "head_exact4": head["exact4"],
                "ridge_target_exact4": ridge_t[0],
                "ridge_target_per_byte": ridge_t[1],
                "ridge_distractor_exact4": ridge_d[0],
                "ridge_distractor_per_byte": ridge_d[1],
            }
            print(f"  {key:<28} BEHAVIOUR {behav_pb:.4f}/{behav_ex:.4f} | "
                  f"HEAD {head['per_byte']:.4f}/{head['exact4']:.4f} | "
                  f"RIDGE(state) target {ridge_t[1]:.4f}/{ridge_t[0]:.4f} "
                  f"distractor {ridge_d[1]:.4f}/{ridge_d[0]:.4f}")
    for pre in ("behav_exact4", "head_exact4", "ridge_target_exact4"):
        nc = out["no-conflict"][pre]
        rf = out["random-key"][pre]
        cf = out["confusable"][pre]
        print(f"  CONFLICT COST [{pre}]: no-conflict {nc:.4f} - random-key "
              f"{rf:.4f} = {nc - rf:+.4f} | - confusable {cf:.4f} = "
              f"{nc - cf:+.4f}")
    print("  (per_byte / exact4 in every cell)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": label, "ckpt": args.ckpt, "gated": gated,
                       "cells": out}, f, indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
