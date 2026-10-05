#!/usr/bin/env python3
"""snr_vs_lr.py -- the cheap test of the OVER-LARGE-STEP diagnosis.

ROADMAP v3 P1 says the SNR blocker is a step-size problem, not a noise floor:
RESULTS.md measured the repo's own telemetry SNR at 0.633 with theta FROZEN
against 0.05 in training, a 12.6x gap, and attributes the gap to the iterate
moving further between updates than the gradient's spatial consistency length.

That diagnosis makes a FALSIFIABLE prediction before any training run:

    telemetry SNR should RISE MONOTONICALLY as lr falls.

If it does not, the over-large-step story is wrong and P1's whole global-LR
sweep is aimed at the wrong thing -- which is worth knowing before spending
8 h of Colab on it.

This is a continuation from the incumbent at t=919,897. It is DIAGNOSTIC ONLY:
it saves nothing, and the checkpoint is restored from the pinned file after
each arm. Nothing here is a result about the model; it is a result about the
diagnosis.

    ~/Projects/tmt/.venv-gpu/bin/python snr_vs_lr.py \
        --ckpt runs/lrroles/arm_lrdecay1x_t919897.npz --steps 600 --warmup 100
"""
import argparse
import json
import os
import sys
import time

# the engine modules live at the repo root, one level above tools/
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np
import torch

import gpu_port as G
import numpy_port as np_
import telemetry as T

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/lrroles/arm_lrdecay1x_t919897.npz")
ap.add_argument("--data", default="data/wikipedia_clean/**/wiki_*")
ap.add_argument("--B", type=int, default=1024)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--lrs", default="5e-4,2.5e-4,1.25e-4,6.25e-5,3.125e-5")
ap.add_argument("--device", default="cuda")
ap.add_argument("--ring-credit", default="auto")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data-seed", type=int, default=0)
ap.add_argument("--out", default="snr_vs_lr.json")
a = ap.parse_args()

lrs = [float(x) for x in a.lrs.split(",")]

import glob
import random
files = sorted(f for f in glob.glob(a.data, recursive=True) if os.path.isfile(f))
if not files:
    sys.exit(f"no corpus files matched {a.data}")

random.Random(a.data_seed).shuffle(files)
print(f"corpus: {len(files)} file(s), data_seed={a.data_seed}")

raw = open(files[0], "rb").read(64_000_000)
streams, endss = np_.Runtime.make_streams(raw, a.B)
L = min(len(s) for s in streams) - 1
mats = torch.from_numpy(np.stack(
    [np.frombuffer(s[:L + 1], dtype=np.uint8) for s in streams],
    axis=1).astype(np.int64)).to(a.device)
ets = torch.from_numpy(np.array([e[:L] for e in endss], dtype=np.float32).T).to(a.device)
print(f"data {len(raw):,}B -> L={L}\n")

theta_pinned = None
rows = []
for lr in lrs:
    torch.manual_seed(a.seed)
    m = G.GpuModel(dim=512, layers=16, device=a.device,
                   ring_credit=a.ring_credit, seed=a.seed, lr=lr)
    m.new_states(a.B)
    if not m.load(a.ckpt):
        sys.exit(f"could not load {a.ckpt}")
    if theta_pinned is None:
        theta_pinned = m.theta.detach().clone()
    else:
        m.theta.copy_(theta_pinned)     # identical starting point for every arm
    m.t = 1                            # reset the optimizer step counter

    eg = torch.zeros_like(m.theta)
    esq = torch.zeros_like(m.theta)
    t0 = time.perf_counter()
    traj = []
    for t in range(a.steps):
        loss = m.step_batch(mats[t], mats[t + 1], ets[t])
        _, snr = T.update_gradient_ema(m.grads, eg, esq)
        if t >= a.warmup and (t + 1) % 50 == 0:
            traj.append((t + 1, snr, float(loss)))
    dt = time.perf_counter() - t0
    post = [s for _, s, _ in traj]
    # the standing metric reads as ||EMA(g)||^2/EMA(||g||^2); report the warm mean
    snr_warm = float(np.mean(post)) if post else float("nan")
    med_step = float(m.theta.detach().abs().median())
    print(f"  lr={lr:<10.4g} SNR(warm, n={len(post)}) = {snr_warm:.4f}"
          f"   loss(med) = {np.median([l for _,_,l in traj]):.4f}"
          f"   {dt:.0f}s   {a.steps*a.B/dt:,.0f} coord-steps/s")
    rows.append({"lr": lr, "snr_warm": snr_warm, "sec": dt,
                 "loss_med": float(np.median([l for _, _, l in traj])),
                 "traj": traj})
    del m, eg, esq
    torch.cuda.empty_cache() if a.device == "cuda" else None

print()
print("=" * 68)
print("PREDICTION OF THE OVER-LARGE-STEP DIAGNOSIS: SNR rises as lr falls")
print("=" * 68)
print(f"{'lr':>12} {'SNR(warm)':>11} {'vs lr=5e-4':>12}")
base = rows[0]["snr_warm"]
for r in rows:
    print(f"{r['lr']:>12.4g} {r['snr_warm']:>11.4f} {r['snr_warm']/base:>11.2f}x")

# The diagnosis predicts SNR RISES as lr FALLS, so compare ascending lr
# against descending lr. (This was written the other way round first and
# printed the wrong branch on monotone data -- the exact failure mode this
# repo has shipped twice, so it is commented rather than silently fixed.)
mono = all(rows[i]["snr_warm"] < rows[i + 1]["snr_warm"] for i in range(len(rows) - 1))
print()
if mono:
    print("  -> MONOTONE. The over-large-step diagnosis survives its own falsifier.")
    print("     SNR rises as lr falls, exactly as predicted, with the params pinned")
    print("     to an identical start point in every arm.")
    print("     P1's global-LR sweep is aimed at a real effect.")
elif rows[-1]["snr_warm"] > base * 1.3:
    print("  -> NOT strictly monotone, but the smallest lr still clears the")
    print(f"     largest by {rows[-1]['snr_warm']/base:.2f}x. Prediction broadly holds;")
    print("     report the non-monotonicity rather than hiding it.")
else:
    print("  -> NOT confirmed. SNR does not improve as lr falls.")
    print("     The over-large-step diagnosis is FALSIFIED and P1's sweep would be")
    print("     aimed at the wrong thing. Do not spend Colab hours on it.")

with open(a.out, "w") as f:
    json.dump({"args": vars(a), "rows": rows, "monotone": bool(mono)}, f, indent=2)
print(f"\nwrote {a.out}")