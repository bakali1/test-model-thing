#!/usr/bin/env python3
"""reconcile_snr.py — ONE capture, EVERY convention, so the two threads can be compared.

Two sessions measured "SNR at frozen parameters" and got numbers 5x apart:

  thread A (rho_probe.py)        frozen theta, 96 steps -> variance SNR 0.174
  thread B (other session)       frozen theta, 96 steps -> 0.8525 (plain avg)
                                                            0.9637 (EMA b=0.9)
and autocorrelation 4x apart:

  thread A   rho(1) = +0.136   (aggregate vector autocorrelation)
  thread B   rho(1) =  0.52    (per-coordinate autocorrelation)

Both are probably right about DIFFERENT quantities. Rather than argue, this
script captures ONE gradient history at ONE frozen theta and prints every
convention side by side, so the disagreement resolves into a definition table.

SNR conventions computed (all from the same hist):
  S1 repo/telemetry : ||EMA_b(g)||^2 / EMA_b(||g||^2)      UNBIAS-CORRECTED
  S2 same, bias-corrected by (1-b^T)
  S3 plain mean     : ||mean_T g||^2 / mean_T(||g||^2)
  S4 EMA vs plain denominator : ||EMA_b(g)||^2 / mean_T(||g||^2)
  S5 per-coordinate : mean_i[ mean_t(g_it)^2 / (mean_t(g_it^2)) ]     <- "consistency"
  S6 aggregate energy share : ||mean_T g||^2 / (1/T)*sum_t ||g_t||^2   = S3

rho conventions:
  R1 aggregate : <g_t, g_{t+tau}> / <g_t, g_t>
  R2 per-coord : mean_i [ <g_it, g_i,t+tau>_t / <g_it, g_it>_t ]
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch

import gpu_port as G
import numpy_port as np_
import telemetry as T

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "runs/mx150_nvrtc/tmt_t101847.npz")
B = int(sys.argv[2]) if len(sys.argv) > 2 else 128
NSTEP = int(sys.argv[3]) if len(sys.argv) > 3 else 96
WARM = 20
B1 = 0.9

torch.manual_seed(0)
model = G.GpuModel(dim=512, layers=16, device="cpu", ring_credit="full", seed=0)
model.new_states(B)
assert model.load(CKPT)
print(f"loaded t={model.t} B={model.B} ring_k={model.ring_k}  "
      f"({os.path.basename(CKPT)})")

raw = open(os.path.join(REPO, "data/eval/tinystories_valid.txt"), "rb").read(6_000_000)
streams, endss = np_.Runtime.make_streams(raw, B)
L = min(len(s) for s in streams) - 1
mats = torch.from_numpy(np.stack(
    [np.frombuffer(s[:L + 1], dtype=np.uint8) for s in streams],
    axis=1).astype(np.int64))
ets = torch.from_numpy(np.array([e[:L] for e in endss], dtype=np.float32).T)

theta0 = model.theta.detach().clone()
print(f"capturing {NSTEP} gradients at ONE frozen theta ...\n")

hist = []
t0 = time.perf_counter()
for t in range(WARM + NSTEP):
    model.theta.copy_(theta0)
    model.step_batch(mats[t], mats[t + 1], ets[t])
    if t >= WARM:
        hist.append(model.grads.detach().clone())
model.theta.copy_(theta0)
print(f"captured in {time.perf_counter()-t0:.0f}s\n")

n = model.theta.numel()
S = torch.stack([h.pow(2).sum() for h in hist])       # ||g_t||^2 per step
Smean = float(S.mean())

print("=" * 78)
print("A. SNR UNDER EVERY CONVENTION  (one history, frozen theta)")
print("=" * 78)
print(f"{'T':>4} | {'S1 repo EMA':>11} {'S2 EMA corr':>11} {'S3 plain mean':>13} "
      f"{'S4 EMA/mean||g||^2':>16} {'S5 per-coord':>12}")
print("-" * 78)
for TT in (8, 32, 96, len(hist)):
    if TT > len(hist):
        continue
    hs = hist[:TT]
    m_ema = torch.zeros(n)
    v_ema = torch.zeros(n)
    for g in hs:
        m_ema.mul_(B1).add_(g, alpha=1 - B1)
        v_ema.mul_(B1).add_(g * g, alpha=1 - B1)
    bc1 = 1 - B1 ** TT
    s1 = float(m_ema.pow(2).sum()) / float(v_ema.sum())
    s2 = float((m_ema / bc1).pow(2).sum()) / float(v_ema.sum())
    gm = torch.stack(hs).mean(0)
    s3 = float(gm.pow(2).sum()) / Smean
    s4 = float(m_ema.pow(2).sum()) / Smean
    G2 = torch.stack([g * g for g in hs])
    per = float(((G2.mean(0)) .pow(0.5)).new_tensor(0))  # placeholder
    mu = G2.new_zeros(n)
    mu = torch.stack(hs).mean(0)
    s5 = float((mu.pow(2) / (G2.mean(0) + 1e-30)).mean())
    print(f"{TT:>4} | {s1:>11.4f} {s2:>11.4f} {s3:>13.4f} {s4:>16.4f} {s5:>12.4f}")

print()
print("S1/S2 = the repo's own telemetry metric (telemetry.py:27-40), uncorrected / corrected")
print("S3    = fraction of gradient ENERGY carried by the time-average")
print("S5    = mean over coordinates of that coordinate's own consistency")
print("NOTE S3/S5 are bounded by 1 and mean 'the time-average is this good'.")
print("     They are NOT the same quantity as the repo's ratio-of-EMAs metric.")

print()
print("=" * 78)
print("B. AUTOCORRELATION UNDER EVERY NORMALISATION")
print("=" * 78)
TAU_MAX = 12
print(f"{'tau':>4} | {'R1 aggregate':>13} {'R2 per-coord':>13}")
print("-" * 78)
for tau in range(1, TAU_MAX + 1):
    num = sum(float(torch.dot(hist[i], hist[i + tau]))
              for i in range(len(hist) - tau))
    den = sum(float(hist[i].pow(2).sum()) for i in range(len(hist) - tau))
    r1 = num / den if den else float("nan")
    hs2 = [h for h in hist]
    a = torch.stack([hs2[i] for i in range(len(hs2) - tau)])
    b = torch.stack([hs2[i + tau] for i in range(len(hs2) - tau)])
    numv = (a * b).sum(0)
    denv = (a * a).sum(0)
    r2 = float((numv / (denv + 1e-30)).mean())
    print(f"{tau:>4} | {r1:>13.4f} {r2:>13.4f}")

print()
print("R1: the SHARED component of the gradient (common mode) -- this is what")
print("    averaging over timesteps can actually accumulate.")
print("R2: each coordinate's OWN persistence -- this is what a per-coordinate")
print("    metric like 'median|m|/RMS(g)' is sensitive to.")
print("A large R2 with a small R1 = coordinate noise that averages away in the")
print("vector sum, which is why the two threads disagree.")

print()
print("=" * 78)
print("C. ACCUMULATION EFFICIENCY (what an accumulation ladder would actually buy)")
print("=" * 78)
print(f"{'k':>5} | {'R1 eff':>8} {'R2 eff':>8}   (efficiency = achieved / ideal 1/k)")
print("-" * 78)
for k in (2, 4, 8, 16, 32, 64, 128, 256):
    if k + 2 > len(hist):
        break
    acc1 = acc2 = 0.0
    for s in range(0, len(hist) - k, max(1, (len(hist) - k) // 12)):
        w = torch.stack(hist[s:s + k])
        gm = w.mean(0)
        # achieved variance reduction of the mean, relative to a single step
        e1 = float(gm.pow(2).sum()) / float(w.pow(2).sum().mean() / k)
        per = (w.pow(2).mean(0))
        e2 = float(((gm.pow(2) / (per + 1e-30)).mean()) * k)
        acc1 += min(e1, 1.0); acc2 += min(e2, 1.0)
    cnt = len(range(0, len(hist) - k, max(1, (len(hist) - k) // 12)))
    print(f"{k:>5} | {acc1/cnt:>8.3f} {acc2/cnt:>8.3f}")
print()
print("efficiency 1.0 = averaging over k consecutive steps gives you the full")
print("ideal 1/k noise reduction. Lower means the steps are correlated.")