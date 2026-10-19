#!/usr/bin/env python3
"""random_vs_real.py — the decisive A/B, using THEIR exact script as arm 1.

The other session measured, at frozen parameters on a 96-step gradient history:
    telemetry-EMA SNR 0.9637,  rho(1) = 0.516,  "the gradient is 96% signal"
and concluded the project's SNR~0.05 blocker is an artifact of step size.

Their script built its batch like this:
    g = torch.Generator().manual_seed(7)
    cs = torch.randint(1, 250, (STEPS + 1, B), generator=g)   # UNIFORM RANDOM BYTES
    ns = torch.randint(1, 250, (STEPS + 1, B), generator=g)

On uniform random bytes there is no local predictability to model, so the
network's response collapses onto a single constant direction -- "predict a byte
uniformly" -- which is identical at every timestep. That manufactures agreement
that is not there. The repo already wrote this warning down for the trace gate
(PLAN-2026-10-04-ROADMAP.md, P0 item 1: "with random bytes the states are iid
and every lag weighting collapses to the same direction, giving a confident
cosine 1.0000 for free").

This script runs THEIR configuration twice -- identical checkpoint, B, step count,
telemetry call, estimator -- changing exactly ONE thing: random bytes vs real
TinyStories bytes. Same frozen parameters. Nothing else.

  arm A  RANDOM bytes   (reproduces their 0.9637 / 0.516)
  arm B  REAL corpus    (the same measurement on data the model can actually model)

Usage:  .venv/bin/python random_vs_real.py [CKPT] [B] [STEPS]
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
    REPO, "runs/lrroles/arm_lrdecay1x_t919897.npz")
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 96

assert os.path.exists(CKPT), CKPT
raw = open(os.path.join(REPO, "data/eval/tinystories_valid.txt"), "rb").read(8_000_000)
print(f"checkpoint {os.path.basename(CKPT)}   B={B}  steps={STEPS}\n")


def run(tag, cs, ns, es):
    m = G.GpuModel(device="cpu", seed=0, ring_credit="full", lr=0.0)  # lr=0 => FROZEN
    assert m.load(CKPT)
    m.new_states(B)

    eg = torch.zeros_like(m.theta)
    esq = torch.zeros_like(m.theta)
    acc = torch.zeros(m.theta.numel(), dtype=torch.float64)
    pw = 0.0
    # rolling window for the autocorrelation, same lags they reported
    LAGS = [1, 2, 4, 8, 16, 32]
    W = max(LAGS) + 1
    window, lag_prod, lag_cnt = [], {l: None for l in LAGS}, {l: 0 for l in LAGS}
    mean, M2 = None, None
    rows = []
    t0 = time.perf_counter()
    for t in range(STEPS):
        m.step_batch(cs[t], ns[t], es[t])
        _, snr_ema = T.update_gradient_ema(m.grads, eg, esq)   # repo's own metric
        gv = m.grads.detach().to(torch.float64)
        acc += gv
        pw += float(gv.dot(gv))
        k = t + 1
        mk = acc / k
        rows.append((k, snr_ema, float(mk.dot(mk)) / (pw / k)))
        if mean is None:
            mean = torch.zeros_like(gv)
            M2 = torch.zeros_like(gv)
        delta = gv - mean
        mean += delta / k
        M2 += delta * (gv - mean)
        window.append(gv)
        if len(window) > W:
            window.pop(0)
        for l in LAGS:
            if len(window) > l:
                pr = window[-1 - l] * (window[-1] - mean)
                lag_prod[l] = pr if lag_prod[l] is None else lag_prod[l] + pr
                lag_cnt[l] += 1

    var = (M2 / STEPS)
    live = var > 1e-12
    print(f"=== arm {tag} ===   ({time.perf_counter()-t0:.0f}s)")
    print(f"  {'k':>5} {'telemetry EMA (b=0.9)':>24} {'plain k-average':>17}")
    for k, a, b in rows:
        if k in (1, 8, 32, 64, 96):
            print(f"  {k:>5} {a:>24.4f} {b:>17.4f}")
    rh = []
    for l in LAGS:
        if lag_cnt[l]:
            rh.append(float(((lag_prod[l] / lag_cnt[l]) / var.clamp_min(1e-30))[live].mean()))
    print(f"  rho(tau) per-coord: " + " ".join(f"{l}={r:+.3f}" for l, r in zip(LAGS, rh)))
    print()
    return rows[-1][1], rows[-1][2], rh


# ---- arm A: their exact construction, uniform random bytes ----------------
g = torch.Generator().manual_seed(7)
cs_r = torch.randint(1, 250, (STEPS + 1, B), generator=g)
ns_r = torch.randint(1, 250, (STEPS + 1, B), generator=g)
es_r = (torch.rand(STEPS, B, generator=g) < 0.02).float()
a_ema, a_k, a_rho = run("A  RANDOM bytes (their construction)", cs_r, ns_r, es_r)

# ---- arm B: identical, except real corpus bytes --------------------------
streams, endss = np_.Runtime.make_streams(raw, B)
L = min(len(s) for s in streams) - 1
L = min(L, STEPS)
cs_t = torch.from_numpy(np.stack(
    [np.frombuffer(s[:L + 1], dtype=np.uint8) for s in streams],
    axis=1).astype(np.int64))
ns_t = cs_t[1:]
cs_t = cs_t[:L]
et = np.array([e[:L] for e in endss], dtype=np.float32).T
es_t = torch.from_numpy(et)
b_ema, b_k, b_rho = run("B  REAL TinyStories bytes (one variable changed)", cs_t, ns_t, es_t)

print("=" * 72)
print("VERDICT")
print("=" * 72)
print(f"  telemetry-EMA SNR at k=96:  random {a_ema:.4f}   real {b_ema:.4f}"
      f"   ratio {a_ema/max(b_ema,1e-9):.2f}x")
print(f"  plain k-average  at k=96:  random {a_k:.4f}   real {b_k:.4f}"
      f"   ratio {a_k/max(b_k,1e-9):.2f}x")
print(f"  rho(1) per-coordinate:     random {a_rho[0]:+.4f}   real {b_rho[0]:+.4f}")
print()
if a_ema > 3 * b_ema:
    print("  -> The high SNR and the high rho are produced by RANDOM BYTES, not by the")
    print("     gradient. On real text the same estimator, same frozen parameters,")
    print("     same everything-else gives the low number. The '96% self-consistent'")
    print("     claim is a data artifact and the repo's existing warning applies.")
else:
    print("  -> Random-vs-real barely differs. The gap must come from something else")
    print("     (checkpoint, B, or step count) and needs a second pass.")