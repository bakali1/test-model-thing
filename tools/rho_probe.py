#!/usr/bin/env python3
"""rho_probe.py — is the 3.8% momentum-retention statistic real, or an artifact?

per_coord_grad.py reports `median|m| / RMS(g) = 3.8%` and RESULTS.md reads it as
"the gradient direction flips nearly every step". That measurement calls
`model.step_batch(...)` on every iteration, and step_batch ENDS in an optimizer
update (gpu_port.py:945-957, `th.sub_(...)`). So the 30 captured gradients were
taken at 30 DIFFERENT parameter values, each ~1.1e-5 from the last. The statistic
therefore mixes two things that must not be mixed:

  (1) data noise at fixed theta            -> what the SNR blocker is about
  (2) J(theta) . dtheta from parameter motion -> what the optimizer is doing

This script separates them. Mode A restores theta after every step_batch, so the
gradient is always evaluated at ONE fixed point. Mode B is the contaminated
original. Then it reports the analytic nulls so the numbers can be read.

Nulls for the statistic `median|m| / RMS(g)`, beta1=0.9, m near-Gaussian:
  iid zero-mean g      -> 0.6745 * sqrt((1-b1)/(1+b1)) = 0.6745 * 0.2294 = 0.1547
  perfect period-2     -> 0.6745 * ((1-b1)/(1+b1))     = 0.6745 * 0.0526 = 0.0355
                         (m_t = c*g_t, g_t = -g_{t-1}  =>  c = (1-b1)/(1+b1))

Usage:
  .venv/bin/python rho_probe.py [CKPT] [B] [CAP]
"""
import os
import sys
import time

REPO = "/home/baku/programing/opensource/new-llm/test-model-thing"
sys.path.insert(0, REPO)

import numpy as np
import torch

import gpu_port as G
import numpy_port as np_
import telemetry as T

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "runs/mx150_nvrtc/tmt_t101847.npz")
B = int(sys.argv[2]) if len(sys.argv) > 2 else 128
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 160
WARM = 20
TAU_MAX = 96
LR = 5e-4
B1, B2, EPS = 0.9, 0.999, 1e-8

torch.manual_seed(0)
model = G.GpuModel(dim=512, layers=16, device="cpu", ring_credit="full", seed=0)
model.new_states(B)
assert model.load(CKPT), f"could not load {CKPT}"
print(f"loaded t={model.t} B={model.B} ring_k={model.ring_k} from "
      f"{os.path.basename(CKPT)}")

raw = open(os.path.join(REPO, "data/eval/tinystories_valid.txt"), "rb").read(6_000_000)
streams, endss = np_.Runtime.make_streams(raw, B)
L = min(len(s) for s in streams) - 1
mats = torch.from_numpy(np.stack(
    [np.frombuffer(s[:L + 1], dtype=np.uint8) for s in streams],
    axis=1).astype(np.int64))
ets = torch.from_numpy(np.array([e[:L] for e in endss], dtype=np.float32).T)
print(f"data {len(raw):,}B -> L={L}  warmup {WARM} capture {CAP}  tau<={TAU_MAX}\n")

n = model.theta.numel()
theta0 = model.theta.detach().clone()


def capture(freeze):
    """Collect CAP gradient vectors. freeze=True pins theta for every step."""
    model.theta.copy_(theta0)
    m_ema = torch.zeros(n)
    hist = []
    for t in range(WARM + CAP):
        if freeze:
            model.theta.copy_(theta0)
        model.step_batch(mats[t], mats[t + 1], ets[t])
        g = model.grads.detach().clone()
        if t >= WARM:
            m_ema.mul_(B1).add_(g, alpha=1 - B1)
            hist.append(g)
    model.theta.copy_(theta0)
    return hist, m_ema / (1 - B1 ** CAP)


def autocorr(hist, tau_max):
    """Time-averaged autocorrelation of the whole gradient vector, and per group."""
    C = len(hist)
    norms = [float(h.pow(2).sum()) for h in hist]
    mean_norm = sum(norms) / C
    out = []
    for tau in range(1, tau_max + 1):
        s = sum(float(torch.dot(hist[i], hist[i + tau]))
                for i in range(C - tau))
        d = sum(norms[i] for i in range(C - tau))
        out.append(s / d if d else float("nan"))
    return out, mean_norm


groups = T.RunTelemetry._groups(None, model)
DECAY = [(a, b) for (nm, a, b) in groups if nm != "shared_trunk"]
DEC = torch.zeros(n, dtype=torch.bool)
for a, b in DECAY:
    DEC[a:b] = True


def group_rho(hist, mask, tau_max):
    hs = [h[mask] for h in hist]
    C = len(hs)
    norms = [float(h.pow(2).sum()) for h in hs]
    out = []
    for tau in range(1, tau_max + 1):
        s = sum(float(torch.dot(hs[i], hs[i + tau])) for i in range(C - tau))
        d = sum(norms[i] for i in range(C - tau))
        out.append(s / d if d else float("nan"))
    return out


IID = 0.6745 * np.sqrt((1 - B1) / (1 + B1))
P2 = 0.6745 * ((1 - B1) / (1 + B1))

results = {}
for tag, freeze in (("A  FROZEN theta", True), ("B  theta MOVES (the original)", False)):
    t0 = time.perf_counter()
    hist, mc = capture(freeze)
    stats = {}
    gsq = sum(float(h.pow(2).sum()) for h in hist) / len(hist)
    rms_g = np.sqrt(gsq / n)
    stat = float(mc.abs().median()) / rms_g
    rho, _ = autocorr(hist, TAU_MAX)
    rho_dec = group_rho(hist, DEC, TAU_MAX)
    # variance-based (repo convention) vs amplitude SNR
    var_snr = float(mc.pow(2).sum()) / gsq
    print(f"=== mode {tag} ===   ({time.perf_counter()-t0:.0f}s)")
    print(f"  median|m|/RMS(g)      = {stat:.4f}"
          f"    iid null {IID:.4f} | period-2 null {P2:.4f}"
          f" | stat/iid {stat/IID:.3f} | stat/period-2 {stat/P2:.3f}")
    print(f"  variance SNR (repo)   = {var_snr:.4f}"
          f"   -> amplitude SNR = sqrt = {np.sqrt(var_snr):.4f}")
    print(f"  rho(tau), ALL coords  = "
          + " ".join(f"{r:+.3f}" for r in rho))
    print(f"  rho(tau), DECAY coords= "
          + " ".join(f"{r:+.3f}" for r in rho_dec))
    print()
    # per-group variance SNR on THIS mode's history
    pg = []
    for nm, a_, b_ in groups:
        gq = sum(float(h[a_:b_].pow(2).sum()) for h in hist) / len(hist)
        v_ = float(mc[a_:b_].pow(2).sum()) / gq if gq else float("nan")
        pg.append((nm, b_ - a_, v_))
    print("  per-group variance SNR (this mode's history):")
    seen = {}
    for nm, cnt, v_ in pg:
        seen.setdefault(nm, []).append((cnt, v_))
    for nm, lst in seen.items():
        print(f"    {nm:16s} x{len(lst):<3d} {lst[0][0]:>9,} params each"
              f"   var SNR {min(v for _, v in lst):.4f}-{max(v for _, v in lst):.4f}"
              f"   amp {min(np.sqrt(max(v,0)) for _, v in lst):.4f}"
              f"-{max(np.sqrt(max(v,0)) for _, v in lst):.4f}")
    print()
    results[tag] = (stat, rho, rho_dec, var_snr)

a, b = results["A  FROZEN theta"], results["B  theta MOVES (the original)"]
print("=== does parameter motion explain the 3.8%? ===")
print(f"  statistic: frozen {a[0]:.4f}  vs  moving {b[0]:.4f}"
      f"   ratio {b[0]/max(a[0],1e-12):.2f}x")
print(f"  rho(1):    frozen {a[1][0]:+.4f}  vs  moving {b[1][0]:+.4f}")
print(f"  variance SNR: frozen {a[3]:.4f}  vs  moving {b[3]:.4f}")
print()
if a[1][0] < -0.3:
    print("  -> rho(1) at FROZEN theta is strongly negative: the gradient really does")
    print("     alternate sign between steps. Real signal, not a motion artifact.")
elif a[1][0] > -0.1:
    print("  -> rho(1) at FROZEN theta is NOT strongly negative. The 3.8% statistic is")
    print("     therefore mostly J(theta).dtheta, i.e. parameter motion, NOT data noise.")
    print("     The SNR blocker, as measured, is confounded.")
else:
    print("  -> rho(1) is weakly negative. Partly signal, partly motion. Report both.")
print()
print("  per-coordinate SNR, frozen theta:")
sl_trunk = slice(0, 0)
for nm, s, e in groups[:3]:
    gsq_g = sum(float(h[s:e].pow(2).sum()) for h in hist) / CAP
    v = float(mc[s:e].pow(2).sum()) / gsq_g if gsq_g else float("nan")
    print(f"    {nm:20s} {e-s:>9,} params  variance SNR {v:.4f}"
          f"  amplitude {np.sqrt(max(v,0)):.4f}")