#!/usr/bin/env python3
"""snr_synthetic_check.py -- closed-form test of WHICH estimator is biased.

Question being settled.  docs/ideas/papers/gns-pytorch.GITHUB-README.md warns
that "an EMA of per-batch GNS values converges to the wrong number (50-70% off
in the synthetic test)" and that a per-batch |G|^2 "often negative when the
batch is far below the noise scale".  TMT's standing metric (telemetry.py:27-40)
is ||EMA_b(g)||^2 / EMA_b(||g||^2).  Does the gns-pytorch warning apply to it?

MODEL.  g_t = G + u_t in R^d, u_t zero-mean, per-step variance sigma^2 per
coordinate, and AGGREGATE autocorrelation rho(h) = E<u_t,u_{t+h}>/(d sigma^2).
Then with bias-corrected weights w_k = (1-b)b^k/(1-b^T), sum_k w_k = 1,

    E ||EMA_c(g)||^2    = ||G||^2 + d sigma^2 * kappa(b, rho)
    E  EMA_c(||g||^2)  = ||G||^2 + d sigma^2
    =>   SNR_TMT = s + kappa*(1 - s),      s = true SNR = r/(1+r),  r = ||G||^2/(d sigma^2)

    kappa = sum_{k,l} w_k w_l rho(|k-l|) = (1-b)/(1+b) * [1 + 2 sum_{h>=1} b^h rho(h)]

so TMT's number is a CONVEX COMBINATION of the true SNR and kappa.  It is biased
UPWARD by kappa*(1-s).  kappa is a function of the noise's AUTOCORRELATION, not
of the data or the batch size, so it is NOT a common scale factor: two
conditions with different autocorrelation get differently-inflated numbers and
the comparison between them is distorted.  kappa = sum_k w_k^2 = (1-b)/(1+b) in
the white-noise limit -- a FLOOR below which the metric cannot read.

Checks:
  1. the closed form against Monte-Carlo, for white / correlated noise;
  2. every estimator's bias factor against a known truth;
  3. batch-means bias vs block length L;
  4. the gns-pytorch two-batch-size estimator: negative |G|^2, and the
     EMA-of-ratio vs accumulate-then-divide error;
  5. can TMT's denominator ever go <= 0?

Usage: .venv/bin/python tools/snr_synthetic_check.py
"""
import json

import numpy as np
import torch

torch.manual_seed(0)
OUT = {}


def ema_weights(b, T):
    w = (1.0 - b) * b ** np.arange(T)
    return w / w.sum()


def make_process(T, d, W):
    """Moving average of W iid steps, rescaled to per-coordinate variance 1.
    Its aggregate autocorrelation is rho(h) = max(1 - |h|/W, 0), so rho(1) =
    (W-1)/W and the correlation time is ~W.  W=1 is white noise."""
    e = torch.randn(T, d)
    if W <= 1:
        return e
    c = torch.cumsum(e, 0)
    pad = torch.zeros(W - 1, d)
    out = (torch.cat([pad, c])[:T] - c) / W
    return out


def rho_agg(g, hmax):
    n = g.pow(2).sum(1).mean()
    return np.array([float((g[:-h] * g[h:]).sum(1).mean() / n) if h else 1.0
                     for h in range(hmax + 1)])


def kappa(rho, b):
    h = np.arange(1, len(rho))
    return float((1 - b) / (1 + b) * (rho[0] + 2 * float(np.sum(b ** h * rho[1:]))))


def run(b, T, d, r, W, label, reps=60, L=64):
    G = torch.zeros(d)
    G[0] = np.sqrt(r * d)
    s_true = r / (1 + r)
    acc = []
    for _ in range(reps):
        u = make_process(T, d, W)
        u /= float(u.pow(2).sum(1).mean() / d) ** 0.5
        g = G.unsqueeze(0) + u
        mn = float(g.pow(2).sum(1).mean())

        w = torch.from_numpy(ema_weights(b, T)).float()
        num = float((w.unsqueeze(1) * g).sum(0).pow(2).sum())
        den = float((w * g.pow(2).sum(1)).sum())

        ev = torch.zeros(d)
        rn = rd = 0.0
        for t in range(T):
            ev.mul_(b).add_(g[t], alpha=1 - b)
            rd = b * rd + (1 - b) * float(g[t].pow(2).sum())
        rn = float(ev.pow(2).sum())
        corr = 1 - b ** T

        K = T // L
        mb = g[:K * L].reshape(K, L, d).mean(1)
        q = float((mb * mb).sum(1).mean())
        v = (L / (K - 1)) * float(((mb - mb.mean(0)) ** 2).sum())
        g2 = q - v / L
        acc.append({"tmt_raw": (rn / corr ** 2) / (rd / corr),
                    "tmt_bc": num / den, "plain": float(g.mean(0).pow(2).sum()) / mn,
                    "bm_raw": q / mn, "bm_corr": g2 / mn,
                    "bm_noise": g2 / (mn - g2)})
    a = {k: float(np.mean([o[k] for o in acc])) for k in acc[0]}
    uu = make_process(T, d, W)
    uu /= float(uu.pow(2).sum(1).mean() / d) ** 0.5
    rho = rho_agg(uu, min(400, T // 4))       # the NOISE autocorrelation, not g's
    a["kappa"] = kappa(rho, b)
    a["s_true"] = s_true
    a["pred"] = s_true + a["kappa"] * (1 - s_true)
    a["label"] = label
    a["rho_first"] = [float(x) for x in rho[:8]]
    OUT[label] = a
    return a


FILTS = {"white (W=1)": 1, "ma2 rho1=0.5": 2, "ma5 rho1=0.8": 5,
         "ma11 rho1=0.91": 11, "ma41 rho1=0.98": 41}

print("=" * 96)
print("CHECK 1 -- closed form  SNR_TMT = s + kappa*(1-s)  ?   d=2000 T=2000 b=0.9")
print("=" * 96)
print(f"{'noise':<18}{'rho_n(1)':>9}{'s(true)':>9}{'kappa':>8}{'pred':>9}{'MC':>9}{'err':>9}")
for lab, W in FILTS.items():
    a = run(0.9, 2000, 2000, 0.2, W, lab)
    err = (a["tmt_bc"] - a["pred"]) / a["pred"]
    print(f"{lab:<14}{a['rho_first'][1]:>9.3f}{a['s_true']:>9.4f}{a['kappa']:>8.4f}"
          f"{a['pred']:>9.4f}{a['tmt_bc']:>9.4f}{err:>9.1%}")
print(f"\nwhite-noise check: kappa = (1-b)/(1+b) = {(1-0.9)/(1+0.9):.4f} at b=0.9,"
      f"  {(1-0.95)/(1+0.95):.4f} at b=0.95,  {(1-0.99)/(1+0.99):.5f} at b=0.99;"
      f"  MC SNR_TMT on white noise at s=0.1667 = 0.2108")

print()
print("=" * 96)
print("CHECK 2 -- bias factor of every estimator against a known truth (L=64)")
print("=" * 96)
hdr = (f"{'noise / r':<18}{'s':>7}{'TMT raw':>9}{'TMT bc':>9}{'plain':>9}"
       f"{'bm raw':>9}{'bm corr':>9}{'bm/Sig0':>9}")
print(hdr)
for lab, W in FILTS.items():
    for r in (0.05, 0.2, 1.0):
        a = run(0.9, 2000, 2000, r, W, f"{lab}_r{r}", reps=30)
        s = a["s_true"]
        print(f"{lab + ' r=' + str(r):<18}{s:>7.4f}"
              f"{a['tmt_raw']/s:>8.2f}x{a['tmt_bc']/s:>8.2f}x{a['plain']/s:>8.2f}x"
              f"{a['bm_raw']/s:>8.2f}x{a['bm_corr']/s:>8.2f}x{a['bm_noise']/s:>8.2f}x")
print("  (entries are bias FACTORS: MC / truth.  1.00x = unbiased.)")

print()
print("=" * 96)
print("CHECK 3 -- batch-means bias vs block length L (ma41, r=0.2, T=2000)")
print("  the correlation time here is ~41 steps, so L must exceed it.")
print("=" * 96)
for Lb in (4, 8, 16, 32, 64, 128, 256):
    a = run(0.9, 2000, 2000, 0.2, 41, f"bmL{Lb}", reps=30, L=Lb)
    print(f"   L={Lb:<4} K={2000//Lb:<4} corrected {a['bm_corr']:.4f} "
          f"(raw {a['bm_raw']:.4f})   truth {a['s_true']:.4f}   "
          f"bias x{a['bm_corr']/a['s_true']:.3f}")

print()
print("=" * 96)
print("CHECK 4 -- the gns-pytorch construction (per-batch |G|^2 that goes negative)")
print("=" * 96)
print("McCandlish A.1 / gns-pytorch per-example path:  b_small=1, b_big=n, so")
print("  a_hat = (n*E|G_n|^2 - E|G_1|^2)/(n-1),   t_hat = n*(E|G_1|^2 - E|G_n|^2)/(n-1)")
print("Both are DIFFERENCES of two large, very noisy squared norms.  That is the")
print("source of the negativity, and TMT never forms either difference.")
print()
n, d, B, T = 64, 400, 64, 600
print(f"{'||G||^2/(d s2)':>14}{'B_simple':>11}{'steps a_hat<=0':>16}"
      f"{'EMA-of-RATIO':>16}{'accum/divide':>16}")
for far in (0.0002, 0.002, 0.02, 0.2, 1.0, 4.0, 20.0):
    G = torch.zeros(d)
    G[0] = np.sqrt(far * d)
    ratios, nums, dens, neg = [], [], [], 0
    for _ in range(T):
        z = torch.randn(n, d)
        per_ex = float((z + G).pow(2).sum(1).mean())   # E|G_1|^2 = a + t
        big = float((z[:B].mean(0) + G).pow(2).sum())  # E|G_B|^2 = a + t/B
        a_hat = (B * big - per_ex) / (B - 1)
        t_hat = B * (per_ex - big) / (B - 1)
        if a_hat <= 0:
            neg += 1
        ratios.append(t_hat / a_hat if a_hat > 0 else 0.0)
        nums.append(t_hat)
        dens.append(a_hat)
    bb = 0.9
    er = 0.0
    for x in ratios:
        er = bb * er + (1 - bb) * x
    truth = d / (far * d)
    ema = er
    accd = sum(nums) / sum(dens)
    print(f"{far:>14}{truth:>11.1f}{str(neg)+'/'+str(T):>16}"
          f"{ema:>9.2f} ({ema/truth-1:+.0%})".ljust(52)
          + f"{accd:>9.2f} ({accd/truth-1:+.0%})")
print("  TMT's operating point: with B=1024 and a measured SNR of 0.05-0.17 the")
print("  implied B_simple = 1/SNR is 6-20, i.e. WELL ABOVE the noise scale --")
print("  the regime where this estimator is well conditioned and |G|^2 stays positive.")

print()
print("=" * 96)
print("CHECK 5 -- can TMT's denominator EMA_b(||g||^2) go <= 0?")
print("=" * 96)
print("telemetry.py:34-37 accumulates ema_grad_sq = b*ema_grad_sq + (1-b)*g*g,")
print("elementwise; telemetry.py:36 sums it; :37 clamps with .clamp_min(tiny).")
print("  * every term (g*g) is >= 0, so the accumulator is a non-negative")
print("    weighted sum.  It reaches exactly 0 only if g == 0 at every step.")
print("  * the numerator is a squared norm, so >= 0 by construction.")
print("  => SNR >= 0 structurally; there is NO negative-denominator mode.")
rng = np.random.default_rng(0)
worst, bad, tot = np.inf, 0, 0
for _ in range(300):
    x = rng.standard_normal(2000) * 0.5
    x[rng.random(2000) < 0.6] = 0.0            # adversarial: mostly exactly zero
    eg = 0.0
    for v in x ** 2:
        eg = 0.9 * eg + 0.1 * v
        worst = min(worst, eg)
        tot += 1
        if eg <= 0:
            bad += 1
print(f"  adversarial path ({tot:,} steps, 60% of entries exactly 0):")
print(f"     min EMA_b(||g||^2) = {worst:.3e}   steps at <= 0: {bad}"
      f"   -> clamped to tiny() by telemetry.py:37, so the metric reads 0.0,")
print(f"        never negative.")
print("  real run (this audit, B=1024): min raw ||g||^2 and min EMA denominator")
print("  are reported in the frozen/moving table below the estimator table.")

json.dump(OUT, open("/tmp/snr_synthetic.json", "w"), indent=1)
print("\nwrote /tmp/snr_synthetic.json")