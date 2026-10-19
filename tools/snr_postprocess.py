#!/usr/bin/env python3
"""snr_postprocess.py -- turn /tmp/snr_audit_full.json into the report table.

WHY THIS EXISTS.  snr_audit_full.py stores the random-projection lag products
WITHOUT dividing by R, so `rho_lag[h]` in the JSON is R x the true aggregate
autocorrelation and additionally diluted by the (steps - win_n) zero rows of
the projection table:

    rho_true(h) = rho_lag[h] * (steps - h) / win_n / R

(checked against the exact short-lag rho measured from full gradient vectors:
rho_lag[1] = 15.686 -> 0.712 vs exact 0.704, 1% agreement.)

Only the RATIOS stored in the JSON are affected.  The block-means, plain-mean
and TMT-EMA numbers in the same file are exact and need no correction.

WHAT IT REPORTS
  * kappa(b) = (1-b)/(1+b) * [1 + 2 sum_h b^h rho(h)] -- the autocorrelation
    inflation of telemetry.py:27-40, and the closed form
        SNR_TMT = s + kappa*(1-s)
    checked against the measured TMT value.
  * the Goodfellow / discrepancy estimator reconstructed from rho(h):
        ||G||^2_hat = ||mean_T g||^2 - (2/T^2) sum_{h=1}^{H} (T-h) rho(h) E||g||^2
    swept over the cutoff H.
  * N_eff = T / (1 + 2 sum_h rho(h)) -- how many independent gradient samples
    the window actually contains.  This is what decides whether ANY estimator
    of ||G||^2 is identified at this window length.
"""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/snr_audit_full.json"
d = json.load(open(path))
A = d["args"]
T, W, R, L = A["steps"], A["warmup"], A["proj"], A["L"]
win = T - W
mn = {k: d["res"][k]["plain"]["mean_norm"] for k in d["res"]}
g2p = {k: d["res"][k]["plain"]["g2"] for k in d["res"]}

rho = {}
for k, r in d["res"].items():
    rl = np.array(r["rho_lag"], dtype=float)
    h = np.arange(len(rl))
    # AUTO-DETECT the legacy convention (R x rho, diluted by the zero rows).
    # A real aggregate autocorrelation cannot exceed 1, so rho_lag[1] > 1.5 is
    # unambiguous.  snr_audit_full.py has been fixed to emit the corrected form.
    if len(rl) > 1 and rl[1] > 1.5:
        rl = rl * (T - h) / win / R
    rho[k] = np.clip(rl, -1.0, 1.5)
    rho[k][0] = 1.0

def kappa_exact(r, b, T):
    """kappa = sum_{k,l} w_k w_l rho(|k-l|) with the EXACT discrete weights
    w_k = (1-b) b^k / (1-b^T).  Truncated at the last available lag."""
    n = min(len(r), T)
    w = (1 - b) * b ** np.arange(n)
    w = w / w.sum()
    h = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    rr = r[np.minimum(h, len(r) - 1)]
    return float((w[:, None] * w[None, :] * rr).sum())


print("=" * 92)
print(f"POST-PROCESSED  {path}")
print(f"B={A['B']}  steps={T} warmup={W} window={win}  L={L}  R={R}")
print("=" * 92)
print("\nTRUE AGGREGATE AUTOCORRELATION rho(h)  (reconstructed, /R correction applied)")
show = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384]
for k in ("frozen", "moving"):
    if k not in rho:
        continue
    print(f"  {k:<7}" + " ".join(f"h{t}:{rho[k][t]:+.3f}" for t in show if t < len(rho[k])))
print("  exact short-lag check, frozen:",
      [round(x, 3) for x in d["res"]["frozen"]["rho_exact_short"][:6]])

print("\nkappa(b) -- the autocorrelation inflation of the TMT metric, and the")
print("closed form  SNR_TMT = s + kappa*(1-s):")
print(f"{'b':>7}{'kappa frozen':>14}{'kappa moving':>14}{'white floor':>13}")
for b in (0.9, 0.95, 0.99):
    kf = kappa_exact(rho["frozen"], b, win)
    km = kappa_exact(rho["moving"], b, win)
    print(f"{b:>7}{kf:>14.4f}{km:>14.4f}{(1-b)/(1+b):>13.4f}")

print("\nCLOSED-FORM CHECK against the measured TMT-EMA value (b=0.9):")
for k in ("frozen", "moving"):
    s = d["res"][k]["plain"]["snr"]
    kk = kappa_exact(rho[k], 0.9, win)
    pred = s + kk * (1 - s)
    meas = d["res"][k]["tmt_0.9"]["raw"]
    print(f"  {k:<7} s(plain) {s:.4f}  kappa {kk:.4f}  ->  pred {pred:.4f}"
          f"   measured TMT {meas:.4f}   (pred/meas {pred/meas:.2f})")

def kappa_exact(r, b, T):
    """kappa = sum_{k,l} w_k w_l rho(|k-l|) with the EXACT discrete weights
    w_k = (1-b) b^k / (1-b^T).  Truncated at the last available lag."""
    n = min(len(r), T)
    w = (1 - b) * b ** np.arange(n)
    w = w / w.sum()
    h = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    rr = r[np.minimum(h, len(r) - 1)]
    return float((w[:, None] * w[None, :] * rr).sum())


print("\nGOODFELLOW / DISCREPANCY estimator, cutoff H swept.")
print("  ||G||^2_hat = ||mean_T g||^2 - (2/T^2) sum_{h<=H} (T-h) rho(h) E||g||^2")
gf_last = {}
for k in ("frozen", "moving"):
    if k not in rho:
        continue
    print(f"  {k}")
    Hmax = len(rho[k]) - 1
    for H in sorted({1, 4, 8, 16, 32, 63, 127, 255, 383, Hmax}):
        if H >= win - 1 or H > Hmax:
            continue
        corr = 2.0 * float(np.sum((win - np.arange(1, H + 1)) * rho[k][1:H + 1])) / (win ** 2) * mn[k]
        gh = g2p[k] - corr
        gf_last[k] = (H, gh / mn[k])
        print(f"    H={H:>4}  correction {corr:>13.1f}   ||G||^2_hat {gh:>13.1f}"
              f"   SNR {gh/mn[k]:>9.4f}")

print("\nIMPLIED TRUE SNR  s, obtained by inverting the closed form on the")
print("MEASURED TMT value:   s = (SNR_TMT - kappa) / (1 - kappa)")
for k in ("frozen", "moving"):
    if k not in rho:
        continue
    kk = kappa_exact(rho[k], 0.9, win)
    meas = d["res"][k]["tmt_0.9"]["raw"]
    s = (meas - kk) / (1 - kk)
    print(f"  {k:<7} kappa(exact) {kk:.4f}   measured TMT {meas:.4f}"
          f"   ->  s = {s:>8.4f}")
    print(f"          and the last Goodfellow cutoff H={gf_last[k][0]} gives "
          f"s = {gf_last[k][1]:.4f}")


print("\nEFFECTIVE INDEPENDENT SAMPLES in the window (this is the real constraint):")
for k in ("frozen", "moving"):
    if k not in rho:
        continue
    for H in (16, 64, 256, 384):
        if H >= len(rho[k]):
            continue
        ssum = 2.0 * float(np.sum(rho[k][1:H + 1]))
        print(f"  {k:<7} lags 1..{H:<4} sum_h rho(h) = {ssum:>8.2f}"
              f"   N_eff = win/(1+2 sum) = {win/(1.0+ssum):>7.2f}")

print("\nBATCH MEANS (exact, no projection involved):")
for k in ("frozen", "moving"):
    b = d["res"][k]["batchmeans"]
    print(f"  {k:<7} K={b['K']} L={L}: mean_k||m_k||^2 = {b['mean_block_sq']:.1f}"
          f" +- {b['jackknife_se']:.1f}   V_LR/step = {b['v_long_run']:.1f}"
          f"   ||G||^2_hat = {b['g2_hat']:.1f}   SNR = {b['snr_total']:.4f}")
