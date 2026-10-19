#!/usr/bin/env python3
"""snr_audit_full.py -- frozen-vs-moving SNR under autocorr-robust estimators.

Companion to tools/snr_estimator_audit.py.  Same frozen-theta mechanism (copied
from tools/rho_probe.py: restore theta0 before EVERY step_batch, because
step_batch ends in an AdamW update at gpu_port.py:940-957).

Why this exists.  snr_estimator_audit.py stores block means on a fixed grid L
and showed that at L=8 the gradients are still strongly correlated across
blocks (block-level rho(1) ~ +0.55 in units of 8 steps).  Batch-means is only
valid when L >> the correlation time, so a fixed small L is NOT the unbiased
route.  This script adds the two things that make the estimate honest:

  * FULL-LAG AUTOCORRELATION via random projection.  With R random unit vectors
    p_r,  P_t[r] = <g_t, p_r>  satisfies
        E[ (1/R) sum_r P_t[r] P_{t+h}[r] ] = <g_t, g_{t+h}>
    so an R-column (T x R) table recovers every lag up to T/2 for 64 floats per
    step instead of 4.5M.  Cheap enough to make the whole lag profile a
    first-class result rather than an afterthought.

  * THE CLOSED FORM OF TMT'S OWN BIAS.  For bias-corrected weights
    w_k = (1-b)b^k/(1-b^T), sum w = 1, and gradient noise with per-step
    variance d*s2 and vector autocorrelation rho(tau):

        E ||EMA_c(g)||^2   = ||G||^2 + d*s2 * kappa(b, rho)
        E  EMA_c(||g||^2) = ||G||^2 + d*s2
        =>   SNR_TMT = s + kappa*(1 - s),   s = true SNR

        kappa = (1-b)/(1+b) * [ rho(0) + 2 sum_{tau>=1} b^tau rho(tau) ]
              = sum_k w_k^2                          (white-noise floor)

    So TMT's number is a CONVEX COMBINATION of the true SNR and kappa: it is
    biased UPWARD, by kappa*(1-s), and kappa is a function of the
    autocorrelation time -- NOT a common scale factor.  If frozen theta and
    moving theta have different autocorrelation, the bias distorts the
    comparison rather than cancelling in it.  This script measures both
    conditions' rho(tau), computes kappa for each, and checks the closed form.

  * GOODFELLOW / DISCREPANCY estimator, the textbook autocorr-corrected one:
        ||G||^2_hat = ||mean_T g||^2 - (2/T^2) sum_{h=1}^{H} (T-h) Chat(h)
    with Chat(h) the mean inner product at lag h and H a truncation cutoff.
    Its truncation bias is 2 sum_{h>H} (1-h/T) tr C_h, so the script sweeps H.

Read-only w.r.t. the repo: no checkpoint is written, theta is restored.

Usage:
  .venv/bin/python tools/snr_audit_full.py --steps 768 --warmup 64 \
      --L 64 --proj 64 --ring 16 --out /tmp/snr_audit_full.json
"""
import argparse
import glob
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np
import torch

import gpu_port as G
import numpy_port as np_

BETAS = (0.9, 0.95, 0.99)
CUTOFFS = (1, 4, 8, 16, 32, 63, 127, 255, 383)


def kappa_of(rho, b):
    """kappa = sum_{k,l} w_k w_l rho(|k-l|) = (1-b)/(1+b) [rho0 + 2 sum b^t rho_t]."""
    rho = np.asarray(rho, dtype=float)
    out = {}
    for bb in BETAS:
        w = np.arange(1, len(rho) + 1)
        inner = np.sum((bb ** w) * rho[1:])
        out[bb] = float((1 - bb) / (1 + bb) * (rho[0] + 2 * inner))
    return out


def capture(model, mats, ets, theta0, args, freeze):
    n = model.theta.numel()
    assert args.proj >= 8
    gen = torch.Generator().manual_seed(1234)
    P = torch.randn(args.proj, n, generator=gen) / np.sqrt(n)   # R x n unit rows

    if freeze:
        model.theta.copy_(theta0)
    model.t = 0

    ema_g = {b: torch.zeros(n) for b in BETAS}
    ema_q = {b: 0.0 for b in BETAS}
    ema_min = {b: float("inf") for b in BETAS}
    ema_neg = {b: 0 for b in BETAS}
    raw_min = float("inf")
    raw_nonpos = 0

    acc = torch.zeros(n)
    win_norm = []
    blocks, acc_blk, blk_n, win_n = [], torch.zeros(n), 0, 0
    proj = np.zeros((args.steps, args.proj))
    ringbuf = []
    rho_num = np.zeros(args.ring + 1)
    rho_den = np.zeros(args.ring + 1)

    for t in range(args.steps):
        if freeze:
            model.theta.copy_(theta0)
        model.step_batch(mats[t], mats[t + 1], ets[t])
        g = model.grads.detach()
        q = float(g.pow(2).sum())
        raw_min = min(raw_min, q)
        if q <= 0:
            raw_nonpos += 1
        for b in BETAS:
            ema_g[b].mul_(b).add_(g, alpha=1.0 - b)
            ema_q[b] = b * ema_q[b] + (1.0 - b) * q
            ema_min[b] = min(ema_min[b], ema_q[b])
            if ema_q[b] <= 0:
                ema_neg[b] += 1
        if t < args.warmup:
            continue
        acc.add_(g)
        win_norm.append(q)
        win_n += 1
        proj[win_n - 1] = (g.unsqueeze(0) @ P.t()).squeeze(0).numpy()
        acc_blk.add_(g)
        blk_n += 1
        if blk_n == args.L:
            blocks.append((acc_blk / args.L).clone())
            acc_blk.zero_()
            blk_n = 0
        for tau in range(1, min(args.ring, len(ringbuf)) + 1):
            rho_num[tau] += float(torch.dot(ringbuf[-tau], g))
            rho_den[tau] += float(ringbuf[-tau].pow(2).sum())
        ringbuf.append(g.clone())
        if len(ringbuf) > args.ring:
            ringbuf.pop(0)
    if freeze:
        model.theta.copy_(theta0)

    out = {"win_n": win_n, "steps": args.steps, "L": args.L}
    T = args.steps
    for b in BETAS:
        corr = 1.0 - b ** T
        num, den = float(ema_g[b].pow(2).sum()), ema_q[b]
        out[f"tmt_{b}"] = {
            "raw": num / max(den, 1e-38),
            "corrected": (num / corr ** 2) / max(den / corr, 1e-38),
            "den_min": ema_min[b], "den_neg_steps": ema_neg[b],
            "correction_factor": corr,
        }
    out["raw_norm_min"] = raw_min
    out["raw_norm_nonpos"] = raw_nonpos
    mean_vec = acc / win_n
    mn = float(np.mean(win_norm))
    g2_plain = float(mean_vec.pow(2).sum())
    out["plain"] = {"g2": g2_plain, "mean_norm": mn, "snr": g2_plain / mn}

    # ---- full-lag aggregate autocorrelation from the random projection ----
    # p_r is a random UNIT direction, so Var_p <g, p> = ||g||^2 / d, not
    # ||g||^2.  Hence  <g_t, g_{t+h}> ~= (d/R) * sum_r P_t[r] P_{t+h}[r].
    # The RATIO below needs no d, but the absolute inner product does -- getting
    # that factor wrong silently zeroes the Goodfellow correction.  (It did,
    # once: the numbers reported here were recomputed from rho_lag by
    # tools/snr_postprocess.py, which applies the /R and row-dilution
    # corrections; see its docstring.  tools/snr_audit_full.py is fixed.)
    Rp = (proj * proj).sum(1) / args.proj           # ~ E||g_t||^2 / d
    rho_lag = [1.0]
    nh = min(args.steps - 4, win_n // 2, 384)
    for h in range(1, nh + 1):
        a = proj[:win_n - h]
        b_ = proj[h:win_n]
        rho_lag.append(float((a * b_).sum(1).mean()) / args.proj
                       / float(Rp[:win_n - h].mean()))
    out["rho_lag"] = rho_lag
    out["rho_exact_short"] = [float(rho_num[t] / rho_den[t]) if rho_den[t] else float("nan")
                              for t in range(1, args.ring + 1)]
    out["n_par"] = args.proj

    # ---- Goodfellow / discrepancy estimator, swept over the cutoff H ----
    gf = {}
    pj = proj[:win_n]
    dpar = float(n) / args.proj      # (d/R); see the note above
    for H in CUTOFFS:
        if H >= win_n - 2:
            continue
        acc2 = 0.0
        for h in range(1, H + 1):
            acc2 += (win_n - h) * float((pj[:win_n - h] * pj[h:]).sum(1).mean())
        g2 = g2_plain - 2.0 * dpar * acc2 / (win_n ** 2)
        gf[str(H)] = {"g2_hat": g2, "snr_total": g2 / mn,
                      "snr_noise": g2 / max(mn - g2, 1e-30)}
    out["goodfellow"] = gf

    # ---- batch means at grid L ----
    if blocks:
        M = torch.stack(blocks)
        K = M.shape[0]
        q = float((M * M).sum(1).mean())
        mbar = M.mean(0)
        dev = float(((M - mbar) ** 2).sum(1).sum())
        v_lr = (args.L / max(K - 1, 1)) * dev
        g2 = q - v_lr / args.L
        sq = (M * M).sum(1)
        tot = float(sq.sum())
        jk = np.array([(tot - float(sq[i])) / (K - 1) for i in range(K)])
        out["batchmeans"] = {
            "K": K, "L": args.L, "mean_block_sq": q,
            "jackknife_se": float(np.sqrt((K - 1) / K * float(((jk - jk.mean()) ** 2).sum()))),
            "v_long_run": v_lr, "g2_hat": g2,
            "snr_total": g2 / mn, "snr_noise": g2 / max(mn - g2, 1e-30),
            "snr_lr": g2 / max(v_lr, 1e-30), "raw": q / mn,
        }
    out["mean_norm"] = mn
    out["rho_agg_mean_norm"] = float(np.mean(win_norm))
    del ema_g, acc, blocks, ringbuf, P
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/lrroles/arm_lrdecay1x_t919897.npz")
    ap.add_argument("--data", default="data/wikipedia_clean/**/wiki_*")
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=768)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--proj", type=int, default=64)
    ap.add_argument("--ring", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--conditions", default="frozen,moving")
    ap.add_argument("--out", default="/tmp/snr_audit_full.json")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(a.data, recursive=True) if os.path.isfile(f))
    random.Random(a.data_seed).shuffle(files)
    raw = open(files[0], "rb").read(64_000_000)
    streams, endss = np_.Runtime.make_streams(raw, a.B)
    Ld = min(len(s) for s in streams) - 1
    mats = torch.from_numpy(np.stack(
        [np.frombuffer(s[:Ld + 1], dtype=np.uint8) for s in streams],
        axis=1).astype(np.int64))
    ets = torch.from_numpy(np.array([e[:Ld] for e in endss], dtype=np.float32).T)
    print(f"corpus {files[0]}  B={a.B} steps={a.steps} warmup={a.warmup} "
          f"L={a.L} proj={a.proj}\n")

    res, theta0 = {}, None
    for tag, freeze in (("frozen", True), ("moving", False)):
        if tag not in a.conditions.split(","):
            continue
        t0 = time.perf_counter()
        torch.manual_seed(a.seed)
        m = G.GpuModel(dim=512, layers=16, device="cpu",
                       ring_credit="full", seed=a.seed)
        m.new_states(a.B)
        if not m.load(a.ckpt):
            sys.exit(f"cannot load {a.ckpt}")
        if theta0 is None:
            theta0 = m.theta.detach().clone()
        else:
            m.theta.copy_(theta0)
        res[tag] = capture(m, mats, ets, theta0, a, freeze)
        m.theta.copy_(theta0)
        del m
        print(f"[{tag}] done {time.perf_counter()-t0:.0f}s", flush=True)
    json.dump({"args": vars(a), "res": res}, open(a.out, "w"), indent=1)

    if len(res) < 2:
        for tag, r in res.items():
            print(f"\n{tag}: plain {r['plain']['snr']:.4f}  "
                  + "  ".join(f"TMT b={b} raw {r[f'tmt_{b}']['raw']:.4f}"
                              for b in BETAS))
        return
    fz, mv = res["frozen"], res["moving"]
    print("\n" + "=" * 84)
    print(f"FROZEN vs MOVING, B={a.B}  T={a.steps}  window={fz['win_n']}  L={a.L}")
    print("=" * 84)
    rows = []
    for b in BETAS:
        rows.append((f"TMT EMA b={b} RAW (shipped)", fz[f"tmt_{b}"]["raw"], mv[f"tmt_{b}"]["raw"]))
        rows.append((f"TMT EMA b={b} BIAS-CORRECTED", fz[f"tmt_{b}"]["corrected"], mv[f"tmt_{b}"]["corrected"]))
    rows.append(("plain time-average  ||mean g||^2/mean||g||^2", fz["plain"]["snr"], mv["plain"]["snr"]))
    bf, bm = fz.get("batchmeans"), mv.get("batchmeans")
    if bf:
        rows.append((f"batch-means K={bf['K']}xL={bf['L']} RAW", bf["raw"], bm["raw"]))
        rows.append((f"batch-means CORRECTED  ||G||^2/E||g||^2", bf["snr_total"], bm["snr_total"]))
        rows.append((f"batch-means CORRECTED  ||G||^2/tr(Sigma_0)", bf["snr_noise"], bm["snr_noise"]))
        rows.append((f"batch-means CORRECTED  ||G||^2/V_LR", bf["snr_lr"], bm["snr_lr"]))
    for H in sorted(int(k) for k in fz["goodfellow"]):
        rows.append((f"Goodfellow autocorr-corr, cutoff H={H}",
                     fz["goodfellow"][str(H)]["snr_total"], mv["goodfellow"][str(H)]["snr_total"]))
    print(f"{'estimator':<46}{'frozen':>11}{'moving':>11}{'ratio':>10}")
    print("-" * 84)
    for nm, x, y in rows:
        x, y = float(x), float(y)
        print(f"{nm:<46}{x:>11.4f}{y:>11.4f}{x/max(y,1e-30):>9.2f}x")

    if bf:
        d = bf["mean_block_sq"] - bm["mean_block_sq"]
        sed = (bf["jackknife_se"] ** 2 + bm["jackknife_se"] ** 2) ** 0.5
        print(f"\njackknife on ||G||^2: frozen {bf['mean_block_sq']:.1f}+-{bf['jackknife_se']:.1f}"
              f"  moving {bm['mean_block_sq']:.1f}+-{bm['jackknife_se']:.1f}"
              f"  z={d/sed:.2f}")
        print(f"V_LR/step: frozen {bf['v_long_run']:.1f}  moving {bm['v_long_run']:.1f}"
              f"  ({bm['v_long_run']/max(bf['v_long_run'],1e-30):.2f}x)")
        print(f"E||g||^2: frozen {bf['snr_total']*0+ (bf['g2_hat']/bf['snr_total']):.1f}"
              f"  moving {bm['g2_hat']/bm['snr_total']:.1f}"
              f"  ({ (bm['g2_hat']/bm['snr_total'])/(bf['g2_hat']/bf['snr_total']):.2f}x)")

    print("\nAGGREGATE AUTOCORRELATION rho(tau) of the gradient vector:")
    for tag in ("frozen", "moving"):
        r = fz["rho_lag"] if tag == "frozen" else mv["rho_lag"]
        show = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
        print(f"  {tag:<7}" + " ".join(f"t{t}:{r[t]:+.3f}" for t in show if t < len(r)))
    print("\nkappa(b, rho) = the white-noise floor / autocorr inflation of TMT:")
    for tag in ("frozen", "moving"):
        k = kappa_of(fz["rho_lag"] if tag == "frozen" else mv["rho_lag"], BETAS[0])
        print(f"  {tag:<7} kappa(0.9) = {k[0.9]:.4f}   "
              f"kappa(0.95) = {k[0.95]:.4f}   kappa(0.99) = {k[0.99]:.4f}")
    print(f"  white-noise floor kappa(0.9) = {(1-0.9)/(1+0.9):.4f} "
          f"(= sum_k w_k^2)  <- TMT cannot read below this")
    # closed-form prediction using the batch-means s and the measured kappa
    for tag in ("frozen", "moving"):
        r = fz if tag == "frozen" else mv
        if not r.get("batchmeans"):
            continue
        k = kappa_of(r["rho_lag"], 0.9)[0.9]
        s = r["batchmeans"]["snr_total"]
        pred = s + k * (1 - s)
        print(f"  {tag:<7} closed form  s + kappa(1-s) = {s:.4f} + "
              f"{k:.4f}*(1-{s:.4f}) = {pred:.4f}   vs TMT raw {r['tmt_0.9']['raw']:.4f}"
              f"   (predicted bias x{pred/max(s,1e-12):.1f})")

    print("\nDENOMINATOR HEALTH:")
    print(f"  min raw ||g||^2 over run: frozen {fz['raw_norm_min']:.4e}  "
          f"moving {mv['raw_norm_min']:.4e}   steps<=0: "
          f"{fz['raw_norm_nonpos']}/{mv['raw_norm_nonpos']}")
    for tag in ("frozen", "moving"):
        for b in BETAS:
            d = res[tag][f"tmt_{b}"]
            print(f"  {tag:<7} b={b:<5} EMA denom min {d['den_min']:.4e}"
                  f"   steps at <=0: {d['den_neg_steps']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()