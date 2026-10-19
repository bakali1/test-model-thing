#!/usr/bin/env python3
"""snr_estimator_audit.py -- is the headline "gradient SNR ~ 0.05" a biased estimator?

READ-ONLY with respect to the repo. Creates no checkpoint, touches no tracked
file, and restores the checkpointed theta exactly on exit. The frozen-theta
mechanism is copied from tools/rho_probe.py (restore theta0 before EVERY
step_batch; step_batch ends in an AdamW update, gpu_port.py:940-957, so theta
moves unless it is pinned).

WHAT IS AUDITED
  TMT's standing metric, telemetry.py:27-40, is

      snr = ||EMA_b(g)||^2 / EMA_b(||g||^2)          b = 0.9, NOT bias-corrected

  which drove the roadmap for weeks.  RESULTS.md measured the SAME formula at a
  FROZEN parameter and got 0.633, versus ~0.05 in training: a 12.6x gap.  Two
  things have to be settled before that gap means anything:

    (1) is the formula itself biased, and does the bias preserve the
        frozen-vs-moving COMPARISON (in which case the gap survives) or does it
        distort it (in which case the gap may be an artifact)?
    (2) can the denominator EMA_b(||g||^2) go negative or unstable?  (docs/
        ideas/papers/gns-pytorch.GITHUB-README.md warns that a *different*
        estimator's |G|^2 can go negative; is that TMT's problem?)

ESTIMATORS REPORTED (all from the SAME capture, both conditions)
  TMT       ||EMA_b(g)||^2 / EMA_b(||g||^2)            as shipped, uncorrected
  TMT-bc    the same with the (1-b^T) bias correction applied to BOTH sides
  plain     ||mean_T g||^2 / mean_T(||g||^2)
  bm        BATCH MEANS over non-overlapping blocks of length L.  Block means
            m_k are ~independent across k, so
                mean_k ||m_k||^2  =  ||G||^2 + V_LR/L
                V_LR               =  (L/(K-1)) sum_k ||m_k - mbar||^2
            gives an autocorr-corrected  ||G||^2  and an unbiased long-run
            variance.  This is the standard route for autocorrelated series and
            it does NOT assume independence between steps.
  jack      jackknife SE of  mean_k ||m_k||^2  over the K blocks, so the
            frozen-vs-moving difference can be tested rather than eyeballed.

Usage:
  .venv/bin/python tools/snr_estimator_audit.py \
      --ckpt runs/lrroles/arm_lrdecay1x_t919897.npz --B 1024 --steps 384 \
      --warmup 64 --L 8 --out /tmp/snr_audit.json
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

BETAS = (0.9, 0.95, 0.99, 0.999)


def batch_means_stats(block_means, norms_sq, L):
    """Autocorrelation-robust statistics from non-overlapping block means."""
    M = torch.stack(block_means)                       # (K, d)
    K = M.shape[0]
    mbar = M.mean(0)
    q = float((M * M).sum(1).mean())                   # mean_k ||m_k||^2
    dev = float(((M - mbar) ** 2).sum(1).sum())         # sum_k ||m_k-mbar||^2
    v_lr = (L / max(K - 1, 1)) * dev                   # long-run variance / step
    g2_bm = q - v_lr / L                               # bias-corrected ||G||^2
    # jackknife SE of q (delete-one-block), the standard autocorr-robust SE
    jk = []
    for i in range(K):
        q_i = (float((M * M).sum(1).sum()) - float((M[i] * M[i]).sum())) / (K - 1)
        jk.append(q_i)
    jk = np.array(jk)
    se_q = float(np.sqrt((K - 1) / K * float(((jk - jk.mean()) ** 2).sum())))
    # one-step noise tr(Sigma_0) = E||g||^2 - ||G||^2
    mean_norm = float(np.mean(norms_sq))
    return {
        "K": K, "L": L,
        "mean_block_sq": q, "jackknife_se": se_q,
        "v_long_run": v_lr, "g2_bm_corrected": g2_bm,
        "mean_norm_sq": mean_norm,
        "sigma0": mean_norm - g2_bm,
        "snr_bm_total": g2_bm / mean_norm,
        "snr_bm_noise": g2_bm / max(mean_norm - g2_bm, 1e-30),
        "snr_bm_lr": g2_bm / max(v_lr, 1e-30),
        "snr_blockmean_raw": q / mean_norm,
    }


def capture(model, mats, ets, theta0, steps, warm, freeze, L, ring):
    """One pass. Returns every statistic, streaming (memory ~ (K+ring)*18 MB)."""
    n = model.theta.numel()
    if freeze:
        model.theta.copy_(theta0)
    model.t = 0

    ema_g = {b: torch.zeros(n) for b in BETAS}
    ema_q = {b: 0.0 for b in BETAS}
    ema_q_min = {b: float("inf") for b in BETAS}
    ema_q_neg = {b: 0 for b in BETAS}

    acc = torch.zeros(n)          # sum over the post-warmup window
    win_norm = []                 # ||g||^2 over the post-warmup window
    all_norm = []                 # ||g||^2 over EVERY step (EMA denominator trace)
    blocks = []                   # block means of the post-warmup window
    acc_blk = torch.zeros(n)
    blk_n = 0
    win_n = 0

    ringbuf = []                  # last `ring` gradients, for rho(tau)
    rho_num = np.zeros(ring + 1)
    rho_den = np.zeros(ring + 1)

    loss_last = float("nan")
    for t in range(steps):
        if freeze:
            model.theta.copy_(theta0)
        loss_last = float(model.step_batch(mats[t], mats[t + 1], ets[t]))
        g = model.grads.detach()
        q = float(g.pow(2).sum())
        all_norm.append(q)
        for b in BETAS:                     # telemetry.py:33-37, uncorrected
            ema_g[b].mul_(b).add_(g, alpha=1.0 - b)
            ema_q[b] = b * ema_q[b] + (1.0 - b) * q
            if ema_q[b] < ema_q_min[b]:
                ema_q_min[b] = ema_q[b]
            if ema_q[b] <= 0.0:
                ema_q_neg[b] += 1
        if t < warm:
            continue

        acc.add_(g)
        win_norm.append(q)
        win_n += 1
        acc_blk.add_(g)
        blk_n += 1
        if blk_n == L:
            blocks.append(acc_blk / L)
            acc_blk.zero_()
            blk_n = 0

        ntau = min(ring, len(ringbuf))
        for tau in range(1, ntau + 1):
            rho_num[tau] += float(torch.dot(ringbuf[-tau], g))
            rho_den[tau] += float(ringbuf[-tau].pow(2).sum())
        ringbuf.append(g.clone())
        if len(ringbuf) > ring:
            ringbuf.pop(0)

    if freeze:
        model.theta.copy_(theta0)

    T = steps
    out = {"loss": loss_last, "steps": T, "warmup": warm, "L": L,
           "ema": {}, "denominator": {}}
    for b in BETAS:
        raw_num = float(ema_g[b].pow(2).sum())
        raw_den = ema_q[b]
        corr = 1.0 - b ** T
        out["ema"][str(b)] = {
            "snr_raw_telemetry": raw_num / max(raw_den, 1e-38),
            "snr_bias_corrected": (raw_num / corr ** 2) / max(raw_den / corr, 1e-38),
            "num_raw": raw_num, "den_raw": raw_den, "correction_factor": corr,
            "den_min_over_run": ema_q_min[b], "den_nonpositive_steps": ema_q_neg[b],
        }
        out["denominator"][str(b)] = {
            "min": min(all_norm), "min_ema": ema_q_min[b],
            "n_nonpositive_raw": sum(1 for x in all_norm if x <= 0.0),
            "n_nonpositive_ema": ema_q_neg[b],
        }

    mean_vec = acc / win_n
    mn = float(np.mean(win_norm))
    out["plain"] = {
        "snr_plain": float(mean_vec.pow(2).sum()) / mn,
        "g2_plain": float(mean_vec.pow(2).sum()),
        "mean_norm_sq": mn,
        "n": win_n,
    }
    # autocorr of the plain numerator, exposed for interpretation
    rho = []
    for tau in range(1, min(ring, T) + 1):
        rho.append(rho_num[tau] / rho_den[tau] if rho_den[tau] else float("nan"))
    out["rho_step"] = rho
    out["rho_ring"] = ring
    out["norm_sq_first"] = all_norm[:5]
    out["norm_sq_min"] = min(all_norm)
    out["norm_sq_max"] = max(all_norm)
    if blocks:
        out["batchmeans"] = batch_means_stats(blocks, win_norm, L)
        out["n_blocks"] = len(blocks)
        # block-level autocorrelation (in units of L steps), free
        M = torch.stack(blocks)
        br = []
        for lag in range(1, min(8, len(blocks))):
            a, bb = M[:-lag], M[lag:]
            br.append(float((a * bb).sum()) / float((a * a).sum()))
        out["rho_block"] = br
    del ema_g, acc, blocks, ringbuf
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/lrroles/arm_lrdecay1x_t919897.npz")
    ap.add_argument("--data", default="data/wikipedia_clean/**/wiki_*")
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=384)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--ring", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/snr_audit.json")
    ap.add_argument("--conditions", default="frozen,moving")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(a.data, recursive=True) if os.path.isfile(f))
    if not files:
        sys.exit(f"no corpus files matched {a.data}")
    random.Random(a.data_seed).shuffle(files)
    raw = open(files[0], "rb").read(64_000_000)
    streams, endss = np_.Runtime.make_streams(raw, a.B)
    L = min(len(s) for s in streams) - 1
    mats = torch.from_numpy(np.stack(
        [np.frombuffer(s[:L + 1], dtype=np.uint8) for s in streams],
        axis=1).astype(np.int64))
    ets = torch.from_numpy(np.array([e[:L] for e in endss], dtype=np.float32).T)
    print(f"corpus {files[0]}  {len(raw):,}B  L={L}  steps={a.steps} "
          f"warmup={a.warmup} block L={a.L} B={a.B}\n")

    results = {}
    theta0 = None
    for tag, freeze in (("frozen", True), ("moving", False)):
        if tag not in a.conditions.split(","):
            continue
        t0 = time.perf_counter()
        torch.manual_seed(a.seed)
        m = G.GpuModel(dim=512, layers=16, device="cpu",
                       ring_credit="full", seed=a.seed)
        m.new_states(a.B)
        if not m.load(a.ckpt):
            sys.exit(f"could not load {a.ckpt}")
        if theta0 is None:
            theta0 = m.theta.detach().clone()
        else:
            m.theta.copy_(theta0)
        r = capture(m, mats, ets, theta0, a.steps, a.warmup, freeze, a.L, a.ring)
        m.theta.copy_(theta0)          # leave nothing perturbed
        results[tag] = r
        del m
        print(f"[{tag}] done in {time.perf_counter()-t0:.0f}s")
    if theta0 is not None:
        del theta0

    json.dump({"args": vars(a), "results": results}, open(a.out, "w"), indent=1)

    # ---------------- report ----------------
    if "frozen" not in results or "moving" not in results:
        for tag, r in results.items():
            print(f"\n=== {tag} only (single-condition run) ===")
            for b in BETAS:
                print(f"  TMT EMA b={b}: raw "
                      f"{r['ema'][str(b)]['snr_raw_telemetry']:.4f}  "
                      f"corrected {r['ema'][str(b)]['snr_bias_corrected']:.4f}")
            print(f"  plain: {r['plain']['snr_plain']:.4f}")
            if "batchmeans" in r:
                print(f"  batch-means corrected ||G||^2/E||g||^2: "
                      f"{r['batchmeans']['snr_bm_total']:.4f}   "
                      f"||G||^2/tr(Sigma_0): {r['batchmeans']['snr_bm_noise']:.4f}")
        return
    fz, mv = results["frozen"], results["moving"]
    print("\n" + "=" * 78)
    print(f"SNR ESTIMATOR AUDIT   B={a.B}  steps={a.steps}  "
          f"window={a.steps-a.warmup}  block L={a.L}")
    print("=" * 78)
    print(f"{'estimator':<46}{'frozen':>12}{'moving':>12}{'ratio f/m':>12}")
    print("-" * 78)
    rows = []
    for b in BETAS:
        k = str(b)
        rows.append((f"TMT EMA b={b} RAW (telemetry.py:27-40, shipped)",
                     fz["ema"][k]["snr_raw_telemetry"],
                     mv["ema"][k]["snr_raw_telemetry"]))
        rows.append((f"TMT EMA b={b} BIAS-CORRECTED (1-b^T)^-1",
                     fz["ema"][k]["snr_bias_corrected"],
                     mv["ema"][k]["snr_bias_corrected"]))
    rows.append(("plain time-avg  ||mean g||^2/mean||g||^2",
                 fz["plain"]["snr_plain"], mv["plain"]["snr_plain"]))
    bf, bm_ = fz.get("batchmeans"), mv.get("batchmeans")
    if bf:
        rows.append((f"batch-means K={bf['K']}xL={bf['L']} RAW  mean_k||m_k||^2/E||g||^2",
                     bf["snr_blockmean_raw"], bm_["snr_blockmean_raw"]))
        rows.append((f"batch-means AUTOCORR-CORRECTED  ||G||^2/E||g||^2",
                     bf["snr_bm_total"], bm_["snr_bm_total"]))
        rows.append((f"batch-means CORRECTED  ||G||^2/tr(Sigma_0)",
                     bf["snr_bm_noise"], bm_["snr_bm_noise"]))
        rows.append((f"batch-means CORRECTED  ||G||^2/V_LR",
                     bf["snr_bm_lr"], bm_["snr_bm_lr"]))
    for name, x, y in rows:
        x = float(x); y = float(y)
        print(f"{name:<46}{x:>12.4f}{y:>12.4f}{x/max(y,1e-30):>11.2f}x")
    print("-" * 78)
    print("ratio f/m > 1  =>  frozen theta has the HIGHER SNR, i.e. the")
    print("iterate moving is what drives the SNR down.\n")

    if bf:
        qf, qm = bf["mean_block_sq"], bm_["mean_block_sq"]
        sef, sem = bf["jackknife_se"], bm_["jackknife_se"]
        d = qf - qm
        sed = (sef ** 2 + sem ** 2) ** 0.5
        print("SIGNIFICANCE of the gap in ||G||^2 (jackknife SE, block means):")
        print(f"  frozen ||G||^2~{qf:.4f} +- {sef:.4f}   moving ~{qm:.4f} +- {sem:.4f}")
        print(f"  diff {d:.4f}  pooled SE {sed:.4f}  z = {d/sed:.2f}   "
              f"ratio {qf/max(qm,1e-30):.2f}x")
        print(f"  autocorr correction V_LR/L: frozen {bf['v_long_run']/bf['L']:.4f}  "
              f"moving {bm_['v_long_run']/bm_['L']:.4f}")
        print(f"  V_LR/step: frozen {bf['v_long_run']:.4f}  "
              f"moving {bm_['v_long_run']:.4f}  "
              f"(moving/frozen {bm_['v_long_run']/max(bf['v_long_run'],1e-30):.2f}x)")
        print(f"  ||G||^2 fraction of E||g||^2: frozen "
              f"{bf['g2_bm_corrected']/bf['mean_norm_sq']:.4f}  moving "
              f"{bm_['g2_bm_corrected']/bm_['mean_norm_sq']:.4f}")

    print("\nAUTOCORRELATION of the gradient vector (aggregate), rho(tau):")
    for tag in ("frozen", "moving"):
        r = results[tag]["rho_step"]
        print(f"  {tag:<7}" + " ".join(f"{x:+.3f}" for x in r[:16]))
    if "rho_block" in fz:
        print("  block-level rho (units of L steps), frozen: "
              + " ".join(f"{x:+.3f}" for x in fz["rho_block"]))
        print("  block-level rho (units of L steps), moving: "
              + " ".join(f"{x:+.3f}" for x in mv["rho_block"]))

    print("\nDENOMINATOR HEALTH (can EMA_b(||g||^2) go <= 0?)")
    print(f"  raw ||g||^2 over the run: min {min(fz['norm_sq_min'],mv['norm_sq_min']):.4e}"
          f"   max {max(fz['norm_sq_max'],mv['norm_sq_max']):.4e}")
    for tag in ("frozen", "moving"):
        for b in BETAS:
            k = str(b)
            d = results[tag]["denominator"][k]
            print(f"  {tag:<7} b={b:<6} EMA denom min {d['min_ema']:.6e}"
                  f"   steps at <= 0: {d['n_nonpositive_ema']}"
                  f"   raw steps at <= 0: {d['n_nonpositive_raw']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()