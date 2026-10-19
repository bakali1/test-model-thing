#!/usr/bin/env python3
"""retention_probe.py -- Priority 1: the causal state-retention (Delta-h) probe.

Direct state-difference probe (PLAN-2026-10-06-MEMORY-BENCHMARKS.md section 1;
STRATEGIC-REVIEW-2026-10-05.md rev 2 section 4 Stage 1; the delta-h version
that supersedes the KL-over-conditions design of tools/copy_probe*.py -- a
direct state-norm difference needs no control conditions at all).

Protocol. Take the frozen checkpoint and feed pairs of sequences that are
identical except for the initial 4-byte key:

    sequence A:  KEY_A  + d distractor bytes
    sequence B:  KEY_B  + exactly the same d distractor bytes
    (KEY_A, KEY_B drawn i.i.d. random 4-byte keys, different within a pair)

At the position right after the d-th distractor byte, measure the causal state
difference at four stages independently:

    stage 1: raw recurrent state (pre-layer-norm)     -> l.states after the step
    stage 2: normalized state   (post-layer-norm)     -> h from batch_layernorm
    stage 3: post-nonlinearity block output           -> silu(h @ W.T)   == y
    stage 4: final logits                             -> x @ Wd.T + bd

Delta-h_d(stage) = sqrt( mean over trials of ||state_A - state_B||^2 ), the
per-trial difference vectors aggregated over the 16 layers by SRSS (which is
the L2 norm of the concatenated 16*512 state) and over trials by RMS. This is
the "typical magnitude of the state difference caused by the key flip".

Permutation null (--perm M): break the pairing by permuting the B-side labels
across trials (each trial's distractor stream is the causal structure under
test, so the paired difference is exactly the key trace). Report the null mean
and 95th percentile of the RMS norm over M permutations.

Decision rule (plan sec 1 / STRATEGIC-REVIEW sec 4):
    Delta-h dies by d ~ 16  -> architecture genuinely suppresses temporal
                              perturbations; LN-ceiling hypothesis strengthened
    Delta-h alive at d=512   -> LN-ceiling hypothesis weakened

Comparison prediction (plan sec 1, STRATEGIC-REVIEW sec 3.ii):
    p_j(d) = a_j^d * sqrt(1 - a_j^2)   with a_j the checkpoint's ACTUAL
    per-channel sigma (sigmoid of layer{i}.decay). Aggregated as the RMS over
    channels (sqrt(mean_j p_j^2)) -- per-channel is the honest S4D picture
    (heterogeneous a_1..a_512 per layer: different channels specialise on
    different d) and the RMS form is scale-comparable to the measured total
    Delta-h. The mean-sigma collapse a^d*sqrt(1-a^2) is also printed; it
    under-predicts the tail because long-decay channels dominate large d.

Usage (from the repo root, engine at root + tools/ must both be importable):
    .venv/bin/python tools/retention_probe.py --distractors random --trials 192 --seed 42
    .venv/bin/python tools/retention_probe.py --distractors real   --trials 192 --seed 42
    .venv/bin/python tools/retention_probe.py --distractors random --trials 192 --out /tmp/rp.json
"""
import os, sys, time, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)                       # engine: numpy_port at repo root
sys.path.insert(0, os.path.join(HERE, "tools"))

import numpy_port as N
from eval_npz import infer_config

DISTANCES = [8, 32, 128, 512, 2048]
VOCAB = 256
STAGES = {
    1: "rawstate(preLN)",
    2: "postLN",
    3: "silu(h.W^T)",
    4: "logits",
}


def fmt(x):
    return f"{x:.6g}"


def feed_seq_probe(m, seq):
    """Feed (B, L) uint8 batch; return per-stage vectors at the LAST position.

    Returns dict {1: (B, L*512), 2: (B, L*512), 3: (B, L*512), 4: (B, 256)}.
    """
    B, L = seq.shape
    m.new_states(B)
    st1, st2, st3 = [], [], []
    for t in range(L):
        enc = m.embed[seq[:, t]]
        x = enc
        for l in m.layers:
            x, st, _d, cache = l.forward(enc, x, l.states)
            l.states = st
            if t == L - 1:
                z, sig, h, _ = cache
                st1.append(st)
                st2.append(h)
                st3.append(z * sig)            # silu
    return {
        1: np.concatenate(st1, axis=1),
        2: np.concatenate(st2, axis=1),
        3: np.concatenate(st3, axis=1),
        4: x @ m.Wd.T + m.bd,
    }


def rms_norm(dx):
    """dx: (T, D). sqrt(mean over trials of per-trial L2^2)."""
    return float(np.sqrt(np.mean(np.einsum("td,td->t", dx, dx, optimize=True))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distractors", choices=["random", "real"], default="random")
    ap.add_argument("--trials", type=int, default=192, help="number of PAIRS (2x sequences)")
    ap.add_argument("--perm", type=int, default=200, help="permutation-null repetitions")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs/lrroles/arm_lrdecay1x_t919897.npz"))
    ap.add_argument("--distractor-file", default=os.path.join(HERE, "data/eval/tinystories_valid.txt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--distances", default=None,
                    help="comma list, e.g. 8,32,128 (override DEFAULT_DISTANCES)")
    ap.add_argument("--out", default=None, help="optional JSON dump path")
    args = ap.parse_args()
    distances = ([int(x) for x in args.distances.split(",")]
                 if args.distances else DISTANCES)
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    print(f"Loading {args.ckpt}", flush=True)
    d = np.load(args.ckpt)
    dim, layers = infer_config(d)
    m = N.Model(dim=dim, layers=layers, temp=0.75, lr=5e-4)
    m.new_states(1); m.load(args.ckpt); m.new_states(1)
    print(f"dim={dim} layers={layers}", flush=True)

    if args.distractors == "real":
        raw = np.frombuffer(open(args.distractor_file, "rb").read(), dtype=np.uint8)
        print(f"real distractors from {args.distractor_file} ({len(raw)} bytes)", flush=True)
    else:
        raw = None

    # ---------------- scalar prediction from the checkpoint's sigma ----------------
    sig = np.concatenate([N.sigmoid(m.layers[i].decay) for i in range(layers)])  # (16*512,)
    mean_sig = float(sig.mean())
    half = 1.0 / np.log2(1.0 / np.clip(sig, 1e-9, 1.0))
    print(
        f"[sigma] per-channel: N={sig.size} mean={mean_sig:.4f} min={sig.min():.4f} "
        f"max={sig.max():.4f}  (>2B hl: {(half > 2).sum()}, >4B: {(half > 4).sum()}, "
        f">8B: {(half > 8).sum()}, >32B: {(half > 32).sum()}) "
        f"mean half-life={half.mean():.2f}B, a^16(mean)={mean_sig ** 16:.2e}", flush=True)
    pred = {}
    for dd in distances:
        p_j = np.power(sig, dd) * np.sqrt(1.0 - sig * sig)      # per-channel (RMS->L2-like)
        pred[dd] = {
            "perchan": float(np.sqrt(np.mean(p_j * p_j))),      # RMS over channels
            "mean_sig": float(mean_sig ** dd * np.sqrt(1.0 - mean_sig * mean_sig)),
        }
    print("prediction a^d*sqrt(1-a^2)  [per-channel RMS | mean-sigma]", flush=True)
    for dd in distances:
        print(f"  d={dd:<6} {pred[dd]['perchan']:.6g}  {pred[dd]['mean_sig']:.6g}", flush=True)

    # ---------------- the probe ----------------
    nA = args.trials
    B = 2 * nA
    print(f"trials(pairs)={args.trials} (sequences={B}), perm={args.perm}",
          flush=True)
    print(f"{'d':>6} {'stage':>14} {'dh':>12} {'null_mean':>12} {'null_p95':>12} "
          f"{'dh/null95':>10} {'t(s)':>8}", flush=True)

    all_dh = {}      # stage -> {d: dh}
    all_null = {}    # stage -> {d: (mean, p95)}
    for dd in distances:
        ts0 = time.time()
        L = 4 + dd
        # shared distractor stream per pair, random keys per trial
        if args.distractors == "random":
            shared = rng.integers(0, VOCAB, size=(nA, dd), dtype=np.uint8)
        else:
            shared = np.zeros((nA, dd), dtype=np.uint8)
            n_avail = max(len(raw) - dd, 1)
            for i in range(nA):
                off = rng.integers(0, n_avail)
                shared[i] = raw[off:off + dd]
        keyA = rng.integers(0, VOCAB, size=(nA, 4), dtype=np.uint8)
        keyB = rng.integers(0, VOCAB, size=(nA, 4), dtype=np.uint8)
        # guarantee keyA != keyB within a pair (redraw row on collision)
        while True:
            same = np.all(keyA == keyB, axis=1)
            if not same.any():
                break
            keyB[same] = rng.integers(0, VOCAB, size=(int(same.sum()), 4), dtype=np.uint8)
        seqA = np.concatenate([keyA, shared], axis=1).astype(np.uint8)
        seqB = np.concatenate([keyB, shared], axis=1).astype(np.uint8)
        seq = np.concatenate([seqA, seqB], axis=0)

        # Feed both legs in ONE batch (halves Python-overhead-bound forward
        # time at the long distances); split the captured stages afterwards.
        v = feed_seq_probe(m, seq)
        t_feed = time.time()
        vA = {s: vv[:nA] for s, vv in v.items()}
        vB = {s: vv[nA:] for s, vv in v.items()}
        print(f"  [d={dd}] forward both legs: {t_feed - ts0:.1f}s", flush=True)

        for stage, name in STAGES.items():
            ts = time.time()
            dx = vA[stage].astype(np.float64) - vB[stage].astype(np.float64)
            dh = rms_norm(dx)
            vB64 = vB[stage].astype(np.float64)
            nulls = np.empty(args.perm)
            for r in range(args.perm):
                perm = rng.permutation(nA)
                nulls[r] = rms_norm(vA[stage].astype(np.float64) - vB64[perm])
            null_mean, null_p95 = float(nulls.mean()), float(np.quantile(nulls, 0.95))
            all_dh.setdefault(stage, {})[dd] = dh
            all_null.setdefault(stage, {})[dd] = (null_mean, null_p95)
            print(f"{dd:>6} {name:>14} {fmt(dh):>12} {fmt(null_mean):>12} {fmt(null_p95):>12} "
                  f"{'YES' if dh > null_p95 else 'no':>10} {time.time()-ts:>8.2f}",
                  flush=True)

    # ---------------- shape summary + verdict ----------------
    base = 8
    print("\nshape summary: ratio to d=8   [measured | perchan pred | mean-sig pred]",
          flush=True)
    for stage, name in STAGES.items():
        row = f"{name:>14}"
        for dd in distances:
            rat = all_dh[stage][dd] / all_dh[stage][base]
            row += f"  d={dd:<5} {rat:8.2e} | {pred[dd]['perchan'] / pred[base]['perchan']:8.2e} | {pred[dd]['mean_sig'] / pred[base]['mean_sig']:8.2e}"
        print(row, flush=True)

    # implied single-channel decay across the first two distances (raw state stage)
    if len(distances) >= 2:
        d_a, d_b = sorted(distances)[:2]
        s_a, s_b = all_dh[1][d_a], all_dh[1][d_b]
        alpha_est = np.exp((np.log(max(s_b, 1e-30)) - np.log(s_a)) / (d_b - d_a))
        print(f"\n[stage1 raw] implied per-channel alpha between d={d_a} and d={d_b}: "
              f"{alpha_est:.4f}  (half-life "
              f"{1.0 / np.log2(1.0 / max(alpha_est, 1e-9)):.2f}B)",
              flush=True)
    else:
        alpha_est = float("nan")

    def dl_ratio(stage, d1, d2):
        av1, av2 = all_dh[stage].get(d1), all_dh[stage].get(d2)
        if av1 is None or av2 is None or av2 <= 0:
            return np.inf
        return av1 / av2

    dlist = sorted(distances)
    has = lambda k: k in dlist

    alive512 = {s: all_dh[s][512] for s in STAGES} if has(512) else None
    ratio = ({s: (alive512[s] / all_null[s][512][1]
                  if all_null[s][512][1] > 0 else np.inf) for s in STAGES}
             if has(512) else None)

    far_d_sel = 2048 if has(2048) else (512 if has(512) else dlist[-1])

    r32 = dl_ratio(4, 32, 8) if has(32) else None
    r512 = dl_ratio(4, 512, 8) if has(512) else None
    r2048 = dl_ratio(4, 2048, 8) if has(2048) else None
    r_far = r2048 if r2048 is not None else r512
    far_d = 2048 if r2048 is not None else 512
    far_abs = all_dh[4].get(far_d, 0.0) if all_dh[4].get(far_d, 0.0) > 0 else 0.0

    print("\nVERDICT (decision rule, plan sec 1):", flush=True)
    # primary branches on the READOUT (logits) stage, all stages tabulated above
    if r_far is not None and r_far < 1e-6:
        print(f"  readout Delta-h decays by >=1e6 between d=8 and d={far_d} "
              f"(d=512 is {r512 if r512 is not None else float('nan'):.1e} of d=8) -> "
              f"retention DIES well before d=512; LN-ceiling hypothesis STRENGTHENED",
              flush=True)
        if r32 is not None and r32 < 0.25:
            print(f"  d=32 readout is already {r32:.1%} of d=8 -> consistent with the "
                  f"copy-probe readout dying between d=8 and d=32", flush=True)
    elif r512 is not None and r512 > 0.05:
        print(f"  readout Delta-h at d=512 is {r512:.1%} of d=8 -> ALIVE at d=512; "
              f"LN-ceiling hypothesis WEAKENED", flush=True)
    else:
        print(f"  readout d=512 delta-h = {alive512[4] if alive512 else float('nan'):.3e} "
              f"(={100 * (r512 if r512 is not None else float('nan')):.3f}% of d=8), "
              f"d=32 = {100 * (r32 if r32 is not None else float('nan')):.1f}% of d=8 -> "
              f"intermediate: trace dies on a d~O(10-100) timescale; review the table",
              flush=True)
    for s, nm in STAGES.items():
        ratio_s = ratio[s] if ratio else None
        far_ratio_s = dl_ratio(s, far_d_sel, 8)
        print(f"  {nm:>14}: d=512/d=8={dl_ratio(s, 512, 8):.2e}  "
              f"d={max(dlist)}/d=8={dl_ratio(s, max(dlist), 8):.2e}  "
              f"dh(null95)@512={ratio_s if ratio_s is not None else float('nan'):.2f}", flush=True)

    print(f"\nall distances+stages done in {time.time() - t_start:.1f}s total", flush=True)

    if args.out:
        out = {
            "ckpt": args.ckpt, "distractors": args.distractors, "seed": args.seed,
            "trials_pairs": args.trials, "perm": args.perm,
            "sigma": {"mean": mean_sig, "min": float(sig.min()), "max": float(sig.max()),
                      "n_gt_2B_hl": int((half > 2).sum()),
                      "n_gt_4B_hl": int((half > 4).sum()),
                      "n_gt_8B_hl": int((half > 8).sum()),
                      "n_gt_32B_hl": int((half > 32).sum()),
                      "mean_half_life_B": float(half.mean())},
            "pred": {str(d): pred[d] for d in distances},
            "dh": {str(s): {str(dd): all_dh[s][dd] for dd in distances}
                   for s in STAGES},
            "null": {str(s): {str(dd): list(all_null[s][dd]) for dd in distances}
                     for s in STAGES},
            "implied_alpha_8_32": float(alpha_est),
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}", flush=True)

    print("DONE")


if __name__ == "__main__":
    main()