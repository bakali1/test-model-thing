#!/usr/bin/env python3
"""copy_probe.py - Measure whether the TMT checkpoint can store and retrieve
a 4-byte key through d distractor bytes.

Protocol per trial:
  Feed: KEY(4B) + d distractors + key_again(4B) + 4 probe bytes
  KEY = [0x41, 0x42, 0x43, 0x44] (fixed for all trials)
  key_again = KEY (same-key) or [0x00,0x01,0x02,0x03] (different-key)

Two measures:
  a) READOUT CONTROL: KL(output | same_key) vs KL(output | diff_key)
     over the 4 probe bytes. Positive KL = state carries the key.
  b) COPY TASK: P(top-1 correct KEY byte) at each probe position,
     vs chance 1/256 = 0.39%.
"""
import os, sys, time, argparse
from math import erfc, sqrt
import numpy as np

print('SCRIPT START', flush=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

print('BEFORE IMPORTS', flush=True)
import numpy_port as N
from eval_npz import infer_config
print('AFTER IMPORTS', flush=True)

KEY = np.array([0x41, 0x42, 0x43, 0x44], dtype=np.uint8)
print('KEYS SET', flush=True)
DIFFERENT_KEY = np.array([0x00, 0x01, 0x02, 0x03], dtype=np.uint8)
DISTANCES = [8, 32, 128, 512, 2048]
VOCAB = 256
CHANCE_PER_BYTE = 1.0 / 256
CHANCE_4BYTE = 1.0 / (256 ** 4)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def binomial_pvalue(k, n, p0):
    """One-sided normal approximation: P(X >= k) under Binomial(n, p0)."""
    if n == 0:
        return 1.0
    mean = n * p0
    std = sqrt(max(n * p0 * (1 - p0), 1e-30))
    if std < 1e-30:
        return 0.0 if k > mean else 1.0
    z = (k - mean) / std
    return 0.5 * erfc(z / sqrt(2.0))


def run_readout_control(m, same_dist, diff_dist, n_same, n_diff, d_dist):
    """Readout control: same-key vs different-key trials.

    Feeds KEY + distractors + key_again + KEY[0:4] for both conditions.
    Returns output distributions at each probe position.
    """
    B = n_same + n_diff
    L_rc = 4 + d_dist + 4  # KEY + distractors + key_again

    # Build same-key sequence: KEY + same_dist + KEY
    same_seq = np.zeros((n_same, L_rc), dtype=np.uint8)
    same_seq[:, :4] = KEY
    same_seq[:, 4:4 + d_dist] = same_dist
    same_seq[:, 4 + d_dist:4 + d_dist + 4] = KEY

    # Build different-key sequence: KEY + diff_dist + DIFFERENT_KEY
    diff_seq = np.zeros((n_diff, L_rc), dtype=np.uint8)
    diff_seq[:, :4] = KEY
    diff_seq[:, 4:4 + d_dist] = diff_dist
    diff_seq[:, 4 + d_dist:4 + d_dist + 4] = DIFFERENT_KEY

    # Concatenate into one batch
    full_seq = np.vstack([same_seq, diff_seq])  # (B, L_rc)

    # Feed through model
    m.new_states(B)
    for t in range(L_rc):
        enc = m.embed[full_seq[:, t]]
        x = enc
        for l in m.layers:
            x, st, _d, _c = l.forward(enc, x, l.states)
            l.states = st

    # x is state after key_again. Get distribution for probe byte 0.
    logits = x @ m.Wd.T + m.bd
    p = softmax(logits)
    p_same = np.zeros((n_same, 4, VOCAB), dtype=np.float64)
    p_diff = np.zeros((n_diff, 4, VOCAB), dtype=np.float64)
    p_same[:, 0] = p[:n_same]
    p_diff[:, 0] = p[n_same:]

    # Feed probe bytes (KEY) one at a time
    for i in range(4):
        enc = m.embed[np.full(B, KEY[i], dtype=np.uint8)]
        x = enc
        for l in m.layers:
            x, st, _d, _c = l.forward(enc, x, l.states)
            l.states = st
        if i < 3:
            logits = x @ m.Wd.T + m.bd
            p = softmax(logits)
            p_same[:, i + 1] = p[:n_same]
            p_diff[:, i + 1] = p[n_same:]

    return p_same, p_diff


def run_copy_task(m, same_dist, n_same, d_dist):
    """Copy task: autoregressively predict KEY after KEY + distractors + KEY.

    Returns predictions (n_same, 4) and probabilities (n_same, 4, VOCAB).
    """
    L_copy = 4 + d_dist + 4
    same_seq = np.zeros((n_same, L_copy), dtype=np.uint8)
    same_seq[:, :4] = KEY
    same_seq[:, 4:4 + d_dist] = same_dist
    same_seq[:, 4 + d_dist:4 + d_dist + 4] = KEY

    m.new_states(n_same)
    for t in range(L_copy):
        enc = m.embed[same_seq[:, t]]
        x = enc
        for l in m.layers:
            x, st, _d, _c = l.forward(enc, x, l.states)
            l.states = st

    # Autoregressively predict 4 bytes
    preds = np.zeros((n_same, 4), dtype=np.int64)
    probs = np.zeros((n_same, 4, VOCAB), dtype=np.float64)

    for i in range(4):
        logits = x @ m.Wd.T + m.bd
        p = softmax(logits)
        probs[:, i] = p
        pred = p.argmax(axis=1)
        preds[:, i] = pred

        if i < 3:
            enc = m.embed[pred.astype(np.uint8)]
            x = enc
            for l in m.layers:
                x, st, _d, _c = l.forward(enc, x, l.states)
                l.states = st

    return preds, probs


def main():
    parser = argparse.ArgumentParser(description="Copy probe: key storage/retrieval through distractors")
    parser.add_argument("--distractors", choices=["random", "real"], default="random")
    parser.add_argument("--trials", type=int, default=384)
    parser.add_argument("--batch", type=int, default=384)
    parser.add_argument("--ckpt", default=os.path.join(HERE, "runs/lrroles/arm_lrdecay1x_t919897.npz"))
    parser.add_argument("--distractor-file", default=os.path.join(HERE, "data/eval/tinystories_valid.txt"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt_data = np.load(args.ckpt)
    dim, layers = infer_config(ckpt_data)
    print(f"  dim={dim}, layers={layers}, state_dim={dim * layers}")
    m = N.Model(dim=dim, layers=layers, temp=0.75, lr=5e-4)
    m.new_states(1)
    m.load(args.ckpt)
    m.new_states(1)

    # Load distractor data
    if args.distractors == "real":
        raw_bytes = np.frombuffer(open(args.distractor_file, "rb").read(), dtype=np.uint8)
        print(f"Loaded {len(raw_bytes)} distractor bytes from {args.distractor_file}")
    else:
        raw_bytes = None

    n_trials = args.trials
    n_same = n_trials // 2
    n_diff = n_trials - n_same

    print(f"Trials: {n_trials} per distance ({n_same} same-key, {n_diff} different-key)")
    print(f"Distances: {DISTANCES}")
    print(f"Key: {list(KEY)} (0x{KEY[0]:02x} 0x{KEY[1]:02x} 0x{KEY[2]:02x} 0x{KEY[3]:02x})")
    print(f"Different key: {list(DIFFERENT_KEY)}")
    print()

    start_time = time.time()
    all_results = []

    for d_dist in DISTANCES:
        t0 = time.time()
        print(f"--- Distance d={d_dist} ---")

        # Generate distractors for all trials
        n_total = n_same + n_diff
        if args.distractors == "random":
            all_distractors = rng.integers(0, VOCAB, size=(n_total, d_dist), dtype=np.uint8)
        else:
            all_distractors = np.zeros((n_total, d_dist), dtype=np.uint8)
            n_available = max(len(raw_bytes) - d_dist, 1)
            for i in range(n_total):
                off = rng.integers(0, n_available)
                all_distractors[i] = raw_bytes[off:off + d_dist]

        same_dist = all_distractors[:n_same]
        diff_dist = all_distractors[n_same:]

        # ============================================================
        # READOUT CONTROL
        # ============================================================
        p_same, p_diff = run_readout_control(m, same_dist, diff_dist, n_same, n_diff, d_dist)

        # Compute KL divergence at each probe position
        kl_per_position = np.zeros(4)
        for i in range(4):
            p_s = p_same[:, i].mean(axis=0)
            p_d = p_diff[:, i].mean(axis=0)
            p_s = np.clip(p_s, 1e-10, 1.0)
            p_d = np.clip(p_d, 1e-10, 1.0)
            p_s /= p_s.sum()
            p_d /= p_d.sum()
            kl = np.sum(p_s * np.log(p_s / p_d))
            kl_per_position[i] = kl

        kl_mean = float(kl_per_position.mean())
        kl_std = float(kl_per_position.std())

        # ============================================================
        # COPY TASK
        # ============================================================
        preds, probs = run_copy_task(m, same_dist, n_same, d_dist)

        correct = (preds == KEY)  # (n_same, 4)
        accuracy_per_byte = correct.mean(axis=0)

        # Binomial test for each byte position
        p_values = np.zeros(4)
        for i in range(4):
            k = int(correct[:, i].sum())
            p_values[i] = binomial_pvalue(k, n_same, CHANCE_PER_BYTE)

        # Overall 4-byte accuracy
        all_correct = correct.all(axis=1)
        acc_4byte = float(all_correct.mean())

        elapsed_d = time.time() - t0

        # Print results
        print(f"  Readout control (KL p_same || p_diff):")
        for i in range(4):
            marker = "POSITIVE" if kl_per_position[i] > 0 else "zero"
            print(f"    Probe byte {i}: KL = {kl_per_position[i]:.6f}  [{marker}]")
        print(f"    Mean KL: {kl_mean:.6f} +/- {kl_std:.6f}")
        readout_positive = bool(kl_mean > 0)

        print(f"  Copy task (top-1 accuracy vs KEY):")
        for i in range(4):
            beats_chance = "YES" if p_values[i] < 0.05 else "no"
            print(f"    Byte {i} (0x{KEY[i]:02x}): acc = {accuracy_per_byte[i]:.6f} "
                  f"(chance = {CHANCE_PER_BYTE:.6f}, p = {p_values[i]:.6e})  [{beats_chance}]")
        print(f"    4-byte accuracy: {acc_4byte:.6e} (chance = {CHANCE_4BYTE:.6e})")
        copy_beats = bool(acc_4byte > CHANCE_4BYTE)

        print(f"  Time: {elapsed_d:.1f}s")
        print()

        all_results.append({
            'd': d_dist,
            'distractor_type': args.distractors,
            'kl_per_position': kl_per_position.copy(),
            'kl_mean': kl_mean,
            'kl_std': kl_std,
            'accuracy_per_byte': accuracy_per_byte.copy(),
            'p_values': p_values.copy(),
            'acc_4byte': acc_4byte,
            'readout_positive': readout_positive,
            'copy_beats': copy_beats,
        })

    total_time = time.time() - start_time

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: distractors={args.distractors}")
    print(f"{'=' * 80}")
    print(f"{'d':>6} {'KL_mean':>10} {'KL_std':>10} {'acc_b0':>10} {'acc_b1':>10} "
          f"{'acc_b2':>10} {'acc_b3':>10} {'acc_4B':>12}")
    for r in all_results:
        print(f"{r['d']:>6} {r['kl_mean']:>10.6f} {r['kl_std']:>10.6f} "
              f"{r['accuracy_per_byte'][0]:>10.6f} {r['accuracy_per_byte'][1]:>10.6f} "
              f"{r['accuracy_per_byte'][2]:>10.6f} {r['accuracy_per_byte'][3]:>10.6f} "
              f"{r['acc_4byte']:>12.6e}")

    readout_any = any(r['readout_positive'] for r in all_results)
    copy_any = any(r['copy_beats'] for r in all_results)
    print(f"\nReadout positive at any d: {'YES' if readout_any else 'no'}")
    print(f"Copy accuracy beats chance at any d: {'YES' if copy_any else 'no'}")
    print(f"Total wall-clock time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
