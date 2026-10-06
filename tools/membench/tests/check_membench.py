#!/usr/bin/env python3
"""check_membench.py -- smoke test / acceptance gate for tools/membench/.

Covers every check the Priority-2 + section-10 deliverables require:

  (a) payload statistically independent of the key  -- chi-square + permutation
      null on generated batches (task B, plus a Q on C's distractor pairs)
  (b) deterministic given seed                      -- two runs with the same
      seed are byte-identical, per task
  (c) chance-level math correct per task            -- analytic values AND
      empirical rates of a null guesser on generated data
  (d) eval_api contract                             -- every §10 metadata key
      + metric present in every return dict; model-backed paths raise
      NotImplementedError
  (e) selective_copy structure                      -- 1..3 distractor
      key/value pairs at valid random positions, random per-sample gaps,
      unique keys

Run:  python3 tools/membench/tests/check_membench.py
Exit 0 = PASS.
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(os.path.dirname(HERE))     # tools/
REPO = os.path.dirname(TOOLS)
for _p in (REPO, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import membench as mb

VOCAB = mb.VOCAB
_PASS = []


def check(name, fn):
    fn()
    _PASS.append(name)
    print(f"[PASS] {name}")


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------
# (a) payload statistically independent of the key
# --------------------------------------------------------------------------
def _chi2_stat(K, P, vocab):
    K = K.reshape(-1).astype(np.int64)
    P = P.reshape(-1).astype(np.int64)
    total = K.size
    tab = np.bincount(K * vocab + P, minlength=vocab * vocab,
                      ).reshape(vocab, vocab).astype(np.float64)
    row = tab.sum(axis=1)
    col = tab.sum(axis=0)
    exp = np.outer(row, col) / total
    m = exp > 0
    stat = float(((tab[m] - exp[m]) ** 2 / exp[m]).sum())
    return stat, tab, exp, total


def test_payload_independent_of_key():
    rng = np.random.default_rng(0)
    n = 100_000
    b = mb.get_batch("associative_copy", 32, n, rng=int(rng.integers(2**32)))
    keys = np.stack([s[: mb.KEY_LEN] for s in b["seq"]])      # (n, 4)
    payloads = b["target"]                                    # (n, 4)

    # Pool all 16 (key_byte, payload_byte) pairs: each is an iid joint draw.
    stat, tab, exp, total = _chi2_stat(keys, payloads, VOCAB)
    dof = (VOCAB - 1) ** 2
    mean, sd = float(dof), math.sqrt(2.0 * float(dof))
    threshold = mean + 6.0 * sd
    _assert(stat < threshold,
            f"chi2 {stat:.1f} exceeds independence threshold {threshold:.1f} "
            f"(mean {mean:.1f}, sd {sd:.1f})")

    # Permutation null (robustness display + sanity): shuffle payload axis.
    nulls = []
    for _ in range(7):
        statp, *_ = _chi2_stat(keys, rng.permutation(payloads), VOCAB)
        nulls.append(statp)
    _assert(stat <= max(nulls) * 1.05 + 1.0,
            f"observed chi2 {stat:.1f} above permutation null max "
            f"{max(nulls):.1f}")
    print(f"        chi2={stat:9.1f}  null(mean={np.mean(nulls):8.1f}, "
          f"max={max(nulls):8.1f})  threshold={threshold:9.1f} "
          f"(df={dof}, n_pairs={total})")

    # Marginals uniform (sanity of the generator itself).
    row_min = tab.sum(axis=1).min()
    _assert(row_min > total / VOCAB - 6 * math.sqrt(total), "key marginals not uniform")
    col_min = tab.sum(axis=0).min()
    _assert(col_min > total / VOCAB - 6 * math.sqrt(total), "payload marginals not uniform")

    # Bonus: C's distractor pair payloads must also be independent of the
    # query key (light check on first distractor pair of each sample).
    bc = mb.get_batch("selective_copy", 128, n // 2, rng=int(rng.integers(2**32)))
    ks, ps = [], []
    for i, (s, infos) in enumerate(zip(bc["seq"], bc["meta"]["distractor_pairs"])):
        if not infos:
            continue
        ks.append(s[: mb.KEY_LEN])
        ps.append(np.frombuffer(infos[0][1], dtype=np.uint8))
    stat2, *_ = _chi2_stat(np.stack(ks), np.stack(ps), VOCAB)
    _assert(stat2 < mean + 6.0 * sd,
            f"distractor payload chi2 {stat2:.1f} above threshold")
    print(f"        [C distractor pairs] chi2={stat2:9.1f} threshold={threshold:9.1f}")


# --------------------------------------------------------------------------
# (b) deterministic given seed
# --------------------------------------------------------------------------
def test_deterministic_given_seed():
    cases = [("delayed_copy", 64), ("associative_copy", 512),
             ("selective_copy", 512), ("long_range_language", 256)]
    for name, d in cases:
        a = mb.get_batch(name, d, 64, rng=1234)
        b = mb.get_batch(name, d, 64, rng=1234)
        _assert(a["meta"]["seed"] == b["meta"]["seed"], f"{name}: seed mismatch")
        _assert(all(np.array_equal(x, y) for x, y in zip(a["seq"], b["seq"])),
                f"{name}: seq not byte-identical for same seed")
        _assert(np.array_equal(a["target"], b["target"]),
                f"{name}: target not byte-identical for same seed")
        if name == "selective_copy":
            _assert(a["meta"]["distractor_pairs"] == b["meta"]["distractor_pairs"],
                    f"{name}: distractor layout not identical for same seed")
            _assert(a["meta"]["pair_positions"] == b["meta"]["pair_positions"],
                    f"{name}: pair positions not identical for same seed")
        # a different seed must (overwhelmingly) produce different data
        c = mb.get_batch(name, d, 12, rng=987654321)
        _assert(not all(np.array_equal(x, y) for x, y in zip(a["seq"][:12], c["seq"])),
                f"{name}: different seeds produced identical data")


# --------------------------------------------------------------------------
# (c) chance-level math correct per task
# --------------------------------------------------------------------------
def test_chance_level_math():
    # --- analytic values ---
    for t in ("delayed_copy", "associative_copy", "selective_copy"):
        cl = mb.CHANCE_LEVELS[t]
        _assert(abs(cl["per_byte"] - 1.0 / VOCAB) < 1e-15, f"{t}: per_byte chance")
        _assert(abs(cl["exact_payload"] - (1.0 / VOCAB) ** mb.PAYLOAD_LEN) < 1e-30,
                f"{t}: exact payload chance")
    cl = mb.CHANCE_LEVELS["long_range_language"]
    _assert(abs(cl["critical_position"] - 0.5) < 1e-15, "D: critical chance 1/2")
    _assert(abs(cl["critical_position_uniform_guess"] - 1.0 / VOCAB) < 1e-15,
            "D: uniform-guess critical chance 1/256")
    _assert(cl["terminator"] == 1.0, "D: terminator deterministic")

    # chance CE expression for D, two lengths
    for L in (8, 512):
        expected = ((L - 3) * math.log(VOCAB) + math.log(2)) / (L - 1)
        _assert(abs(mb.chance_ce("long_range_language", L=L) - expected) < 1e-12,
                f"chance_ce(D, L={L}) mismatch")

    # --- empirical: a null guesser hits chance on generated data ---
    rng = np.random.default_rng(1)

    # A/B/C: uniform byte guess vs 4-byte payloads -> ~1/256 per byte.
    for t in ("delayed_copy", "selective_copy"):
        b = mb.get_batch(t, 128, 16_384, rng=int(rng.integers(2**32)))
        guess = rng.integers(0, VOCAB, size=b["target"].shape)
        rate = float((guess == b["target"]).mean())
        p0 = 1.0 / VOCAB
        se = math.sqrt(p0 * (1 - p0) / b["target"].size)
        _assert(abs(rate - p0) <= 6 * se,
                f"{t}: null-guess byte match rate {rate:.5f} vs chance {p0:.5f} "
                f"(6se={6 * se:.5f})")

    # D: structure + the "cannot be decided from the last few bytes" property.
    L = 512
    b = mb.get_batch("long_range_language", L, 32_768, rng=int(rng.integers(2**32)))
    markers = np.array([s[0] for s in b["seq"]], dtype=np.uint8)
    crit = np.array([s[L - 2] for s in b["seq"]], dtype=np.uint8)
    _assert(set(markers.tolist()) <= {0x00, 0x01}, "D: marker alphabet")
    expected = np.where(markers == 0x00, 0x41, 0x42).astype(np.uint8)
    _assert(np.array_equal(crit, expected), "D: marker controls critical byte")
    _assert(all(int(s[L - 1]) == mb.TERMINATOR for s in b["seq"]),
            "D: terminator byte")

    se256 = math.sqrt((1 / VOCAB) * (1 - 1 / VOCAB) / len(crit))
    se2 = math.sqrt(0.5 * 0.5 / len(crit))
    # (i) uniform-256 guesser hits the binary controlled byte at 1/256
    g256 = rng.integers(0, VOCAB, size=len(crit))
    r256 = float((g256 == crit).mean())
    _assert(abs(r256 - 1 / VOCAB) <= 6 * se256,
            f"D: uniform guess rate {r256:.5f} vs 1/256")
    # (ii) alphabet-aware guesser (knows A/B alphabet, not the marker): 1/2
    g2 = rng.choice(mb.CONTROLLED_ALPHABET, size=len(crit))
    r2 = float((g2 == crit).mean())
    _assert(abs(r2 - 0.5) <= 6 * se2,
            f"D: alphabet-aware rate {r2:.5f} vs 1/2")
    # (iii) the last-few-bytes rule "repeat previous byte" is useless: 1/256
    prev = np.array([s[L - 3] for s in b["seq"]], dtype=np.uint8)
    rp = float((prev == crit).mean())
    _assert(abs(rp - 1 / VOCAB) <= 6 * se256,
            f"D: last-byte rule rate {rp:.5f} vs 1/256 (must NOT be decidable)")

    print(f"        A/C null-guess ~1/256; D: uniform {r256:.4f}, "
          f"alphabet-aware {r2:.4f}, last-byte {rp:.4f} "
          f"(n={len(crit)}, L={L})")


# --------------------------------------------------------------------------
# (d) eval_api contract
# --------------------------------------------------------------------------
def test_eval_api_contract():
    METADATA = ("cue_type", "gap_distribution", "distractor_type",
                "state_reset_policy", "seed", "sample_count", "chance_level")
    cases = [
        (mb.retention_curve, dict(d_list=[8, 32, 128]), "retention_curve"),
        (mb.selective_copy_accuracy, dict(d=128), "selective_copy_accuracy"),
        (mb.associative_retrieval_accuracy, dict(d=128),
         "associative_retrieval_accuracy"),
        (mb.language_ce, dict(L=512), "language_ce"),
        (mb.transfer_ce, {}, "transfer_ce"),
        (mb.cross_domain_ce, {}, "cross_domain_ce"),
    ]
    for fn, kw, name in cases:
        r = fn(**kw)
        for k in METADATA:
            _assert(k in r and r[k] is not None, f"{name}: missing §10 key {k!r}")
        _assert(r["metric"] == name, f"{name}: metric name wrong: {r['metric']!r}")
        _assert("value" in r and "unit" in r, f"{name}: metric value missing")
        _assert(r["metric_source"] != "model", f"{name}: should be a baseline")
        _assert(r["requires_model"] is True, f"{name}: real metric needs a model")
        # model-backed path must raise NotImplementedError
        try:
            fn(model=object(), **kw)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{name}: expected NotImplementedError with a model")

    # numeric spot checks on the model-free baselines
    r = mb.selective_copy_accuracy(128)
    _assert(abs(r["value"] - 2 ** -32) < 1e-30, "selective chance != 2^-32")
    r = mb.associative_retrieval_accuracy(512)
    _assert(abs(r["value"] - 2 ** -32) < 1e-30, "associative chance != 2^-32")
    r = mb.language_ce(L=512)
    _assert(abs(r["value"] - mb.chance_ce("long_range_language", L=512)) < 1e-12,
            "language_ce chance != chance_ce(D)")
    r = mb.retention_curve([8, 32, 128])
    _assert(r["value"] == {8: 0.0, 32: 0.0, 128: 0.0}, "retention null must be 0")
    _assert(abs(mb.transfer_ce()["value"] - math.log(VOCAB)) < 1e-12,
            "transfer chance != ln 256")
    _assert(abs(mb.cross_domain_ce()["value"] - math.log(VOCAB)) < 1e-12,
            "cross_domain chance != ln 256")


# --------------------------------------------------------------------------
# (e) selective_copy structure
# --------------------------------------------------------------------------
def test_selective_copy_structure():
    SEED, D, N = 5, 512, 300
    b = mb.get_batch("selective_copy", D, N, rng=SEED)
    meta = b["meta"]
    gaps = meta["gaps"]
    counts = meta["distractor_counts"]

    _assert(all(g in mb.GAP_LADDER for g in gaps), "gap outside ladder")
    _assert(all(g <= D for g in gaps), "gap above requested d")
    _assert(len(set(gaps.tolist())) >= 2,
            f"gaps not random per sample (only {sorted(set(gaps.tolist()))})")
    _assert(all(1 <= c <= 3 for c in counts), "distractor pair count outside 1..3")

    for i, s in enumerate(b["seq"]):
        gap = int(gaps[i])
        _assert(len(s) == 12 + gap, f"sample {i}: seq length {len(s)} != 12+{gap}")
        _assert(np.array_equal(s[:4], s[len(s) - 4:]),
                f"sample {i}: query key not repeated at the end")
        _assert(bytes(s[4:8]) == b["target"][i].tobytes(),
                f"sample {i}: target != query payload")
        region = s[8:8 + gap]
        n_dist = int(counts[i])
        poslist = meta["pair_positions"][i]
        pairs = meta["distractor_pairs"][i]
        _assert(len(poslist) == n_dist == len(pairs),
                f"sample {i}: recorded pairs mismatch")
        seen_keys = {s[:4].tobytes()}
        for (kbytes, pbytes), pos in zip(pairs, poslist):
            _assert(0 <= pos <= gap - 8, f"sample {i}: pair pos {pos} out of gap")
            _assert(region[pos:pos + 4].tobytes() == kbytes,
                    f"sample {i}: key bytes not at recorded position")
            _assert(region[pos + 4:pos + 8].tobytes() == pbytes,
                    f"sample {i}: payload bytes not at recorded position")
            _assert(kbytes not in seen_keys, f"sample {i}: duplicate key")
            seen_keys.add(kbytes)

    # length/dtype contract of the batch container
    _assert(all(len(s) % 1 == 0 and s.dtype == np.uint8 for s in b["seq"]),
            "seq must be uint8")
    _assert(b["target"].dtype == np.uint8 and b["target"].shape == (N, 4),
            "target shape/dtype wrong")
    print(f"        gaps seen: {sorted(set(gaps.tolist()))}; pair counts: "
          f"{sorted(set(counts.tolist()))} (n={N})")


# --------------------------------------------------------------------------
def main():
    print(f"membench {mb.__version__} -- smoke test (Python "
          f"{sys.version.split()[0]}, numpy {np.__version__})")
    check("(a) payload independent of key (chi2 + permutation null)", 
          test_payload_independent_of_key)
    check("(b) deterministic given seed", test_deterministic_given_seed)
    check("(c) chance-level math correct per task", test_chance_level_math)
    check("(d) eval_api §10 contract on every return dict",
          test_eval_api_contract)
    check("(e) selective_copy: 1-3 distractor pairs + random gaps",
          test_selective_copy_structure)
    print(f"\nALL {len(_PASS)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())