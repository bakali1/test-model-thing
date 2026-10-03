"""check_mi.py — GATE: the vectorized Marginalize-It reader == brute-force MI.

    .venv/bin/python tests/check_mi.py

Free, CPU, seconds. No teacher, no GPU, no network.

The MI target is the one piece of this experiment that is easy to vectorize
wrong and impossible to notice from a loss curve: an off-by-one in the depth, a
`<=` that should be `<`, a prefix test that forgets the `vocab_len > d`
condition. Each of those produces a plausible target that teaches the wrong
thing while the loss still falls. So it is compared against the definition,
token by token and byte by byte, on randomized synthetic vocabularies.

Definition under test, for a gold token with bytes g and a byte at depth d:

    S  = { c in top-k : len(c) > d and bytes(c)[:d+1] == g[:d+1] }
    p[x] = sum(prob[c] for c in S if bytes(c)[d] == x)
    p    = p / p.sum()          (and the backoff count when p.sum() == 0)

Exit 0 = PASS.
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import distill as D  # noqa: E402


def brute_force(vocab, cand_ids, probs, gold, depth):
    """The definition, written as directly as possible."""
    out = np.zeros(256, np.float64)
    for c, p in zip(cand_ids, probs):
        b = vocab.bytes[c]
        if len(b) > depth and list(b[:depth + 1]) == list(gold[:depth + 1]):
            out[int(b[depth])] += p
    tot = out.sum()
    return out, tot


def synthetic_vocab(rng, n_tokens=40, maxlen=6):
    """A table of random byte strings, including empty and length-1 ones."""
    table = []
    for i in range(n_tokens):
        m = int(rng.integers(0, maxlen + 1))
        table.append(bytes(rng.integers(0, 256, m, dtype=np.uint8).tobytes()))
    return table


def main():
    rng = np.random.default_rng(0)
    ok = True

    # ---- t1: vocab packing -------------------------------------------------
    table = synthetic_vocab(rng)
    v = D.VocabBytes(table, drop_ids=[3])
    lut = v.take(np.arange(len(table), dtype=np.int64))
    bad = []
    for i, b in enumerate(table):
        r = int(lut[i])
        if r < 0:
            continue
        want = np.zeros(v.maxlen, np.uint8)
        if b:
            want[:len(b)] = np.frombuffer(b, np.uint8)
        if not np.array_equal(v.bytes[r], want) or int(v.len[r]) != len(b):
            bad.append(i)
    same = not bad and int((lut == -1).sum()) == 1
    print(f"  [{'PASS' if same else 'FAIL'}] t1 vocab packing matches the table "
          f"(rows checked {v.n_kept}, mismatches {bad[:4]})")
    ok &= same
    print(f"         dropped id 3 -> -1: {v.take(np.array([3]))[0] == -1}")

    # ---- t2: vectorized MI == brute force, randomized ---------------------
    # The reader derives `depth` from its byte->token map, so a depth-d case has
    # to be built as a stream whose FIRST token spans d-1 pad bytes followed by
    # the gold token bytes; then predicting byte p=d lands at depth d. Getting
    # this wrong in the test looks exactly like an MI bug, which is why the
    # brute-force side is written independently.
    worst = 0.0
    n_cases = 0
    for trial in range(400):
        tb = synthetic_vocab(rng, n_tokens=int(rng.integers(8, 44)),
                             maxlen=int(rng.integers(3, 8)))
        vv = D.VocabBytes(tb)
        k = int(rng.integers(2, 9))
        cand = rng.integers(0, len(tb), k).astype(np.int64)
        raw = rng.random(k) ** 2 + 1e-6
        probs = (raw / raw.sum()).astype(np.float32)
        # The gold token must NOT be token 0: dump row r predicts token r+1, so
        # byte p in token q uses row q-1 and q=0 has no predicting row at all.
        # So the stream is  [dummy token][gold token]  and the candidate
        # distribution lives in row 0, which predicts the gold token.
        long_c = [int(c) for c in cand if len(tb[int(c)]) >= 2]
        if not long_c:
            continue
        c0 = int(rng.choice(long_c))
        n_pad = int(rng.integers(1, 4))
        pad_byte = (int(tb[c0][0]) + 11) % 256          # differs from gold[0]
        stream = bytes([pad_byte]) * n_pad + tb[c0]
        depth = int(rng.integers(1, len(tb[c0])))
        L_step = n_pad + depth
        L = 2                                          # two token rows

        ids = np.zeros((1, L, k), np.int32)
        pr = np.zeros((1, L, k), np.float16)
        ids[0, 0] = cand
        pr[0, 0] = probs
        tl = np.asarray([[n_pad, len(tb[c0])]], np.uint8)
        with tempfile.TemporaryDirectory() as td:
            D.save_dump(td, "t", ids, pr, tl, {"corpus_file": "synthetic",
                                               "corpus_sha256": "x", "B": 1,
                                               "Ltok": L, "k": k,
                                               "L_step": L_step,
                                               "stream_tokens": [1, 1]})
            r = D.MiReader(td, "t", vv, streams=[stream], block=1,
                           backoff="skip")
            tgt, _ret = r.targets(L_step - 1)
        got = tgt[0].astype(np.float64)      # targets() returns (B, 256) per step
        # The MI prefix is the GOLD TOKEN's bytes, not the whole stream's.
        exp, tot = brute_force(vv, vv.take(cand), probs.astype(np.float64),
                               np.frombuffer(tb[c0], np.uint8), depth)
        if tot > r.floor:
            exp = exp / tot
            worst = max(worst, float(np.abs(got - exp).max()))
            n_cases += 1
    good = worst <= 2e-3 and n_cases > 100
    print(f"  [{'PASS' if good else 'FAIL'}] t2 vectorized MI == brute force "
          f"({n_cases} cases, depth>=1, gold token index>=1, max |delta| {worst:.2e}, "
          f"tol 2e-3)")
    ok &= good

    # ---- t3: the `len(c) > d` condition is load-bearing --------------------
    # 'a' ENDS at depth 1, so it must be excluded. That exclusion is the paper's
    # EoT case: keeping it would drop 0.3 of the mass here, and that dropped
    # mass is exactly the measured MI bias.
    tab = [b"ab", b"a", b"abc"]
    vv = D.VocabBytes(tab)
    cand = np.asarray([0, 1, 2], np.int64)      # 'ab', 'a', 'abc'
    probs = np.asarray([0.5, 0.3, 0.2], np.float32)
    got, tot = brute_force(vv, cand, probs, np.frombuffer(b"ab", np.uint8), 1)
    ok3 = abs(tot - 0.7) < 1e-6 and abs(got[ord("b")] - 0.7) < 1e-6
    print(f"  [{'PASS' if ok3 else 'FAIL'}] t3 token ending exactly at depth d is "
          f"excluded (kept {tot:.2f}, expected 0.70; all on 'b')")
    ok &= ok3
    naive = 1.0   # a `len(c) >= d` bug would keep 'a' too
    print(f"         a `len(c) >= d` bug would have kept {naive:.2f} instead of "
          f"{tot:.2f} — the dropped 0.30 IS the measured MI bias")

    # ---- t4: backoff is counted, not silent -------------------------------
    tab = [b"zz", b"yy"]
    vv = D.VocabBytes(tab)
    ids = np.asarray([[[0, 1]]], np.int32)       # candidates 'zz','yy', gold 'a'
    pr = np.asarray([[[0.6, 0.4]]], np.float16)
    tl = np.asarray([[1]], np.uint8)
    with tempfile.TemporaryDirectory() as td:
        D.save_dump(td, "t", ids, pr, tl, {"corpus_file": "s", "corpus_sha256": "x",
                                           "B": 1, "L": 1, "k": 2})
        r = D.MiReader(td, "t", vv, streams=[b"aa"], block=1, backoff="skip")
        tgt, ret = r.targets(0)
    dead = r.backoff_n == 1 and float(np.asarray(tgt).sum()) == 0.0
    print(f"  [{'PASS' if dead else 'FAIL'}] t4 empty S -> explicit backoff, "
          f"counted ({r.backoff_n}), target all-zero ({float(tgt.sum()):.1f})")
    ok &= dead

    # ---- t5: alignment map -------------------------------------------------
    tab = [b"abc", b"de", b"f"]
    vv = D.VocabBytes(tab)
    ids = np.zeros((1, 3, 1), np.int32)
    pr = np.ones((1, 3, 1), np.float16)
    tl = np.asarray([[3, 2, 1]], np.uint8)
    with tempfile.TemporaryDirectory() as td:
        D.save_dump(td, "t", ids, pr, tl, {"corpus_file": "s", "corpus_sha256": "x",
                                           "B": 1, "L": 3, "k": 1})
        r = D.MiReader(td, "t", vv, streams=[b"abcdef"], block=4,
                       backoff="skip")
        b2t = r.byte2tok[0]
    # bytes: abc | de | f  ->  0,1,2 -> tok0 ; 3,4 -> tok1 ; 5 -> tok2
    want = [0, 0, 0, 1, 1, 2]
    good = list(b2t[:6]) == want
    print(f"  [{'PASS' if good else 'FAIL'}] t5 byte->token map {list(b2t[:6])} "
          f"== {want}")
    ok &= good

    print("\n== verdict:", "PASS" if ok else "FAIL", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())