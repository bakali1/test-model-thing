"""check_mi.py — GATE: the distillation target == the paper's definition.

    .venv/bin/python tests/check_mi.py

Free, CPU, seconds. No teacher, no GPU, no network.

⚠ THIS GATE'S SEMANTICS CHANGED ON 2026-10-03, deliberately. It previously
asserted the *gold-prefix-conditioned* target (the paper's depth loop read
literally) and passed — while that target was measurably a POINT MASS at the
gold byte (entropy exactly 0.000000 nats, argmax == gold on 100.000000% of
positions), which made `alpha*KL + (1-alpha)*CE` a reweighted CE. A gate that
encodes a wrong definition is worse than no gate, so the definition is now the
one that carries information:

    depth 0  (byte BEGINS a token, ~1 position in 4.1):
        p_target[x] = sum(prob[c] for c in top-k if first_byte(c) == x), then / sum
        -- the paper's Case I, with NO conditioning on the gold byte
    depth >=1 (inside a token): NO TARGET. Declined and counted, because one
        teacher forward pass conditioned on the gold prefix cannot represent byte
        uncertainty there.

t6 is the tripwire that would have caught the original defect: the depth-0
target must put mass on bytes OTHER than the gold one. If it is one-hot again,
the KD term is doing nothing and the gate says so.

Exit 0 = PASS.
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import distill as D  # noqa: E402


def synthetic_vocab(rng, n_tokens=40, maxlen=6):
    table = []
    for _ in range(n_tokens):
        m = int(rng.integers(1, maxlen + 1))          # non-empty by construction
        table.append(bytes(rng.integers(0, 256, m, dtype=np.uint8).tobytes()))
    return table


def brute_force_case1(vocab, cand_rows, probs):
    """The paper's Case I: aggregate the kept tokens by FIRST BYTE."""
    out = np.zeros(256, np.float64)
    for row, p in zip(cand_rows, probs):
        out[int(vocab.first[row])] += p
    tot = out.sum()
    return (out / tot if tot > 0 else out), tot


def make_reader(td, table, vocab, streams, ntok_rows=2, L_step=None, k=4, B=1):
    ids = np.zeros((B, ntok_rows, k), np.int32)
    pr = np.zeros((B, ntok_rows, k), np.float16)
    for r in range(ntok_rows):
        ids[0, r, 0] = r % len(table)
        pr[0, r, 0] = 1.0
    tl = np.asarray([[len(s)] for s in streams], np.uint8)
    D.save_dump(td, "t", ids, pr, tl,
                {"corpus_file": "synthetic", "corpus_sha256": "x", "B": B,
                 "Ltok": ntok_rows, "k": k, "L_step": L_step or len(streams[0]),
                 "stream_tokens": [1] * B})
    return D.MiReader(td, "t", vocab, streams=streams, block=4, backoff="skip")


def main():
    rng = np.random.default_rng(0)
    ok = True

    def t(name, good, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if good else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        ok &= bool(good)

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
        want[:len(b)] = np.frombuffer(b, np.uint8)
        if not np.array_equal(v.bytes[r], want) or int(v.len[r]) != len(b):
            bad.append(i)
    t("t1 vocab packing matches the table",
      not bad and int((lut == -1).sum()) == 1,
      f"rows {v.n_kept}, mismatches {bad[:4]}, dropped id 3 -> -1")

    # ---- t2: depth-0 target == brute-force Case I -------------------------
    worst = 0.0
    n_cases = 0
    for _ in range(300):
        tb = synthetic_vocab(rng, n_tokens=int(rng.integers(6, 40)),
                             maxlen=int(rng.integers(2, 8)))
        vv = D.VocabBytes(tb)
        k = int(rng.integers(2, 9))
        cand = rng.integers(0, len(tb), k).astype(np.int64)
        raw = rng.random(k) ** 2 + 1e-6
        probs = (raw / raw.sum()).astype(np.float32)
        # stream = [pad token][gold token]; byte len(pad) is depth 0 of token 1,
        # which is what row 0 predicts. Non-empty tokens only.
        g = int(rng.integers(1, len(tb)))
        stream = bytes([int(tb[g][0]) % 200 + 10]) + tb[g]
        with tempfile.TemporaryDirectory() as td:
            ids = np.zeros((1, 2, k), np.int32)
            pr = np.zeros((1, 2, k), np.float16)
            ids[0, 0] = cand
            pr[0, 0] = probs
            tl = np.asarray([[1, len(tb[g])]], np.uint8)
            D.save_dump(td, "t", ids, pr, tl,
                        {"corpus_file": "s", "corpus_sha256": "x", "B": 1,
                         "Ltok": 2, "k": k, "L_step": 1, "stream_tokens": [1, 1]})
            r = D.MiReader(td, "t", vv, streams=[stream], block=1, backoff="skip")
            tgt, produced = r.targets(0)
        if float(produced[0]) == 0.0:
            continue
        got = np.asarray(tgt[0], np.float64).ravel()
        exp, tot = brute_force_case1(vv, vv.take(cand), probs.astype(np.float64))
        if tot > 0:
            worst = max(worst, float(np.abs(got - exp).max()))
            n_cases += 1
    t("t2 depth-0 target == brute-force Case I (no gold conditioning)",
      worst <= 2e-3 and n_cases > 100,
      f"{n_cases} cases, max |delta| {worst:.2e}, tol 2e-3 (fp16 storage)")

    # ---- t3: depths >= 1 are DECLINED, not silently given a one-hot -------
    # stream [a][bc][de] = "abcde": byte1 starts token 'bc', byte2 is depth 1,
    # byte3 starts token 'de', byte4 is depth 1. Byte 0 is never PREDICTED (it is
    # the input), which is why a token start at byte 0 produces nothing.
    tb = [b"abc", b"de", b"fgh"]
    vv = D.VocabBytes(tb)
    ids = np.zeros((1, 2, 3), np.int32)
    pr = np.zeros((1, 2, 3), np.float16)
    ids[0, 0] = [0, 1, 2]          # row 0 predicts token 1 ('de')
    pr[0, 0] = [0.5, 0.3, 0.2]
    ids[0, 1] = [0, 1, 2]
    pr[0, 1] = [0.2, 0.3, 0.5]
    tl = np.asarray([[1, 2, 2]], np.uint8)
    ids = np.concatenate([ids, ids[:, :1, :]], axis=1)
    pr = np.concatenate([pr, pr[:, :1, :]], axis=1)
    with tempfile.TemporaryDirectory() as td:
        D.save_dump(td, "t", ids, pr, tl,
                    {"corpus_file": "s", "corpus_sha256": "x", "B": 1, "Ltok": 3,
                     "k": 3, "L_step": 5, "stream_tokens": [3]})
        r = D.MiReader(td, "t", vv, streams=[b"abcde"], block=8, backoff="skip")
        got = [float(r.targets(i)[1][0]) for i in range(4)]
        tgt0 = np.asarray(r.targets(0)[0])
    # steps 0 and 2 predict the first byte of 'bc' and 'de' -> produced;
    # steps 1 and 3 predict depth-1 bytes -> declined
    declined_ok = (got[0] == 1.0 and got[1] == 0.0 and got[2] == 1.0
                   and got[3] == 0.0)
    t("t3 only depth 0 is produced; depths >= 1 are declined", declined_ok,
      f"produced by step: {got} (want [1,0,1,0]); declined={r.declined} of "
      f"{r.positions}")
    t("t3b depth-0 byte distribution is an aggregate, not a point mass",
      float(tgt0.sum()) > 0.99 and int((tgt0[0] > 1e-6).sum()) >= 2,
      f"mass {float(tgt0.sum()):.4f} over {int((tgt0[0] > 1e-6).sum())} bytes")

    # ---- t4: all-empty candidates -> no target, counted -------------------
    # byte 1 must be a TOKEN START for this to test the empty path rather than
    # the declined path (that mistake made an earlier version of this test pass
    # for the wrong reason). tokens are both EMPTY, so nothing contributes.
    tb = [b"", b""]
    vv = D.VocabBytes(tb)
    with tempfile.TemporaryDirectory() as td:
        ids = np.asarray([[[0, 1], [0, 1]]], np.int32)
        pr = np.asarray([[[0.6, 0.4], [0.6, 0.4]]], np.float16)
        tl = np.asarray([[1, 1]], np.uint8)
        D.save_dump(td, "t", ids, pr, tl,
                    {"corpus_file": "s", "corpus_sha256": "x", "B": 1, "Ltok": 2,
                     "k": 2, "L_step": 2, "stream_tokens": [2]})
        r = D.MiReader(td, "t", vv, streams=[b"zz"], block=1, backoff="skip")
        tg, prd = r.targets(0)
    t("t4 token start with no usable candidate -> no target, counted as empty",
      float(prd[0]) == 0.0 and float(np.asarray(tg).sum()) == 0.0
      and r.empty >= 1,
      f"empty={r.empty} (this is the EMPTY path, not the declined one)")

    # ---- t5: byte->token map ---------------------------------------------
    tb = [b"abc", b"de", b"f"]
    vv = D.VocabBytes(tb)
    with tempfile.TemporaryDirectory() as td:
        ids = np.zeros((1, 3, 1), np.int32)
        pr = np.ones((1, 3, 1), np.float16)
        tl = np.asarray([[3, 2, 1]], np.uint8)
        D.save_dump(td, "t", ids, pr, tl,
                    {"corpus_file": "s", "corpus_sha256": "x", "B": 1, "Ltok": 3,
                     "k": 1, "L_step": 6, "stream_tokens": [3]})
        r = D.MiReader(td, "t", vv, streams=[b"abcdef"], block=4, backoff="skip")
        b2t = list(r.byte2tok[0][:6])
    t("t5 byte->token map", b2t == [0, 0, 0, 1, 1, 2], f"{b2t} == [0,0,0,1,1,2]")

    # ---- t6: TRIPWIRE — the target must NOT be one-hot at the gold byte ----
    ent = []
    for _ in range(400):
        tb = synthetic_vocab(rng, n_tokens=int(rng.integers(8, 40)),
                             maxlen=int(rng.integers(2, 8)))
        vv = D.VocabBytes(tb)
        k = int(rng.integers(4, 12))
        cand = rng.integers(0, len(tb), k).astype(np.int64)
        raw = rng.random(k) + 1e-6
        probs = (raw / raw.sum()).astype(np.float32)
        g = int(rng.integers(0, len(tb)))
        stream = bytes([int(tb[g][0]) % 200 + 10]) + tb[g]
        with tempfile.TemporaryDirectory() as td:
            ids = np.zeros((1, 2, k), np.int32)
            pr = np.zeros((1, 2, k), np.float16)
            ids[0, 0] = cand
            pr[0, 0] = probs
            tl = np.asarray([[1, len(tb[g])]], np.uint8)
            D.save_dump(td, "t", ids, pr, tl,
                        {"corpus_file": "s", "corpus_sha256": "x", "B": 1,
                         "Ltok": 2, "k": k, "L_step": 1, "stream_tokens": [1, 1]})
            r = D.MiReader(td, "t", vv, streams=[stream], block=1, backoff="skip")
            tgt, produced = r.targets(0)
        if float(produced[0]) == 0.0:
            continue
        row = np.asarray(tgt[0], np.float64)
        ent.append(float(-(row * np.log(row + 1e-12)).sum()))
    ent = np.asarray(ent)
    good = (ent.size > 100 and (ent > 1e-3).mean() > 0.5)
    t("t6 TRIPWIRE depth-0 target carries uncertainty (not one-hot)", good,
      f"{ent.size} cases, entropy mean {ent.mean():.3f} nats, "
      f"{(ent > 1e-3).mean():.1%} non-degenerate (a one-hot target = 0% "
      f"non-degenerate, which is the defect this gate now catches)")

    print("\n== verdict:", "PASS" if ok else "FAIL", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())