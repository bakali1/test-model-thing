"""check_mi.py — GATE: the distillation target == the paper's definition.

    .venv/bin/python tests/check_mi.py

Free, CPU, seconds. No teacher, no GPU, no network.

⚠ THIS GATE'S SEMANTICS CHANGED TWICE. Read the history before trusting it.

1. It originally asserted the gold-prefix-conditioned target (the paper's depth
   loop read literally) and passed — while that target was measurably a POINT
   MASS at the gold byte (entropy exactly 0.000000 nats, argmax == gold on
   100.000000% of positions), which made `alpha*KL + (1-alpha)*CE` a reweighted
   CE.
2. It then changed to "depth 0 only, decline every depth >= 1", justified by a
   claim the paper contradicts on the same page: that one forward pass cannot
   represent byte uncertainty inside a token. It cannot represent the mass of
   tokens that END mid-token -- that is what <eot> is for -- but the
   distribution over the NEXT byte given the emitted prefix is soft, and the
   paper's own worked example (isu 0.8 / isk 0.2) says so. That change threw
   away 66% of the available signal and t3 PINNED IT, which is how a wrong
   definition survives review.
3. Now: the paper's construction at every depth. At depth d, condition on the
   bytes already emitted (0..d-1) and aggregate over byte d.

    S_d   = { c : bytes(c)[:d] == gold[:d] and len(c) > d }
    p_tgt[x] = sum_{c in S_d, bytes(c)[d] == x} p(c) / sum_{c in S_d} p(c)

t7 is the regression test for the whole episode: it pins the paper's own
worked example (P(isu)=0.8, P(isk)=0.2 from a 0.5/0.125/0.125 teacher row), so
an implementation that conditions on the byte it is predicting fails loudly.
t3 checks that depth >= 1 is produced AND soft; t6 keeps the one-hot tripwire.

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


def brute_force_depth(gold, blob, cand_bytes, probs, start, depth):
    """Paper Marginalize-It at one (start, depth): condition on gold[:depth],
    aggregate over byte `depth`, keep len(c) > depth. Returns (target|None, mass)."""
    out = np.zeros(256, np.float64)
    pref = blob[start:start + depth]
    tot = 0.0
    for b, pr in zip(cand_bytes, probs):
        if len(b) > depth and bytes(b[:depth]) == bytes(pref):
            out[b[depth]] += pr
            tot += pr
    return (out / tot if tot > 0 else None), tot


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

    # ---- t2: target == brute-force Marginalize-It, at EVERY depth ---------
    # stream = [pad][gold token]; every byte position inside `gold token` is a
    # separate depth, and row q-1 must be the row that predicted token q.
    worst, n_cases, depths_hit = 0.0, 0, set()
    for _ in range(300):
        tb = synthetic_vocab(rng, n_tokens=int(rng.integers(6, 40)),
                             maxlen=int(rng.integers(2, 8)))
        vv = D.VocabBytes(tb)
        k = int(rng.integers(2, 9))
        cand = rng.integers(0, len(tb), k).astype(np.int64)
        raw = rng.random(k) ** 2 + 1e-6
        probs = (raw / raw.sum()).astype(np.float32)
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
                         "Ltok": 2, "k": k, "L_step": len(stream) - 1,
                         "stream_tokens": [1, 1]})
            r = D.MiReader(td, "t", vv, streams=[stream], block=8, backoff="skip",
                           floor=0.0)
            cbytes = [tb[int(c)] for c in cand]
            for depth in range(len(tb[g])):
                got, produced = r.targets(depth)
                exp, tot = brute_force_depth(tb[g], stream, cbytes,
                                             probs.astype(np.float64), 1, depth)
                if tot <= 0:
                    continue
                depths_hit.add(depth)
                g_row = np.asarray(got[0], np.float64).ravel()
                if float(produced[0]) == 0.0:
                    worst = 9.9
                    continue
                worst = max(worst, float(np.abs(g_row - exp).max()))
                n_cases += 1
    t("t2 target == brute-force Marginalize-It at EVERY depth",
      worst <= 2e-3 and n_cases > 100 and len(depths_hit) >= 2,
      f"{n_cases} cases over depths {sorted(depths_hit)}, max |delta| "
      f"{worst:.2e}, tol 2e-3 (fp16 storage)")

    # ---- t3: depth >= 1 is PRODUCED, and is soft ------------------------
    # The fixture must be SELF-CONSISTENT: the stream is a concatenation of
    # table tokens, and tok_len must describe it exactly. An earlier version
    # used stream "abcde" with tok_len=[1,2,2] over a table of ["abc","de",
    # "fgh"], so the gold token "bc" was not in the vocabulary at all and the
    # prefix test could never match. The depth-0-only implementation passed that
    # test because depth 0 never reads a gold byte -- which is exactly why a
    # wrong target definition can hide behind a green gate.
    #
    # stream = a|bc|de. Row 0 predicts 'bc', row 1 predicts 'de'.
    tb = [b"a", b"bc", b"de", b"bcd", b"bxy"]
    vv = D.VocabBytes(tb)
    ids = np.zeros((1, 3, 4), np.int32)
    pr = np.zeros((1, 3, 4), np.float16)
    ids[0, 0] = [1, 3, 4, 0]                  # bc, bcd, bxy, a
    pr[0, 0] = [0.4, 0.3, 0.2, 0.1]
    ids[0, 1] = [2, 3, 0, 0]                 # de, bcd, a, (pad)
    pr[0, 1] = [0.5, 0.3, 0.2, 0.0]
    ids[0, 2] = [0, 1, 2, 0]
    pr[0, 2] = [0.5, 0.3, 0.2, 0.0]
    tl = np.asarray([[1, 2, 2]], np.uint8)
    stream = b"abcde"
    assert sum(tl[0]) == len(stream), (int(tl[0].sum()), len(stream))
    with tempfile.TemporaryDirectory() as td:
        D.save_dump(td, "t", ids, pr, tl,
                    {"corpus_file": "s", "corpus_sha256": "x", "B": 1, "Ltok": 3,
                     "k": 4, "L_step": len(stream) - 1, "stream_tokens": [3]})
        r = D.MiReader(td, "t", vv, streams=[stream], block=8, backoff="skip")
        got = [float(r.targets(i)[1][0]) for i in range(4)]
        t_d1 = np.asarray(r.targets(1)[0][0])      # depth 1 of 'bc', gold 'c'
        t_d0 = np.asarray(r.targets(0)[0][0])      # depth 0 of 'bc', gold 'b'
    all_depths = all(x == 1.0 for x in got)
    t("t3 depths >= 1 are produced, not declined", all_depths,
      f"produced by step: {got} (want [1,1,1,1]); declined={r.declined} of "
      f"{r.positions}")
    # depth 0: 'b' from bc/bcd/bxy (0.9) vs 'a' (0.1) -> soft, no conditioning
    d0_ok = (abs(float(t_d0[ord("b")]) - 0.9) < 5e-3
             and abs(float(t_d0[ord("a")]) - 0.1) < 5e-3)
    t("t3b depth-0 target is the unconditioned first-byte marginal", d0_ok,
      f"'b'={float(t_d0[ord('b')]):.3f} 'a'={float(t_d0[ord('a')]):.3f} "
      f"(want 0.900 / 0.100)")
    # depth 1: prefix 'b', so bcd and bxy survive alongside bc; byte 1 is 'c'
    # (from bc 0.4 + bcd 0.3) or 'x' (from bxy 0.2). Retained mass is 0.9, so
    # the renormalised target is 0.7/0.9 and 0.2/0.9. The target MUST carry the
    # 'x' branch -- that is the whole point.
    d1_ok = (abs(float(t_d1[ord("c")]) - 0.7 / 0.9) < 5e-3
             and abs(float(t_d1[ord("x")]) - 0.2 / 0.9) < 5e-3)
    t("t3c depth-1 target keeps the non-gold branch (soft, not one-hot)", d1_ok,
      f"'c'={float(t_d1[ord('c')]):.3f} 'x'={float(t_d1[ord('x')]):.3f} "
      f"(want {0.7 / 0.9:.3f} / {0.2 / 0.9:.3f}; one-hot 'c'=1.0 means the byte "
      f"being predicted was conditioned on -- the 2026-10-03 bug)")

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
                         "Ltok": 2, "k": k, "L_step": len(stream) - 1,
                         "stream_tokens": [1, 1]})
            r = D.MiReader(td, "t", vv, streams=[stream], block=8, backoff="skip",
                           floor=0.0)
            tgts = []
            for depth in range(len(tb[g])):
                tgt, produced = r.targets(depth)
                if float(produced[0]) > 0:
                    tgts.append(np.asarray(tgt[0], np.float64))
        for row in tgts:
            ent.append(float(-(row * np.log(row + 1e-12)).sum()))
    ent = np.asarray(ent)
    good = (ent.size > 100 and (ent > 1e-3).mean() > 0.5)
    t("t6 TRIPWIRE target carries uncertainty (not one-hot)", good,
      f"{ent.size} cases, entropy mean {ent.mean():.3f} nats, "
      f"{(ent > 1e-3).mean():.1%} non-degenerate (a one-hot target = 0% "
      f"non-degenerate, which is the defect this gate now catches)")

    # ---- t7: THE PAPER'S OWN WORKED EXAMPLE ------------------------------
    # Section 2.1, Figure 2. Teacher row: isu 0.5, is 0.125, isk 0.125. Gold
    # token is `isu`, so the byte being predicted at depth 2 is 'u' and the
    # emitted prefix is "is". MI drops `is` (it ENDS at depth 2, so len(c) > d
    # fails) and renormalises {isu, isk} to 0.5/0.125 -> 0.8 / 0.2.
    #
    # An implementation that conditions on the byte it is predicting returns
    # 1.000 at 'u' instead. An implementation that declines depth >= 1 returns
    # no target at all. This test is the one that would have caught BOTH.
    tb = [b"z", b"is", b"isu", b"isk"]
    vv = D.VocabBytes(tb)
    blob = b"z" + b"isu"                        # [z] pad then the gold token
    tl = np.asarray([[1, 3]], np.uint8)
    assert int(tl[0].sum()) == len(blob), (int(tl[0].sum()), len(blob))
    with tempfile.TemporaryDirectory() as td:
        ids = np.zeros((1, 2, 3), np.int32)
        pr = np.zeros((1, 2, 3), np.float16)
        ids[0, 0] = [2, 1, 3]                   # isu, is, isk
        pr[0, 0] = [0.5, 0.125, 0.125]
        D.save_dump(td, "t", ids, pr, tl,
                    {"corpus_file": "s", "corpus_sha256": "x", "B": 1, "Ltok": 2,
                     "k": 3, "L_step": len(blob) - 1, "stream_tokens": [2]})
        r = D.MiReader(td, "t", vv, streams=[blob], block=8, backoff="skip")
        # step 0 predicts byte 1 (depth 0 of 'isu'); step 1 predicts byte 2
        # (depth 1); step 2 predicts byte 3 (depth 2) <- the paper's B3.
        t_d0 = np.asarray(r.targets(0)[0][0])
        t_d1 = np.asarray(r.targets(1)[0][0])
        t_d2 = np.asarray(r.targets(2)[0][0])
        p_d2 = float(r.targets(2)[1][0])
    i_b, s_b, u, k_b = ord("i"), ord("s"), ord("u"), ord("k")
    # depth 0: all three candidates start with 'i' -> 1.0 at 'i'
    d0_ok = abs(float(t_d0[i_b]) - 1.0) < 1e-3
    # depth 1: prefix 'i', len>1 -> isu and isk both give 's' -> 1.0 at 's'
    d1_ok = abs(float(t_d1[s_b]) - 1.0) < 1e-3
    # depth 2: THE PAPER'S NUMBER. 'is' is excluded by len(c) > 2, so its 0.125
    # is dropped and {isu, isk} renormalise 0.5/0.125 -> 0.8 / 0.2.
    d2_ok = (abs(float(t_d2[u]) - 0.8) < 5e-3
             and abs(float(t_d2[k_b]) - 0.2) < 5e-3
             and float(t_d2[i_b]) == 0.0
             and p_d2 == 1.0)
    t("t7 paper's worked example: B3 is 0.8/0.2, not 1.0 and not absent",
      d0_ok and d1_ok and d2_ok,
      f"depth0 'i'={float(t_d0[i_b]):.3f} | depth1 's'={float(t_d1[s_b]):.3f} "
      f"| depth2 u={float(t_d2[u]):.3f} k={float(t_d2[k_b]):.3f} (paper: isu "
      f"0.8 / isk 0.2; the 2026-10-03 bug gives 1.000/0.000, the 2026-10-04 "
      f"misdiagnosis gives no target)")

    print("\n== verdict:", "PASS" if ok else "FAIL", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())