#!/usr/bin/env python3
"""tools/d5_probe.py -- the D5 discriminator (PLAN-2026-10-21-CONTINUAL-LEARNING.md §6).

THE QUESTION
------------
The recurrent state is a per-coordinate leaky integrator of the byte
embedding (`state = states*a + enc`, gpu_port.py:247) -- a WEIGHTED MIXTURE of
embeddings over time.  Two facts are in tension:

  (i)  the delta-h retention probe says the state HOLDS a lot: 37.5 % of the
       control floor at 8 B, 17 % at 32 B, 95.4 B half-life (P4 init);
  (ii) a linear readout of the state recovers the PAYLOAD only for ~12 B of
       lag (12.4 %/byte at 12 B, 1.69 % at 36 B, nothing at 132 B) while the
       TRAILING KEY decodes at 25-41 %.

Capacity (delta-h) and usable content (decodable payload) diverge by an order
of magnitude.  Three candidate causes, one discriminator each:

  (a) the WRITE is order-losing -- the payload is in the state as a mixture
      but not ADDRESSABLE;
  (b) the READOUT is key-blind -- one global linear map must serve every
      key, so it can only read what is linearly separable ACROSS samples;
  (c) delta-h OVERSTATES the capacity -- it registers a difference that no
      readout can use.

WHAT THIS TOOL DOES
-------------------
For a given gap d and sample count n it reports, per gap, the following
readout cells.  Every readout is a 256-way one-hot dual ridge, using the
FIXED `_ridge_score` of tools/membench/model_backend.py -- imported, never
re-implemented, never modified.

  payload      decode the 4 payload bytes from the state (the GLOBAL readout)
  key          decode the 4 KEY bytes from the state (the GLOBAL readout)
  trailing     decode the 4 bytes the sequence ENDS with (the trailing key --
               the instrument's power reference).  In delayed_copy the trailing
               key IS the head key, so this is the same fit as `key`; it is
               reported separately as a consistency check.
  perkey       a PER-KEY readout: for each distinct key group, fit a ridge on
               that group's samples and decode the payload with it
  perkey_veto  the per-key readout against PERMUTED payload labels (control)
  nuisance     the payload readout where the payload region of the sequence is
               REPLACED by fresh random bytes from the same alphabet, so the
               labelled payload is no longer at the payload position ("no
               signal" null at the same alphabet size)

EVERY cell reports: per-byte accuracy, exact-4-byte rate, sample count, the
chosen lambda, a MATCHED NULL (same split, same training rows, permuted
labels), the class coverage and a verdict.  A cell is interpretable only if it
beats its own null; a coverage-limited cell is labelled UNPOWERED.

INSTRUMENT REQUIREMENTS HONOURED (CL-1)
---------------------------------------
  * the state is read AT THE CUE.  The default path uses a FIXED-GAP
    generator (`generators.delayed_copy(d, gap_mode="fixed")`), so every
    sequence is exactly 12+d bytes long and the state read at t == L-1 IS the
    state at the cue -- no zero padding is ever inserted.  `--mode varlen`
    instead uses the variable-length batch and reads every sample at ITS OWN
    last real byte (`states_at_positions`).  The tool always prints which
    method it used.  (The pre-existing cell, tools/membench/model_backend.py
    `model_copy_readout`, pads to the longest sample and reads `final_states`
    at t == L-1 -- the D4 defect.)
  * per-byte is reported next to the exact-4-byte rate, and a MATCHED
    ALPHABET is available: `--alphabet {full,16,4}` restricts the payload
    bytes to 16 or 4 symbols, so the null is 1/alphabet_size per byte and the
    exact-4-byte metric is not hopeless.
  * the state is reset per sample (each sample is an independent row of the
    batch; no state crosses samples) and no weight is updated.

USAGE
-----
    .venv/bin/python tools/d5_probe.py \
        --ckpt runs/taxis/A_p4_wd0_c.npz --gaps 8,12,32,64,128 \
        --n 2048 --alphabet 16 --out d5_a16.json
    .venv/bin/python tools/d5_probe.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

import tools.membench.model_backend as mb
from tools.membench import generators as G

# the FIXED lambda grid of mb._ridge_score (its default; passed explicitly so
# the tool's convention is visible in one place)
LAMS = (1.0, 1e3, 1e5)

KEY_LEN = G.KEY_LEN          # 4
PAYLOAD_LEN = G.PAYLOAD_LEN  # 4


# ---------------------------------------------------------------------------
# alphabets
# ---------------------------------------------------------------------------
def alphabet_symbols(kind):
    """The payload alphabet.  'full' -> None (all 256 byte values);
    '16'/'4' -> 16/4 evenly spread byte values (so the symbols are not
    clustered in the control-byte range)."""
    if kind in (None, "full"):
        return None
    k = int(kind)
    if k <= 1 or 256 % k:
        raise ValueError(f"alphabet size must divide 256; got {k}")
    return np.array([(i * 256) // k for i in range(k)], dtype=np.uint8)


def _draw(rng, alpha, size):
    if alpha is None:
        return rng.integers(0, 256, size=size).astype(np.uint8)
    return alpha[rng.integers(0, len(alpha), size=size)]


# ---------------------------------------------------------------------------
# batch generation (fixed gap -> no padding, ever)
# ---------------------------------------------------------------------------
def gen_fixed_gap_batch(d, n, rng, alpha, key_alpha=None):
    """A FIXED-GAP delayed-copy batch.

    Layout (generators.delayed_copy, gap_mode="fixed"):
        [KEY 4][PAYLOAD 4][gap d][KEY 4]      -> exactly 12+d bytes per sample
    so every sample has the SAME length: the state read at t == L-1 is the
    state AT THE CUE, with no zero padding in between.

    `alpha` restricts the PAYLOAD bytes to that alphabet; `key_alpha`
    restricts the KEY bytes (used by the exact per-key arm, where the key has
    to repeat for a per-key readout to be well posed).

    Returns dict(seq (n, 12+d) uint8, key (n,4), payload (n,4), gaps).
    """
    d = int(d)
    L = 12 + d
    seq = np.zeros((n, L), dtype=np.uint8)
    key = np.zeros((n, KEY_LEN), dtype=np.uint8)
    payload = np.zeros((n, PAYLOAD_LEN), dtype=np.uint8)
    for i in range(n):
        s, t, info = G.delayed_copy(d, rng=rng, gap_mode="fixed")
        k = info["key"].copy()
        if key_alpha is not None:
            k = _draw(rng, key_alpha, KEY_LEN)
        p = t.copy()
        if alpha is not None:
            p = _draw(rng, alpha, PAYLOAD_LEN)
        seq[i, :] = s
        seq[i, 0:KEY_LEN] = k
        seq[i, 4:4 + PAYLOAD_LEN] = p
        seq[i, 8 + d:12 + d] = k        # the cue (the trailing key)
        key[i] = k
        payload[i] = p
    return {"seq": seq, "key": key, "payload": payload, "length": L,
            "gap": d}


def nuisance_batch(batch, rng, alpha):
    """The NUISANCE arm: the payload region of every sequence is replaced by
    fresh random bytes drawn from the SAME alphabet, while the labels stay
    the original payload.  The labelled payload is therefore no longer at the
    payload position: any accuracy above the null would be a leak, not a
    readout.  (Chance collisions of the alphabet symbols inside the
    distractor bytes exist in the main arm as well, so the two arms are
    matched in that respect.)"""
    seq = batch["seq"].copy()
    n = seq.shape[0]
    fresh = _draw(rng, alpha, (n, PAYLOAD_LEN))
    seq[:, 4:4 + PAYLOAD_LEN] = fresh
    return {"seq": seq, "key": batch["key"], "payload": batch["payload"]}


def gen_varlen_batch(d, n, rng, alpha):
    """Variable-length batch (generators.get_batch, gap_mode="random") plus the
    per-sample read position (each sample's OWN last real byte).  Used by
    --mode varlen; the fixed-gap path is the default."""
    b = G.get_batch("delayed_copy", d, n, rng=rng, gap_mode="random")
    seqs = b["seq"]
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    L = int(lengths.max())
    X = np.zeros((n, L), dtype=np.uint8)
    key = np.zeros((n, KEY_LEN), dtype=np.uint8)
    payload = np.zeros((n, PAYLOAD_LEN), dtype=np.uint8)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
        k = s[0:KEY_LEN].copy()
        p = s[4:4 + PAYLOAD_LEN].copy()
        if alpha is not None:
            p = _draw(rng, alpha, PAYLOAD_LEN)
            X[i, 4:4 + PAYLOAD_LEN] = p
            # keep the trailing cue key in sync with the head key
            X[i, len(s) - KEY_LEN:len(s)] = k
        key[i] = k
        payload[i] = p
    return {"seq": X, "key": key, "payload": payload,
            "length": L, "read_pos": lengths - 1,
            "lengths": lengths}


# ---------------------------------------------------------------------------
# frozen forward (state at the cue)
# ---------------------------------------------------------------------------
def states_at_cue(m, X):
    """State after the LAST byte of a fixed-length batch.  With the fixed-gap
    generator the sequence ENDS with the 4 cue-key bytes, so this is the
    state AT THE CUE.  Mirrors mb._frozen_forward exactly, minus the (unused)
    per-position logits."""
    B, L = X.shape
    m.new_states(B)                      # per-sample state reset
    for t in range(L):
        enc = m.embed[X[:, t]]
        x = enc
        for l in m.layers:
            x, st, _d, _c = l.forward(enc, x, l.states)
            l.states = st
    return np.concatenate([l.states for l in m.layers],
                          axis=1).astype(np.float32)


def states_at_positions(m, X, pos):
    """State at a PER-SAMPLE position (pos[i] = sample i's last real byte).
    The recurrence is causal, so the padding that follows a sample cannot
    change the state captured at its own last byte."""
    B, L = X.shape
    m.new_states(B)
    out = np.zeros((B, m.dim * len(m.layers)), dtype=np.float32)
    want = {}
    for i, p in enumerate(pos):
        want.setdefault(int(p), []).append(i)
    for t in range(L):
        enc = m.embed[X[:, t]]
        x = enc
        for l in m.layers:
            x, st, _d, _c = l.forward(enc, x, l.states)
            l.states = st
        if t in want:
            cat = np.concatenate([l.states for l in m.layers],
                                 axis=1).astype(np.float32)
            for i in want[t]:
                out[i] = cat[i]
    return out


# ---------------------------------------------------------------------------
# readout cells
# ---------------------------------------------------------------------------
def _se_per_byte(p, n):
    """SE of a per-byte rate over the test half (the readout's own split:
    the first n//2 rows train, the rest test)."""
    n_test = max(1, n - max(4, n // 2))
    return float(math.sqrt(max(p * (1.0 - p), 0.0) / n_test))


def _ridge(X, Y, seed):
    """One global 256-way one-hot dual ridge (the FIXED mb._ridge_score)."""
    exact, per_byte, se, lam = mb._ridge_score(X, Y, X, Y, lams=LAMS, seed=seed)
    return {"exact4": float(exact), "per_byte": float(per_byte),
            "se": float(se), "lam": float(lam)}


def _split(n, seed):
    """The split mb._ridge_score uses internally: permutation by `seed`,
    first half train, remainder test.  Reproduced here ONLY so that the
    coverage report refers to the same rows the readout was fit on."""
    idx = np.random.default_rng(seed).permutation(n)
    n_train = max(4, n // 2)
    return idx[:n_train], idx[n_train:]


def coverage_report(y, tr, te, alpha_size):
    """Class coverage of the readout.

    coverage      fraction of TEST-half label values that also occur in the
                  training half (pooled over the byte positions)
    coverage_pos  the same, per byte position
    train_distinct_frac  fraction of the EFFECTIVE alphabet (alpha_size) that
                  the training half actually contains
    train_distinct_frac256  the same over the full 256-symbol byte space
    """
    y = np.asarray(y)
    cov_pos = []
    all_tr = set()
    for j in range(y.shape[1]):
        vals = np.unique(y[tr, j])
        all_tr.update(vals.tolist())
        hit = np.isin(y[te, j], vals)
        cov_pos.append(float(hit.mean()))
    pooled_hits, pooled_n = 0, 0
    for j in range(y.shape[1]):
        vals = np.unique(y[tr, j])
        pooled_hits += int(np.isin(y[te, j], vals).sum())
        pooled_n += len(te)
    return {
        "coverage": float(pooled_hits / max(pooled_n, 1)),
        "coverage_pos": cov_pos,
        "train_distinct": len(all_tr),
        "train_distinct_frac": float(len(all_tr) / max(alpha_size, 1)),
        "train_distinct_frac256": float(len(all_tr) / 256.0),
    }


def perkey_ridge(X, Y, groups, seed, min_rows=16):
    """A PER-KEY readout: a separate ridge per key group, each decoding the
    4 payload bytes.  Groups with fewer than `min_rows` samples are skipped
    (and counted).  The matched null is the same call with the labels
    permuted over samples: same rows, same split, same groups."""
    Y = np.asarray(Y)
    groups = np.asarray(groups)
    used = skipped = 0
    test_rows = 0
    acc_b = 0.0
    acc_e = 0.0
    lams_used = []
    train_rows = []
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        if len(rows) < min_rows:
            skipped += 1
            continue
        e, pb, se, lam = mb._ridge_score(X[rows], Y[rows], X[rows], Y[rows],
                                         lams=LAMS, seed=seed)
        n_train = max(4, len(rows) // 2)
        n_test = len(rows) - n_train
        train_rows.append(n_train)
        acc_b += pb * n_test
        acc_e += e * n_test
        test_rows += n_test
        used += 1
        lams_used.append(lam)
    if test_rows == 0:
        return {"per_byte": None, "exact4": None, "se": None, "lam": None,
                "se_per_byte": None,
                "groups_total": int(len(np.unique(groups))),
                "groups_used": 0, "groups_skipped": skipped,
                "rows_used": 0, "test_rows": 0,
                "note": "no key group had enough samples to fit a readout"}
    pb = float(acc_b / test_rows)
    return {"per_byte": pb,
            "exact4": float(acc_e / test_rows),
            "se": float(math.sqrt(max(acc_e * (1 - acc_e), 0.0) / test_rows)),
            "se_per_byte": _se_per_byte(pb, 2 * test_rows),
            "lam": (float(np.median(lams_used)) if lams_used else None),
            "groups_total": int(len(np.unique(groups))),
            "groups_used": used, "groups_skipped": skipped,
            "rows_used": int(test_rows), "test_rows": int(test_rows),
            "median_train_rows_per_group": (float(np.median(train_rows))
                                            if train_rows else 0.0)}


def group_ids(key, mode):
    """The per-key grouping.

    head4  the high nibble of the key's first byte  (16 groups -- the
           coarsest conditioning that is still well posed at n=2048)
    head1  the key's first byte                   (256 groups)
    full   the whole 4-byte key                   (n groups -- one per key)
    """
    k = np.asarray(key)
    if mode == "head4":
        return (k[:, 0] >> 4).astype(np.int64)
    if mode == "head1":
        return k[:, 0].astype(np.int64)
    if mode == "full":
        return np.array([int.from_bytes(bytes(r), "big") for r in k],
                        dtype=np.int64)
    raise ValueError(f"unknown per-key grouping {mode!r}")


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
def verdict(cell, cov, alpha_size, null_per_byte):
    """SIGNAL / NULL / UNPOWERED.

    UNPOWERED  the training half does not cover the effective alphabet (or
               the test half's values are largely unseen) -- the cell cannot
               be interpreted either way.
    SIGNAL     the cell beats its own MATCHED NULL by more than 2 SE (the SE
               of the per-byte rate).
    NULL       the cell does not beat its own matched null.
    """
    if cell.get("per_byte") is None:
        return "UNPOWERED"
    cov_ok = (cov["coverage"] >= 0.9 and
              cov["train_distinct_frac"] >= 0.9)
    if not cov_ok:
        return "UNPOWERED"
    se = cell.get("se_per_byte")
    if se is None:
        se = cell.get("se") or 0.0
    if cell["per_byte"] > null_per_byte + 2.0 * se:
        return "SIGNAL"
    return "NULL"


def make_cell(name, X, Y, seed, label_alpha_size, note=None,
              expect_signal=True):
    """A global cell + its MATCHED NULL (permuted labels, same split, same
    training rows) + coverage + verdict.

    `label_alpha_size` is the size of the LABEL's alphabet (the payload
    alphabet for the payload cells, 256 for the key cells), which fixes both
    the chance level and the coverage denominator.
    """
    n = X.shape[0]
    Y = np.asarray(Y, dtype=np.int64)
    cell = _ridge(X, Y, seed)
    perm = np.random.default_rng(seed + 7919).permutation(n)
    null = _ridge(X, Y[perm], seed)
    tr, te = _split(n, seed)
    cov = coverage_report(Y, tr, te, label_alpha_size)
    out = {
        "cell": name, "n": int(n),
        "per_byte": cell["per_byte"], "exact4": cell["exact4"],
        "se": cell["se"], "lam": cell["lam"],
        "se_per_byte": _se_per_byte(cell["per_byte"], n),
        "null_per_byte": null["per_byte"], "null_exact4": null["exact4"],
        "null_se": null["se"], "null_lam": null["lam"],
        "null_se_per_byte": _se_per_byte(null["per_byte"], n),
        "coverage": cov,
        "label_alphabet_size": int(label_alpha_size),
        "chance_per_byte": 1.0 / label_alpha_size,
        "chance_exact4": (1.0 / label_alpha_size) ** Y.shape[1],
        "null_method": ("permuted labels, same split and same training rows "
                        "(same seed) as the cell itself"),
    }
    if note:
        out["note"] = note
    out["verdict"] = verdict(out, cov, label_alpha_size, null["per_byte"])
    if not expect_signal:
        out["expected"] = "NULL (this is a control; SIGNAL here would be a leak)"
    return out


# ---------------------------------------------------------------------------
# one gap
# ---------------------------------------------------------------------------
def run_gap(m, d, n, seed, args, alpha):
    t_start = time.time()
    rng = np.random.default_rng((seed * 1_000_003 + d) % (2 ** 32))
    alpha_size = 256 if alpha is None else len(alpha)
    t0 = time.time()
    if args.mode == "fixed":
        batch = gen_fixed_gap_batch(d, n, rng, alpha)
        X = states_at_cue(m, batch["seq"])
        read_method = ("fixed-gap generator: every sequence is exactly "
                       f"{12 + d} bytes, state read at t == L-1 == AT THE CUE, "
                       "no zero padding")
        read_positions = None
    else:
        batch = gen_varlen_batch(d, n, rng, alpha)
        X = states_at_positions(m, batch["seq"], batch["read_pos"])
        read_method = ("variable-length batch: state read at EACH SAMPLE'S "
                       "OWN last real byte (no sample is read after its "
                       "padding)")
        read_positions = batch["read_pos"]
    t_fwd = time.time() - t0

    Y_payload = batch["payload"].astype(np.int64)
    Y_key = batch["key"].astype(np.int64)

    t0 = time.time()
    cells = []
    cells.append(make_cell("payload", X, Y_payload, seed, alpha_size,
                            note="global readout of the 4 payload bytes"))
    cells.append(make_cell("key", X, Y_key, seed, 256,
                            note=("global readout of the 4 KEY bytes (the cue); "
                                  "the key uses the FULL 256-symbol alphabet, "
                                  "so its chance is 1/256")))
    # the trailing key: in delayed_copy the sequence ENDS with the cue key,
    # so the trailing key IS the head key -> the same fit as `key`.  Reported
    # as its own cell (consistency check) with the same numbers.
    cells.append(dict(cells[1], cell="trailing",
                      note=("the 4 bytes the sequence ends with = the cue key; "
                            "identical to the `key` cell by construction "
                            "(same fit, reported separately)")))
    t_ridge = time.time() - t0

    # -- the nuisance arm: the payload region is replaced by fresh random
    #    bytes of the same alphabet, so the labelled payload is NOT in the
    #    state.  Needs its own forward (the sequence changed).
    t0 = time.time()
    nb = nuisance_batch(batch, rng, alpha)
    if args.mode == "fixed":
        Xn = states_at_cue(m, nb["seq"])
    else:
        Xn = states_at_positions(m, nb["seq"], read_positions)
    t_fwd_nuis = time.time() - t0
    t0 = time.time()
    cells.append(make_cell("nuisance", Xn, Y_payload, seed, alpha_size,
                            note=("payload region replaced by fresh random "
                                  "bytes of the same alphabet; the labels (the "
                                  "original payload) are no longer at the "
                                  "payload position, so any accuracy above "
                                  "the null would be a leak (chance alphabet "
                                  "collisions in the distractor bytes are "
                                  "present in the main arm too)"),
                            expect_signal=False))
    t_ridge_nuis = time.time() - t0

    # -- the per-key arm ---------------------------------------------------
    perkey_info = None
    if args.perkey:
        t0 = time.time()
        g = group_ids(batch["key"], args.perkey_group)
        pk = perkey_ridge(X, Y_payload, g, seed, min_rows=args.perkey_min_rows)
        perm = np.random.default_rng(seed + 7919).permutation(n)
        pkv = perkey_ridge(X, Y_payload[perm], g, seed,
                            min_rows=args.perkey_min_rows)
        tr, te = _split(n, seed)
        cov = coverage_report(Y_payload, tr, te, alpha_size)
        pk.update({"cell": "perkey", "n": int(n), "null_per_byte": pkv["per_byte"],
                   "null_exact4": pkv["exact4"], "null_se": pkv["se"],
                   "null_lam": pkv["lam"], "coverage": cov,
                   "chance_per_byte": 1.0 / alpha_size,
                   "chance_exact4": (1.0 / alpha_size) ** Y_payload.shape[1],
                   "grouping": args.perkey_group,
                   "note": (f"PER-KEY readout (grouping {args.perkey_group}): "
                            f"{pk['groups_used']}/{pk['groups_total']} key "
                            f"groups testable, median "
                            f"{pk.get('median_train_rows_per_group')} training "
                            f"rows per group"),
                   "null_method": ("permuted payload labels, same rows, same "
                                   "split, same groups")})
        pk["verdict"] = verdict(pk, cov, alpha_size, pkv["per_byte"])
        pkv2 = dict(pkv)
        pkv2.update({"cell": "perkey_veto", "n": int(n), "coverage": cov,
                     "chance_per_byte": 1.0 / alpha_size,
                     "chance_exact4": (1.0 / alpha_size) ** Y_payload.shape[1],
                     "note": ("the per-key readout against PERMUTED payload "
                              "labels -- the control for the perkey cell"),
                     "verdict": "NULL (control)"})
        cells.append(pk)
        cells.append(pkv2)
        perkey_info = {"grouping": args.perkey_group,
                       "groups_total": pk["groups_total"],
                       "groups_used": pk["groups_used"],
                       "groups_skipped": pk["groups_skipped"]}
        t_ridge += time.time() - t0

    # -- the EXACT per-key arm: a low-entropy key so that keys REPEAT -----
    if args.perkey and args.perkey_exact:
        t0 = time.time()
        kalpha = np.array([0, 128], dtype=np.uint8)      # 2 symbols/byte
        rng2 = np.random.default_rng((seed * 7_919 + d + 1) % (2 ** 32))
        b2 = gen_fixed_gap_batch(d, n, rng2, alpha, key_alpha=kalpha)
        X2 = states_at_cue(m, b2["seq"])
        t_fwd += time.time() - t0
        Y2 = b2["payload"].astype(np.int64)
        t0 = time.time()
        g2 = group_ids(b2["key"], "full")     # the whole 4-byte key
        pk2 = perkey_ridge(X2, Y2, g2, seed, min_rows=args.perkey_min_rows)
        perm2 = np.random.default_rng(seed + 7919).permutation(n)
        pkv2 = perkey_ridge(X2, Y2[perm2], g2, seed,
                             min_rows=args.perkey_min_rows)
        tr, te = _split(n, seed)
        cov2 = coverage_report(Y2, tr, te, alpha_size)
        pk2.update({"cell": "perkey_exact", "n": int(n),
                    "null_per_byte": pkv2["per_byte"],
                    "null_exact4": pkv2["exact4"], "null_se": pkv2["se"],
                    "null_lam": pkv2["lam"], "coverage": cov2,
                    "chance_per_byte": 1.0 / alpha_size,
                    "chance_exact4": (1.0 / alpha_size) ** Y2.shape[1],
                    "grouping": "full 4-byte key (low-entropy key: 2 symbols "
                                "per byte -> 16 distinct keys)",
                    "note": ("TRUE per-key readout: the key has only 16 "
                             "distinct values, so each key group can hold a "
                             "ridge"),
                    "null_method": "permuted payload labels, same rows/split"})
        pk2["verdict"] = verdict(pk2, cov2, alpha_size, pkv2["per_byte"])
        # the global readout on the SAME arm: the apples-to-apples control
        # for "is the global map key-blind?"
        gl2 = make_cell("perkey_exact_global", X2, Y2, seed, alpha_size,
                        note=("the GLOBAL payload readout on the low-entropy-"
                              "key arm -- compare with perkey_exact"))
        cells.append(pk2)
        cells.append(gl2)
        t_ridge += time.time() - t0

    return {
        "gap": int(d), "n": int(n), "alpha_size": alpha_size,
        "read_method": read_method,
        "sequence_length": int(batch.get("length", 0)),
        "cells": cells,
        "perkey_info": perkey_info,
        "timing": {"forward_s": t_fwd, "nuisance_forward_s": t_fwd_nuis,
                   "ridge_s": t_ridge, "total_s": time.time() - t_start},
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def pct(x):
    if x is None:
        return "   n/a"
    return f"{100.0 * x:6.2f}%"


def print_gap_table(gap_res, alpha_size):
    print(f"\n--- gap {gap_res['gap']}  (n={gap_res['n']}, payload alphabet "
          f"{alpha_size}, {gap_res['sequence_length']} B/sequence) ---")
    print(f"    read method: {gap_res['read_method']}")
    hdr = (f"    {'cell':<20} {'per_byte':>9} {'exact4':>9} {'null_pb':>9} "
           f"{'exact4n':>9} {'cov':>7} {'tr_dist':>8} {'lam':>7}  verdict")
    print(hdr)
    for c in gap_res["cells"]:
        cov = c.get("coverage", {})
        print(f"    {c['cell']:<20} {pct(c.get('per_byte')):>9} "
              f"{pct(c.get('exact4')):>9} {pct(c.get('null_per_byte')):>9} "
              f"{pct(c.get('null_exact4')):>9} "
              f"{pct(cov.get('coverage')):>7} "
              f"{cov.get('train_distinct', '-'):>8} "
              f"{(c.get('lam') if c.get('lam') is not None else '-'):>7}  "
              f"{c.get('verdict')}")
        if c.get("note"):
            print(f"        note: {c['note']}")


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def selftest(ckpt, seed=0):
    """(i) the readout recovers a linearly decodable payload and does not
    recover a random one; (ii) the state is read AT THE CUE: changing the
    bytes AFTER the cue does not change the extracted features (the direct
    test for the D4 padding bug)."""
    ok = True
    print("== D5 selftest ==")

    # ---- (i) readout power, on synthetic features (fast, deterministic) ---
    # a GENUINELY linearly decodable target: 16 well-separated prototypes,
    # one per class.  The 256-way one-hot ridge must recover it; the same
    # readout on a random (uncorrelated) label vector must not.
    rng = np.random.default_rng(seed)
    n, f, k = 600, 64, 16
    proto = rng.standard_normal((k, f))
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    y_lin = rng.integers(0, k, size=n)
    Xs = (3.0 * proto[y_lin] + 0.3 * rng.standard_normal((n, f))).astype(np.float32)
    r = _ridge(Xs, y_lin.astype(np.int64), seed)
    r_rand = _ridge(Xs, rng.integers(0, k, size=n).astype(np.int64), seed)
    power_ok = r["per_byte"] > 0.9 and r_rand["per_byte"] < 0.25
    print(f"  (i) linearly decodable target : per_byte={pct(r['per_byte'])} "
          f"(expect > 90%)   -> {'OK' if r['per_byte'] > 0.9 else 'FAIL'}")
    print(f"      random (uncorrelated) label: per_byte={pct(r_rand['per_byte'])} "
          f"(chance {pct(1.0 / k)}, expect < 25%)"
          f"   -> {'OK' if r_rand['per_byte'] < 0.25 else 'FAIL'}")
    ok &= bool(power_ok)

    # ---- (ii) the state is read at the cue -------------------------------
    m, info = mb.load_frozen(ckpt)
    d = 8
    rng2 = np.random.default_rng(seed + 1)
    b = gen_fixed_gap_batch(d, 16, rng2, None)
    cue = 12 + d - 1                       # the last cue byte
    # (a) the same prefix, DIFFERENT bytes after the cue
    Xa = b["seq"].copy()
    Xb = b["seq"].copy()
    pad = 37
    Xb = np.concatenate([Xb, rng2.integers(0, 256, size=(16, pad)).astype(np.uint8)],
                        axis=1)
    sa = states_at_positions(m, Xa, np.full(16, cue))
    sb = states_at_positions(m, Xb, np.full(16, cue))
    dmax = float(np.abs(sa - sb).max())
    cue_ok = dmax == 0.0
    print(f"  (ii) bytes after the cue changed -> state at the cue changed by "
          f"{dmax:.3g} (max abs)  -> {'OK (identical)' if cue_ok else 'FAIL'}")
    ok &= cue_ok

    # (b) the D4 bug demonstrated: the padded-batch read at t == L-1 is NOT
    #     the state at the cue
    sbug = states_at_cue(m, Xb)
    dbug = float(np.abs(sbug - sb).max())
    rel = dbug / max(float(np.abs(sb).max()), 1e-12)
    print(f"      D4 fingerprint: reading the padded batch at t == L-1 "
          f"differs from the cue read by {dbug:.3g} "
          f"({100 * rel:.1f} % of the state scale) -> the padding bug is "
          f"REAL in that path")
    print(f"      fixed-gap read == cue read: "
          f"{'OK' if float(np.abs(states_at_cue(m, Xa) - sa).max()) == 0.0 else 'FAIL'}")
    ok &= float(np.abs(states_at_cue(m, Xa) - sa).max()) == 0.0

    # (c) the lean forward used by the probe agrees with mb._frozen_forward
    Xc = b["seq"][:8]
    ref = mb._frozen_forward(m, Xc)["final_states"]
    mine = states_at_cue(m, Xc)
    same = float(np.abs(ref - mine).max())
    print(f"      probe forward == mb._frozen_forward (max abs diff {same:.3g})"
          f"  -> {'OK' if same == 0.0 else 'FAIL'}")
    ok &= same == 0.0

    print(f"== selftest {'PASS' if ok else 'FAIL'} ==")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/taxis/A_p4_wd0_c.npz",
                    help="frozen checkpoint (default: %(default)s)")
    ap.add_argument("--gaps", default="8,12,32,64,128",
                    help="comma separated gaps d (default: %(default)s)")
    ap.add_argument("--n", type=int, default=2048,
                    help="samples per gap (default: %(default)s)")
    ap.add_argument("--alphabet", default="16", choices=("full", "16", "4"),
                    help="payload alphabet (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    ap.add_argument("--perkey", action="store_true",
                    help="enable the per-key readout (slower)")
    ap.add_argument("--perkey-group", default="head4",
                    choices=("head4", "head1", "full"),
                    help="per-key grouping (default: %(default)s)")
    ap.add_argument("--perkey-min-rows", type=int, default=16,
                    help="minimum rows for a key group to be testable")
    ap.add_argument("--perkey-exact", action="store_true",
                    help="also run the TRUE per-key arm (low-entropy key)")
    ap.add_argument("--mode", default="fixed", choices=("fixed", "varlen"),
                    help="state-read method: fixed-gap (no padding) or "
                         "variable-length (per-sample read position)")
    ap.add_argument("--gated", action="store_true",
                    help="the checkpoint is a P3 GATED one (its theta carries "
                         "a layer{i}.gate vector per layer). The model builder "
                         "must be told, or the parameter LAYOUT would not "
                         "match the checkpoint's and the load would be "
                         "refused. See docs/design/P3-GATE-ANALYSIS.md 5.1 "
                         "for why the gated arm runs the dense trace.")
    ap.add_argument("--selftest", action="store_true",
                    help="run the readout-power and read-at-the-cue selftest")
    args = ap.parse_args(argv)

    # line-buffered stdout: the tables must survive a redirected run
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if args.selftest:
        return selftest(args.ckpt, seed=args.seed)

    gaps = [int(g) for g in args.gaps.split(",") if g.strip()]
    alpha = alphabet_symbols(args.alphabet)
    alpha_size = 256 if alpha is None else len(alpha)

    t0 = time.time()
    m, info = mb.load_frozen(args.ckpt, gated=args.gated)
    t_load = time.time() - t0

    print("=" * 78)
    print("D5 probe -- capacity (delta-h) vs usable content (decodable payload)")
    print("=" * 78)
    print(f"checkpoint : {args.ckpt}")
    print(f"            dim={info['dim']} layers={info['layers']} "
          f"sigma_mean={info['sigma_mean']:.4f} "
          f"half_life_mean={info['half_life_mean_B']:.1f} B "
          f"n_gt_8B={info['n_gt_8B']} n_gt_16B={info['n_gt_16B']}")
    print(f"payload alphabet: {args.alphabet} ({alpha_size} symbols) "
          f"-> chance {1.0 / alpha_size:.4f}/byte, "
          f"{(1.0 / alpha_size) ** 4:.3g} exact-4")
    print(f"read mode : {args.mode}")
    print(f"n         : {args.n} per gap   gaps: {gaps}   seed: {args.seed}")
    print(f"per-key   : {'on (grouping ' + args.perkey_group + ')' if args.perkey else 'off'}"
          f"{'  + exact low-entropy-key arm' if (args.perkey and args.perkey_exact) else ''}")
    print(f"readout   : mb._ridge_score (imported, unmodified), lambdas {LAMS}")
    print("-" * 78)

    results = []
    for d in gaps:
        res = run_gap(m, d, args.n, args.seed, args, alpha)
        results.append(res)
        print_gap_table(res, alpha_size)
        t = res["timing"]
        print(f"    [timing] forward {t['forward_s']:.1f}s "
              f"(+nuisance {t['nuisance_forward_s']:.1f}s)  "
              f"ridge {t['ridge_s']:.1f}s  total {t['total_s']:.1f}s")

    total = time.time() - t0
    print("=" * 78)
    print(f"total runtime {total:.1f}s (model load {t_load:.1f}s)")

    out = {
        "tool": "tools/d5_probe.py",
        "question": "D5 -- capacity (delta-h) vs usable content (decodable payload)",
        "checkpoint": args.ckpt,
        "ckpt_info": info,
        "n": args.n, "seed": args.seed, "gaps": gaps,
        "alphabet": args.alphabet, "alphabet_size": alpha_size,
        "mode": args.mode, "perkey": args.perkey,
        "perkey_group": args.perkey_group,
        "perkey_exact": bool(args.perkey and args.perkey_exact),
        "readout": "tools/membench/model_backend.py:_ridge_score (unmodified)",
        "lambdas": list(LAMS),
        "state_reset": ("per-sample: the batch is fed with m.new_states(B) so "
                        "every sample starts from a zero state; no state "
                        "crosses samples; no weight updates"),
        "runtime_s": total,
        "results": results,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
