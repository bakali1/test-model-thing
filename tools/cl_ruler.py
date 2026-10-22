#!/usr/bin/env python3
"""tools/cl_ruler.py -- the CL-RULER (PLAN-2026-10-21-CONTINUAL-LEARNING.md
§1, step CL-0).

THE CRITERION
-------------
Three numbers, one ruler, a control on every cell (plan §1):

  ACQUIRE        after a stream of new facts, the model recalls the new facts
  RETAIN         the model still recalls the facts it knew BEFORE the stream
  NO REGRESSION  the held-out CE on the general corpus has not degraded

The pre-registered thresholds (plan §1, fixed before any run):
  ACQUIRE       >= 0.50 exact 4-byte payload recall at d=32, ON A TRAINED
                model.  A probe on a frozen checkpoint is a DIAGNOSTIC.
  RETAIN        >= 0.90 of the pre-stream recall, on a replayed set of
                pre-stream facts that were not in the stream.
  NO REGRESSION  dCE <= 0.02 nats/byte on the general ruler.

WHAT THIS TOOL IS
-----------------
A MEASUREMENT INSTRUMENT, built so that it cannot lie in the ways this
programme's instruments have already lied (D1/D3/D4, RESULTS.md §1-§7):

  * every cell carries its OWN MATCHED NULL (permuted labels, same split,
    same training rows) AND a NUISANCE CONTROL (the same readout on a target
    that is PROVABLY ABSENT from the state);
  * a cell that does not beat its own null is NULL, never PASS;
  * CLASS COVERAGE is reported next to every accuracy, and a coverage-limited
    cell is UNPOWERED;
  * the payload uses a MATCHED ALPHABET (16 or 4 symbols) and the alphabet
    size is PART OF THE RULER SETTING -- at 256 symbols the plan's
    exact-4-byte threshold is unreachable by arithmetic (0.0225^4 = 2.6e-7);
  * the state is read AT THE CUE: every sequence is a FIXED-LENGTH sequence
    built from the fixed-gap generator (generators.delayed_copy,
    gap_mode="fixed"), so the state read at t == L-1 IS the state at the
    cue and NO zero padding is ever inserted (defect D4).  The length is
    ASSERTED and reported;
  * every number carries its ruler setting (n, seed, alphabet, gap, facts).
    "A CE without its SE is not a result."

THE INSTRUMENTS ARE REUSED, NOT REINVENTED
-------------------------------------------
  * the readout: tools/membench/model_backend.py `_ridge_score` (the FIXED
    one, imported, never re-implemented);
  * the readout conventions -- matched null, coverage report, verdict
    (SIGNAL / NULL / UNPOWERED), the state-at-the-cue forward and the
    selftest -- are IMPORTED from tools/d5_probe.py, so the two instruments
    cannot drift apart;
  * the fixed-gap fact stream: tools/membench/generators.py `delayed_copy`
    (gap_mode="fixed") and `get_batch`;
  * the CE cell: tools/membench/model_backend.py `model_transfer_ce`, which
    is the CE cell of tools/eval.py (same slices, same bytes, same SE).

THE FACT STREAM AND THE TWO CELLS
---------------------------------
The stream is N facts (`--facts`, default 64), each fact = (4-byte key ->
4-byte payload), presented IN ORDER, one stream per sample.  For a gap d:

  S  (the stream cell)   [F_1][F_2]...[F_N][ d random gap bytes][CUE=KEY_N]
                         length 8N+d+4, CONSTANT across samples
      ACQUIRE      readout target = PAY_N (the NEWEST fact, lag d behind
                   the cue -- the plan's "at d=32")
      RETAIN_POST  readout target = PAY_1 (the OLDEST fact, lag 8(N-1)+d
                   behind the same cue)

  P  (the pre-stream cell) [F_1][ 8(N-1)+d random bytes ][CUE=KEY_N]
                         length 8N+d+4 -- the SAME length, the SAME first
                         fact, the SAME cue; the ONLY difference from S is
                         that the new facts are absent (their bytes are
                         replaced by noise of the same length)
      RETAIN_PRE   readout target = PAY_1 (the same old fact, at the same
                   lag, before the new facts were written)

  RETAIN = RETAIN_POST / RETAIN_PRE at the SAME lag, same n, same seed, each
  side against its own matched null and its own nuisance control.

  NUISANCE: the same sequences with the TARGET payload(s) replaced by fresh
  random bytes OF THE SAME ALPHABET, so the target is provably absent from
  the state.  A readout that beats its null there is a leak, not a memory.

WHAT MODE `probe` IS (AND IS NOT)
---------------------------------
`probe` scores a FROZEN checkpoint on a synthetic fact stream.  The stream
is real (an ordered stream of key->payload facts), but NO weight is updated,
so the ACQUIRE/RETAIN numbers are DIAGNOSTICS of the read path, not the
goal.  The plan says so: "a probe on a frozen checkpoint is diagnostic, not
a pass."  `train` (the facts IN the loss) is the goal; it REFUSES to run
here and names exactly what is missing -- see the refusal text.

USAGE
-----
    .venv/bin/python tools/cl_ruler.py \
        --ckpt runs/taxis/A_p4_wd0_c.npz --mode probe \
        --gaps 8,32 --n 2048 --alphabet 16 --out cl_ruler.json
    .venv/bin/python tools/cl_ruler.py --selftest
    .venv/bin/python tools/cl_ruler.py --resummarize cl_ruler.json --out x.json
        # re-derive the top-level verdicts of a stored run with THIS code
        # (deterministic: it re-reads the stored numbers, it measures nothing)
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
from tools import d5_probe as d5

# reused, not re-implemented -------------------------------------------------
LAMS = d5.LAMS                       # the FIXED lambda grid of mb._ridge_score
alphabet_symbols = d5.alphabet_symbols
states_at_cue = d5.states_at_cue     # the state AT THE CUE (fixed-length batch)
states_at_positions = d5.states_at_positions
make_cell = d5.make_cell              # readout + MATCHED NULL + coverage + verdict
coverage_report = d5.coverage_report

KEY_LEN = G.KEY_LEN                  # 4
PAYLOAD_LEN = G.PAYLOAD_LEN          # 4
FACT_LEN = KEY_LEN + PAYLOAD_LEN     # 8

# the pre-registered thresholds (plan §1).  Fixed BEFORE any run.
ACQUIRE_EXACT_THRESHOLD = 0.50       # exact 4-byte recall at d=32, trained model
RETAIN_RATIO_THRESHOLD = 0.90        # post-stream / pre-stream recall
NO_REGRESSION_MAX_DCE = 0.02         # nats/byte

# The state's USABLE HORIZON (D5, PLAN §0): the payload decodes to ~128 B and
# is gone by ~132 B.  A cell whose target sits further back than this
# measures NOTHING -- it is not a weak signal, it is no signal at all.  The
# RETAIN cell is the one that gets this wrong by construction: with the
# ruler's default --facts 64 the oldest fact of the stream sits
# 8*(64-1)+d = 512+d bytes behind the cue, i.e. FOUR TIMES the horizon, and
# the ratio is then a ratio of two noises.  Every lag is checked against
# this constant and a cell beyond it is UNMEASURED, never a PASS.
HORIZON_B = 128


# ---------------------------------------------------------------------------
# the fact stream
# ---------------------------------------------------------------------------
def _draw(rng, alpha, size):
    if alpha is None:
        return rng.integers(0, 256, size=size).astype(np.uint8)
    return alpha[rng.integers(0, len(alpha), size=size)]


def gen_fact_stream(n_facts, n, rng, alpha):
    """n INDEPENDENT streams of n_facts facts each.

    A stream is an ordered list of (key -> payload) facts.  Keys use the full
    256-symbol byte space (they are addresses); payloads use the MATCHED
    ALPHABET (`alpha`), so the null is 1/|alpha| per byte and the exact-4
    metric is not hopeless (plan §1 ALPHABET CAVEAT).

    Returns dict with
      keys      (n, n_facts, 4) uint8
      payloads  (n, n_facts, 4) uint8
      facts     (n, n_facts, 8) uint8  -- key||payload per fact
    """
    n_facts = int(n_facts)
    n = int(n)
    if n_facts < 2:
        raise ValueError("the stream needs at least 2 facts (the newest one "
                         "is the ACQUIRE target, the oldest one the RETAIN "
                         "target)")
    keys = rng.integers(0, 256, size=(n, n_facts, KEY_LEN)).astype(np.uint8)
    payloads = _draw(rng, alpha, (n, n_facts, PAYLOAD_LEN))
    facts = np.concatenate([keys, payloads], axis=2)
    return {"keys": keys, "payloads": payloads, "facts": facts}


def build_sequences(kind, d, stream, rng, alpha):
    """Build the fixed-length byte sequences of one cell.

    kind:
      "stream"        S  = [F_1..F_N][d gap bytes][CUE=KEY_N]
      "prestream"     P  = [F_1][8(N-1)+d noise bytes][CUE=KEY_N]
      "stream_nuis"   S with the ACQUIRE target payload (PAY_N) and the
                      RETAIN target payload (PAY_1) replaced by fresh random
                      bytes OF THE SAME ALPHABET
      "prestream_nuis" P with PAY_1 replaced the same way
      "stream_nuis1"   S with ONLY the ACQUIRE target payload (PAY_N)
                       replaced (the newest fact).  This is the nuisance
                       control for a cell whose target is the NEWEST fact:
                       the target is then provably absent from the state,
                       while the older facts -- and therefore the state's
                       content -- are untouched.

    The tail of the "stream" cell is produced by the FIXED-GAP GENERATOR
    (generators.delayed_copy(d, gap_mode="fixed")): [KEY][PAYLOAD][gap][KEY],
    so the cue is the last KEY_LEN bytes of the sequence and the state read
    at t == L-1 IS the state AT THE CUE.  No zero padding is ever inserted.

    Every row has EXACTLY the same length L = 8N + d + 4 (asserted).
    """
    d = int(d)
    facts = stream["facts"]
    keys = stream["keys"]
    payloads = stream["payloads"]
    n, n_facts = facts.shape[0], facts.shape[1]
    L = FACT_LEN * n_facts + d + KEY_LEN

    if kind in ("stream_nuis", "prestream_nuis", "stream_nuis1"):
        pay = payloads.copy()
        if kind == "stream_nuis":
            pay[:, n_facts - 1] = _draw(rng, alpha, (n, PAYLOAD_LEN))
            pay[:, 0] = _draw(rng, alpha, (n, PAYLOAD_LEN))
        elif kind == "stream_nuis1":
            pay[:, n_facts - 1] = _draw(rng, alpha, (n, PAYLOAD_LEN))
        else:
            pay[:, 0] = _draw(rng, alpha, (n, PAYLOAD_LEN))
        facts = np.concatenate([keys, pay], axis=2)

    seq = np.zeros((n, L), dtype=np.uint8)
    for i in range(n):
        if kind.startswith("stream"):
            # the fixed-gap generator gives [KEY][PAYLOAD][gap d][KEY]
            cell, _t, _info = G.delayed_copy(d, rng=rng, gap_mode="fixed")
            cell = np.asarray(cell, dtype=np.uint8).copy()
            cell[0:KEY_LEN] = keys[i, n_facts - 1]
            cell[KEY_LEN:KEY_LEN + PAYLOAD_LEN] = facts[i, n_facts - 1, KEY_LEN:]
            cell[KEY_LEN + PAYLOAD_LEN + d:] = keys[i, n_facts - 1]
            seq[i] = np.concatenate([facts[i, 0:n_facts - 1].ravel(), cell])
        else:
            noise = rng.integers(0, 256, size=FACT_LEN * (n_facts - 1) + d)
            noise = noise.astype(np.uint8)
            seq[i] = np.concatenate([facts[i, 0].ravel(), noise,
                                     keys[i, n_facts - 1]])
    # the fixed-length assertion (defect D4): every row is exactly L bytes,
    # so the state read at t == L-1 is the state at the cue, with no padding.
    assert seq.shape == (n, L), (seq.shape, (n, L))
    assert L == FACT_LEN * n_facts + d + KEY_LEN
    return seq


# ---------------------------------------------------------------------------
# one gap
# ---------------------------------------------------------------------------
def setting_of(args, gap, alpha_size, extra=None):
    """The ruler setting of every number: n, seed, alphabet, gap, facts."""
    s = {
        "n": int(args.n),
        "sample_count": int(args.n),
        "seed": int(args.seed),
        "alphabet": str(args.alphabet),
        "alphabet_size": int(alpha_size),
        "gap": int(gap),
        "facts": int(args.facts),
        "sequence_length": int(FACT_LEN * args.facts + gap + KEY_LEN),
        "readout": "tools/membench/model_backend.py:_ridge_score (imported, "
                   "unmodified)",
        "lambdas": list(LAMS),
        "state_read": ("AT THE CUE: fixed-length sequences built from "
                       "generators.delayed_copy(gap_mode='fixed'); the "
                       "sequence ENDS with the cue bytes, so the state read "
                       "at t == L-1 is the state at the cue; NO zero padding"),
        "state_reset": ("per-sample: the batch is fed with m.new_states(B), "
                        "so every sample starts from a zero state; no state "
                        "crosses samples"),
        "mode": args.mode,
    }
    if extra:
        s.update(extra)
    return s


def run_gap(m, d, args, alpha):
    """Every cell of one gap, with its own null, nuisance, coverage, verdict."""
    t_start = time.time()
    alpha_size = 256 if alpha is None else len(alpha)
    rng = np.random.default_rng((args.seed * 1_000_003 + d) % (2 ** 32))

    # -- the fact streams (one per sample) --------------------------------
    t0 = time.time()
    stream = gen_fact_stream(args.facts, args.n, rng, alpha)
    t_gen = time.time() - t0

    # -- the sequences ------------------------------------------------------
    t0 = time.time()
    seq_S = build_sequences("stream", d, stream, rng, alpha)
    seq_P = build_sequences("prestream", d, stream, rng, alpha)
    seq_Sn = build_sequences("stream_nuis", d, stream, rng, alpha)
    seq_Pn = build_sequences("prestream_nuis", d, stream, rng, alpha)
    L = seq_S.shape[1]
    assert seq_P.shape[1] == L and seq_Sn.shape[1] == L and seq_Pn.shape[1] == L
    t_gen += time.time() - t0

    # -- the forwards (the state AT THE CUE) -------------------------------
    t0 = time.time()
    X_S = states_at_cue(m, seq_S)
    t_fwd = time.time() - t0
    t0 = time.time()
    X_P = states_at_cue(m, seq_P)
    t_fwd += time.time() - t0
    t0 = time.time()
    X_Sn = states_at_cue(m, seq_Sn)
    t_fwd += time.time() - t0
    t0 = time.time()
    X_Pn = states_at_cue(m, seq_Pn)
    t_fwd += time.time() - t0

    # -- the targets --------------------------------------------------------
    Y_new = stream["payloads"][:, args.facts - 1].astype(np.int64)   # PAY_N
    Y_old = stream["payloads"][:, 0].astype(np.int64)               # PAY_1
    setg = setting_of(args, d, alpha_size)

    # -- the cells ----------------------------------------------------------
    t0 = time.time()
    cells = []
    c = make_cell("acquire", X_S, Y_new, args.seed, alpha_size,
                  note=("the NEWEST fact of the stream: the target payload is "
                        f"d={d} bytes behind the cue, in a stream of "
                        f"{args.facts} facts"),
                  )
    c["setting"] = dict(setg, lag_bytes=int(d), target="newest fact (PAY_N)")
    cells.append(c)

    c = make_cell("retain_post", X_S, Y_old, args.seed, alpha_size,
                  note=("the OLDEST fact of the stream, read from the SAME "
                        "state at the SAME cue: it sits "
                        f"{FACT_LEN * (args.facts - 1) + d} bytes behind the "
                        "cue, after the other facts were written"))
    c["setting"] = dict(setg, lag_bytes=int(FACT_LEN * (args.facts - 1) + d),
                         target="oldest fact (PAY_1), after the stream")
    cells.append(c)

    c = make_cell("retain_pre", X_P, Y_old, args.seed, alpha_size,
                  note=("the SAME old fact BEFORE the new facts were written: "
                        "same length, same first fact, same cue; the bytes "
                        "where the new facts would be are random noise"))
    c["setting"] = dict(setg, lag_bytes=int(FACT_LEN * (args.facts - 1) + d),
                         target="oldest fact (PAY_1), before the stream")
    cells.append(c)

    c = make_cell("nuisance_acquire", X_Sn, Y_new, args.seed, alpha_size,
                  note=("NUISANCE CONTROL: the ACQUIRE target payload is "
                        "replaced by fresh random bytes of the SAME alphabet, "
                        "so it is PROVABLY ABSENT from the state.  A readout "
                        "that beats its null here is a leak, not a memory."),
                  expect_signal=False)
    c["setting"] = dict(setg, lag_bytes=int(d),
                         target="newest fact (PAY_N), PROVABLY ABSENT")
    cells.append(c)

    c = make_cell("nuisance_retain_post", X_Sn, Y_old, args.seed, alpha_size,
                  note=("NUISANCE CONTROL for RETAIN_POST: the RETAIN target "
                        "payload is replaced by fresh random bytes of the "
                        "same alphabet (provably absent from the state)."),
                  expect_signal=False)
    c["setting"] = dict(setg, lag_bytes=int(FACT_LEN * (args.facts - 1) + d),
                         target="oldest fact (PAY_1), PROVABLY ABSENT")
    cells.append(c)

    c = make_cell("nuisance_retain_pre", X_Pn, Y_old, args.seed, alpha_size,
                  note=("NUISANCE CONTROL for RETAIN_PRE: the RETAIN target "
                        "payload is replaced by fresh random bytes of the "
                        "same alphabet (provably absent from the state)."),
                  expect_signal=False)
    c["setting"] = dict(setg, lag_bytes=int(FACT_LEN * (args.facts - 1) + d),
                         target="oldest fact (PAY_1), PROVABLY ABSENT")
    cells.append(c)
    t_ridge = time.time() - t0

    by = {c["cell"]: c for c in cells}

    # -- the RETAIN ratio (post / pre at the SAME lag) ---------------------
    pre, post = by["retain_pre"], by["retain_post"]
    ratio = None
    ratio_se = None
    ratio_exact = None
    if pre["per_byte"] and post["per_byte"] and pre["per_byte"] > 0:
        ratio = post["per_byte"] / pre["per_byte"]
        ratio_se = ratio * math.sqrt(
            (pre["se_per_byte"] / pre["per_byte"]) ** 2 +
            (post["se_per_byte"] / post["per_byte"]) ** 2)
    if pre["exact4"] is not None and post["exact4"] is not None:
        if pre["exact4"] > 0:
            ratio_exact = post["exact4"] / pre["exact4"]
    if pre["verdict"] != "SIGNAL" or post["verdict"] != "SIGNAL":
        retain_status = "UNMEASURED"
        retain_why = ("one side of the ratio does not beat its own matched "
                      f"null (pre={pre['verdict']}, post={post['verdict']}): "
                      "the ratio would be a ratio of two nulls")
    elif ratio is None:
        retain_status = "UNMEASURED"
        retain_why = "the pre-stream recall is zero; the ratio is undefined"
    else:
        retain_status = "MEASURED"
        retain_why = ""
    retain = {
        "status": retain_status,
        "why": retain_why,
        "ratio_per_byte": ratio,
        "ratio_per_byte_se": ratio_se,
        "ratio_exact4": ratio_exact,
        "threshold": RETAIN_RATIO_THRESHOLD,
        "threshold_note": ("plan §1: RETAIN passes at >= 0.90 of the "
                           "pre-stream recall, measured on a replayed set of "
                           "pre-stream facts that were NOT in the stream.  In "
                           "probe mode the 'pre-stream' fact is the OLDEST "
                           "fact of the same stream (there is no earlier "
                           "training phase), so this ratio is the "
                           "interference-retention proxy, not the plan's "
                           "replay metric."),
        "setting": dict(setg),
        "pre": {k: pre[k] for k in ("per_byte", "exact4", "se", "se_per_byte",
                                    "null_per_byte", "null_exact4",
                                    "coverage", "verdict", "lam")},
        "post": {k: post[k] for k in ("per_byte", "exact4", "se",
                                      "se_per_byte", "null_per_byte",
                                      "null_exact4", "coverage", "verdict",
                                      "lam")},
    }

    # -- the ACQUIRE verdict ----------------------------------------------
    acq = by["acquire"]
    nuis_a = by["nuisance_acquire"]
    if acq["verdict"] != "SIGNAL":
        acq_status = "UNMEASURED"
        acq_why = (f"the acquire cell does not beat its own matched null "
                   f"(verdict {acq['verdict']})")
    elif nuis_a["verdict"] == "SIGNAL":
        acq_status = "UNMEASURED"
        acq_why = ("the NUISANCE control beats its own null: the readout "
                   "invents signal, so the acquire cell cannot be trusted")
    else:
        acq_status = "DIAGNOSTIC"
        acq_why = ("measured on a FROZEN checkpoint with the facts NOT in "
                   "the loss: a diagnostic of the read path, not the goal "
                   "(plan §1: 'a probe on a frozen checkpoint is diagnostic, "
                   "not a pass')")
    acquire = {
        "status": acq_status,
        "why": acq_why,
        "per_byte": acq["per_byte"],
        "per_byte_se": acq["se_per_byte"],
        "exact4": acq["exact4"],
        "exact4_se": acq["se"],
        "null_per_byte": acq["null_per_byte"],
        "null_exact4": acq["null_exact4"],
        "coverage": acq["coverage"],
        "verdict": acq["verdict"],
        "threshold": ACQUIRE_EXACT_THRESHOLD,
        "threshold_note": ("plan §1: ACQUIRE passes at >= 0.50 exact 4-byte "
                           "recall at d=32 ON A TRAINED MODEL; the threshold "
                           f"is only measurable at a matched alphabet "
                           f"({alpha_size} symbols here: chance "
                           f"{(1.0 / alpha_size) ** PAYLOAD_LEN:.3g} "
                           "exact-4).  In probe mode there is no pass/fail."),
        "setting": dict(setg),
    }

    return {
        "gap": int(d),
        "n": int(args.n),
        "facts": int(args.facts),
        "sequence_length": int(L),
        "fixed_length": True,
        "padding": "none (every sequence is exactly L bytes; asserted)",
        "read_method": ("the state at the last byte of a fixed-length batch "
                        "== the state AT THE CUE (the sequence ends with the "
                        "cue bytes)"),
        "cells": cells,
        "acquire": acquire,
        "retain": retain,
        "timing": {"stream_s": t_gen, "forward_s": t_fwd, "ridge_s": t_ridge,
                   "total_s": time.time() - t_start},
    }


# ---------------------------------------------------------------------------
# the CE cell (NO REGRESSION)
# ---------------------------------------------------------------------------
def run_ce_cell(m, args):
    """The held-out CE, on tools/eval.py's own slices and bytes.

    Reuses `mb.model_transfer_ce`, which is the CE cell of the existing ruler
    (the same `_slices` windows over data/eval, the same next-byte CE, the
    same SE across slices).  The pre-registered NO REGRESSION metric is
    dCE = post-stream CE - pre-stream CE <= 0.02 nats/byte.

    In PROBE mode the checkpoint is frozen and the fact stream is not in the
    loss, so no parameter changes: pre == post and dCE == 0 BY CONSTRUCTION.
    That is NOT evidence of no regression, and the cell says so.
    """
    t0 = time.time()
    res = mb.model_transfer_ce(m, ckpt=args.ckpt, seed=args.seed, verbose=True)
    dt = time.time() - t0
    ce = float(res["value"])
    se = float(res["se"]) if res["se"] == res["se"] else None
    if args.mode == "probe":
        status = "UNMEASURED"
        why = ("the checkpoint is FROZEN and the fact stream is not in the "
               "loss, so no parameter changes and dCE == 0 BY CONSTRUCTION.  "
               "A zero dCE here is not evidence of no regression; the CE "
               "itself (the floor the model must not fall below) IS "
               "measured and reported.")
        dce = 0.0
    else:
        status = "MEASURED"
        why = ""
        dce = None
    return {
        "status": status,
        "why": why,
        "held_out_ce_nats_per_byte": ce,
        "held_out_ce_se": se,
        "n_slices": int(res["n_slices"]),
        "scored_bytes": int(res["total_bytes"]),
        "unigram_ce": float(res["unigram"]),
        "bigram_ce": float(res["bigram"]),
        "delta_ce": dce,
        "delta_ce_se": None if dce is None else (se * math.sqrt(2)
                                                if se else None),
        "threshold": NO_REGRESSION_MAX_DCE,
        "threshold_note": ("plan §1: NO REGRESSION passes at dCE <= 0.02 "
                           "nats/byte.  NOTE: the model is 0.56 nats/byte "
                           "ABOVE the order-8 byte n-gram on the same bytes "
                           "(RESULTS.md §6), so the CE ruler is the FLOOR, "
                           "not the headline."),
        "estimator": res.get("estimator"),
        "setting": {"n": int(args.n), "seed": int(args.seed),
                    "slices": int(res["n_slices"]),
                    "bytes_per_slice": 20000,
                    "corpus": "data/eval/**/* (tools/eval.py's own slices)",
                    "mode": args.mode},
        "runtime_s": dt,
    }


# ---------------------------------------------------------------------------
# the TRAINED mode: the CL-2 arm, scored with the plan's three numbers
# ---------------------------------------------------------------------------
def cl2_meta(ckpt):
    """The CL-2 metadata a CL-2 checkpoint carries (tools/cl2_train.py)."""
    d = np.load(ckpt, allow_pickle=False)
    meta = {}
    for k in d.files:
        if k.startswith("cl2_"):
            v = d[k]
            meta[k[4:]] = v.item() if v.shape == () else v
    return meta, d


def board_stream(board, n, n_facts, rng, alpha):
    """A stream of facts drawn from a BOARD (the trainer's make_board)."""
    nb = board["keys"].shape[0]
    idx = rng.integers(0, nb, size=(n, n_facts))
    keys = board["keys"][idx]
    pays = board["payloads"][idx]
    return {"keys": keys, "payloads": pays,
            "facts": np.concatenate([keys, pays], axis=2)}


def make_head_cell(name, X, Y, W, b, seed, alpha_size, prior_byte=None,
                   note=None):
    """A cell read out by the TRAINED MEMORY HEAD -- the model's OWN trained
    read path (a 256-way head per payload byte on the state at the cue).

    The same discipline as d5.make_cell: a MATCHED NULL (the SAME readout on
    PERMUTED labels, same rows), the class COVERAGE, and the same verdict
    rule (SIGNAL / NULL / UNPOWERED, 2 SE over the null).  The head is NOT
    refit here -- it is the model's own parameter -- so the null is the
    readout's agreement with a permuted labelling rather than a refit on
    permuted labels; that is the honest null for a FIXED readout.

    The GLOBAL BYTE PRIOR is reported next to it: the accuracy of the
    constant predictor "always the most frequent byte of the corpus".  The
    FSP arm was declared a null because its head scored 0.685 against a
    prior of 0.817 -- the objective was satisfied without the state.  A head
    that does not beat the prior is not a memory.
    """
    n = X.shape[0]
    Y = np.asarray(Y, dtype=np.int64)
    logits = X @ W + b
    pred = logits.reshape(n, Y.shape[1], 256).argmax(-1)
    per_byte = float((pred == Y).mean())
    exact4 = float((pred == Y).all(axis=1).mean())
    perm = np.random.default_rng(seed + 7919).permutation(n)
    null_per_byte = float((pred == Y[perm]).mean())
    null_exact4 = float((pred == Y[perm]).all(axis=1).mean())
    tr, te = d5._split(n, seed)
    cov = coverage_report(Y, tr, te, alpha_size)
    se = float(math.sqrt(max(exact4 * (1 - exact4), 0.0) / max(len(te), 1)))
    out = {
        "cell": name, "n": int(n),
        "readout": ("the TRAINED memory head (the CL-2 checkpoint's Wm/bm) "
                    "applied to the state at the cue"),
        "per_byte": per_byte, "exact4": exact4, "se": se,
        "se_per_byte": d5._se_per_byte(per_byte, n),
        "null_per_byte": null_per_byte, "null_exact4": null_exact4,
        "null_method": ("permuted labels, same rows, the SAME fixed readout "
                        "(a fixed readout is not refit on the null)"),
        "coverage": cov,
        "label_alphabet_size": int(alpha_size),
        "chance_per_byte": 1.0 / alpha_size,
        "chance_exact4": (1.0 / alpha_size) ** Y.shape[1],
    }
    if prior_byte is not None:
        hit = (Y == int(prior_byte))
        out["prior_byte"] = int(prior_byte)
        out["prior_per_byte"] = float(hit.mean())
        out["prior_exact4"] = float(hit.all(axis=1).mean())
        out["beats_prior"] = bool(per_byte > out["prior_per_byte"])
    if note:
        out["note"] = note
    out["verdict"] = d5.verdict(out, cov, alpha_size, null_per_byte)
    return out


def run_gap_trained(models, heads, d, args, alpha, boards, prior_byte):
    """Every cell of one gap, for a TRAINED (CL-2) arm.

    The cells are built from the CL-2 BOARDS, not from fresh random streams:
      board B  the facts of the STREAM (the ones the phase-2 loss saw)
      board C  a HELD-OUT board, never trained on
      board A  the PRE-STREAM facts, replayed (the RETAIN set)
    """
    t_start = time.time()
    alpha_size = 256 if alpha is None else len(alpha)
    rng = np.random.default_rng((args.seed * 1_000_003 + d) % (2 ** 32))
    setg = setting_of(args, d, alpha_size)
    lag_cue = int(d) + PAYLOAD_LEN          # the payload's distance to the cue
    cells = []

    # -- the ACQUIRE cells ------------------------------------------------
    for cname, bname, bnote in (
            ("acquire", "B",
             "the NEW facts of the stream (board B), read at the cue"),
            ("acquire_heldout", "C",
             "a HELD-OUT board (board C) that was NEVER trained on: the same "
             "geometry, the same readout, facts the model has never seen.  A "
             "memorisation of board B cannot pass this cell.")):
        board = boards[bname]
        stream = board_stream(board, args.n, args.facts, rng, alpha)
        seq = build_sequences("stream", d, stream, rng, alpha)
        X = states_at_cue(models["post"], seq)
        Y = stream["payloads"][:, args.facts - 1].astype(np.int64)
        c = d5.make_cell(cname, X, Y, args.seed, alpha_size, note=bnote)
        c["setting"] = dict(setg, lag_bytes=int(d), lag_bytes_cue=lag_cue,
                             board=bname, readout="ridge (mb._ridge_score)")
        cells.append(c)
        hc = make_head_cell(cname + "_head", X, Y, heads["post"][0],
                            heads["post"][1], args.seed, alpha_size,
                            prior_byte=prior_byte, note=bnote)
        hc["setting"] = dict(setg, lag_bytes=int(d), lag_bytes_cue=lag_cue,
                              board=bname, readout="the trained memory head")
        cells.append(hc)
        # the NUISANCE control: the target payload is replaced by fresh
        # random bytes of the same alphabet, so it is PROVABLY ABSENT.
        seqn = build_sequences("stream_nuis1", d, stream, rng, alpha)
        Xn = states_at_cue(models["post"], seqn)
        cn = d5.make_cell(cname + "_nuisance", Xn, Y, args.seed, alpha_size,
                           note="NUISANCE: the target payload is replaced by "
                                "fresh random bytes of the same alphabet, so "
                                "it is provably absent from the state.",
                           expect_signal=False)
        cn["setting"] = dict(setg, lag_bytes=int(d), lag_bytes_cue=lag_cue,
                              board=bname, readout="ridge (mb._ridge_score)")
        cells.append(cn)
        hn = make_head_cell(cname + "_nuisance_head", Xn, Y, heads["post"][0],
                            heads["post"][1], args.seed, alpha_size,
                            prior_byte=prior_byte,
                            note="NUISANCE (head): the target payload is "
                                 "provably absent from the state.")
        hn["setting"] = dict(setg, lag_bytes=int(d), lag_bytes_cue=lag_cue,
                              board=bname, readout="the trained memory head")
        cells.append(hn)

    # -- the RETAIN replay (board A) at BOTH checkpoints --------------------
    board = boards["A"]
    stream = board_stream(board, args.n, args.facts, rng, alpha)
    seq = build_sequences("stream", d, stream, rng, alpha)
    Y = stream["payloads"][:, args.facts - 1].astype(np.int64)
    Xpre = states_at_cue(models["pre"], seq)
    Xpost = states_at_cue(models["post"], seq)
    seqn = build_sequences("stream_nuis1", d, stream, rng, alpha)
    Xn = states_at_cue(models["post"], seqn)
    ret = {"lag_bytes": int(d), "lag_bytes_cue": lag_cue,
           "board": "A (the pre-stream facts, replayed)",
           "in_horizon": bool(lag_cue <= HORIZON_B)}
    for tag, X, mdl, hd in (("pre", Xpre, models["pre"], heads["pre"]),
                            ("post", Xpost, models["post"], heads["post"])):
        c = d5.make_cell(f"retain_replay_{tag}", X, Y, args.seed, alpha_size,
                         note=("the PRE-STREAM facts (board A) replayed, read "
                               f"at the cue of the {tag}-stream checkpoint"))
        c["setting"] = dict(setg, lag_bytes=int(d), lag_bytes_cue=lag_cue,
                            board="A", readout="ridge (mb._ridge_score)")
        cells.append(c)
        hc = make_head_cell(f"retain_replay_{tag}_head", X, Y, hd[0], hd[1],
                            args.seed, alpha_size, prior_byte=prior_byte,
                            note=("the PRE-STREAM facts (board A) replayed, "
                                  f"read with the {tag}-stream model's own "
                                  "head"))
        hc["setting"] = dict(setg, lag_bytes=int(d), lag_bytes_cue=lag_cue,
                              board="A", readout="the trained memory head")
        cells.append(hc)
        ret[tag] = {"ridge": c, "head": hc}
    cn = d5.make_cell("retain_replay_nuisance", Xn, Y, args.seed, alpha_size,
                      note="NUISANCE for the replay: the target payload is "
                           "provably absent from the state.",
                      expect_signal=False)
    cn["setting"] = dict(setg, lag_bytes=int(d), lag_bytes_cue=lag_cue,
                          board="A", readout="ridge (mb._ridge_score)")
    cells.append(cn)
    hn = make_head_cell("retain_replay_nuisance_head", Xn, Y, heads["post"][0],
                        heads["post"][1], args.seed, alpha_size,
                        prior_byte=prior_byte,
                        note="NUISANCE (head) for the replay.")
    hn["setting"] = dict(setg, lag_bytes=int(d), lag_bytes_cue=lag_cue,
                          board="A", readout="the trained memory head")
    cells.append(hn)

    # -- the RETAIN ratio (post / pre, the same rows, the same readout) ---
    ratios = {}
    for readout in ("ridge", "head"):
        pre = ret["pre"][readout]
        post = ret["post"][readout]
        r = {"pre_per_byte": pre["per_byte"], "post_per_byte": post["per_byte"],
             "pre_exact4": pre["exact4"], "post_exact4": post["exact4"],
             "pre_verdict": pre["verdict"], "post_verdict": post["verdict"],
             "pre_null_per_byte": pre["null_per_byte"],
             "post_null_per_byte": post["null_per_byte"],
             "pre_coverage": pre["coverage"], "post_coverage": post["coverage"]}
        r["ratio_per_byte"] = (post["per_byte"] / pre["per_byte"]
                               if pre["per_byte"] > 0 else None)
        r["ratio_exact4"] = (post["exact4"] / pre["exact4"]
                             if pre["exact4"] > 0 else None)
        if not ret["in_horizon"]:
            r["status"] = "UNMEASURED"
            r["why"] = (f"the target sits {lag_cue} bytes behind the cue, "
                        f"beyond the state's usable horizon ({HORIZON_B} B, "
                        "D5): the cell measures nothing")
        elif pre["verdict"] != "SIGNAL" or post["verdict"] != "SIGNAL":
            r["status"] = "UNMEASURED"
            r["why"] = (f"one side does not beat its own matched null "
                        f"(pre={pre['verdict']}, post={post['verdict']}): the "
                        "ratio would be a ratio of two nulls")
        elif r["ratio_per_byte"] is None:
            r["status"] = "UNMEASURED"
            r["why"] = "the pre-stream recall is zero; the ratio is undefined"
        else:
            r["status"] = "MEASURED"
            r["why"] = ""
        ratios[readout] = r

    by = {c["cell"]: c for c in cells}
    # -- the ACQUIRE verdict (the pre-registered threshold is at d=32) ----
    acq = {"per_readout": {}}
    for readout, cname in (("ridge", "acquire"),
                            ("head", "acquire_head")):
        c = by[cname]
        nu = by[cname.replace("acquire", "acquire_nuisance")] if readout == "ridge" \
            else by["acquire_nuisance_head"]
        if c["verdict"] != "SIGNAL":
            status = "UNMEASURED"
            why = (f"the {readout} cell does not beat its own matched null "
                   f"(verdict {c['verdict']})")
        elif nu["verdict"] == "SIGNAL":
            status = "UNMEASURED"
            why = ("the NUISANCE control beats its own null: the readout "
                   "invents signal, so the cell cannot be trusted.  NOTE: on "
                   "a TRAINED board the nuisance (the key is kept, the payload "
                   "is replaced) is not provably absent from the MODEL's "
                   "knowledge -- the model may answer from its WEIGHTS, which "
                   "is a memorisation signature and not a state read; on the "
                   "HELD-OUT board no such association can exist, so a leak "
                   "there would be a real leak.")
        else:
            status = "MEASURED"
            why = ""
        acq["per_readout"][readout] = {
            "status": status, "why": why,
            "per_byte": c["per_byte"], "exact4": c["exact4"],
            "se": c.get("se"), "se_per_byte": c.get("se_per_byte"),
            "null_per_byte": c["null_per_byte"],
            "null_exact4": c["null_exact4"],
            "coverage": c["coverage"], "verdict": c["verdict"],
            "prior_per_byte": c.get("prior_per_byte"),
            "prior_exact4": c.get("prior_exact4"),
            "beats_prior": c.get("beats_prior"),
            "nuisance_verdict": nu["verdict"],
            "nuisance_per_byte": nu["per_byte"],
            "nuisance_null_per_byte": nu["null_per_byte"],
            "threshold": ACQUIRE_EXACT_THRESHOLD,
            "setting": c["setting"],
        }
    # the HELD-OUT acquire cell (the criterion's own wording: "a held-out set
    # of the same shape")
    for readout, cname in (("ridge", "acquire_heldout"),
                            ("head", "acquire_heldout_head")):
        c = by[cname]
        nu = by["acquire_heldout_nuisance"] if readout == "ridge" \
            else by["acquire_heldout_nuisance_head"]
        if c["verdict"] != "SIGNAL":
            status = "UNMEASURED"
            why = (f"the {readout} cell does not beat its own matched null "
                   f"(verdict {c['verdict']})")
        elif nu["verdict"] == "SIGNAL":
            status = "UNMEASURED"
            why = "the NUISANCE control beats its own null"
        else:
            status = "MEASURED"
            why = ""
        acq["per_readout"][readout + "_heldout"] = {
            "status": status, "why": why,
            "per_byte": c["per_byte"], "exact4": c["exact4"],
            "se": c.get("se"), "se_per_byte": c.get("se_per_byte"),
            "null_per_byte": c["null_per_byte"],
            "null_exact4": c["null_exact4"],
            "coverage": c["coverage"], "verdict": c["verdict"],
            "prior_per_byte": c.get("prior_per_byte"),
            "prior_exact4": c.get("prior_exact4"),
            "beats_prior": c.get("beats_prior"),
            "nuisance_verdict": nu["verdict"],
            "nuisance_per_byte": nu["per_byte"],
            "nuisance_null_per_byte": nu["null_per_byte"],
            "threshold": ACQUIRE_EXACT_THRESHOLD,
            "setting": c["setting"],
        }
    return {
        "gap": int(d), "n": int(args.n), "facts": int(args.facts),
        "sequence_length": int(seq.shape[1]), "fixed_length": True,
        "cells": cells, "acquire": acq, "retain": ratios,
        "retain_lag": {"lag_bytes": int(d), "lag_bytes_cue": lag_cue,
                       "horizon_B": HORIZON_B, "in_horizon": ret["in_horizon"]},
        "timing": {"total_s": time.time() - t_start},
    }


# ---------------------------------------------------------------------------
# the train mode -- REFUSES
# ---------------------------------------------------------------------------
TRAIN_REFUSAL = """
[REFUSED] --mode train: this checkpoint is not a CL-2 checkpoint.

The CL-RULER's TRAIN path scores a model whose facts were IN THE LOSS.  That
is now the CL-2 trainer (tools/cl2_train.py), and the ruler's train mode
needs its checkpoint: one that carries the cl2_* metadata (the boards, the
alphabet, the fact geometry) and the trained memory head (Wm/bm).

The five things this mode used to name as missing, and where they live now:
  1. THE FACT IN THE LOSS ............ tools/cl2_train.py: the memory loss
     (a 256-way head on the state AT THE CUE) plus the behavioural CE.
  2. A BYTE-EXACT FACT-CORPUS LOADER .. tools/cl2_train.py: the corpus is
     built as raw uint8 (no UTF-8 decoder anywhere near it).
  3. PHASES AND A REPLAY SET .......... tools/cl2_train.py --pre-steps /
     --phase1-steps / --phase2-steps, board A (pre) and board B (stream),
     with the optimizer state carried across the boundary.
  4. A BEHAVIOURAL READOUT ............ tools/cl2_train.py: the model's own
     output at the four slots after the cue.
  5. A FACT-TRAINED CHECKPOINT ........ runs/cl2/<tag>_<phase>_end.npz.

WHAT IS STILL NOT MEASURED BY THIS TOOL (and is said so in the output):
  * the READOUT of a fixed head is the model's own; the ridge readout is
    reported beside it as the state-information number;
  * a cell that does not beat its own matched null is UNMEASURED, never a
    PASS;
  * a target beyond the state's usable horizon (128 B, D5) is UNMEASURED.
"""


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def selftest(ckpt, seed=0):
    """(i) power: a linearly decodable target IS recovered;
       (ii) no invented signal: an uncorrelated label is NOT recovered;
       (iii) the state is read AT THE CUE: bytes AFTER the cue do not move
              the features (the direct test for the D4 padding bug);
       (iv) the fixed-length assertion holds (no zero padding in a cell)."""
    ok = True
    print("== CL-RULER selftest ==")

    # ---- (i)+(ii) readout power / no invented signal -------------------
    rng = np.random.default_rng(seed)
    n, f, k = 600, 64, 16
    proto = rng.standard_normal((k, f))
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    y_lin = rng.integers(0, k, size=n)
    Xs = (3.0 * proto[y_lin] + 0.3 * rng.standard_normal((n, f))).astype(np.float32)
    r = d5._ridge(Xs, y_lin.astype(np.int64), seed)
    r_rand = d5._ridge(Xs, rng.integers(0, k, size=n).astype(np.int64), seed)
    print(f"  (i)  linearly decodable target : per_byte={pct(r['per_byte'])} "
          f"(expect > 90%)  -> {'OK' if r['per_byte'] > 0.9 else 'FAIL'}")
    print(f"      random (uncorrelated) label: per_byte={pct(r_rand['per_byte'])} "
          f"(chance {pct(1.0 / k)}, expect < 25%)  "
          f"-> {'OK' if r_rand['per_byte'] < 0.25 else 'FAIL'}")
    ok &= bool(r["per_byte"] > 0.9 and r_rand["per_byte"] < 0.25)

    # ---- (iii) the state is read at the cue ----------------------------
    m, info = mb.load_frozen(ckpt)
    d = 8
    rng2 = np.random.default_rng(seed + 1)
    stream = gen_fact_stream(4, 16, rng2, None)
    seq = build_sequences("stream", d, stream, rng2, None)
    L = seq.shape[1]
    cue = L - 1                      # the last cue byte
    Xa = seq.copy()
    Xb = np.concatenate([seq, rng2.integers(0, 256, size=(16, 37)).astype(np.uint8)],
                        axis=1)
    sa = states_at_positions(m, Xa, np.full(16, cue))
    sb = states_at_positions(m, Xb, np.full(16, cue))
    dmax = float(np.abs(sa - sb).max())
    print(f"  (iii) bytes AFTER the cue changed -> the features at the cue "
          f"moved by {dmax:.3g} (max abs)  -> "
          f"{'OK (identical)' if dmax == 0.0 else 'FAIL'}")
    ok &= (dmax == 0.0)
    sbug = states_at_cue(m, Xb)
    dbug = float(np.abs(sbug - sb).max())
    print(f"      D4 fingerprint: reading the PADDED batch at t == L-1 "
          f"differs from the cue read by {dbug:.3g} -> the padding bug is "
          f"real in that path (this is why the cells are fixed-length)")
    same = float(np.abs(states_at_cue(m, Xa) - sa).max())
    print(f"      fixed-length read == cue read: "
          f"{'OK' if same == 0.0 else 'FAIL'}")
    ok &= (same == 0.0)

    # ---- (iv) the fixed-length assertion --------------------------------
    lens = {seq.shape[1], build_sequences("prestream", d, stream, rng2, None).shape[1]}
    fixed = (len(lens) == 1 and seq.shape[1] == 8 * 4 + d + 4)
    print(f"  (iv) every cell sequence is exactly "
          f"{FACT_LEN * 4 + d + 4} B (no padding) -> "
          f"{'OK' if fixed else 'FAIL'}")
    ok &= bool(fixed)

    print(f"== selftest {'PASS' if ok else 'FAIL'} ==")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# the three numbers, one verdict each
# ---------------------------------------------------------------------------
def summarize(results, ce, mode):
    """The top-level verdicts block, as a pure function of the stored
    per-gap numbers and the CE cell (so it can be re-derived from a stored
    run with --resummarize: it measures nothing).

    In PROBE mode nothing can be a PASS: the plan says a probe on a frozen
    checkpoint is a DIAGNOSTIC, not a pass (PLAN §1).  A cell that does not
    beat its own null is UNMEASURED, whatever the mode.
    """
    measured_label = "DIAGNOSTIC" if mode == "probe" else "MEASURED"
    acq_status = (measured_label
                  if all(r["acquire"]["status"] != "UNMEASURED" for r in results)
                  else "UNMEASURED")
    ret_status = (measured_label
                  if all(r["retain"]["status"] == "MEASURED" for r in results)
                  else "UNMEASURED")
    return {
        "acquire": {
            "status": acq_status,
            "note": ("DIAGNOSTIC on a frozen checkpoint (the facts are not in "
                     "the loss); the pre-registered threshold applies to a "
                     "TRAINED model only"),
            "per_gap": [{"gap": r["gap"], "status": r["acquire"]["status"],
                         "per_byte": r["acquire"]["per_byte"],
                         "per_byte_se": r["acquire"]["per_byte_se"],
                         "exact4": r["acquire"]["exact4"],
                         "null_per_byte": r["acquire"]["null_per_byte"],
                         "null_exact4": r["acquire"]["null_exact4"],
                         "coverage": r["acquire"]["coverage"],
                         "verdict": r["acquire"]["verdict"],
                         "threshold": ACQUIRE_EXACT_THRESHOLD,
                         "setting": r["acquire"]["setting"]}
                        for r in results],
        },
        "retain": {
            "status": ret_status,
            "note": ("the post/pre ratio at the SAME lag, same n, same seed; "
                     "each side against its own matched null and its own "
                     "nuisance control"),
            "per_gap": [{"gap": r["gap"], "status": r["retain"]["status"],
                         "ratio_per_byte": r["retain"]["ratio_per_byte"],
                         "ratio_per_byte_se": r["retain"]["ratio_per_byte_se"],
                         "ratio_exact4": r["retain"]["ratio_exact4"],
                         "pre": r["retain"]["pre"], "post": r["retain"]["post"],
                         "threshold": RETAIN_RATIO_THRESHOLD,
                         "setting": r["retain"]["setting"]}
                        for r in results],
        },
        "no_regression": {"status": ce.get("status"), "why": ce.get("why", ""),
                           "held_out_ce_nats_per_byte":
                               ce.get("held_out_ce_nats_per_byte"),
                           "held_out_ce_se": ce.get("held_out_ce_se"),
                           "delta_ce": ce.get("delta_ce"),
                           "threshold": NO_REGRESSION_MAX_DCE,
                           "setting": ce.get("setting", {})},
    }


# ---------------------------------------------------------------------------
# the three numbers of a TRAINED arm: PASS / FAIL / UNMEASURED
# ---------------------------------------------------------------------------
def summarize_trained(results, ce_post, ce_pre, threshold_gap=32):
    """The criterion, as a verdict against the pre-registered thresholds.

      ACQUIRE       exact-4 >= 0.50 at d=32, on a HELD-OUT set of the same
                    shape, read with the model's OWN trained head.  The ridge
                    readout is reported beside it (the state-information
                    number) and the trained-board cell beside that.
      RETAIN        post/pre >= 0.90 on the REPLAYED pre-stream facts, at a
                    lag inside the state's usable horizon.
      NO REGRESSION dCE <= 0.02 nats/byte between the pre-stream and the
                    post-stream checkpoint.

    UNMEASURED is never PASS: a cell that does not beat its own matched
    null, a cell whose nuisance control leaks, a cell beyond the horizon and
    a missing pre-stream checkpoint are all UNMEASURED.
    """
    gaps = [r["gap"] for r in results]
    g = threshold_gap if threshold_gap in gaps else (max(gaps) if gaps else None)
    res = next((r for r in results if r["gap"] == g), None)

    def verdict_of(cell, threshold, higher_is_better=True, key="exact4"):
        if cell is None:
            return {"status": "UNMEASURED", "why": "the cell was not measured"}
        if cell.get("status") == "UNMEASURED":
            return {"status": "UNMEASURED", "why": cell.get("why", "")}
        v = cell.get(key)
        if v is None:
            return {"status": "UNMEASURED", "why": "the cell has no value"}
        ok = (v >= threshold) if higher_is_better else (v <= threshold)
        return {"status": "PASS" if ok else "FAIL",
                "why": (f"{key} = {v:.4f} vs the pre-registered threshold "
                        f"{threshold}")}

    acq = {}
    if res is not None:
        pr = res["acquire"]["per_readout"]
        for key in ("head", "head_heldout", "ridge", "ridge_heldout"):
            c = pr.get(key)
            acq[key] = {
                "status": c["status"] if c else "UNMEASURED",
                "why": c.get("why", "") if c else "",
                "per_byte": c.get("per_byte") if c else None,
                "exact4": c.get("exact4") if c else None,
                "null_per_byte": c.get("null_per_byte") if c else None,
                "null_exact4": c.get("null_exact4") if c else None,
                "coverage": c.get("coverage") if c else None,
                "verdict": c.get("verdict") if c else None,
                "prior_per_byte": c.get("prior_per_byte") if c else None,
                "prior_exact4": c.get("prior_exact4") if c else None,
                "beats_prior": c.get("beats_prior") if c else None,
            }
    acq_verdict = (verdict_of(res["acquire"]["per_readout"].get("head_heldout"),
                               ACQUIRE_EXACT_THRESHOLD) if res else
                   {"status": "UNMEASURED", "why": "no gap was measured"})
    acq_verdict_trained = (verdict_of(res["acquire"]["per_readout"].get("head"),
                                       ACQUIRE_EXACT_THRESHOLD) if res else
                           {"status": "UNMEASURED", "why": "no gap was measured"})

    ret = {}
    ret_verdict = {"status": "UNMEASURED", "why": "no gap was measured"}
    if res is not None:
        for readout in ("head", "ridge"):
            r = res["retain"][readout]
            ret[readout] = r
        r = res["retain"]["head"]
        if r["status"] == "UNMEASURED":
            ret_verdict = {"status": "UNMEASURED", "why": r["why"]}
        else:
            v = r["ratio_per_byte"]
            ret_verdict = {
                "status": "PASS" if (v is not None and
                                     v >= RETAIN_RATIO_THRESHOLD) else "FAIL",
                "why": (f"post/pre = {v:.4f} vs the pre-registered threshold "
                        f"{RETAIN_RATIO_THRESHOLD} (the head's own recall of "
                        "the replayed pre-stream facts)")}

    # NO REGRESSION
    if ce_post.get("held_out_ce_nats_per_byte") is None:
        nr = {"status": "UNMEASURED", "why": "the CE cell was skipped",
              "delta_ce": None}
    elif ce_pre.get("held_out_ce_nats_per_byte") is None:
        nr = {"status": "UNMEASURED",
              "why": ("no PRE-STREAM checkpoint was given, so the DELTA cannot "
                      "be formed (dCE is a difference, not a level)"),
              "held_out_ce_nats_per_byte":
                  ce_post.get("held_out_ce_nats_per_byte"),
              "delta_ce": None}
    else:
        dce = (ce_post["held_out_ce_nats_per_byte"]
               - ce_pre["held_out_ce_nats_per_byte"])
        nr = {"status": "PASS" if dce <= NO_REGRESSION_MAX_DCE else "FAIL",
              "why": f"dCE = {dce:.4f} vs the pre-registered threshold "
                     f"{NO_REGRESSION_MAX_DCE} nats/byte",
              "held_out_ce_pre": ce_pre["held_out_ce_nats_per_byte"],
              "held_out_ce_post": ce_post["held_out_ce_nats_per_byte"],
              "delta_ce": dce}
    return {
        "threshold_gap": g,
        "acquire": {"verdict": acq_verdict,
                    "verdict_trained_board": acq_verdict_trained,
                    "per_readout": acq,
                    "threshold": ACQUIRE_EXACT_THRESHOLD,
                    "note": ("the criterion's headline is the HELD-OUT board "
                             "(board C, never trained on); the trained-board "
                             "cell is the 'did it memorise the stream' "
                             "number")},
        "retain": {"verdict": ret_verdict, "per_readout": ret,
                   "threshold": RETAIN_RATIO_THRESHOLD},
        "no_regression": dict(nr, threshold=NO_REGRESSION_MAX_DCE),
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def pct(x):
    if x is None:
        return "   n/a"
    return f"{100.0 * x:6.2f}%"


def print_gap(res):
    print(f"\n--- gap {res['gap']}  (n={res['n']}, {res['facts']} facts in the "
          f"stream, {res['sequence_length']} B/sequence, "
          f"alphabet {res['cells'][0]['setting']['alphabet_size']}) ---")
    print(f"    read: {res['read_method']}")
    print(f"    {'cell':<22} {'per_byte':>9} {'exact4':>9} {'null_pb':>9} "
          f"{'null_ex4':>9} {'cov':>7} {'trdist':>7} {'lag':>6}  verdict")
    for c in res["cells"]:
        cov = c.get("coverage", {})
        print(f"    {c['cell']:<22} {pct(c.get('per_byte')):>9} "
              f"{pct(c.get('exact4')):>9} {pct(c.get('null_per_byte')):>9} "
              f"{pct(c.get('null_exact4')):>9} "
              f"{pct(cov.get('coverage')):>7} "
              f"{cov.get('train_distinct', '-'):>7} "
              f"{c['setting'].get('lag_bytes', '-'):>6}  {c.get('verdict')}")
    a = res["acquire"]
    print(f"    ACQUIRE  : per_byte {pct(a['per_byte'])} +/- "
          f"{pct(a['per_byte_se'])} | exact4 {pct(a['exact4'])} | null "
          f"{pct(a['null_per_byte'])}/{pct(a['null_exact4'])} | coverage "
          f"{pct(a['coverage'].get('coverage'))} | {a['status']}")
    r = res["retain"]
    print(f"    RETAIN   : pre {pct(r['pre']['per_byte'])} (null "
          f"{pct(r['pre']['null_per_byte'])}, {r['pre']['verdict']}) | post "
          f"{pct(r['post']['per_byte'])} (null {pct(r['post']['null_per_byte'])}, "
          f"{r['post']['verdict']}) | ratio "
          f"{('%.3f' % r['ratio_per_byte']) if r['ratio_per_byte'] is not None else 'n/a'}"
          f" +/- {('%.3f' % r['ratio_per_byte_se']) if r['ratio_per_byte_se'] is not None else 'n/a'}"
          f" | {r['status']}")
    t = res["timing"]
    print(f"    [timing] forward {t['forward_s']:.1f}s  ridge {t['ridge_s']:.1f}s"
          f"  total {t['total_s']:.1f}s")


# ---------------------------------------------------------------------------
# the TRAINED mode (--mode train): a CL-2 arm, scored
# ---------------------------------------------------------------------------
def run_trained(args, gaps):
    """The three numbers of the criterion, on a CL-2 checkpoint.

    The cells are built from the CL-2 BOARDS (the trainer's own make_board,
    imported -- never re-implemented), so the geometry is the TRAINING
    geometry, and the RETAIN replay is the pre-stream board A measured at
    the pre-stream checkpoint and at the post-stream one.
    """
    t0 = time.time()
    meta, dz = cl2_meta(args.ckpt)
    if "alphabet" not in meta or "board_seed_a" not in meta:
        print(TRAIN_REFUSAL, file=sys.stderr)
        return 2
    import tools.cl2_train as cl2          # the boards, imported
    alpha_kind = str(meta["alphabet"])
    alpha = cl2.alphabet_symbols(alpha_kind)
    alpha_size = 256 if alpha is None else len(alpha)
    facts = int(meta.get("facts", args.facts))
    if facts != args.facts:
        print(f"[geometry] the checkpoint was trained with {facts} facts per "
              f"stream (the CLI said {args.facts}): the CHECKPOINT's geometry "
              f"is used -- the cells must match the training")
        args.facts = facts
    board_size = int(meta.get("board_size", 512))
    seeds = {"A": int(meta.get("board_seed_a", 101)),
             "B": int(meta.get("board_seed_b", 202)),
             "C": int(meta.get("board_seed_c", 303))}
    boards = {k: cl2.make_board(s, board_size, alpha) for k, s in seeds.items()}
    try:
        prior = json.loads(meta.get("prior", "{}"))
    except Exception:
        prior = {}
    prior_byte = prior.get("argmax_alpha")

    print("=" * 78)
    print("CL-RULER -- TRAINED mode: the criterion of PLAN-2026-10-21 §1")
    print("=" * 78)
    print(f"checkpoint : {args.ckpt} (phase {meta.get('phase')}, "
          f"step {meta.get('step')})")
    if args.ckpt_pre:
        print(f"pre-stream: {args.ckpt_pre}")
    else:
        print("pre-stream: NONE  -> the RETAIN ratio and the dCE are "
              "UNMEASURED (they are differences, not levels)")
    print(f"geometry  : {facts} facts/stream, alphabet {alpha_kind} "
          f"({alpha_size} symbols), boards A/B/C = {board_size} facts each "
          f"(seeds {seeds['A']}/{seeds['B']}/{seeds['C']})")
    print(f"ruler     : n={args.n} seed={args.seed} gaps={gaps} | "
          f"the ACQUIRE threshold is at d=32 (exact-4 >= "
          f"{ACQUIRE_EXACT_THRESHOLD}), the RETAIN threshold is a ratio >= "
          f"{RETAIN_RATIO_THRESHOLD}, the NO REGRESSION threshold is dCE <= "
          f"{NO_REGRESSION_MAX_DCE} nats/B")
    print(f"readouts  : the model's OWN trained head (primary) and "
          f"mb._ridge_score (the state-information number); the global byte "
          f"prior: byte {prior_byte}")

    m_post, info = mb.load_frozen(args.ckpt)
    models = {"post": m_post}
    heads = {"post": (np.asarray(dz["Wm"], np.float32),
                      np.asarray(dz["bm"], np.float32))}
    if args.ckpt_pre:
        m_pre, _i2 = mb.load_frozen(args.ckpt_pre)
        dp = np.load(args.ckpt_pre)
        models["pre"] = m_pre
        heads["pre"] = (np.asarray(dp["Wm"], np.float32),
                        np.asarray(dp["bm"], np.float32))
    print("-" * 78)

    results = []
    for d in gaps:
        res = run_gap_trained(models, heads, d, args, alpha, boards, prior_byte)
        results.append(res)
        print(f"\n--- gap {d} (n={args.n}, {facts} facts, "
              f"{res['sequence_length']} B/sequence) ---")
        print(f"    {'cell':<32} {'per_byte':>9} {'exact4':>9} {'null_pb':>9} "
              f"{'null_ex4':>9} {'cov':>7}  verdict")
        for c in res["cells"]:
            cov = c.get("coverage", {})
            print(f"    {c['cell']:<32} {pct(c.get('per_byte')):>9} "
                  f"{pct(c.get('exact4')):>9} {pct(c.get('null_per_byte')):>9} "
                  f"{pct(c.get('null_exact4')):>9} "
                  f"{pct(cov.get('coverage')):>7}  {c.get('verdict')}")
        for k, c in res["acquire"]["per_readout"].items():
            print(f"    ACQUIRE[{k:<18}] per_byte {pct(c['per_byte'])} | "
                  f"exact4 {pct(c['exact4'])} | null {pct(c['null_per_byte'])}"
                  f"/{pct(c['null_exact4'])} | prior {pct(c.get('prior_per_byte'))}"
                  f"/{pct(c.get('prior_exact4'))} | coverage "
                  f"{pct(c['coverage'].get('coverage'))} | {c['status']}")
        for ro in ("head", "ridge"):
            r = res["retain"][ro]
            print(f"    RETAIN[{ro:<6}] pre {pct(r['pre_per_byte'])} "
                  f"({r['pre_verdict']}) | post {pct(r['post_per_byte'])} "
                  f"({r['post_verdict']}) | ratio "
                  f"{('%.3f' % r['ratio_per_byte']) if r['ratio_per_byte'] is not None else 'n/a'}"
                  f" | {r['status']}")
        print(f"    [retain lag] {res['retain_lag']['lag_bytes_cue']} B behind "
              f"the cue (the horizon is {HORIZON_B} B): "
              f"{'inside' if res['retain_lag']['in_horizon'] else 'OUTSIDE'}")

    # -- the CE cells: the pre-stream and the post-stream checkpoint ------
    def ce_cell(m, ckpt):
        if args.no_ce:
            return {"status": "SKIPPED", "why": "--no-ce was given",
                    "held_out_ce_nats_per_byte": None, "held_out_ce_se": None,
                    "delta_ce": None, "threshold": NO_REGRESSION_MAX_DCE}
        r = mb.model_transfer_ce(m, ckpt=ckpt, seed=args.seed, verbose=True)
        return {"status": "MEASURED", "why": "",
                "held_out_ce_nats_per_byte": float(r["value"]),
                "held_out_ce_se": float(r["se"]),
                "n_slices": int(r["n_slices"]),
                "scored_bytes": int(r["total_bytes"]),
                "unigram_ce": float(r["unigram"]),
                "bigram_ce": float(r["bigram"]),
                "delta_ce": None, "threshold": NO_REGRESSION_MAX_DCE,
                "estimator": r.get("estimator"),
                "setting": {"corpus": "data/eval/**/* (tools/eval.py's own "
                                      "slices)", "ckpt": ckpt}}

    print("\n--- NO REGRESSION (the held-out CE, pre-stream and post-stream) ---")
    ce_post = ce_cell(m_post, args.ckpt)
    print(f"    post-stream CE {ce_post['held_out_ce_nats_per_byte']}")
    if args.ckpt_pre:
        ce_pre = ce_cell(models["pre"], args.ckpt_pre)
        print(f"    pre-stream  CE {ce_pre['held_out_ce_nats_per_byte']}")
    else:
        ce_pre = {"status": "MISSING", "why": "no --ckpt-pre",
                  "held_out_ce_nats_per_byte": None}

    verdicts = summarize_trained(results, ce_post, ce_pre)
    print("\n" + "=" * 78)
    print("THE THREE NUMBERS")
    print("=" * 78)
    a = verdicts["acquire"]
    print(f"  ACQUIRE       : {a['verdict']['status']}  ({a['verdict']['why']})")
    print(f"                 held-out board, trained head: "
          f"per_byte {pct(a['per_readout'].get('head_heldout', {}).get('per_byte'))}"
          f" | exact4 {pct(a['per_readout'].get('head_heldout', {}).get('exact4'))}"
          f" | null {pct(a['per_readout'].get('head_heldout', {}).get('null_per_byte'))}"
          f" | prior {pct(a['per_readout'].get('head_heldout', {}).get('prior_per_byte'))}")
    print(f"                 trained board, trained head: "
          f"per_byte {pct(a['per_readout'].get('head', {}).get('per_byte'))}"
          f" | exact4 {pct(a['per_readout'].get('head', {}).get('exact4'))}"
          f" | null {pct(a['per_readout'].get('head', {}).get('null_per_byte'))}"
          f" | prior {pct(a['per_readout'].get('head', {}).get('prior_per_byte'))}")
    print(f"                 held-out board, ridge: per_byte "
          f"{pct(a['per_readout'].get('ridge_heldout', {}).get('per_byte'))}"
          f" | exact4 {pct(a['per_readout'].get('ridge_heldout', {}).get('exact4'))}"
          f" | null {pct(a['per_readout'].get('ridge_heldout', {}).get('null_per_byte'))}")
    r = verdicts["retain"]
    print(f"  RETAIN        : {r['verdict']['status']}  ({r['verdict']['why']})")
    nr = verdicts["no_regression"]
    print(f"  NO REGRESSION : {nr['status']}  ({nr['why']})")

    out = {
        "tool": "tools/cl_ruler.py",
        "criterion": "CL-RULER (PLAN-2026-10-21-CONTINUAL-LEARNING.md §1, CL-0)",
        "mode": "train",
        "mode_note": ("TRAINED: the facts were IN THE LOSS (tools/cl2_train.py "
                      "CL-2).  The three numbers carry their pre-registered "
                      "thresholds.  The primary readout is the model's OWN "
                      "trained memory head; the ridge is the state-information "
                      "number beside it."),
        "checkpoint": args.ckpt,
        "ckpt_pre": args.ckpt_pre,
        "ckpt_info": info,
        "cl2_meta": {k: (v if isinstance(v, (int, float, str)) else str(v))
                     for k, v in meta.items()},
        "ruler_setting": {
            "n": int(args.n), "seed": int(args.seed),
            "alphabet": alpha_kind, "alphabet_size": alpha_size,
            "facts": facts, "gaps": gaps,
            "board_size": board_size, "board_seeds": seeds,
            "chance_per_byte": 1.0 / alpha_size,
            "chance_exact4": (1.0 / alpha_size) ** PAYLOAD_LEN,
            "readout": ("the trained memory head (primary) + "
                        "tools/membench/model_backend.py:_ridge_score"),
            "read_at_cue": True, "padding": "none (fixed-length, asserted)",
            "horizon_B": HORIZON_B,
        },
        "thresholds": {"acquire_exact4": ACQUIRE_EXACT_THRESHOLD,
                       "retain_ratio": RETAIN_RATIO_THRESHOLD,
                       "no_regression_max_dce": NO_REGRESSION_MAX_DCE},
        "results": results,
        "no_regression": ce_post,
        "no_regression_pre": ce_pre,
        "verdicts": verdicts,
        "runtime_s": time.time() - t0,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/taxis/A_p4_wd0_c.npz",
                    help="frozen checkpoint (default: %(default)s)")
    ap.add_argument("--ckpt-pre", default=None,
                    help="the PRE-STREAM checkpoint (the end of phase f1).  "
                         "The RETAIN ratio and the dCE are differences, so "
                         "without it they are UNMEASURED, not zero")
    ap.add_argument("--facts", type=int, default=64,
                    help="facts in the stream (default: %(default)s).  In "
                         "--mode train the CHECKPOINT's own fact count is "
                         "used (the geometry must match the training)")
    ap.add_argument("--gaps", default="8,32",
                    help="comma separated gaps d (default: %(default)s)")
    ap.add_argument("--n", type=int, default=2048,
                    help="samples per cell (default: %(default)s)")
    ap.add_argument("--alphabet", default="16", choices=("full", "16", "4"),
                    help="payload alphabet (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    ap.add_argument("--mode", default="probe", choices=("probe", "train"),
                    help="'probe' = frozen checkpoint scored on a fact stream "
                         "(a DIAGNOSTIC); 'train' = a CL-2 checkpoint whose "
                         "facts were IN THE LOSS: the three numbers of the "
                         "criterion, with their thresholds")
    ap.add_argument("--selftest", action="store_true",
                    help="run the power / no-invented-signal / read-at-the-cue "
                         "selftest")
    ap.add_argument("--no-ce", action="store_true",
                    help="skip the NO REGRESSION CE cell (it costs ~22 min on "
                         "this box; it is deterministic given the checkpoint)")
    ap.add_argument("--resummarize", default=None, metavar="RUN.json",
                    help="re-derive the top-level 'verdicts' block of a stored "
                         "run with THIS code (deterministic: it re-reads the "
                         "stored per-gap numbers, it measures nothing) and "
                         "write it to --out")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if args.resummarize:
        with open(args.resummarize) as fh:
            stored = json.load(fh)
        stored["verdicts"] = summarize(stored.get("results", []),
                                       stored.get("no_regression", {}),
                                       stored.get("mode", "probe"))
        out_path = args.out or args.resummarize
        with open(out_path, "w") as fh:
            json.dump(stored, fh, indent=2, default=float)
        print(f"re-summarized {args.resummarize} -> {out_path}")
        for k, v in stored["verdicts"].items():
            print(f"  {k}: {v['status']}")
        return 0

    if args.selftest:
        return selftest(args.ckpt, seed=args.seed)

    gaps = [int(g) for g in args.gaps.split(",") if g.strip()]
    if args.mode == "train":
        return run_trained(args, gaps)
    alpha = alphabet_symbols(args.alphabet)
    alpha_size = 256 if alpha is None else len(alpha)

    t0 = time.time()
    m, info = mb.load_frozen(args.ckpt)
    t_load = time.time() - t0

    print("=" * 78)
    print("CL-RULER -- the criterion of PLAN-2026-10-21-CONTINUAL-LEARNING.md §1")
    print("=" * 78)
    print(f"checkpoint : {args.ckpt}")
    print(f"            dim={info['dim']} layers={info['layers']} "
          f"sigma_mean={info['sigma_mean']:.4f} "
          f"half_life_mean={info['half_life_mean_B']:.1f} B")
    print(f"mode       : {args.mode}"
          + ("  -- DIAGNOSTIC: frozen checkpoint, the facts are NOT in the "
             "loss; the goal needs --mode train (which refuses, see below)"
             if args.mode == "probe" else ""))
    print(f"ruler setting: n={args.n} seed={args.seed} alphabet={args.alphabet} "
          f"({alpha_size} symbols) facts={args.facts} gaps={gaps}")
    print(f"             chance {1.0 / alpha_size:.4f}/byte, "
          f"{(1.0 / alpha_size) ** PAYLOAD_LEN:.3g} exact-4; "
          f"the exact-4 threshold ({ACQUIRE_EXACT_THRESHOLD}) is only "
          f"measurable at a matched alphabet")
    print(f"readout    : mb._ridge_score (imported, unmodified), lambdas {LAMS}")
    print("-" * 78)

    results = []
    for d in gaps:
        res = run_gap(m, d, args, alpha)
        results.append(res)
        print_gap(res)

    print("\n--- NO REGRESSION (held-out CE, tools/eval.py's own slices) ---")
    if args.no_ce:
        ce = {"status": "SKIPPED", "why": "--no-ce was given",
              "held_out_ce_nats_per_byte": None, "held_out_ce_se": None,
              "n_slices": 0, "scored_bytes": 0, "unigram_ce": None,
              "bigram_ce": None, "delta_ce": None, "delta_ce_se": None,
              "threshold": NO_REGRESSION_MAX_DCE, "threshold_note": "",
              "estimator": None, "runtime_s": 0.0,
              "setting": {"n": int(args.n), "seed": int(args.seed)}}
        print("    skipped (--no-ce)")
    else:
        ce = run_ce_cell(m, args)
        print(f"    held-out CE {ce['held_out_ce_nats_per_byte']:.4f} +/- "
              f"{ce['held_out_ce_se']:.4f} nats/byte ({ce['n_slices']} slices, "
              f"{ce['scored_bytes']:,} B) | unigram {ce['unigram_ce']:.4f} | "
              f"bigram {ce['bigram_ce']:.4f}")
        print(f"    dCE = {ce['delta_ce']} (threshold {ce['threshold']}) -> "
              f"{ce['status']}")
        if ce["why"]:
            print(f"    {ce['why']}")

    total = time.time() - t0
    print("=" * 78)
    print(f"total runtime {total:.1f}s (model load {t_load:.1f}s)")

    # -- the three numbers, one verdict each ------------------------------
    verdicts = summarize(results, ce, args.mode)

    out = {
        "tool": "tools/cl_ruler.py",
        "criterion": "CL-RULER (PLAN-2026-10-21-CONTINUAL-LEARNING.md §1, CL-0)",
        "mode": args.mode,
        "mode_note": ("PROBE: a frozen checkpoint scored on a synthetic fact "
                      "stream.  The stream is real (an ordered stream of "
                      "key->payload facts) but NO weight is updated, so "
                      "ACQUIRE/RETAIN are DIAGNOSTICS of the read path, not "
                      "the goal.  The goal needs --mode train, which REFUSES "
                      "and names what is missing."),
        "checkpoint": args.ckpt,
        "ckpt_info": info,
        "ruler_setting": {
            "n": int(args.n), "seed": int(args.seed),
            "alphabet": str(args.alphabet), "alphabet_size": alpha_size,
            "facts": int(args.facts), "gaps": gaps,
            "chance_per_byte": 1.0 / alpha_size,
            "chance_exact4": (1.0 / alpha_size) ** PAYLOAD_LEN,
            "readout": "tools/membench/model_backend.py:_ridge_score (unmodified)",
            "lambdas": list(LAMS),
            "read_at_cue": True,
            "padding": "none (fixed-length sequences, asserted)",
        },
        "thresholds": {"acquire_exact4": ACQUIRE_EXACT_THRESHOLD,
                       "retain_ratio": RETAIN_RATIO_THRESHOLD,
                       "no_regression_max_dce": NO_REGRESSION_MAX_DCE},
        "results": results,
        "no_regression": ce,
        "verdicts": verdicts,
        "runtime_s": total,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
