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

    if kind in ("stream_nuis", "prestream_nuis"):
        pay = payloads.copy()
        if kind == "stream_nuis":
            pay[:, n_facts - 1] = _draw(rng, alpha, (n, PAYLOAD_LEN))
            pay[:, 0] = _draw(rng, alpha, (n, PAYLOAD_LEN))
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
# the train mode -- REFUSES
# ---------------------------------------------------------------------------
TRAIN_REFUSAL = """
[REFUSED] --mode train: the CL-RULER's TRAIN path cannot be run in this repo
yet.  The repo's rule is to refuse rather than emit a plausible wrong number
(RESULTS.md: the D1 probe, the D4 padding read and the 14 plans that
measured a proxy instead of the goal).  EXACTLY what is missing:

1. THE FACT IN THE LOSS (a per-position loss MASK / WEIGHT).
   The trainer's only objective is next-byte CE: `numpy_port.Model.step_batch`
   (numpy_port.py:433) computes ONE mean CE per stream over consecutive byte
   pairs; the torch twin is gpu_port.py:1701+.  There is no way to say "only
   the 4 payload positions after the cue count, and weight them X".  A fact
   stream needs the loss restricted (or up-weighted) to the
   payload-after-the-cue positions; without that the stream is just text.
   Code path to change: the loss assembly in step_batch (numpy_port.py:491+).

2. A BYTE-EXACT FACT-CORPUS LOADER.
   `numpy_port.Runtime.train` (numpy_port.py:960) opens every training file
   with `open(file, "r", encoding="utf-8", errors="ignore")`, so arbitrary
   fact bytes (random 4-byte payloads) are silently DROPPED or mangled by the
   UTF-8 decoder.  A fact stream cannot be fed to the trainer as bytes
   today; it needs a binary path (or a reversible byte-safe encoding) in
   Runtime.train / train_gpu.py's MiReader.

3. PHASES AND A REPLAY SET.
   The trainer runs ONE corpus with ONE optimizer state and no phase
   boundary.  The ruler needs phase 1 (the pre-stream facts) -> phase 2 (the
   new facts + a replay set of the phase-1 facts), with the optimizer state
   carried across the boundary and a checkpoint saved at the boundary.  Two
   runs with a checkpoint handoff would give the boundary; the replay
   INTERLEAVING does not exist.

4. A BEHAVIOURAL READOUT (the model's OWN output after the cue).
   Once the fact is in the loss, "recalling" it is a BEHAVIOUR: the model
   must emit the payload bytes after the cue key.  Every read-path number in
   RESULTS.md comes from an OFFLINE ridge over the frozen state
   (mb._ridge_score), which measures the state, not the behaviour.  There is
   no behavioural recall readout in the repo; --mode probe cannot stand in
   for it, because the probe's readout is a lag decoder over the state.

5. A FACT-TRAINED CHECKPOINT.
   Nothing in runs/ has ever been trained on a fact stream, so the
   pre-registered ACQUIRE threshold (>= 0.50 exact 4-byte recall at d=32)
   has never been measured against anything.  Until one exists, ACQUIRE and
   RETAIN are DIAGNOSTICS on a frozen model, and NO REGRESSION is degenerate
   (dCE == 0 by construction).

ALSO UNKNOWN (an experiment, not a missing file): the write path is an
order-losing per-coordinate EMA (`state = states*a + enc`,
numpy_port.py:142), so a key->payload association has to be learned by a
GLOBAL linear head over a MIXTURE state.  D5 measured that the global
readout is not key-blind (the cue key decodes 65-165x over its null) but
that the CONTENT decays within ~12-32 B.  Whether byte-level CE can train
that association at all is the open question of CL-2 -- and it is exactly
what --mode train is for.
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
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/taxis/A_p4_wd0_c.npz",
                    help="frozen checkpoint (default: %(default)s)")
    ap.add_argument("--facts", type=int, default=64,
                    help="facts in the stream (default: %(default)s)")
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
                         "(a DIAGNOSTIC); 'train' = the facts are in the loss "
                         "and the model is updated (the goal) -- REFUSES, see "
                         "the message it prints")
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

    if args.mode == "train":
        print(TRAIN_REFUSAL, file=sys.stderr)
        return 2

    gaps = [int(g) for g in args.gaps.split(",") if g.strip()]
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
