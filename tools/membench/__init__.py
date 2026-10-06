#!/usr/bin/env python3
"""membench -- synthetic long-memory benchmark suite.

Implements Priority 2 (tasks A-D) and section 10 (memory evaluation API) of
docs/plans/PLAN-2026-10-06-MEMORY-BENCHMARKS.md, as corrected by
docs/reviews/RESPONSE-2026-10-06-EXTERNAL-CRITIQUE.md section 2.5.

Self-contained: stdlib + numpy only.

Quick start
-----------
    import membench as mb

    # one batch of assoc-copy samples, gap <= 512, seeded
    b = mb.get_batch("associative_copy", 512, 128, seed=2026)
    seq, target, meta = b["seq"], b["target"], b["meta"]

    # chance-level math for every task
    mb.CHANCE_LEVELS            # exact per-byte chance per task
    mb.chance_ce("long_range_language", L=512)

    # eval API skeletons return the full section-10 dict with the exact
    # model-free baseline (and raise NotImplementedError if a model is passed)
    mb.selective_copy_accuracy(512)          # value == 2**-32
    mb.language_ce(L=512)                    # chance CE in nats/byte

Run the smoke test:  python3 tools/membench/tests/check_membench.py
"""
from __future__ import annotations

from . import generators as generators
from . import eval_api as eval_api

from .generators import (
    VOCAB, DTYPE, KEY_LEN, PAYLOAD_LEN, GAP_LADDER,
    MARKER_ALPHABET, CONTROLLED_BY_MARKER, CONTROLLED_ALPHABET, TERMINATOR,
    CHANCE_LEVELS, STATE_RESET_POLICY, TASKS,
    delayed_copy, associative_copy, selective_copy, long_range_language,
    chance_ce, get_batch,
)
from .eval_api import (
    METADATA_KEYS, retention_curve, selective_copy_accuracy,
    associative_retrieval_accuracy, language_ce, transfer_ce, cross_domain_ce,
)

__version__ = "0.1.0"
__all__ = [
    # generators
    "delayed_copy", "associative_copy", "selective_copy", "long_range_language",
    "get_batch", "chance_ce", "TASKS", "CHANCE_LEVELS",
    "VOCAB", "DTYPE", "KEY_LEN", "PAYLOAD_LEN", "GAP_LADDER",
    "MARKER_ALPHABET", "CONTROLLED_BY_MARKER", "CONTROLLED_ALPHABET",
    "TERMINATOR", "STATE_RESET_POLICY",
    # eval api
    "METADATA_KEYS", "retention_curve", "selective_copy_accuracy",
    "associative_retrieval_accuracy", "language_ce", "transfer_ce",
    "cross_domain_ce",
    # submodules
    "generators", "eval_api",
]