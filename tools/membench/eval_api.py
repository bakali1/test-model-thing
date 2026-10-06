#!/usr/bin/env python3
"""membench.eval_api -- Memory evaluation API (PLAN-2026-10-06 section 10).

The plan requires exactly these functions:

    retention_curve(d)
    selective_copy_accuracy(d)
    associative_retrieval_accuracy(d)
    language_ce()
    transfer_ce()
    cross_domain_ce()

Contract (section 10): *every* result is a dict that ALWAYS carries the
metadata

    cue type
    gap distribution
    distractor type
    state reset policy
    seed
    sample count
    chance level

plus the metric name and value.

Model-dependent parts: the real metric needs a trained/evaluable model (or a
frozen checkpoint + forward pass). Those code paths raise NotImplementedError
as soon as a `model` is passed. Called WITHOUT a model, each function still
returns the exact model-free baseline -- chance level / trivial null -- inside
the full section-10 contract dict, so the contract is testable today and the
chance rulers are pinned down before any training happens.

Self-contained: stdlib + numpy only.
"""
from __future__ import annotations

import math

from . import generators as G

# Every memory result must carry these keys (plan section 10).
METADATA_KEYS = ("cue_type", "gap_distribution", "distractor_type",
                 "state_reset_policy", "seed", "sample_count", "chance_level")

STATE_RESET_POLICY = (
    "per-sample: recurrent state reset to zero at the start of every sequence; "
    "samples i.i.d.; no state carried across samples; no weight updates during "
    "evaluation; fixed corpus split / fixed sample IDs"
)

CHANCE_PER_BYTE = 1.0 / G.VOCAB
EXACT_PAYLOAD_CHANCE = CHANCE_PER_BYTE ** G.PAYLOAD_LEN   # 2^-32
CHANCE_CE_NATS = math.log(G.VOCAB)                        # ln 256

_SEED_DEFAULT = 0
_COUNT_DEFAULT = 1024


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _meta(seed, sample_count, cue_type, gap_distribution, distractor_type,
          chance_level):
    return {
        "cue_type": cue_type,
        "gap_distribution": gap_distribution,
        "distractor_type": distractor_type,
        "state_reset_policy": STATE_RESET_POLICY,
        "seed": int(seed),
        "sample_count": int(sample_count),
        "chance_level": chance_level,
    }


def _require_model(model, what):
    if model is not None:
        raise NotImplementedError(
            f"{what}: the real metric needs a trained/evaluable model (or a "
            "frozen checkpoint plus forward pass). The model-backed eval "
            "backend is not built yet; without a model this function returns "
            "the exact chance/null baseline instead of raising.")


def _finish(meta, metric, value, unit, source, model):
    meta["metric"] = metric
    meta["value"] = value
    meta["unit"] = unit
    meta["metric_source"] = source
    meta["requires_model"] = True
    meta["model"] = None if model is None else "<unsupported>"
    return meta


def _check_d(d, what):
    d = int(d)
    if d not in G.GAP_LADDER:
        raise ValueError(f"{what}: d must be one of G.GAP_LADDER="
                         f"{G.GAP_LADDER}, got {d}")
    return d


# --------------------------------------------------------------------------
# retention curve
# --------------------------------------------------------------------------
def retention_curve(d_list, seed=_SEED_DEFAULT, sample_count=_COUNT_DEFAULT,
                    model=None):
    """d -> ||h_A(d) - h_B(d)|| with conditions identical except for the first
    4-byte key (Priority 1 protocol). Needs a frozen checkpoint -> model-backed
    path raises NotImplementedError. Model-free null computed exactly: a no-
    memory model produces identical states, i.e. delta-h = 0 at every d.
    """
    d_list = [int(x) for x in d_list]
    if not d_list or any(x < 1 for x in d_list):
        raise ValueError(f"d_list must be a non-empty list of positive "
                         f"distances, got {d_list!r}")
    meta = _meta(
        seed, sample_count,
        cue_type="state retention: causal state-norm difference h_A - h_B "
                 "after sequences identical except for the first 4-byte key",
        gap_distribution=f"fixed gap per evaluated d in {tuple(d_list)}",
        distractor_type="uniform random bytes (vocab 256); identical "
                        "distractor stream across the two conditions",
        chance_level={
            "per_byte": CHANCE_PER_BYTE,
            "exact_payload": EXACT_PAYLOAD_CHANCE,
            "note": "null (no-memory) model: delta_h(d) = 0 at every d; the "
                    "scalar prediction a^d*sqrt(1-a^2) needs the checkpoint's "
                    "actual decay sigma",
        })
    _require_model(model, "retention_curve")
    return _finish(meta, metric="retention_curve",
                   value={int(x): 0.0 for x in d_list},
                   unit="normalized state norm (null = 0)",
                   source="analytic null (no-memory baseline)", model=model)


# --------------------------------------------------------------------------
# selective copy accuracy
# --------------------------------------------------------------------------
def selective_copy_accuracy(d, n_pairs=3, seed=_SEED_DEFAULT,
                            sample_count=_COUNT_DEFAULT, model=None):
    """Exact-payload retrieval accuracy on task C at gap parameter d.
    Model-free baseline returned exactly: chance = 2^-32 for an exact 4-byte
    payload (1/256 per byte); distractors do not change the chance level.
    """
    d = _check_d(d, "selective_copy_accuracy")
    meta = _meta(
        seed, sample_count,
        cue_type="key_query: retrieve the one target pair payload while "
                 "ignoring 1..n_pairs distractor key->payload pairs and noise",
        gap_distribution=("random per sample: uniform over "
                          f"{{g in {G.GAP_LADDER} : g <= d}} (d={d})"),
        distractor_type=("uniform random bytes + 1..n_pairs random "
                         "key->payload pairs at random positions"),
        chance_level={"per_byte": CHANCE_PER_BYTE,
                      "exact_payload": EXACT_PAYLOAD_CHANCE,
                      "n_pairs_max": int(n_pairs)})
    _require_model(model, "selective_copy_accuracy")
    return _finish(meta, metric="selective_copy_accuracy",
                   value=EXACT_PAYLOAD_CHANCE,
                   unit="fraction of exact 4-byte payload matches (chance)",
                   source="chance baseline (uniform byte guessing)",
                   model=model)


# --------------------------------------------------------------------------
# associative retrieval accuracy
# --------------------------------------------------------------------------
def associative_retrieval_accuracy(d, seed=_SEED_DEFAULT,
                                   sample_count=_COUNT_DEFAULT, model=None):
    """Exact-payload retrieval accuracy on task B at gap parameter d.
    Model-free baseline returned exactly: chance = 2^-32 (1/256 per byte).
    """
    d = _check_d(d, "associative_retrieval_accuracy")
    meta = _meta(
        seed, sample_count,
        cue_type="key_query: retrieve the payload paired with the query key "
                 "across a random per-sample gap",
        gap_distribution=("random per sample: uniform over "
                          f"{{g in {G.GAP_LADDER} : g <= d}} (d={d})"),
        distractor_type="uniform random bytes (vocab 256)",
        chance_level={"per_byte": CHANCE_PER_BYTE,
                      "exact_payload": EXACT_PAYLOAD_CHANCE})
    _require_model(model, "associative_retrieval_accuracy")
    return _finish(meta, metric="associative_retrieval_accuracy",
                   value=EXACT_PAYLOAD_CHANCE,
                   unit="fraction of exact 4-byte payload matches (chance)",
                   source="chance baseline (uniform byte guessing)",
                   model=model)


# --------------------------------------------------------------------------
# language CE (task D)
# --------------------------------------------------------------------------
def language_ce(L=512, seed=_SEED_DEFAULT, sample_count=_COUNT_DEFAULT,
                model=None):
    """Next-byte CE on long_range_language sequences of length L. Baseline
    (returned without a model): the generator's own chance CE, nats per scored
    byte -- ((L-3)*ln 256 + ln 2)/(L-1). A model with perfect retention of the
    marker achieves 0 nats at the critical position.
    """
    L = int(L)
    if L < 4:
        raise ValueError(f"L must be >= 4, got {L}")
    cl = dict(G.CHANCE_LEVELS["long_range_language"])
    cl["chance_ce_nats_per_byte"] = G.chance_ce("long_range_language", L=L)
    meta = _meta(
        seed, sample_count,
        cue_type="marker_byte: byte at position 0 controls the byte at "
                 "position L-2 (non-local dependency)",
        gap_distribution=("deterministic per sample: one sequence of length "
                          f"L = {L}"),
        distractor_type="uniform random filler bytes (vocab 256)",
        chance_level=cl)
    _require_model(model, "language_ce")
    return _finish(meta, metric="language_ce",
                   value=cl["chance_ce_nats_per_byte"],
                   unit="nats per scored byte (chance)",
                   source="analytic chance CE of the generating distribution",
                   model=model)


# --------------------------------------------------------------------------
# transfer / cross-domain CE
# --------------------------------------------------------------------------
def transfer_ce(seed=_SEED_DEFAULT, sample_count=_COUNT_DEFAULT, model=None):
    """CE after transfer: capability trained on the synthetic memory suite,
    evaluated on held-out real text. Baseline (returned without a model):
    uniform-byte chance CE on the target domain, ln 256 nats/byte.
    """
    meta = _meta(
        seed, sample_count,
        cue_type="transfer: memory capability trained on synthetic tasks "
                 "(A-D), evaluated on held-out real text",
        gap_distribution="n/a (real text, no synthetic gap)",
        distractor_type="n/a -- the question is whether a fixed model trained "
                        "on synthetic memory tasks transfers to text",
        chance_level={"per_byte": CHANCE_PER_BYTE,
                      "chance_ce_nats_per_byte": math.log(G.VOCAB)})
    _require_model(model, "transfer_ce")
    return _finish(meta, metric="transfer_ce", value=math.log(G.VOCAB),
                   unit="nats per byte (chance)",
                   source="chance baseline (uniform byte guessing)",
                   model=model)


def cross_domain_ce(seed=_SEED_DEFAULT, sample_count=_COUNT_DEFAULT,
                    model=None):
    """Cross-domain CE: in-domain (e.g. Wikipedia) vs out-of-domain (e.g.
    Gutenberg novels). Baseline (returned without a model): uniform-byte
    chance CE, ln 256 nats/byte.
    """
    meta = _meta(
        seed, sample_count,
        cue_type="cross-domain: byte-CE of a model on an out-of-domain corpus "
                 "vs its in-domain CE",
        gap_distribution="n/a (real text corpora)",
        distractor_type="n/a -- domain shift, not distractors",
        chance_level={"per_byte": CHANCE_PER_BYTE,
                      "chance_ce_nats_per_byte": math.log(G.VOCAB)})
    _require_model(model, "cross_domain_ce")
    return _finish(meta, metric="cross_domain_ce", value=math.log(G.VOCAB),
                   unit="nats per byte (chance)",
                   source="chance baseline (uniform byte guessing)",
                   model=model)


EVAL_FUNCTIONS = (retention_curve, selective_copy_accuracy,
                  associative_retrieval_accuracy, language_ce, transfer_ce,
                  cross_domain_ce)