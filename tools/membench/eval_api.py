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
from . import model_backend
from .model_backend import (model_fsp_bag, model_retention_curve,
                             model_selective_copy_accuracy,
                            model_associative_retrieval_accuracy,
                            model_language_ce, model_transfer_ce,
                            model_cross_domain_ce, load_frozen)

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



def _run_model(model, backend_fn, *args, **kwargs):
    """Run a model-backed backend against a numpy_port Model or a checkpoint path.

    model=None -> caller keeps the model-free baseline.  A str is a checkpoint
    path (loaded frozen, zero state).  Any other object must be a numpy_port
    Model.  Anything else raises NotImplementedError (keeps the section-10
    contract contract testable today)."""
    if model is None:
        # The model-free baseline. Every caller keeps its own analytic
        # chance/null path below this dispatch, so returning None rather than
        # raising is what makes the section-10 contract testable today
        # without a checkpoint. Raising here (as this did at first) made
        # `fn()` with no model unreachable and turned check_membench red.
        return None
    if isinstance(model, str):
        m, _info = load_frozen(model)
    elif hasattr(model, "new_states") and hasattr(model, "load"):
        m = model
    else:
        raise NotImplementedError(
            f"model must be a checkpoint path (str), a numpy_port Model, or "
            f"None for the model-free baseline; got {type(model).__name__}")
    return backend_fn(m, *args, **kwargs)


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
    r = _run_model(model, model_retention_curve, d_list, seed=seed,
                   sample_count=sample_count)
    if r is not None:
        return r
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
    r = _run_model(model, model_selective_copy_accuracy, d, seed=seed,
                   sample_count=sample_count)
    if r is not None:
        return r
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
    r = _run_model(model, model_associative_retrieval_accuracy, d, seed=seed,
                   sample_count=sample_count)
    if r is not None:
        return r
    _require_model(model, "associative_retrieval_accuracy")
    return _finish(meta, metric="associative_retrieval_accuracy",
                   value=EXACT_PAYLOAD_CHANCE,
                   unit="fraction of exact 4-byte payload matches (chance)",
                   source="chance baseline (uniform byte guessing)",
                   model=model)


# --------------------------------------------------------------------------
# language CE (task D)
# --------------------------------------------------------------------------
def fsp_bag_accuracy(model=None, raw_path=None, tau=64, r=16,
                     seed=_SEED_DEFAULT, sample_count=1024, L=512):
    """Does the model's OWN FSP head predict the future byte-bag better than
    a static prior can?

    The D7 test. A bag of 64 bytes holds ~22 distinct symbols, and those
    symbols are largely the common ones, so a head that ignores the state
    entirely and learns the unigram distribution already scores well. The
    cell therefore reports the head's F1 AGAINST two static baselines
    measured on the same positions: the global byte-frequency prior and a
    local 4-gram prior. A head that does not beat them has learned the
    prior, and the auxiliary loss was satisfied without the state carrying
    anything.

    Baseline (returned without a model): None -- this cell has no chance
    level, it is a comparison between predictors, not an absolute score.
    """
    if raw_path is None:
        raise ValueError("fsp_bag_accuracy needs raw_path (the held-out "
                         "corpus the bag is measured on)")
    meta = _meta(
        seed, sample_count,
        cue_type="none: the target is the bag of the next tau bytes, not a "
                 "cue-target pair",
        gap_distribution=f"the FSP target window is tau={int(tau)} bytes",
        distractor_type="none",
        chance_level={"note": "no chance level: the cell compares the "
                              "model's FSP head against static priors"})
    r = _run_model(model, model_fsp_bag, raw_path=raw_path, tau=tau, r=r,
                    seed=seed, sample_count=sample_count, L=L)
    if r is not None:
        return r
    _require_model(model, "fsp_bag_accuracy")
    return _finish(meta, metric="fsp_bag_accuracy", value=None,
                   unit="F1 of the predicted byte-set (no chance level)",
                   source="a model without an FSP head cannot be scored here",
                   model=model)


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
    r = _run_model(model, model_language_ce, L=L, seed=seed,
                   sample_count=sample_count)
    if r is not None:
        return r
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
    # NOTE: the backend is model_transfer_ce, NOT transfer_ce. Dispatching to
    # this function's own name would recurse forever (transfer_ce ->
    # _run_model -> transfer_ce), which is what an earlier revision of this
    # dispatch did; the correct line sat below it, unreachable.
    r = _run_model(model, model_transfer_ce, seed=seed,
                   sample_count=sample_count)
    if r is not None:
        return r
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
    # model_cross_domain_ce, not cross_domain_ce -- see the note in transfer_ce.
    r = _run_model(model, model_cross_domain_ce, seed=seed,
                   sample_count=sample_count)
    if r is not None:
        return r
    _require_model(model, "cross_domain_ce")
    return _finish(meta, metric="cross_domain_ce", value=math.log(G.VOCAB),
                   unit="nats per byte (chance)",
                   source="chance baseline (uniform byte guessing)",
                   model=model)


EVAL_FUNCTIONS = (retention_curve, selective_copy_accuracy,
                  associative_retrieval_accuracy, language_ce, transfer_ce,
                  cross_domain_ce)