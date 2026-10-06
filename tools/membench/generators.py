#!/usr/bin/env python3
"""membench.generators -- synthetic long-memory benchmark suite.

Priority 2 of docs/plans/PLAN-2026-10-06-MEMORY-BENCHMARKS.md (tasks A-D) as
corrected by docs/reviews/RESPONSE-2026-10-06-EXTERNAL-CRITIQUE.md section 2.5
(payload random and independent of the key; random gaps; distractor pairs).

Every sample is a numpy uint8 byte string (vocab 256). Universal shape of the
copy tasks (A/B/C):

    KEY(4B) PAYLOAD(4B) [ gap bytes ] KEY(4B)      -> target = PAYLOAD(4B)

    A. delayed_copy(d)      gap fixed = d (plan label: "fixed known gap").
                            Control: step-counting fixed dynamics suffice.
    B. associative_copy(d)  per-sample gap drawn uniformly at random from
                            GAP_LADDER = {8,32,128,512,2048} with g <= d, so
                            fixed dynamics cannot solve it; payload is uniform
                            random and statistically independent of the key.
    C. selective_copy(d)    random gap (same ladder), AND the gap contains
                            1..n_pairs distractor key->payload pairs placed at
                            random positions, plus uniform random noise.
    D. long_range_language(L): marker byte at position 0 controls the byte at
                            position L-2 (binary alphabet); positions 1..L-3
                            are uniform random filler; position L-1 is a fixed
                            terminator. The answer at the critical position
                            cannot be decided from the last few bytes.

Chance levels are exact and exposed per task (1/256 per payload byte,
2^-32 for an exact 4-byte payload, 1/2 at D's critical position for an
alphabet-aware guesser).

Dependency note: the per-sample gap in B/C is a *function of the task's d* --
uniform over the ladder elements that do not exceed the requested d, so d is
the maximum distance at which the model is probed.

Self-contained: stdlib + numpy only.
"""
from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------
# Public constants
# --------------------------------------------------------------------------
VOCAB = 256
DTYPE = np.uint8
KEY_LEN = 4
PAYLOAD_LEN = 4
GAP_LADDER = (8, 32, 128, 512, 2048)

# long_range_language: marker byte at index 0 -> controlled byte at index L-2.
MARKER_ALPHABET = np.array([0x00, 0x01], dtype=DTYPE)
CONTROLLED_BY_MARKER = {0x00: 0x41, 0x01: 0x42}        # 'A' / 'B'
CONTROLLED_ALPHABET = sorted(CONTROLLED_BY_MARKER.values())
TERMINATOR = 0xFF

# Exact analytic chance levels per task (per-byte, or per 4-byte payload).
CHANCE_LEVELS = {
    "delayed_copy": {
        "per_byte": 1.0 / VOCAB,
        "exact_payload": (1.0 / VOCAB) ** PAYLOAD_LEN,
        "note": "target payload bytes are uniform random and independent of "
                "the key; distractors are uniform random -> local frequency "
                "statistics are useless",
    },
    "associative_copy": {
        "per_byte": 1.0 / VOCAB,
        "exact_payload": (1.0 / VOCAB) ** PAYLOAD_LEN,
        "note": "same as delayed copy; the random per-sample gap adds a "
                "timing uncertainty but does not change the chance level",
    },
    "selective_copy": {
        "per_byte": 1.0 / VOCAB,
        "exact_payload": (1.0 / VOCAB) ** PAYLOAD_LEN,
        "note": "distractor pairs are uniform random and independent of the "
                "query pair, so they do not change the chance level",
    },
    "long_range_language": {
        "filler_per_byte": 1.0 / VOCAB,
        "critical_position": 1.0 / len(CONTROLLED_ALPHABET),   # 1/2
        "critical_position_uniform_guess": 1.0 / VOCAB,        # 1/256
        "terminator": 1.0,
        "note": "filler bytes uniform; the controlled byte is binary but its "
                "symbol depends on the marker  L-2 positions away; a guesser "
                "ignorant of that non-local dependency is stuck at 1/2 "
                "(alphabet-aware) or 1/256 (uniform)",
    },
}

STATE_RESET_POLICY = (
    "per-sample: recurrent state reset to zero at the start of every sequence; "
    "samples i.i.d.; no state carried across samples; evaluations never "
    "update weights"
)

TASKS = ("delayed_copy", "associative_copy", "selective_copy", "long_range_language")

_CUE_TYPE = {
    "delayed_copy": "key_echo: re-emit the stored 4-byte payload after an "
                    "identical key cue at a fixed, known gap",
    "associative_copy": "key_query: retrieve the payload paired with the query "
                        "key across a random per-sample gap (content-dependent "
                        "retention required)",
    "selective_copy": "key_query: retrieve the one target pair payload while "
                      "ignoring 1..n_pairs distractor key->payload pairs and "
                      "uniform noise",
    "long_range_language": "marker_byte: byte at position 0 controls the byte "
                           "at position L-2 (non-local dependency)",
}


# --------------------------------------------------------------------------
# RNG plumbing
# --------------------------------------------------------------------------
def _as_generator(rng):
    """Accept an int seed, a np.random.Generator, or None (nondeterministic)."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    raise TypeError(
        f"rng/seed must be an int, np.random.Generator, or None; got {type(rng)!r}")


def _rand_bytes(rng, n):
    """n uniform random bytes as an (n,) uint8 array."""
    return np.frombuffer(rng.bytes(int(n)), dtype=np.uint8).copy()


def _draw_gap(rng, d):
    """Per-sample gap: uniform over ladder elements <= the task's d choice."""
    choices = np.array([g for g in GAP_LADDER if g <= d], dtype=np.int64)
    if choices.size == 0:
        raise ValueError(f"no ladder gaps <= {d}; GAP_LADDER={GAP_LADDER}")
    return int(rng.choice(choices))


def _eq(a, b):
    return np.array_equal(a, b)


def _check_d(d, name):
    d = int(d)
    if d not in GAP_LADDER:
        raise ValueError(
            f"{name}: d must be one of GAP_LADDER={GAP_LADDER} (d is the "
            f"maximum gap at which the task is probed); got {d}")
    return d


# --------------------------------------------------------------------------
# Task A -- delayed copy (fixed known gap, control)
# --------------------------------------------------------------------------
def delayed_copy(d, seed=None, rng=None, gap_mode="fixed"):
    """Task A: KEY(4B) PAYLOAD(4B) [d distractors] KEY(4B) -> target PAYLOAD.

    gap_mode="fixed" (default): the gap is exactly d for every sample -- the
    plan's "fixed known gap" control, solvable by step-counting fixed dynamics.
    gap_mode="random": gap drawn per sample from ladder <= d (turns the control
    into a timing-robust variant; identical to associative_copy's layout).

    Returns (seq (12+d,) uint8, target (4,) uint8, info dict).
    """
    rng = _as_generator(seed if rng is None else rng)
    d = int(d)
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    if gap_mode not in ("fixed", "random"):
        raise ValueError(f"gap_mode must be 'fixed' or 'random', got {gap_mode!r}")
    gap = d if gap_mode == "fixed" else _draw_gap(rng, d)
    key = _rand_bytes(rng, KEY_LEN)
    payload = _rand_bytes(rng, PAYLOAD_LEN)
    distractors = _rand_bytes(rng, gap)
    seq = np.concatenate([key, payload, distractors, key])
    info = {
        "task": "delayed_copy", "gap": gap,
        "gap_distribution": f"fixed: gap = {d} for every sample",
        "key": key, "payload": payload,
    }
    return seq, payload.copy(), info


# --------------------------------------------------------------------------
# Task B -- associative copy (random gap; payload independent of the key)
# --------------------------------------------------------------------------
def associative_copy(d, seed=None, rng=None):
    """Task B: same shape as A, but the per-sample gap is drawn uniformly at
    random from ladder elements <= d, so a model must retain and retrieve the
    KEY->PAYLOAD association rather than count steps. Payload is uniform random
    and statistically independent of the key.

    Returns (seq (12+gap,) uint8, target (4,) uint8, info dict).
    """
    rng = _as_generator(seed if rng is None else rng)
    d = _check_d(d, "associative_copy")
    gap = _draw_gap(rng, d)
    key = _rand_bytes(rng, KEY_LEN)
    payload = _rand_bytes(rng, PAYLOAD_LEN)
    distractors = _rand_bytes(rng, gap)
    seq = np.concatenate([key, payload, distractors, key])
    info = {
        "task": "associative_copy", "gap": gap,
        "gap_distribution": ("random per sample: uniform over "
                             f"{{g in {GAP_LADDER} : g <= d}} (d={d})"),
        "key": key, "payload": payload,
    }
    return seq, payload.copy(), info


# --------------------------------------------------------------------------
# Task C -- selective copy (random gap + distractor key/value pairs)
# --------------------------------------------------------------------------
def selective_copy(d, n_pairs=3, seed=None, rng=None):
    """Task C: the Mamba-style discriminator. Random per-sample gap (ladder
    <= d). Inside the gap, 1..n_pairs uniform-random distractor key->payload
    pairs are placed at random (gap-relative) positions, interleaved with
    uniform random noise (exactly `gap` bytes total). One target pair
    (QUERY_KEY, QUERY_PAYLOAD) sits at the front of the sequence and is
    queried by repeating QUERY_KEY at the end. All keys within a sample are
    pairwise distinct, so exactly one answer exists.

    Returns (seq (12+gap,) uint8, target (4,) uint8, info dict with
    n_dist, pair_positions (gap-relative), distractor_pairs (bytes)).
    """
    rng = _as_generator(seed if rng is None else rng)
    d = _check_d(d, "selective_copy")
    n_pairs = int(n_pairs)
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")

    gap = _draw_gap(rng, d)
    max_dist = gap // (KEY_LEN + PAYLOAD_LEN)          # floor(d/8)
    n_dist = int(rng.integers(1, min(n_pairs, max_dist) + 1))

    query_key = _rand_bytes(rng, KEY_LEN)
    query_payload = _rand_bytes(rng, PAYLOAD_LEN)

    # Distractor pairs: uniform random, keys pairwise distinct and distinct
    # from the query key (unique answer).
    pairs = []
    for _ in range(n_dist):
        while True:
            k = _rand_bytes(rng, KEY_LEN)
            if not (_eq(k, query_key) or any(_eq(k, e[0]) for e in pairs)):
                break
        pairs.append((k, _rand_bytes(rng, PAYLOAD_LEN)))

    # Random noise segmentation: n_dist+1 noise blocks around n_dist pairs,
    # total noise = gap - 8*n_dist (>= 0 by construction of n_dist).
    noise_total = gap - (KEY_LEN + PAYLOAD_LEN) * n_dist
    if noise_total > 0:
        parts = rng.multinomial(noise_total,
                                np.full(n_dist + 1, 1.0 / (n_dist + 1)))
    else:
        parts = np.zeros(n_dist + 1, dtype=np.int64)

    order = rng.permutation(n_dist)
    placed = [pairs[int(o)] for o in order]          # placement order
    blocks, positions = [], []
    pos = 0
    for i in range(n_dist):
        n0 = int(parts[i])
        blocks.append(_rand_bytes(rng, n0))
        pos += n0
        k_block, p_block = placed[i]
        blocks.append(k_block)
        blocks.append(p_block)
        positions.append(pos)
        pos += KEY_LEN + PAYLOAD_LEN
    blocks.append(_rand_bytes(rng, int(parts[-1])))
    distractor_region = np.concatenate(blocks)
    assert distractor_region.shape[0] == gap

    seq = np.concatenate([query_key, query_payload, distractor_region,
                          query_key])
    info = {
        "task": "selective_copy", "gap": gap,
        "gap_distribution": ("random per sample: uniform over "
                             f"{{g in {GAP_LADDER} : g <= d}} (d={d})"),
        "n_dist": n_dist,
        "pair_positions": positions,                     # gap-relative offsets
        "distractor_pairs": [(k.tobytes(), p.tobytes()) for k, p in placed],
        "query_key": query_key.tobytes(),
        "query_payload": query_payload.tobytes(),
    }
    return seq, query_payload.copy(), info


# --------------------------------------------------------------------------
# Task D -- long-range language (structured, non-local dependency)
# --------------------------------------------------------------------------
def long_range_language(L, seed=None, rng=None):
    """Task D: one structured byte string of length L.

        [0]        marker byte        uniform over {0x00, 0x01}
        [1 .. L-3] filler             uniform random bytes (vocab 256)
        [L-2]      controlled byte    'A'(0x41) if marker==0x00 else 'B'(0x42)
        [L-1]      terminator         0xFF (fixed)

    The answer at position L-2 is a function of position 0, so it cannot be
    decided from the last few bytes (the filler is uniform and uninformative).
    target = the controlled byte.

    Returns (seq (L,) uint8, target (1,) uint8, info dict).
    """
    rng = _as_generator(seed if rng is None else rng)
    L = int(L)
    if L < 4:
        raise ValueError(f"L must be >= 4 (marker + filler + controlled + "
                         f"terminator), got {L}")
    marker = rng.choice(MARKER_ALPHABET)
    filler = _rand_bytes(rng, L - 3)
    controlled = np.array([CONTROLLED_BY_MARKER[int(marker)]], dtype=DTYPE)
    terminator = np.array([TERMINATOR], dtype=DTYPE)
    seq = np.concatenate([np.array([marker], dtype=DTYPE), filler,
                          controlled, terminator])
    info = {
        "task": "long_range_language", "L": L,
        "gap_distribution": ("deterministic per sample: one sequence of length "
                             f"L = {L}; dependency from position 0 to position "
                             "L-2 (non-local)"),
        "marker": int(marker),
        "critical_position": L - 2,
    }
    return seq, controlled.copy(), info


# --------------------------------------------------------------------------
# Chance-level helpers
# --------------------------------------------------------------------------
def chance_ce(task="long_range_language", L=512):
    """Analytic chance-level cross-entropy, nats per scored byte, under the
    generating distribution (what a perfect distribution-model scores):
      - A/B/C: ln(256) per payload byte (uniform byte guess);
      - D:      ((L-3)*ln 256 + ln 2 + 0)/(L-1) over scored positions
        1..L-1 (L-3 uniform fillers, 1 binary controlled byte, 1 fixed
        terminator that contributes 0 nats to the generator's own loss).
    """
    if task in ("delayed_copy", "associative_copy", "selective_copy"):
        return math.log(VOCAB)
    if task == "long_range_language":
        L = int(L)
        if L < 4:
            raise ValueError(f"L must be >= 4, got {L}")
        return ((L - 3) * math.log(VOCAB) + math.log(len(CONTROLLED_ALPHABET))
                ) / (L - 1)
    raise ValueError(f"unknown task {task!r}")


# --------------------------------------------------------------------------
# Batch API
# --------------------------------------------------------------------------
_TASK_FUNCS = {
    "delayed_copy": delayed_copy,
    "associative_copy": associative_copy,
    "selective_copy": selective_copy,
    "long_range_language": long_range_language,
}


def _task_name(task):
    if callable(task):
        return task.__name__
    if isinstance(task, str):
        return task
    raise TypeError(f"task must be one of {TASKS} or a generator callable; "
                    f"got {task!r}")


def _chance_level(name, d):
    cl = dict(CHANCE_LEVELS[name])
    if name == "long_range_language":
        cl["chance_ce_nats_per_byte"] = chance_ce(name, L=d)
    else:
        cl["query_key_repeat"] = {
            "per_byte_without_memory": 1.0 / VOCAB,
            "exact_without_memory": (1.0 / VOCAB) ** KEY_LEN,
            "exact_with_perfect_retention": 1.0,
        }
    return cl


def get_batch(task, d, n, rng, **kwargs):
    """Draw a batch of n samples from one generator.

    task : str in TASKS, or a generator callable (then d/n/rng are forwarded).
    d    : max/target gap; for long_range_language it is the sequence length L.
    n    : number of samples (>= 1).
    rng  : int seed, np.random.Generator, or None. A master seed is derived
           from rng and recorded in meta["seed"]; two identical (task, d, n,
           seed) calls are byte-identical, and meta["seed"] can be replayed.

    Returns dict(seq, target, meta):
      seq    : list of n numpy uint8 arrays, one per sample (variable length
               for selective_copy because the per-sample gap is random; use
               meta["lengths"] / meta["gaps"]).
      target : (n, k) uint8 array -- k=4 payload bytes for A/B/C, k=1
               controlled byte for D.
      meta   : dict carrying every PLAN-2026-10-06 section-10 field:
               cue_type, gap_distribution, distractor_type, state_reset_policy,
               seed, sample_count, chance_level (+ task-specific extras).
    """
    name = _task_name(task)
    if name not in _TASK_FUNCS:
        raise ValueError(f"unknown task {name!r}; choose from {TASKS}")
    d = int(d)
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if name in ("associative_copy", "selective_copy"):
        d = _check_d(d, name)
    if name == "long_range_language" and d < 4:
        raise ValueError(f"long_range_language: L (= d) must be >= 4, got {d}")

    master = _as_generator(rng)
    seed = int(master.integers(0, 2 ** 32))
    local = np.random.default_rng(seed)

    if any(k in kwargs for k in ("seed", "rng")):
        raise TypeError("get_batch: pass the seed/rng as the `rng` argument; "
                        "kwargs are task options only (e.g. n_pairs, gap_mode)")

    fn = _TASK_FUNCS[name]
    seqs, targets, infos = [], [], []
    for _ in range(n):
        s, t, info = fn(d, seed=local, **kwargs)
        seqs.append(s)
        targets.append(t)
        infos.append(info)

    meta = {
        "task": name,
        "d": d,
        "seed": seed,
        "sample_count": n,
        "vocab_size": VOCAB,
        "dtype": str(np.dtype(DTYPE)),
        "cue_type": _CUE_TYPE[name],
        "gap_distribution": infos[0]["gap_distribution"],
        "distractor_type": (
            "uniform random noise bytes (vocab 256) + "
            f"{infos[0].get('n_dist', '1..n_pairs')} distractor key->payload "
            "pairs at random positions" if name == "selective_copy" else
            "uniform random bytes (vocab 256), i.i.d. per position"),
        "state_reset_policy": STATE_RESET_POLICY,
        "chance_level": _chance_level(name, d),
        "lengths": np.array([len(s) for s in seqs], dtype=np.int64),
        "gaps": np.array([inf.get("gap", d) for inf in infos], dtype=np.int64),
    }
    if name == "selective_copy":
        meta["distractor_counts"] = np.array([inf["n_dist"] for inf in infos],
                                             dtype=np.int64)
        meta["pair_positions"] = [inf["pair_positions"] for inf in infos]
        meta["distractor_pairs"] = [inf["distractor_pairs"] for inf in infos]
    if name == "long_range_language":
        meta["markers"] = np.array([inf["marker"] for inf in infos],
                                   dtype=DTYPE)
        meta["critical_position"] = d - 2

    return {"seq": seqs, "target": np.stack(targets).astype(DTYPE),
            "meta": meta}