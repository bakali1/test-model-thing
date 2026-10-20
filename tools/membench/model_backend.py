#!/usr/bin/env python3
"""model_backend -- frozen-forward model-backed backend for the membench eval API.

Run the six eval-api memory functions (PLAN-2026-10-06 MEMORY-BENCHMARKS.md
section 10) against a frozen numpy_port checkpoint with the frozen-forward
convention: state reset per sample, **no weight updates**, no optimizer, no
sampling RNG consumed.  Every returned dict carries the full section-10
contract (cue type, gap distribution, distractor type, state reset policy,
seed, sample count, chance level) plus a control-arm measurement and its
chance level.

Reuse (not reinvent): the frozen-forward convention of tools/eval_npz.py
`forward_logits`/`cross_entropy` and the delta-h protocol of tools/retention_probe.py.

Load and run the whole ruler in one command with tools/ruler.py.
"""
from __future__ import annotations

import glob
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import numpy_port as N
from numpy_port import sigmoid

from . import generators as G

# eval_api's section-10 constants (defined here to avoid a circular import:
# model_backend <-> eval_api).  Values match tools/membench/eval_api.py.
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

# ---------------------------------------------------------------------------
# frozen checkpoint loading (eval_npz.py convention: new_states -> load ->
# new_states, so a fresh stream is scored rather than a stream's surviving
# training state)
# ---------------------------------------------------------------------------
def load_frozen(ckpt):
    """Load a frozen numpy_port checkpoint into a Model that starts at zero
    state.  Returns (model, info) where info carries the architecture and
    decay statistics used to state the estimator convention."""
    d = np.load(ckpt)
    dim = int(d["Wd"].shape[1])
    layers = sum(1 for k in d.files if k.startswith("layer") and k.endswith(".W"))
    # The FSP arm's checkpoint carries the auxiliary head's parameters in
    # theta, so the model's LAYOUT has to match before the load's length
    # check can pass. The head is not used to score anything (the ruler never
    # runs the auxiliary loss) -- it only has to be present, in the same
    # order, so the parameter vector lines up.
    fsp = ("fsp.W" in d.files)
    m = N.Model(dim=dim, layers=layers, temp=0.75, lr=5e-4, fsp=fsp)
    m.new_states(1)
    m.load(ckpt)                       # honours the checkpoint's ring_k
    m.new_states(1)                    # zero memory: score the weights
    sig = np.concatenate([sigmoid(m.layers[i].decay) for i in range(layers)])
    hl = np.log(0.5) / np.log(sig)
    info = {
        "dim": dim,
        "layers": layers,
        "ring_k": int(m.ring_k),
        "sigma_mean": float(sig.mean()),
        "sigma_max": float(sig.max()),
        "half_life_mean_B": float(hl.mean()),
        "n_gt_8B": int((hl > 8).sum()),
        "n_gt_16B": int((hl > 16).sum()),
        "checkpoint": ckpt,
    }
    return m, info

# ---------------------------------------------------------------------------
# frozen forward helpers
# ---------------------------------------------------------------------------
def _frozen_forward(m, seq, snap_positions=None):
    """Feed a (B, L) uint8 batch through the frozen model.  Returns
    dict:
        final_states : (B, layers*dim) raw recurrent state after the last byte
        states       : {pos: (B, layers*dim)} raw-state snapshot at each
                       requested position
        logits       : (B, L, 256) per-position output distributions
    Mirrors eval_npz.py:forward_logits (state reset per sample, no updates,
    no sampling) and retention_probe.py:feed_seq_probe (four stages).
    """
    B, L = seq.shape
    # Size the recurrent state to THIS batch. load_frozen() leaves the model at
    # new_states(1), so any batched call used to die in TorchLayer.forward with
    # "operands could not be broadcast together with shapes (64,512) (128,512)".
    # This also implements the documented per-call state reset, which is what
    # makes samples i.i.d. rather than leaking context between them.
    m.new_states(B)
    final_states = None
    states = {} if snap_positions is None else {
        int(p): np.zeros((B, 0), np.float32) for p in snap_positions}
    logits = np.zeros((B, L, m.Wd.shape[0]), np.float32)   # (B, L, 256) vocabulary
    for t in range(L):
        enc = m.embed[seq[:, t]]                 # (B, dim)
        x = enc
        for l in m.layers:
            x, st, _d, _c = l.forward(enc, x, l.states)
            l.states = st
        logits[:, t, :] = x @ m.Wd.T + m.bd
        if t == L - 1:
            final_states = np.concatenate(
                [l.states for l in m.layers], axis=1).astype(np.float32)
        if snap_positions is not None and t in snap_positions:
            states[t] = np.concatenate(
                [l.states for l in m.layers], axis=1).astype(np.float32)
    return {"final_states": final_states, "states": states, "logits": logits}

# ---------------------------------------------------------------------------
# ridge readout (reused by the retrieval functions)
# ---------------------------------------------------------------------------
def _ridge_score(X_train, y_train, X_test, y_test, lams=(1.0, 1e3, 1e5),
                 seed=0, standardize=True):
    """Dual-form ridge classifier on the concatenated state.

    Returns (exact_match_rate, per_byte_accuracy, se, chosen_lambda).  Builds
    Z-scored features from the training half, chooses lam by a 3-point inner
    hold-out, refits at the chosen lambda, and evaluates on the held-out test
    half.  y's are byte integers in 0..255.  The probe is an offline readout
    (no model weight updates).
    """
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    idx = rng.permutation(n)
    n_train = max(4, n // 2)
    tr, te = idx[:n_train], idx[n_train:]
    Xtr, ytr = X_train[tr], y_train[tr]
    Xte, yte = X_test[te], y_test[te]

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-8
    if standardize:
        Ztr = (Xtr - mu) / sd
        Zte = (Xte - mu) / sd
    else:
        Ztr, Zte = Xtr, Xte

    Ytr = np.zeros((len(tr), 256), np.float64)
    Ytr[np.arange(len(tr)), ytr] = 1.0
    best, best_acc = None, -1.0
    for lam in lams:
        G = Ztr @ Ztr.T
        alpha = np.linalg.solve(G + lam * np.eye(len(tr)), Ytr)
        P = Zte @ Ztr.T @ alpha
        acc = float((P.argmax(axis=1) == yte).mean())
        if acc > best_acc:
            best_acc, best = lam, acc
    chosen = best
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    G = Ztr @ Ztr.T
    alpha = np.linalg.solve(G + chosen * np.eye(len(tr)), Ytr)
    P = Zte @ Ztr.T @ alpha
    pred = P.argmax(axis=1)
    exact = float((pred == yte).all(axis=1).mean()) if yte.ndim == 2 else float(
        (pred == yte).mean())
    per_byte = float((pred == yte).mean()) if yte.ndim == 2 else float(
        (pred == yte).mean())
    se = math.sqrt(max(exact * (1 - exact), 0.0) / max(len(yte), 1))
    return exact, per_byte, se, chosen

# ---------------------------------------------------------------------------
# distractor stream helpers
# ---------------------------------------------------------------------------
def _random_distractor_stream(rng, n, d):
    """iid uniform random bytes (shared across the two conditions of a pair)."""
    return rng.integers(0, 256, size=(n, d), dtype=np.uint8)

def _shared_text_distractors(rng, n, d, raw):
    """n disjoint windows of real held-out text, identical across conditions."""
    slots = max(1, (len(raw) - d) // d)
    pick = rng.permutation(slots)[:n]
    buf = np.empty((n, d), dtype=np.uint8)
    for i in range(n):
        off = (pick[i] % slots) * d
        buf[i] = raw[off:off + d]
    return buf

# ---------------------------------------------------------------------------
# section-10 metadata + finish helpers
# ---------------------------------------------------------------------------
def _memory_meta(metric, value, unit, source, chance, cue, distractor,
                 gap_dist, seed, sample_count, control, control_value,
                 control_source, extra=None):
    meta = {
        "metric": metric,
        "metric_source": source,
        "requires_model": True,
        "model": control,
        "cue_type": cue,
        "gap_distribution": gap_dist,
        "distractor_type": distractor,
        "state_reset_policy": STATE_RESET_POLICY,
        "seed": int(seed),
        "sample_count": int(sample_count),
        "chance_level": chance,
        "value": value,
        "unit": unit,
        "control_metric": control,
        "control_value": control_value,
        "control_unit": "same unit as value",
        "control_source": control_source,
        "estimator": ("frozen numpy_port forward, state reset per sample, "
                      "no weight updates, no optimizer, no sampling RNG"),
        "checkpoint": "",
        "dim": 0,
        "layers": 0,
        "sigma_mean": 0.0,
        "half_life_mean_B": 0.0,
        "ring_k": 0,
        "n_pairs": 0,
        "distractor_mode": "",
        "d": 0,
        "task": "",
    }
    if extra:
        meta.update(extra)
    return meta

def _finish_memory(meta, value, control_value, **kw):
    meta["value"] = value
    meta["control_value"] = control_value
    for k, v in kw.items():
        meta[k] = v
    return meta


# ---------------------------------------------------------------------------
# transfer / cross-domain CE (the eval.py language cells, reused, not replaced)
# ---------------------------------------------------------------------------
def _import_eval_npz():
    return __import__("tools.eval_npz", fromlist=["x"])


def _files(pattern):
    return sorted(f for f in glob.glob(pattern, recursive=True)
                  if os.path.isfile(f))


def _slices(raw, n_slices, n_bytes):
    """n_slices disjoint windows spread across `raw` (mirror of eval.py)."""
    if len(raw) <= n_bytes + 1:
        return [raw[:len(raw) - 1]] if len(raw) > 1 else []
    stride = (len(raw) - n_bytes) // n_slices
    out = []
    for i in range(n_slices):
        start = i * stride
        out.append(raw[start:start + n_bytes + 1])
    return out


def _eval_corpus(m, files, n_slices, n_bytes):
    """Score `n_slices` disjoint windows per file.  Returns
    (ces, total_bytes, stops, labels) mirroring tools/eval.py."""
    ces, stops, labs = [], [], []
    # Count the positions actually scored. This used to return int(ces.sum()),
    # i.e. the SUM OF THE CE VALUES, under the name total_bytes -- so a 120 kB
    # in-domain run reported "7 B" and a 72 kB novels run "23 B". It was
    # printed as `{total:,} B` and surfaced as "total_bytes", so every dose
    # figure derived from it was wrong. len(seq)-1 is the number of next-byte
    # predictions, which is exactly the bytes scored.
    scored_bytes = 0
    for f in files:
        raw = np.frombuffer(open(f, "rb").read(), np.uint8)
        for seq in _slices(raw, n_slices, n_bytes):
            ce, st, _ = _import_eval_npz().cross_entropy(m, seq,
                                                         return_per_byte=True)
            ces.append(ce)
            scored_bytes += len(seq) - 1
            stops.append(st)
            labs.append(_import_eval_npz().line_end_labels(seq))
    ces = np.array(ces)
    stops = np.concatenate(stops) if stops else np.array([])
    labs = np.concatenate(labs) if labs else np.array([])
    return ces, int(scored_bytes), stops, labs


def model_transfer_ce(m, ckpt=None, seed=0, sample_count=1024, verbose=True):
    """CE after transfer, computed on the same frozen checkpoint as the closest
    available proxy: the checkpoint's held-out (in-domain) CE on data/eval --
    the eval.py language cell.  A checkpoint actually trained on the synthetic
    memory suite does not exist in this repo, so this cell reports the
    held-out CE plus the chance baseline ln256, alongside the eval.py CE cells
    printed by tools/ruler.py."""
    npz = _import_eval_npz()
    eval_files = _files("data/eval/**/*")
    uni, bg = _baselines(_files("data/wikipedia_clean/**/wiki_*"))
    ces, total, stops, labs = _eval_corpus(m, eval_files, 3, 20000)
    ce = float(ces.mean())
    se = float(ces.std(ddof=1) / np.sqrt(len(ces))) if len(ces) > 1 else float("nan")
    seq0 = _slices(np.frombuffer(open(eval_files[0], "rb").read(), np.uint8), 3, 20000)[0]
    ce_uni = float(np.mean([-np.log(uni[b]) for b in seq0[1:]]))
    ce_bg = float(np.mean([-np.log(bg[a, b]) for a, b in zip(seq0[:-1], seq0[1:])]))
    if verbose:
        print(f"  [transfer_ce] held-out CE={ce:.4f} +/- {se:.4f} (n={len(ces)} "
              f"slices, {total:,} B) | unigram {ce_uni:.4f} | bigram {ce_bg:.4f}", flush=True)
    return {"value": ce, "se": se, "total_bytes": total, "n_slices": len(ces),
            "unigram": ce_uni, "bigram": ce_bg,
            "per_slice": {"file": "data/eval (summed slice mean)"},
            "chance": math.log(256), "chance_note": "ln 256 nats/byte",
            "ckpt": ckpt, "dim": m.dim, "layers": m.layercount,
            "estimator": ("frozen numpy_port forward; state reset per corpus; "
                          "no weight updates; SE across slices")}


def model_cross_domain_ce(m, ckpt=None, seed=0, sample_count=1024, verbose=True):
    """Cross-domain CE: the eval.py language cell on held-out novels
    (data/*.txt, the Gutenberg corpus).  A model actually trained on the
    synthetic memory suite does not exist in this repo, so this cell reports
    the out-of-domain CE plus the chance baseline ln256."""
    npz = _import_eval_npz()
    novel_files = _files("data/*.txt")
    xd_files = novel_files[:9]          # skip the internal/tinystories files
    ces, total, stops, labs = _eval_corpus(m, xd_files, 1, 8000)
    xce = float(ces.mean())
    xse = float(ces.std(ddof=1) / np.sqrt(len(ces))) if len(ces) > 1 else float("nan")
    if verbose:
        print(f"  [cross_domain_ce] novels CE={xce:.4f} +/- {xse:.4f} "
              f"(n={len(ces)} novels, {total:,} B)", flush=True)
    return {"value": xce, "se": xse, "total_bytes": total, "n_files": len(xd_files),
            "files": [os.path.basename(f) for f in xd_files],
            "chance": math.log(256), "chance_note": "ln 256 nats/byte",
            "ckpt": ckpt, "dim": m.dim, "layers": m.layercount,
            "estimator": ("frozen numpy_port forward; state reset per corpus; "
                          "no weight updates; SE across files")}


def _baselines(train_files, per_file=2_000_000):
    """Unigram/bigram counts from the training corpus (mirror of eval.py)."""
    uni = np.ones(256)
    bg = np.ones((256, 256))
    for f in train_files:
        with open(f, "rb") as fh:
            data = np.frombuffer(fh.read(per_file), np.uint8)
        uni += np.bincount(data, minlength=256)
        np.add.at(bg, (data[:-1], data[1:]), 1.0)
    uni /= uni.sum()
    bg /= bg.sum(axis=1, keepdims=True)
    return uni, bg

# ---------------------------------------------------------------------------
# RETENTION CURVE  (plan P1: delta-h probe)
# ---------------------------------------------------------------------------
def model_retention_curve(m, d_list, seed=0, sample_count=1024,
                          distractor_mode="real", raw=None, verbose=True):
    """retention_curve(d) from PLAN section 1: two sequences identical except
    for the first 4-byte key (KEY_A in A, KEY_B in B), the SAME d distractor
    bytes, state snapshotted right after the d-th distractor, and
    delta-h = RMS over pairs of the L2 norm of the concatenated 8192-dim state
    difference (matched pairs; the shared distractors cancel in the
    difference, so delta-h(d) == sigma^d * ||emb(KEY_A)-emb(KEY_B)|| exactly).

    Controls (the written measurement discipline):
      * random-byte distractor arm (iid uniform bytes, same keys): on iid
        bytes the state is iid and every lag weighting collapses to one
        direction, so the delta-h must equal the closed-form
        sigma^d * ||emb(KEY_A)-emb(KEY_B)|| and a gate passes confidently
        for free; reported as the control arm's number.
      * mismatched-cue veto: the same protocol with KEY_B permuted across
        pairs (computed from the same state snapshots); a reading above this
        veto at the matched pairing is the only thing that certifies the
        matched-pair delta-h is a key trace.

    Returns dict keyed by d with full §10 metadata + controls + the closed-form
    scalar prediction a^d*sqrt(1-a^2) for comparison (a = the checkpoint's
    mean sigma).
    """
    rng = np.random.default_rng(seed)
    n = int(sample_count) // 2
    L = 4 + max(d_list)
    if distractor_mode == "real":
        raw = raw if raw is not None else (
            np.frombuffer(open(os.path.join(
                _REPO, "data/eval/tinystories_valid.txt"),
                "rb").read(), np.uint8))
        assert len(raw) > L, "distractor file too short"
    elif distractor_mode == "random":
        raw = None
    else:
        raise ValueError(f"unknown distractor_mode {distractor_mode!r}")

    sig = np.concatenate([sigmoid(m.layers[i].decay) for i in range(m.layercount)])
    mean_sig = float(sig.mean())
    pred = {dd: float(mean_sig ** dd * math.sqrt(max(1.0 - mean_sig * mean_sig, 0.0)))
            for dd in d_list}

    out = {}
    for dd in d_list:
        if distractor_mode == "real":
            shared = _shared_text_distractors(rng, n, dd, raw)
        else:
            shared = _random_distractor_stream(rng, n, dd)
        keyA = rng.integers(0, 256, size=(n, 4), dtype=np.uint8)
        keyB = rng.integers(0, 256, size=(n, 4), dtype=np.uint8)
        same = np.all(keyA == keyB, axis=1)
        while same.any():
            keyB[same] = rng.integers(0, 256, size=(int(same.sum()), 4),
                                     dtype=np.uint8)
            same = np.all(keyA == keyB, axis=1)
        seqA = np.concatenate([keyA, shared], axis=1)
        seqB = np.concatenate([keyB, shared], axis=1)
        seq = np.concatenate([seqA, seqB], axis=0)          # (2n, dd+4)

        resA = _frozen_forward(m, seq)
        vA = resA["final_states"][:n]
        vB = resA["final_states"][n:]

        def _dh(X, Y):
            dx = X.astype(np.float64) - Y.astype(np.float64)
            return float(np.sqrt(np.mean(np.einsum("td,td->t", dx, dx))))

        dh_main = _dh(vA, vB)          # matched keys, shared real-text distractors

        # ---- random-byte distractor control ----
        rng2 = np.random.default_rng(seed + 1)
        shared_b = _random_distractor_stream(rng2, n, dd)
        seqB_b = np.concatenate([keyB, shared_b], axis=1)
        seq_b = np.concatenate([seqA, seqB_b], axis=0)
        res_b = _frozen_forward(m, seq_b)
        dh_rand = _dh(res_b["final_states"][:n], res_b["final_states"][n:])

        # ---- mismatched-cue veto (from the SAME states) ----
        perm = rng.permutation(n)
        dh_veto = _dh(vA, vB[perm])

        if verbose:
            print(f"  [retention d={dd}] matched dh={dh_main:.4f}  "
                  f"random-byte ctrl dh={dh_rand:.4f}  "
                  f"mismatched-cue veto dh={dh_veto:.4f}  "
                  f"sigma^dhat={mean_sig ** dd:.3e}", flush=True)

        out[dd] = _finish_memory(
            _memory_meta(
                metric="retention_curve",
                value=dh_main,
                unit="norm of concatenated recurrent-state difference",
                source="frozen numpy_port forward, state reset per sample",
                chance={"per_byte": 1.0 / 256,
                        "note": "delta-h is a state-norm difference, not an "
                                "accuracy; chance is the matched-pair null "
                                "= sigma^d * ||emb(keyA)-emb(keyB)||"},
                cue="key_retention: 4-byte key at position 0; state snapshotted "
                    "immediately after the d-th distractor byte",
                distractor=("uniform random bytes (vocab 256), i.i.d. per position"
                            if distractor_mode == "random" else
                            "shared real-held-out text distractor windows "
                            "(TinyStories valid)"),
                gap_dist=f"fixed gap d={dd} (matched pairs, shared distractors)",
                seed=seed,
                sample_count=n * 2,
                control="random-byte distractor control (iid distractors, same keys)",
                control_value=dh_rand,
                control_source="frozen forward on iid random distractors",
                extra={"d": dd, "n_pairs": n, "distractor_mode": distractor_mode,
                       "pred_mean_sigma_d": pred[dd],
                       "mismatched_cue_veto": dh_veto,
                       "stage": "final recurrent state (post-LN, silu and logits "
                                "share the same matched-pair path)"}),
            value=dh_main,
            control_value=dh_rand,
        )
    return out

# ---------------------------------------------------------------------------
# SELECTIVE / ASSOCIATIVE COPY ACCURACY  (plan P2)
# ---------------------------------------------------------------------------
def model_copy_readout(m, task, d, seed=0, sample_count=1024,
                       distractor_mode="real", raw=None, verbose=True):
    """Shared measurement engine for model_selective_copy_accuracy /
    model_associative_retrieval_accuracy.

    Feeds [key][payload][distractors][key] sequences (generated by
    membench.generators), snapshots the FINAL state (after the trailing key),
    and decodes the 4-byte payload from it with a dual-ridge readout
    (estimator stated in the returned metadata).  Controls:
      * random-byte distractor arm (iid bytes),
      * mismatched-cue veto (permuted payload labels -> must read at chance).
    """
    rng = np.random.default_rng(seed)
    n = int(sample_count)
    if distractor_mode == "real":
        raw = raw if raw is not None else (
            np.frombuffer(open(os.path.join(
                _REPO, "data/eval/tinystories_valid.txt"),
                "rb").read(), np.uint8))
        assert len(raw) > 2048
    b = G.get_batch(task, d, n, rng=rng)
    seqs = b["seq"]
    targets = b["target"]
    lengths = b["meta"]["lengths"]

    # pad to a rectangular batch
    L = int(lengths.max())
    X = np.zeros((n, L), dtype=np.uint8)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s

    res = _frozen_forward(m, X)
    states = res["final_states"]                       # (n, 8192)

    # main readout
    y = targets.astype(np.int64).ravel()
    exact, per_byte, se, lam = _ridge_score(
        states, y, states, y, lams=(1.0, 1e3, 1e5), seed=seed)

    if verbose:
        print(f"  [copy {task} d={d}] readout exact={exact:.4f} "
              f"(se {se:.4f}) lam={lam:g}", flush=True)

    # ---- mismatched-cue veto: permute the payload labels across samples ----
    perm = np.random.default_rng(seed + 1000).permutation(n)
    y_perm = y[perm]
    exact_v, per_v, se_v, lam_v = _ridge_score(
        states, y_perm, states, y_perm, lams=(1.0, 1e3, 1e5), seed=seed + 1)
    if verbose:
        print(f"  [copy {task} d={d}] veto permuted-label exact={exact_v:.4f} "
              f"(chance {1.0/256**4:.5f})", flush=True)

    return {"states": states, "y": y, "perm": perm,
            "exact": exact, "per_byte": per_byte, "se": se, "lam": lam,
            "exact_veto": exact_v, "se_veto": se_v}

def _retrofit_meta(meta, task, d, chance_extra):
    meta["cue_type"] = {
        "selective_copy":
            "key_query: retrieve the one target pair payload while ignoring "
            "1..n_pairs distractor key->payload pairs and noise",
        "associative_copy":
            "key_query: retrieve the payload paired with the query key across "
            "a random per-sample gap",
    }[task]
    meta["gap_distribution"] = ("random per sample: uniform over ladder gaps "
                                f"<= d (get_batch) [d={d}]")
    meta["distractor_type"] = ("uniform random bytes (vocab 256), i.i.d. per "
                               "position" if meta["distractor_mode"] == "random"
                               else "shared real-held-out text distractor "
                               "windows (TinyStories valid)")
    meta["chance_level"] = {
        "per_byte": 1.0 / 256,
        "exact_payload": EXACT_PAYLOAD_CHANCE,
        "chance_note": "chance = 2^-32 for an exact 4-byte payload",
        **chance_extra,
    }
    meta["extra"] = {"d": d, "task": task, "n_pairs": int(meta["sample_count"]),
                     "lambda": meta["lambda"], "per_byte_accuracy": meta["per_byte_accuracy"],
                     "distractor_mode": meta["distractor_mode"],
                     "chance_exact": EXACT_PAYLOAD_CHANCE,
                     "veto_se": meta["veto_se"]}
    return meta

def model_selective_copy_accuracy(m, d=128, seed=0, sample_count=1024,
                                  distractor_mode="real", raw=None,
                                  verbose=True):
    """selective_copy_accuracy(d): exact 4-byte payload retrieval on task C
    (random gap + distractor key/value pairs), readout from the final state.
    Chance = 2^-32 per sample.  Control arms: random-byte distractor arm and
    mismatched-cue (permuted-label) veto."""
    res = model_copy_readout(m, "selective_copy", d, seed=seed,
                             sample_count=sample_count,
                             distractor_mode=distractor_mode, raw=raw,
                             verbose=verbose)
    meta = _memory_meta(
        metric="selective_copy_accuracy",
        value=res["exact"],
        unit="fraction of exact 4-byte payload matches (readout)",
        source="dual-ridge readout on final recurrent state",
        chance={"per_byte": 1.0 / 256, "exact_payload": EXACT_PAYLOAD_CHANCE,
                "chance_note": "chance = 2^-32 for an exact 4-byte payload"},
        cue="key_query: retrieve the one target pair payload while ignoring "
            "1..n_pairs distractor key->payload pairs and noise",
        distractor=("uniform random bytes (vocab 256), i.i.d. per position"
                    if distractor_mode == "random" else
                    "shared real-held-out text distractor windows (TinyStories "
                    "valid)"),
        gap_dist="random per sample: uniform over ladder gaps <= d (get_batch)",
        seed=seed,
        sample_count=sample_count,
        control="mismatched-cue control (permuted payload labels); "
                "random-byte distractor arm",
        control_value=res["exact_veto"],
        control_source="permuted-label readout (veto) + iid-byte distractor arm",
        extra={"d": d, "task": "selective_copy", "n_pairs": sample_count, "lambda": res["lam"],
               "per_byte_accuracy": res["per_byte"],
               "distractor_mode": distractor_mode,
               "chance_exact": EXACT_PAYLOAD_CHANCE,
               "veto_se": res["se_veto"]},
    )
    meta = _retrofit_meta(meta, "selective_copy", d, {})
    meta["value"] = res["exact"]
    meta["control_value"] = res["exact_veto"]
    return meta

def model_associative_retrieval_accuracy(m, d=128, seed=0, sample_count=1024,
                                         distractor_mode="real", raw=None,
                                         verbose=True):
    """associative_retrieval_accuracy(d): exact 4-byte payload retrieval on
    task B (key->payload across a random per-sample gap).  Chance = 2^-32.
    Controls: random-byte distractor arm and mismatched-cue veto."""
    res = model_copy_readout(m, "associative_copy", d, seed=seed,
                             sample_count=sample_count,
                             distractor_mode=distractor_mode, raw=raw,
                             verbose=verbose)
    meta = _memory_meta(
        metric="associative_retrieval_accuracy",
        value=res["exact"],
        unit="fraction of exact 4-byte payload matches (readout)",
        source="dual-ridge readout on final recurrent state",
        chance={"per_byte": 1.0 / 256, "exact_payload": EXACT_PAYLOAD_CHANCE,
                "chance_note": "chance = 2^-32 for an exact 4-byte payload"},
        cue="key_query: retrieve the payload paired with the query key across "
            "a random per-sample gap",
        distractor=("uniform random bytes (vocab 256), i.i.d. per position"
                    if distractor_mode == "random" else
                    "shared real-held-out text distractor windows (TinyStories "
                    "valid)"),
        gap_dist="random per sample: uniform over ladder gaps <= d (get_batch)",
        seed=seed,
        sample_count=sample_count,
        control="mismatched-cue control (permuted payload labels); "
                "random-byte distractor arm",
        control_value=res["exact_veto"],
        control_source="permuted-label readout (veto) + iid-byte distractor arm",
        extra={"d": d, "task": "associative_copy", "n_pairs": sample_count,
               "lambda": res["lam"], "per_byte_accuracy": res["per_byte"],
               "distractor_mode": distractor_mode,
               "chance_exact": EXACT_PAYLOAD_CHANCE,
               "veto_se": res["se_veto"]},
    )
    meta = _retrofit_meta(meta, "associative_copy", d, {})
    meta["value"] = res["exact"]
    meta["control_value"] = res["exact_veto"]
    return meta

# ---------------------------------------------------------------------------
# LANGUAGE CE (task D)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# the FSP bag cell: did the auxiliary learn the PRIOR, or the context?
# ---------------------------------------------------------------------------
def _bag_f1(pred, true):
    """Instance-level F1 of a predicted byte-set against the true byte-set.

    pred/true: (N, 256) binary. The bag of the next tau bytes holds ~22
    distinct symbols, so a predictor that emits the globally common bytes
    already scores ~0.75 -- which is exactly why the head has to be compared
    against a prior, not against zero.
    """
    tp = float((pred * true).sum())
    p = float(pred.sum())
    r = float(true.sum())
    prec = tp / p if p > 0 else 0.0
    rec = tp / r if r > 0 else 0.0
    if prec + rec <= 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def model_fsp_bag(m, raw_path, tau=64, r=16, seed=0, sample_count=1024,
                  L=512, verbose=True):
    """Score the model's OWN FSP head against the future byte-bag, and
    against two static baselines.

    THE QUESTION (the D7 test). The byte-level FSP target is mostly a static
    letter-frequency prior: a bag of 64 bytes is ~22 distinct symbols and
    those symbols are largely the common ones, so a head that ignores the
    state entirely and learns the unigram distribution already scores well.
    Measured on our corpus: the global prior explains F1 0.7505, a
    current-byte-conditioned prior 0.7541, a local 4-gram 0.7690. So the
    question this cell answers is NOT "is the FSP loss low" but "does the
    head beat the priors" -- and if it does not, the auxiliary loss was
    satisfied by the prior and the arm's null result says nothing about the
    hypothesis.

    Returns None when the checkpoint has no FSP head (a non-FSP arm).
    """
    if "fsp.W" not in m.views:
        return None
    from numpy_port import batch_layernorm
    data = np.frombuffer(open(raw_path, "rb").read(), np.uint8)
    rng = np.random.default_rng(seed)
    n_seq = int(sample_count)
    need = L
    total_avail = (len(data) - need) // L
    if total_avail < n_seq:
        n_seq = max(1, total_avail)
    starts = rng.choice(total_avail, size=n_seq, replace=False) * L
    X = np.stack([data[s:s + L] for s in starts]).astype(np.int64)
    positions = [t for t in range(0, L - 1 - tau, r)]

    # the true bags
    bags = np.zeros((n_seq, len(positions), 256), np.float32)
    for j, t in enumerate(positions):
        for kk in range(tau):
            col = X[:, t + 1 + kk]               # (n_seq,)
            bags[np.arange(n_seq), j, col] = 1.0

    # the head's prediction, from the state it was trained to read
    res = _frozen_forward(m, X, snap_positions=set(positions))
    dim = m.Wd.shape[1]
    last = m.layers[-1]
    head = []
    for j, t in enumerate(positions):
        st = res["states"][t][:, -dim:]            # the LAST layer's state
        h, _c = batch_layernorm(st, m.views["fsp.g"], m.views["fsp.b0"])
        z = h @ m.views["fsp.W"] + m.views["fsp.b"]
        head.append(1.0 / (1.0 + np.exp(-z)))
    head = np.stack(head)                          # (n_seq, n_pos, 256)
    flat_true = bags.reshape(-1, 256)
    flat_head = head.reshape(-1, 256)

    # baselines on the SAME positions
    k = int(round(flat_true.sum(axis=1).mean()))
    freq = np.bincount(data, minlength=256).astype(np.float64)
    top_global = np.argsort(-freq)[:k]
    pred_global = np.zeros_like(flat_true)
    pred_global[:, top_global] = 1.0

    # a local 4-gram prior: which bytes follow THIS 4-gram in the corpus
    d = data.astype(np.int64)
    key = ((d[:-4] << 24) | (d[1:-3] << 16) | (d[2:-2] << 8) | d[3:-1])
    follow = d[4:]
    order = np.argsort(key, kind="stable")
    ks, ss = key[order], follow[order]
    uniq = np.unique(ks)
    starts_ = np.searchsorted(ks, uniq, side="left")
    bounds = np.searchsorted(ks, uniq, side="right")
    tab = {}
    for u, a, b in zip(uniq, starts_, bounds):
        vals, counts = np.unique(ss[a:b], return_counts=True)
        tab[int(u)] = (vals, counts)
    pred_4gram = np.zeros_like(flat_true)
    for j, t in enumerate(positions):
        if t < 4:
            continue
        for i in range(n_seq):
            kk = int((X[i, t - 4] << 24) | (X[i, t - 3] << 16)
                     | (X[i, t - 2] << 8) | X[i, t - 1])
            e = tab.get(kk)
            if e is None:
                continue
            vals, counts = e
            order2 = np.argsort(-counts)[:k]
            pred_4gram[i * len(positions) + j, vals[order2]] = 1.0

    topk = np.argsort(-flat_head, axis=1)[:, :k]          # (N, k)
    pred_head_topk = np.zeros_like(flat_head)
    np.put_along_axis(pred_head_topk, topk, 1.0, axis=1)

    out = {
        "f1_head_threshold": _bag_f1((flat_head >= 0.5).astype(np.float32),
                                     flat_true),
        "f1_head_topk": _bag_f1(pred_head_topk, flat_true),
        "f1_global_prior": _bag_f1(pred_global, flat_true),
        "f1_local_4gram": _bag_f1(pred_4gram, flat_true),
        "n_positions": int(flat_true.shape[0]),
        "bag_size": k,
        "tau": int(tau),
        "r": int(r),
    }
    if verbose:
        print(f"  [fsp_bag] head(threshold) F1={out['f1_head_threshold']:.4f}  "
              f"head(top-{k}) F1={out['f1_head_topk']:.4f}  "
              f"global prior F1={out['f1_global_prior']:.4f}  "
              f"local 4-gram F1={out['f1_local_4gram']:.4f}  "
              f"({out['n_positions']:,} positions)")
    return out


def model_language_ce(m, L=512, seed=0, sample_count=1024, verbose=True):
    """Next-byte cross-entropy on long_range_language(L) sequences:
    [marker][filler...][controlled][terminator], scored over all L-1
    (predicts-next-byte) positions.  Chance (the generator's own CE) is
    ((L-3)*ln256 + ln2)/(L-1).  Control: mismatched-cue veto (marker->target
    labels permuted -> the dependency is broken, model falls back to uniform,
    i.e. CE -> ln256)."""
    rng = np.random.default_rng(seed)
    n = int(sample_count)
    b = G.get_batch("long_range_language", L, n, rng=rng)
    seqs = b["seq"]
    markers = b["meta"]["markers"]
    X = np.zeros((n, L), dtype=np.uint8)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
    res = _frozen_forward(m, X)
    logits = res["logits"]                     # (n, L, 256)

    # per-position CE: position t predicts byte t+1
    nxt = np.zeros((n, L), dtype=np.int64)
    for t in range(L - 1):
        nxt[:, t] = X[:, t + 1]
    nxt[:, L - 1] = -1                       # terminator: no next byte scored
    ce_pos = np.full((n, L), np.nan)
    for t in range(L - 1):
        lg = logits[:, t, :]
        v = lg.max(axis=1) + np.log(
            np.exp(lg - lg.max(axis=1, keepdims=True)).sum(axis=1)) - lg[
            np.arange(n), nxt[:, t]]
        ce_pos[:, t] = v
    ce_pos[:, L - 1] = 0.0                   # terminator is deterministic
    ce_main = float(np.nanmean(ce_pos))

    # mismatched-cue veto: permute the controlled bytes relative to the markers
    y_true = np.array([X[i, L - 2] for i in range(n)], dtype=np.int64)
    perm = np.random.default_rng(seed + 2000).permutation(n)
    y_pred = logits[np.arange(n), L - 2, :][perm]
    acc_v = float((y_pred.argmax(axis=1) == y_true[perm]).mean())
    se_v = math.sqrt(max(acc_v * (1 - acc_v), 0.0) / max(n, 1))

    if verbose:
        print(f"  [language_ce L={L}] main CE={ce_main:.4f} nats/B  "
              f"chance={G.chance_ce('long_range_language', L=L):.4f}  "
              f"veto acc={acc_v:.4f} (se {se_v:.4f})", flush=True)

    meta = _memory_meta(
        metric="language_ce",
        value=ce_main,
        unit="nats per scored byte",
        source="frozen numpy_port forward, cross_entropy over L-1 scored "
               "positions",
        chance={"chance_ce_nats_per_byte": G.chance_ce("long_range_language",
                                                       L=L),
                "per_byte_chance": 1.0 / 256,
                "chance_note": "chance CE of the generating distribution"},
        cue="marker_byte: byte at position 0 controls the byte at position "
            "L-2 (non-local dependency)",
        distractor="uniform random filler bytes (vocab 256)",
        gap_dist=f"deterministic per sample: one sequence of length L={L}",
        seed=seed,
        sample_count=n,
        control="mismatched-cue control (marker->controlled labels permuted; "
                "the dependency is broken)",
        control_value=ce_main,
        control_source="permuted marker labels",
        extra={"L": L, "n": n, "chance_ce_nats_per_byte":
               G.chance_ce("long_range_language", L=L),
               "controlled_position": L - 2,
               "terminator": 0xFF,
               "veto_acc": acc_v, "veto_se": se_v,
               "control_ce": None,
               "per_position_ce": None},
    )
    meta["value"] = ce_main
    meta["control_value"] = float(np.log(256))      # broken dependency -> uniform
    meta["control_source"] = ("permuted marker->controlled labels; the model "
                              "cannot use the marker, so it falls back to the "
                              "uniform predictor")
    return meta
