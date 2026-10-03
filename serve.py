#!/usr/bin/env python3
"""
A chat window for talking to TMT, and for watching its memory while you do.

    .venv/bin/python serve.py            then open http://127.0.0.1:8080

Ported from mini-AGI's serve.py, which is the same idea against a different
model. The shape is deliberately identical: one Flask app, four routes, an
SSE stream that yields a byte at a time, a single lock around the model, and
one HTML page with no build step. What changed is the panel in the middle --
mini-AGI can show you its working set because it pages experts, and TMT has no
experts, so the panel shows the two mechanisms TMT actually has.

  * TEST-TIME TRAINING. The conversation is fed back through step_batch, the
    same call train_gpu.py makes on a corpus. One AdamW update per `chunk`
    bytes, exactly like mini-AGI's LiveLearner and for the same reason: there
    is no second mechanism to reason about. Turn on by default, because
    test-time training is one of the five properties this model was built
    around, and hiding it behind a flag would make the page a worse
    demonstration of the thing it exists to show. `--no-learn` serves
    read-only.

  * RECURRENT MEMORY. This is TMT's flagship and the reason the page is worth
    more than a terminal. The chips are the dimensions whose decay gives them
    a half-life of more than `--long-bytes` bytes, each showing its half-life
    and, in the tooltip, the RMS of the state it is actually holding. That
    pairing is the point: RESULTS.md and PLAN-2026-09-22 both say, in their
    own words, not to infer liveness from sigma or half-life alone, because a
    dimension can have a 32-byte half-life and carry nothing at all. Showing
    half-life next to state RMS is the cheapest way to make that check
    possible by eye while you talk.

  * GRADIENT SNR. ||EMA(g)||^2 / EMA(||g||^2), from telemetry.py. The
    measured value over the whole 889,897-step run was ~0.05 and never
    recovered, which is the standing blocker on the science queue. It costs
    one already-computed gradient to show, so it is on the page.

TURN FORMAT. By default this feeds the model exactly what `numpy_port.Runtime
.chat` feeds it: raw bytes, no markers, and the stop head decides where the
reply ends. That is deliberate. The recorded chat results in RESULTS.md were
measured on that path, and a web page that quietly switched formats would stop
being comparable to them. `--markers` switches to <user>/<bot> turns for a
stricter separation, which the current model handles badly enough to leak the
markers back -- RESULTS.md records exactly that -- so it is opt-in.

CHECKPOINTS. Learning is on, so talking to this model changes the checkpoint on
disk. Saves go through Runtime.save_checkpoint, which is the project's
tagged/atomic/keep-k path: it writes {base}_t{step:06d}.npz, hard-links the
rolling path to the newest, and prunes all but `--keep-k`. A result's
checkpoint is never silently destroyed by chatting to the model. `--path`
points somewhere else than the rolling `tmt.npz` to keep a served model off
the training checkpoint entirely.

NO DEPENDENCIES. flask if it is installed, otherwise the stdlib
`http.server` behind the same four routes -- this project ships with numpy and
torch and nothing else, and a chat window is not a reason to add a package.
Binds to localhost.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from collections import Counter

# The engine lives at the repo root and this file is imported by name from
# tools, bundles and tests, so put the root on the path no matter where this
# was launched from.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np

import numpy_port

# telemetry pulls in torch, and torch is only needed for the gradient-SNR
# meter. The chat itself is numpy. Import it optionally so a numpy-only
# environment still serves, with one meter reporting nothing rather than the
# whole page refusing to start.
try:
    import torch
    import telemetry
    _HAVE_TORCH = True
except ImportError:                                          # pragma: no cover
    torch = None
    telemetry = None
    _HAVE_TORCH = False

# ---------------------------------------------------------------- transport
# One router, two servers. flask when it is there, because it is nicer; the
# stdlib when it is not, because it always is.

ROUTES = {}


def route(path, method="GET"):
    def deco(fn):
        ROUTES[(path, method)] = fn
        return fn
    return deco


def _json_response(payload, status=200):
    return status, {"Content-Type": "application/json"}, json.dumps(payload).encode()


# ------------------------------------------------------------------- state
# One model, one conversation at a time. Two overlapping requests would
# interleave their recurrent state updates -- which here is not a corrupted
# KV cache but a corrupted *memory*, and the failure would be silent: the
# model would just answer as if it had forgotten something. That is worse than
# a garbled reply, so the lock is not optional.
LOCK = threading.Lock()
STATE = {
    "runtime": None,      # numpy_port.Runtime, owns the model
    "model": None,
    "path": None,
    "learn": True,
    "pending": 0,         # bytes fed since the last optimiser step
    "chunk": 1024,        # bytes per optimiser step
    "steps": 0,           # optimiser steps taken by this conversation
    "t0": 0,              # model.t at load, for the "steps" readout
    "ema": None,          # per-role gradient EMAs
    "snr": None,          # per-role SNR dict
    "gmean": None,        # per-role mean |grad|
    "loss": None,
    "ce": None,
    "ce_best": None,
    "gen_s_total": 0.0,   # session wall clock, writing
    "learn_s_total": 0.0, # session wall clock, training on the exchange
    "hot": [],            # last memory_report(), for diffing
    "hot_set": None,      # last tail membership, for the churn stat
    "cancel": False,      # set by /api/cancel; checked between generated bytes
    "session": [],        # one record per exchange, for the session export
    "meta": {},           # what the session started from
}

# How many trailing bytes the instrument panels cover. 200 is a bit under the
# 400-byte default cap, so a full reply is usually shown whole.
HIST = 200

# The probability below which a byte counts as a guess rather than a choice.
# It lives here and travels to the page in the done frame, because the number
# the panel draws its rule at, the number the page underlines bytes under, and
# the number the export counts are one number, and three copies of a constant
# is how they stop being one number.
UNSURE_P = 0.1

# The half-life buckets RESULTS.md reports in. A dimension's half-life L solves
# sigma**L = 0.5, so L = log(0.5)/log(sigma); sigma at the init default of 0.5
# is exactly one byte, which is where 8,064 of 8,192 dims have always sat.
HL_BUCKETS = [(0.0, 1.0, "<1"), (1.0, 2.0, "1-2"), (2.0, 4.0, "2-4"),
              (4.0, 8.0, "4-8"), (8.0, 16.0, "8-16"),
              (16.0, 32.0, "16-32"), (32.0, float("inf"), ">32")]

PRIME = ""
U0, U1, B0, B1 = "<user>", "</user>", "<bot>", "</bot>"


# ------------------------------------------------------------------- prime
def load_prime(chars, root=None, prefer="tinystories_valid.txt"):
    """
    Corpus text to open the conversation with, or "" if there is none.

    Cut at a LINE boundary and only take whole lines. TMT's stop head is
    trained on end-of-line, and the tail-credit and ring paths both key off
    line structure, so a prime that stops mid-sentence hands the model a
    half-formed byte pair to predict from. mini-AGI cuts where its markers
    balance for the same class of reason.

    Deterministic: the first usable file, every time. A prime picked at random
    would make the same conversation give different answers across restarts,
    and for this model a repeatable answer is worth more than variety -- the
    whole point of the page is that two runs are comparable.
    """
    if chars <= 0:
        return ""
    # Relative to the repo root, not the cwd: every tool in this project runs
    # from any directory, and a prime that silently disappears because you
    # launched from /tmp would change the model's answers without saying so.
    root = root or os.path.join(ROOT, "data", "eval")
    if not os.path.isdir(root):
        return ""
    names = sorted(f for f in os.listdir(root) if f.endswith(".txt"))
    names = ([n for n in names if n == prefer]
             + [n for n in names if n != prefer])
    for name in names:
        path = os.path.join(root, name)
        try:
            with open(path, "rb") as f:
                raw = f.read(chars * 3)
        except OSError:
            continue
        out, used = [], 0
        for line in raw.splitlines(keepends=True):
            if used + len(line) > chars:
                break
            out.append(line)
            used += len(line)
        if used >= chars // 2:
            return b"".join(out).decode("utf-8", errors="replace")
    return ""


# ------------------------------------------------------------------- memory
def long_dims(model, threshold=4.0, limit=32):
    """
    The dimensions that remember for more than `threshold` bytes.

    Half-life in bytes: the state is multiplied by sigma once per byte, so
    sigma**hl == 0.5, i.e. hl = log(0.5)/log(sigma). A sigma at or below 0.5
    is at or under one byte of memory and is not what this is looking for.

    Each entry carries the RMS of the state that dim is ACTUALLY holding, not
    just the rate it would decay at. A dimension can have a 32-byte half-life
    and sit at zero, and the project's own rule is not to call that memory --
    so the two numbers travel together, all the way to the tooltip.
    """
    out = []
    for i, l in enumerate(model.layers):
        sig = 1.0 / (1.0 + np.exp(-np.clip(l.decay, -60, 60)))
        hot = sig > 0.5
        if not hot.any():
            continue
        state = np.asarray(l.states[0], dtype=np.float64)
        rms = np.sqrt(np.mean(state * state))
        # per-dim RMS, not a single scalar for the layer
        dims = np.where(hot)[0]
        for j in dims:
            s = float(sig[j])
            if s <= 0.5:
                continue
            hl = float(np.log(0.5) / np.log(s))
            if hl < threshold:
                continue
            v = float(state[j])
            out.append({"id": "L%d.%d" % (i, j), "layer": i, "dim": int(j),
                        "sigma": round(s, 4), "hl": round(hl, 2),
                        "rms": round(abs(v), 5),
                        "live": abs(v) >= 1e-3})
    # longest memory first; that is the one a reader is looking for
    out.sort(key=lambda d: -d["hl"])
    return out[:limit]


def memory_report(model, threshold=4.0, limit=32):
    """The chip list plus the numbers the meter is drawn from."""
    dims = long_dims(model, threshold, limit)
    n_live = sum(1 for d in dims if d["live"])
    state_rms = float(np.sqrt(np.mean(
        np.concatenate([np.asarray(l.states[0], dtype=np.float64).ravel()
                        for l in model.layers]) ** 2))) if model.layers else 0.0
    return {"dims": dims, "n_long": len(dims), "n_live": n_live,
            "state_rms": round(state_rms, 4),
            "threshold": threshold}


def _half_lives(model):
    """Half-life in bytes for every dimension, as a flat (layers*dim,) array."""
    out = []
    for l in model.layers:
        sig = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(l.decay, dtype=np.float64),
                                           -60, 60)))
        # sigma <= 0.5 is a sub-byte half-life; sigma == 1 never decays. Clip
        # the ends so the division cannot produce inf or a negative.
        s = np.clip(sig, 1e-6, 1.0 - 1e-9)
        out.append(np.log(0.5) / np.log(s))
    return np.concatenate(out) if out else np.zeros(0)


def halflife_histogram(model, long_threshold=4.0):
    """
    EVERY dimension bucketed by half-life, not just the long-lived ones.

    The chip row shows the handful of dims past the threshold. The number that
    actually describes this model is the one RESULTS.md keeps quoting and the
    page never showed: how many of the 8,192 dims are sitting at about one
    byte of memory. The top bucket being nearly empty IS the finding.

    Each bucket also reports how many of its dims are LIVE, because the
    project's rule is that a long half-life is not memory on its own -- the
    count and the live count are two different facts and both are here.
    """
    hl = _half_lives(model)
    if not hl.size:
        return {"buckets": [], "total": 0, "n_long": 0, "n_live_long": 0,
                "threshold": long_threshold}
    # per-dim state magnitude, so a bucket can say whether its dims are used
    mag = np.concatenate([np.abs(np.asarray(l.states[0], dtype=np.float64))
                          for l in model.layers])
    buckets, n_long, n_live_long = [], 0, 0
    for lo, hi, label in HL_BUCKETS:
        m = (hl >= lo) & (hl < hi)
        n = int(m.sum())
        live = int((mag[m] >= 1e-3).sum())
        buckets.append({"label": label, "lo": lo,
                        "hi": (None if hi == float("inf") else hi),
                        "n": n, "live": live,
                        # a float, NOT int(round(...)) -- that truncated the
                        # sub-byte bucket's mean of 0.8 to "0B", which reads
                        # as "these dims have no memory at all" when they in
                        # fact have most of a byte
                        "bytes": (round(float(hl[m].mean()), 2) if n else None)})
        if lo >= long_threshold:
            n_long += n
            n_live_long += live
    return {"buckets": buckets, "total": int(hl.size), "n_long": n_long,
            "n_live_long": n_live_long, "threshold": long_threshold}


def degeneracy(data):
    """
    The reply's own shape, as numbers -- the defects cross-entropy cannot see.

    RESULTS.md: "FS-N has a newline-collapse mode (16 consecutive newlines)
    that the other three never approach. Plausibly a paragraph-boundary
    attractor in Gutenberg. It is invisible to CE, which is exactly the class
    of defect this check exists to catch." That was measured by hand, offline,
    with a fixed-prompt protocol. It takes ten lines to compute live, and it
    is the one thing on this page that can disagree with the CE.

    `repeat` is the fraction of adjacent byte pairs that are not the first
    sighting of that pair -- 0 is every pair novel, and it climbs toward 1 as
    the model loops.
    """
    data = bytes(data)
    n = len(data)
    if n < 2:
        return None
    run = best_run = 1
    nl = best_nl = 0
    for i, b in enumerate(data):
        if i and b == data[i - 1]:
            run += 1
        else:
            run = 1
        best_run = max(best_run, run)
        nl = nl + 1 if b == 10 else 0
        best_nl = max(best_nl, nl)
    pairs = Counter(zip(data[:-1], data[1:]))
    repeat = sum(c - 1 for c in pairs.values()) / max(1, n - 1)
    counts = Counter(data)
    ent = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return {"n": n, "unique": len(counts),
            "max_run": int(best_run),
            "max_newlines": int(best_nl),
            "repeat": round(float(repeat), 4),
            "entropy": round(float(ent), 3),
            # the documented collapse modes, flagged rather than left to the eye
            "degenerate": bool(best_nl >= 8 or best_run >= 8
                               or len(counts) <= 4 or repeat > 0.5)}


def hot_set(model):
    """The exact-credit set as (layer, dim) pairs."""
    t = getattr(model, "tail", None)
    if t is None or not getattr(t, "cols", None):
        return set()
    return {(int(l), int(d)) for l, cols in t.cols.items() for d in cols}


def hot_churn(model):
    """
    Dims entering and leaving the dense exact-credit set since the last look.

    RESULTS.md flags an unexplained regression shared by two arms at 28k->60k
    that coincided with the hot set exploding (5 -> 54 dims) and calls it
    "possibly more important for long runs than the corpus choice".
    tail_credit.sync() already computes adopt/drop every REBASE_EVERY steps and
    throws the counts away; this keeps them, so the growth is a curve rather
    than something you reconstruct from checkpoints afterwards.
    """
    now = hot_set(model)
    prev = STATE.get("hot_set")
    STATE["hot_set"] = now
    if prev is None:
        return {"n": len(now), "entered": 0, "left": 0}
    return {"n": len(now), "entered": len(now - prev),
            "left": len(prev - now)}


def learn_state():
    """How far the stream is from its next optimiser step."""
    if not STATE.get("learn"):
        return None
    return {"pending": int(STATE["pending"]), "chunk": int(STATE["chunk"]),
            "steps": int(STATE["steps"]),
            "model_t": int(STATE["model"].t) if STATE.get("model") else 0}


# ------------------------------------------------------- parameter roles
def _role_of(name):
    if name == "embed":
        return "embedding"
    if name in ("Wd", "bd", "Ws", "bs"):
        return "decoder"
    if name.endswith(".decay"):
        return "decay"
    if name.endswith(".W"):
        return "layer weights"
    if name.endswith(".g"):
        return "layer norm gain"
    if name.endswith(".b"):
        return "layer norm bias"
    return name


def build_groups(model):
    """Flat index arrays for the parameter roles.

    The queued experiment is a shared-weight multiplier against a decay
    multiplier, so `decay` (every layer*.decay) and `trunk` (everything else)
    are the two groups that matter. `named` splits the trunk further, but
    only for the delta view, where the question is "which tensor moved".
    `per_layer` exists because the decay question is not uniform across the
    stack -- averaging 16 layers into one bar cannot tell you whether every
    layer's decay is dead or three of them are carrying the rest.

    Built once out of `model.shapes`, which IS the flat buffer's layout, so
    a group cannot silently drift out of step with the buffer it indexes.
    """
    decay, named, per_layer, per_layer_decay = [], {}, {}, {}
    off = 0
    for name, shape in model.shapes:
        size = int(np.prod(shape))
        idx = np.arange(off, off + size, dtype=np.int64)
        off += size
        if name.endswith(".decay"):
            decay.append(idx)
        if name.startswith("layer"):
            tag = name.split(".")[0]                  # "layer7"
            per_layer.setdefault(tag, []).append(idx)
            if name.endswith(".decay"):
                per_layer_decay[tag] = idx
        named.setdefault(_role_of(name), []).append(idx)
    allidx = np.arange(off, dtype=np.int64)
    decay_idx = (np.concatenate(decay) if decay
                 else np.zeros(0, dtype=np.int64))
    return {"decay": decay_idx, "trunk": np.setdiff1d(allidx, decay_idx),
            "named": {k: np.concatenate(v) for k, v in named.items()},
            "per_layer": {k: np.concatenate(v) for k, v in per_layer.items()},
            "per_layer_decay": per_layer_decay,
            "all": allidx, "n": off}


def model_groups(model):
    """Groups for this model, cached on it and rebuilt if the buffer changes."""
    g = getattr(model, "_serve_groups", None)
    if g is None or g["n"] != model.grads.size:
        g = build_groups(model)
        model._serve_groups = g
    return g


def update_snr(model):
    """
    Gradient SNR PER ROLE, using telemetry.py's definition unchanged.

    Split because the standing blocker is currently a single number, and one
    number hides the question. An overall SNR of ~0.05 -- measured over the
    whole 889,897-step run and never recovered -- is consistent with two very
    different worlds:

      * the trunk is barely learning at all, or
      * the trunk is fine and the DECAY group has never had a coherent signal.

    numpy_port's own comment on decay_init argues for the second: the default
    init puts sigma at ~0.5, a ONE-byte half-life, so "the recurrent state is
    too short-lived for the decay param to get a coherent gradient, which is
    why decay sits at 0.5 forever". If these two bars come back showing a
    healthy trunk and a dead decay group, then the queued role-specific LR
    factorial is treating a symptom whose cause is the init, and no learning
    rate will fix it. That is worth knowing before spending quota on it.

    Cost is one optimiser step's worth, not one byte's: this only runs when a
    chunk fills.
    """
    if not _HAVE_TORCH:
        return
    groups = model_groups(model)
    if STATE["ema"] is None:
        STATE["ema"] = {}
    g = model.grads
    # "all" is the whole buffer, so take it zero-copy off the contiguous array
    # rather than fancy-indexing 4.5M floats into a copy for no reason
    want = {"trunk": groups["trunk"], "decay": groups["decay"],
            "all": None}
    for tag, idx in groups["per_layer"].items():
        want[tag] = idx                          # every param in that layer
        want[tag + ".d"] = groups["per_layer_decay"][tag]
    out, means = {}, {}
    allview = None
    for key, idx in want.items():
        if idx is None:
            if allview is None:
                allview = torch.from_numpy(
                    np.ascontiguousarray(g, dtype=np.float32))
            seg, size = allview, g.size
        else:
            seg = torch.from_numpy(
                np.ascontiguousarray(g[idx], dtype=np.float32))
            size = int(idx.size)
        slot = STATE["ema"].get(key)
        if slot is None or slot[0].numel() != size:
            slot = (torch.zeros(size, dtype=torch.float32),
                    torch.zeros(size, dtype=torch.float32))
            STATE["ema"][key] = slot
        mn, snr = telemetry.update_gradient_ema(seg, slot[0], slot[1],
                                                decay=0.9)
        out[key] = snr
        # the magnitude the EMA already computed and that used to be thrown
        # away. SNR is direction consistency; magnitude is scale. A group can
        # have decent SNR and a gradient of 1e-9, and only both numbers say
        # whether it is learning.
        means[key] = mn
    STATE["snr"] = out
    STATE["gmean"] = means


def snr_state():
    """Per-role SNR for the meters. None until the first optimiser step."""
    s = STATE.get("snr")
    if s is None:
        return None
    gm = STATE.get("gmean") or {}
    layers = []
    for tag in sorted(s):
        if not tag.startswith("layer") or tag.endswith(".d"):
            continue
        # `layer3` is the whole layer, `layer3.d` its decay vector; only the
        # whole-layer tags drive the rows, and the decay figures are attached
        try:
            n = int(tag[len("layer"):])
        except ValueError:
            continue
        layers.append({
            "layer": n,
            "snr": round(float(s[tag]), 5),
            "gmean": float("%.3g" % gm.get(tag, 0.0)),
            "dsnr": round(float(s.get(tag + ".d", 0.0)), 5),
            "dgmean": float("%.3g" % gm.get(tag + ".d", 0.0)),
        })
    layers.sort(key=lambda r: r["layer"])
    return {"trunk": round(float(s["trunk"]), 4),
            "decay": round(float(s["decay"]), 4),
            "all": round(float(s["all"]), 4),
            "trunk_g": float("%.3g" % gm.get("trunk", 0.0)),
            "decay_g": float("%.3g" % gm.get("decay", 0.0)),
            "layers": layers,
            # how many optimiser steps the EMAs have actually seen. One sample
            # gives EXACTLY 0.1 for every group whatever the gradient is, so an
            # early reading of 0.1/0.1 carries no information about which role
            # is better fed -- and the ratio between them looks reassuringly
            # like 100%. The page says so rather than letting the number be
            # read as a verdict.
            "steps": int(STATE["steps"]),
            "warm": bool(STATE["steps"] < 20),
            # ce is the TRAINING ce on the exchange, which the model is being
            # fitted to as it goes -- it is not a held-out measurement and not
            # a quality number. It is labelled as such in the page.
            "ce": None if STATE.get("ce") is None else round(float(STATE["ce"]), 4),
            "best_ce": (None if STATE.get("ce_best") is None
                        else round(float(STATE["ce_best"]), 4))}


# ------------------------------------------------------------ projections
def project_2d(points):
    """
    PCA the trailing latents onto two axes.

    The latent is what the JEPA loss actually predicts and what the decoder
    reads, so its shape is a fact about the model that cross-entropy cannot
    report. A trace that wanders is different information from a trace that
    loops, and neither shows up in a CE.

    Returns the variance explained per axis alongside the points, because a
    2D scatter of the first two principal components of a 512-vector means
    very little if those two components hold 0.4% of the variance -- and
    drawing that without saying so is how a projection becomes a lie.
    """
    if len(points) < 3:
        return {"points": [], "var": [0.0, 0.0], "n": len(points)}
    A = np.asarray(points, dtype=np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    try:
        _, S, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"points": [], "var": [0.0, 0.0], "n": len(points)}
    comp = A @ Vt[:2].T
    denom = max(1e-12, (len(A) - 1) * A.shape[1])
    var = (S[:2] ** 2) / denom
    return {"points": [[round(float(p[0]), 4), round(float(p[1]), 4)]
                       for p in comp],
            "var": [round(float(var[0]), 5), round(float(var[1]), 5)],
            "n": len(points)}


def state_heat(model):
    """Per-layer recurrent-state RMS right now, for the heatmap's last column."""
    return [round(float(np.sqrt(np.mean(
        np.asarray(l.states[0], dtype=np.float64) ** 2))), 5)
        for l in model.layers]


def theta_delta(model, before):
    """
    What one exchange did to the weights, per role.

    `rel` is the update-to-parameter ratio the plan asks for: RMS of the
    change over RMS of the parameter. At the default --learn-chunk 1024 this
    is small, and seeing the actual number is more useful than the phrase
    "it learns while you talk to it".
    """
    if before is None:
        return []
    groups = model_groups(model)["named"]
    delta = model.theta - before
    rows = []
    for role in sorted(groups):
        idx = groups[role]
        seg = delta[idx].astype(np.float64)
        base = model.theta[idx].astype(np.float64)
        rms = float(np.sqrt(np.mean(seg * seg)))
        brms = float(np.sqrt(np.mean(base * base)))
        rows.append({"role": role, "n": int(idx.size),
                     "rms": float("%.3g" % rms),
                     "rel": float("%.3g" % (rms / brms)) if brms > 0 else 0.0})
    return rows


# ------------------------------------------------------------------ prompt
def build_prompt(text, markers=None):
    """
    ONE new turn, as the model reads it -- not the whole conversation.

    This is where TMT departs from mini-AGI, and the difference is the whole
    reason the page is worth having. mini-AGI rebuilds its prompt from the
    full message list on every request, because a transformer with a fresh KV
    cache has no memory between requests and the history IS its context. TMT's
    memory is the recurrent state, and that state persists between requests
    here exactly as it does in `Runtime.chat`. Replaying the history would
    therefore apply every previous reply to the state a SECOND time -- the
    model would be answering as though it had already said everything it has
    said, twice.

    So the rule is the terminal chat's rule: feed the user's bytes, let the
    reply happen in the state, and feed nothing back. The conversation lives
    in the state, which is also why it is not truncated and why there is no
    context budget to trim against.
    """
    if markers is None:
        markers = markers_on()
    text = (text or "").strip()
    if not text:
        return ""
    if not markers:
        return text + "\n"
    return f"{U0}\n{text}\n{U1}\n"


# --------------------------------------------------------------- generation
def stream(first_byte, max_new, threshold):
    """
    Yield one byte at a time, the way the model writes it.

    `observe_stats` advances the recurrent state and returns the sampled
    byte, the stop-head score, the JEPA latent, and what the sampler knew
    when it picked that byte. No gradient, no optimiser --
    generation must not change the weights, or the reply you are watching
    would be the model training on its own half-sentence, and the learning
    meter would be lying about when the step happens.

    The stop head is a learned signal and can sit below the threshold forever,
    so --max-tokens is a hard cap, the same guard Runtime.chat uses. Hitting
    it is reported, not hidden: a reply that stops because the head fired and
    a reply that stops because it ran out of budget are different facts.

    Everything measured is appended to a fixed-length history rather than
    accumulated, so a long conversation cannot grow the payload without bound.

    Returns its measurements as the generator's return value, which the caller
    reads after the loop -- there is no channel out of a `for ... yield` loop
    otherwise, and threading a mutable dict through would be the same thing
    with worse typing.
    """
    model = STATE["model"]
    b = int(first_byte)
    produced = 0
    capped = False
    latents, stops, heat = [], [], []
    # what the sampler knew at each byte: the probability it gave the byte it
    # actually emitted, the entropy of the distribution it drew from, and the
    # effective temperature. These are the numbers the misspelling panel is
    # made of -- cross-entropy has already averaged them away by the time it
    # reaches a loss, so nothing downstream of the sampler still has them.
    probs, ents, temps = [], [], []
    for _ in range(int(max_new)):
        if STATE.get("cancel"):
            yield {"note": "stopped at your request, after %d bytes" % produced}
            break
        b, stop, latent, conf = model.observe_stats(b)
        latents.append(np.asarray(latent, dtype=np.float32).copy())
        stops.append(round(float(stop), 5))
        heat.append(state_heat(model))
        probs.append(round(float(conf["p"]), 4))
        ents.append(round(float(conf["entropy"]), 4))
        temps.append(round(float(conf["temp"]), 4))
        # A byte-level model emits bytes that are not printable characters,
        # and a few of the 256 are outside the UTF-8 range entirely. Dropping
        # them silently would make the reader see a five-byte reply while the
        # counter said twelve, which is exactly the kind of quiet mismatch this
        # project's history is full of. So they are SENT, flagged, and the page
        # draws a placeholder -- while the reply string stays clean.
        if 32 <= b < 127 or b in (9, 10, 13):
            yield {"t": chr(b), "p": probs[-1]}
        else:
            yield {"t": "", "b": int(b), "p": probs[-1]}
        produced += 1
        if stop > threshold:
            break
        if produced >= int(max_new):
            capped = True
            break
    if capped:
        yield {"note": "stopped at the %d-byte cap - the stop head never "
                       "fired" % int(max_new)}
    return {"latents": latents[-HIST:], "stops": stops[-HIST:],
            "heat": heat[-HIST:], "probs": probs[-HIST:],
            "ents": ents[-HIST:], "temps": temps[-HIST:],
            "produced": produced}


# ------------------------------------------------------------------ learning
def remember(user_text, bot_text, save_every):
    """
    Read the finished exchange the way the trainer reads a corpus.

    ONLY the exchange that was just spoken goes in, fed as a B=1 stream with
    the stop-head target set on the byte pair that ends a line. That is
    exactly what `Runtime.ingest` does and what `Runtime.train` does over a
    file, so chat and training are one path.

    Re-feeding the whole prompt each time would put the earliest turns through
    the model once per exchange -- re-reading rather than reading on, which is
    the dense regime the training docs warn about and which is why the
    gradient SNR is where it is.

    A step needs `chunk` bytes and an exchange is usually shorter than that,
    so the first several turns take none at all. That is reported every time
    rather than left ambiguous: silence here is indistinguishable from
    learning being broken, which is the wrong thing to leave ambiguous.

    This is a GENERATOR, and it yields progress while it trains. That is not a
    nicety: remember() is a full step_batch per byte at B=1, and the measured
    split is ~97% of the turn's wall clock. The reply is finished long before
    the weights are, and for those seconds the page used to show a blinking
    caret and a static line of text -- indistinguishable from a hang, on the
    part of the turn that takes longest. The caller reads the return value off
    StopIteration, as it already does for stream().
    """
    rt = STATE.get("runtime")
    if rt is None or not STATE.get("learn"):
        return
    model = STATE["model"]
    if not markers_on():
        text = (user_text or "").strip() + "\n" + (bot_text or "")
    else:
        text = f"{U0}\n{(user_text or '').strip()}\n{U1}\n{B0}\n{bot_text or ''}\n{B1}\n"
    data = text.encode("utf-8", errors="replace")
    if len(data) < 2:
        return
    steps, last = 0, None
    # the loss over the exchange, one point per byte, so "did the loss fall
    # while it was being fitted" is a shape on the page and not a feeling
    trace = []
    total = len(data) - 1
    t0 = time.time()
    # ~60 updates whatever the exchange length, and always on a step boundary
    every = max(1, total // 60)
    try:
        for i, (c, n) in enumerate(zip(data[:-1], data[1:])):
            # ends=True on the pair that finishes a line, so the stop head
            # gets the same targets it gets in training
            is_end = data[i + 1:i + 2] == b"\n"
            # step_batch returns the total loss and sets .last_ce (the CE
            # component alone, in nats/byte, which is the number the ruler
            # reports). Track both: the CE is the one worth watching.
            last = model.step_batch([c], [n], [is_end])
            STATE["pending"] += 1
            steps += 1
            STATE["loss"] = last
            STATE["ce"] = model.last_ce
            trace.append(round(float(model.last_ce), 4))
            took_step = False
            if STATE.get("ce_best") is None or model.last_ce < STATE["ce_best"]:
                STATE["ce_best"] = float(model.last_ce)
            if STATE["pending"] >= STATE["chunk"]:
                # one optimiser step per chunk bytes; the SNR is read off the
                # gradient step_batch already computed, so it costs nothing
                update_snr(model)
                STATE["pending"] = 0
                STATE["steps"] += 1
                took_step = True
                if save_every and STATE["steps"] % save_every == 0:
                    rt.save_checkpoint()
            # on a step boundary (the step count has to be exact), on the
            # interval, and on the last byte (which is the frame that says the
            # pass is over, and a progress bar that stops at 98% is a lie)
            if took_step or i % every == 0 or i == total - 1:
                yield {"train": {
                    "i": i + 1, "total": total, "steps": steps,
                    "ce": round(float(STATE["ce"]), 4),
                    "loss": (None if last is None
                             else round(float(last), 4)),
                    "pending": STATE["pending"], "chunk": STATE["chunk"],
                    "snr": STATE.get("snr"),
                    "elapsed": round(time.time() - t0, 2)}}
    except Exception as e:                                    # noqa: BLE001
        # a failed step must not cost the reply that was already written
        print(f"[learn] failed: {e}", file=sys.stderr)
        return {"error": str(e)}
    return {"bytes": steps, "pending": STATE["pending"],
            "chunk": STATE["chunk"], "total": STATE["steps"],
            "loss": None if STATE.get("loss") is None else round(float(STATE["loss"]), 4),
            "ce": None if STATE.get("ce") is None else round(float(STATE["ce"]), 4),
            # the trace is capped for the same reason the panels are: an
            # all-night session of exchanges must not grow one array forever
            "trace": trace[-HIST:],
            "trace_from": max(0, total - len(trace[-HIST:])),
            "snr": STATE.get("snr")}


_MARKERS = [False]


def markers_on():
    return bool(_MARKERS[0])


# ------------------------------------------------------------------ routing
@route("/api/state")
def api_state():
    """
    What the three mechanisms are doing between replies.

    Asked once on load and after every exchange, so the counters are right
    when nothing is being generated. Otherwise "how far from the next update"
    would only exist while the model is talking.
    """
    model = STATE.get("model")
    rep = memory_report(model, STATE.get("long_threshold", 4.0)) if model else None
    return _json_response({
        "learn": learn_state(),
        "snr": snr_state(),
        "memory": rep,
        "histogram": (halflife_histogram(model, STATE.get("long_threshold", 4.0))
                      if model else None),
        "churn": ({"n": len(hot_set(model))} if model else None),
        "max_tokens": STATE.get("max_tokens", 400),
        "threshold": STATE.get("threshold", 0.35),
        "markers": markers_on(),
        "path": STATE.get("path"),
        "turns": len(STATE["session"]),
        "meta": STATE.get("meta", {}),
    })


@route("/api/prime")
def api_prime():
    """
    What the model was opened with, before the first message.

    The prime is fed into the recurrent state ONCE, at load, and is not part
    of any turn. That is a deliberate difference from mini-AGI, which
    prepends the prime to every prompt: that is free there, because a
    transformer with a fresh cache each request has no state to disturb, but
    here it would re-apply a thousand bytes of corpus to the memory on every
    single turn and bury the conversation.

    It still needs to be inspectable. The model is answering from state the
    person talking to it cannot see, and that is the sort of thing that
    should be shown rather than trusted.
    """
    return _json_response({"chars": len(PRIME), "fed_at": "load, once",
                           "text": PRIME})


def record_session(user_text, reply, done):
    """
    Append one row per exchange, in the shape RESULTS.md wants.

    This project's whole discipline is that a number without its seed, its
    byte count, its ruler setting and its checkpoint is not a result. A chat
    page generates exactly those numbers and then throws them away, so every
    observation made through it is unreproducible by default. The export is
    the fix: the checkpoint it started from and its digest, and per turn the
    prompt, the reply, the byte counts, the CE, the per-role SNR, and what the
    stop head did, and now what the sampler believed about each byte it wrote.

    Capped so an all-night session cannot grow without bound.
    """
    stops = done.get("stops") or []
    # the per-byte series, plus the two aggregates a results table actually
    # wants: a mean confidence and a count of the bytes that were guesses
    conf = done.get("conf") or {}
    cp = conf.get("p") or []
    STATE["session"].append({
        "user": user_text,
        "reply": reply,
        "n": done.get("n"),
        "nonprint": done.get("nonprint"),
        "cps": done.get("cps"),
        "gen_s": done.get("gen_s"),
        "learn_s": done.get("learn_s"),
        "threshold": done.get("threshold"),
        "model_t": (done.get("learn") or {}).get("model_t"),
        "learned": done.get("learned"),
        "snr": done.get("snr"),
        "stop_max": round(max(stops), 4) if stops else None,
        "stop_fired": bool(stops and max(stops) > (done.get("threshold") or 9e9)),
        # what the sampler knew per byte, which is the one reading CE has
        # already destroyed by the time a loss exists
        "conf": conf,
        "mean_p": round(sum(cp) / len(cp), 4) if cp else None,
        "n_unsure": sum(1 for v in cp if v < conf.get("u", UNSURE_P)) if cp else None,
        "long_dims": (done.get("memory") or {}).get("n_long"),
        # the readings CE cannot see, and the one number that can disagree
        "degeneracy": done.get("degeneracy"),
        "churn": done.get("churn"),
        "delta": done.get("delta"),
    })
    del STATE["session"][:-2000]


@route("/api/session")
def api_session():
    """The whole session, for download. See record_session."""
    return _json_response({"meta": STATE.get("meta", {}),
                           "turns": STATE["session"]})


@route("/api/cancel", "POST")
def api_cancel():
    """
    Stop the reply in progress.

    Deliberately does NOT take LOCK. The lock is held for the whole
    generation, so a cancel that waited on it would wait for the thing it is
    trying to stop. It only sets a flag the byte loop checks between bytes.
    """
    STATE["cancel"] = True
    return _json_response({"cancel": True})


@route("/api/chat", "POST")
def api_chat():
    body = read_json()
    messages = body.get("messages", [])
    max_new = int(body.get("max_new") or STATE.get("max_tokens", 400))
    threshold = float(body.get("threshold", STATE.get("threshold", 0.35)))
    # the half of the exchange the model did not predict, which is where the
    # signal in a conversation is. Only the LAST user turn is fed: the state
    # already holds everything before it.
    last_user = next((m.get("content") for m in reversed(messages)
                      if m.get("role") == "user"), "")

    turn = build_prompt(last_user)
    data = turn.encode("utf-8", errors="replace")
    if len(data) < 2:
        return sse(iter(()))

    def events():
        t0, n, produced, nonprint = time.time(), 0, [], 0
        learned, obs = None, {}
        before = None
        STATE["cancel"] = False
        try:
            with LOCK:
                model = STATE["model"]
                # one 18 MB snapshot per turn, not per byte: this is what makes
                # "what did this conversation change" a measurement
                before = (model.theta.copy() if STATE.get("learn") else None)
                # the prompt bytes go in as state WITHOUT training. A gradient
                # on the prompt would make the reply depend on the optimiser
                # state of a half-fed stream, and would also mean the user
                # message is learned from before it has been answered.
                for c in data[:-1]:
                    model.frozen_step(c)
                STATE["hot"] = memory_report(model, STATE.get("long_threshold", 4.0))["dims"]
                yield pack({"swap": {"at": 0, "moved": None,
                                     "memory": STATE["hot"]}})
                # the clock for the RATE starts here, not at the top of the
                # turn. dt used to cover the whole turn including remember(),
                # which is a full step_batch per byte at B=1 and takes an
                # order of magnitude longer than generation -- so the number
                # labelled "bytes/s" understated generation by 5-10x. Measured
                # at 5.4 B/s while the model was actually writing at >130.
                tgen = time.time()
                # `next()` in a loop rather than `for ... yield`, because the
                # generator returns its measurements as its StopIteration value
                # and a plain for-loop discards those.
                gen = stream(data[-1], max_new, threshold)
                while True:
                    try:
                        ev = next(gen)
                    except StopIteration as stop:
                        obs = stop.value or {}
                        break
                    if "t" in ev:
                        if ev["t"]:
                            n += 1
                            produced.append(ev["t"])
                        else:
                            nonprint += 1
                    yield pack(ev)
                gen_dt = max(time.time() - tgen, 1e-6)
                # The reply is finished but the exchange has not been trained
                # on yet, and remember() is a full step_batch per byte at B=1
                # -- so this window is seconds, not milliseconds. Without a
                # frame saying so, the page looks stuck: the text stops moving
                # but the caret keeps blinking and the status still says it is
                # thinking. Say which of the two is happening.
                yield pack({"learning": True, "n": n,
                            "gen_s": round(gen_dt, 2)})
                tlearn = time.time()
                # remember() is a generator now, for the reason its docstring
                # gives: this window is the overwhelming majority of the turn
                # and it used to be a silent one. Drained with next() in a
                # loop, because the return value -- the learned record the
                # done frame carries -- arrives as StopIteration.value and a
                # for-loop throws that away.
                lgen = remember(last_user, "".join(produced),
                                STATE.get("save_every", 8))
                while True:
                    try:
                        prog = next(lgen)
                    except StopIteration as stop:
                        learned = stop.value or None
                        break
                    yield pack(prog)
                learn_dt = time.time() - tlearn
        except Exception as e:                                # noqa: BLE001
            yield pack({"error": str(e)})
            gen_dt = learn_dt = 0.0
        total = max(time.time() - t0, 1e-6)
        model = STATE.get("model")
        # session totals: the split between writing and training is the most
        # actionable number this page produces, and a single turn's figure
        # says nothing about whether it is stable
        STATE["gen_s_total"] = STATE.get("gen_s_total", 0.0) + gen_dt
        STATE["learn_s_total"] = STATE.get("learn_s_total", 0.0) + learn_dt
        done = {"done": True, "n": n, "nonprint": nonprint,
                # generation rate, over the generation window only
                "cps": round(n / (gen_dt or total), 1),
                "gen_s": round(gen_dt, 2),
                # and the training pass separately, because it is a different
                # cost and folding it into the rate hides both
                "learn_s": round(learn_dt, 2),
                "gen_s_total": round(STATE["gen_s_total"], 1),
                "learn_s_total": round(STATE["learn_s_total"], 1),
                "learn": learn_state(),
                "snr": snr_state(), "threshold": threshold}
        if model is not None:
            done["memory"] = memory_report(model, STATE.get("long_threshold", 4.0))
            done["histogram"] = halflife_histogram(
                model, STATE.get("long_threshold", 4.0))
            done["churn"] = hot_churn(model)
        # the reply's own shape, which is the one reading here that can
        # disagree with the CE
        reply_bytes = "".join(produced).encode("utf-8", errors="replace")
        done["degeneracy"] = degeneracy(reply_bytes)
        # ---- the instruments ----
        if obs:
            done["latent"] = project_2d(obs.get("latents") or [])
            done["stops"] = obs.get("stops") or []
            done["heat"] = obs.get("heat") or []
            # The per-byte series are kept to HIST bytes, so on a long reply
            # they are the TAIL and not the whole thing. Saying which window
            # is on screen is the difference between reading a plot of the
            # reply and reading a plot of an unexplained suffix of it.
            cp = obs.get("probs") or []
            # The window is indexed by BYTES, not by `n`: n counts only the
            # printable ones, and the page's per-byte index counts the
            # placeholder it draws for each of the others, so a byte that
            # never made it into `n` is still a column on the plot.
            tot = n + nonprint
            done["conf"] = {"p": cp, "h": obs.get("ents") or [],
                            "t": obs.get("temps") or [], "u": UNSURE_P,
                            "from": max(0, tot - len(cp)), "to": tot}
        if before is not None and model is not None:
            done["delta"] = theta_delta(model, before)
        if learned:
            done["learned"] = learned
        record_session(last_user, "".join(produced), done)
        yield pack(done)

    return sse(events())


# ------------------------------------------------------------------- output
def pack(ev):
    return "data: " + json.dumps(ev) + "\n\n"


def sse(iterator):
    """(status, headers, body-iterator) -- the shape both servers speak."""
    return (200, {"Content-Type": "text/event-stream",
                  "Cache-Control": "no-cache",
                  "X-Accel-Buffering": "no"}, iterator)


_BODY = {}


def read_json():
    return _BODY.get("__json__", {})


# ------------------------------------------------------------------- assets
# The page is three files under web/, not a string in this file.
#
# It used to be one r'''...''' literal, and that was defensible for a
# four-route demo and is not defensible at 1,100 lines: a CSS rule or a JS
# function that cannot be linted, syntax-checked, or diffed on its own is code
# that gets edited carefully. node --check on app.js is now a one-line gate,
# and the browser is no longer the only thing that can tell you a brace is
# missing.
#
# Read per request and cached on mtime, so editing app.js while the server is
# running does not need a restart. The cost is one stat() per asset.
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
_ASSET_CACHE = {}


def asset(name, ctype):
    path = os.path.join(WEB, name)
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        # A missing asset is a broken page, so it says which file is missing
        # rather than 404ing a browser request that looks like a routing bug.
        return (500, {"Content-Type": "text/plain; charset=utf-8"},
                [("web/%s is missing -- the page cannot be served "
                  "without it\n" % name).encode()])
    hit = _ASSET_CACHE.get(name)
    if hit is None or hit[0] != stamp:
        with open(path, "rb") as f:
            hit = (stamp, f.read())
        _ASSET_CACHE[name] = hit
    return (200, {"Content-Type": ctype, "Cache-Control": "no-store"},
            [hit[1]])


def web_source(name):
    """The text of an asset, uncached. For the tests, which assert against the
    page's own source, and for anyone who wants to read it without a browser.
    Deliberately not the cached bytes: a test that reads a stale cache is a
    test that passes against code that is no longer there."""
    with open(os.path.join(WEB, name), encoding="utf-8") as f:
        return f.read()


@route("/")
def index():
    return asset("index.html", "text/html; charset=utf-8")


@route("/panels.css")
def panels_css():
    return asset("panels.css", "text/css; charset=utf-8")


@route("/app.js")
def app_js():
    return asset("app.js", "text/javascript; charset=utf-8")


# -------------------------------------------------------------- flask server
def flask_app():
    from flask import Flask, Response, request
    app = Flask(__name__)

    def make(path, method, fn):
        def view():
            if method == "POST":
                _BODY["__json__"] = request.get_json(force=True) or {}
            else:
                _BODY.pop("__json__", None)
            status, headers, body = fn()
            if hasattr(body, "__iter__") and not isinstance(body, (bytes, list)):
                return Response(body, status=status, headers=headers)
            if isinstance(body, list):
                body = b"".join(body)
            return Response(body, status=status, headers=headers)
        # flask derives the endpoint from the function name by default, and
        # every route here is a closure called `view`, so the default collides
        # on the second one. Name them by path and method instead.
        endpoint = "tmt_%s_%s" % (method.lower(), path.strip("/").replace("/", "_")
                                  or "root")
        app.add_url_rule(path, endpoint, view, methods=[method])

    for (path, method), fn in ROUTES.items():
        make(path, method, fn)
    return app


# ------------------------------------------------------------- stdlib server
def stdlib_app(host, port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):        # the page is the log
            pass

        def _dispatch(self, method):
            fn = ROUTES.get((self.path.split("?")[0], method))
            if fn is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                _BODY["__json__"] = json.loads(raw or b"{}")
            except ValueError:
                _BODY["__json__"] = {}
            status, headers, body = fn()
            streaming = hasattr(body, "__iter__") and not isinstance(body, (bytes, list))
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            if streaming:
                # no Content-Length and a closed connection is what tells the
                # browser this is a stream rather than a document
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    for chunk in body:
                        self.wfile.write(chunk.encode("utf-8")
                                         if isinstance(chunk, str) else chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # the reader hung up (Esc, closed tab). Closing the
                    # generator throws GeneratorExit at its current yield,
                    # which exits the `with LOCK` and frees the model --
                    # without this a cancelled reply would keep generating to
                    # its cap while every later request queued behind it.
                    STATE["cancel"] = True
                return
            if isinstance(body, list):
                body = b"".join(body)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return ThreadingHTTPServer((host, port), Handler)


# ----------------------------------------------------------------- the page
# Same structure as mini-AGI's: header, a panel for the mechanisms, the
# thread, the input. Same care with textContent over innerHTML, same
# follow-only-if-at-the-bottom scroll, same collapse-before-measure textarea.
# The meters are the ones TMT has.


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="chat with TMT and watch its memory while you do")
    ap.add_argument("--path", default="tmt.npz",
                    help="the checkpoint to serve. Learning writes here, "
                         "through the project's tagged/keep-k save path")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-learn", dest="learn", action="store_false",
                    help="serve without learning. Nothing is written back")
    ap.add_argument("--learn-chunk", type=int, default=1024,
                    help="bytes of conversation per optimiser step. The "
                         "trainer's own chunk is tuned for reading a corpus; "
                         "an exchange is often shorter than that, so lower it "
                         "for a short session at the cost of a denser diet "
                         "than documents give")
    ap.add_argument("--save-every", type=int, default=8,
                    help="optimiser steps between writing the checkpoint out")
    ap.add_argument("--keep-k", type=int, default=8,
                    help="tagged checkpoints to keep, same default as the "
                         "trainer")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="hard cap on bytes per reply. The stop head is a "
                         "learned signal and can sit below its threshold "
                         "forever, so this is not optional")
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="stop-head cutoff, the Runtime.chat default")
    ap.add_argument("--prime-bytes", type=int, default=1024,
                    help="bytes of held-out corpus to open the conversation "
                         "with. 0 disables it")
    ap.add_argument("--long-bytes", type=float, default=4.0,
                    help="half-life in bytes above which a dimension counts "
                         "as long memory. The tail-credit hot set sits around "
                         "sigma**18 > 1%%, i.e. roughly 2.7 bytes, so 4 is "
                         "just past it")
    ap.add_argument("--markers", action="store_true",
                    help="tag turns <user>/<bot> instead of feeding raw "
                         "bytes. OFF by default because that is what the "
                         "recorded chat results in RESULTS.md were measured "
                         "on, and this model tends to leak the tags back")
    ap.add_argument("--no-flask", action="store_true",
                    help="force the stdlib server even if flask is installed")
    args = ap.parse_args()

    _MARKERS[0] = bool(args.markers)

    if not os.path.exists(args.path):
        raise SystemExit(
            f"no checkpoint at {args.path}. Train one first:\n"
            f"    .venv/bin/python train_gpu.py {args.path} train --dataset 'data/**/wiki_*'\n"
            f"or bank one:\n"
            f"    .venv/bin/python checkpoint_hub.py pull {args.path}")

    # Reuse Runtime rather than reimplementing the trainer: it owns the model
    # and, more importantly, save_checkpoint() -- the tagged, atomic, keep-k
    # path that exists so a result's checkpoint is never silently destroyed.
    # Talking to the model must not be a way to lose one.
    # Read the shape off the checkpoint rather than assuming the 4.5m config.
    # A checkpoint saved at a different dim or depth is a real thing to have
    # (tools/bench_variants.py exists to make them), and loading it into a
    # 512/16 model would fail deep inside the optimizer with a shape error
    # that says nothing about which parameter was wrong.
    dim, layers = 512, 16
    try:
        with np.load(args.path) as d:
            if "embed" in d.files:
                dim = int(d["embed"].shape[1])
            n_layers = len([k for k in d.files
                            if k.startswith("layer") and k.endswith(".W")])
            if n_layers:
                layers = n_layers
    except Exception as e:                                    # noqa: BLE001
        raise SystemExit(f"cannot read {args.path}: {e}")

    rt = numpy_port.Runtime(path=args.path, save_every=args.save_every,
                            keep_k=args.keep_k, tagged=True,
                            dim=dim, layers=layers)
    model = rt.model
    model.new_states(1)
    model.load(args.path)

    global PRIME
    PRIME = load_prime(args.prime_bytes)
    if args.prime_bytes and not PRIME:
        print("  no corpus to prime from - the model will answer from the "
              "prompt alone", file=sys.stderr)

    # the prime goes into the state NOW, once, and is never fed again. See
    # /api/prime: prepending it to each turn would re-apply a thousand bytes
    # of corpus to the memory every time and bury the conversation.
    if PRIME:
        for c in PRIME.encode("utf-8", errors="replace")[:-1]:
            model.frozen_step(c)
    print(f"[loaded] {args.path} t={model.t} bytes={model.bytes_seen}",
          file=sys.stderr)
    STATE.update(runtime=rt, model=model, path=args.path, learn=args.learn,
                 chunk=max(1, args.learn_chunk), pending=0, steps=0,
                 max_tokens=args.max_tokens, threshold=args.threshold,
                 long_threshold=args.long_bytes, save_every=args.save_every,
                 ema=None, snr=None, loss=None, ce=None,
                 cancel=False, session=[])
    # What this session started from, so an exported transcript says which
    # checkpoint produced it.
    #
    # The FULL sha256, not a first+last-page fingerprint. The fingerprint was
    # an attempt to keep startup fast, and it was wrong on its own terms: a
    # value labelled "digest" that does not match `sha256sum`, and does not
    # match the .sha256 sidecars checkpoint_hub writes, is a number that
    # means something other than what it says -- which is the exact defect
    # this project's bug log is full of. It is also the number a reader will
    # actually check the export filename against. It costs ~0.4s once.
    t_digest = time.time()
    try:
        h = hashlib.sha256()
        with open(args.path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        digest = h.hexdigest()
    except OSError:
        digest = None
    print(f"[digest] {args.path} sha256 {digest} "
          f"({time.time() - t_digest:.2f}s)", file=sys.stderr)
    STATE["meta"] = {"path": os.path.abspath(args.path), "digest": digest,
                     "t": int(model.t), "bytes_seen": int(model.bytes_seen),
                     "dim": int(model.dim), "layers": int(model.layercount),
                     "ring_k": int(model.ring_k),
                     "params": int(model.count()),
                     "learn": bool(args.learn), "chunk": int(STATE["chunk"]),
                     "max_tokens": int(args.max_tokens),
                     "threshold": float(args.threshold),
                     "long_bytes": float(args.long_bytes),
                     "markers": bool(args.markers),
                     "prime_bytes": int(len(PRIME.encode("utf-8"))),
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print(f"  {model.count():,} parameters, dim={model.dim}, "
          f"layers={model.layercount}, ring_k={model.ring_k}")

    if args.learn:
        print(f"\n  LEARNING IS ON. Talking to this model CHANGES {args.path}.\n"
              f"  One step per {STATE['chunk']} bytes; saved to "
              f"{args.path} every {args.save_every} steps, keeping "
              f"{args.keep_k} tags.\n  Use --no-learn to serve it read-only.\n")
    else:
        print("  read-only: nothing will be written back\n")

    if PRIME:
        print(f"  primed with {len(PRIME.encode('utf-8'))} bytes of held-out "
              f"corpus")

    print(f"\n  http://{args.host}:{args.port}\n")

    if not args.no_flask:
        try:
            import flask  # noqa: F401
            app = flask_app()
            print("  [flask]")
            app.run(host=args.host, port=args.port, threaded=True)
            return
        except ImportError:
            pass
    srv = stdlib_app(args.host, args.port)
    print("  [stdlib http.server - flask is not installed]")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if args.learn:
            # do not lose the conversation to a ^C
            try:
                rt.save_checkpoint()
                print(f"\n[saved] {args.path}", file=sys.stderr)
            except Exception as e:                            # noqa: BLE001
                print(f"[save failed] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
