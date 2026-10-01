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
import os
import sys
import threading
import time

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
    "loss": None,
    "ce": None,
    "hot": [],            # last memory_report(), for diffing
    "cancel": False,      # set by /api/cancel; checked between generated bytes
    "session": [],        # one record per exchange, for the session export
    "meta": {},           # what the session started from
}

# How many trailing bytes the instrument panels cover. 200 is a bit under the
# 400-byte default cap, so a full reply is usually shown whole.
HIST = 200

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

    Built once out of `model.shapes`, which IS the flat buffer's layout, so
    a group cannot silently drift out of step with the buffer it indexes.
    """
    decay, named = [], {}
    off = 0
    for name, shape in model.shapes:
        size = int(np.prod(shape))
        idx = np.arange(off, off + size, dtype=np.int64)
        off += size
        if name.endswith(".decay"):
            decay.append(idx)
        named.setdefault(_role_of(name), []).append(idx)
    allidx = np.arange(off, dtype=np.int64)
    decay_idx = (np.concatenate(decay) if decay
                 else np.zeros(0, dtype=np.int64))
    return {"decay": decay_idx, "trunk": np.setdiff1d(allidx, decay_idx),
            "named": {k: np.concatenate(v) for k, v in named.items()},
            "n": off}


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
    out = {}
    for role, idx in (("trunk", groups["trunk"]), ("decay", groups["decay"]),
                      ("all", groups["named"] and np.arange(g.size, dtype=np.int64))):
        slot = STATE["ema"].get(role)
        if slot is None or slot[0].numel() != np.size(idx):
            slot = (torch.zeros(np.size(idx), dtype=torch.float32),
                    torch.zeros(np.size(idx), dtype=torch.float32))
            STATE["ema"][role] = slot
        seg = np.ascontiguousarray(g[idx], dtype=np.float32)
        _, snr = telemetry.update_gradient_ema(torch.from_numpy(seg), slot[0],
                                               slot[1], decay=0.9)
        out[role] = snr
    STATE["snr"] = out


def snr_state():
    """Per-role SNR for the meters. None until the first optimiser step."""
    s = STATE.get("snr")
    if s is None:
        return None
    return {"trunk": round(float(s["trunk"]), 4),
            "decay": round(float(s["decay"]), 4),
            # `all` is the figure RESULTS.md quotes (~0.05 over 889,897 steps),
            # so it is the one to hover. loss/ce travel on the learned record
            # instead -- sending them twice is how the two start disagreeing.
            "all": round(float(s["all"]), 4),
            # how many optimiser steps the EMAs have actually seen. One sample
            # gives EXACTLY 0.1 for every group whatever the gradient is, so an
            # early reading of 0.1/0.1 carries no information about which role
            # is better fed -- and the ratio between them looks reassuringly
            # like 100%. The page says so rather than letting the number be
            # read as a verdict.
            "steps": int(STATE["steps"]),
            "warm": bool(STATE["steps"] < 20),
            "ce": None if STATE.get("ce") is None else round(float(STATE["ce"]), 4)}


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

    `observe` advances the recurrent state and returns the sampled byte, the
    stop-head score, and the JEPA latent. No gradient, no optimiser --
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
    for _ in range(int(max_new)):
        if STATE.get("cancel"):
            yield {"note": "stopped at your request, after %d bytes" % produced}
            break
        b, stop, latent = model.observe(b)
        latents.append(np.asarray(latent, dtype=np.float32).copy())
        stops.append(round(float(stop), 5))
        heat.append(state_heat(model))
        # A byte-level model emits bytes that are not printable characters,
        # and a few of the 256 are outside the UTF-8 range entirely. Dropping
        # them silently would make the reader see a five-byte reply while the
        # counter said twelve, which is exactly the kind of quiet mismatch this
        # project's history is full of. So they are SENT, flagged, and the page
        # draws a placeholder -- while the reply string stays clean.
        if 32 <= b < 127 or b in (9, 10, 13):
            yield {"t": chr(b)}
        else:
            yield {"t": "", "b": int(b)}
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
            "heat": heat[-HIST:], "produced": produced}


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
    """
    rt = STATE.get("runtime")
    if rt is None or not STATE.get("learn"):
        return None
    model = STATE["model"]
    if not markers_on():
        text = (user_text or "").strip() + "\n" + (bot_text or "")
    else:
        text = f"{U0}\n{(user_text or '').strip()}\n{U1}\n{B0}\n{bot_text or ''}\n{B1}\n"
    data = text.encode("utf-8", errors="replace")
    if len(data) < 2:
        return None
    steps, last = 0, None
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
            if STATE["pending"] >= STATE["chunk"]:
                # one optimiser step per chunk bytes; the SNR is read off the
                # gradient step_batch already computed, so it costs nothing
                update_snr(model)
                STATE["pending"] = 0
                STATE["steps"] += 1
                if save_every and STATE["steps"] % save_every == 0:
                    rt.save_checkpoint()
    except Exception as e:                                    # noqa: BLE001
        # a failed step must not cost the reply that was already written
        print(f"[learn] failed: {e}", file=sys.stderr)
        return {"error": str(e)}
    return {"bytes": steps, "pending": STATE["pending"],
            "chunk": STATE["chunk"], "total": STATE["steps"],
            "loss": None if STATE.get("loss") is None else round(float(STATE["loss"]), 4),
            "ce": None if STATE.get("ce") is None else round(float(STATE["ce"]), 4),
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
    stop head did.

    Capped so an all-night session cannot grow without bound.
    """
    stops = done.get("stops") or []
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
        "long_dims": (done.get("memory") or {}).get("n_long"),
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
                learned = remember(last_user, "".join(produced),
                                   STATE.get("save_every", 8))
                learn_dt = time.time() - tlearn
        except Exception as e:                                # noqa: BLE001
            yield pack({"error": str(e)})
            gen_dt = learn_dt = 0.0
        total = max(time.time() - t0, 1e-6)
        model = STATE.get("model")
        done = {"done": True, "n": n, "nonprint": nonprint,
                # generation rate, over the generation window only
                "cps": round(n / (gen_dt or total), 1),
                "gen_s": round(gen_dt, 2),
                # and the training pass separately, because it is a different
                # cost and folding it into the rate hides both
                "learn_s": round(learn_dt, 2),
                "learn": learn_state(),
                "snr": snr_state(), "threshold": threshold}
        if model is not None:
            done["memory"] = memory_report(model, STATE.get("long_threshold", 4.0))
        # ---- the instruments ----
        if obs:
            done["latent"] = project_2d(obs.get("latents") or [])
            done["stops"] = obs.get("stops") or []
            done["heat"] = obs.get("heat") or []
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


@route("/")
def index():
    return (200, {"Content-Type": "text/html; charset=utf-8"},
            [PAGE.encode("utf-8")])


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
PAGE = r'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TMT</title>
<style>
  :root{
    --bg:#ffffff; --fg:#1f2328; --dim:#6b7280; --line:#e5e7eb;
    --user:#f3f4f6; --accent:#2563eb; --err:#b91c1c;
    --ok:#15803d; --warn:#b45309; --glow:rgba(37,99,235,.28);
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#0d1117; --fg:#e6edf3; --dim:#8b949e; --line:#21262d;
           --user:#1c2128; --accent:#58a6ff; --err:#f85149;
           --ok:#3fb950; --warn:#d29922; --glow:rgba(88,166,255,.30); }
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       display:flex;flex-direction:column}
  header{border-bottom:1px solid var(--line);padding:10px 16px;
         display:flex;align-items:center;gap:12px;flex:0 0 auto}
  header b{font-weight:600}
  header .sub{color:var(--dim);font-size:12px}
  header .sp{flex:1}
  button{font:inherit;color:var(--fg);background:none;cursor:pointer;
         border:1px solid var(--line);border-radius:6px;padding:5px 11px}
  button:hover{border-color:var(--accent);color:var(--accent)}
  #log{flex:1 1 auto;min-height:0;overflow-y:auto;padding:24px 16px}
  .wrap{max-width:720px;margin:0 auto}
  .msg{margin:0 0 20px}
  .who{font-size:12px;color:var(--dim);margin-bottom:5px}
  .body{white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere}
  .user .body{background:var(--user);padding:10px 13px;border-radius:10px}
  .err{color:var(--err)}
  .caret::after{content:"\258B";color:var(--accent);
                animation:blink 1s step-end infinite}
  @keyframes blink{50%{opacity:0}}
  details.prime{border:1px solid var(--line);border-radius:8px;
                margin:0 0 22px;background:var(--user)}
  details.prime summary{cursor:pointer;padding:8px 12px;font-size:12px;
                        color:var(--dim);list-style:none}
  details.prime summary::-webkit-details-marker{display:none}
  details.prime summary::before{content:"\25B8  ";display:inline-block;
                                transition:transform .15s}
  details.prime[open] summary::before{transform:rotate(90deg)}
  details.prime pre{margin:0;padding:0 12px 12px;white-space:pre-wrap;
                     word-wrap:break-word;overflow-wrap:anywhere;
                     font-size:12px;color:var(--dim);max-height:320px;overflow-y:auto}
  /* Side by side: instruments left, conversation right.
     This is the layout the panels actually want. Stacked, the instrument
     block measured 483px on an 800px screen and had to live behind a scroll
     ceiling, so it was always either cropped or squashing the thread. As a
     column it gets the full height of the window instead, and the meters stay
     pinned while only the instruments scroll -- the live numbers are the
     thing you glance at, the plots are the thing you study. */
  #split{flex:1 1 auto;min-height:0;display:flex;align-items:stretch}
  #mech{flex:0 0 clamp(330px, 44%, 660px);min-width:0;
        border-right:1px solid var(--line);padding:9px 16px 10px;
        display:flex;flex-direction:column;overflow:hidden}
  #mechscroll{flex:1 1 auto;min-height:0;overflow-y:auto;overscroll-behavior:contain}
  #mech.collapsed{display:none}
  /* When the panels are folded, the conversation takes the whole width. */
  /* When folded, a one-line strip keeps the headline numbers visible, so
     folding the panels away does not mean losing the state of the model. */
  #mechbar{border-bottom:1px solid var(--line);padding:6px 16px;flex:0 0 auto;
           font-size:11px;color:var(--dim);display:flex;gap:16px;
           flex-wrap:wrap;font-variant-numeric:tabular-nums}
  #mechbar b{color:var(--fg);font-weight:600}
  #mechbar[hidden]{display:none}
  #chatcol{flex:1 1 auto;min-width:0;display:flex;flex-direction:column}
  .mrow{flex:0 0 auto;display:grid;
        /* 2x2 rather than a flex row: at 563px four meters wrapped 3+1, which
           reads as a layout accident. A grid is 2+2 at every column width. */
        grid-template-columns:1fr 1fr;gap:9px 18px;align-items:center}
  .meter{min-width:0}
  .mlab{font-size:11px;color:var(--dim);display:flex;
        justify-content:space-between;align-items:baseline;gap:8px;
        margin-bottom:4px;white-space:nowrap}
  .mlab span{overflow:hidden;text-overflow:ellipsis}
  .mlab b{font-weight:600;color:var(--fg);font-variant-numeric:tabular-nums;
          white-space:nowrap;flex:0 0 auto}
  .bar{height:5px;border-radius:3px;background:var(--line);overflow:hidden}
  .fill{height:100%;width:0;border-radius:3px;background:var(--accent);
        transition:width .18s linear}
  .fill.mem{background:var(--ok)}
  .fill.snr{background:var(--warn)}
  @keyframes flare{0%{transform:scale(1);box-shadow:0 0 0 0 var(--glow)}
                   40%{transform:scale(1.03);box-shadow:0 0 0 7px transparent}
                   100%{transform:scale(1);box-shadow:0 0 0 0 transparent}}
  .fired{animation:flare .75s ease-out}
  .note{font-size:11px;color:var(--ok);opacity:0;transition:opacity .2s;
        white-space:nowrap}
  .note.on{opacity:1}
  .note.swap{color:var(--accent)}
  #chips{display:flex;flex-wrap:wrap;gap:3px;margin-top:5px}
  .chip{font:10px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
        padding:1px 5px;border-radius:3px;border:1px solid var(--line);
        color:var(--dim);transition:background .3s,color .3s,border-color .3s}
  .chip.live{color:var(--ok);border-color:var(--ok)}
  .chip.new{background:var(--accent);border-color:var(--accent);color:#fff}
  .chiplab{font-size:10px;color:var(--dim);margin:7px 0 0;
           display:flex;justify-content:space-between;gap:12px;
           align-items:baseline}
  /* the instrument panels */
  #instr{max-width:720px}
  .irow{display:flex;gap:18px;flex-wrap:wrap;margin-top:13px}
  .ipanel{flex:1 1 186px;min-width:168px}
  .ipanel.wide{flex:1 1 100%}
  .ihdr{font-size:10px;color:var(--dim);margin-bottom:5px;
        display:flex;justify-content:space-between;gap:10px;
        align-items:baseline}
  .ihdr em{font-style:normal;color:var(--fg);opacity:.7}
  /* Explicit CSS heights, not width:100%/height:auto on a canvas with no
     intrinsic size -- which is what dropping the width/height attributes for
     the a11y labels left behind, so every plot settled on the 300x150 default
     ratio blown up to the column width and ate the whole column. A plot's
     height is a decision, not a side effect of having no intrinsic size. */
  canvas{display:block;width:100%;height:150px;border:1px solid var(--line);
         border-radius:6px;background:var(--bg)}
  canvas#heat{height:178px}
  table#delta{width:100%;border-collapse:collapse;font-size:11px;margin-top:2px}
  table#delta th{text-align:left;font-weight:600;color:var(--dim);
                 padding:2px 10px 4px 0;font-size:10px;text-transform:uppercase;
                 letter-spacing:.03em}
  table#delta td{padding:3px 10px 3px 0;color:var(--fg);
                 font-variant-numeric:tabular-nums;
                 border-top:1px solid var(--line)}
  table#delta td.n{color:var(--dim);text-align:right}
  table#delta td.r{text-align:right}
  table#delta tr.hl td{color:var(--warn);font-weight:600}
  .empty{color:var(--dim);text-align:center;margin-top:18vh;line-height:1.9}
  .empty code{background:var(--user);padding:2px 6px;border-radius:4px;
              font-size:13px}
  /* Narrow, or a short window: the side-by-side is not worth more than the
     conversation, so go back to stacked -- instruments first, capped, and the
     thread keeps the majority of what is left. */
  @media (max-width:900px), (max-height:560px){
    #split{flex-direction:column}
    #mech{flex:0 0 auto;max-height:min(46vh,430px);width:auto;
          border-right:none;border-bottom:1px solid var(--line)}
    #mechscroll{max-height:inherit}
    #chatcol{flex:1 1 auto;min-height:0}
  }
  footer{border-top:1px solid var(--line);padding:12px 16px;flex:0 0 auto}
  form{max-width:720px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
  textarea{flex:1;resize:none;font:inherit;color:var(--fg);
           background:var(--bg);border:1px solid var(--line);
           border-radius:9px;padding:10px 12px;max-height:180px;overflow-y:auto}
  textarea:focus{outline:none;border-color:var(--accent)}
  .stat{max-width:720px;margin:7px auto 0;font-size:12px;color:var(--dim);
        height:16px;display:flex;justify-content:space-between;gap:12px}
  .hint{color:var(--dim);white-space:nowrap}
  button[disabled]{opacity:.4;cursor:default}
  button[disabled]:hover{border-color:var(--line);color:var(--fg)}
  /* the caret and the flare are the only motion on the page; both are
     decorative and both are unreadable as animation */
  @media (prefers-reduced-motion: reduce){
    .caret::after{animation:none;opacity:.7}
    .fired{animation:none}
    .fill{transition:none}
    details.prime summary::before{transition:none}
  }
</style></head>
<body>
<header>
  <b>TMT</b><span class="sub" id="hdr"></span>
  <span class="sub" id="prov"></span>
  <span class="sp"></span>
  <span class="note" id="note"></span>
  <button id="toggle" aria-expanded="true" aria-controls="mech">Hide panels</button>
  <button id="reset">New chat</button>
  <button id="export" disabled title="download this session as JSON — the checkpoint, its digest, and every turn's numbers">Export</button>
</header>

<div id="split">
  <aside id="mech" aria-label="what the model is doing to itself">
  <div class="mrow">
    <div class="meter" id="m-learn">
      <div class="mlab"><span>test-time training</span><b id="learn-n">-</b></div>
      <div class="bar"><div class="fill" id="learn-f"></div></div>
    </div>
    <div class="meter" id="m-mem">
      <div class="mlab"><span>long memory</span><b id="mem-n">-</b></div>
      <div class="bar"><div class="fill mem" id="mem-f"></div></div>
    </div>
    <div class="meter" id="m-snr">
      <div class="mlab"><span>SNR trunk</span><b id="snr-n">-</b></div>
      <div class="bar"><div class="fill snr" id="snr-f"></div></div>
    </div>
    <div class="meter" id="m-dec">
      <div class="mlab"><span>SNR decay</span><b id="dec-n">-</b></div>
      <div class="bar"><div class="fill snr" id="dec-f"></div></div>
    </div>
    </div>
  <div id="mechscroll">
  <div class="chiplab">
    <span>long-lived dimensions &mdash; half-life, and whether the state is actually live</span>
  </div>
  <div id="chips"></div>
  <div id="instr">
    <div class="irow">
      <div class="ipanel">
        <div class="ihdr"><span>JEPA latent, PCA to 2D</span><em id="lat-var"></em></div>
        <canvas id="lat" role="img" aria-label="trajectory of the model's latent vector across the bytes of this reply, projected onto its first two principal components"></canvas>
      </div>
      <div class="ipanel">
        <div class="ihdr"><span>stop head per byte</span><em id="stop-note"></em></div>
        <canvas id="stop" role="img" aria-label="the stop head's score for each byte of this reply, against the cut-off"></canvas>
      </div>
    </div>
    <div class="irow">
      <div class="ipanel wide">
        <div class="ihdr"><span>recurrent state RMS per layer, across the reply</span><em id="heat-note"></em></div>
        <canvas id="heat" role="img" aria-label="the recurrent state's root-mean-square magnitude in each of the sixteen layers, for each byte of this reply"></canvas>
      </div>
    </div>
    <div class="irow">
      <div class="ipanel wide">
        <div class="ihdr"><span>what this exchange changed</span><em id="delta-note"></em></div>
        <table id="delta"></table>
      </div>
    </div>
  </div>
  </div>
  </aside>

  <section id="chatcol" aria-label="conversation">
    <div id="mechbar" hidden></div>
    <div id="log"><div class="wrap"><div id="primebox"></div><div id="thread"></div></div></div>

    <footer>
      <form id="f">
        <textarea id="q" rows="1" placeholder="Say something"></textarea>
        <button id="send" type="submit">Send</button>
      </form>
      <div class="stat"><span id="stat" role="status" aria-live="polite"></span><span class="hint">Enter to send &middot; Shift+Enter newline &middot; Esc to stop</span></div>
    </footer>
  </section>
</div>

<script>
const thread = document.getElementById('thread');
const log    = document.getElementById('log');
const q      = document.getElementById('q');
const stat   = document.getElementById('stat');
const send   = document.getElementById('send');
let messages = [];
let busy = false;
let abortFetch = null;
let CUR = {body: null, reply: '', learning: false, stopped: false};
const $ = id => document.getElementById(id);

// Both counters and the memory report are the SERVER's. The page mirrors
// them per byte so the model's behaviour is visible while it is happening,
// and re-syncs from the server whenever a step or a reply resets something
// it cannot work out on its own.
let MECH = {pending:0, chunk:0, steps:0, ids:[], maxTokens:400,
            threshold:0.35, layers:16, longBytes:4, long: undefined,
            snrTrunk: undefined, snrDecay: undefined};
let LAST = {latent:null, stops:null, heat:null, delta:null};

function note(text, kind){
  const n = $('note');
  n.textContent = text;
  n.className = 'note on' + (kind ? ' ' + kind : '');
  clearTimeout(note._t);
  note._t = setTimeout(() => { n.className = 'note'; }, 2600);
}
function flare(el){ el.classList.remove('fired');
                    void el.offsetWidth;
                    el.classList.add('fired'); }

function drawLearn(){
  const {pending, chunk} = MECH;
  if (!chunk){ $('learn-n').textContent = 'off'; $('learn-f').style.width = '0';
               return; }
  const left = Math.max(0, chunk - pending);
  // the step is taken after the reply finishes, not mid-sentence, so a full
  // bar is true but would read as though nothing were going to happen
  $('learn-n').textContent = left
      ? left + ' B to next update'
      : (busy ? 'updates after this reply' : 'ready to update');
  $('learn-f').style.width = (100 * Math.min(pending, chunk) / chunk) + '%';
}

// Long memory: the count of dimensions past the half-life threshold, and how
// many of those are actually holding state. A dimension with a 32-byte
// half-life and zero state is not memory, and counting it as such is the
// mistake both plans warn about -- so the two are reported separately.
function drawMem(rep){
  if (!rep) return;
  MECH.long = rep.n_long;
  $('mem-n').textContent = rep.n_long + ' dims > ' + rep.threshold + 'B'
                         + (rep.n_live < rep.n_long
                            ? '  (' + rep.n_live + ' live)' : '');
  // the bar is a proportion of a stated ceiling, not a raw count: 8 dims is
  // "every long-lived dimension this checkpoint has ever grown", so the full
  // bar means memory has saturated rather than that there are 8 of them
  $('mem-f').style.width = Math.min(100, 100 * rep.n_live / 8) + '%';
  $('m-mem').title = rep.n_live + ' of ' + rep.n_long + ' dims over '
                  + rep.threshold + ' bytes are holding state; the bar fills '
                  + 'at 8, which is every long-lived dim this checkpoint has '
                  + 'ever grown';
  drawChips(rep);
}

// SNR is the standing blocker and it is now TWO bars, not one. An overall
// ~0.05 is consistent with a trunk that is barely learning AND with a healthy
// trunk beside a decay group that has never had a coherent signal -- and
// numpy_port's decay_init comment argues for the second. Both bars are scaled
// to 1.0, so 0.05 is visibly nothing.
function drawSnr(s){
  const set = (nid, fid, v) => {
    if (v === undefined || v === null){
      $(nid).textContent = 'no step yet'; $(fid).style.width = '0'; return;
    }
    $(nid).textContent = v;
    $(fid).style.width = Math.max(0, Math.min(100, 100 * v)) + '%';
  };
  if (!s){ set('snr-n','snr-f',null); set('dec-n','dec-f',null); return; }
  set('snr-n','snr-f', s.trunk);
  set('dec-n','dec-f', s.decay);
  // Until the EMAs have seen a few dozen steps the two numbers are both
  // pinned near 0.1 by construction and their RATIO is noise that happens to
  // look like "they agree". Marking it is the difference between a warm-up
  // reading and a conclusion about which role the optimiser feeds.
  if (s.warm){
    $('snr-n').textContent = s.trunk + '  warming up (' + s.steps + ' steps)';
    $('dec-n').textContent = s.decay + '  not yet comparable';
  } else {
    const r = (s.decay > 0) ? s.decay / Math.max(s.trunk, 1e-9) : 0;
    $('snr-n').textContent = s.trunk
      + (s.ce !== null && s.ce !== undefined ? '  ce ' + s.ce : '');
    $('dec-n').textContent = s.decay
      + (r > 0 ? '  (' + (r < 0.01 ? '<1%' : Math.round(r * 100) + '%')
                         + ' of trunk)' : '');
  }
  MECH.snrTrunk = s.trunk; MECH.snrDecay = s.decay;
  // `all` is the whole-vector figure RESULTS.md quotes, so it is a hover
  // rather than a third bar -- three bars for two questions is noise.
  $('m-snr').title = 'whole-vector SNR ' + s.all
                  + '  ·  ' + s.steps + ' optimiser steps'
                  + '  ·  ||EMA(g)||^2 / EMA(||g||^2), telemetry.py';
}

function drawChips(rep){
  const seen = new Set(MECH.ids);
  const now = rep.dims.map(d => d.id);
  const fresh = MECH.ids.length ? new Set(now.filter(i => !seen.has(i))) : null;
  MECH.ids = now;
  const box = $('chips');
  box.innerHTML = '';
  for (const d of rep.dims){
    const c = document.createElement('span');
    c.className = 'chip' + (d.live ? ' live' : '')
                + (fresh && fresh.has(d.id) ? ' new' : '');
    c.textContent = d.hl + 'B';
    // colour alone cannot be the only signal, so the same fact is in the
    // text for anyone who cannot see the green and in the label for a
    // screen reader
    c.setAttribute('role', 'img');
    const held = d.live ? 'holding state, RMS ' + d.rms
                        : 'not holding anything';
    c.setAttribute('aria-label', d.id + ': half-life ' + d.hl
                              + ' bytes, ' + held);
    c.title = d.id + ' — half-life ' + d.hl + ' bytes, sigma ' + d.sigma
            + ', state ' + (d.live ? d.rms : '0 (not holding anything)');
    box.appendChild(c);
  }
  if (fresh && fresh.size) setTimeout(() => {
    box.querySelectorAll('.chip.new').forEach(c => c.classList.remove('new'));
  }, 1400);
}

// ------------------------------------------------ the instrument panels
// Three canvases and a table. All of them exist to show a thing the ruler
// cannot: what the latent does, what the stop head does, where the recurrent
// state actually is, and what the weights did. They are drawn on the done
// frame rather than per byte, so a long reply does not repaint 400 times.
const CSSVAR = n => getComputedStyle(document.documentElement)
                           .getPropertyValue(n).trim() || '#888';

function fit(cv){
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || cv.width, h = cv.clientHeight || cv.height;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return {g, w, h};
}

// PCA of the trailing latents. The variance readout is not decoration: a
// scatter whose first two axes hold 1% of the variance is basically noise,
// and drawing it without saying so is how a projection becomes a lie.
function drawLatent(d){
  const cv = $('lat');
  if (!d) return;
  const {g, w, h} = fit(cv);
  const pts = d.points || [];
  $('lat-var').textContent = pts.length
      ? pts.length + ' bytes · axes hold '
        + Math.round(100 * (d.var[0] + d.var[1])) + '% of variance'
      : 'too few bytes';
  if (pts.length < 2) return;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (x1 - x0) > 1e-12 ? (w - 16) / (x1 - x0) : 0;
  const sy = (y1 - y0) > 1e-12 ? (h - 16) / (y1 - y0) : 0;
  const px = p => 8 + (p[0] - x0) * sx, py = p => h - 8 - (p[1] - y0) * sy;
  g.strokeStyle = CSSVAR('--accent'); g.lineWidth = 1.4; g.globalAlpha = .85;
  g.beginPath();
  pts.forEach((p, i) => i ? g.lineTo(px(p), py(p)) : g.moveTo(px(p), py(p)));
  g.stroke();
  // the newest byte, marked -- the rest of the line is the past
  g.globalAlpha = 1; g.fillStyle = CSSVAR('--accent');
  const last = pts[pts.length - 1];
  g.beginPath(); g.arc(px(last), py(last), 3.2, 0, 6.284); g.fill();
}

// the stop head, per byte, against the threshold. This is what turns "the
// 600k checkpoint hit the 2000-token cap" from an anecdote into something
// you can look at.
function drawStop(stops, thr){
  const cv = $('stop');
  const {g, w, h} = fit(cv);
  if (!stops || !stops.length){ $('stop-note').textContent = ''; return; }
  const peak = Math.max(...stops);
  const hi = Math.max(thr * 1.35, peak * 1.1, 0.1);
  const y = v => h - 6 - (v / hi) * (h - 12);
  // the threshold as a rule, not a number you have to imagine
  g.strokeStyle = CSSVAR('--err'); g.lineWidth = 1; g.setLineDash([4, 3]);
  g.beginPath(); g.moveTo(0, y(thr)); g.lineTo(w, y(thr)); g.stroke();
  g.setLineDash([]);
  g.strokeStyle = CSSVAR('--accent'); g.lineWidth = 1.5;
  g.beginPath();
  stops.forEach((v, i) => {
    const px = 6 + (i / Math.max(1, stops.length - 1)) * (w - 12);
    i ? g.lineTo(px, y(v)) : g.moveTo(px, y(v));
  });
  g.stroke();
  const fired = peak > thr;
  $('stop-note').textContent = 'peak ' + peak.toFixed(3) + ' vs cut ' + thr
                             + (fired ? ' · FIRED' : ' · never fired');
  $('stop-note').style.color = fired ? CSSVAR('--err') : '';
}

// per-layer state RMS across the reply. Every layer lights up here even
// though only ~11 dims anywhere remember past 4 bytes -- that CONTRAST is
// the finding: the state is busy everywhere, long memory is almost nowhere.
function drawHeat(heat, nlayers){
  const cv = $('heat');
  const {g, w, h} = fit(cv);
  if (!heat || !heat.length || !nlayers) return;
  const n = heat.length;
  let hi = 0;
  for (const row of heat) for (const v of row) if (v > hi) hi = v;
  hi = hi || 1;
  const cw = w / n, ch = h / nlayers;
  for (let i = 0; i < n; i++)
    for (let L = 0; L < nlayers; L++){
      const v = heat[i][L] / hi;
      g.fillStyle = 'rgba(63,185,80,' + (0.06 + 0.94 * v).toFixed(3) + ')';
      g.fillRect(i * cw, h - (L + 1) * ch, Math.ceil(cw), Math.ceil(ch));
    }
  g.fillStyle = CSSVAR('--dim'); g.font = '10px ui-monospace,Menlo,monospace';
  g.textBaseline = 'middle';
  for (let L = 0; L < nlayers; L += Math.max(1, Math.round(nlayers / 8)))
    g.fillText('L' + L, 2, h - (L + 0.5) * ch);
  $('heat-note').textContent = n + ' bytes · ' + nlayers + ' layers · peak '
                             + hi.toFixed(3) + ' RMS';
}

// what the exchange did to theta. `rel` is the update-to-parameter ratio,
// so the decay row is the one to read: it is the role the queued factorial
// is about to spend quota on.
function drawDelta(rows, learn){
  const t = $('delta');
  const panel = t.closest('.ipanel');
  if (!rows || !rows.length){
    // an empty table under a heading reads as a bug; say WHY it is empty
    t.innerHTML = '';
    panel.style.display = learn ? '' : 'none';
    $('delta-note').textContent = learn ? 'no change recorded' : '';
    return;
  }
  panel.style.display = '';
  const head = '<tr><th>role</th><th class="n">params</th>'
             + '<th class="r">RMS &Delta;&theta;</th>'
             + '<th class="r">&Delta;/&theta;</th></tr>';
  const body = rows.map(r =>
    '<tr' + (r.role === 'decay' ? ' class="hl"' : '') + '>'
    + '<td>' + r.role + '</td>'
    + '<td class="n">' + r.n.toLocaleString() + '</td>'
    + '<td class="r">' + r.rms.toExponential(2) + '</td>'
    + '<td class="r">' + (r.rel * 100).toExponential(2) + '%</td></tr>').join('');
  t.innerHTML = head + body;
  $('delta-note').textContent = learn ? 'update-to-parameter ratio' : '';
}

let MODE = {learn: true};
function applyState(d){
  if (d.learn){ MECH.pending = d.learn.pending; MECH.chunk = d.learn.chunk;
                MECH.steps = d.learn.steps; MECH.learn = true; }
  else if (d.learn === null){ MECH.learn = false; }
  // the SERVER is the authority for the byte cap and the stop threshold.
  // The page used to hardcode max_new, which silently ignored --max-tokens:
  // asking for 70 and getting 400 back is the kind of quiet mismatch that
  // makes a number mean something other than what it says.
  if (d.max_tokens !== undefined) MECH.maxTokens = d.max_tokens;
  if (d.threshold !== undefined) MECH.threshold = d.threshold;
  if (d.meta && d.meta.layers) MECH.layers = d.meta.layers;
  if (d.meta && d.meta.long_bytes) MECH.longBytes = d.meta.long_bytes;
  if (d.snr !== undefined) drawSnr(d.snr);
  if (d.memory) drawMem(d.memory);
  drawLearn();
  if ($('mech').classList.contains('collapsed')) drawStrip();
}

fetch('/api/state').then(r => r.json()).then(d => {
  if (d.path) $('hdr').textContent = d.path;
  if (d.markers) $('hdr').textContent += '  ·  turn markers on';
  if (d.learn === null) MODE.learn = false; else if (d.learn) MODE.learn = true;
  if (d.meta && d.meta.t !== undefined){
    // the digest is ON SCREEN, not only in a tooltip: this tool's whole ethos
    // is that a number belongs to the checkpoint that produced it
    $('prov').textContent = 'sha256 ' + (d.meta.digest || '?').slice(0, 12)
                         + ' · t=' + d.meta.t
                         + ' · ' + (d.meta.params || 0).toLocaleString() + 'p';
    $('hdr').title = d.meta.path + '  ·  bytes=' + d.meta.bytes_seen
                  + '  dim=' + d.meta.dim + '  layers=' + d.meta.layers
                  + '  ring_k=' + d.meta.ring_k
                  + '  prime=' + d.meta.prime_bytes + 'B';
  }
  applyState(d);
}).catch(() => {});

// what the model is answering from, besides what you typed. It is in the
// state before the first message and is not part of any turn, so it is shown
// here rather than left to be taken on trust.
fetch('/api/prime').then(r => r.json()).then(p => {
  if (!p.chars) return;
  const d = document.createElement('details');
  d.className = 'prime';
  d.innerHTML = '<summary></summary><pre></pre>';
  d.querySelector('summary').textContent =
    'primed with ' + p.chars + ' bytes of held-out corpus, fed into the '
    + 'state once at startup before your first message';
  d.querySelector('pre').textContent = p.text;
  $('primebox').appendChild(d);
}).catch(() => {});

// The panels are tall and the conversation is the thing. Fold them away, keep
// the headline numbers in a one-line strip so the model's state is still
// readable at a glance, and remember the choice. Default: folded on a short
// viewport, because that is where they were eating the thread.
const MECH_KEY = 'tmt.mech';
function drawStrip(){
  // the strip mirrors the meters, so folding the panels away does not lose
  // the numbers. Called from applyState as well as from applyMech, because
  // the first render happens before /api/state has answered and would
  // otherwise sit there claiming the model was read-only.
  const parts = [];
  parts.push('training <b>' + (MECH.learn
      ? (MECH.pending + '/' + MECH.chunk + ' B to next step')
      : 'read-only') + '</b>');
  if (MECH.long !== undefined)
    parts.push('long memory <b>' + MECH.long + ' dims</b>');
  if (MECH.snrTrunk !== undefined)
    parts.push('SNR <b>' + MECH.snrTrunk + '</b> / <b>' + MECH.snrDecay + '</b>');
  $('mechbar').innerHTML = parts.join(' &middot; ');
}
function applyMech(collapsed){
  $('mech').classList.toggle('collapsed', collapsed);
  $('mechbar').hidden = !collapsed;
  $('toggle').textContent = collapsed ? 'Show panels' : 'Hide panels';
  $('toggle').setAttribute('aria-expanded', String(!collapsed));
  // the split is the default at any width that can hold two columns; folding
  // is the way to buy the whole window for the conversation, so the old
  // "fold on a short viewport" default only applies once the layout has
  // already fallen back to stacked
  if (collapsed) drawStrip();
  try { localStorage.setItem(MECH_KEY, collapsed ? '1' : '0'); } catch (e) {}
  // the canvases are sized off the panel being visible, so they have to be
  // redrawn after the fold or they come back at the wrong width
  if (!collapsed) requestAnimationFrame(redrawAll);
}
$('toggle').onclick = () =>
  applyMech(!$('mech').classList.contains('collapsed'));
function redrawAll(){
  if (LAST.latent) drawLatent(LAST.latent);
  if (LAST.stops) drawStop(LAST.stops, MECH.threshold);
  if (LAST.heat) drawHeat(LAST.heat, MECH.layers);
  if (LAST.delta) drawDelta(LAST.delta, MODE.learn);
}
// The canvases are sized from clientWidth, so anything that changes the
// column width leaves them stretched or blank. That is now three things --
// folding, a window resize, and crossing the narrow-viewport breakpoint --
// so it is handled once, here, rather than at each call site.
let resizeTimer = null;
addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(redrawAll, 120);
});
// read the saved preference, else fold it only when the layout has already
// fallen back to stacked and the panels would eat the thread again
let saved = null;
try { saved = localStorage.getItem(MECH_KEY); } catch (e) {}
const stacked = () => matchMedia('(max-width:900px), (max-height:560px)').matches;
applyMech(saved === null ? stacked() : saved === '1');
if (stacked()) matchMedia('(max-width:900px), (max-height:560px)')
  .addEventListener('change', () => redrawAll());

function empty(){
  thread.innerHTML = '<div class="empty">Ask it something.<br>'
    + 'It is 4.5M parameters and mid-training, and it will show.</div>';
}
function add(role, text){
  if (thread.querySelector('.empty')) thread.innerHTML = '';
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = '<div class="who"></div><div class="body"></div>';
  d.querySelector('.who').textContent = role === 'user' ? 'You' : 'TMT';
  d.querySelector('.body').textContent = text || '';
  thread.appendChild(d);
  return d.querySelector('.body');
}
// Only follow the text down while the reader is already at the bottom, so
// scrolling up to re-read something is not undone by the next byte.
function atBottom(){ return log.scrollHeight - log.scrollTop - log.clientHeight < 60; }
function follow(was){ if (was) log.scrollTop = log.scrollHeight; }
// Collapse to zero before measuring: 'auto' does not shrink a textarea
// inside a flex row, so scrollHeight reports the height it already has and
// the box only ever grows.
function grow(){ q.style.height = '0px';
                 q.style.height = Math.min(q.scrollHeight, 180) + 'px'; }
q.addEventListener('input', grow);
q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault();
                                         document.getElementById('f').requestSubmit(); }
});
$('reset').onclick = () => { if (busy) return; messages = []; empty();
                             stat.textContent = '';
                             LAST = {latent:null, stops:null, heat:null, delta:null};
                             for (const id of ['lat','stop','heat'])
                               $(id).getContext('2d').clearRect(0,0,$(id).width,$(id).height);
                             $('lat-var').textContent = '';
                             $('stop-note').textContent = '';
                             $('heat-note').textContent = '';
                             drawDelta(null, MODE.learn);
                             $('export').disabled = true;
                             $('export').textContent = 'Export';
                             q.focus(); };

// Export the session as JSON. Every number this page produces is one the
// project would otherwise throw away, and RESULTS.md's rule is that a number
// without its checkpoint, its ruler setting and its byte count is not a
// result -- so the export carries all three.
$('export').onclick = async () => {
  try {
    const r = await fetch('/api/session');
    const j = await r.json();
    const blob = new Blob([JSON.stringify(j, null, 2)],
                          {type: 'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'tmt-session-' + (j.meta.digest || 'nodigest').slice(0, 16)
                + '.json';
    a.click();
    URL.revokeObjectURL(a.href);
    $('export').textContent = 'saved ' + j.turns.length;
    setTimeout(() => { $('export').textContent = 'Export'; }, 2500);
  } catch (e) { note('export failed: ' + e); }
};

// Esc stops the reply. The server is also told, because aborting the fetch
// alone would leave it generating to its cap while every later request
// queued behind the lock. The current reply is kept on CUR so this handler,
// which lives outside the submit closure, can see it.
//
// It CANNOT stop the learning pass. Once remember() has begun the weights are
// already moving and that pass runs to its end regardless -- measured: a
// cancel sent inside that window still took a step and advanced t. So the
// status says so instead of implying the turn was halted.
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape' || !busy) return;
  e.preventDefault();
  if (CUR.learning){
    stat.textContent = 'the reply is done; the exchange is still being '
                     + 'trained on and cannot be interrupted now'
                     + (MECH.chunk ? ' (' + MECH.pending + '/' + MECH.chunk
                                    + ' B to the next step)' : '');
    note('too late to stop — already training');
    return;
  }
  if (abortFetch) abortFetch.abort();
  fetch('/api/cancel', {method: 'POST'}).catch(() => {});
  CUR.stopped = true;
  if (CUR.body){
    CUR.body.classList.remove('caret');
    CUR.body.textContent = CUR.reply + '\n\n[stopped]';
  }
  stat.textContent = 'stopped after ' + CUR.reply.length + ' bytes';
  note('stopped');
  busy = false; send.disabled = false;
  q.focus();
});

$('f').onsubmit = async (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text || busy) return;
  busy = true; send.disabled = true;
  q.value = ''; grow();

  add('user', text);
  messages.push({role:'user', content:text});
  const body = add('bot', '');
  CUR = {body: body, reply: '', learning: false, stopped: false};
  body.classList.add('caret');
  log.scrollTop = log.scrollHeight;
  stat.textContent = 'thinking...';

  let reply = '';
  abortFetch = new AbortController();
  try {
    const r = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      signal: abortFetch.signal,
      body: JSON.stringify({messages, max_new: MECH.maxTokens,
                            threshold: MECH.threshold})
    });
    // A 500 comes back as an HTML error page, which is not SSE: every frame
    // check below would silently discard it and the turn would look like it
    // produced nothing. Check the status before trying to read a stream.
    if (!r.ok){
      throw new Error('server returned ' + r.status
                      + (r.status === 404 ? ' (no checkpoint loaded?)' : ''));
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for(;;){
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      // SSE frames are separated by a blank line, and a frame can arrive
      // split across two reads, so anything after the last separator stays
      // in buf.
      const frames = buf.split('\n\n');
      buf = frames.pop();
      for (const f of frames){
        if (!f.startsWith('data: ')) continue;
        const d = JSON.parse(f.slice(6));
        if (d.t !== undefined){
          const was = atBottom();
          // a byte that is not a printable character still counts, and is
          // drawn as a placeholder -- the reader should never see a shorter
          // reply than the model produced without being told why
          reply += (d.t || '\u00b7');
          // CUR.reply is what the Esc handler shows, so it has to track the
          // same string -- when it did not, Esc wiped the whole reply and left
          // only "[stopped]", which is destroying the work rather than
          // stopping it
          CUR.reply = reply;
          body.textContent = reply;
          follow(was);
          MECH.pending++; drawLearn();
        }
        else if (d.swap){
          // the swap frame carries ONLY a dim list. drawChips needs nothing
          // else, so nothing else is invented here: the earlier version
          // passed n_long/n_live derived from the list length under a
          // hardcoded threshold, which is a count the server never sent.
          if (d.memory) drawChips({dims: d.memory, threshold: MECH.longBytes});
        }
        else if (d.learning){
          // the bytes are done; the weights are not, and this is the window
          // in which Esc can no longer stop anything
          CUR.learning = true;
          body.classList.remove('caret');
          stat.textContent = 'reply done — training on the exchange...';
        }
        else if (d.note){ note(d.note); }
        else if (d.error){ body.classList.add('err');
                           body.textContent = reply + '\n\n[' + d.error + ']'; }
        else if (d.done){
          // the server's counters are authoritative: a step resets pending to
          // something the page cannot work out on its own
          const L = d.learned;
          applyState(d);
          // kept so folding the panels away and back can redraw them: a
          // canvas sized while its panel was display:none comes back blank
          LAST.stops = d.stops; LAST.latent = d.latent;
          LAST.heat = d.heat;   LAST.delta = d.delta;
          drawStop(d.stops, d.threshold);
          drawLatent(d.latent);
          drawHeat(d.heat, MECH.layers);
          drawDelta(d.delta, MODE.learn);
          $('export').disabled = false;
          if (L){
            if (L.error) note('learning failed: ' + L.error);
            else if (L.bytes && L.pending === 0){
              flare($('m-learn'));
              note('weight update' + (L.ce !== null && L.ce !== undefined
                                      ? ', ce ' + L.ce : ''));
            }
          }
          let line = d.n + ' bytes, ' + d.cps + '/s writing'
                   + (d.nonprint ? ' (' + d.nonprint + ' non-printable)' : '')
                   + (d.gen_s !== undefined ? ' in ' + d.gen_s + 's' : '');
          if (d.learn_s !== undefined && d.learn_s > 0.05)
            line += '  ·  +' + d.learn_s + 's learning on it';
          if (!L) line += '  ·  not learning';
          else if (L.error) line += '  ·  learning failed';
          else line += '  ·  ' + (L.pending || 0) + '/' + L.chunk
                    + ' bytes to the next step  ·  ' + L.total + ' total';
          stat.textContent = line;
        }
      }
    }
  } catch (err) {
    // a dropped connection is the server going away, not the model failing;
    // the browser reports both as "Failed to fetch", so say which is likely
    const gone = String(err).includes('Failed to fetch')
              || String(err).includes('aborted');
    body.classList.add('err');
    if (!gone) body.textContent = reply + '\n\n[' + err + ']';
    // The Esc path was fixed so it could not leave the status reading
    // "thinking..."; the transport-drop path had the same failure and was
    // left with it. A dropped connection is the SERVER going away -- often a
    // training run holding the card -- so say that rather than leaving a
    // spinner nobody can stop.
    if (gone && !CUR.stopped){
      stat.textContent = 'connection lost — is the server still running?'
                       + (reply ? ' (' + reply.length + ' bytes kept)' : '');
    }
  }
  body.classList.remove('caret');
  // Keep the turn for the DISPLAY. The server only reads the last user turn
  // each request -- the model's own replies live in its recurrent state, not
  // in the prompt -- so this history is for you, not for it.
  messages.push({role:'bot', content:reply});
  busy = false; send.disabled = false; q.focus();
};

empty(); q.focus();
// after layout, not during it: at the end of the script the flex row has no
// resolved width, so the placeholder wraps and scrollHeight reports a tall box
requestAnimationFrame(grow);
</script>
</body></html>'''


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
