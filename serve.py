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
    "ema_grad": None,
    "ema_grad_sq": None,
    "snr": None,
    "loss": None,
    "ce": None,
    "hot": [],            # last memory_report(), for diffing
}

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


def snr_state():
    """Gradient SNR for the shared trunk, or None before the first step."""
    if STATE.get("snr") is None:
        return None
    return {"snr": round(float(STATE["snr"]), 4),
            "loss": None if STATE.get("loss") is None else round(float(STATE["loss"]), 4),
            "ce": None if STATE.get("ce") is None else round(float(STATE["ce"]), 4),
            "available": _HAVE_TORCH}


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

    `frozen_step` advances the recurrent state and returns the sampled next
    byte plus the stop head. No gradient, no optimiser -- generation must not
    change the weights, or the reply you are watching would be the model
    training on its own half-sentence, and the learning meter would be lying
    about when the step happens.

    The stop head is a learned signal and can sit below the threshold forever,
    so --max-tokens is a hard cap, the same guard Runtime.chat uses. Hitting
    it is reported, not hidden: a reply that stops because the head fired and
    a reply that stops because it ran out of budget are different facts.
    """
    model = STATE["model"]
    b = int(first_byte)
    produced = 0
    capped = False
    for _ in range(int(max_new)):
        b, stop = model.frozen_step(b)
        # A byte-level model emits bytes that are not printable characters,
        # and a few of the 256 are outside the UTF-8 range entirely. Dropping
        # them silently would make the reader see a five-byte reply while the
        # counter said twelve, which is exactly the kind of quiet mismatch this
        # project's history is full of. So they are SENT, flagged, and the page
        # draws a placeholder -- while the reply string stays clean, which
        # matters now that the state, not the prompt, carries the history.
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
                _update_snr(model)
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
            "snr": None if STATE.get("snr") is None else round(float(STATE["snr"]), 4)}


def _update_snr(model):
    """
    SNR over the whole parameter vector, using telemetry's own definition.

    numpy_port's `grads` is a numpy array and `torch.from_numpy` on a float32
    CPU array is zero-copy, so this costs one float32 view rather than a
    4.5M-element conversion per step. The EMAs are torch tensors so the
    definition is literally telemetry.update_gradient_ema's and not a
    second, subtly different one.
    """
    if STATE["ema_grad"] is None:
        if not _HAVE_TORCH:
            return
        n = model.grads.size
        STATE["ema_grad"] = torch.zeros(n, dtype=torch.float32)
        STATE["ema_grad_sq"] = torch.zeros(n, dtype=torch.float32)
    g = torch.from_numpy(np.ascontiguousarray(model.grads, dtype=np.float32))
    _, snr = telemetry.update_gradient_ema(g, STATE["ema_grad"],
                                           STATE["ema_grad_sq"], decay=0.9)
    STATE["snr"] = snr


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
        learned = None
        try:
            with LOCK:
                model = STATE["model"]
                # the prompt bytes go in as state WITHOUT training. A gradient
                # on the prompt would make the reply depend on the optimiser
                # state of a half-fed stream, and would also mean the user
                # message is learned from before it has been answered.
                for c in data[:-1]:
                    model.frozen_step(c)
                STATE["hot"] = memory_report(model, STATE.get("long_threshold", 4.0))["dims"]
                yield pack({"swap": {"at": 0, "moved": None,
                                     "memory": STATE["hot"]}})
                for ev in stream(data[-1], max_new, threshold):
                    if "t" in ev:
                        if ev["t"]:
                            n += 1
                            produced.append(ev["t"])
                        else:
                            nonprint += 1
                    yield pack(ev)
                learned = remember(last_user, "".join(produced),
                                   STATE.get("save_every", 8))
        except Exception as e:                                # noqa: BLE001
            yield pack({"error": str(e)})
        dt = max(time.time() - t0, 1e-6)
        model = STATE.get("model")
        done = {"done": True, "n": n, "nonprint": nonprint,
                "cps": round(n / dt, 1), "learn": learn_state(),
                "snr": snr_state()}
        if model is not None:
            done["memory"] = memory_report(model, STATE.get("long_threshold", 4.0))
        if learned:
            done["learned"] = learned
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
                for chunk in body:
                    self.wfile.write(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
                    self.wfile.flush()
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
  #log{flex:1 1 auto;overflow-y:auto;padding:24px 16px}
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
  #mech{border-bottom:1px solid var(--line);padding:9px 16px;flex:0 0 auto}
  .mrow{max-width:720px;margin:0 auto;display:flex;gap:22px;
        align-items:center;flex-wrap:wrap}
  .meter{flex:1 1 190px;min-width:170px}
  .mlab{font-size:11px;color:var(--dim);display:flex;
        justify-content:space-between;margin-bottom:4px;gap:8px}
  .mlab b{font-weight:600;color:var(--fg);font-variant-numeric:tabular-nums;
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .bar{height:5px;border-radius:3px;background:var(--line);overflow:hidden}
  .fill{height:100%;width:0;border-radius:3px;background:var(--accent);
        transition:width .18s linear}
  .fill.mem{background:var(--ok)}
  .fill.snr{background:var(--warn)}
  @keyframes flare{0%{transform:scale(1);box-shadow:0 0 0 0 var(--glow)}
                   40%{transform:scale(1.03);box-shadow:0 0 0 7px transparent}
                   100%{transform:scale(1);box-shadow:0 0 0 0 transparent}}
  .fired{animation:flare .75s ease-out}
  .note{font-size:11px;color:var(--ok);opacity:0;transition:opacity .2s}
  .note.on{opacity:1}
  .note.swap{color:var(--accent)}
  #chips{max-width:720px;margin:9px auto 0;display:flex;flex-wrap:wrap;gap:3px}
  .chip{font:10px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
        padding:1px 5px;border-radius:3px;border:1px solid var(--line);
        color:var(--dim);transition:background .3s,color .3s,border-color .3s}
  .chip.live{color:var(--ok);border-color:var(--ok)}
  .chip.new{background:var(--accent);border-color:var(--accent);color:#fff}
  .chiplab{font-size:10px;color:var(--dim);max-width:720px;margin:7px auto 0}
  .empty{color:var(--dim);text-align:center;margin-top:18vh;line-height:1.9}
  .empty code{background:var(--user);padding:2px 6px;border-radius:4px;
              font-size:13px}
  footer{border-top:1px solid var(--line);padding:12px 16px;flex:0 0 auto}
  form{max-width:720px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
  textarea{flex:1;resize:none;font:inherit;color:var(--fg);
           background:var(--bg);border:1px solid var(--line);
           border-radius:9px;padding:10px 12px;max-height:180px;overflow-y:auto}
  textarea:focus{outline:none;border-color:var(--accent)}
  .stat{max-width:720px;margin:7px auto 0;font-size:12px;color:var(--dim);
        height:16px;display:flex;justify-content:space-between;gap:12px}
  .hint{color:var(--dim);white-space:nowrap}
</style></head>
<body>
<header>
  <b>TMT</b><span class="sub" id="hdr"></span>
  <span class="sp"></span>
  <button id="reset">New chat</button>
</header>

<div id="mech">
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
      <div class="mlab"><span>gradient SNR</span><b id="snr-n">-</b></div>
      <div class="bar"><div class="fill snr" id="snr-f"></div></div>
    </div>
    <span class="note" id="note"></span>
  </div>
  <div class="chiplab" id="chiplab">long-lived dimensions &mdash; half-life, and whether the state is actually live</div>
  <div id="chips"></div>
</div>

<div id="log"><div class="wrap"><div id="primebox"></div><div id="thread"></div></div></div>

<footer>
  <form id="f">
    <textarea id="q" rows="1" placeholder="Say something"></textarea>
    <button id="send" type="submit">Send</button>
  </form>
  <div class="stat"><span id="stat"></span><span class="hint">Enter to send, Shift+Enter for a newline</span></div>
</footer>

<script>
const thread = document.getElementById('thread');
const log    = document.getElementById('log');
const q      = document.getElementById('q');
const stat   = document.getElementById('stat');
const send   = document.getElementById('send');
let messages = [];
let busy = false;
const $ = id => document.getElementById(id);

// Both counters and the memory report are the SERVER's. The page mirrors
// them per byte so the model's behaviour is visible while it is happening,
// and re-syncs from the server whenever a step or a reply resets something
// it cannot work out on its own.
let MECH = {pending:0, chunk:0, steps:0, ids:[]};

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
      ? left + ' bytes to the next update'
      : (busy ? 'updates when this reply finishes' : 'ready to update');
  $('learn-f').style.width = (100 * Math.min(pending, chunk) / chunk) + '%';
}

// Long memory: the count of dimensions past the half-life threshold, and how
// many of those are actually holding state. A dimension with a 32-byte
// half-life and zero state is not memory, and counting it as such is the
// mistake both plans warn about -- so the two are reported separately.
function drawMem(rep){
  if (!rep) return;
  $('mem-n').textContent = rep.n_long + ' dims > ' + rep.threshold + 'B'
                         + (rep.n_live < rep.n_long
                            ? '  (' + rep.n_live + ' live)' : '');
  $('mem-f').style.width = Math.min(100, 100 * rep.n_live / 8) + '%';
  drawChips(rep);
}

// SNR is the standing blocker: ~0.05 for the whole 889,897-step run. The bar
// is scaled to 1.0 so a healthy signal fills it and 0.05 is visibly nothing.
function drawSnr(s){
  if (!s){ $('snr-n').textContent = 'no step yet'; $('snr-f').style.width='0';
           return; }
  $('snr-n').textContent = s.snr + (s.ce !== null && s.ce !== undefined
                                    ? '   ce ' + s.ce : '');
  $('snr-f').style.width = Math.max(0, Math.min(100, 100 * s.snr)) + '%';
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
    c.title = d.id + ' — half-life ' + d.hl + ' bytes, sigma ' + d.sigma
            + ', state ' + (d.live ? d.rms : '0 (not holding anything)');
    box.appendChild(c);
  }
  if (fresh && fresh.size) setTimeout(() => {
    box.querySelectorAll('.chip.new').forEach(c => c.classList.remove('new'));
  }, 1400);
}

function applyState(d){
  if (d.learn){ MECH.pending = d.learn.pending; MECH.chunk = d.learn.chunk;
                MECH.steps = d.learn.steps; }
  if (d.snr !== undefined) drawSnr(d.snr);
  if (d.memory) drawMem(d.memory);
  drawLearn();
}

fetch('/api/state').then(r => r.json()).then(d => {
  if (d.path) $('hdr').textContent = d.path;
  if (d.markers) $('hdr').textContent += '  ·  turn markers on';
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
                             stat.textContent = ''; q.focus(); };

$('f').onsubmit = async (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text || busy) return;
  busy = true; send.disabled = true;
  q.value = ''; grow();

  add('user', text);
  messages.push({role:'user', content:text});
  const body = add('bot', '');
  body.classList.add('caret');
  log.scrollTop = log.scrollHeight;
  stat.textContent = 'thinking...';

  let reply = '';
  try {
    const r = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages, max_new: 400})
    });
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
          if (d.t) { reply += d.t; body.textContent = reply; }
          else { reply += '\u00b7'; body.textContent = reply; }
          follow(was);
          MECH.pending++; drawLearn();
        }
        else if (d.swap){ if (d.memory) drawMem({dims:d.memory, n_long:d.memory.length,
                              n_live:d.memory.filter(x=>x.live).length,
                              threshold:4, state_rms:0}); }
        else if (d.note){ note(d.note); }
        else if (d.error){ body.classList.add('err');
                           body.textContent = reply + '\n\n[' + d.error + ']'; }
        else if (d.done){
          // the server's counters are authoritative: a step resets pending to
          // something the page cannot work out on its own
          const L = d.learned;
          applyState(d);
          if (L){
            if (L.error) note('learning failed: ' + L.error);
            else if (L.bytes && L.pending === 0){
              flare($('m-learn'));
              note('weight update' + (L.ce !== null && L.ce !== undefined
                                      ? ', ce ' + L.ce : ''));
            }
          }
          let line = d.n + ' bytes, ' + d.cps + '/s'
                   + (d.nonprint ? ' (' + d.nonprint + ' non-printable)' : '');
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
    const gone = String(err).includes('Failed to fetch');
    body.classList.add('err');
    body.textContent = reply + '\n\n[' + (gone
      ? 'lost the server - it stopped while answering. Check the terminal it '
        + 'is running in.' : err) + ']';
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
                 ema_grad=None, ema_grad_sq=None, snr=None, loss=None, ce=None)

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
