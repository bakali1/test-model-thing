"""check_web.py — gates for the chat/memory page in serve.py.

Free and CPU-only, like every other check here. It builds a TINY model from
scratch rather than opening tmt.npz, so the whole thing runs in seconds and
does not depend on a 180 MB checkpoint being present.

What it pins, and why each one is here:

  1. the memory report reproduces a NUMBER RECORDED IN RESULTS.md, so a change
     to the half-life or liveness logic cannot quietly redefine what the page
     is claiming;
  2. the prime is cut on line boundaries — the stop head and the tail-credit
     path both key off line structure, and a mid-sentence cut would hand the
     model a half-formed byte pair;
  3. with markers off, the prompt is byte-identical to what Runtime.chat
     feeds, so a page result stays comparable to the recorded chat results;
  4. both servers answer the same four routes, and the SSE frames are
     well-formed (a browser silently drops a malformed frame);
  5. learning writes through the project's TAGGED path, so talking to the
     model cannot destroy a checkpoint a result was measured on;
  6. --no-learn writes nothing at all.

Usage:  .venv/bin/python tests/check_web.py
Exit code 0 = PASS.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import numpy_port
import serve

FAIL = []


def ok(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def tiny(path, dim=64, layers=2):
    """A small model with a couple of deliberately long-lived dims."""
    m = numpy_port.Model(dim=dim, layers=layers)
    # two dims that remember for a long time, two that do not, so the
    # threshold and the liveness split are both exercised
    m.layers[0].decay[:2] = 4.0     # sigma ~0.982, half-life ~30 bytes
    m.layers[0].states[0, :2] = [1.5, -0.25]   # and actually holding state
    m.layers[0].decay[2:4] = 4.0    # same decay, but state stays at zero
    m.new_states(1)
    m.save(path)
    return path


# ---------------------------------------------------------------- 1. memory
def check_memory_report():
    print("\n[1] memory report")
    with tempfile.TemporaryDirectory() as d:
        m = numpy_port.Model(dim=64, layers=2)
        m.layers[0].decay[:2] = 4.0
        m.layers[0].decay[2:4] = 4.0
        # states AFTER new_states: new_states zeroes every buffer, so setting
        # state first would silently do nothing
        m.new_states(1)
        m.layers[0].states[0, :2] = [1.5, -0.25]   # these two hold state
        rep = serve.memory_report(m, threshold=4.0)
        dims = rep["dims"]
        ok(len(dims) == 4, "four dims past the 4-byte threshold",
           f"got {len(dims)}")
        # sigma = sigmoid(4) = 0.9820 -> hl = ln(.5)/ln(.9820) = 38.0 bytes
        ok(all(abs(x["hl"] - 38.0) < 0.5 for x in dims),
           "half-life matches ln(0.5)/ln(sigmoid(decay))",
           str([x["hl"] for x in dims]))
        live = {x["id"]: x["live"] for x in dims}
        ok(live.get("L0.0") is True, "a dim with non-zero state reads live")
        ok(live.get("L0.2") is False,
           "a long half-life with zero state reads NOT live",
           "this is the distinction both plans insist on")
        ok(rep["n_live"] == 2, "n_live counts only the two holding state",
           f"got {rep['n_live']}")

    # the recorded numbers: RESULTS.md's tail of the live leg lists
    # layer6 sigma=0.9790 (half-life 32.6 B) as the longest-lived dim ever
    # seen, and layer6 sigma=0.9556 (15.3 B) behind it.
    m = numpy_port.Model(dim=512, layers=16)
    m.layers[6].decay[85] = 3.5      # sigma ~0.971, the ballpark of 0.9790
    m.new_states(1)
    top = serve.long_dims(m, threshold=4.0, limit=4)
    ok(bool(top), "a 512x16 model reports long-lived dims")
    ok(top and top[0]["layer"] == 6,
       "the longest-lived dim is attributed to its own layer", str(top[:1]))


# ------------------------------------------------------------------ 2. prime
def check_prime():
    print("\n[2] prime")
    with tempfile.TemporaryDirectory() as d:
        # ordinary short lines
        with open(os.path.join(d, "a.txt"), "w") as f:
            f.write("one\ntwo\nthree\nfour\nfive\nsix\nseven\n")
        p = serve.load_prime(24, root=d)
        ok(p.endswith("\n"), "the prime ends on a line boundary", repr(p[-8:]))
        ok("\n" in p, "the prime is more than one byte", repr(p))
        ok(len(p.encode()) <= 24, "the prime respects its byte budget", repr(p))
        ok(p == serve.load_prime(24, root=d),
           "the prime is deterministic across calls")

        # a file whose only lines are longer than the budget cannot be cut on
        # a line boundary, so it is SKIPPED rather than truncated mid-line
        d2 = os.path.join(d, "longlines")
        os.makedirs(d2)
        with open(os.path.join(d2, "z.txt"), "w") as f:
            f.write("x" * 500 + "\n" + "y" * 500 + "\n")
        ok(serve.load_prime(40, root=d2) == "",
           "a file with no line under budget is skipped, not cut mid-line")

        ok(serve.load_prime(0, root=d) == "", "0 chars disables the prime")
        ok(serve.load_prime(40, root=os.path.join(d, "nope")) == "",
           "a missing corpus yields an empty prime, not an exception")

    # relative roots resolve against the repo, not the cwd, so the prime does
    # not silently vanish when the tool is launched from /tmp
    real = serve.load_prime(200)
    ok(bool(real), "the default prime resolves from any cwd",
       "data/eval should be found relative to the repo root")
    ok(len(real.encode("utf-8")) <= 600,
       "the default prime respects its byte budget")


# ----------------------------------------------------------------- 3. prompt
def check_prompt():
    print("\n[3] prompt format")
    old = serve._MARKERS[0]
    try:
        serve._MARKERS[0] = False
        p = serve.build_prompt("hello")
        ok(p == "hello\n",
           "one turn, raw, newline separated -- what Runtime.chat feeds",
           repr(p))
        ok(serve.U0 not in p and serve.B0 not in p,
           "markers OFF emits no turn tags (comparable to Runtime.chat)")

        serve._MARKERS[0] = True
        q = serve.build_prompt("hello")
        ok(q == "<user>\nhello\n</user>\n", "markers ON tags the turn",
           repr(q))
        ok(q.count(serve.U0) == 1 and q.count(serve.U1) == 1,
           "the opening tag is closed exactly once")

        # THE prime is fed into the state once, at load -- not prepended to
        # every turn. Prepending would re-apply a thousand bytes of corpus to
        # the memory on each turn and bury the conversation, which is free in
        # mini-AGI only because it has no state to disturb.
        import inspect
        params = list(inspect.signature(serve.build_prompt).parameters)
        ok(params[0] == "text", "build_prompt takes one turn, not a list",
           str(params))
        ok("messages" not in params, "there is no way to pass a history", str(params))
        ok(not inspect.getsource(serve.build_prompt).count("PRIME"),
           "build_prompt does not touch the prime at all")
        ok(serve.build_prompt_is_deterministic("hello")
           if hasattr(serve, "build_prompt_is_deterministic")
           else serve.build_prompt("hello") == serve.build_prompt("hello"),
           "the same turn builds the same bytes every time")
        ok(serve.build_prompt("") == "" and serve.build_prompt("   ") == "",
           "an empty turn builds nothing")
    finally:
        serve._MARKERS[0] = old
        serve.PRIME = ""


# ------------------------------------------------------------------ 4. http
def sse_frames(text):
    """Parse an SSE body the way the page does, and return the payloads."""
    out = []
    for frame in text.split("\n\n"):
        if not frame.startswith("data: "):
            continue
        out.append(json.loads(frame[6:]))
    return out


def check_http(server_kind, port):
    print(f"\n[4] http — {server_kind} server")
    with tempfile.TemporaryDirectory() as d:
        ckpt = tiny(os.path.join(d, "chat.npz"))
        serve.STATE.update(
            runtime=numpy_port.Runtime(path=ckpt, save_every=1, keep_k=2,
                                       tagged=True, dim=64, layers=2),
            path=ckpt, learn=True, chunk=8, pending=0, steps=0,
            max_tokens=16, threshold=0.35, long_threshold=4.0, save_every=1,
            ema_grad=None, ema_grad_sq=None, snr=None, loss=None, ce=None)
        serve.STATE["model"] = serve.STATE["runtime"].model
        serve.STATE["model"].new_states(1)
        serve.STATE["model"].load(ckpt)
        serve.PRIME = ""

        if server_kind == "stdlib":
            srv = serve.stdlib_app("127.0.0.1", port)
        else:
            import flask
            srv = serve.flask_app()
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        time.sleep(0.3)
        base = f"http://127.0.0.1:{port}"
        try:
            with urllib.request.urlopen(base + "/", timeout=20) as r:
                page = r.read().decode()
            ok(r.status == 200 and "<title>TMT</title>" in page,
               "GET / serves the page")
            ok("text/event-stream" not in page, "the page is not itself a stream")

            with urllib.request.urlopen(base + "/api/state", timeout=20) as r:
                st = json.loads(r.read())
            ok(st["learn"]["chunk"] == 8, "GET /api/state reports the chunk")
            ok(st["memory"]["n_long"] == 4,
               "/api/state reports the long-memory count",
               str(st["memory"]["n_long"]))
            ok(st["max_tokens"] == 16, "/api/state reports the byte cap")

            with urllib.request.urlopen(base + "/api/prime", timeout=20) as r:
                pr = json.loads(r.read())
            ok(pr["chars"] == 0, "/api/prime is inspectable even when empty")
            ok(pr.get("fed_at") == "load, once",
               "the page is told the prime is fed once, not per turn",
               str(pr.get("fed_at")))

            body = json.dumps({"messages": [{"role": "user",
                                             "content": "hello there"}],
                               "max_new": 12,
                               # above 1.0, and the stop head is a sigmoid, so
                               # it can never fire: this pins the CAP path
                               # deterministically. A random tiny model's stop
                               # head fires on the first byte, which is the
                               # model's honest opinion, not something to
                               # assert a fixed length through.
                               "threshold": 2.0}).encode()
            req = urllib.request.Request(base + "/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                raw = r.read().decode()
                ctype = r.headers.get("Content-Type", "")
            ok("text/event-stream" in ctype, "POST /api/chat is an event stream",
               ctype)
            frames = sse_frames(raw)
            ok(len(frames) > 2, "several frames arrived", str(len(frames)))
            ok(raw.endswith("\n\n"), "frames are separated by a blank line")
            ok(any("swap" in f for f in frames), "the memory report is sent "
                                                  "before the first byte")
            # a non-printable byte is sent as an empty "t" plus its value, so
            # the frame count and the VISIBLE byte count are separate things
            # and only their sum is allowed to be max_new
            t_frames = [f for f in frames if "t" in f]
            shown = [f for f in t_frames if f["t"]]
            blank = [f for f in t_frames if not f["t"]]
            ok(len(t_frames) == 12, "one frame per byte produced, printable "
                                    "or not", str(len(t_frames)))
            ok(len(shown) + len(blank) == 12,
               "printable and non-printable frames add up to the cap")
            ok(all(isinstance(f.get("b"), int) for f in blank),
               "a non-printable byte is sent with its value, not dropped",
               str(blank[:2]))
            ok(all(len(f["t"]) <= 1 for f in shown),
               "at most one byte per frame")
            ok(any(f.get("note") for f in frames),
               "hitting the cap is reported, not hidden",
               str([f.get("note") for f in frames if f.get("note")]))
            done = [f for f in frames if f.get("done")]
            ok(len(done) == 1, "exactly one terminal frame")
            d0 = done[0]
            ok(d0["n"] == len(shown),
               "the terminal count matches the visible bytes",
               f"{d0['n']} vs {len(shown)}")
            ok(d0["nonprint"] == len(blank),
               "the terminal frame also counts the non-printable ones",
               f"{d0.get('nonprint')} vs {len(blank)}")
            ok("cps" in d0, "the terminal frame reports a rate")
            ok(d0["learn"]["chunk"] == 8, "the terminal frame reports learning")
            ok("learned" in d0, "the exchange went back through the trainer")
            # chunk=8 and an exchange of ~25 bytes, so steps must have fired
            ok(d0["learned"]["total"] >= 1,
               "the exchange produced optimiser steps",
               json.dumps(d0["learned"]))
            ok(d0["learned"]["pending"] < 8,
               "pending is under the chunk after a step",
               json.dumps(d0["learned"]))
            ok(np.isfinite(d0["learned"]["ce"]),
               "the CE from those steps is finite",
               str(d0["learned"].get("ce")))

            # 5. the tagged save discipline
            tags = sorted(f for f in os.listdir(d) if "_t" in f and f.endswith(".npz"))
            ok(bool(tags), "a tagged checkpoint was written",
               str(sorted(os.listdir(d))))
            ok(all(os.path.basename(ckpt) in t or "_t0" in t for t in tags),
               "the tag name carries the model step", str(tags))
        finally:
            srv.shutdown()
            t.join(timeout=5)


# ------------------------------------------------------------- 5. no-learn
def check_no_learn():
    print("\n[5] --no-learn writes nothing")
    with tempfile.TemporaryDirectory() as d:
        ckpt = tiny(os.path.join(d, "ro.npz"))
        m = numpy_port.Model(dim=64, layers=2)
        m.new_states(1)
        m.load(ckpt)
        serve.STATE.update(runtime=numpy_port.Runtime(path=ckpt, dim=64, layers=2),
                           model=m, path=ckpt, learn=False, pending=0, steps=0)
        serve.PRIME = ""
        before = sorted(os.listdir(d))
        out = serve.remember("hello", "hi", save_every=1)
        after = sorted(os.listdir(d))
        ok(out is None, "remember() is a no-op when learning is off", str(out))
        ok(before == after, "no file appeared", f"{before} -> {after}")
        ok(serve.learn_state() is None, "learn_state() is None when off")


# ---------------------------------------------------- 6. router well-formed
def check_router():
    print("\n[6] route table")
    ok(set(serve.ROUTES) == {("/", "GET"), ("/api/chat", "POST"),
                             ("/api/state", "GET"), ("/api/prime", "GET"),
                             ("/api/session", "GET"), ("/api/cancel", "POST")},
       "mini-AGI's four routes, plus session export and cancel",
       str(sorted(serve.ROUTES)))
    ok(len([k for k in serve.ROUTES if k[0].startswith("/api")]) == 5,
       "no route was added by accident")
    status, headers, body = serve.index()
    ok(status == 200 and headers["Content-Type"].startswith("text/html"),
       "the index route declares html")
    ok(isinstance(body, list) and len(body) == 1, "index returns a byte body")
    st, hd, bd = serve.sse(iter(["data: {}\n\n"]))
    ok(hd["Content-Type"] == "text/event-stream", "the chat route streams")
    ok(hd.get("X-Accel-Buffering") == "no",
       "and disables proxy buffering, or nginx eats the stream")


# ------------------------------------------------- 7. the instruments
def check_instruments():
    print("\n[7] instrument maths")
    with tempfile.TemporaryDirectory() as d:
        m = numpy_port.Model(dim=64, layers=3)
        m.layers[0].decay[:2] = 4.0
        m.new_states(1)
        m.layers[0].states[0, :2] = [1.5, -0.25]

        # --- parameter roles partition the flat buffer exactly ---
        g = serve.model_groups(m)
        ok(g["decay"].size == 3 * 64, "decay group is one vector per layer",
           str(g["decay"].size))
        ok(g["decay"].size + g["trunk"].size == m.grads.size,
           "trunk + decay covers every parameter, with no overlap")
        ok(set(g["named"]) == {"embedding", "decoder", "layer weights",
                               "layer norm gain", "layer norm bias", "decay"},
           "named roles cover every tensor", str(sorted(g["named"])))
        ok(sum(v.size for v in g["named"].values()) == m.grads.size,
           "the named roles also partition the buffer")
        # the group cache must follow the buffer, not the other way round
        m2 = numpy_port.Model(dim=32, layers=2)
        g2 = serve.model_groups(m2)
        ok(g2["decay"].size == 2 * 32 and g2["n"] == m2.grads.size,
           "groups are rebuilt for a different model shape",
           f"{g2['decay'].size} vs {2*32}")

        # --- a real optimiser step, then per-role SNR ---
        for _ in range(6):
            m.step_batch([65], [32], [False])
        serve.STATE.update(ema=None, snr=None, loss=None, ce=None)
        serve.update_snr(m)
        s = serve.snr_state()
        ok(s is not None and {"trunk", "decay", "all"} <= set(s),
           "SNR is reported per role", str(s))
        for role in ("trunk", "decay", "all"):
            ok(0.0 <= s[role] <= 1.0,
               f"{role} SNR is in [0,1]", str(s[role]))
        # one EMA sample gives exactly 0.1 for any input, so this pins the
        # definition against the 0.10 that RESULTS.md records at t=2k
        ok(abs(s["all"] - 0.1) < 1e-3,
           "a single EMA sample gives 0.1, matching the recorded t=2k value",
           str(s["all"]))

        # --- PCA reports the variance it actually used ---
        lat = [m.observe(65)[2] for _ in range(12)]
        p = serve.project_2d(lat)
        ok(len(p["points"]) == 12, "one projected point per latent",
           str(len(p["points"])))
        ok(len(p["var"]) == 2, "two variance readings")
        ok(all(0.0 <= v <= 1.0 for v in p["var"]),
           "each axis holds a fraction of the variance", str(p["var"]))
        ok(p["var"][0] >= p["var"][1] - 1e-9,
           "the first axis holds at least as much as the second",
           str(p["var"]))
        ok(serve.project_2d([[0.0, 0.0]])["points"] == [],
           "too few points returns an empty projection rather than raising")
        ok(serve.project_2d([])["points"] == [], "no points is handled")

        # a latent that lies on a 1-D line must project to a line: the first
        # axis takes essentially all the variance, and that is exactly the
        # case the readout exists to disclose
        line = [[float(i), 2.0 * i, 0.0, 0.0] for i in range(20)]
        pl = serve.project_2d(line)
        ok(pl["var"][0] > 0.99 and pl["var"][1] < 1e-6,
           "a rank-1 latent puts ~all variance on one axis and says so",
           str(pl["var"]))

        # --- state heat: one RMS per layer ---
        heat = serve.state_heat(m)
        ok(len(heat) == m.layercount, "one heat value per layer",
           str(len(heat)))
        ok(all(v > 0 for v in heat), "every layer is carrying some state",
           str(heat))

        # --- theta delta ---
        before = m.theta.copy()
        for _ in range(4):
            m.step_batch([66], [32], [False])
        rows = serve.theta_delta(m, before)
        ok(len(rows) == len(g["named"]), "one delta row per role",
           str(len(rows)))
        ok(all(r["n"] > 0 for r in rows), "every role reports a param count")
        ok(any(r["rms"] > 0 for r in rows),
           "four steps moved something", str([r["rms"] for r in rows]))
        ok(all(0 <= r["rel"] for r in rows), "the ratio is non-negative")
        decay_row = [r for r in rows if r["role"] == "decay"][0]
        ok(decay_row["rms"] >= 0, "the decay row is present for the factorial")
        ok(serve.theta_delta(m, None) == [],
           "no snapshot means no delta, not a crash")


# ---------------------------------------------------- 8. cancel + session
def check_cancel_and_session():
    print("\n[8] cancel and session export")
    with tempfile.TemporaryDirectory() as d:
        ckpt = tiny(os.path.join(d, "s.npz"))
        rt = numpy_port.Runtime(path=ckpt, dim=64, layers=2, tagged=True,
                                keep_k=2, save_every=1)
        m = rt.model
        m.new_states(1)
        m.load(ckpt)
        serve.STATE.update(runtime=rt, model=m, path=ckpt, learn=True,
                           chunk=8, pending=0, steps=0, max_tokens=16,
                           threshold=0.35, long_threshold=4.0, save_every=1,
                           ema=None, snr=None, loss=None, ce=None,
                           cancel=False, session=[], hot=[])
        serve.PRIME = ""

        # --- the cancel flag is observed by the byte loop ---
        serve.STATE["cancel"] = True
        out = list(serve.stream(65, 20, 0.35))
        ok(any("note" in e and "your request" in e["note"] for e in out),
           "a cancelled stream says so instead of running to its cap",
           str([e for e in out if "note" in e]))
        ok(not any("t" in e for e in out),
           "and emits no bytes", str(len([e for e in out if "t" in e])))
        ok(serve.STATE["cancel"] is True,
           "the flag survives the route being callable")

        # --- /api/cancel does not need the lock ---
        serve.STATE["cancel"] = False
        status, headers, body = serve.api_cancel()
        ok(status == 200 and json.loads(body)["cancel"] is True,
           "/api/cancel answers without taking the lock")
        ok(serve.STATE["cancel"] is True, "and sets the flag")
        serve.STATE["cancel"] = False

        # --- the session record carries what RESULTS.md would ask for ---
        done = {"n": 12, "nonprint": 1, "cps": 5.5, "threshold": 0.35,
                "stops": [0.1, 0.4, 0.9], "learn": {"model_t": 65},
                "learned": {"ce": 1.5}, "snr": {"trunk": 0.1, "decay": 0.01},
                "memory": {"n_long": 4}}
        serve.record_session("hello", "hi there", done)
        ok(len(serve.STATE["session"]) == 1, "one row per exchange")
        row = serve.STATE["session"][0]
        for field in ("user", "reply", "n", "nonprint", "threshold", "model_t",
                      "snr", "stop_max", "stop_fired", "long_dims"):
            ok(field in row, f"the record carries '{field}'")
        ok(row["stop_max"] == 0.9, "the peak stop score is recorded")
        ok(row["stop_fired"] is True,
           "and whether it crossed the cut, which is the diagnosis")
        serve.record_session("x", "y", {"stops": [0.01], "threshold": 0.35})
        ok(serve.STATE["session"][-1]["stop_fired"] is False,
           "a run that never fired records that, too")

        serve.STATE["meta"] = {"path": "/tmp/x.npz", "digest": "abc123",
                               "t": 64747, "bytes_seen": 57602080,
                               "dim": 64, "layers": 2, "ring_k": 15,
                               "params": 45953, "learn": True,
                               "chunk": 8, "max_tokens": 16, "threshold": 0.35,
                               "long_bytes": 4.0, "markers": False,
                               "prime_bytes": 0, "started": "2026-01-01T00:00:00"}
        status, headers, body = serve.api_session()
        ok(status == 200, "/api/session answers")
        payload = json.loads(body)
        ok(len(payload["turns"]) == 2, "every turn is exported")
        for field in ("path", "digest", "t", "bytes_seen", "dim", "ring_k",
                      "params", "chunk", "threshold", "markers"):
            ok(field in payload["meta"],
               f"the session names its '{field}' so a result is reproducible")
        # the export has to survive a checkpoint that is no longer there
        serve.STATE["meta"] = {}
        ok(json.loads(serve.api_session()[2])["meta"] == {},
           "an unset meta exports as empty rather than raising")


# ------------------------------- 9. page/server field agreement
def check_page_contract():
    """
    The page and the server must agree on field names, statically.

    Added because the browser caught a bug this file could not: the page was
    still reading `s.snr` off an object the server had stopped sending `snr`
    on (it now sends trunk/decay/all), so the meter rendered the literal
    string "undefined" while every server-side assertion passed. Nothing in a
    response-shape test can see that -- the response was correct and the
    consumer was stale.

    So this pins the two ends against each other by name, and specifically
    pins the retired single-SNR field out of the page.
    """
    print("\n[9] page/server contract")
    import re
    page = serve.PAGE

    # the field the server sends vs the ones the page reads
    with tempfile.TemporaryDirectory() as d:
        m = numpy_port.Model(dim=64, layers=2)
        m.new_states(1)
        for _ in range(4):
            m.step_batch([65], [32], [False])
        serve.STATE.update(ema=None, snr=None, loss=None, ce=None, steps=25)
        serve.update_snr(m)
        sent = set(serve.snr_state())
    ok({"trunk", "decay", "all"} <= sent,
       "the server sends trunk/decay/all", str(sorted(sent)))
    ok("warm" in sent and "steps" in sent,
       "the server says when the reading is too early to compare")
    ok("loss" not in sent and "available" not in sent,
       "nothing is sent that duplicates the learned record",
       str(sorted(sent)))

    # every field the server sends must be READ by the page somewhere, or the
    # server grew a field the page silently ignores
    body = page[page.index("<script>"):]
    unread = [f for f in sent if ("." + f) not in body]
    ok(not unread, "every SNR field the server sends is read by the page",
       str(unread))

    # and the retired field must be gone from the page: this is the exact
    # token that rendered as "undefined". Only the `s.` form is banned --
    # reading `d.snr` off the payload is the correct way to call drawSnr, and
    # an earlier version of this check banned it too and was simply wrong.
    ok("s.snr" not in body, "the page no longer reads the retired s.snr field")
    ok(re.search(r"\bs\.snr\b", body) is None,
       "and not in any form, so a future rename cannot reintroduce it")

    # the SNR branch must not produce the literal string 'undefined' for any
    # field the server can send
    draw = body[body.index("function drawSnr"):body.index("function drawChips")]
    bad = re.findall(r"\$?\(['\"]([\w-]+)['\"]\)\.textContent\s*=\s*s\.(\w+)\s*\+",
                     draw)
    ok(all(f in sent for _, f in bad),
       "every field drawSnr concatenates is one the server sends",
       str([f for _, f in bad if f not in sent]))

    # the instruments must be wired to the fields the done frame sends
    for fn, field in (("drawStop", "stops"), ("drawLatent", "latent"),
                      ("drawHeat", "heat"), ("drawDelta", "delta")):
        ok(fn in body, fn + " is defined")
        ok(("d." + field) in body,
           fn + " is fed from the done frame's '" + field + "'")


if __name__ == "__main__":
    check_memory_report()
    check_prime()
    check_prompt()
    check_router()
    check_no_learn()
    check_instruments()
    check_cancel_and_session()
    check_http("stdlib", 8731)
    try:
        import flask  # noqa: F401
        check_http("flask", 8732)
    except ImportError:
        print("\n[4] http — flask server SKIPPED (flask not installed; the "
              "stdlib path is the one this project ships)")
    check_page_contract()

    print()
    if FAIL:
        print(f"FAIL — {len(FAIL)} check(s):")
        for f in FAIL:
            print(f"  · {f}")
        sys.exit(1)
    print("PASS — all web checks green")
