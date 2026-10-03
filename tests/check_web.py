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
def drain(gen):
    """Run a generator to its end: (the values it yielded, its return value).

    remember() returns its learned record through StopIteration.value, the
    same way stream() does, so a test that wants either half has to drive it
    the way the SSE handler does.
    """
    out = []
    try:
        while True:
            out.append(next(gen))
    except StopIteration as stop:
        return out, stop.value


def page_sources():
    """index.html, panels.css and app.js as one string.

    The page used to be a single inline literal, so a heap of checks below
    assert against CSS selectors and JS identifiers as though they were all
    in the same blob -- and they still are, for the purpose of those checks.
    The split into three files is real for the browser and invisible here.
    """
    return "\n".join(serve.web_source(n) for n in
                     ("index.html", "panels.css", "app.js"))


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
            # the page is three files now, so the document has to actually
            # reference the other two or the browser gets an unstyled page and
            # a console full of 404s that look like a routing bug
            ok('href="panels.css"' in page and 'src="app.js"' in page,
               "the document links the stylesheet and the script")
            for asset_path, ctype, needle in (
                    ("/panels.css", "text/css", "#mechscroll"),
                    ("/app.js", "text/javascript", "function drawConf")):
                with urllib.request.urlopen(base + asset_path, timeout=20) as r:
                    got, got_type = r.read().decode(), r.headers.get(
                        "Content-Type", "")
                ok(r.status == 200 and ctype in got_type and needle in got,
                   f"GET {asset_path} serves {ctype}",
                   f"{r.status} {got_type}")

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
            # ---- what the sampler knew about each byte ----
            # The page underlines a byte the sampler barely believed and
            # titles it with its probability, so a byte frame without one
            # would be a panel quietly drawing nothing.
            ok(all("p" in f for f in t_frames),
               "every byte frame carries the probability of that byte",
               str([f for f in t_frames if "p" not in f][:2]))
            ok(all(isinstance(f["p"], float) and 0.0 <= f["p"] <= 1.0
                   for f in t_frames),
               "and it is a probability, not a logit or a rank")
            # a fresh tiny model is close to uniform over 256, so this also
            # pins that p belongs to the byte that was WRITTEN and not to the
            # argmax: a bug that reported the max probability instead would
            # sit far above 1/256 and fail here rather than on a real reply
            ok(0 < sum(f["p"] for f in t_frames) / len(t_frames) < 0.25,
               "an untrained model's mean p is near 1/256",
               str(sum(f["p"] for f in t_frames) / len(t_frames)))
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
            # ---- the confidence series on the terminal frame ----
            c = d0.get("conf")
            ok(isinstance(c, dict) and len(c.get("p") or []) > 0,
               "the terminal frame carries the per-byte series",
               str(type(c)))
            ok(c["p"] == [f["p"] for f in t_frames][-len(c["p"]):],
               "the series is the tail of the streamed ones, in order")
            ok(len(c["p"]) == len(d0["stops"]) == len(d0["heat"]),
               "p, stops and state heat are index-aligned",
               f"{len(c['p'])}/{len(d0['stops'])}/{len(d0['heat'])}")
            ok(all(0.0 <= v <= 1.0 for v in c["h"]),
               "entropy is reported in bits/byte, on 0..1")
            ok(all(t > 0 for t in c["t"]), "the effective temperature is positive")
            ok(c["u"] == serve.UNSURE_P,
               "the page is told the threshold the server counts with, rather "
               "than keeping its own copy of the number",
               str(c.get("u")))
            # The window is indexed by BYTES. `n` counts only printable ones,
            # and the page's per-byte index includes the placeholder it draws
            # for each of the others, so keying the window off `n` puts the
            # "bytes 0-11 of 12" label on the wrong columns by exactly the
            # number of non-printable bytes in the reply.
            ok(c["to"] == d0["n"] + d0["nonprint"],
               "'to' is the last byte index, counting non-printables too",
               f"{c['to']} vs {d0['n'] + d0['nonprint']}")
            ok(c["from"] == c["to"] - len(c["p"]),
               "'from' is where the plotted window starts")
            # ---- the training pass reports as it goes ----
            # remember() is 97% of the turn's wall clock. If these frames stop
            # arriving the page is a blinking caret again, and only a test can
            # notice, because the pass still completes either way.
            train = [f["train"] for f in frames if "train" in f]
            ok(train, "the training pass reports progress", str(len(train)))
            ok([t["i"] for t in train] == sorted(t["i"] for t in train),
               "progress only moves forward")
            ok(train[-1]["i"] == train[-1]["total"],
               "the last frame is the last byte, so the bar cannot stop short",
               f"{train[-1]['i']}/{train[-1]['total']}")
            ok(len(train) > 3, "and it reports more than once for a short "
                              "exchange", str(len(train)))
            ok(all(t["ce"] is not None for t in train),
               "every progress frame carries a ce")
            ok(all("elapsed" in t and t["elapsed"] >= 0 for t in train),
               "and the elapsed seconds, so the cost of the pass is visible")
            # the frames have to arrive BETWEEN the reply finishing and the
            # terminal frame, or the page has nothing to draw them into
            order = [k for k, f in enumerate(frames)
                     if "learning" in f or "train" in f or f.get("done")]
            kinds = ["learning" if "learning" in frames[i]
                     else ("done" if frames[i].get("done") else "train")
                     for i in order]
            ok(kinds[0] == "learning" and kinds[-1] == "done"
               and set(kinds[1:-1]) == {"train"},
               "the order is: reply done, then progress, then the terminal "
               "frame", str(kinds[:3] + ["..."] + kinds[-2:]))
            # and the trace the loss panel draws is the per-byte record of the
            # same pass, not a summary of it
            ok(L_trace := (d0["learned"].get("trace") or []),
               "the learned record keeps the per-byte loss trace")
            ok(len(L_trace) == d0["learned"]["bytes"],
               "one point per byte fitted",
               f"{len(L_trace)} vs {d0['learned']['bytes']}")
            ok("trace_from" in d0["learned"],
               "and says where the kept window starts, when it is capped")

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
        prog, out = drain(serve.remember("hello", "hi", save_every=1))
        after = sorted(os.listdir(d))
        ok(out is None, "remember() is a no-op when learning is off", str(out))
        ok(prog == [], "and yields no progress frames either", str(prog[:1]))
        ok(before == after, "no file appeared", f"{before} -> {after}")
        ok(serve.learn_state() is None, "learn_state() is None when off")


# ---------------------------------------------------- 6. router well-formed
def check_router():
    print("\n[6] route table")
    # The four mini-AGI routes, plus session export and cancel, plus the two
    # static assets the page is made of. The API surface is still the same
    # five endpoints it was -- the assets are not API and are checked apart
    # below, so a new /api/* route still trips the no-accidents line.
    ok(set(serve.ROUTES) == {("/", "GET"), ("/api/chat", "POST"),
                             ("/api/state", "GET"), ("/api/prime", "GET"),
                             ("/api/session", "GET"), ("/api/cancel", "POST"),
                             ("/app.js", "GET"), ("/panels.css", "GET")},
       "the four API routes, plus the two static assets the page is made of",
       str(sorted(serve.ROUTES)))
    ok(len([k for k in serve.ROUTES if k[0].startswith("/api")]) == 5,
       "no API route was added by accident")
    ok(sorted(k for k in serve.ROUTES if k[0].startswith("/api")) ==
       [("/api/cancel", "POST"), ("/api/chat", "POST"),
        ("/api/prime", "GET"), ("/api/session", "GET"),
        ("/api/state", "GET")],
       "the API surface is still two POSTs and three GETs, unchanged",
       str(sorted(k for k in serve.ROUTES if k[0].startswith("/api"))))
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
# ------------------------------------------------- 11. the added stats
def check_new_stats():
    print("\n[11] half-life histogram, reply shape, per-layer SNR, churn")
    m = numpy_port.Model(dim=64, layers=3)
    m.new_states(1)

    # --- the histogram covers EVERY dimension, not the long ones ---
    h = serve.halflife_histogram(m, 4.0)
    ok(h["total"] == 3 * 64, "the histogram counts every dimension",
       f"{h['total']} vs {3*64}")
    ok(sum(b["n"] for b in h["buckets"]) == h["total"],
       "the buckets are a partition, so nothing is dropped between them")
    ok([b["label"] for b in h["buckets"]]
       == ["<1", "1-2", "2-4", "4-8", "8-16", "16-32", ">32"],
       "the buckets are the ones RESULTS.md reports in")
    ok(all(np.isfinite(b["lo"]) for b in h["buckets"]),
       "no bucket boundary is inf or nan")
    ok(h["n_long"] == 0, "at init nothing is long-lived, as documented",
       str(h["n_long"]))
    ok(h["buckets"][0]["n"] + h["buckets"][1]["n"] == h["total"],
       "every dim sits at or below one byte of half-life at init")
    m.layers[2].decay[0] = 8.0
    h2 = serve.halflife_histogram(m, 4.0)
    top = h2["buckets"][-1]
    ok(top["n"] == 1, "a dim pinned to a long half-life reaches >32",
       str(top["n"]))
    ok(h2["n_long"] == 1, "and is counted as long-lived", str(h2["n_long"]))
    ok(np.isfinite(top["bytes"]) and top["bytes"] > 100,
       "a decay of 8 gives a half-life of hundreds of bytes",
       str(top["bytes"]))
    # the sub-byte bucket's mean must not be rounded to a whole byte: 0.8B
    # displayed as "0B" says these dims hold nothing, which is the opposite
    # of the truth, and it is exactly the defect class this project keeps
    # hitting -- a number that does not mean what it appears to mean
    sub = h2["buckets"][0]
    ok(sub["n"] == 0 or (sub["bytes"] is None or sub["bytes"] != 0),
       "a bucket of sub-byte half-lives is never reported as 0 bytes",
       str(sub))
    ok(h2["buckets"][1]["bytes"] is not None
       and isinstance(h2["buckets"][1]["bytes"], float),
       "bucket means are floats, so a fractional byte survives",
       str(type(h2["buckets"][1]["bytes"])))
    # sigma == 1 exactly (a saturated decay) must not produce inf or nan, and
    # must not drop the dim out of the partition
    m.layers[1].decay[3] = 1e9
    h3 = serve.halflife_histogram(m, 4.0)
    ok(sum(b["n"] for b in h3["buckets"]) == h3["total"],
       "an effectively-infinite decay still lands in a bucket")
    ok(np.isfinite(serve._half_lives(m)).all(),
       "no half-life is inf or nan even at a saturated decay")
    ok(h3["n_live_long"] <= h3["n_long"],
       "live cannot exceed long", f"{h3['n_live_long']} vs {h3['n_long']}")

    # --- reply shape: the documented collapse modes are caught ---
    ok(serve.degeneracy(b"") is None, "an empty reply has no shape")
    ok(serve.degeneracy(b"a") is None, "nor does a single byte")
    nl = serve.degeneracy(b"\n" * 16)
    ok(nl["max_newlines"] == 16, "a 16-newline run is measured",
       str(nl["max_newlines"]))
    ok(nl["degenerate"],
       "and flagged -- RESULTS.md's FS-N mode is invisible to CE")
    ok(nl["max_run"] == 16, "the longest byte run is the newlines")
    rep = serve.degeneracy(b"the " * 40)
    ok(rep["unique"] == 4, "unique bytes are counted", str(rep["unique"]))
    ok(rep["repeat"] > 0.9, "a strictly repeating block reads as repeats",
       str(rep["repeat"]))
    ok(rep["degenerate"],
       "and is flagged: four distinct bytes over 160 is a loop")
    # a long run of DISTINCT bytes is not a loop, however many there are, and
    # must not be flagged -- the flag has to mean something
    ran = serve.degeneracy(bytes(range(64)) * 3)
    ok(ran["unique"] == 64 and ran["max_run"] == 1,
       "a cycling run of 64 distinct bytes has no run at all",
       str(ran["max_run"]))
    prose = serve.degeneracy(b"The quick brown fox jumps over the lazy dog. "
                             b"It was the best of times, it was the worst "
                             b"of times, it was the age of wisdom.")
    ok(not prose["degenerate"], "prose is not called degenerate",
       str(prose))
    ok(0.0 <= prose["repeat"] <= 1.0, "repeat is a fraction",
       str(prose["repeat"]))
    ok(0.0 < prose["entropy"] <= 8.0,
       "entropy is bits per byte, so it cannot exceed 8",
       str(prose["entropy"]))
    ok(abs(serve.degeneracy(b"a" * 100)["entropy"]) < 1e-9,
       "one repeated byte is 0 bits of entropy")

    # --- per-layer SNR and the gradient magnitude that used to be dropped ---
    for _ in range(25):
        m.step_batch([65], [32], [False])
    serve.STATE.update(ema=None, snr=None, gmean=None, steps=25)
    serve.update_snr(m)
    s = serve.snr_state()
    ok(len(s["layers"]) == m.layercount, "one SNR row per layer",
       str(len(s["layers"])))
    ok([r["layer"] for r in s["layers"]] == list(range(m.layercount)),
       "the rows are in layer order")
    for r in s["layers"]:
        ok(0.0 <= r["snr"] <= 1.0, f"layer {r['layer']} SNR in [0,1]")
        ok(r["gmean"] >= 0.0, f"layer {r['layer']} |g| is not negative")
        ok(r["dsnr"] >= 0.0, f"layer {r['layer']} decay SNR exists")
    ok(all(r["gmean"] > 0 for r in s["layers"]),
       "every layer has a non-zero gradient, so |g| is really read")
    ok(s["trunk_g"] > 0 and s["decay_g"] > 0,
       "the role gradients carry a magnitude too")
    ok(s["warm"] is False and s["steps"] == 25,
       "25 steps is past the warm-up, so the reading is comparable")
    serve.STATE.update(steps=1)
    ok(serve.snr_state()["warm"] is True,
       "one step is still warm, because one sample is always 0.1")
    serve.STATE.update(steps=25)

    # --- group cache: per-layer groups must sit inside the buffer ---
    g = serve.model_groups(m)
    ok(len(g["per_layer"]) == m.layercount, "one group per layer",
       str(len(g["per_layer"])))
    ok(sum(v.size for v in g["per_layer"].values()) <= m.grads.size,
       "the per-layer groups are inside the buffer")
    ok(set(g["per_layer_decay"]) == set(g["per_layer"]),
       "every layer has its decay indexed too")
    ok(all(g["per_layer_decay"][k].size == 64
           for k in g["per_layer_decay"]),
       "each layer's decay group is one vector of dim")
    ok(g["decay"].size + g["trunk"].size == m.grads.size,
       "trunk and decay still partition the buffer")

    # --- churn: the exact-credit set diff ---
    import tail_credit
    serve.STATE.update(hot_set=None)
    m.tail = tail_credit.TailCredit(1, 4)
    m.tail.cols = {0: np.array([3, 9, 11]), 1: np.array([7])}
    c = serve.hot_churn(m)
    ok(c["n"] == 4, "the set size is counted", str(c["n"]))
    ok(c["entered"] == 0 and c["left"] == 0,
       "the first look has no baseline, so nothing has entered or left")
    m.tail.cols = {0: np.array([9, 11, 40]), 1: np.array([7, 13])}
    c = serve.hot_churn(m)
    ok(c["entered"] == 2 and c["left"] == 1,
       "adopted dims enter and dropped dims leave", str(c))
    m.tail.cols = {0: np.array([9]), 1: np.array([7, 13])}
    ok(serve.hot_churn(m)["left"] == 2,
       "two more leaving is counted against the previous look")
    ok(serve.hot_churn(m)["entered"] == 0,
       "an unchanged set reports no churn")
    m.tail = None
    serve.STATE.update(hot_set=None)
    ok(serve.hot_churn(m)["n"] == 0, "a model with no tail has no hot set")

    # --- the session clock accumulates rather than reporting one turn ---
    serve.STATE.update(gen_s_total=0.0, learn_s_total=0.0)
    serve.STATE["gen_s_total"] += 1.5
    serve.STATE["learn_s_total"] += 38.5
    ok(abs(serve.STATE["gen_s_total"] - 1.5) < 1e-9,
       "generation time accumulates across a session")
    ok(serve.STATE["learn_s_total"]
       / (serve.STATE["gen_s_total"] + serve.STATE["learn_s_total"]) > 0.9,
       "the measured split is ~97% training")

    # --- ce must be labelled as a training reading, not a result ---
    serve.STATE.update(ce=0.0077, ce_best=None, steps=30)
    ok(serve.snr_state()["ce"] == 0.0077, "the training ce is carried")
    page = page_sources()
    ok("train ce" in page,
       "the page names it a TRAINING ce, since a low value here is the model "
       "fitting the text it was just fed, not a held-out measurement")
    ok("not held out" in page,
       "and says so in words, because 0.0077 looks like a triumph")
    ok("0.03" in page,
       "and quotes the project's noise bar, so a reader knows what 0.0077 "
       "is being compared against")


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
    page = page_sources()
    js = serve.web_source("app.js")

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
    body = js
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

    # ---- the composer has three states, and a stop affordance that reaches
    # all of them. Esc alone is a keyboard affordance, and this page gets
    # opened on phones.
    ok('id="stopbtn"' in page, "there is a stop button, not just an Esc key")
    ok("$('stopbtn').onclick = stopTurn" in body,
       "and it calls the same function Esc does, so the two cannot diverge")
    ok("function stopTurn" in body and body.count("function stopTurn") == 1,
       "there is exactly one stop implementation")
    ok("CUR.learning" in body and "'Training…'" in body,
       "and it says why it cannot be pressed during the training pass")

    # ---- ids. A duplicated id is silent: $('stop') returns the FIRST match,
    # so a button that shares an id with a canvas takes the canvas's place and
    # the button does nothing at all, with no error anywhere. This is the
    # cheapest possible check against the whole class.
    html = serve.web_source("index.html")
    all_ids = re.findall(r'\bid="([^"]+)"', html + "\n" + js)
    dupes = sorted({i for i in all_ids if all_ids.count(i) > 1})
    ok(not dupes, "no id appears twice in the page", str(dupes))
    # every element the script reaches for has to exist. Ids the script
    # creates itself count, which is why the script is searched too.
    known = set(all_ids)
    used = set(re.findall(r"\$\('([^']+)'\)", js))
    ok(not sorted(used - known),
       "every element the script looks up exists in the page",
       str(sorted(used - known)))
    # ---- the connection pill, because "the model is thinking" and "the
    # server is gone" look identical from the other side of a dead socket
    ok('id="conn"' in page and "server unreachable" in body,
       "a dropped server is reported as such, not as a slow model")
    ok("connWrap" in body and body.count("connWrap(") >= 3,
       "and every request reports through it, so it cannot claim to be "
       "connected while the last call failed")
    # ---- seed prompts and per-message copy
    ok("const SEEDS" in body and "class=\"seeds\"" in page,
       "the empty state offers prompts that make the model do something")
    ok("copyText" in body and "execCommand" in body,
       "copy has a fallback, because clipboard needs a secure context and "
       "this page is opened on LAN addresses")
    # "ask again" must not be called regenerate: the model has been trained on
    # the exchange, so a second answer is a different state, not a resample
    ok("b.textContent = 'ask again'" in body
       and "'regenerate'" not in body and '"regenerate"' not in body,
       "the retry is not named as a resample, because it is not one")

    # ---- the A/B comparison
    ok("while (PINS.length > 2) PINS.shift()" in body,
       "the pin tray holds two, and drops the oldest past that")
    ok("LAST.pin" not in body,
       "a pin record belongs to its own message, not to one global slot -- a "
       "global would file the newest turn's numbers under an older message")
    ok("offerAgain(msgEl, CUR.prompt || '', rec)" in body,
       "and each message is handed the record built from its own turn")
    # a difference against a value that was never measured must not be
    # printed: that is how a comparison claims a change nobody observed
    ok("if (kind !== 'text' && a != null && b != null)" in body,
       "no difference is printed unless both sides have the number")
    ok("return '—'" in body,
       "and a missing number prints as a dash, not as a zero")

    # ---- the byte cursor: one control, not one per canvas
    ok('id="xrow"' in page and 'tabindex="0"' in page,
       "the byte inspector is one focusable control")
    ok("ArrowRight" in body and "ArrowLeft" in body,
       "and the arrow keys move it along the reply")
    ok("e.stopPropagation()" in body,
       "Esc inside it does not also stop a running turn")
    # it has to be drawn on the per-byte panels, and those are the three that
    # share a byte axis; the latent has none, so it gets a ring instead
    ok(body.count("cursor(g,") >= 3,
       "the cursor is drawn on the byte-axis panels")
    ok("pts[mark]" in body,
       "and ringed on the latent, which has no byte axis to draw a line on")
    ok("dataset.i" in body and "data-i" in body,
       "hovering a byte in the reply itself moves the same cursor")

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

    # --- things this file has already got wrong once ---
    print("\n[10] regressions this gate now pins")

    # the panels measured 483px on an 800px screen and pushed the empty-state
    # greeting to y=745..802, i.e. off the bottom of the viewport
    ok("max-height" in page and "#mech" in page,
       "#mech has a height ceiling, so the thread cannot be squeezed out")
    ok("addEventListener('resize'" in body,
       "the canvases are redrawn on resize, or they stay stretched")
    ok("matchMedia" in body and "stacked()" in body,
       "the fold default follows the layout's own breakpoint")

    # the side-by-side
    ok('id="split"' in page and 'id="chatcol"' in page,
       "instruments and conversation are siblings, not stacked")
    ok("#split{display:flex" in page.replace(" ", " ") or
       "#split{flex:1 1 auto;min-height:0;display:flex" in page,
       "#split is a row")
    ok("#mechscroll" in page and "#mechscroll{flex:1 1 auto" in page,
       "only the instruments scroll; the meters stay pinned above them")
    ok("max-width:900px" in page and "max-height:560px" in page,
       "and it falls back to stacked on a narrow or short viewport")
    ok("flex-direction:column" in page,
       "the fallback really does stack the split")
    # the fold has to buy the whole window for the conversation, not just
    # un-hide it -- verified in the browser at 717px -> 1280px of chat
    ok("#chatcol{flex:1 1 auto" in page,
       "the chat column absorbs the space the panels were using")
    # two rows of meters, not a flex wrap that lands 3+1 at this width
    ok(".mrow{flex:0 0 auto;display:grid" in page and
       "grid-template-columns:1fr 1fr" in page,
       "the meters are a deliberate 2x2 grid")
    # the a11y edit dropped the canvases' width/height attributes, leaving
    # width:100%/height:auto on a canvas with no intrinsic size -- every
    # plot then took the 300x150 default ratio at full column width and the
    # instrument column no longer fit on the screen
    ok("canvas{" in page and "height:auto" not in page.split('canvas{')[1][:120],
       "the plots have an explicit height rather than an inherited default")
    ok("canvas#heat{height:" in page, "and the heatmap its own")
    # latent and stop head must still pair inside a ~530px column
    ok(".ipanel{flex:1 1 186px" in page,
       "the instrument panels are narrow enough to stay side by side")
    ok('aria-label="conversation"' in page,
       "the conversation is a labelled landmark")
    ok('id="note"' in page and "</header>" in page,
       "the transient note has a home in the header, visible in both layouts")
    ok("localStorage" in body, "the folded/expanded choice is remembered")
    ok("redrawAll" in body and "LAST.stops" in body,
       "unfolding redraws the canvases, which come back blank otherwise")

    # a dropped connection used to leave the status on "thinking..."
    ok("connection lost" in body,
       "a transport failure says so instead of leaving 'thinking...'")
    ok("!r.ok" in body,
       "a non-2xx from /api/chat is raised rather than read as an empty stream")

    # an empty table under a heading reads as a bug; in read-only there is
    # nothing to compare against and the panel should not pretend
    ok("panel.style.display = learn ? '' : 'none'" in body,
       "the delta panel is hidden outright when learning is off")
    ok("no change recorded" in body,
       "and says why it is empty when learning is on but nothing moved")

    # cancelling during the learning pass does NOT stop the training -- the
    # weights are already moving. Saying "stopped" there is a lie.
    ok("CUR.learning" in body and "cannot be interrupted" in body,
       "Esc says it is too late once the exchange is being trained on")

    # Esc used to wipe the reply it was stopping
    ok("CUR.reply = reply;" in body,
       "the reply Esc keeps tracks what actually streamed")

    # the swap frame carries only a dim list; rebuilding the meter from it put
    # a count and a hardcoded threshold on screen that the server never sent
    ok("threshold: MECH.longBytes" in body,
       "the swap path uses --long-bytes, not a hardcoded 4")
    ok("threshold:4" not in body, "and no hardcoded 4 survives anywhere")
    ok("n_long: d.memory.length" not in body,
       "the swap path does not fake n_long from the list length")

    # the mem bar is a proportion of a stated ceiling, so the ceiling is
    # stated rather than being a magic 8
    ok("/ 8" in body and "fills " in body and "$('m-mem').title" in body,
       "the 8-dim ceiling is explained in the meter's own label")

    # provenance belongs on screen for a tool built around it
    ok("$('prov')" in body, "the checkpoint digest is rendered in the header")
    ok('id="prov"' in page, "and there is an element to render it into")

    # accessibility: the panels are the content
    canvases = re.findall(r'<canvas[^>]*>', page)
    ok(len(canvases) >= 3, "the instrument panels are canvases",
       str(len(canvases)))
    # What matters is that EVERY canvas has a role and a label, and not that
    # the total comes to a particular number: a count is a test that a new
    # panel fails until someone remembers to edit it, which is a test that
    # gets deleted instead of satisfied. The per-canvas form cannot be passed
    # by adding a plot and forgetting it.
    ok(all('role="img"' in c for c in canvases),
       "every canvas exposes a role",
       str([c[:44] for c in canvases if 'role="img"' not in c]))
    ok(all('aria-label=' in c for c in canvases),
       "every canvas carries a label",
       str([c[:44] for c in canvases if 'aria-label=' not in c]))
    # and the label has to say something: an empty aria-label satisfies the
    # count above and tells a screen reader nothing at all
    ok(all(len(re.search(r'aria-label="([^"]*)"', c).group(1)) > 20
           for c in canvases),
       "no canvas has an empty label",
       str([c[:44] for c in canvases
            if len(re.search(r'aria-label="([^"]*)"', c).group(1)) <= 20]))
    # the confidence panel is a canvas like the rest, and is held to the same
    # rule rather than being the one that was added without a label
    ok('id="conf" role="img" aria-label=' in page,
       "the confidence panel is labelled like the other three")
    # the chips are built in JS, so their label is a setAttribute rather than
    # markup -- counting only the literal attribute would miss it
    ok("setAttribute('aria-label'" in body,
       "the chips carry a label too, set where they are created")
    ok("holding state, RMS" in body and "not holding anything" in body,
       "and the chip label states liveness in words, not only in colour")
    ok('role="status"' in page and 'aria-live="polite"' in page,
       "the status line announces itself to a screen reader")
    ok("prefers-reduced-motion" in page,
       "the caret and flare animations respect reduced-motion")
    ok("'aria-expanded'" in body or '"aria-expanded"' in body,
       "the panel toggle reports its state")


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
    check_new_stats()

    print()
    if FAIL:
        print(f"FAIL — {len(FAIL)} check(s):")
        for f in FAIL:
            print(f"  · {f}")
        sys.exit(1)
    print("PASS — all web checks green")
