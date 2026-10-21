#!/usr/bin/env python3
"""check_data_resume.py -- a resume must CONTINUE the corpus, not replay it.

THE BUG THIS EXISTS FOR. The trainer's file loop always starts at the first
file, so a resumed run re-reads the bytes it had already trained on. With a
10,000-step checkpoint cadence that caps an arm's UNIQUE data at ~10 MB: a
165 MB "epoch" would degenerate into "the same 10 MB, sixteen times", and the
arm would never see the rest of the corpus at all. The checkpoint now carries
the stream position -- (file index, window index within that file) -- and the
trainer skips what it has already trained on.

THE CHECK, in three parts:
  1. the checkpoint carries the position, and it ADVANCES with training;
  2. a resumed run continues at that position -- the first window it consumes
     is the window the checkpoint points at, and its byte is the byte that
     position means (computed here from the corpus, independently);
  3. a fresh run still starts at the beginning (the fix did not change the
     from-scratch behaviour), and the numpy reference stores the position
     the same way.

Run:  python tests/check_data_resume.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy_port as np_  # noqa: E402

_fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def make_corpus(path, nbytes=400_000):
    """A deterministic pseudo-text corpus, big enough for several windows."""
    rng = np.random.default_rng(0)
    letters = np.frombuffer(b"abcdefghij klmnopqrst uvwxyz.", dtype=np.uint8)
    data = letters[rng.integers(0, len(letters), size=nbytes)]
    with open(path, "wb") as fh:
        fh.write(data.tobytes())
    return data


def stream_bytes(path, B, n_windows):
    """The byte each window starts on, computed independently of the trainer."""
    with open(path, "rb") as fh:
        data = fh.read()
    streams, _ends = np_.Runtime.make_streams(data, B)
    return [int(streams[0][t]) for t in range(n_windows)]


def run_trainer(ckpt, corpus, steps, extra=()):
    cmd = [sys.executable, os.path.join(HERE, "train_gpu.py"), ckpt, "train",
           "--dataset", corpus, "--batch", "8", "--seqlen", "1",
           "--steps", str(steps), "--warmup", "0", "--no-tagged",
           "--ring-k", "8", "--device", "cpu", "--seed", "0", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + p.stderr


def main():
    print("=" * 78)
    print("DATA RESUME: a resumed run continues the corpus instead of replaying it")
    print("=" * 78)
    tmp = tempfile.mkdtemp()
    corpus = os.path.join(tmp, "corpus.txt")
    make_corpus(corpus)
    ckpt = os.path.join(tmp, "ck.npz")

    # ---- 1. the checkpoint carries the position, and it advances ----------
    print("(1) the checkpoint carries the stream position")
    out = run_trainer(ckpt, corpus, 5)
    d = np.load(ckpt)
    check("the checkpoint has data_file/data_off",
          "data_file" in d.files and "data_off" in d.files,
          f"keys: {[k for k in d.files if k.startswith('data')]}")
    if "data_off" in d.files:
        pos = (int(d["data_file"]), int(d["data_off"]))
        check("the position is where the run stopped", pos[1] >= 4, f"position {pos}")
    else:
        check("the position is where the run stopped", False, "no position in the checkpoint")
        pos = (0, 0)

    # ---- 2. a resume CONTINUES at that position --------------------------
    print("(2) a resumed run continues at that position")
    out2 = run_trainer(ckpt, corpus, 3)
    line = [l for l in out2.splitlines() if "first window after the resume" in l]
    check("the resume announces where it continues", bool(line),
          line[0].strip() if line else "no such line in the log")
    ok_byte = False
    if line:
        # "the first window after the resume starts at window T of file F, byte B"
        try:
            body = line[0].split("the first window after the resume", 1)[1]
            t_win = int(body.split("window", 1)[1].split("of", 1)[0].strip())
            b_seen = int(body.split("byte", 1)[1].strip())
            want = stream_bytes(corpus, 8, t_win + 1)[t_win]
            ok_byte = (b_seen == want)
            check("the first byte after the resume is the byte that position means",
                  ok_byte, f"resumed at byte {b_seen}, the corpus says {want}")
        except Exception as exc:                        # noqa: BLE001
            check("the first byte after the resume is the byte that position means",
                  False, f"could not parse the log line: {exc}")
    d2 = np.load(ckpt)
    pos2 = (int(d2["data_file"]), int(d2["data_off"])) if "data_off" in d2.files else (0, 0)
    check("the position advanced past the resume point", pos2[1] > pos[1],
          f"{pos} -> {pos2}")

    # ---- 3. a fresh run still starts at the beginning ---------------------
    print("(3) a fresh run still starts at the beginning")
    ck3 = os.path.join(tmp, "fresh.npz")
    out3 = run_trainer(ck3, corpus, 3)
    d3 = np.load(ck3)
    pos3 = (int(d3["data_file"]), int(d3["data_off"])) if "data_off" in d3.files else (0, 0)
    check("a fresh run starts at window 0", pos3[1] < 4, f"position {pos3}")
    check("a fresh run does not print a resume line",
          "RESUMING the stream" not in out3)

    # ---- 4. the numpy reference stores it the same way --------------------
    print("(4) the numpy reference stores the position too")
    m = np_.Model(dim=16, layers=2)
    m.new_states(4)
    m.data_pos = (1, 123)
    p4 = os.path.join(tmp, "np.npz")
    m.save(p4)
    d4 = np.load(p4)
    check("numpy_port writes data_file/data_off",
          "data_file" in d4.files and "data_off" in d4.files,
          f"keys: {[k for k in d4.files if k.startswith('data')]}")
    m2 = np_.Model(dim=16, layers=2)
    m2.new_states(4)
    m2.load(p4)
    check("numpy_port restores it",
          getattr(m2, "data_pos", None) == (1, 123),
          f"got {getattr(m2, 'data_pos', None)}")

    print()
    if _fails:
        print(f"FAILED: {len(_fails)} check(s): {_fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
