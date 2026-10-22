"""tmt CLI gate: every subcommand's --help, plus one tiny end-to-end run.

    .venv/bin/python tests/check_cli.py

What it pins:
  * `tmt --help` and `tmt {chat,train,eval,bench} --help` all exit 0.
  * `tmt train` on a generated tiny corpus produces a checkpoint.
  * `tmt eval` on that checkpoint exits 0 and prints the REAL numbers:
    held-out CE with its SE, the parameter count, the ring and the
    half-life.
  * `tmt bench` and `tmt chat` run.
  * A missing local file (the corpora are gitignored, local-only) prints a
    clear message with the fetch command -- NOT a traceback.

Everything runs on the CPU reference engine, in a temp dir; nothing here
writes into the repo.
"""
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable
TMT = [PY, "-m", "tmt"]

TRAIN_PARAS = [
    "The quick brown fox jumps over the lazy dog while the sun sets behind the hills.",
    "A small compiler reads a line of text and turns it into a stream of bytes.",
    "Water in a narrow glass moves slowly, and the glass keeps the shape of the hand.",
    "Every measurement has an error, and the error has to be reported next to the number.",
    "The train leaves the station at dawn, and the fields are still wet with rain.",
    "She counted the steps on the staircase and wrote the total in the margin of the page.",
]
HOLD_PARAS = [
    "The old lighthouse keeper wrote a letter every week and mailed it to the mainland.",
    "A copper kettle whistles when the water is nearly ready, and the room fills with steam.",
    "They repaired the bridge in the autumn, replacing every plank that had started to rot.",
    "An old dog sleeps in the last patch of sunlight on the porch and does not move.",
]

FAILURES = []


def run(args, timeout=1800):
    proc = subprocess.run(TMT + args, cwd=REPO, capture_output=True, text=True,
                          timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}   {detail}")
        FAILURES.append(name)


def write_corpus(directory, paras, files, size, seed=7):
    os.makedirs(directory, exist_ok=True)
    rnd = random.Random(seed)
    paths = []
    for i in range(files):
        lines = []
        while sum(len(x) + 1 for x in lines) < size:
            lines.append(rnd.choice(paras))
        p = os.path.join(directory, f"part{i:02d}.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        paths.append(p)
    return paths


def main():
    print("== 1. every --help exits 0 ==")
    for args in ([], ["chat"], ["train"], ["eval"], ["bench"]):
        rc, out, err = run(args + ["--help"])
        check(f"tmt {' '.join(args) or '(root)'} --help exits 0", rc == 0,
              f"rc={rc} err={err[:200]}")
        check(f"tmt {' '.join(args) or '(root)'} --help prints usage",
              "usage:" in out, out[:120])

    print("== 2. the chat help says the GPU engine has no generation path ==")
    rc, out, err = run(["chat", "--help"])
    check("chat --help states there is no GPU generation path",
          "NO GENERATION PATH" in out, out[:200])

    print("== 3. tiny end-to-end: train -> eval ==")
    with tempfile.TemporaryDirectory(prefix="tmt_cli_gate_") as td:
        corpus = os.path.join(td, "corpus")
        held = os.path.join(td, "held")
        write_corpus(corpus, TRAIN_PARAS, 3, 6000)
        write_corpus(held, HOLD_PARAS, 2, 4000)
        ckpt = os.path.join(td, "tiny.npz")

        rc, out, err = run([
            "train", "--data-dir", os.path.join(corpus, "*.txt"),
            "--steps", "20", "--ckpt", ckpt, "--batch", "8",
            "--dim", "64", "--layers", "2", "--device", "cpu",
            "--from-scratch", "--no-tagged",
        ])
        check("tmt train exits 0", rc == 0, f"rc={rc} err={err[:400]}")
        check("tmt train wrote the checkpoint", os.path.exists(ckpt))
        check("tmt train printed the parameter count", "params" in out,
              out[-400:])
        check("tmt train printed the half-life", "half-life" in out,
              out[-400:])
        check("tmt train printed the ring", "ring" in out, out[-400:])
        check("tmt train printed no traceback", "Traceback" not in out + err)

        rc, out, err = run([
            "eval", ckpt, "--slices", "2", "--bytes", "1000",
            "--train-glob", os.path.join(corpus, "*.txt"),
            "--eval-glob", os.path.join(held, "*.txt"),
            "--shift-glob", "", "--novel-glob", "", "--xd-bytes", "0",
        ])
        check("tmt eval exits 0", rc == 0, f"rc={rc} err={err[:400]}")
        check("tmt eval printed the held-out CE", "held-out CE" in out,
              out[-400:])
        check("tmt eval printed the CE with its SE", "+/-" in out, out[-400:])
        check("tmt eval printed the parameter count", "params" in out,
              out[-400:])
        check("tmt eval printed the ring credit mode",
              "ring credit" in out, out[-400:])
        check("tmt eval printed the half-life", "half-life" in out, out[-400:])
        check("tmt eval printed no traceback", "Traceback" not in out + err)

        rc, out, err = run([
            "bench", "--ckpt", ckpt, "--batch", "32", "--iters", "5",
            "--device", "cpu",
        ])
        check("tmt bench exits 0", rc == 0, f"rc={rc} err={err[:400]}")
        check("tmt bench printed no traceback", "Traceback" not in out + err)

        rc, out, err = run(["chat", "--ckpt", ckpt, "--prompt", "The train",
                            "--n", "64"])
        check("tmt chat exits 0", rc == 0, f"rc={rc} err={err[:400]}")
        check("tmt chat generated bytes", "bytes  :" in out, out[-400:])
        check("tmt chat says the GPU engine has no generation path",
              "NO generation path" in out, out[-400:])

        print("== 4. a missing local file is a message, not a traceback ==")
        rc, out, err = run(["eval", ckpt,
                            "--eval-glob", os.path.join(td, "nothing", "*.txt")])
        check("tmt eval on missing held-out data exits non-zero", rc != 0)
        check("tmt eval on missing held-out data says no traceback",
              "Traceback" not in out + err)
        check("tmt eval on missing held-out data prints the fetch command",
              "fetch_eval_data.py" in out + err, (out + err)[-300:])

        rc, out, err = run(["train", "--data-dir",
                            os.path.join(td, "nothing", "*.txt"),
                            "--steps", "1", "--ckpt", ckpt])
        check("tmt train on missing corpus exits non-zero", rc != 0)
        check("tmt train on missing corpus prints the fetch command",
              "fetch_corpus.py" in out + err, (out + err)[-300:])

        rc, out, err = run(["chat", "--ckpt", os.path.join(td, "no.npz")])
        check("tmt chat on a missing checkpoint exits non-zero", rc != 0)
        check("tmt chat on a missing checkpoint says no traceback",
              "Traceback" not in out + err)

    print()
    if FAILURES:
        print(f"GATE: FAIL ({len(FAILURES)} check(s) failed: "
              f"{', '.join(FAILURES)})")
        return 1
    print("GATE: PASS (all checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
