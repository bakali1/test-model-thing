"""tmt -- the community CLI for the byte-level recurrent LM proof of concept.

Four commands, and NONE of them reimplements the engine:

  chat   generation, on the numpy_port CPU path (tools/chat_models.py)
  train  the trainer          (train_gpu.py, the gpu_port/torch engine)
  eval   the ruler            (tools/eval.py)
  bench  the ring-credit bench (tools/bench_ring_credit.py)

HONESTY (read the --help of each command):
  * The corpora are NOT in git. They are local-only and gitignored, so a
    fresh clone has no data/ and every command that needs text says so and
    prints the exact fetch command instead of a traceback.
  * The GPU engine (gpu_port.py) has NO GENERATION PATH. It can train and
    it can be scored, but there is no Model.sample / frozen_step port, so
    `tmt chat` always samples with the numpy_port CPU reference engine --
    whatever checkpoint you point it at.
  * The model is small (~4.6M params, byte level). On byte-level text it
    loses to an order-8 byte n-gram. See README-FORK.md.
"""
import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")

# The fetch commands, quoted verbatim where they are printed.
FETCH_CORPUS = (".venv/bin/python fetch_corpus.py --layers A,B,C --gb 4.5 "
                "--out ./corpus")
FETCH_EVAL = ".venv/bin/python fetch_eval_data.py"

CE_RE = re.compile(r"mean\s+([0-9]+\.[0-9]+)\s*\+/-\s*([0-9]+\.[0-9]+)")


# ---------------------------------------------------------------- helpers --
def _need(path, what, hint):
    """Missing local file -> a message with the fix, never a traceback."""
    if not os.path.exists(path):
        print(f"FAIL: {what} not found: {path}", file=sys.stderr)
        print(f"  {hint}", file=sys.stderr)
        sys.exit(1)


def _engine(path, what):
    _need(path, what, "the engine file is part of the repo; run the CLI from "
                      "a clone of it (pip install -e .)")


def _repo_python():
    """Prefer the repo venv's interpreter (it has torch); fall back to ours."""
    venv = os.path.join(REPO, ".venv", "bin", "python")
    if os.path.exists(venv):
        return venv
    return sys.executable


def _glob_files(pattern):
    import glob as _g
    return sorted(_g.glob(pattern, recursive=True))


def _rel(path):
    try:
        r = os.path.relpath(path, os.getcwd())
    except ValueError:
        return path
    return path if r.startswith("..") else r


def _ckpt_summary(path, engine, credit_mode):
    """Print the model's REAL numbers, read out of the checkpoint itself."""
    import numpy as np

    with np.load(path, allow_pickle=False) as d:
        files = list(d.files)
        dim = int(d["Wd"].shape[1])
        layers = sum(1 for k in files
                     if k.startswith("layer") and k.endswith(".W"))
        named = ("embed", "Wd", "bd", "Ws", "bs", "fsp.W")
        pkeys = [k for k in files
                 if k in named
                 or (k.startswith("layer")
                     and k.rsplit(".", 1)[-1] in ("W", "g", "b", "decay"))]
        params = 0
        for k in pkeys:
            shp = d[k].shape
            n = 1
            for s in shp:
                n *= int(s)
            params += n
        ring_k = int(np.ravel(d["ring_k"])[0]) if "ring_k" in files else 0
        t = int(np.ravel(d["t"])[0]) if "t" in files else -1
        nb = int(np.ravel(d["bytes"])[0]) if "bytes" in files else -1
        B = int(np.ravel(d["B"])[0]) if "B" in files else -1
        decay = np.stack([d[f"layer{i}.decay"] for i in range(layers)])
        sigma = 1.0 / (1.0 + np.exp(-decay))
        with np.errstate(divide="ignore", invalid="ignore"):
            hl = np.log(0.5) / np.log(sigma)
        hot = int((sigma ** max(ring_k, 1) > 0.01).sum()) if ring_k else 0

    print("\n--- tmt: checkpoint summary (real numbers, read from the file) ---")
    print(f"checkpoint     : {path}")
    print(f"params         : {params:,}  (dim={dim}, layers={layers})")
    print(f"ring           : K={ring_k} lags")
    print(f"ring credit    : {credit_mode}")
    print(f"half-life      : mean={np.nanmean(hl):.2f} B | "
          f"min={np.nanmin(hl):.2f} B | max={np.nanmax(hl):.2f} B | "
          f"hot dims (sigma^K>1%): {hot}")
    if t >= 0:
        seen = f"{nb:,}" if nb >= 0 else f"~{t * max(B, 1):,}"
        print(f"steps / bytes  : t={t:,} | bytes seen {seen}")
    print(f"engine         : {engine}")
    return {"params": params, "dim": dim, "layers": layers, "ring_k": ring_k,
            "half_life": float(np.nanmean(hl)), "ce": None, "se": None}


# ------------------------------------------------------------------ chat --
def cmd_chat(args):
    """Generation. numpy_port CPU path ONLY -- gpu_port has no sampler."""
    _engine(os.path.join(TOOLS, "chat_models.py"), "the generation tool")
    ckpt = args.ckpt or os.path.join(REPO, "tmt.npz")
    if not os.path.exists(ckpt):
        print(f"FAIL: checkpoint not found: {_rel(ckpt)}", file=sys.stderr)
        print("  Train one first:  tmt train --data-dir <dir> --steps 200 "
              "--ckpt tmt.npz", file=sys.stderr)
        print(f"  Or use the checkpoint shipped with the repo: "
              f"{_rel(os.path.join(REPO, 'tmt.npz'))}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, REPO)
    sys.path.insert(0, TOOLS)
    import chat_models

    try:
        model, info = chat_models.load_model(ckpt)
    except Exception as exc:                        # unreadable checkpoint
        print(f"FAIL: cannot load checkpoint {_rel(ckpt)}: {exc}", file=sys.stderr)
        print("  The file may be truncated or from a different engine "
              "version.", file=sys.stderr)
        sys.exit(1)

    prompt = args.prompt.encode("utf-8", errors="ignore")
    gen, stops, dt, ent, eff_t = chat_models.generate(
        model, prompt, args.n, temp=args.temp, seed=args.seed)

    print("--- tmt chat (numpy_port CPU generation path) ---")
    print(f"prompt : {prompt.decode('utf-8', errors='replace')!r}")
    print(f"bytes  : {len(gen)} generated in {dt:.2f}s "
          f"({len(gen) / max(dt, 1e-9):.0f} B/s) | stop-head hits {stops} | "
          f"mean entropy {ent:.3f} | effective temp {eff_t:.3f}")
    print("--- generation (raw bytes, escaped) ---")
    print(chat_models.escape(gen))
    print("--- end of generation ---")
    print("NOTE: this sampled with the numpy_port CPU reference engine. The "
          "GPU engine (gpu_port.py) has NO generation path -- no "
          "Model.sample / frozen_step port exists there yet.")

    summary = _ckpt_summary(ckpt, engine="numpy_port (CPU reference)",
                            credit_mode="numpy reference (CPU, exact credit, "
                                        "no fused kernel)")
    return summary


# ----------------------------------------------------------------- train --
def cmd_train(args):
    """Delegate to train_gpu.py -- the trainer is NOT reimplemented here."""
    trainer = os.path.join(REPO, "train_gpu.py")
    _engine(trainer, "the trainer")

    dataset = args.data_dir
    files = _glob_files(dataset)
    if not files:
        print(f"FAIL: no training files match {_rel(dataset)!r}", file=sys.stderr)
        print("  The corpora are local-only (not in git). Fetch them with:",
              file=sys.stderr)
        print(f"    {FETCH_CORPUS}", file=sys.stderr)
        print("  (the fetcher writes plain text into ./corpus; point "
              "--data-dir at it)", file=sys.stderr)
        sys.exit(1)

    ckpt = os.path.abspath(args.ckpt)
    cmd = [_repo_python(), trainer, ckpt, "train",
           "--dataset", dataset,
           "--steps", str(args.steps)]
    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    if args.dim is not None:
        cmd += ["--dim", str(args.dim)]
    if args.layers is not None:
        cmd += ["--layers", str(args.layers)]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.device:
        cmd += ["--device", args.device]
    if args.ring_credit:
        cmd += ["--ring-credit", args.ring_credit]
    if args.from_scratch:
        cmd += ["--from-scratch"]
    if args.no_tagged:
        cmd += ["--no-tagged"]
    if args.keep is not None:
        cmd += ["--keep-k", str(args.keep)]
    if args.log_every:
        cmd += ["--log-every", str(args.log_every)]
    if args.save_every is not None:
        cmd += ["--save-every", str(args.save_every)]
    cmd += args.rest

    print("--- tmt train -> train_gpu.py (the existing trainer) ---")
    print("  " + " ".join(cmd))
    print(f"  data: {len(files)} file(s) matching {dataset!r}", flush=True)
    res = subprocess.run(cmd, cwd=REPO)
    if res.returncode != 0:
        print(f"\nFAIL: train_gpu.py exited {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)
    if not os.path.exists(ckpt):
        print(f"\nFAIL: training finished but no checkpoint at {_rel(ckpt)}",
              file=sys.stderr)
        sys.exit(1)
    _ckpt_summary(ckpt, engine="gpu_port (torch)",
                  credit_mode=f"{args.ring_credit or 'auto'} (requested; "
                              f"train_gpu.py resolves 'auto' on the device: "
                              f"full on CPU, fused nvrtc/compiled on CUDA)")
    return None


# ------------------------------------------------------------------ eval --
def cmd_eval(args):
    """Delegate to tools/eval.py -- the ruler is NOT reimplemented here."""
    ruler = os.path.join(TOOLS, "eval.py")
    _engine(ruler, "the ruler")
    ckpt = os.path.abspath(args.ckpt)
    if not os.path.exists(ckpt):
        print(f"FAIL: checkpoint not found: {_rel(ckpt)}", file=sys.stderr)
        print("  Train one first:  tmt train --data-dir <dir> --steps 200 "
              "--ckpt tmt.npz", file=sys.stderr)
        sys.exit(1)

    eval_files = _glob_files(args.eval_glob)
    if not eval_files:
        print(f"FAIL: no held-out files match {_rel(args.eval_glob)!r}",
              file=sys.stderr)
        print("  The corpora are local-only (not in git). Fetch the held-out "
              "split with:", file=sys.stderr)
        print(f"    {FETCH_EVAL}", file=sys.stderr)
        print(f"  (it writes data/eval/ and data/eval_shift/)",
              file=sys.stderr)
        sys.exit(1)

    cmd = [_repo_python(), ruler, ckpt,
           "--slices", str(args.slices), "--bytes", str(args.bytes)]
    cmd += ["--eval-glob", args.eval_glob]
    if args.train_glob is not None:
        cmd += ["--train-glob", args.train_glob]
    if args.shift_glob is not None:
        cmd += ["--shift-glob", args.shift_glob]
    if args.novel_glob is not None:
        cmd += ["--novel-glob", args.novel_glob]
    if args.xd_bytes is not None:
        cmd += ["--xd-bytes", str(args.xd_bytes)]
    cmd += args.rest

    print("--- tmt eval -> tools/eval.py (the existing ruler) ---")
    print("  " + " ".join(cmd))
    print("  held-out: %d file(s) matching %r" % (len(eval_files),
                                                  args.eval_glob),
          flush=True)

    # Stream the ruler's output live AND keep it, so the CE +/- SE can be
    # repeated in the summary block below.
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    ce = se = None
    for line in proc.stdout:
        print(line, end="")
        m = CE_RE.search(line)
        if m and ce is None:
            ce, se = float(m.group(1)), float(m.group(2))
    proc.wait()
    if proc.returncode != 0:
        print(f"\nFAIL: tools/eval.py exited {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)

    summary = _ckpt_summary(ckpt, engine="numpy_port (CPU reference, weights "
                                         "only -- the ruler does not train)",
                            credit_mode="numpy reference (CPU, exact credit); "
                                        "the checkpoint does not record the "
                                        "GPU credit mode, and the ruler scores "
                                        "weights, so the credit path does not "
                                        "change the numbers")
    if ce is not None:
        print(f"held-out CE    : {ce:.4f} +/- {se:.4f} nats/byte "
              f"(SE across slices)")
    else:
        print("held-out CE    : (not parsed from the ruler's output)")
    return summary


# ----------------------------------------------------------------- bench --
def cmd_bench(args):
    """Delegate to tools/bench_ring_credit.py."""
    bench = os.path.join(TOOLS, "bench_ring_credit.py")
    _engine(bench, "the bench tool")
    ckpt = os.path.abspath(args.ckpt) if args.ckpt else None
    if ckpt and not os.path.exists(ckpt):
        print(f"FAIL: checkpoint not found: {_rel(ckpt)}", file=sys.stderr)
        print("  Train one first:  tmt train --data-dir <dir> --steps 200 "
              "--ckpt tmt.npz", file=sys.stderr)
        sys.exit(1)
    if ckpt is None:
        default = os.path.join(REPO, "tmt.npz")
        if os.path.exists(default):
            ckpt = default
        else:
            print("FAIL: no checkpoint given and no tmt.npz in the repo",
                  file=sys.stderr)
            print("  tmt bench --ckpt <path>", file=sys.stderr)
            sys.exit(1)

    cmd = [_repo_python(), bench, "--ckpt", ckpt]
    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    if args.iters is not None:
        cmd += ["--iters", str(args.iters)]
    if args.ring_k is not None:
        cmd += ["--ring-k", str(args.ring_k)]
    if args.device:
        cmd += ["--device", args.device]
    cmd += args.rest

    print("--- tmt bench -> tools/bench_ring_credit.py (the existing bench) ---")
    print("  " + " ".join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=REPO)
    if res.returncode != 0:
        print(f"\nFAIL: bench_ring_credit.py exited {res.returncode}",
              file=sys.stderr)
        sys.exit(res.returncode)
    _ckpt_summary(ckpt, engine="gpu_port (torch)",
                  credit_mode="auto (engine default; the bench compares the "
                              "credit paths against each other)")
    return None


# ------------------------------------------------------------------ main --
def build_parser():
    parser = argparse.ArgumentParser(
        prog="tmt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=("HONESTY: the corpora are not in git (local-only); the GPU "
                "engine has no generation path, so chat runs on the numpy "
                "CPU reference; this model is small enough to lose to an "
                "order-8 byte n-gram. Unknown flags are forwarded to the "
                "underlying tool."))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "chat", help="Sample a checkpoint (numpy_port CPU generation path).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("Generate a continuation of --prompt from a checkpoint.\n"
                     "\n"
                     "THE GPU ENGINE HAS NO GENERATION PATH: gpu_port.py can\n"
                     "train and it can be scored, but it has no Model.sample /\n"
                     "frozen_step port, so this command ALWAYS samples with the\n"
                     "numpy_port CPU reference engine, whatever checkpoint you\n"
                     "point it at. The reused generation code lives in\n"
                     "tools/chat_models.py (load_model / generate)."),
        epilog="Unknown flags are forwarded to tools/chat_models.py.")
    p.add_argument("--ckpt", default=None,
                   help="checkpoint .npz (default: tmt.npz in the repo)")
    p.add_argument("--prompt", default="hello", help="prompt bytes (UTF-8)")
    p.add_argument("--n", type=int, default=200, help="bytes to generate")
    p.add_argument("--temp", type=float, default=None,
                   help="sampling temperature (default: the model's)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser(
        "train", help="Train a checkpoint (delegates to train_gpu.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("Train on a glob of text files. Delegates to "
                     "train_gpu.py -- the existing trainer; this CLI does "
                     "not reimplement it."),
        epilog=("Unknown flags are forwarded to train_gpu.py. The corpora "
                "are local-only; fetch them with:\n  " + FETCH_CORPUS))
    p.add_argument("--data-dir", default="data/wikipedia_clean/**/wiki_*",
                   help="glob of training files (default: the local wiki set)")
    p.add_argument("--steps", type=int, default=1000, help="training steps")
    p.add_argument("--ckpt", default="tmt.npz", help="output checkpoint path")
    p.add_argument("--batch", type=int, default=None, help="streams per step")
    p.add_argument("--dim", type=int, default=None, help="model width")
    p.add_argument("--layers", type=int, default=None, help="layer count")
    p.add_argument("--lr", type=float, default=None, help="learning rate")
    p.add_argument("--device", default=None, help="cuda | cpu")
    p.add_argument("--ring-credit", default=None,
                   choices=("auto", "full", "compiled", "triton", "nvrtc"),
                   help="ring-credit path (GPU engine only)")
    p.add_argument("--from-scratch", action="store_true",
                   help="start from t=0 instead of resuming")
    p.add_argument("--no-tagged", action="store_true",
                   help="do not write the tagged t######.npz siblings")
    p.add_argument("--keep", type=int, default=None, help="tagged saves to keep")
    p.add_argument("--log-every", type=int, default=None,
                   help="print the loss/rate line every N steps")
    p.add_argument("--save-every", type=int, default=None,
                   help="checkpoint cadence in steps")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser(
        "eval", help="Score a checkpoint on held-out text (the ruler).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("Held-out CE +/- SE, parameter count, ring, half-life. "
                     "Delegates to tools/eval.py -- the existing ruler."),
        epilog=("Unknown flags are forwarded to tools/eval.py. The held-out "
                "split is local-only; fetch it with:\n  " + FETCH_EVAL))
    p.add_argument("ckpt", nargs="?", default="tmt.npz",
                   help="checkpoint to score (default: tmt.npz)")
    p.add_argument("--slices", type=int, default=3,
                   help="disjoint held-out slices (the SE comes from these)")
    p.add_argument("--bytes", type=int, default=20000, help="bytes per slice")
    p.add_argument("--train-glob", default=None, help="training glob")
    p.add_argument("--eval-glob", default="data/eval/**/*",
                   help="held-out glob")
    p.add_argument("--shift-glob", default=None, help="shift-distribution glob")
    p.add_argument("--novel-glob", default=None, help="cross-domain glob")
    p.add_argument("--xd-bytes", type=int, default=None,
                   help="cross-domain bytes per novel (0 disables)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser(
        "bench", help="Measure the ring-credit paths (bench_ring_credit.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("Time the ring-credit paths against each other. "
                     "Delegates to tools/bench_ring_credit.py."),
        epilog="Unknown flags are forwarded to tools/bench_ring_credit.py.")
    p.add_argument("--ckpt", default=None, help="checkpoint to load")
    p.add_argument("--batch", type=int, default=None, help="streams per step")
    p.add_argument("--iters", type=int, default=None, help="timed iterations")
    p.add_argument("--ring-k", type=int, default=None, help="explicit ring_k")
    p.add_argument("--device", default=None, help="cuda | cpu")
    p.set_defaults(func=cmd_bench)

    return parser


def main(argv=None):
    parser = build_parser()
    ns, rest = parser.parse_known_args(argv)
    ns.rest = rest
    ns.func(ns)


if __name__ == "__main__":
    main()
