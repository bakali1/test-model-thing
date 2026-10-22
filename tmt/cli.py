"""tmt CLI — four commands for newcomers. Uses numpy_port for chat (gpu_port
has no generation path; ~60 lines would port Model.sample/frozen_step/Runtime.chat
to gpu_port.GpuModel). Reuses train_gpu.py, tools/eval.py, bench_ring_credit.py.

HONESTY: corpora are not in git; CPU-only works; CUDA needs torch; a model
at ~4.5M params on byte-level text is expected to lose to an order-8 byte
n-gram (see README-FORK.md)."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _repo_root():
    # cli may be invoked from anywhere after install
    return HERE

# ----- chat (numpy path only) -----
def cmd_chat(args):
    root = _repo_root()
    sys.path.insert(0, root)
    import numpy_port as np_
    import numpy as np
    path = args.ckpt or os.path.join(root, "tmt.npz")
    if not os.path.exists(path):
        print(f"FAIL: checkpoint not found: {path}")
        sys.exit(1)
    # Load via numpy_port (CPU path; gpu_port has no generation equivalent)
    rt = np_.Runtime.__new__(np_.Runtime)
    rt.load(path)
    # Print real model metadata from the loaded state
    total = 0
    d = np.load(path, allow_pickle=True)
    for k in d.files:
        v = d[k]
        if hasattr(v, "shape") and isinstance(v.shape, tuple):
            total += int(np.prod(v.shape)) if hasattr(np, "prod") else int(np.multiply.reduce(v.shape))
    ring_mode = "unknown"
    half_life = "unknown"
    for k in d.files:
        if k == "ring": ring_mode = str(d[k].shape) if hasattr(d[k],"shape") else "present"
        if k in ("half_life", "half_life_init"): half_life = str(d[k])[:40]
    print(f"CHECKPOINT: {path}")
    print(f"  param elements (approx): {total}")
    print(f"  ring mode: {ring_mode}")
    print(f"  half-life info: {half_life}")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rt.chat(save=td, frozen=True)
    print("NOTE: generation used numpy_port path (NOT gpu_port).")
    print("Port to GPU: implement Model.sample / frozen_step (~60 lines) on gpu_port.GpuModel.")

def cmd_train(args):
    root = _repo_root()
    # Delegate to the trainer the repo already has; do not reimplement.
    cmd = [sys.executable, os.path.join(root, "train_gpu.py")]
    if args.data_dir: cmd += ["--data-dir", args.data_dir]
    if args.steps: cmd += ["--steps", str(args.steps)]
    if args.ckpt: cmd += ["--ckpt", args.ckpt]
    if args.keep: cmd += ["--keep", str(args.keep)]
    if args.tagged: cmd += ["--tagged"]
    cmd += args.rest
    print("DELEGATE to train_gpu.py (GPU/CUDA engine; CPU works via torch.device cpu):")
    print(" ", " ".join(cmd))
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

def cmd_eval(args):
    root = _repo_root()
    # Delegate to the ruler the repo already has.
    cmd = [sys.executable, os.path.join(root, "tools", "eval.py"), args.ckpt]
    if args.slices: cmd += ["--slices", str(args.slices)]
    if args.bytes: cmd += ["--bytes", str(args.bytes)]
    cmd += args.rest
    print("DELEGATE to tools/eval.py (held-out CE + SE, parameter/info, ring mode, half-life):")
    print(" ", " ".join(cmd))
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

def cmd_bench(args):
    root = _repo_root()
    # Delegate to bench_ring_credit (step-rate measurement).
    cmd = [sys.executable, os.path.join(root, "tools", "bench_ring_credit.py")]
    if args.ckpt: cmd += ["--ckpt", args.ckpt]
    cmd += args.rest
    print("DELEGATE to tools/bench_ring_credit.py (step rate / ring credit):")
    print(" ", " ".join(cmd))
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

# ----- main -----
parser = argparse.ArgumentParser(
    prog="tmt",
    description="TMT CLI — onboarding for the byte-level recurrent LM PoC. "
                "Reuses train_gpu.py (train), tools/eval.py (ruler), bench_ring_credit.py (bench), "
                "numpy_port (chat). Honest: corpora not in git; CPU-only OK; CUDA needs torch; "
                "model is ~0.56 nats/byte worse than order-8 byte n-gram.")
sub = parser.add_subparsers(dest="cmd", required=True)

# chat
p_chat = sub.add_parser("chat", help="Talk to a checkpoint (numpy generation path).")
p_chat.add_argument("--ckpt", default=None, help="checkpoint (.npz); default tmt.npz at repo root")
p_chat.add_argument("--prompt", default="hello", help="prompt string")
p_chat.add_argument("--threshold", type=float, default=0.35)
p_chat.add_argument("--batch", type=int, default=8)
p_chat.add_argument("--maxtokens", type=int, default=100)
p_chat.add_argument("--max-steps", type=int, default=1)
p_chat.add_argument("--data-seed", type=int, default=0)
p_chat.set_defaults(func=cmd_chat)

# train
p_train = sub.add_parser("train", help="Train on a directory of text (delegates to train_gpu.py).")
p_train.add_argument("--data-dir", default="data/wikipedia_clean")
p_train.add_argument("--steps", type=int, default=1000)
p_train.add_argument("--ckpt", default="tmt.npz")
p_train.add_argument("--keep", type=int, default=3)
p_train.add_argument("--tagged", action="store_true")
p_train.add_argument("rest", nargs=argparse.REMAINDER, help="extra flags for train_gpu.py")
p_train.set_defaults(func=cmd_train)

# eval
p_eval = sub.add_parser("eval", help="Score a checkpoint on held-out text (the ruler; prints CE +/- SE, params, ring, half-life).")
p_eval.add_argument("ckpt", nargs="?", default="tmt.npz", help="checkpoint file")
p_eval.add_argument("--slices", type=int, default=3)
p_eval.add_argument("--bytes", type=int, default=20000)
p_eval.add_argument("rest", nargs=argparse.REMAINDER)
p_eval.set_defaults(func=cmd_eval)

# bench
p_bench = sub.add_parser("bench", help="Measure step rate / ring credit (delegates to bench_ring_credit.py).")
p_bench.add_argument("--ckpt", default=None)
p_bench.add_argument("rest", nargs=argparse.REMAINDER)
p_bench.set_defaults(func=cmd_bench)

def main(args=None):
    ns = parser.parse_args(args)
    if ns.cmd == "chat":
        ns.func(ns)
    elif ns.cmd in ("train", "eval", "bench"):
        ns.func(ns)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
