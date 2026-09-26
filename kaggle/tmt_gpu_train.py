#!/usr/bin/env python3
"""Kaggle GPU entrypoint for TMT (review #16: the Kaggle side was runbook-only).

One command, same engine and same env contract as Colab:

    TMT_HUB_URL=user/repo TMT_RUN_ID=expD-W TMT_STEPS=120000 \
    python kaggle/tmt_gpu_train.py

Everything is configured by environment variables so a Kaggle notebook cell is
just `!python kaggle/tmt_gpu_train.py` with secrets attached:

  TMT_STEPS          steps to train FROM scratch/resume (relative)       [120000]
  TMT_BATCH          streams per step (A/B arms must state it)            [1024]
  TMT_DIM / TMT_LAYERS                                                   [512 / 16]
  TMT_RING_K         explicit retained-lag count; checkpoint mismatch fails [18]
  TMT_LR             learning rate                                         [5e-4]
  TMT_SEED           from-scratch model seed                               [0]
  TMT_SAVE_EVERY     local tagged-save interval                           [20000]
  TMT_HUB_SAVE_EVERY selected Hub-upload interval; 0 = final only       [40000]
  TMT_KEEP_K         local tags to keep; 0 = all                          [0]
  TMT_TELEMETRY_EVERY observation-only JSONL interval; 0 = disabled     [2000]
  TMT_MEASURE_AFTER  relative steps before steady-state timing starts     [1000]
  TMT_VERIFY_FINAL   1 = clean reload + digest verification at final     [1]
  TMT_DATA_SEED      file-order seed (pin per arm)                        [0]
  TMT_CHECKPOINT     rolling checkpoint path                              [tmt.npz]
  TMT_DATASET        training glob                    [corpus/current/**/corpus_*]
  TMT_FETCH          1 = run fetch_corpus.py first (in-session download)  [0]
  TMT_FETCH_ARGS     extra args for fetch_corpus.py
                     [--layers A,B,C --gb 0.9 --mb 6]
  TMT_HUB_URL        HF repo for pull+push      (TMT_HUB_REPO also accepted)
  TMT_TOKEN          HF write token (or use Kaggle secrets as env)
  TMT_RUN_ID         arm/run label recorded in checkpoint sidecars
  TMT_FRESH          1 = force seeded training from scratch               [0]
  TMT_DEVICE         cuda | cpu                                            [cuda]
  TMT_TAIL           1 = exact dense tail credit                           [1]
  TMT_RING_CREDIT    auto | full | compiled | triton                       [auto]
  TMT_EXTRA          extra args appended to train_gpu.py

Quota discipline: this script NEVER discovers datasets or spends time on
anything but the leg you asked for. It prints its exact command before
running it, and `--dry-run` stops there. Log the session in SCHEDULE.md.
"""
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def env(name, default=None):
    return os.environ.get(name, default)


def main():
    dry = "--dry-run" in sys.argv
    steps = int(env("TMT_STEPS", "120000"))
    batch = int(env("TMT_BATCH", "1024"))
    ckpt = env("TMT_CHECKPOINT", "tmt.npz")
    # fetch_corpus.py publishes <out>/current by default. Keep this path
    # aligned with that default so TMT_FETCH=1 can feed the trainer directly.
    dataset = env("TMT_DATASET", "corpus/current/**/corpus_*")
    repo = env("TMT_HUB_REPO") or env("TMT_HUB_URL")
    device = env("TMT_DEVICE", "cuda")
    run_id = env("TMT_RUN_ID")
    fetch = env("TMT_FETCH", "0") == "1"

    print(f"[kaggle] root={ROOT}")
    print(f"[kaggle] run_id={run_id or '-'} hub={repo or '-'} device={device}")
    print(f"[kaggle] steps={steps} (relative) batch={batch} ckpt={ckpt}")
    print(f"[kaggle] fresh={env('TMT_FRESH', '0') == '1'} "
          f"seed={env('TMT_SEED', '0')} data_seed={env('TMT_DATA_SEED', '0')}")
    print(f"[kaggle] ring_k={env('TMT_RING_K', '18')} "
          f"tail={env('TMT_TAIL', '1')} precision=fp32 tf32=disabled")
    print(f"[kaggle] dataset={dataset}")
    try:
        import torch
        if device == "cuda":
            print(f"[kaggle] torch {torch.__version__} "
                  f"cuda={torch.cuda.is_available()} "
                  f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    except ImportError as e:
        sys.exit(f"torch missing ({e}); on Kaggle use the GPU image")

    if fetch:
        fargs = shlex.split(env(
            "TMT_FETCH_ARGS", "--layers A,B,C --gb 0.9 --mb 6 --seed 42"))
        fcmd = [sys.executable, os.path.join(ROOT, "fetch_corpus.py")] + fargs
        print(f"[kaggle] fetch: {' '.join(shlex.quote(c) for c in fcmd)}")
        if not dry:
            subprocess.run(fcmd, check=True, cwd=ROOT)

    # GpuModel reads this before construction. Set it here so the printed
    # command and the actual process cannot disagree about tail semantics.
    os.environ["TMT_TAIL"] = env("TMT_TAIL", "1")
    cmd = [sys.executable, os.path.join(ROOT, "train_gpu.py"), ckpt, "train",
           "--dataset", dataset, "--batch", str(batch),
           "--dim", env("TMT_DIM", "512"),
           "--layers", env("TMT_LAYERS", "16"),
           "--lr", env("TMT_LR", "5e-4"),
           "--seed", env("TMT_SEED", "0"),
           "--save-every", env("TMT_SAVE_EVERY", "20000"),
           "--hub-save-every", env("TMT_HUB_SAVE_EVERY", "40000"),
           "--keep-k", env("TMT_KEEP_K", "0"),
           "--telemetry-every", env("TMT_TELEMETRY_EVERY", "2000"),
           "--measure-after", env("TMT_MEASURE_AFTER", "1000"),
           "--data-seed", env("TMT_DATA_SEED", "0"),
           "--steps", str(steps), "--device", device,
           "--ring-credit", env("TMT_RING_CREDIT", "auto"),
           "--ring-k", env("TMT_RING_K", "18")]
    if env("TMT_TELEMETRY_PATH"):
        cmd += ["--telemetry-path", env("TMT_TELEMETRY_PATH")]
    if env("TMT_VERIFY_FINAL", "1") == "1":
        cmd += ["--verify-final"]
    if env("TMT_FRESH", "0") == "1":
        cmd += ["--from-scratch"]
    if repo:
        cmd += ["--hub-repo", repo]
    if run_id:
        os.environ["TMT_RUN_ID"] = run_id
    cmd += shlex.split(env("TMT_EXTRA", ""))
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"[kaggle] train: {printable}")
    if dry:
        print("[kaggle] dry run; not executing")
        return 0
    checkpoint_parent = os.path.dirname(os.path.abspath(os.path.join(ROOT, ckpt)))
    os.makedirs(checkpoint_parent, exist_ok=True)
    if not os.path.exists(os.path.join(ROOT, ckpt)) and not repo \
            and env("TMT_FRESH", "0") != "1":
        sys.exit(f"no checkpoint at {ckpt} and no TMT_HUB_URL to pull one "
                 f"from — attach the hub repo, upload the .npz as a dataset, "
                 f"or set TMT_FRESH=1 to train from scratch")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    print(f"[kaggle] leg finished rc={rc} in {(time.time()-t0)/3600:.2f} h")
    print(f"[kaggle] LOG THIS SESSION in SCHEDULE.md "
          f"(platform=kaggle, run_id={run_id or '-'}, hours={(time.time()-t0)/3600:.2f})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
