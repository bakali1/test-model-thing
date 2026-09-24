#!/usr/bin/env python3
"""Kaggle GPU entrypoint for TMT (review #16: the Kaggle side was runbook-only).

One command, same engine and same env contract as Colab:

    TMT_HUB_URL=user/repo TMT_RUN_ID=expD-W TMT_STEPS=120000 \
    python kaggle/tmt_gpu_train.py

Everything is configured by environment variables so a Kaggle notebook cell is
just `!python kaggle/tmt_gpu_train.py` with secrets attached:

  TMT_STEPS       steps to train FROM THE RESUMED CHECKPOINT (relative)  [120000]
  TMT_BATCH       streams per step (A/B arms must state it)              [1024]
  TMT_LR          learning rate                                            [5e-4]
  TMT_SAVE_EVERY  steps between tagged saves                              [1000]
  TMT_KEEP_K      local tagged checkpoints to keep                        [8]
  TMT_DATA_SEED   file-order seed (pin per arm)                           [0]
  TMT_CHECKPOINT  rolling checkpoint path                                 [tmt.npz]
  TMT_DATASET     training glob                                           [corpus/current/**/corpus_*]
  TMT_FETCH       1 = run fetch_corpus.py first (in-session download)     [0]
  TMT_FETCH_ARGS  extra args for fetch_corpus.py            [--layers A,B,C --gb 4.5]
  TMT_HUB_URL     HF repo for pull+push      (TMT_HUB_REPO also accepted)
  TMT_TOKEN       HF write token (or use Kaggle secrets as env)
  TMT_RUN_ID      arm/run label recorded in checkpoint sidecars
  TMT_FRESH       1 = allow training from scratch (no checkpoint/hub needed) [0]
  TMT_DEVICE      cuda | cpu                                               [cuda]
  TMT_RING_ALGORITHM  legacy | factored (experimental)                   [legacy]
  TMT_TAIL_ALGORITHM  legacy | factored (experimental)                   [legacy]
  TMT_RING_CREDIT     full | cold | kloop | compiled | triton | segmented [full]
  TMT_EXTRA       extra args appended to train_gpu.py

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
        fargs = shlex.split(env("TMT_FETCH_ARGS", "--layers A,B,C --gb 4.5"))
        fcmd = [sys.executable, os.path.join(ROOT, "fetch_corpus.py")] + fargs
        print(f"[kaggle] fetch: {' '.join(shlex.quote(c) for c in fcmd)}")
        if not dry:
            subprocess.run(fcmd, check=True, cwd=ROOT)

    cmd = [sys.executable, os.path.join(ROOT, "train_gpu.py"), ckpt, "train",
           "--dataset", dataset, "--batch", str(batch),
           "--lr", env("TMT_LR", "5e-4"),
           "--save-every", env("TMT_SAVE_EVERY", "1000"),
           "--keep-k", env("TMT_KEEP_K", "8"),
           "--data-seed", env("TMT_DATA_SEED", "0"),
           "--steps", str(steps), "--device", device,
           "--ring-algorithm", env("TMT_RING_ALGORITHM", "legacy"),
           "--tail-algorithm", env("TMT_TAIL_ALGORITHM", "legacy"),
           "--ring-credit", env("TMT_RING_CREDIT", "full")]
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
