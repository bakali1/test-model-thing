"""train_gpu.py — drive the gpu_port engine like numpy_port's trainer.

Same CLI semantics as numpy_port.py train mode (file globbing, line-bucketed
B streams via numpy_port.Runtime.make_streams, --data-seed order, tagged
checkpoints with --keep-k, --steps budget) but the step loop runs on
gpu_port.GpuModel on the selected device (cuda if available, else cpu).

Checkpoint continuity: .npz both ways — a numpy-written tmt.npz loads here
directly (bit-identical params/moments/ring/tail) and this trainer writes the
same format, so the CPU numpy trainer and eval tools can resume a GPU leg.

Optional hub sync (guarded by env, never on by default):
    TMT_HUB_REPO=<hf repo id>   pull latest tmt*.npz before training,
                                upload after each tagged save.
    TMT_TOKEN=<hf token>        (or logged in via huggingface_hub)
"""
import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import torch

import gpu_port
from gpu_port import GpuModel
import numpy_port as np_

import checkpoint_hub as hub

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _with_hub(repo):
    """Point checkpoint_hub at --hub-repo for the duration of the call."""
    def wrap(fn, *a, **k):
        old = os.environ.get("TMT_HUB_URL")
        if repo:
            os.environ["TMT_HUB_URL"] = repo
        try:
            return fn(*a, **k)
        finally:
            if old is None:
                os.environ.pop("TMT_HUB_URL", None)
            else:
                os.environ["TMT_HUB_URL"] = old
    return wrap


def tagged_save(model, rolling, keep_k, tag_dir=None):
    """Phase 0.3 pattern: tmt_t{step:06d}.npz, keep last K, hard-link rolling."""
    tag = f"tmt_t{model.t:06d}.npz"
    tag_path = os.path.join(tag_dir or ".", tag)
    model.save(tag_path)
    if keep_k > 0:
        olds = sorted(glob.glob(os.path.join(tag_dir or ".", "tmt_t*.npz")))
        for old in olds[:-keep_k]:
            try:
                os.remove(old)
            except OSError:
                pass
    try:
        if os.path.exists(rolling):
            os.remove(rolling)
        os.link(tag_path, rolling)
    except OSError:
        model.save(rolling)
    return tag_path


class Trainer:
    def __init__(self, path, batch=64, save_every=1000, keep_k=8,
                 tagged=True, max_steps=0, data_seed=0, device=Device,
                 **model_kw):
        self.model = GpuModel(device=str(device), **model_kw)
        self.path = path
        self.batch = batch
        self.save_every = save_every
        self.keep_k = keep_k
        self.tagged = tagged and keep_k >= 0
        self.max_steps = max_steps
        self.data_seed = data_seed
        self.device = device

    def train(self, save, dataset, hub_repo=None):
        m = self.model
        m.new_states(self.batch)
        with_hub = _with_hub(hub_repo)
        pulled = with_hub(hub.pull, self.path) if hub_repo else None
        if pulled is None and (os.path.exists(self.path)
                               and os.path.getsize(self.path) > 0):
            m.load(self.path)
        files = sorted(glob.glob(dataset, recursive=True))
        if not files:
            raise FileNotFoundError(f"no training files: {dataset!r}")
        random.Random(self.data_seed).shuffle(files)
        print(f"[data] seed={self.data_seed} | {len(files)} file(s): "
              f"{os.path.basename(files[0])} ... {os.path.basename(files[-1])}",
              flush=True)
        total = 0
        loss_win = ce_win = 0.0
        n_win = 0
        t0 = time.time()
        while True:
            for file in files:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    data = f.read().encode("utf-8", errors="ignore")
                if len(data) < self.batch * 4:
                    continue
                streams, endss = np_.Runtime.make_streams(data, self.batch)
                L = min(len(s) for s in streams) - 1
                if L < 1:
                    continue
                # vectorized per-file byte matrix (L+1, B) + end flags (L, B)
                mats = torch.from_numpy(np.stack(
                    [np.frombuffer(s[:L + 1], dtype=np.uint8) for s in streams],
                    axis=1).astype(np.int64)).to(self.device)
                ets = torch.from_numpy(np.array(
                    [e[:L] for e in endss], dtype=np.float32).T).to(self.device)
                for t in range(L):
                    loss_win += m.step_batch(mats[t], mats[t + 1], ets[t])
                    ce_win += m.last_ce
                    n_win += 1
                    total += self.batch
                    if self.max_steps and m.t >= self.max_steps:
                        print(f"  [reached step budget {self.max_steps}, "
                              f"stopping]")
                        if save:
                            tag = tagged_save(m, self.path,
                                              self.keep_k if self.tagged else 0)
                            if hub_repo:
                                with_hub(hub.push, tag)
                            print(f"  [final] saved {os.path.basename(tag)}",
                                  flush=True)
                        return
                    if save and self.save_every and m.t % self.save_every == 0:
                        tag = tagged_save(m, self.path,
                                          self.keep_k if self.tagged else 0)
                        if hub_repo:
                            with_hub(hub.push, tag)
                            print(f"  [hub] pushed {os.path.basename(tag)}",
                                  flush=True)
                    if total % (self.batch * 100) == 0:
                        dt = time.time() - t0
                        print(f"  {total/1e6:.1f} MB | {total/dt:.0f} B/s | "
                              f"step {m.t} | loss {loss_win/n_win:.3f} | "
                              f"ce {ce_win/n_win:.3f}", flush=True)
                        loss_win = ce_win = 0.0
                        n_win = 0

    def bench(self, Bs=(1024, 2048, 4096), steps=30):
        m = self.model
        cs = torch.randint(1, 250, (max(Bs),), device=self.device)
        ns = torch.randint(1, 250, (max(Bs),), device=self.device)
        ends = torch.zeros(max(Bs), dtype=torch.float32, device=self.device)
        print(f"device {m.device} ring_k {m.ring_k} params {m.theta.shape[0]:,}")
        for B in Bs:
            m.new_states(B)
            for _ in range(3):
                m.step_batch(cs[:B], ns[:B], ends[:B])
            t0 = time.time()
            for _ in range(steps):
                m.step_batch(cs[:B], ns[:B], ends[:B])
            dt = (time.time() - t0) / steps
            print(f"  B={B:5d}: {1/dt:7.1f} steps/s = {B/dt:9.0f} B/s "
                  f"({dt*1000:6.1f} ms/step)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("mode", choices=["train", "bench"])
    ap.add_argument("--dataset", default="data/wikipedia_clean/**/wiki_*")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--decay-init", type=float, default=None)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep-k", type=int, default=8)
    ap.add_argument("--no-tagged", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--hub-repo", default=None,
                    help="HF model repo id (or $TMT_HUB_REPO): pull before, push after saves")
    ap.add_argument("--bench-bs", default="1024,2048,4096")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.hub_repo is None:
        args.hub_repo = os.environ.get("TMT_HUB_REPO")

    t = Trainer(path=args.path, batch=args.batch, save_every=args.save_every,
                keep_k=args.keep_k, tagged=not args.no_tagged,
                max_steps=args.steps, data_seed=args.data_seed,
                device=args.device, dim=args.dim, layers=args.layers,
                lr=args.lr, decay_init=args.decay_init)
    print(f"parameters: {t.model.theta.shape[0]:,} | device {args.device}",
          flush=True)
    if args.mode == "train":
        t.train(not args.no_save, args.dataset, args.hub_repo)
    else:
        t.bench([int(x) for x in args.bench_bs.split(",")])


if __name__ == "__main__":
    main()