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


def save_checkpoint(model, rolling, keep_k, tagged, tag_dir=None):
    """Phase 0.3 pattern, crash-safe.

    tagged:  write tmt_t{step:06d}.npz, prune to last K, then point the rolling
             path at it via link-to-tmp + os.replace (atomic).
    untagged: write the rolling path directly through model.save's own
             .tmp + os.replace (atomic); no tags are created.
    """
    tag_dir = tag_dir or "."
    if not tagged:
        model.save(rolling)                      # atomic inside model.save
        return rolling
    tag = f"tmt_t{model.t:06d}.npz"
    tag_path = os.path.join(tag_dir, tag)
    model.save(tag_path)
    if keep_k > 0:
        olds = sorted(glob.glob(os.path.join(tag_dir, "tmt_t*.npz")))
        for old in olds[:-keep_k]:
            try:
                os.remove(old)
            except OSError:
                pass
    # rolling <- tag, atomically: link/copy to a temp name, then os.replace
    tmp = rolling + ".linktmp"
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
        os.link(tag_path, tmp)
    except OSError:
        model.save(tmp)                          # atomic, cross-fs fallback
    os.replace(tmp, rolling)
    return tag_path


class Trainer:
    def __init__(self, path, batch=1024, save_every=1000, keep_k=8,
                 tagged=True, max_steps=0, data_seed=0, device=Device,
                 **model_kw):
        self.model = GpuModel(device=str(device), **model_kw)
        self.path = path
        self.batch = batch
        self.save_every = save_every
        self.keep_k = keep_k
        self.tagged = tagged
        self.max_steps = max_steps          # "train N MORE steps" (relative)
        self.data_seed = data_seed
        self.device = device

    def train(self, save, dataset, hub_repo=None):
        m = self.model
        m.new_states(self.batch)
        with_hub = _with_hub(hub_repo)
        if hub_repo:
            with_hub(hub.pull, self.path)     # may write/link self.path
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            ck = np.load(self.path)
            ck_B = int(ck["B"]) if "B" in ck else None
            if ck_B is not None and ck_B != self.batch:
                print(f"  [warn] checkpoint B={ck_B} but --batch {self.batch}: "
                      f"recurrent rows resized (numpy_port does the same). A/B "
                      f"arms must state B + LR (PLAN §6).", flush=True)
            m.load(self.path)
            del ck
        # --steps is RELATIVE: train N more steps from wherever we resumed
        self.start_t = m.t
        self.stop_t = self.start_t + self.max_steps if self.max_steps else 0
        print(f"[train] resume t={self.start_t} -> "
              f"{self.stop_t or '∞'} | B={self.batch} | device {m.device}",
              flush=True)
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
                    if self.stop_t and m.t >= self.stop_t:
                        print(f"  [reached {self.max_steps} more steps "
                              f"(t={m.t}), stopping]", flush=True)
                        if save:
                            out = save_checkpoint(m, self.path, self.keep_k,
                                                  self.tagged)
                            if hub_repo and self.tagged:
                                with_hub(hub.push, out)
                            print(f"  [final] saved {os.path.basename(out)}",
                                  flush=True)
                        return
                    if save and self.save_every and m.t % self.save_every == 0:
                        out = save_checkpoint(m, self.path, self.keep_k,
                                              self.tagged)
                        if hub_repo and self.tagged:
                            with_hub(hub.push, out)
                        print(f"  [save] {os.path.basename(out)} t={m.t}",
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
        cuda = str(self.device).startswith("cuda")
        cs = torch.randint(1, 250, (max(Bs),), device=self.device)
        ns = torch.randint(1, 250, (max(Bs),), device=self.device)
        ends = torch.zeros(max(Bs), dtype=torch.float32, device=self.device)
        print(f"device {m.device} ring_k {m.ring_k} params {m.theta.shape[0]:,}")
        for B in Bs:
            m.new_states(B)
            for _ in range(4):
                m.step_batch(cs[:B], ns[:B], ends[:B])
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(steps):
                m.step_batch(cs[:B], ns[:B], ends[:B])
            if cuda:
                torch.cuda.synchronize()          # else timing is meaningless
            dt = (time.time() - t0) / steps
            peak = (torch.cuda.max_memory_allocated() / 1e9) if cuda else 0.0
            print(f"  B={B:5d}: {1/dt:7.1f} steps/s = {B/dt:9.0f} B/s "
                  f"({dt*1000:6.1f} ms/step) peak_alloc={peak:.2f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("mode", choices=["train", "bench"])
    ap.add_argument("--dataset", default="data/wikipedia_clean/**/wiki_*")
    ap.add_argument("--batch", type=int, default=1024,
                    help="streams per step (live checkpoint is "
                         "B=1024; a different B is a different "
                         "learning process — state B + LR in "
                         "every A/B arm)")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--decay-init", type=float, default=None)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep-k", type=int, default=8)
    ap.add_argument("--no-tagged", action="store_true",
                    help="overwrite the rolling checkpoint; no tags")
    ap.add_argument("--steps", type=int, default=0,
                    help="train N MORE steps from the resumed t "
                         "(0 = forever)")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--hub-repo", default=None,
                    help="HF model repo id: pull before, push after saves")
    ap.add_argument("--push", action="store_true",
                    help="shorthand: use $TMT_HUB_REPO/$TMT_HUB_URL for "
                         "pull+push")
    ap.add_argument("--bench-bs", default="1024,2048,4096")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.hub_repo is None and args.push:
        args.hub_repo = (os.environ.get("TMT_HUB_REPO")
                         or os.environ.get("TMT_HUB_URL"))
        if args.hub_repo is None:
            sys.exit("--push needs --hub-repo, $TMT_HUB_REPO or $TMT_HUB_URL")
    if args.hub_repo is None:
        args.hub_repo = (os.environ.get("TMT_HUB_REPO")
                         or os.environ.get("TMT_HUB_URL"))

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