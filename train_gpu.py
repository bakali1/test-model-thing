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
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch

import gpu_port
import distill as _distill
from gpu_port import GpuModel
import numpy_port as np_
import teacher_table as _tt

import checkpoint_hub as hub
from telemetry import RunTelemetry

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
    """Phase 0.3 pattern, crash-safe, A/B-safe.

    The tag name is DERIVED FROM THE ROLLING BASENAME, so two arms in one
    directory never collide: armW.npz -> armW_t000001.npz, armN.npz ->
    armN_t000001.npz, each pruned only against its own prefix. (Rolling names
    like ``tmt.npz`` keep the historical ``tmt_t*.npz`` convention.)

    tagged:  write <base>_t{step:06d}.npz, prune to last K, then point the
             rolling path at it via link-to-tmp + os.replace (atomic).
    untagged: write the rolling path directly through model.save's own
             .tmp + os.replace (atomic); no tags are created.
    """
    rolling = os.path.abspath(rolling)
    base = os.path.splitext(os.path.basename(rolling))[0]
    base = re.sub(r"_t\d+$", "", base)          # rolling may itself be a tag
    tag_dir = tag_dir or os.path.dirname(rolling)
    if not tagged:
        model.save(rolling)                      # atomic inside model.save
        return rolling
    tag = f"{base}_t{model.t:06d}.npz"
    tag_path = os.path.join(tag_dir, tag)
    model.save(tag_path)
    if keep_k > 0:
        olds = sorted(glob.glob(os.path.join(tag_dir, f"{base}_t*.npz")))
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


class DistillTargets:
    """Per-file Marginalize-It targets for the trainer.

    Built lazily, once per corpus file, from the dump written by
    tools/mi_dump.py. The alignment contract is the whole point: the dump was
    bucketed with the SAME `make_streams(data, B)` call, and MiReader refuses to
    load unless the dump's token lengths imply exactly the stream lengths it was
    handed. A mismatch is a hard error, because the alternative is a KD arm
    quietly training on targets belonging to other bytes.

    Every KD arm prints coverage at the end. Coverage is NOT 1.0: the paper's
    Marginalize-It is defined at EVERY byte depth, but two things remove
    positions. (a) The first byte of a stream has no teacher row. (b) Under
    top-k truncation the surviving mass at deeper positions falls under
    `floor` and the position is declined rather than built from noise --
    measured 64.6% coverage on real text (RESULTS.md 2026-10-04), against
    23.1% for the depth-0-only target this replaced.
    """

    def __init__(self, dump_dir, corpus_file, batch, block=32,
                 repo="LiquidAI/LFM2.5-230M-Base", verbose=True, floor=0.05):
        self.path = dump_dir
        self.batch = batch
        self.block = block
        self.repo = repo
        self.verbose = verbose
        self.floor = floor
        self.corpus_file = corpus_file
        self.stem = os.path.splitext(os.path.basename(corpus_file))[0]
        self.reader = None
        self._bind = 0
        self._vocab = None
        self._torch = torch

    def _ensure(self):
        if self.reader is not None:
            return self.reader
        tok_path = _tt.fetch_tokenizer_json(self.repo)
        table, meta = _tt.build_table(tok_path)
        control = set(meta["control_ids"]) | set(meta["nonbyte_ids"])
        vocab = _distill.VocabBytes(table, drop_ids=control)
        with open(self.corpus_file, "rb") as fh:
            data = fh.read()
        streams, _ends = np_.Runtime.make_streams(data, self.batch)
        r = _distill.MiReader(self.path, self.stem, vocab, streams=streams,
                              block=self.block, floor=self.floor)
        if self.verbose:
            print(f"  [distill] {self.stem}: B={r.B} L_step={r.L_step} "
                  f"Ltok={r.Ltok} k={r.k}", flush=True)
        self.reader = r
        self._vocab = vocab
        return r

    def at(self, model_step):
        """(B,256) float32 for this step.

        `model_step` is the ENGINE's counter, which step_batch increments BEFORE
        calling the provider. With `bind(m.t)` taken while m.t still pointed at
        the previous step, the first call for this file arrives as bind+1, so
        the file-local step is `model_step - bind - 1`.
        """
        r = self._ensure()
        t = int(model_step) - self._bind - 1
        if not (0 <= t < r.L_step):
            return None
        tgt, _ret = r.targets(t)
        return self._torch.from_numpy(np.ascontiguousarray(tgt)).float()

    def bind(self, model_step_before_file):
        self._bind = int(model_step_before_file)
        return self

    def stats(self):
        return self.reader.stats() if self.reader is not None else {}


class Trainer:
    def __init__(self, path, batch=1024, save_every=1000, keep_k=8,
                 tagged=True, max_steps=0, data_seed=0, device=Device,
                 from_scratch=False, seed=0, log_every=0, warmup=0,
                 hub_save_every=0, telemetry_every=0, telemetry_path=None,
                 measure_after=0, verify_final=False, distill_dir=None,
                 distill_alpha=0.0, distill_block=32, distill_repo=None,
                 distill_null_target=False, distill_floor=0.05, **model_kw):
        # The init is drawn from a numpy Generator seeded inside GpuModel, so
        # --seed is threaded through to make starting weights reproducible. Two
        # from-scratch A/B arms with the same seed start bit-identical, which
        # is what makes them byte-matched on the data axis alone.
        #
        # The distillation block runs BEFORE the model is built: `model_kw` is
        # consumed here, so injecting distill_alpha after this line silently
        # trained a CE arm while the log claimed alpha was set.
        self.distill_dir = distill_dir
        self.distill_alpha = float(distill_alpha)
        self.distill_block = int(distill_block)
        self.distill_repo = distill_repo or os.environ.get(
            "TMT_TEACHER", "LiquidAI/LFM2.5-230M-Base")
        self.distill = None
        self.distill_null_target = bool(distill_null_target)
        self.distill_floor = float(distill_floor)
        if self.distill_alpha > 0.0:
            if self.distill_null_target:
                # The half-CE control. alpha > 0 so ce_weight == 1-alpha and the
                # CE gradient is down-weighted exactly as in a real KD arm; the
                # target provider returns zeros so the KD term contributes no
                # gradient. NO dump is read, so this arm cannot be confounded by
                # target content -- only by the weighting.
                model_kw["distill_alpha"] = self.distill_alpha
                print(f"  [distill] NULL-TARGET CONTROL: alpha="
                      f"{self.distill_alpha}, CE weight "
                      f"{1.0 - self.distill_alpha}, KD contributes nothing")
            elif not distill_dir:
                raise SystemExit("--distill-alpha > 0 requires --distill-dir "
                                 "(the dump made by tools/mi_dump.py)")
            else:
                model_kw["distill_alpha"] = self.distill_alpha
        self.model = GpuModel(device=str(device), seed=seed, **model_kw)
        self.path = path
        self.seed = int(seed)
        # 0 => log every batch*100 steps (the historical cadence). A short
        # calibration run needs a much smaller interval to report a rate.
        self.log_every = int(log_every) or batch * 100
        # Steps run before the throughput clock starts (compile, allocator,
        # autotune). A calibration run MUST set this or it measures warmup.
        self.warmup = int(warmup)
        self.from_scratch = bool(from_scratch)
        self.batch = batch
        self.save_every = save_every
        self.keep_k = keep_k
        self.tagged = tagged
        self.max_steps = max_steps          # "train N MORE steps" (relative)
        self.data_seed = data_seed
        self.device = device
        # Local saves and Hub uploads intentionally have independent cadences.
        # hub_save_every=0 means "periodic local saves only"; the final tagged
        # checkpoint is always pushed when a Hub repo was supplied.
        self.hub_save_every = int(hub_save_every)
        self.telemetry_every = int(telemetry_every)
        self.telemetry_path = telemetry_path or (self.path + ".telemetry.jsonl")
        self.measure_after = int(measure_after)
        self.verify_final = bool(verify_final)
        self.telemetry = None

    def _kd_provider(self, file, m):
        """Fresh per-file provider, bound to the engine's step counter."""
        if self.distill_alpha <= 0.0:
            return None
        if self.distill_null_target:
            zero = torch.zeros(self.batch, 256)
            m.set_target_fn(lambda step, _z=zero: _z)
            return None
        p = DistillTargets(self.distill_dir, file, self.batch,
                           self.distill_block, self.distill_repo,
                           floor=self.distill_floor)
        p.bind(m.t)
        prov = p
        m.set_target_fn(lambda step, _p=prov: _p.at(step))
        return p

    def _kd_report(self, m):
        if self.distill_alpha <= 0.0:
            return
        st = m.kd_stats()
        if self.distill_null_target:
            print(f"  [distill] NULL-TARGET CONTROL: alpha={st['alpha']}, "
                  f"CE weight {1.0 - st['alpha']}, KD covered "
                  f"{st['covered']}/{st['positions']} (expected 0)", flush=True)
            return
        rs = self.distill.stats() if self.distill is not None else {}
        # Coverage is NOT a property of the target definition alone. It is
        # (definition x floor), and over the plausible floor range it swings from
        # 85.2% (floor=0) to 55.3% (floor=0.20) on one fixed 46k-token dump --
        # a 30-point spread. Two runs on the same dump reported 64.38% and
        # 60.50% and looked like a reproducibility failure; they were
        # floor=0.05 and floor~=0.09 on identical bytes. So the floor is
        # printed with the coverage, always. See RESULTS.md 2026-10-04.
        extra = ""
        if rs:
            extra = (f" | floor={rs.get('floor')}"
                     f" retained_mean={rs.get('retained_mean'):.3f}"
                     f" entropy={rs['target_entropy_nats']:.3f} nats"
                     f" mean_depth={rs['target_mean_depth']:.2f}"
                     f" top1==gold={rs['top1_gold_fraction']:.1%}")
        print(f"  [distill] alpha={st['alpha']} | KD target supplied at "
              f"{st['covered']:,}/{st['positions']:,} positions "
              f"({(st['coverage'] or 0) * 100:.2f}% coverage at the floor "
              f"above; the rest fell back to CE-only){extra}", flush=True)

    def train(self, save, dataset, hub_repo=None):
        m = self.model
        m.new_states(self.batch)
        with_hub = _with_hub(hub_repo)
        if self.from_scratch:
            # Deliberately t=0: no pull, no load, no inherited state. The
            # experiment asks what a model can learn from ONE corpus, so any
            # prior exposure to the arm's own data would confound the verdict.
            print(f"  [from-scratch] t=0, seeded init (seed={self.seed}), "
                  f"ring_k={m.ring_k}; no checkpoint pulled or loaded",
                  flush=True)
        else:
            if hub_repo:
                with_hub(hub.pull, self.path, expect_n=m.theta.numel(),
                         require_digest=True)     # may write/link self.path
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                ck = np.load(self.path)
                ck_B = int(ck["B"]) if "B" in ck else None
                if ck_B is not None and ck_B != self.batch:
                    print(f"  [warn] checkpoint B={ck_B} but --batch {self.batch}: "
                          f"recurrent rows resized (numpy_port does the same). "
                          f"A/B arms must state B + LR (PLAN §6).", flush=True)
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
        # Warmup steps run BEFORE the clock starts. Without this, a short
        # calibration run reports torch.compile's Inductor time as if it were
        # steady-state throughput: measured 1,081 B/s against a true 27.5 kB/s,
        # purely because compile dominated a 24-step window.
        if self.warmup > 0:
            cuda_w = str(self.device).startswith("cuda")
            wsteps = 0
            for file in files:
                if wsteps >= self.warmup:
                    break
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    data = f.read().encode("utf-8", errors="ignore")
                if len(data) < self.batch * 4:
                    continue
                streams, endss = np_.Runtime.make_streams(data, self.batch)
                L = min(len(s) for s in streams) - 1
                if L < 1:
                    continue
                mats = torch.from_numpy(np.stack(
                    [np.frombuffer(s[:L + 1], dtype=np.uint8) for s in streams],
                    axis=1).astype(np.int64)).to(self.device)
                ets = torch.from_numpy(np.array(
                    [e[:L] for e in endss], dtype=np.float32).T).to(self.device)
                self.distill = self._kd_provider(file, m)
                for t in range(L):
                    m.step_batch(mats[t], mats[t + 1], ets[t])
                    wsteps += 1
                    if wsteps >= self.warmup:
                        break
            if cuda_w:
                torch.cuda.synchronize()
            print(f"  [warmup] {wsteps} steps discarded before timing",
                  flush=True)
            # Warmup advanced m.t, which would otherwise eat the requested step
            # budget (stop_t was computed from start_t before warmup). Re-base
            # so --steps still means "this many TIMED steps"; warmup is extra.
            if self.max_steps:
                self.start_t = m.t
                self.stop_t = m.t + self.max_steps
        if self.telemetry_every:
            self.telemetry = RunTelemetry(
                self.telemetry_path, self.batch,
                measure_after=self.start_t + self.measure_after)
            if self.measure_after == 0:
                self.telemetry.mark_measurement(m)

        def digest_of(path):
            return hub.sha256_file(path) if self.telemetry is not None else None

        def push_selected(path, final=False):
            # A zero cadence disables periodic uploads, never the final safety
            # push. This prevents a 200+ MB upload from stalling every local
            # 20k checkpoint.
            periodic = (self.hub_save_every > 0 and
                        m.t % self.hub_save_every == 0)
            if hub_repo and self.tagged and (final or periodic):
                with_hub(hub.push, path)
                return True
            return False

        def verify_saved(path):
            """Reload through a clean engine and verify transport identity."""
            probe = GpuModel(device="cpu", dim=m.dim, layers=m.layercount,
                             ring_k=m.ring_k, ring_credit="full")
            probe.new_states(self.batch)
            probe.load(path)
            with np.load(path) as d:
                ck_t = int(d["t"])
                ck_B = int(d["B"])
                ck_K = int(d["ring_k"])
            if (probe.t, probe.B, probe.ring_k) != (ck_t, ck_B, ck_K):
                raise RuntimeError(
                    f"checkpoint reload identity mismatch: engine="
                    f"{(probe.t, probe.B, probe.ring_k)} file={(ck_t, ck_B, ck_K)}")
            if (ck_t, ck_B, ck_K) != (m.t, self.batch, m.ring_k):
                raise RuntimeError(
                    f"checkpoint content mismatch: engine={(m.t, self.batch, m.ring_k)} "
                    f"file={(ck_t, ck_B, ck_K)}")
            sha = hub.sha256_file(path)
            side = path + ".sha256"
            if os.path.exists(side):
                side_sha = hub._read_sidecar(side)
                if side_sha != sha:
                    raise RuntimeError(f"local digest sidecar mismatch for {path}")
                print(f"  [verify] clean reload PASS t={ck_t} B={ck_B} K={ck_K} "
                      f"sha256={sha}", flush=True)
            else:
                print(f"  [verify] clean reload PASS t={ck_t} B={ck_B} K={ck_K} "
                      f"sha256={sha} (no sidecar yet)", flush=True)
            return sha

        def emit_telemetry(digest=None):
            if self.telemetry is not None:
                self.telemetry.maybe_emit(m, self.telemetry_every, digest)

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
                self.distill = self._kd_provider(file, m)
                for t in range(L):
                    loss = m.step_batch(mats[t], mats[t + 1], ets[t])
                    if self.telemetry is not None:
                        self.telemetry.observe(loss, m.last_ce)
                        if m.t == self.start_t + self.measure_after:
                            self.telemetry.mark_measurement(m)
                    if not math.isfinite(float(loss)):
                        if self.telemetry is not None:
                            self.telemetry.nonfinite_loss(m, loss, m.last_ce)
                        print(f"[fatal] non-finite loss at t={m.t}: {loss}",
                              flush=True)
                        raise FloatingPointError(f"non-finite loss at t={m.t}")
                    loss_win += loss
                    ce_win += m.last_ce
                    n_win += 1
                    total += self.batch
                    if self.stop_t and m.t >= self.stop_t:
                        print(f"  [reached {self.max_steps} more steps "
                              f"(t={m.t}), stopping]", flush=True)
                        digest = None
                        if save:
                            out = save_checkpoint(m, self.path, self.keep_k,
                                                  self.tagged)
                            push_selected(out, final=True)
                            digest = digest_of(out)
                            if self.verify_final:
                                digest = verify_saved(out)
                            print(f"  [final] saved {os.path.basename(out)}",
                                  flush=True)
                        self._kd_report(m)
                        emit_telemetry(digest if save else None)
                        return
                    if save and self.save_every and m.t % self.save_every == 0:
                        out = save_checkpoint(m, self.path, self.keep_k,
                                              self.tagged)
                        push_selected(out, final=False)
                        digest = digest_of(out)
                        if self.verify_final:
                            # Full clean reload is required only for the final
                            # artifact; a periodic tag is covered by that final
                            # gate without doubling 200+ MB reload costs.
                            pass
                        print(f"  [save] {os.path.basename(out)} t={m.t}",
                              flush=True)
                        emit_telemetry(digest)
                        continue
                    emit_telemetry()
                    if self.log_every and total % self.log_every == 0:
                        dt = time.time() - t0
                        # Async metric mode keeps CUDA work queued; synchronize
                        # only at the logging boundary, not once per step.
                        loss_value = float(loss_win / n_win)
                        ce_value = float(ce_win / n_win)
                        print(f"  {total/1e6:.1f} MB | {total/dt:.0f} B/s | "
                              f"step {m.t} | loss {loss_value:.3f} | "
                              f"ce {ce_value:.3f}", flush=True)
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
    ap.add_argument("--ring-k", type=int, default=None,
                    help="explicit retained-lag count; when set, loading a "
                         "checkpoint with a different stored ring_k fails")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-decay", type=float, default=None,
                    help="separate learning rate for the 16 per-layer .decay "
                         "slices (8,192 coords = 0.183%% of all parameters); "
                         "everything else keeps --lr. Omitted = one LR "
                         "everywhere, the historical behaviour. Resumable: the "
                         "AdamW state format is unchanged, so arms can branch "
                         "off the incumbent checkpoint")
    ap.add_argument("--decay-init", type=float, default=None)
    ap.add_argument("--from-scratch", action="store_true",
                    help="start at t=0 with a seeded random init; no pull, no "
                         "load. For corpus A/B arms where inherited exposure to "
                         "the arm's own data would confound the verdict.")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch seed for the from-scratch init (0 = the "
                         "historical default). Use the SAME seed across arms "
                         "so their starting weights are bit-identical.")
    ap.add_argument("--log-every", type=int, default=0,
                    help="print the B/s/loss line every N steps (0 = every "
                         "batch*100). Short calibration runs need this to "
                         "report a throughput rate at all.")
    ap.add_argument("--warmup", type=int, default=0,
                    help="run N steps before starting the throughput clock "
                         "(torch.compile/allocator/autotune). Required for a "
                         "short calibration run to report a real rate.")
    ap.add_argument("--ring-credit",
                    choices=("auto", "full", "compiled", "triton", "nvrtc"),
                    default="auto",
                    help="auto = Inductor-fused on CUDA (+27%% measured), full "
                         "on CPU, and nvrtc below compute capability 7.0 where "
                         "Triton cannot run at all; triton = fused-kernel "
                         "fallback; nvrtc = NVRTC fused decay+credit+scatter, "
                         "the only fused path on pre-Volta GPUs (+33%% there)")
    ap.add_argument("--grad-assembly", choices=("cat", "loop"), default="cat",
                    help="cat = one fused gradient-buffer write (default, "
                         "bit-identical); loop = 69-copy reference for A/B")
    ap.add_argument("--async-metrics", action="store_true",
                    help="do not synchronize loss/CE every step; enables compile/graph experiments")
    ap.add_argument("--compile-step", action="store_true",
                    help="wrap the steady-state step in torch.compile/reduce-overhead "
                         "(the measured A3 candidate; requires --async-metrics)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep-k", type=int, default=8,
                    help="local tagged checkpoints to keep; 0 keeps all")
    ap.add_argument("--hub-save-every", type=int, default=0,
                    help="push a local tag only when t is divisible by this; "
                         "0 pushes the final tag only")
    ap.add_argument("--telemetry-every", type=int, default=0,
                    help="emit observation-only JSONL every N absolute steps "
                         "(0 disables telemetry)")
    ap.add_argument("--telemetry-path", default=None,
                    help="JSONL path (default: CHECKPOINT.telemetry.jsonl)")
    ap.add_argument("--measure-after", type=int, default=0,
                    help="start steady-state throughput clock after N relative "
                         "training steps; training is not rewound or repeated")
    ap.add_argument("--distill-dir", default=os.environ.get("TMT_MI_DUMP"),
                    help="directory written by tools/mi_dump.py; one dump per "
                         "corpus file, bucketed with the SAME B as training")
    ap.add_argument("--distill-alpha", type=float,
                    default=float(os.environ.get("TMT_KD_ALPHA", "0")),
                    help="KD weight. alpha=0 is the CE baseline and is "
                         "BIT-EXACT vs the pre-KD engine (tests/check_kd.py). "
                         "The paper's mixed objective is alpha*KL + "
                         "(1-alpha)*CE.")
    ap.add_argument("--distill-block", type=int, default=32,
                    help="steps of targets computed per vectorized block")
    ap.add_argument("--distill-floor", type=float, default=0.05,
                    help="drop a target whose surviving top-k mass is <= this. "
                         "QUOTE COVERAGE WITH IT: on one fixed 46k-token dump "
                         "coverage runs 85.2%% (floor=0) / 64.4%% (0.05) / "
                         "60.5%% (0.09) / 55.3%% (0.20). Coverage is "
                         "(definition x floor), not a target statistic on its "
                         "own.")
    ap.add_argument("--distill-null-target", action="store_true",
                    help="CONTROL ARM: run with alpha>0 but an all-zero target, "
                         "so the CE term is down-weighted to (1-alpha) exactly "
                         "as in a KD arm while the KD term contributes nothing. "
                         "This is the arm that separates 'distillation hurts' "
                         "from 'halving CE hurts': without it, a KD arm losing "
                         "to a CE control is uninterpretable. Needs no dump.")
    ap.add_argument("--distill-repo", default=None,
                    help="HF repo for the teacher's tokenizer.json "
                         "(default $TMT_TEACHER)")
    ap.add_argument("--verify-final", action="store_true",
                    help="reload the final tag through a clean CPU engine and "
                         "verify t/B/K plus its local SHA-256 sidecar")
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
    if args.compile_step and not args.async_metrics:
        sys.exit("--compile-step requires --async-metrics (a host float(loss) "
                 "prevents CUDA graph/compile capture)")
    if args.telemetry_every and args.async_metrics:
        sys.exit("--telemetry-every cannot be combined with --async-metrics; "
                 "telemetry needs finite host loss/CE values at its interval")
    if args.no_tagged and (args.push or args.hub_repo
                          or os.environ.get("TMT_HUB_REPO")
                          or os.environ.get("TMT_HUB_URL")):
        sys.exit("--no-tagged writes only the rolling checkpoint, which the "
                 "hub protocol never uploads (it pushes tagged files). Drop "
                 "--no-tagged, or drop the hub flags.")
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
                ring_k=args.ring_k, lr=args.lr, lr_decay=args.lr_decay,
                decay_init=args.decay_init,
                ring_credit=args.ring_credit,
                grad_assembly=args.grad_assembly,
                sync_metrics=not args.async_metrics,
                from_scratch=args.from_scratch, seed=args.seed,
                log_every=args.log_every, warmup=args.warmup,
                hub_save_every=args.hub_save_every,
                telemetry_every=args.telemetry_every,
                telemetry_path=args.telemetry_path,
                measure_after=args.measure_after,
                verify_final=args.verify_final,
                distill_dir=args.distill_dir,
                distill_alpha=args.distill_alpha,
                distill_block=args.distill_block,
                distill_repo=args.distill_repo,
                distill_null_target=args.distill_null_target,
                distill_floor=args.distill_floor)
    if args.distill_alpha > 0.0 and not args.distill_null_target:
        if not args.distill_dir or not os.path.isdir(args.distill_dir):
            sys.exit(f"--distill-alpha {args.distill_alpha} but --distill-dir "
                     f"{args.distill_dir!r} is not a directory; run "
                     f"tools/mi_dump.py first (the KD arm must not silently "
                     f"degrade to CE)")
        print(f"[distill] alpha={args.distill_alpha} | dump={args.distill_dir} "
              f"| block={args.distill_block} | B={args.batch} (the dump must "
              f"have been written with this same B)", flush=True)
    print(f"parameters: {t.model.theta.shape[0]:,} | device {args.device}",
          flush=True)
    if args.mode == "train":
        t.train(not args.no_save, args.dataset, args.hub_repo)
    else:
        t.bench([int(x) for x in args.bench_bs.split(",")])


if __name__ == "__main__":
    main()