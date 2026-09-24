#!/usr/bin/env python3
"""Phase 0 GPU optimization board.

Every variant gets a new GpuModel loaded from the same immutable checkpoint and
runs on exactly the same deterministic bytes.  The board intentionally reports
state error as max-abs / reference-RMS; it is a measurement tool, not a claim
that factored fp32 arithmetic is bitwise identical.

Example (on a T4):
    .venv/bin/python bench_variants.py tmt.npz --B 1024 --steps 12 --profile

CPU is supported for smoke tests, but CUDA timing/profiling is the useful mode.
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import torch

import gpu_port
from gpu_port import GpuModel

VARIANTS = {
    # name: (ring_credit, grad_assembly) -- only measured survivors remain.
    #   full   -- the portable reference (materialize + mask + index_add)
    #   auto   -- THE DEFAULT on CUDA: compiled (Inductor-fused ring credit)
    #   triton -- the fused-kernel fallback when Inductor is unavailable
    "full": ("full", "cat"),
    "auto": ("auto", "cat"),
    "triton": ("triton", "cat"),
    "g-loop": ("full", "loop"),
}
DEFAULT_VARIANTS = "full,auto,triton,g-loop"


def bytes_for(B, steps, device, seed=0xBEEF):
    # Inputs are generated on device, but are identical for every fresh model.
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = torch.randint(1, 250, (steps, B), generator=g, dtype=torch.int64)
    n = torch.randint(1, 250, (steps, B), generator=g, dtype=torch.int64)
    e = (torch.rand((steps, B), generator=g) < 0.02).float()
    return c.to(device), n.to(device), e.to(device)


TINY_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
            "main/TinyStories-train.txt")
TEXT_CACHE = os.environ.get("TMT_BENCH_TEXT", "tinystories_bench.txt")


def real_bytes_for(B, steps, device, mb=64):
    """The REAL training case: a 64 MB slice of TinyStories-train.txt shaped
    through numpy_port.Runtime.make_streams exactly as train_gpu does, so the
    benchmark sees the same byte matrix, end-of-stream density and stream
    structure as a live leg instead of uniform random bytes."""
    import urllib.request
    import numpy_port as np_
    # make_streams splits the buffer into B streams, so L (steps available per
    # stream) is roughly len(data)/B. Ask for 2.5x the requirement so the
    # shortest stream still clears `steps` with margin.
    need = int(steps * B * 2.5)
    if not (os.path.exists(TEXT_CACHE)
            and os.path.getsize(TEXT_CACHE) >= need):
        want = max(mb * 1_000_000, need)
        req = urllib.request.Request(TINY_URL, headers={"Range": f"bytes=0-{want}"})
        print(f"[data] fetching {want/1e6:.0f} MB TinyStories slice ...", flush=True)
        with urllib.request.urlopen(req, timeout=180) as r, \
                open(TEXT_CACHE + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(TEXT_CACHE + ".part", TEXT_CACHE)
    with open(TEXT_CACHE, "rb") as f:
        blob = f.read()
    # make_streams' bytes-per-step ratio is not simply len/B, so grow the
    # prefix until the shortest stream clears `steps` (the cache is 64 MB).
    prefix = max(need, int(steps * B * 4))
    last = None
    for _ in range(8):
        data = blob[:prefix]
        streams, endss = np_.Runtime.make_streams(data, B)
        L = min(len(s) for s in streams) - 1
        if L >= steps:
            break
        last = L
        prefix *= 2
    else:
        raise RuntimeError(f"corpus slice yields only {last} steps, need {steps} "
                           f"(have {len(blob)/1e6:.1f} MB, B={B})")
    cs = np.stack([np.frombuffer(s[:steps], dtype=np.uint8) for s in streams], axis=1)
    ns = np.stack([np.frombuffer(s[1:steps + 1], dtype=np.uint8) for s in streams], axis=1)
    es = np.array([e[:steps] for e in endss], dtype=np.float32).T
    print(f"[data] real TinyStories: {len(data)/1e6:.2f} MB, {steps} steps at "
          f"B={B}, end-of-stream density {float((es > 0).mean()):.3%}", flush=True)
    return (torch.from_numpy(cs.astype(np.int64)).to(device),
            torch.from_numpy(ns.astype(np.int64)).to(device),
            torch.from_numpy(es).to(device))


def model_for(path, B, device, variant):
    rc, ga = VARIANTS[variant]
    # The board is intended for the live 512x16 checkpoint, but infer the
    # shape so a small CPU fixture can exercise the same harness.
    with np.load(path) as d:
        dim = int(d["embed"].shape[1])
        layers = 1 + max((int(k.split(".")[1]) for k in d.files
                          if k.startswith("state.")), default=0)
    m = GpuModel(dim=dim, layers=layers, device=str(device),
                 ring_credit=rc, grad_assembly=ga)
    m.new_states(B)
    if not m.load(path):
        raise FileNotFoundError(path)
    return m


def snapshot(m):
    """Small CPU-side state snapshot used for scale-aware parity reporting."""
    out = {}
    for name, x in (("theta", m.theta), ("m", m.m), ("v", m.v),
                    ("grads", m.grads), ("ring_b", m.ring_b)):
        out[name] = x.detach().cpu().numpy().copy()
    for i, l in enumerate(m.layers):
        raw = l.ring_w.detach()
        out[f"ring_w.{i}"] = raw.cpu().numpy().copy()
        if hasattr(l.ring_w, "_factorized"):
            factor = l.ring_w._factor
            out[f"ring_factor.{i}"] = factor.detach().cpu().numpy().copy()
            out[f"ring_effective.{i}"] = (raw * factor).detach().cpu().numpy().copy()
        else:
            out[f"ring_effective.{i}"] = raw.cpu().numpy().copy()
        if m.tail is not None and i in m.tail.E:
            e = m.tail.E[i]
            out[f"tail_E.{i}"] = e.detach().cpu().numpy().copy()
            if i in m.tail.factor:
                factor = m.tail.factor[i]
                out[f"tail_factor.{i}"] = factor.detach().cpu().numpy().copy()
                out[f"tail_effective.{i}"] = (
                    e * factor[:, None, :]).detach().cpu().numpy().copy()
            else:
                out[f"tail_effective.{i}"] = e.detach().cpu().numpy().copy()
    return out


def state_error(a, b):
    """Return (max absolute, max absolute/reference RMS) over all state tensors."""
    max_abs = 0.0
    max_nrel = 0.0
    for key in sorted(set(a) & set(b)):
        if key == "_loss" or key.startswith("ring_w.") or key.startswith("tail_E."):
            continue
        if key.startswith("ring_factor.") or key.startswith("tail_factor."):
            continue
        x = np.asarray(a[key], dtype=np.float64)
        y = np.asarray(b[key], dtype=np.float64)
        if x.shape != y.shape:
            return float("inf"), float("inf")
        d = np.abs(x - y)
        if d.size:
            max_abs = max(max_abs, float(d.max()))
            rms = float(np.sqrt(np.mean(y * y)))
            max_nrel = max(max_nrel, float(d.max()) / (rms + 1e-30))
    return max_abs, max_nrel


def run_variant(path, B, steps, device, cs, ns, ends, variant, reference):
    m = model_for(path, B, device, variant)
    for i in range(2):  # warm compile, tail initialization, allocator
        m.step_batch(cs[i], ns[i], ends[i])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    losses = []
    t0 = time.perf_counter()
    for i in range(steps):
        # Keep the loss on-device: calling float(loss) here would force a host
        # sync inside the timed region and silently cancel the point of the
        # A3 async-metrics candidate.
        loss = m.step_batch(cs[i + 2], ns[i + 2], ends[i + 2])
        losses.append(loss.detach() if torch.is_tensor(loss) else loss)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - t0) / steps
    peak = (torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else 0.0)
    last_loss = (float(losses[-1].float().cpu())
                 if torch.is_tensor(losses[-1]) else float(losses[-1]))
    state = snapshot(m)
    if reference:
        err_abs, err_nrel = state_error(reference, state)
        reference_loss = reference.get("_loss", losses[-1])
    else:
        err_abs, err_nrel = 0.0, 0.0
        reference_loss = last_loss
    # Release the large ring/tail before constructing the next fresh model.
    hot = (sum(m.tail.nhot(i) for i in range(m.layercount))
           if m.tail is not None else 0)
    result = {
        "variant": variant, "B/s": B / elapsed, "ms": elapsed * 1e3,
        "peak_GB": peak, "loss": last_loss, "hot": hot,
        "loss_delta": abs(last_loss - reference_loss),
        "state_abs": err_abs, "state_nrel": err_nrel,
    }
    del m, state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def rebase_check(path, B, device, variant, total=620, window=100):
    """Does the candidate survive a tail rebase without a recompile stall?

    `_tail_sync` runs Python-level membership work inside step_batch, so a
    compiled step can recompile (or graph-break) every REBASE_EVERY steps. A
    short board run never reaches the first rebase, which is exactly how the
    12-step and 200-step runs missed this. Reports per-window B/s so a stall
    at the rebase step is visible.
    """
    m = model_for(path, B, device, variant)
    cs, ns, ends = bytes_for(B, total + 4, device)
    for i in range(2):
        m.step_batch(cs[i], ns[i], ends[i])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"\n[rebase-check] {variant}: {total} steps, REBASE_EVERY="
          f"{gpu_port.REBASE_EVERY}, per-{window}-step B/s")
    rates = []
    for w0 in range(0, total, window):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for i in range(w0, w0 + window):
            m.step_batch(cs[2 + i], ns[2 + i], ends[2 + i])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) / window
        rates.append(B / dt)
        print(f"  steps {w0:4d}-{w0+window:4d}: {B/dt:8.0f} B/s"
              f"{'   <-- rebase window' if w0 // 500 != (w0+window-1) // 500 else ''}")
    del m
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rates


def profile_baseline(path, B, steps, device, nsteps=3):
    """Kernel + PHASE attribution.

    The phase table is the actionable part: it uses the record_function regions
    in gpu_port.step_batch, so time is attributed to ring_credit / tail_credit /
    optimizer / fwd_layers / bwd_layers rather than to anonymous kernels. The
    'accounted' line compares the sum of leaf kernel device time with the wall
    clock of the same steps, so the gap (launch overhead, gaps, unaccounted
    ops) is explicit instead of being silently assumed away.
    """
    from torch.profiler import ProfilerActivity, profile
    m = model_for(path, B, device, "full")
    cs, ns, ends = bytes_for(B, max(steps, nsteps) + 2, device)
    for i in range(2):
        m.step_batch(cs[i], ns[i], ends[i])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)

    def us_of(e):
        # torch >=2.6 renamed self_cuda_time_total -> self_device_time_total;
        # probe the ladder so this works across versions.
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(e, attr, None)
            if v:
                return float(v), "device"
        return float(getattr(e, "self_cpu_time_total", 0.0) or 0.0), "cpu"

    with profile(activities=acts, record_shapes=True) as prof:
        t_wall = time.perf_counter()
        for i in range(nsteps):
            m.step_batch(cs[2 + i], ns[2 + i], ends[2 + i])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_us = (time.perf_counter() - t_wall) / nsteps * 1e6
    events = prof.key_averages()
    src = "device" if device.type == "cuda" else "cpu"
    # phase table
    phases, kernels = {}, []
    REGIONS = ("fwd_layers", "decoder", "loss_metrics", "bwd_decoder",
               "bwd_layers", "ring_credit", "tail_credit", "decaytrace",
               "grad_copy", "optimizer")
    for e in events:
        us, kind = us_of(e)
        if us <= 0:
            continue
        key = e.key
        if not key.startswith("aten::"):
            kernels.append((us, kind, key, int(e.count)))
        short = key.split("/")[-1] if "/" in key else key
        if short in REGIONS:
            phases[short] = phases.get(short, 0.0) + us
    print(f"\n[profile] baseline, B={B}, {nsteps} steps, "
          f"wall {wall_us/1000:.2f} ms/step, time source = {src}")
    if phases:
        tot = sum(phases.values())
        print(f"  {'phase':<14s} {'ms/step':>9s} {'% wall':>7s}")
        for name, us in sorted(phases.items(), key=lambda kv: -kv[1]):
            # key_averages() totals span the WHOLE profiled window, so divide by
            # nsteps before comparing against a per-step wall clock (otherwise
            # the table sums to nsteps x 100% and lies).
            print(f"  {name:<14s} {us/1000/nsteps:9.3f} "
                  f"{100*us/tot:6.1f}%")
        print(f"  {'TOTAL':<14s} {tot/1000/nsteps:9.3f} "
              f"{100.0:6.1f}%")
    else:
        print("  (no phase regions found -- was gpu_port regenerated?)")
    acct = sum(phases.values())
    print(f"  accounted by phase regions: {acct/1000/nsteps:.3f} ms/step "
          f"= {100*acct/(wall_us*nsteps):.1f}% of wall; the remainder is "
          f"launch overhead / profiler gaps / ops outside the regions")
    if device.type == "cuda" and acct < 0.7 * wall_us * nsteps:
        print("  WARNING: phases explain <70% of wall on CUDA -- the missing "
              "time is outside the instrumented regions (check for graph "
              "replay, unplanned syncs, or allocator stalls).")
    print(f"  top kernels ({src} time, per step; NOTE aten:: parents include "
          f"their children, so do NOT sum this column):")
    for us, kind, key, count in sorted(kernels, reverse=True)[:20]:
        print(f"  {us/1000/nsteps:9.3f} ms  n={count//nsteps:<4d} {key[:110]}")


def atomic_microbench(B, device, steps=10):
    """Isolated Phase-C diagnostic at the production scatter shape."""
    D, K = 512, 18
    vals = torch.randn(B * K, D, device=device)
    rows = torch.randint(0, 256, (B * K,), device=device)
    order = torch.argsort(rows, stable=True)
    lengths = torch.bincount(rows, minlength=256)
    cases = {
        "index_add": lambda: torch.zeros((256, D), device=device).index_add_(0, rows, vals),
        "sort+segment": lambda: torch.zeros((256, D), device=device).add_(
            torch.segment_reduce(vals.index_select(0, order), "sum", lengths=lengths)),
    }
    print("\n[microbench] isolated ring scatter")
    for name, fn in cases.items():
        for _ in range(2):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) / steps
        print(f"  {name:14s} {dt*1e3:8.3f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", default="full,auto,triton,g-loop")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-steps", type=int, default=3,
                    help="steps to profile (>=1); more steps amortize profiler "
                         "overhead and make the phase table stabler")
    ap.add_argument("--atomic-microbench", action="store_true")
    ap.add_argument("--data", choices=["synth", "text"], default="synth",
                    help="synth = uniform random bytes; text = real "
                         "TinyStories sliced through make_streams like a leg")
    ap.add_argument("--rebase-check", default="",
                    help="comma-separated variants to run for 620 steps with "
                         "per-100-step rates (catches recompile stalls at the "
                         "tail rebase that short runs never reach)")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable")
    if not os.path.exists(args.checkpoint):
        ap.error(f"checkpoint not found: {args.checkpoint}")
    if args.steps < 3:
        ap.error("--steps must be >= 3 (two warmups plus at least one timed step)")

    cs, ns, ends = (real_bytes_for(args.B, args.steps + 2, device)
                    if args.data == "text" else
                    bytes_for(args.B, args.steps + 2, device))
    names = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = [x for x in names if x not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants: {unknown}; choose from {sorted(VARIANTS)}")
    print(f"device={device} B={args.B} steps={args.steps} data={args.data} "
          f"checkpoint={args.checkpoint}")
    # The baseline is always built first and is the state/quality reference.
    base = run_variant(args.checkpoint, args.B, args.steps, device, cs, ns, ends,
                       "full", {})
    base_ref_state = None
    # Re-run the reference once to retain its state for comparisons; this is
    # also explicit about the fresh-model rule instead of reusing a mutated one.
    mref = model_for(args.checkpoint, args.B, device, "full")
    for i in range(2 + args.steps):
        mref.step_batch(cs[i], ns[i], ends[i])
    base_ref_state = snapshot(mref)
    base_ref_state["_loss"] = base["loss"]
    del mref
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = [base]
    for name in names:
        if name == "full":
            continue
        if name == "triton" and triton_missing(device):
            print(f"skip {name}: Triton/CUDA unavailable")
            continue
        result = run_variant(args.checkpoint, args.B, args.steps, device, cs, ns,
                             ends, name, base_ref_state)
        results.append(result)

    print("\n[board] variant             B/s       ms/step   peak GB   loss d   state abs  state/RMS  hot")
    for r in sorted(results, key=lambda x: x["B/s"], reverse=True):
        print(f"  {r['variant']:<18s} {r['B/s']:8.0f} {r['ms']:9.3f} "
              f"{r['peak_GB']:8.2f} {r['loss_delta']:9.2e} "
              f"{r['state_abs']:10.2e} {r['state_nrel']:10.2e} {r.get('hot',0):4d}")
    if args.rebase_check:
        for v in [x.strip() for x in args.rebase_check.split(",") if x.strip()]:
            if v in ("a2-compiled", "a3-compile") and device.type != "cuda":
                continue
            try:
                rebase_check(args.checkpoint, args.B, device, v)
            except Exception as e:
                print(f"[rebase-check] {v} failed: {type(e).__name__}: {e}")
    if args.profile:
        try:
            profile_baseline(args.checkpoint, args.B, args.steps, device,
                             nsteps=args.profile_steps)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[profile] unavailable: {type(e).__name__}: {e}")
    if args.atomic_microbench:
        try:
            atomic_microbench(args.B, device)
        except Exception as e:
            print(f"[microbench] unavailable: {type(e).__name__}: {e}")


def triton_missing(device):
    return device.type != "cuda" or gpu_port.triton is None


if __name__ == "__main__":
    main()
