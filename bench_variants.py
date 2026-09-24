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
    # name: (ring_algorithm, tail_algorithm, ring_credit)
    "baseline": ("legacy", "legacy", "full"),
    "a1-cold": ("legacy", "legacy", "cold"),
    "a2-kloop": ("legacy", "legacy", "kloop"),
    "a2-compiled": ("legacy", "legacy", "compiled"),
    "a3-compile": ("legacy", "legacy", "full"),
    "b1-ring": ("factored", "legacy", "full"),
    "b2-tail": ("legacy", "factored", "full"),
    "b1-b2": ("factored", "factored", "full"),
    "c-segmented": ("legacy", "legacy", "segmented"),
    "c-triton": ("legacy", "legacy", "triton"),
    "b1-triton": ("factored", "legacy", "triton"),
}


def bytes_for(B, steps, device, seed=0xBEEF):
    # Inputs are generated on device, but are identical for every fresh model.
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = torch.randint(1, 250, (steps, B), generator=g, dtype=torch.int64)
    n = torch.randint(1, 250, (steps, B), generator=g, dtype=torch.int64)
    e = (torch.rand((steps, B), generator=g) < 0.02).float()
    return c.to(device), n.to(device), e.to(device)


def model_for(path, B, device, variant):
    ra, ta, rc = VARIANTS[variant]
    # The board is intended for the live 512x16 checkpoint, but infer the
    # shape so a small CPU fixture can exercise the same harness.
    with np.load(path) as d:
        dim = int(d["embed"].shape[1])
        layers = 1 + max((int(k.split(".")[1]) for k in d.files
                          if k.startswith("state.")), default=0)
    m = GpuModel(dim=dim, layers=layers, device=str(device),
                 ring_algorithm=ra, tail_algorithm=ta, ring_credit=rc,
                 # The explicit board selection is the migration request for
                 # factored candidates; the baseline remains strict.
                 allow_algo_migration=(ra != "legacy" or ta != "legacy"))
    m.new_states(B)
    if not m.load(path):
        raise FileNotFoundError(path)
    if variant == "a3-compile":
        if str(device) == "cpu":
            raise RuntimeError("a3-compile is a CUDA graph/compile candidate")
        # This is the measured A3 candidate. Async metrics are mandatory: a
        # Python float(loss) forces a host sync inside every captured step.
        m.sync_metrics = False
        m.step_batch = torch.compile(m.step_batch, mode="reduce-overhead", dynamic=False)
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
    result = {
        "variant": variant, "B/s": B / elapsed, "ms": elapsed * 1e3,
        "peak_GB": peak, "loss": last_loss,
        "loss_delta": abs(last_loss - reference_loss),
        "state_abs": err_abs, "state_nrel": err_nrel,
    }
    del m, state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def profile_baseline(path, B, steps, device):
    """Record actual kernel attribution rather than relying on traffic estimates."""
    from torch.profiler import ProfilerActivity, profile
    m = model_for(path, B, device, "baseline")
    cs, ns, ends = bytes_for(B, steps + 2, device)
    for i in range(2):
        m.step_batch(cs[i], ns[i], ends[i])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, record_shapes=True) as prof:
        m.step_batch(cs[2], ns[2], ends[2])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    print("\n[profile] baseline top kernel/operator averages")
    rows = []
    for e in prof.key_averages():
        # torch >=2.6 exposes self_device_time_total; older builds use
        # self_cuda_time_total (CUDA) / self_cpu_time_total.
        us = (getattr(e, "self_device_time_total", None)
              or getattr(e, "self_cuda_time_total", None)
              or e.self_cpu_time_total or 0.0)
        if us > 0:
            rows.append((float(us), e.key, int(e.count)))
    for us, name, count in sorted(rows, reverse=True)[:25]:
        print(f"  {us:10.1f} us  n={count:<3d} {name}")


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
    ap.add_argument("--variants", default="baseline,a1-cold,a2-kloop,b1-ring,b2-tail,b1-b2,c-segmented")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--atomic-microbench", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable")
    if not os.path.exists(args.checkpoint):
        ap.error(f"checkpoint not found: {args.checkpoint}")
    if args.steps < 3:
        ap.error("--steps must be >= 3 (two warmups plus at least one timed step)")

    cs, ns, ends = bytes_for(args.B, args.steps + 2, device)
    names = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = [x for x in names if x not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants: {unknown}; choose from {sorted(VARIANTS)}")
    print(f"device={device} B={args.B} steps={args.steps} checkpoint={args.checkpoint}")
    # The baseline is always built first and is the state/quality reference.
    base = run_variant(args.checkpoint, args.B, args.steps, device, cs, ns, ends,
                       "baseline", {})
    base_ref_state = None
    # Re-run baseline once to retain its state for comparisons; this is also
    # explicit about the fresh-model rule instead of reusing a mutated model.
    mref = model_for(args.checkpoint, args.B, device, "baseline")
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
        if name == "baseline":
            continue
        if name in ("a2-compiled", "a3-compile") and device.type != "cuda":
            print("skip a2-compiled on CPU (compile is a CUDA benchmark candidate)")
            continue
        if name in ("c-triton", "b1-triton") and (triton_missing(device)):
            print(f"skip {name}: Triton/CUDA unavailable")
            continue
        result = run_variant(args.checkpoint, args.B, args.steps, device, cs, ns,
                             ends, name, base_ref_state)
        results.append(result)

    print("\n[board] variant             B/s       ms/step   peak GB   loss d   state abs  state/RMS")
    for r in sorted(results, key=lambda x: x["B/s"], reverse=True):
        print(f"  {r['variant']:<18s} {r['B/s']:8.0f} {r['ms']:9.3f} "
              f"{r['peak_GB']:8.2f} {r['loss_delta']:9.2e} "
              f"{r['state_abs']:10.2e} {r['state_nrel']:10.2e}")
    if args.profile:
        try:
            profile_baseline(args.checkpoint, args.B, args.steps, device)
        except Exception as e:
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
