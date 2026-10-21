"""tools/bench_step_paths.py -- the WHOLE step, once per ring-credit path.

The credit benchmark (tools/bench_ring_credit.py) times the credit ALONE,
which is the right instrument for "is the credit the wall" but the wrong one
for "what does the credit cost the step": the paths differ in more than the
credit (Inductor's per-call overhead, for one). This times `step_batch`
itself, once per path, with a FRESH model per arm so that the arms see the
same seed, the same bytes and the same state -- and it prints the final
loss beside the timing, because the paths must agree on it to the last
digit. That is the bit-identity claim, end to end, in one number.

Measured on a T4 (B=1024, dim=512, 16 layers, 40 timed steps):

    full      36.02 ms/step   loss 7.58001
    compiled  33.02 ms/step   loss 7.58001   <- the default on CUDA
    nvrtc     28.22 ms/step   loss 7.58001   <- 1.17x the default
    grouped   28.43 ms/step   loss 7.58001

Usage (on a GPU box):
    python tools/bench_step_paths.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gpu_port  # noqa: E402

B, DIM, LAYERS = 1024, 512, 16


def run(path, dev, steps=40, warmup=8):
    """One arm: a fresh model, the same seed, the same bytes."""
    m = gpu_port.GpuModel(device=str(dev), dim=DIM, layers=LAYERS, seed=0,
                           ring_k=18)
    m.new_states(B)
    m.ring_credit = path
    m._ring_resolved = None
    g = torch.Generator().manual_seed(0)
    cs = torch.randint(1, 250, (B,), generator=g).to(dev)
    ns = torch.randint(1, 250, (B,), generator=g).to(dev)
    es = (torch.rand(B, generator=g) < 0.02).float().to(dev)
    for _ in range(warmup):
        m.step_batch(cs, ns, es)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        m.step_batch(cs, ns, es)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    return dt, float(m.last_loss_tensor)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}  B={B} dim={DIM} layers={LAYERS}")
    print("the byte ids here are UNIFORM (random bytes) -- the grouped "
          "path's best case;\nwith a real corpus's byte histogram the "
          "commonest byte owns ~16% of the rows.")
    for path in ("full", "compiled", "nvrtc", "grouped"):
        try:
            dt, loss = run(path, dev)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {path:>9}: unavailable ({type(exc).__name__}: "
                  f"{str(exc)[:60]})")
            continue
        print(f"  {path:>9}: {dt*1e3:7.2f} ms/step   loss {loss:.5f}")
    print("the losses must agree to the last digit: the paths are the same "
          "arithmetic.")


if __name__ == "__main__":
    main()
