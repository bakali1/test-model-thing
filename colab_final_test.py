#!/usr/bin/env python3
"""colab_final_test.py — FINAL validation on a real Colab GPU (T4).

Runs once, after the full local gate (check_parity / check_fetch / check_ring_k
/ check_gradients all PASS on CPU). Measures the actual GPU speed of the port,
isolates CUDA-only math drift (CUDA torch vs CPU torch on the same bytes and
the live tmt.npz), exercises the tail on GPU (tmt.npz has 57 hot dims), and
proves a tagged save/load round trip on device.

Upload next to the repo modules (gpu_port.py, train_gpu.py, tail_credit.py,
numpy_port.py, checkpoint_hub.py) and tmt.npz in /content, then:
    python3 colab_final_test.py 2>&1
Prints [RESULT] lines; a final RESULT OVERALL = PASS/FAIL.
"""
import os
import sys
import time

CKPT = "tmt.npz"
STEPS = 20

os.environ.setdefault("TMT_RING_TOL", "1e-4")

import numpy as np
import torch


def banner(s):
    print(f"\n===== {s} =====", flush=True)


# ---------------- A: environment ----------------
banner("A: environment")
print("python", sys.version.split()[0], "| torch", torch.__version__)
print("cuda_available", torch.cuda.is_available(),
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
      "| vram_total_GB",
      round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
      if torch.cuda.is_available() else 0.0)
if not torch.cuda.is_available():
    print("[RESULT] OVERALL = FAIL (no CUDA on this VM)")
    sys.exit(1)

# ---------------- B: bench ----------------
import gpu_port
from gpu_port import GpuModel
import train_gpu as tg

banner("B: bench (fresh init, fp32, tail off)")
print("ring_k(init) =", GpuModel(dim=512, layers=16).ring_k)
m = GpuModel(dim=512, layers=16)
cs = torch.randint(1, 250, (4096,), device=m.device)
ns = torch.randint(1, 250, (4096,), device=m.device)
ends = torch.zeros(4096, dtype=torch.float32, device=m.device)
rows = []
for B in (1024, 2048, 4096):
    m.new_states(B)
    for _ in range(4):
        m.step_batch(cs[:B], ns[:B], ends[:B])
    t0 = time.time(); N = 40
    for _ in range(N):
        m.step_batch(cs[:B], ns[:B], ends[:B])
    dt = (time.time() - t0) / N
    line = f"[RESULT] bench B={B}: {1/dt:.1f} steps/s = {B/dt:,.0f} B/s ({dt*1000:.1f} ms/step)"
    print(line, flush=True)
    rows.append(line)

# ---------------- C: CUDA vs CPU torch on the live checkpoint ----------------
banner("C: live checkpoint continuation — CUDA vs CPU torch (same bytes)")
for dev in ("cpu", "cuda"):
    mm = GpuModel(dim=512, layers=16, device=dev)
    mm.new_states(1024)
    assert mm.load(CKPT), f"load failed on {dev}"
    print(f"  {dev}: t={mm.t} bytes={mm.bytes_seen} ring_k={mm.ring_k} "
          f"tail_dims={sum(mm.tail.nhot(i) for i in range(16))}", flush=True)
    if dev == "cuda":
        cuda_m = mm
    else:
        cpu_m = mm

rng = np.random.default_rng(5)
cs_n = rng.integers(1, 250, (STEPS, 1024)).tolist()
ns_n = rng.integers(1, 250, (STEPS, 1024)).tolist()
losses = {}
for dev, mm in (("cpu", cpu_m), ("cuda", cuda_m)):
    ls = []
    for t in range(STEPS):
        c = torch.tensor(cs_n[t], device=dev)
        n = torch.tensor(ns_n[t], device=dev)
        e = torch.zeros(1024, dtype=torch.float32, device=dev)
        ls.append(mm.step_batch(c, n, e))
    losses[dev] = ls
maxrel = max(abs(a - b) / max(1.0, abs(a))
             for a, b in zip(losses["cpu"], losses["cuda"]))
print(f"[RESULT] CUDA-vs-CPU maxrel loss over {STEPS} steps: {maxrel:.3e}")
for t in range(STEPS):
    print(f"  step {cpu_m.t - STEPS + t + 1:6d}: cpu {losses['cpu'][t]:.6f} "
          f"cuda {losses['cuda'][t]:.6f}", flush=True)

# ---------------- D: tagged save + reload on device + continue ----------------
banner("D: tagged save / reload / continue on CUDA")
cuda_m.save("tmt_gpu_test.npz")
m2 = GpuModel(dim=512, layers=16, device="cuda")
m2.new_states(1024)
m2.load("tmt_gpu_test.npz")
print(f"  reloaded t={m2.t} bytes={m2.bytes_seen} (match {cuda_m.t}/{cuda_m.bytes_seen})")
c = torch.tensor(cs_n[0], device="cuda")
n = torch.tensor(ns_n[0], device="cuda")
e = torch.zeros(1024, dtype=torch.float32, device="cuda")
l_before = cuda_m.step_batch(c, n, e)          # continue the saved model
l_after = m2.step_batch(c, n, e)               # continue the reloaded model
print(f"  continue 1 step: saved-model loss {l_before:.6f} "
      f"reloaded-model loss {l_after:.6f} (rel {abs(l_before-l_after)/max(1.,abs(l_before)):.2e})")

# ---------------- verdict ----------------
ok = all(1e-3 > abs(a - b) / max(1.0, abs(a))
         for a, b in zip(losses["cpu"], losses["cuda"])) \
     and abs(l_before - l_after) / max(1.0, abs(l_before)) < 1e-3 \
     and m2.t == cuda_m.t
print(f"\n[RESULT] OVERALL = {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)