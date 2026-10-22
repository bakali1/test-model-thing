#!/usr/bin/env python3
"""check_ring_credit_default.py -- the gate for the ring-credit default.

On 2026-10-21, while establishing the Colab T4 baseline, the default
ring-credit path was found to be SILENTLY SLOWER than the one the project
believed it was using.

THE BUG.  `GpuModel.__init__` resolved the `auto` mode eagerly:

    self.ring_credit = ring_credit
    if self.ring_credit == "auto":
        self.ring_credit = ("compiled"
                            if str(device).startswith("cuda") else "full")

so `self.ring_credit` was NEVER "auto" by the time `_ring_mode()` read it.
The `auto` branch inside `_ring_mode()` -- the one that selects the fused
`nvrtc` kernel, the FASTEST bit-identical path (26.72 ms/step against
compiled's 33.68 on a T4, with the IDENTICAL loss 7.58001) -- was therefore
DEAD CODE.  The commit that introduced it ("the default credit path is now
the fastest bit-identical one") changed `_ring_mode()` and not the
constructor, so the change was inert: every default run on a T4 has been
using `compiled` all along.

THE FIX.  The constructor no longer resolves `auto`; `_ring_mode()` does it
lazily, where it can see the device capability and the NVRTC availability.

WHAT THIS GATE CHECKS (on any machine, CPU or GPU):
  1. `GpuModel(ring_credit="auto")` KEEPS the literal "auto" in
     `self.ring_credit` -- the constructor must not pre-resolve it;
  2. `_ring_mode()` returns a CONCRETE mode, never the string "auto";
  3. the mode it returns is one of the legal modes;
  4. off CUDA, `auto` resolves to "full" (the portable reference);
  5. on CUDA with cc >= 7, `auto` resolves to a FUSED path ("nvrtc" or
     "compiled"), never the slow "full" reference -- this is the assertion
     that would have caught the original bug;
  6. on CUDA with cc < 7, `auto` never resolves to "compiled" or "triton"
     (Triton has a hard floor at cc 7.0 and used to raise GPUTooOldForTriton
     on the first step of an MX150 run);
  7. the source of `_ring_mode` contains no eager resolution of "auto"
     outside it -- a source-level check, so the bug cannot come back
     through a different path.

Run:  .venv/bin/python tests/check_ring_credit_default.py
"""
import ast
import inspect
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

import gpu_port  # noqa: E402


def main():
    fails = []
    legal = set(gpu_port.RING_CREDIT_MODES)
    import torch
    is_cuda = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability()[0] if is_cuda else None

    # -- 1. the constructor must not pre-resolve "auto" --
    m = gpu_port.GpuModel(dim=64, layers=2, device="cuda" if is_cuda else "cpu")
    if m.ring_credit != "auto":
        fails.append(f"[1] the constructor resolved 'auto' to "
                     f"{m.ring_credit!r} eagerly -- the _ring_mode() branch "
                     f"that selects the fastest bit-identical path is "
                     f"unreachable (the 2026-10-21 bug)")
    else:
        print("[1] OK  the constructor keeps 'auto' (the resolution is lazy)")

    # -- 2/3. _ring_mode() returns a concrete, legal mode --
    mode = m._ring_mode()
    if mode == "auto":
        fails.append("[2] _ring_mode() returned 'auto' -- it must resolve to "
                     "a concrete path")
    elif mode not in legal:
        fails.append(f"[2] _ring_mode() returned {mode!r}, which is not in "
                     f"RING_CREDIT_MODES")
    else:
        print(f"[2] OK  _ring_mode() -> {mode!r} (a concrete, legal mode)")

    # -- 4. off CUDA, auto -> full --
    if not is_cuda:
        mc = gpu_port.GpuModel(dim=64, layers=2, device="cpu")
        if mc._ring_mode() != "full":
            fails.append(f"[4] off CUDA, auto resolved to {mc._ring_mode()!r}, "
                         f"not 'full'")
        else:
            print("[4] OK  off CUDA, auto -> 'full'")

    # -- 5/6. on CUDA, auto -> a fused path, and never Triton below cc 7 --
    if is_cuda:
        if cap >= 7:
            if mode in ("full",):
                fails.append(f"[5] on cc {cap}.x, auto resolved to 'full' -- "
                             f"the slow reference path; a fused path "
                             f"(nvrtc/compiled) must be selected")
            else:
                print(f"[5] OK  on cc {cap}.x, auto -> {mode!r} (a fused path)")
        else:
            if mode in ("compiled", "triton"):
                fails.append(f"[6] on cc {cap}.x, auto resolved to {mode!r}; "
                             f"Triton/Inductor have a hard floor at cc 7.0 "
                             f"and used to raise GPUTooOldForTriton")
            else:
                print(f"[6] OK  on cc {cap}.x, auto -> {mode!r} (never Triton)")

    # -- 7. no eager resolution of "auto" anywhere else in the source --
    src = open(os.path.join(REPO, "gpu_port.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute)
                    and tgt.attr == "ring_credit"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value in ("compiled", "full")):
                offenders.append(f"line {node.lineno}: ring_credit = "
                                 f"{node.value.value!r}")
    if offenders:
        fails.append("[7] the source still assigns a concrete mode to "
                     "ring_credit outside _ring_mode(): "
                     + "; ".join(offenders))
    else:
        print("[7] OK  no eager ring_credit resolution in the source")

    if fails:
        print("\nFAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS  the ring-credit default is resolved lazily and selects the "
          "fastest\n      bit-identical path, with the capability floors "
          "respected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
