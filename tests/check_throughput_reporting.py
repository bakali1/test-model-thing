#!/usr/bin/env python3
"""check_throughput_reporting.py -- the gate for the trainer's own numbers.

Two bugs in `train_gpu.py` were found on 2026-10-21 while running the gate
suite on a second GPU (an MX150, cc 6.1).  Both are in the REPORTING of
throughput, not in the arithmetic, which is why no earlier run caught them:
the losses and the checkpoints were always right, only the B/s line was
wrong.

  BUG 1 -- the shadowed clock.  The train loop's wall-clock start was
      `t0 = time.time()`, and the window loop below it was
      `for c, n_, e_, Tw, t0 in self._windows(...)`.  The window's BYTE
      OFFSET therefore shadowed the clock for the whole loop, and
      `dt = time.time() - t0` measured the time since the epoch rather than
      the time since the run started.  The B/s line printed ~0 for every
      run.  Fixed by renaming the clock to `wall_t0` and the window offset
      to `win_off`.

  BUG 2 -- the bench measured the wrong T and counted the wrong bytes.
      `bench()` built a (B,) call, which `gpu_port.step_batch` unsqueezes
      to (B,1) and reads as T=1 -- so a `--seqlen 4` bench silently
      reported the T=1 throughput under a T=4 heading.  And the byte count
      was `B/dt` rather than `B*T/dt`, which is only right at T=1.

This gate checks the two properties that the fixes restore, WITHOUT needing
a GPU and without a long run:

  1. no variable named `t0` in `Trainer.train` can shadow a wall clock --
     the gate parses the source and asserts that the only `time.time()`
     subtraction in the train loop uses a name that is not assigned by a
     for-loop header;
  2. `bench()` is T-aware: it builds a window whose T matches the
     trainer's seqlen, and it reports bytes/step as B*T.

Run:  .venv/bin/python tests/check_throughput_reporting.py
"""
import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, "train_gpu.py")


def main():
    src = open(TRAIN, encoding="utf-8").read()
    fails = []

    # -- 1. no loop variable may shadow the wall clock --
    tree = ast.parse(src)
    trainer = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            trainer = node
            break
    if trainer is None:
        print("FAIL: no Trainer class in train_gpu.py")
        return 1

    train_fn = None
    for node in trainer.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            train_fn = node
            break
    if train_fn is None:
        print("FAIL: no Trainer.train in train_gpu.py")
        return 1

    # every name bound by a for-loop header inside train()
    loop_vars = set()
    for node in ast.walk(train_fn):
        if isinstance(node, ast.For):
            tgt = node.target
            for sub in ast.walk(tgt):
                if isinstance(sub, ast.Name):
                    loop_vars.add(sub.id)

    # every name that a time.time() subtraction uses
    clock_names = set()
    for node in ast.walk(train_fn):
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub)
                and isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Attribute)
                and node.left.func.attr == "time"):
            for sub in ast.walk(node.right):
                if isinstance(sub, ast.Name):
                    clock_names.add(sub.id)

    shadowed = clock_names & loop_vars
    if shadowed:
        fails.append(f"[1] a wall-clock subtraction uses {sorted(shadowed)}, "
                     f"which is ALSO a loop variable in Trainer.train -- the "
                     f"loop shadows the clock (the BUG-1 shape)")
    else:
        print(f"[1] OK  no clock name is shadowed by a loop variable "
              f"(clocks: {sorted(clock_names) or 'none'})")

    # -- 2a. bench() must be T-aware --
    bench = None
    for node in trainer.body:
        if isinstance(node, ast.FunctionDef) and node.name == "bench":
            bench = node
            break
    if bench is None:
        fails.append("[2] no Trainer.bench in train_gpu.py")
    else:
        bsrc = ast.get_source_segment(src, bench) or ""
        # the window must carry a T dimension
        if not re.search(r"\(max\(Bs\),\s*T\)", bsrc):
            fails.append("[2] bench() does not build a (B,T) window -- it "
                         "measures T=1 whatever --seqlen says (the BUG-2 shape)")
        else:
            print("[2] OK  bench() builds a (B,T) window")
        # the byte count must be B*T
        if not re.search(r"B\s*\*\s*T\s*/\s*dt", bsrc):
            fails.append("[2] bench() does not report bytes as B*T/dt -- the "
                         "byte count is wrong for T>1 (the BUG-2 shape)")
        else:
            print("[2] OK  bench() reports bytes as B*T/dt")

    # -- 3. the train loop's own byte counter must count B*T --
    tsrc = ast.get_source_segment(src, train_fn) or ""
    if not re.search(r"total\s*\+=\s*self\.batch\s*\*\s*Tw", tsrc):
        fails.append("[3] the train loop's byte counter does not add "
                     "batch*Tw -- it would under-count by a factor of T")
    else:
        print("[3] OK  the train loop counts batch*Tw bytes per window")

    if fails:
        print("\nFAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS  the throughput reporting cannot be shadowed, cannot "
          "measure the wrong T,\n      and cannot under-count bytes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
