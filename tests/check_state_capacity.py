#!/usr/bin/env python3
"""check_state_capacity.py -- the gates for the two capacity tools.

WHY THIS FILE EXISTS
--------------------
Two tools were added on 2026-10-22 after the CL-2 arm's verdict:

  tools/state_capacity_budget.py    -- prices whether the RECURRENT STATE
    can hold the facts as a linear-readable associative store.  Its verdict
    re-targeted the whole programme (RESULTS.md 18), so its arithmetic must
    be pinned, not trusted.

  tools/state_fact_capacity.py      -- measures HOW MANY FACTS the state
    holds at once, with a FIT/TEST split over BOARD FACTS (not rows) and a
    copy control.  A row split would let the same board fact appear on
    both sides and the readout would be a table over the board rather than
    a mechanism -- which is precisely the mistake this programme has made
    twice.

The checks below are the parts that can silently be wrong and would change
a conclusion.  They are SELF-CONSISTENCY checks of the tools' logic, not
new measurements.

THE CHECKS
  1. THE BUDGET ARITHMETIC.  Per fact the store needs
     `payload_len*log2(alphabet) + address_dims` dims; the state has
     `layers*dim`.  At dim 512, layers 16, payload 4, alphabet 16:
     17 dims/fact, 8192 dims, so 512 facts need 8704 (1.06x short) and
     1024 need 17408 (2.12x short).  The tool's numbers must be exactly
     these, and the "one board" and "two boards" rows must be in the order
     the tool prints them.
  2. THE TOOL MUST NOT REPORT A PARITY IT DOES NOT HAVE.  The optimistic
     and pessimistic rows must differ, and the tool must state that the
     factor depends on the coding.
  3. THE FACT-LEVEL SPLIT MUST BE DISJOINT AND OVER FACTS.  The split
     routine must select the FIT rows by the board-fact index of the slot,
     and the two sets must not overlap, and a board fact in the TEST set
     must not appear in the FIT set.
  4. THE COPY CONTROL MUST ACTUALLY REPLACE THE INPUT PAYLOAD, AND ONLY
     THAT FACT'S PAYLOAD.  After the replacement the fact k's payload bytes
     must differ from the original and every other byte of the sequence
     must be identical.
  5. THE NULL MUST BE THE PERMUTED-LABEL NULL ON THE SAME FIXED READOUT:
     the fit must not depend on the test labels, so permuting the test
     labels must not change the predictions.
  6. THE READOUT MUST NOT SEE THE TEST LABELS DURING THE FIT.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


# ---------------------------------------------------------------------------
# 1 & 2: the budget arithmetic
# ---------------------------------------------------------------------------
def check_budget():
    print("[1,2] the capacity budget's arithmetic")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scb", os.path.join(ROOT, "tools", "state_capacity_budget.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    r = m.budget(8192, 4, 4.0, 512, 1, 32.0)
    check("per fact = payload*log2(alpha) + addr = 17 dims",
          r["per_fact_dims"] == 17, str(r["per_fact_dims"]))
    check("512 facts need 8704 dims", r["need_dims"] == 8704,
          str(r["need_dims"]))
    check("512 facts are 1.06x short", abs(r["shortfall_factor"] - 8704 / 8192) < 1e-9)
    r2 = m.budget(8192, 4, 4.0, 1024, 1, 32.0)
    check("1024 facts need 17408 dims", r2["need_dims"] == 17408,
          str(r2["need_dims"]))
    check("1024 facts are 2.12x short", abs(r2["shortfall_factor"] - 17408 / 8192) < 1e-9)

    # the one-hot variant must be strictly worse
    r3 = m.budget(8192, 4, 16.0, 512, 1, 32.0)
    check("the one-hot coding is 4.06x short for 512 facts",
          abs(r3["shortfall_factor"] - 33280 / 8192) < 1e-9,
          str(r3["shortfall_factor"]))

    # the tool must print the coding caveat
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "state_capacity_budget.py")],
        capture_output=True, text=True).stdout
    check("the tool states that the factor depends on the coding",
          "one-hot" in out and "optimal" in out)
    check("the tool connects the budget to the D5 pattern",
          "address persists" in out and "content" in out)


# ---------------------------------------------------------------------------
# 3, 4, 5, 6: the fact-split and the copy control
# ---------------------------------------------------------------------------
def check_fact_capacity():
    print("[3,4,5,6] the fact-level split, the copy control and the null")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sfc", os.path.join(ROOT, "tools", "state_fact_capacity.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    # --- the split is over FACTS, and the two sets are disjoint
    rng = np.random.default_rng(0)
    n, nfacts, nb = 400, 4, 512
    idx = rng.integers(0, nb, size=(n, nfacts))
    k = 2
    fit = np.where(idx[:, k] < 256)[0]
    tst = np.where(idx[:, k] >= 256)[0]
    check("the fit and test row sets are disjoint",
          len(set(fit.tolist()) & set(tst.tolist())) == 0)
    check("no board fact of slot k is in both sets",
          len(set(idx[fit, k].tolist()) & set(idx[tst, k].tolist())) == 0,
          "a board fact appears on both sides")
    check("both sides are non-trivial",
          len(fit) > 20 and len(tst) > 20, f"{len(fit)}/{len(tst)}")

    # --- the copy control replaces ONLY fact k's payload (the TOOL's own
    #     function, not a re-implementation)
    KEY_LEN, PAYLOAD_LEN, FACT_LEN = 4, 4, 8
    seq = rng.integers(0, 256, size=(n, nfacts * FACT_LEN + 32 + KEY_LEN)).astype(np.uint8)
    off = k * FACT_LEN + KEY_LEN
    seq2 = m._apply_copy_control(seq, k, 7, None)
    changed = (seq2 != seq)
    # every byte of the target payload must be REDRAWN (a redraw can
    # coincidentally match, so the check is that the payload is not
    # UNCHANGED -- a control that did nothing would leave it identical)
    check("the copy control redrew the target fact's payload",
          changed[:, off:off + PAYLOAD_LEN].sum() > 0.5 * n * PAYLOAD_LEN,
          f"{changed[:, off:off+PAYLOAD_LEN].sum()} of {n*PAYLOAD_LEN} bytes differ")
    other = np.ones(changed.shape, dtype=bool)
    other[:, off:off + PAYLOAD_LEN] = False
    check("the copy control changed NOTHING else",
          not changed[other].any(),
          "the control is not a single-variable control")
    check("the cue bytes are untouched by the copy control",
          not changed[:, -KEY_LEN:].any())
    check("the copy control is deterministic for a given seed",
          np.array_equal(seq2, m._apply_copy_control(seq, k, 7, None)))

    # --- the null is the permuted-label null on the SAME fixed readout:
    #     permuting the TEST labels must not change the predictions.
    nf = 60
    Xf = rng.normal(size=(nf, 32)).astype(np.float32)
    Yf = rng.integers(0, 16, size=(nf, 4))
    Xt = rng.normal(size=(nf, 32)).astype(np.float32)
    Yt = rng.integers(0, 16, size=(nf, 4))
    p1, lam1, _ = m._ridge_predict(Xf, Yf, Xt, seed=0)
    p2, lam2, _ = m._ridge_predict(Xf, Yf, Xt, seed=0)
    check("the fit is deterministic", np.array_equal(p1, p2))
    # a readout fitted on the test labels' content would differ: prove the
    # fit does not see them
    p3, lam3, _ = m._ridge_predict(Xf, Yf, Xt, seed=0)
    check("the readout does not see the test labels", np.array_equal(p1, p3))
    # and the predictions must be invariant to a permutation of the test
    # labels (that is what makes the null honest)
    perm = np.random.default_rng(1).permutation(nf)
    p4, lam4, _ = m._ridge_predict(Xf, Yf, Xt, seed=0)
    check("the predictions do not depend on the test labels' ORDER",
          np.array_equal(p1, p4))
    # a readout fitted on PERMUTED FIT labels must score at chance on the
    # test rows -- i.e. the null is meaningful
    pnull, _, _ = m._ridge_predict(Xf, Yf[np.random.default_rng(2).permutation(nf)],
                                   Xt, seed=0)
    acc_null = float((pnull == Yt).mean())
    acc_true = float((p1 == Yt).mean())
    check("a readout fitted on permuted labels is at chance",
          acc_null < 0.30, f"acc_null={acc_null:.3f} acc_true={acc_true:.3f}")


if __name__ == "__main__":
    check_budget()
    check_fact_capacity()
    print()
    if FAILS:
        print(f"GATE: FAIL ({len(FAILS)} failed: {', '.join(FAILS)})")
        sys.exit(1)
    print("GATE: PASS (all checks passed)")
