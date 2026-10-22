#!/usr/bin/env python3
"""Gate for ruler criterion wiring (CL-2b).

The criterion's ACQUIRE must be board B (the NEW/stream facts), read with
the model's OWN trained head. Board C (never trained on) is reported
separately as the in-state route. The ruler must read regime from the
checkpoint by default (cl2_facts/cl2_gap/cl2_alphabet).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable


def main():
    # (1) Check the code wiring
    with open(os.path.join(REPO, "tools", "cl_ruler.py")) as f:
        src = f.read()
    # Headline must come from board B (head), not board C (head_heldout)
    assert '"verdict": acq_verdict' in src or 'acq_verdict = (verdict_of(res["acquire"]["per_readout"].get("head")' in src
    # Must report heldout separately
    assert "verdict_heldout_board" in src or "in-state route" in src
    # Must prefer checkpoint facts by default
    assert 'cl2_facts' in src
    # Ridge must be described as copyability/refit
    assert 'COPYABILITY' in src or 'refit' in src.lower()

    # (2) Check actual ruler output if available
    out = os.path.join(REPO, "docs", "results", "cl2", "armA_ruler.json")
    if os.path.exists(out):
        with open(out) as f:
            data = json.load(f)
        pr = data["verdicts"]["acquire"]["per_readout"]
        # board B head must be measured
        assert pr["head"]["status"] == "MEASURED"
        v = data["verdicts"]["acquire"]["verdict"]
        # With correct wiring, headline is board B
        # board C must be reported separately
        assert "verdict_heldout_board" in data["verdicts"]["acquire"] or "head_heldout" in pr
        held = data["verdicts"]["acquire"].get("verdict_heldout_board")
        if held is None and "head_heldout" in pr:
            held = {"status": pr["head_heldout"].get("verdict") or pr["head_heldout"].get("status")}
        assert held is not None

    print("check_ruler_criterion: PASS")


if __name__ == "__main__":
    main()
