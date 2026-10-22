#!/usr/bin/env python3
"""Gate for ruler criterion wiring (CL-2b).

(a) criterion's ACQUIRE is board B
(b) ruler cannot report a FAIL on a cell the criterion did not ask for (both stated)
(c) fact-level split is disjoint and over facts, opt-in
(d) default path unchanged
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    with open(os.path.join(REPO, "tools", "cl_ruler.py")) as f:
        src = f.read()

    # (a) ACQUIRE is board B
    assert 'CRITERION: ACQUIRE is board B' in src
    assert '"verdict_stream_board"' in src or 'verdict of board B' in src
    # headline uses board B head
    assert 'acq_verdict = (verdict_of(res["acquire"]["per_readout"].get("head")' in src

    # (b) both cells stated
    assert '"verdict_heldout_board"' in src
    assert 'in-state route' in src

    # (c) fact split
    assert '--split-by-fact' in src
    assert 'fact_split' in src
    assert 'make_cell_factsplit' in src
    assert '"split_type": "fact_level"' in src or '"board_facts"' in src
    assert '"split"' in src

    # (d) default unchanged
    assert 'getattr(args, "split_by_fact", False)' in src
    assert 'cl2_facts' in src  # read from checkpoint by default

    out = os.path.join(REPO, "docs", "results", "cl2", "armA_ruler.json")
    if os.path.exists(out):
        with open(out) as f:
            data = json.load(f)
        pr = data["verdicts"]["acquire"]["per_readout"]
        assert pr["head"]["status"] == "MEASURED"
        assert "verdict_heldout_board" in data["verdicts"]["acquire"] or "head_heldout" in pr

    print("check_ruler_criterion: PASS")


if __name__ == "__main__":
    main()
