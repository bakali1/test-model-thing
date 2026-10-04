"""mi_report.py — GATE 2: what is actually in the dump? (PLAN §6 gauge 2)

    .venv/bin/python tools/mi_report.py --dump /tmp/mini_dump --corpus '/tmp/mini_corpus/corpus_*' \
        --steps 4000 --B 8

Walks the real reader over the real dump and reports, per byte depth:

  * retained prefix mass (mean / p05 / p01) and the fraction below the 0.05 floor
  * the fraction of positions whose target argmax equals the gold byte
  * the explicit backoff count
  * the mean entropy of the target vs the student-free expectation

This is the statistic that decides whether the KD target is worth training on,
and it is computed from the SAME code path the trainer will use — not from a
separate analysis script. The independent measurement in RESULTS.md (0.803 /
0.371 / 0.320 at depths 0/1/2, via tools/teacher_eval.py) is the cross-check: if
these two disagree, one of them is wrong and the disagreement is the finding.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import distill as D  # noqa: E402
import numpy_port as NP  # noqa: E402
import teacher_table as tt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--corpus", required=True, help="the SAME glob the dump used")
    ap.add_argument("--repo", default="LiquidAI/LFM2.5-230M-Base")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--B", type=int, default=None, help="default: the dump's B")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.dump, "manifest.json")))
    files = sorted(f for f in glob.glob(args.corpus, recursive=True) if os.path.isfile(f))
    if not files:
        raise SystemExit(f"no corpus files match {args.corpus}")
    tok_path = tt.fetch_tokenizer_json(args.repo)
    table, meta = tt.build_table(tok_path)
    control = set(meta["control_ids"]) | set(meta["nonbyte_ids"])
    vocab = D.VocabBytes(table, drop_ids=control)

    B_STREAMS_MIN = 1024
    report = {"dump": args.dump, "steps": args.steps, "by_depth": {},
              "reader_stats": {}}
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        p = os.path.join(args.dump, f"{stem}.meta.json")
        if not os.path.exists(p):
            print(f"[skip] no dump for {stem}")
            continue
        data = open(path, "rb").read()
        streams, _ends = NP.Runtime.make_streams(data, man["B"])
        r = D.MiReader(args.dump, stem, vocab, streams=streams, block=args.block)
        if int(np.asarray(r.tok_len).sum()) < B_STREAMS_MIN or r.Ltok < 2:
            raise SystemExit(
                f"[FATAL] {stem}: the dump is degenerate (Ltok={r.Ltok}). "
                f"An all-padding dump has the CORRECT bytes-per-token and is "
                f"invisible to the storage check; re-dump it.")
        L = min(args.steps, r.L_step - 1)
        n_prod = 0
        n_decl = 0
        hit = 0
        ents = []
        firsts = []
        per_depth = {}
        for t in range(0, L, args.block):
            tgt, produced, n = r.block(t)
            pos = np.minimum(t + 1 + np.arange(n), r.L_step)
            gold = np.zeros((r.B, n), np.int64)
            dep = np.zeros((r.B, n), np.int64)
            for j in range(r.B):
                s = streams[j]
                for i in range(n):
                    pp = int(pos[i])
                    if pp >= len(s):
                        continue
                    gold[j, i] = int(s[pp])
                    dep[j, i] = pp - int(r.tok_start[j, int(r.byte2tok[j, pp])])
            am = tgt.argmax(axis=2)
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -(tgt * np.log(tgt + 1e-12)).sum(axis=2)
            pr = produced > 0
            n_prod += int(pr.sum())
            n_decl += int((~pr).sum())
            hit += int((am[pr] == gold[pr]).sum())
            if pr.any():
                ents.append(ent[pr])
                firsts.append(tgt.max(axis=2)[pr])
            for d in range(8):
                m = dep == d
                if not m.any():
                    continue
                cur = per_depth.setdefault(d, [0, 0])
                cur[0] += int((m & pr).sum())
                cur[1] += int(m.sum())
        tot = max(1, n_prod + n_decl)
        ent = np.concatenate(ents) if ents else np.zeros(1)
        firsts = np.concatenate(firsts) if firsts else np.zeros(1)
        print(f"\n[{stem}] B={r.B} L_step={r.L_step} walked {L} steps | "
              f"{tot:,} positions | {n_prod:,} produced ({n_prod / tot:.2%}), "
              f"{n_decl:,} declined")
        print(f"  target entropy (nats) : mean {ent.mean():.4f}  "
              f"p05 {np.percentile(ent, 5):.4f}  max {ent.max():.4f}")
        print(f"  top-1 mass             : mean {firsts.mean():.4f}")
        print(f"  top-1 == gold byte     : {hit / max(1, n_prod):.4%}")
        print(f"  {'depth':>5} {'positions':>10} {'produced':>9} {'coverage':>9}")
        for d in sorted(per_depth):
            prod, n = per_depth[d]
            print(f"  {d:>5} {n:>10,} {prod:>9,} {prod / max(1, n):>9.2%}")
        report["by_depth"][stem] = {
            "positions": tot, "produced": n_prod,
            "coverage": round(n_prod / tot, 4),
            "declined": n_decl,
            "target_entropy_nats": round(float(ent.mean()), 4),
            "top1_mass_mean": round(float(firsts.mean()), 4),
            "top1_equals_gold": round(hit / max(1, n_prod), 4),
            "per_depth": {str(d): {"produced": v[0], "positions": v[1],
                                   "coverage": round(v[0] / max(1, v[1]), 4)}
                          for d, v in sorted(per_depth.items())},
        }
        report["reader_stats"][stem] = r.stats()

    out = args.out or os.path.join(ROOT, "docs", "results", "mi_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[report] wrote {out}")
    print("depth 0 is the paper's Case I (the unconditioned first-byte")
    print("marginal). depths >= 1 are the prefix-conditioned case: condition on")
    print("the bytes already emitted, aggregate over the NEXT one. Both are")
    print("produced. An earlier version of this tool printed 'depth 0 is the")
    print("ONLY depth that yields a target', which was the depth-0-only bug, not")
    print("a property of the paper.")
    print("Quote coverage WITH its floor: coverage is (definition x floor), and")
    print(f"the reader's floor was {r.stats().get('floor')}.")


if __name__ == "__main__":
    main()
