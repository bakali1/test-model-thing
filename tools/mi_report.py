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

    report = {"dump": args.dump, "steps": args.steps, "by_depth": {}}
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        p = os.path.join(args.dump, f"{stem}.meta.json")
        if not os.path.exists(p):
            print(f"[skip] no dump for {stem}")
            continue
        data = open(path, "rb").read()
        streams, _ends = NP.Runtime.make_streams(data, man["B"])
        r = D.MiReader(args.dump, stem, vocab, streams=streams, block=args.block)
        L = min(args.steps, r.L_step - 1)
        ret_by_d = {d: [] for d in range(4)}
        hit_by_d = {d: [0, 0] for d in range(4)}
        ent_by_d = {d: [] for d in range(4)}
        for t in range(0, L, args.block):
            tgt, ret, n = r.block(t)
            p_pos = np.minimum(t + 1 + np.arange(n), r.L_step)
            gold = np.zeros((r.B, n), np.int64)
            dep = np.zeros((r.B, n), np.int64)
            for j in range(r.B):
                s = streams[j]
                for i in range(n):
                    pp = int(p_pos[i])
                    if pp >= len(s):
                        continue
                    gold[j, i] = int(s[pp])
                    dep[j, i] = pp - int(r.tok_start[j, int(r.byte2tok[j, pp])])
            live = ret > r.floor
            am = tgt.argmax(axis=2)
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -(tgt * np.log(tgt + 1e-12)).sum(axis=2)
            for d in range(4):
                m = dep == d          # EVERY position at this depth, live or not
                if not m.any():
                    continue
                ret_by_d[d].append(ret[m])
                hit_by_d[d][0] += int((am[m] == gold[m]).sum())
                hit_by_d[d][1] += int(m.sum())
                ent_by_d[d].append(ent[m])
        print(f"\n[{stem}] B={r.B} L_step={r.L_step} walked {L} steps | "
              f"{r.retained_n:,} positions | backoff {r.backoff_n:,} "
              f"({r.backoff_n / max(1, r.retained_n):.2%})")
        print(f"  {'depth':>5} {'n':>8} {'ret_mean':>9} {'ret_p05':>8} {'ret_p01':>8} "
              f"{'KD@>0.05':>9} {'KD@>0.01':>9} {'argmax=gold':>11} {'entropy':>8}")
        rows = {}
        for d in range(4):
            if not ret_by_d[d]:
                continue
            v = np.concatenate(ret_by_d[d])
            h, tot = hit_by_d[d]
            e = np.concatenate(ent_by_d[d])
            # coverage = fraction of positions where MI still carries signal
            c05 = float((v > 0.05).mean())
            c01 = float((v > 0.01).mean())
            rows[d] = {"n": int(v.size), "retained_mean": round(float(v.mean()), 4),
                       "retained_p05": round(float(np.percentile(v, 5)), 4),
                       "retained_p01": round(float(np.percentile(v, 1)), 4),
                       "coverage_floor_0.05": round(c05, 4),
                       "coverage_floor_0.01": round(c01, 4),
                       "argmax_equals_gold": round(h / max(1, tot), 4),
                       "target_entropy_nats": round(float(e[e > 0].mean()) if (e > 0).any() else 0.0, 4)}
            print(f"  {d:>5} {v.size:>8,} {v.mean():>9.4f} "
                  f"{np.percentile(v, 5):>8.4f} {np.percentile(v, 1):>8.4f} "
                  f"{c05:>9.2%} {c01:>9.2%} {h / max(1, tot):>11.4f} "
                  f"{rows[d]['target_entropy_nats']:>8.4f}")
        report["by_depth"][stem] = rows
        report.setdefault("reader_stats", {})[stem] = r.stats()

    out = args.out or os.path.join(ROOT, "docs", "results", "mi_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[report] wrote {out}")
    print("cross-check against RESULTS.md 2026-03/10 retained mass "
          "(0.803 / 0.371 / 0.320 at depths 0/1/2)")


if __name__ == "__main__":
    main()