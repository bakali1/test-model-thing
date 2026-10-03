"""teacher_eval.py — how good IS the teacher? (PLAN-2026-10-03-DISTILL.md §7)

    .venv/bin/python tools/teacher_eval.py --bytes 100000
    .venv/bin/python tools/teacher_eval.py --bytes 20000 --retained-only

Two questions, one pass:

1. **Teacher quality.** What CE / BPB does the teacher itself reach on the
   ruler's held-out text? Without this, "distil a strong open model" is an
   assumption. If the teacher's CE is worse than the ground truth we already
   train against, the KD target is not obviously better than the labels we
   already have, and that must be known BEFORE a GPU hour is spent on it.

2. **Retained prefix mass** (the statistic from the plan's gate 2, computed
   here for free). Marginalize-It at byte position k renormalizes over the
   stored top-k tokens whose bytes START WITH the gold byte prefix. The mass
   those tokens hold, as a fraction of the full teacher distribution, is the
   fidelity of the target — and under top-k it is silently lossy. This is the
   number most likely to produce a plausible-looking wrong experiment, so it
   is reported per byte-depth k=0,1,2 and per file, with percentiles.

Metric convention: BPB follows the paper (App. A.5) — total token cross-entropy
summed over the data, divided by the number of BYTES, times log2(e). No
Marginalize-It conversion is involved: this measures the teacher as a language
model, not as a distillation target.

Both numbers are inputs to `RESULTS.md`, not conclusions.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import teacher_table as tt  # noqa: E402

DEFAULT_FILES = [
    "data/eval/tinystories_valid.txt",
    "data/eval/wiki_tinystories_005.txt",
    "data/eval_shift/tinystories_v2gpt4_valid.txt",
]


def load_teacher(repo, dtype="float32"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=getattr(torch, dtype))
    model.eval()
    return model


@torch.no_grad()
def score_and_retain(model, ids, table, batch_tokens=1024, topk=32, retain_every=1):
    """Return (ce_sum_nats, n_pred_tokens, retention_records).

    retention record: (byte_depth_k, matched_mass, stored_mass, full_mass).
    """
    ce_sum, n_pred = 0.0, 0
    recs = []
    dev = "cpu"
    for start in range(0, len(ids) - 1, batch_tokens):
        chunk = torch.tensor([ids[start:start + batch_tokens + 1]],
                             dtype=torch.long, device=dev)
        logits = model(chunk).logits[0].float()
        probs = torch.softmax(logits[:-1], dim=-1)
        tgt = chunk[0, 1:]
        ce = torch.nn.functional.cross_entropy(logits[:-1], tgt, reduction="sum")
        ce_sum += float(ce)
        n_pred += int(tgt.numel())
        if retain_every <= 1 or (start // batch_tokens) % retain_every == 0:
            tv, ti = probs.topk(topk, dim=-1)          # (T, k)
            full = probs.sum(dim=-1)                    # sanity: == 1
            stored = tv.sum(dim=-1)
            # byte depth of the token being predicted
            pred_tok = tgt.tolist()
            for r, tid in enumerate(pred_tok):
                gold = table[tid]
                ids_r = ti[r].tolist()
                vals_r = tv[r].tolist()
                live = list(range(len(ids_r)))
                for k in range(min(len(gold), 3)):
                    gb = gold[k]
                    live = [j for j in live if table[ids_r[j]][:k + 1]
                            == gold[:k + 1] and len(table[ids_r[j]]) > k]
                    matched = sum(vals_r[j] for j in live)
                    recs.append((k, matched, stored[r].item(), full[r].item()))
                    # deeper positions need the token to continue past k
                    live = [j for j in live if len(table[ids_r[j]]) > k]
    return ce_sum, n_pred, recs


def retention_stats(recs):
    out = {}
    for k in sorted({r[0] for r in recs}):
        m = np.array([r[1] for r in recs if r[0] == k])
        s = np.array([r[2] for r in recs if r[0] == k])
        f = np.array([r[3] for r in recs if r[0] == k])
        denom = np.maximum(s, 1e-9)
        out[f"k{k}"] = {
            "n": int(m.size),
            "matched_mass_mean": float(m.mean()),
            "matched_mass_p05": float(np.percentile(m, 5)),
            "matched_mass_p01": float(np.percentile(m, 1)),
            "matched_over_stored_mean": float((m / denom).mean()),
            "stored_mass_mean": float(s.mean()),
            "full_mass_mean": float(f.mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="LiquidAI/LFM2.5-230M-Base",
                    help="HF repo id used for tokenizer.json (and as the label "
                         "in the results file)")
    ap.add_argument("--model", default=None,
                    help="local directory or HF id to load weights from "
                         "(default: --repo)")
    ap.add_argument("--files", default=",".join(DEFAULT_FILES))
    ap.add_argument("--bytes", type=int, default=100_000)
    ap.add_argument("--batch-tokens", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--retain-every", type=int, default=1,
                    help="score retention on every Nth batch (retention is the "
                         "slow part: it is pure python)")
    ap.add_argument("--retained-only", action="store_true",
                    help="skip CE, retention only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok_path = tt.fetch_tokenizer_json(args.repo)
    table, meta = tt.build_table(tok_path)
    tk = Tokenizer.from_file(tok_path)
    tk.post_processor = None
    print(f"[teacher] {args.repo} | vocab {meta['n_ids']} | "
          f"tokenizer sha256 {meta['sha256'][:12]} | top-k {args.topk}")

    t0 = time.time()
    model = None if args.retained_only else load_teacher(args.model or args.repo)
    if model is not None:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[teacher] loaded: {n_par:,} params in {time.time() - t0:.1f}s")

    results = {"repo": args.repo, "model": args.model or args.repo,
               "topk": args.topk,
               "tokenizer_sha256": meta["sha256"], "files": {}}
    for f in args.files.split(","):
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            print(f"  [skip] {f}")
            continue
        raw = open(path, "rb").read()[:args.bytes]
        text = raw.decode("utf-8", "ignore")
        ids = tt.encode_without_post(tk, text)
        tb = time.time()
        if model is None:
            results["files"][f] = {"bytes": len(raw), "tokens": len(ids)}
            continue
        ce_sum, n_pred, recs = score_and_retain(
            model, ids, table, args.batch_tokens, args.topk, args.retain_every)
        dt = time.time() - tb
        nbytes = len(raw)
        ce_nats_byte = ce_sum / max(1, nbytes)
        entry = {
            "bytes": nbytes, "tokens": len(ids),
            "bytes_per_token": nbytes / max(1, len(ids)),
            "ce_nats_per_token": ce_sum / max(1, n_pred),
            "ce_nats_per_byte": ce_nats_byte,
            "bits_per_byte": ce_nats_byte * np.log2(np.e),
            "seconds": round(dt, 1),
            "tokens_per_second": round(n_pred / max(1e-9, dt), 1),
            "retention": retention_stats(recs) if recs else {},
        }
        results["files"][f] = entry
        print(f"[teacher] {f}")
        print(f"    {nbytes:,} B / {len(ids):,} tok | {entry['bytes_per_token']:.3f} B/tok")
        print(f"    CE {entry['ce_nats_per_token']:.4f} nats/token | "
              f"{entry['ce_nats_per_byte']:.4f} nats/byte | "
              f"BPB {entry['bits_per_byte']:.4f}")
        print(f"    {entry['tokens_per_second']:.1f} tok/s ({dt:.1f}s)")
        for k, v in entry["retention"].items():
            print(f"    retained prefix mass {k}: mean {v['matched_mass_mean']:.4f} "
                  f"p05 {v['matched_mass_p05']:.4f} p01 {v['matched_mass_p01']:.4f} "
                  f"(n={v['n']:,})")

    out = args.out or os.path.join(ROOT, "docs", "results", "teacher_eval.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[teacher] wrote {out} (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()