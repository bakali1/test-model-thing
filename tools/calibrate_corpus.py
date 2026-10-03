"""calibrate_corpus.py — pick the teacher's generation settings by MEASUREMENT.

    .venv/bin/python tools/calibrate_corpus.py --stories 24

The distillation corpus is written by a 230 M **base** model, and a base model
given a short prefix and no pattern collapses into a verbatim loop (measured:
8-fold repetition of one sentence, batch=1, no padding). So the generation
settings are not taste — they decide whether step 2 measures *register* or
measures *degeneracy*.

This sweeps a few settings and scores each against the real corpus with the
same degeneracy metrics the repo already uses for generated text
(`chat_models.degeneracy_metrics`). The target is the real-corpus row, not zero:

    real TinyStories (train and eval, identical):
        degenerate 0.0 %   unique_4gram 0.687   unique_8gram 0.913
        rep_4gram_frac 0.0148   max_run_newlines 1   unique_bytes 36.1

A setting is only acceptable if the generated corpus lands near those numbers.
The winner's config is then pasted into gen_teacher_corpus.py's defaults AND
recorded, because a corpus whose quality was eyeballed instead of measured is a
corpus that will silently confound every arm trained on it.
"""
import argparse
import glob
import json
import os
import random
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import teacher_table as tt  # noqa: E402
import chat_models as CM  # noqa: E402
import gen_teacher_corpus as G  # noqa: E402

BASELINE_FILES = (
    "data/wikipedia_clean/wikistories/wiki_tinystories_003.txt",
    "data/eval/tinystories_valid.txt",
)

SETTINGS = [
    # label,               shots, temp, top_p, rep_pen, no_rep_ngram, max_new
    ("short-260",            2, 0.85, 0.95, 1.10, 4, 260),
    ("plan-700",             2, 0.85, 0.95, 1.10, 4, 700),
    ("shots3-t09",           3, 0.90, 0.95, 1.05, 4, 700),
    ("hot-t10",              2, 1.00, 0.95, 1.00, 4, 700),
    ("cold-t07",             2, 0.70, 0.95, 1.15, 4, 700),
    ("cold-norep8",          2, 0.80, 0.95, 1.05, 8, 700),
]


def baseline(n):
    rows = []
    for f in BASELINE_FILES:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        blob = open(p, "rb").read(3_000_000)
        st = [s.strip() for s in blob.split(b"\n\n") if len(s.strip()) > 200]
        rows += [CM.degeneracy_metrics(s) for s in st[:n]]
    return rows


def summarize(rows):
    if not rows:
        return {}
    return {
        "n": len(rows),
        "degenerate_fraction": round(sum(1 for r in rows if r["degenerate"]) / len(rows), 4),
        "unique_4gram": round(statistics.fmean(r["unique_4gram"] for r in rows), 4),
        "unique_8gram": round(statistics.fmean(r["unique_8gram"] for r in rows), 4),
        "rep_4gram_frac": round(statistics.fmean(r["rep_4gram_frac"] for r in rows), 5),
        "rep_8gram_frac": round(statistics.fmean(r["rep_8gram_frac"] for r in rows), 5),
        "unique_bytes": round(statistics.fmean(r["unique_bytes"] for r in rows), 2),
        "max_run_newlines": max(r["max_run_newlines"] for r in rows),
        "mean_bytes": round(statistics.fmean(r["bytes"] for r in rows), 1),
    }


def run_setting(model, tk, table, control, prompts, files, cfg, batch, device, dtype):
    label, shots_n, temp, top_p, rep, nrng, max_new = cfg
    shots = G.load_shots(files, shots_n, 42) if shots_n else []
    out_rows, total_bytes, total_tok = [], 0, 0
    t0 = time.time()
    for start in range(0, len(prompts), batch):
        chunk = prompts[start:start + batch]
        encs = []
        for _f, p in chunk:
            body = p.decode("utf-8", "ignore")
            if shots:
                body = "\n\n".join(s.decode("utf-8", "ignore") for s in shots) + "\n\n" + body
            ids = tk.encode(body).ids
            if ids and not any(i in control for i in ids):
                encs.append(ids)
        if not encs:
            continue
        width = max(len(e) for e in encs)
        inp = torch.zeros((len(encs), width), dtype=torch.long)
        mask = torch.zeros((len(encs), width), dtype=torch.long)
        for r, e in enumerate(encs):
            inp[r, width - len(e):] = torch.tensor(e)
            mask[r, width - len(e):] = 1
        inp, mask = inp.to(device), mask.to(device)
        torch.manual_seed(1000 + start)
        gkw = dict(temperature=temp, top_p=top_p, repetition_penalty=rep)
        if nrng:
            gkw["no_repeat_ngram_size"] = nrng
        with torch.no_grad():
            out = model.generate(inp, attention_mask=mask, max_new_tokens=max_new,
                                 do_sample=True, pad_token_id=0, **gkw)
        for ids_r in out[:, width:].cpu().tolist():
            real = [i for i in ids_r if i not in control]
            if not real:
                continue
            cont = tt.reconstruct(table, real)
            if not cont.strip():
                continue
            out_rows.append(CM.degeneracy_metrics(cont))
            total_bytes += len(cont)
            total_tok += len(real)
    dt = time.time() - t0
    s = summarize(out_rows)
    s.update({"setting": list(cfg), "seconds": round(dt, 1),
              "bytes": total_bytes, "tokens": total_tok,
              "bytes_per_second": round(total_bytes / max(1e-9, dt), 1),
              "tokens_per_second": round(total_tok / max(1e-9, dt), 1)})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("TMT_TEACHER",
                                                    "LiquidAI/LFM2.5-230M-Base"))
    ap.add_argument("--model", default=os.environ.get("TMT_TEACHER_PATH"))
    ap.add_argument("--stories", type=int, default=24)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "results",
                                                  "corpus_calibration.json"))
    args = ap.parse_args()

    prompts, _audit, files, _shots = G.load_prompts(
        G.DEFAULT_PROMPT_GLOBS, 10_000_000, 42, shots=0)
    prompts = prompts[:args.stories]
    tok_path = tt.fetch_tokenizer_json(args.repo)
    table, meta = tt.build_table(tok_path)
    control = set(meta["control_ids"]) | set(meta["nonbyte_ids"])
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    tk = Tokenizer.from_file(tok_path)
    tk.post_processor = None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model or args.repo, dtype=getattr(torch, args.dtype)).to(dev).eval()

    base = summarize(baseline(400))
    print("BASELINE (real TinyStories): " + json.dumps(base))
    print(f"\n{'setting':14s} {'degen':>6} {'u4':>6} {'u8':>6} {'rep4':>7} "
          f"{'ubytes':>7} {'maxnl':>5} {'B/s':>8}  verdict")
    rows = []
    for cfg in SETTINGS:
        s = run_setting(model, tk, table, control, prompts, files, cfg,
                        args.batch, dev, args.dtype)
        rows.append(s)
        ok = (s.get("degenerate_fraction", 1) <= base.get("degenerate_fraction", 0)
              and s.get("unique_4gram", 0) >= base.get("unique_4gram", 1) * 0.95
              and s.get("rep_4gram_frac", 1) <= base.get("rep_4gram_frac", 0) * 1.5)
        s["meets_baseline"] = ok
        print(f"{cfg[0]:14s} {s['degenerate_fraction']:6.2%} {s['unique_4gram']:6.3f} "
              f"{s['unique_8gram']:6.3f} {s['rep_4gram_frac']:7.4f} "
              f"{s['unique_bytes']:7.1f} {s['max_run_newlines']:5d} "
              f"{s['bytes_per_second']:8.0f}  {'OK' if ok else 'below baseline'}")

    good = [r for r in rows if r["meets_baseline"]]
    winner = max(good, key=lambda r: r["bytes_per_second"]) if good else \
        max(rows, key=lambda r: r["unique_4gram"])
    print(f"\nWINNER: {winner['setting']}  ({winner['bytes_per_second']:.0f} B/s, "
          f"degenerate {winner['degenerate_fraction']:.1%}, u4 {winner['unique_4gram']})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"baseline": base, "settings": rows, "winner": winner["setting"],
                   "device": dev, "dtype": args.dtype,
                   "stories_per_setting": args.stories}, fh, indent=2)
    print(f"[calib] wrote {args.out}")


if __name__ == "__main__":
    main()