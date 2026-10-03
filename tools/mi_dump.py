"""mi_dump.py — write Marginalize-It targets for a corpus (the frozen 192 B/token).

    .venv/bin/python tools/mi_dump.py --corpus 'data/wikipedia_clean/**/wiki_*' \
        --out ./mi_dump --k 32 --batch 1024

Format, alignment contract and the renormalization rule live in `distill.py` (one
definition, imported by both sides).

Alignment is the thing that would silently ruin this: the trainer buckets a file
into B streams by round-robin over LINES, so stream j position t is not a
contiguous byte range. The writer therefore calls the SAME
`numpy_port.Runtime.make_streams(data, B)` and, for every stream, asserts

    reconstruct(table, ids) == stream_bytes        (exact, byte for byte)

before saving. A failure aborts the dump. The post-processor is disabled so no
15-byte BOS enters any sequence, and continuations are rebuilt from the
hand-built table, never `tokenizer.decode()` (lossy for 791 ids -- gate 0).

The retained prefix mass is accumulated here over the whole corpus and printed:
it is the fidelity of the target, and it is the number most likely to make a
wrong result look right.
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import distill as D  # noqa: E402
import teacher_table as tt  # noqa: E402
import numpy_port as NP  # noqa: E402


def load_teacher(repo, model, dtype, device):
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(model or repo,
                                             dtype=getattr(torch, dtype)).to(device).eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"[dump] teacher {n:,} params | {device} {dtype} | load {time.time() - t0:.1f}s")
    return m


def tokenize_stream(tk, data, table, control, B):
    """One stream -> ids + tok_len, with exact reconstruction asserted."""
    text = data.decode("utf-8", "ignore")
    ids = tk.encode(text).ids
    if any(i in control for i in ids):
        return None, None                      # a control id in real text: refuse
    if tt.reconstruct(table, ids) != data:
        raise RuntimeError(
            "token->byte reconstruction is NOT exact for this stream; refusing to "
            "write a dump whose targets would be silently misaligned")
    tok_len = np.asarray([len(table[i]) for i in ids], np.uint8)
    return np.asarray(ids, np.int64), tok_len


def dump_file(model, tk, table, control, path, out_dir, B, k, batch_tokens, device):
    data = open(path, "rb").read()
    streams, _ends = NP.Runtime.make_streams(data, B)
    L_step = min(len(s) for s in streams) - 1        # trainer's loop length (BYTES)
    stem = os.path.splitext(os.path.basename(path))[0]

    # Tokenize every stream FIRST. The arrays' token axis is sized by the real
    # token counts, NOT by the byte-position loop length: padding it to bytes
    # cost 4.1x the storage (measured 791 B/token against a 193 B/token spec).
    print(f"[dump] {os.path.basename(path)}: {len(data):,} B -> B={B} streams, "
          f"L_step={L_step} byte positions each")
    ids_list, len_list = [], []
    for j, s in enumerate(streams):
        ids, tok_len = tokenize_stream(tk, s, table, control, B)
        if ids is None or len(ids) < 2:
            ids, tok_len = np.zeros(1, np.int64), np.zeros(1, np.uint8)
        ids_list.append(ids)
        len_list.append(tok_len)
    ntok_j = np.asarray([len(x) for x in ids_list], np.int64)
    Ltok = int(ntok_j.max())
    print(f"[dump] token axis: {Ltok:,} rows (longest stream), "
          f"{ntok_j.sum():,} tokens total")

    ids_all = np.zeros((B, Ltok, k), np.int32)
    probs_all = np.zeros((B, Ltok, k), np.float16)
    len_all = np.zeros((B, Ltok), np.uint8)
    t0 = time.time()
    ntok = 0
    for j in range(B):
        ids = ids_list[j]
        n = min(Ltok, len(ids))
        for start in range(0, n, batch_tokens):
            chunk = torch.tensor([ids[start:start + batch_tokens + 1]],
                                 dtype=torch.long, device=device)
            # (1, N): LFM2's forward needs a batch dim, and a 1-D input
            # collapses hidden_states to 2-D and crashes inside hub_kernels.
            with torch.no_grad():
                probs = torch.softmax(model(chunk).logits[0].float(), dim=-1)
            v, i = probs.topk(k, dim=-1)
            m = min(n - start, v.shape[0] - 1)
            if m <= 0:
                break
            ids_all[j, start:start + m] = i[:m].cpu().numpy().astype(np.int32)
            probs_all[j, start:start + m] = v[:m].cpu().numpy().astype(np.float16)
            ntok += m
        len_all[j, :n] = len_list[j][:n]
        if (j + 1) % 128 == 0:
            el = time.time() - t0
            print(f"[dump]   stream {j + 1}/{B} | {ntok:,} tok | "
                  f"{ntok / max(1e-9, el):.0f} tok/s | "
                  f"eta {(B - j - 1) * el / (j + 1) / 60:.1f} min", flush=True)

    meta = {
        "corpus_file": path,
        "corpus_sha256": D._sha256(path),
        "corpus_bytes": len(data),
        "B": B,
        "L_step": L_step,
        "Ltok": Ltok,
        "stream_tokens": [int(x) for x in ntok_j],
        "stream_bytes": [len(s) for s in streams],
        "k": k,
        "bucketing": "numpy_port.Runtime.make_streams (line round-robin)",
        "dtype": "fp16 probabilities, int32 ids",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    paths = D.save_dump(out_dir, stem, ids_all, probs_all, len_all, meta)
    size = sum(os.path.getsize(paths[x]) for x in ("ids", "probs", "tok_len"))
    bpt = size / max(1, ntok_j.sum())
    print(f"[dump] wrote {stem}: {ntok_j.sum():,} tokens, {size / 1e6:.1f} MB "
          f"({bpt:.1f} B/token; frozen target 193)")
    return {"stem": stem, "file": path, "tokens": int(ntok_j.sum()),
            "bytes": size, "B": B, "Ltok": Ltok, "L_step": L_step,
            "bytes_per_token": round(bpt, 1),
            "seconds": round(time.time() - t0, 1),
            "tokens_per_second": round(int(ntok_j.sum()) / max(1e-9, time.time() - t0), 1),
            **{f"{k2}_sha256": v for k2, v in meta.items() if k2.endswith("_sha256")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/wikipedia_clean/**/wiki_*")
    ap.add_argument("--out", default=os.environ.get("TMT_MI_DUMP", "./mi_dump"))
    ap.add_argument("--repo", default=os.environ.get("TMT_TEACHER",
                                                    "LiquidAI/LFM2.5-230M-Base"))
    ap.add_argument("--model", default=os.environ.get("TMT_TEACHER_PATH"))
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1024, help="TRAINER B (alignment)")
    ap.add_argument("--batch-tokens", type=int, default=1024)
    ap.add_argument("--dtype", default="float32",
                    help="teacher compute dtype. fp32 is the default ON PURPOSE: "
                         "the plan's gate 1b compares it against bf16 before "
                         "anyone trusts a dump.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--files", default=None, help="comma list instead of --corpus")
    args = ap.parse_args()

    files = ([f for f in args.files.split(",") if f] if args.files
             else sorted(glob.glob(args.corpus, recursive=True)))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit(f"no corpus files match {args.corpus!r}")
    print(f"[dump] {len(files)} corpus file(s), k={args.k}, B={args.batch}")

    tok_path = tt.fetch_tokenizer_json(args.repo)
    table, meta = tt.build_table(tok_path)
    control = set(meta["control_ids"]) | set(meta["nonbyte_ids"])
    vocab = D.VocabBytes(table, drop_ids=control)
    print(f"[dump] vocab {vocab.n_kept:,} usable ids (dropped {len(control)} "
          f"control), max token bytes {vocab.maxlen}")

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tok_path)
    tk.post_processor = None                       # no BOS phantom bytes
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_teacher(args.repo, args.model, args.dtype, dev)

    results = []
    for path in files:
        results.append(dump_file(model, tk, table, control, path, args.out,
                                 args.batch, args.k, args.batch_tokens, dev))

    man = {"tool": "mi_dump.py", "teacher": args.repo, "k": args.k,
           "B": args.batch, "dtype": args.dtype,
           "tokenizer_sha256": meta["sha256"],
           "storage_bytes_per_token": round(sum(r["bytes"] for r in results)
                                            / max(1, sum(r["tokens"] for r in results)), 1),
           "files": results,
           "created_utc": datetime.now(timezone.utc).isoformat()}
    mpath = os.path.join(args.out, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(man, fh, indent=2)
    print(f"\n[dump] manifest {mpath} — {man['storage_bytes_per_token']} B/token "
          f"(frozen target: 193)")


if __name__ == "__main__":
    main()