"""gen_teacher_corpus.py — write the distillation corpus with the teacher (step 1).

    .venv/bin/python tools/gen_teacher_corpus.py --target-bytes 64000000
    .venv/bin/python tools/gen_teacher_corpus.py --limit 20 --dry-run   # smoke

Prefix-conditioned continuation, in the ruler's register. For every output
story: take a real prefix from a TRAINING corpus file, let the teacher continue
it, and keep the continuation. Unconditional generation would throw away the
prefix structure, and the ruler scores *continuations of real text*.

Three things in here are traps that were already paid for once (gate 0):

* **Never `tokenizer.decode()`.** 791 ids decode to U+FFFD. Continuations are
  rebuilt from the hand-built table (`teacher_table.reconstruct`), which gate 0
  proved byte-exact.
* **Prompt sources exclude the ruler.** Default prompt globs are the TRAINING
  files only. `data/eval/**` and `data/eval_shift/**` are refused outright:
  prompting the teacher with held-out text and keeping the continuation is
  contamination with no offset trick to hide it.
* **Left padding must not be reconstructed.** Batched generation pads; `<|pad|>`
  is a 7-byte control string. The attention mask selects real tokens first.

The generator does NOT decontaminate. It publishes raw parts and records their
digests; `tools/decontaminate.py` then filters train ∪ eval ∪ shift and the
manifest records what survived. Keeping the two steps separate is what makes the
contamination number checkable after the fact.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import teacher_table as tt  # noqa: E402
import chat_models as CM  # noqa: E402

# Training-side prompt sources. Eval and shift globs are deliberately absent.
DEFAULT_PROMPT_GLOBS = ("data/wikipedia_clean/**/wiki_*",)
FORBIDDEN_GLOBS = ("data/eval/**", "data/eval_shift/**")

STORY_SEP = b"\n\n"
DEFAULT_MAX_NEW = 700          # tokens per generated story (~2.9 kB at 4.1 B/tok)
DEFAULT_SHOTS = 2              # few-shot stories, from the TRAINING files only

# Why few-shot + no_repeat_ngram are not optional knobs:
# a 230 M BASE model given a ~40-token prefix and no pattern collapses into a
# verbatim 8-fold loop of one sentence (measured, batch=1, no padding). Adding
# two complete in-register stories as a pattern plus `no_repeat_ngram_size=4`
# is what produces prose; `repetition_penalty` alone does NOT (measured 1.05 and
# 1.10, both still looped). Every generated corpus carries degeneracy metrics so
# this cannot regress silently.


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while blk := fh.read(chunk):
            h.update(blk)
    return h.hexdigest()


def story_prefixes(path, min_bytes=300, max_bytes=900):
    """Yield story prefixes from a corpus file, one per story."""
    blob = open(path, "rb").read()
    for seg in blob.split(STORY_SEP):
        seg = seg.strip()
        if len(seg) < min_bytes:
            continue
        # cut at a sentence-ish boundary so the teacher continues mid-narrative
        cut = min(max_bytes, len(seg) // 2)
        cut = seg.rfind(b". ", 0, cut)
        cut = cut + 2 if cut > min_bytes // 2 else cut
        if cut >= min_bytes:
            yield seg[:cut]


def load_shots(files, n, seed, min_bytes=300, max_bytes=1200):
    """Few-shot pattern stories, drawn from the TRAINING files only."""
    if n <= 0:
        return []
    rng = random.Random(seed + 1)
    pool = []
    for f in files:
        with open(f, "rb") as fh:
            blob = fh.read(40_000_000)
        for seg in blob.split(STORY_SEP):
            seg = seg.strip()
            if min_bytes < len(seg) < max_bytes:
                pool.append(seg)
        if len(pool) > 4000:
            break
    if len(pool) < n:
        raise SystemExit(f"not enough few-shot stories ({len(pool)} < {n})")
    return rng.sample(pool, n)


def load_prompts(globs, limit_bytes, seed=42, shots=DEFAULT_SHOTS):
    files = []
    for pattern in globs:
        files.extend(sorted(glob.glob(pattern, recursive=True)))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit(f"no prompt files match {globs}")
    # Refuse any prompt source the ruler can score, by realpath (glob.fnmatch
    # was removed in Python 3.13; realpath membership is the stronger test).
    ruler = set()
    for pat in FORBIDDEN_GLOBS:
        for f in glob.glob(pat, recursive=True):
            if os.path.isfile(f):
                ruler.add(os.path.realpath(f))
    bad = [f for f in files if os.path.realpath(f) in ruler]
    if bad:
        raise SystemExit(
            f"[FATAL] prompt source overlaps the ruler: {bad}\n"
            f"  prompting the teacher with held-out text and keeping the "
            f"continuation is contamination with no offset trick to hide it")
    shot_stories = {s for s in load_shots(files, shots, seed)}
    random.Random(seed).shuffle(files)
    prompts, used = [], []
    total = 0
    for f in files:
        for p in story_prefixes(f):
            if p in shot_stories:      # never continue the shot itself
                continue
            prompts.append((f, p))
            total += len(p)
            used.append({"file": f, "bytes": len(p)})
            if total >= limit_bytes:
                return prompts, used, files, shot_stories
    return prompts, used, files, shot_stories


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("TMT_TEACHER",
                                                    "LiquidAI/LFM2.5-230M-Base"))
    ap.add_argument("--model", default=os.environ.get("TMT_TEACHER_PATH"))
    ap.add_argument("--out", default=os.environ.get("TMT_GEN_OUT", "./corpus_gen"))
    ap.add_argument("--target-bytes", type=int,
                    default=int(os.environ.get("TMT_GEN_TARGET", 64_000_000)))
    ap.add_argument("--part-mb", type=float,
                    default=float(os.environ.get("TMT_GEN_PART_MB", 6)))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("TMT_GEN_BATCH", 32)))
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument("--shots", type=int, default=DEFAULT_SHOTS,
                    help="few-shot pattern stories from the TRAINING files")
    ap.add_argument("--no-repeat-ngram", type=int, default=4,
                    help="0 disables. NOT optional in practice: see the note "
                         "at the top of this file.")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float,
                    default=float(os.environ.get("TMT_GEN_REP_PENALTY", 1.10)),
                    help="a 230 M base model loops verbatim without this; the "
                         "model card itself recommends 1.05 for generation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="cap generated stories")
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "float32", "bfloat16"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--speed", type=float, default=None,
                    help="measured B/s for the ETA (default: first 10 batches)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prompts, prompt_audit, files, shots = load_prompts(
        DEFAULT_PROMPT_GLOBS, max(1, args.target_bytes), args.seed, args.shots)
    n_stories = args.limit or len(prompts)
    print(f"[gen] {len(prompts):,} prompt prefixes from {len(files)} file(s) + "
          f"{len(shots)} few-shot pattern story/ies; generating {n_stories:,} "
          f"stories (~{n_stories * args.max_new_tokens * 4.1 / 1e6:.1f} MB)")
    if args.dry_run:
        for f in files[:5]:
            print(f"    prompt source: {f}")
        print("[gen] dry run: nothing written")
        return

    tok_path = tt.fetch_tokenizer_json(args.repo)
    table, meta = tt.build_table(tok_path)
    control = set(meta["control_ids"]) | set(meta["nonbyte_ids"])
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    tk = Tokenizer.from_file(tok_path)
    tk.post_processor = None                     # no BOS phantom bytes
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model or args.repo,
                                                 dtype=dtype).to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[gen] teacher {args.repo} | {n_par:,} params | {dev} {args.dtype} | "
          f"tokenizer sha256 {meta['sha256'][:12]} | load {time.time() - t0:.1f}s")

    version = f"v{int(time.time())}"
    outdir = os.path.join(args.out, version)
    os.makedirs(outdir, exist_ok=True)
    part_target = int(args.part_mb * 1e6)

    parts, part_docs, part_bytes, layer_mix = [], 0, 0, {}
    gen_bytes, gen_tokens, batches, speed = 0, 0, 0, args.speed
    started = time.time()
    cur_name = "corpus_000.txt"
    fh = open(os.path.join(outdir, cur_name), "wb")
    degen = []          # one degeneracy_metrics() row per generated story

    def close_part():
        """Flush the open part into `parts` — including the final partial one,
        which used to be closed and then silently dropped from the manifest."""
        nonlocal fh, part_docs, part_bytes, cur_name
        fh.close()
        if part_bytes > 0:
            parts.append({"file": cur_name, "bytes": part_bytes,
                          "stories": part_docs})
            cur_name = f"corpus_{len(parts):03d}.txt"
            fh = open(os.path.join(outdir, cur_name), "wb")
            part_docs, part_bytes = 0, 0
    seed_rng = random.Random(args.seed)
    truncated = 0
    try:
        for start in range(0, n_stories, args.batch):
            chunk = prompts[start:start + args.batch]
            encs, kept = [], []
            for f, p in chunk:
                body = p.decode("utf-8", "ignore")
                if shots:
                    body = "\n\n".join(s.decode("utf-8", "ignore") for s in shots) \
                        + "\n\n" + body
                ids = tk.encode(body).ids
                if not ids or any(i in control for i in ids):
                    continue                     # never prompt with a control id
                encs.append(ids)
                kept.append(p)
            if not encs:
                continue
            width = max(len(e) for e in encs)
            pad = 0                             # <|pad|> id from the config
            inp = torch.full((len(encs), width), pad, dtype=torch.long)
            mask = torch.zeros((len(encs), width), dtype=torch.long)
            for r, e in enumerate(encs):        # LEFT pad, so generation continues
                inp[r, width - len(e):] = torch.tensor(e)
                mask[r, width - len(e):] = 1
            inp, mask = inp.to(dev), mask.to(dev)
            torch.manual_seed(seed_rng.randrange(1 << 30))
            with torch.no_grad():
                gkw = dict(temperature=args.temperature, top_p=args.top_p,
                           repetition_penalty=args.repetition_penalty)
                if args.no_repeat_ngram:
                    gkw["no_repeat_ngram_size"] = args.no_repeat_ngram
                out = model.generate(
                    inp, attention_mask=mask,
                    max_new_tokens=args.max_new_tokens, do_sample=True,
                    pad_token_id=pad, **gkw)
            gen = out[:, width:]
            truncated += int((gen == pad).sum())
            g = gen.cpu().tolist()
            for r, ids_r in enumerate(g):
                real = [i for i in ids_r if i not in control]   # drop pad/eos
                if not real:
                    continue
                # `real` is the NEW tokens only (we sliced `gen` off the prompt),
                # so the continuation is the whole reconstruction. An earlier
                # version sliced off len(prompt) here and silently ate the first
                # few hundred bytes of every story.
                cont = tt.reconstruct(table, real)
                if not cont.strip():
                    continue
                fh.write(cont)
                fh.write(STORY_SEP)
                part_docs += 1
                part_bytes += len(cont)
                gen_bytes += len(cont)
                gen_tokens += len(real)
                # Measured, not eyeballed: a 230 M base model can loop or go
                # degenerate, and CE on the trained corpus cannot see it.
                degen.append(CM.degeneracy_metrics(cont))
            gen[:, :] = gen                # keep the tensor alive for the counter
            batches += 1
            if part_bytes >= part_target:
                close_part()
            if batches % 10 == 0:
                dt = time.time() - started
                speed = gen_bytes / dt
                print(f"[gen] batch {batches:4d} | {gen_bytes / 1e6:7.2f} MB | "
                      f"{speed:8.0f} B/s | {gen_tokens / dt:6.0f} tok/s | "
                      f"eta {(args.target_bytes - gen_bytes) / max(1, speed) / 3600:5.2f} h",
                      flush=True)
                if args.target_bytes and gen_bytes >= args.target_bytes:
                    break
    finally:
        close_part()

    for p in parts:
        path = os.path.join(outdir, p["file"])
        p["sha256"] = sha256_file(path)
        p["full"] = True
    summary = {}
    if degen:
        import statistics
        for key in CM.METRIC_FIELDS:
            vals = [d[key] for d in degen]
            if vals:
                summary[key] = round(statistics.fmean(vals), 4)
        summary["degenerate_stories"] = sum(1 for d in degen if d["degenerate"])
        summary["stories_measured"] = len(degen)
        summary["degenerate_fraction"] = round(
            summary["degenerate_stories"] / len(degen), 4)
    manifest = {
        "tool": "gen_teacher_corpus.py",
        "teacher": args.repo,
        "teacher_params": n_par,
        "dtype": args.dtype,
        "device": dev,
        "tokenizer_sha256": meta["sha256"],
        "generation": {
            "mode": "prefix-conditioned continuation",
            "max_new_tokens": args.max_new_tokens,
            "shots": args.shots,
            "no_repeat_ngram": args.no_repeat_ngram,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "batch": args.batch,
        },
        "prompt_sources": files,
        "prompt_globs": list(DEFAULT_PROMPT_GLOBS),
        "prompt_ruler_globs_refused": list(FORBIDDEN_GLOBS),
        "prompt_stories_used": len(prompt_audit),
        "stories": sum(p["stories"] for p in parts),
        "staged_bytes": gen_bytes,
        "generated_tokens": gen_tokens,
        "control_tokens_dropped": truncated,
        "parts": parts,
        "degeneracy": summary,
        "speed_bytes_per_second": round(speed or gen_bytes / max(1e-9, time.time() - started), 1),
        "elapsed_s": round(time.time() - started, 1),
        "decontaminated": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    mpath = os.path.join(outdir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    link = os.path.join(args.out, "current")
    tmp = link + ".tmp"
    if os.path.islink(tmp) or os.path.exists(tmp):
        os.unlink(tmp)
    os.symlink(version, tmp)
    os.replace(tmp, link)
    print(f"\n[gen] done {time.time() - started:.0f}s -> {gen_bytes / 1e6:.2f} MB in "
          f"{len(parts)} part(s) + manifest")
    if summary:
        print(f"[gen] degeneracy over {summary['stories_measured']:,} stories: "
              f"degenerate {summary['degenerate_stories']:,} "
              f"({summary['degenerate_fraction']:.2%}) | "
              f"unique_4gram {summary.get('unique_4gram')} | "
              f"rep_4gram_frac {summary.get('rep_4gram_frac')} | "
              f"max_run_newlines {summary.get('max_run_newlines')}")
    print(f"[gen] published {outdir} ; trainer glob is {link}/**/corpus_*")
    print(f"[gen] NEXT: run tools/decontaminate.py over {link}/*.txt and flip "
          f"corpus_gen/current only after the contamination numbers are recorded")


if __name__ == "__main__":
    main()