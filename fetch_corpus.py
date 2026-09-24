#!/usr/bin/env python3
"""fetch_corpus.py — in-session corpus staging for TMT GPU runs.

Runs INSIDE the Colab/Kaggle session (or locally). Downloads the pinned corpus
layers straight from HuggingFace Hub, strips/normalizes, concatenates documents
in pinned order into ~40 MB plain-text files under ./corpus/, and prints an ETA
table (with --speed, or TMT_SPEED env) so the amount of data for a run can be
chosen from that session's measured bytes/s — bigger sessions get bigger slices.

Layers (pinned spec; changing entries changes the manifest spec_hash):
  A  gutenberg fiction       Despina/project_gutenberg   config fiction_books
                             6,322 complete public-domain novels (MIT) —
                             the long-range memory driver (PLAN-2026-09-23-GPU §4).
  B  fineweb-edu top-score   HuggingFaceFW/fineweb-edu    docs with score >= 3.0
                             (odc-by) — clean general English breadth.
  C  ultrachat dialogue      HuggingFaceH4/ultrachat_200k split train_sft (MIT)
                             — dialogue turn-taking for the chat wall.

No data is ever shipped from the laptop: downloads happen where the training
happens. Only the SPEC (this file) is code. Repeated fetches hit the
HF cache; Kaggle can additionally save ./corpus as a dataset version once so
later sessions mount instead of downloading.

Usage:
  python fetch_corpus.py --layers A,B,C --gb 4.5 --out ./corpus
  python fetch_corpus.py --dry-run                       # list what a fetch would pull
  python fetch_corpus.py --layers C --limit 50 --out /tmp/corpus_smoke   # quick real test
"""
import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

USER_AGENT = "tmt-fetch-corpus/1.0"

# ----------------------------------------------------------------------------
# Pinned corpus spec
# ----------------------------------------------------------------------------
SPEC = {
    "A": {
        "label": "gutenberg_fiction",
        "repo": "Despina/project_gutenberg",
        "file_pat": "fiction_books/train-*.parquet",
        "cols": ["text", "book_id", "title", "author"],
        "transform": "gutenberg",
        "note": "6,322 novels, MIT",
    },
    "B": {
        "label": "fineweb_edu_top",
        "repo": "HuggingFaceFW/fineweb-edu",
        "file_pat": "data/*/train-*.parquet",
        "cols": ["text", "score", "id"],
        "transform": "raw",
        "score_min": 3.0,
        "note": "web text, score>=3, odc-by",
    },
    "C": {
        "label": "ultrachat_dialogue",
        "repo": "HuggingFaceH4/ultrachat_200k",
        "file_pat": "data/train_sft-*.parquet",
        "cols": ["messages", "prompt_id"],
        "transform": "dialogue",
        "note": "208k dialogues, MIT",
    },
}

# Canonical staged mix in GB (PLAN-2026-09-23-GPU §4: A 2.7 / B 1.5 / C 0.3
# at a 4.5 GB total). `--gb` scales these shares; `--gb-per-layer` overrides.
DEFAULT_MIX_GB = {"A": 2.7, "B": 1.5, "C": 0.3}

PG_HEAD = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
PG_TAIL = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)


def spec_hash(layers, gb, seed, gb_per_layer=None):
    h = hashlib.sha256()
    h.update(json.dumps({k: SPEC[k] for k in layers}, sort_keys=True).encode())
    h.update(f"gb={gb} seed={seed} per_layer={gb_per_layer}".encode())
    return h.hexdigest()[:16]


def resolve_budgets(layers, gb, gb_per_layer):
    """Per-layer staged budgets in bytes.

    A single global budget consumed in layer order starves the last layers:
    A alone stages ~2.67 GB of 4.5, then B eats the remainder and C never runs.
    So the default is a fixed mix scaled to the requested total (A 2.7 / B 1.5
    / C 0.3 GB at --gb 4.5). `--gb-per-layer A=2.7,B=1.5,C=0.3` sets budgets
    exactly; unnamed layers are skipped. `--gb 0` means uncapped per layer.
    """
    if gb_per_layer:
        raw = {}
        for item in gb_per_layer.split(","):
            k, _, v = item.partition("=")
            raw[k.strip().upper()] = float(v)
        bad = [k for k in raw if k not in SPEC]
        if bad:
            sys.exit(f"unknown layer(s) in --gb-per-layer: {bad}; known {sorted(SPEC)}")
        return ({l: (None if raw.get(l, 0.0) <= 0 else int(raw[l] * 1e9))
                 for l in layers}, raw)
    if gb <= 0:
        return {l: None for l in layers}, {l: None for l in layers}
    shares = {l: DEFAULT_MIX_GB.get(l, 1.0) for l in layers}
    tot = sum(shares.values()) or 1.0
    mix = {l: gb * shares[l] / tot for l in layers}
    return {l: int(mix[l] * 1e9) for l in layers}, mix


# ----------------------------------------------------------------------------
# Transforms (row dict -> text bytes); each DOCUMENT stays intact
# ----------------------------------------------------------------------------
def _clean(s):
    if not s:
        return ""
    return s.replace("\r\n", "\n").replace("\r", "\n")


def transform_gutenberg(row):
    text = _clean(row.get("text") or "")
    m = PG_HEAD.search(text)
    if m:
        text = text[m.end():]
    m = PG_TAIL.search(text)
    if m:
        text = text[: m.start()]
    text = text.strip("\n")
    if not text:
        return None
    return text, f"gutenberg_{row.get('book_id')}_{(row.get('title') or '')[:60]}"


def transform_raw(row):
    text = _clean(row.get("text") or "")
    if row.get("score") is not None and SPEC["B"].get("score_min") is not None:
        if float(row["score"]) < SPEC["B"]["score_min"]:
            return None
    text = text.strip("\n")
    if not text:
        return None
    return text, f"fineweb_{row.get('id', '')[:16]}"


def transform_dialogue(row):
    parts = []
    for msg in row.get("messages") or []:
        role = (msg.get("role") or "user").title()
        parts.append(f"{role}: {_clean(msg.get('content') or '')}")
    text = "\n".join(parts).strip("\n")
    if not text:
        return None
    return text + "\n---\n", f"ultrachat_{str(row.get('prompt_id', ''))[:16]}"


TRANSFORMS = {"gutenberg": transform_gutenberg, "raw": transform_raw,
              "dialogue": transform_dialogue}


# ----------------------------------------------------------------------------
# Repo file listing + sizes (plain HF JSON API, no extra deps)
# ----------------------------------------------------------------------------
def repo_files(repo):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true&expand=false"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as r:
        tree = json.load(r)
    return [f for f in tree if f.get("type") == "file"]


def match_files(repo, pattern):
    """files matching glob pattern, sorted; excludes the README collision."""
    import fnmatch
    out = []
    for f in repo_files(repo):
        p = f["path"]
        if fnmatch.fnmatch(p, pattern) and not p.lower().startswith("readme"):
            out.append((p, f.get("size", 0)))
    out.sort()
    return out


def download(repo, filename, install_ok):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        if install_ok:
            _auto_install(["huggingface_hub"])
            from huggingface_hub import hf_hub_download
        else:
            sys.exit("huggingface_hub missing; run: python -m pip install -q huggingface_hub")
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")


def _read_parquet(path, cols, batch_size=20000):
    import pyarrow.parquet as pq
    with pq.ParquetFile(path) as pf:
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols,
                                     use_threads=False):
            for row in batch.to_pylist():
                yield row


# ----------------------------------------------------------------------------
# Staging writer: book-ordered ~40 MB files, docs never split
# ----------------------------------------------------------------------------
class CorpusWriter:
    def __init__(self, outdir, mb):
        self.dir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.target = int(mb * 1e6)
        self.parts = 0
        self.buf = []
        self.bufbytes = 0
        self.docs = 0
        self.parts_docs = []          # (name, part) audit trail
        self.order = io.StringIO()

    def add(self, name, text):
        b = text.encode("utf-8", "ignore")
        if self.bufbytes and self.bufbytes + len(b) > self.target:
            self.flush()
        self.buf.append(b)
        self.bufbytes += len(b)
        self.docs += 1
        self.parts_docs.append((name, self.parts))
        self.order.write(f"{self.parts:05d}\t{name}\n")

    def flush(self):
        if not self.buf:
            return
        with open(os.path.join(self.dir, f"corpus_{self.parts:05d}.txt"), "wb") as f:
            f.write(b"".join(self.buf))
        self.parts += 1
        self.buf = []
        self.bufbytes = 0

    def close(self, manifest):
        self.flush()
        with open(os.path.join(self.dir, "ORDER.txt"), "w") as f:
            f.write(self.order.getvalue())
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)


def _auto_install(pkgs):
    print(f"[fetch] installing {', '.join(pkgs)} ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)


def eta_str(total_bytes, speed):
    if speed <= 0:
        return "?"
    s = total_bytes / speed
    return f"{s / 3600:.2f} h" if s >= 3600 else f"{s / 60:.1f} min"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default="A,B,C", help="layers to fetch (comma list)")
    ap.add_argument("--gb", type=float, default=4.5,
                    help="budget of STAGED text in GB (0 = uncapped)")
    ap.add_argument("--gb-per-layer", default=None,
                    help="exact per-layer staged GB, e.g. A=2.7,B=1.5,C=0.3 "
                         "(overrides --gb; unnamed layers skipped)")
    ap.add_argument("--out", default="./corpus")
    ap.add_argument("--mb", type=float, default=40.0, help="target output file size")
    ap.add_argument("--seed", type=int, default=42, help="pinned ordering seed")
    ap.add_argument("--speed", type=float, default=None,
                    help="bytes/s for ETA (default $TMT_SPEED or 150000)")
    ap.add_argument("--limit", type=int, default=0, help="cap docs (smoke tests)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-install", action="store_true")
    args = ap.parse_args()

    layers = [s.strip().upper() for s in args.layers.split(",") if s.strip()]
    bad = [l for l in layers if l not in SPEC]
    if bad:
        sys.exit(f"unknown layer(s) {bad}; known: {sorted(SPEC)}")
    gb = args.gb
    speed = args.speed if args.speed is not None \
        else float(os.environ.get("TMT_SPEED", "150000"))
    sh = spec_hash(layers, gb, args.seed, args.gb_per_layer)
    budgets, mix = resolve_budgets(layers, gb, args.gb_per_layer)

    # ---- dry run: cost table, no data moved ----
    # Staged-yield per downloaded file (approx; B is speculative, A/C exact):
    YIELD_GB = {   # staged text per file: compressed GB -> text GB
        "A": 2.667 / 3,        # 2.667 GB fiction text over 3 files
        "B": 0.25,             # ~10% of docs kept by score>=3, ~2.4 KB/doc text
        "C": 1.397 / 2,        # 1.397 GB dialogs -> first 2 of 3 files (0.73GB dl)
    }
    if args.dry_run:
        print(f"spec_hash {sh}  layers {','.join(layers)}  "
              f"staged budget {'uncapped' if gb <= 0 else f'{gb} GB'}  mix {mix}")
        print(f"{'layer':5s} {'repo':36s} {'files':>5s} {'avail-dl':>9s} "
              f"{'staged-budget':>13s} {'est-files':>9s} {'est-dl':>8s}")
        gtot = est_dl_total = staged_est = 0.0
        for l in layers:
            spec = SPEC[l]
            files = match_files(spec["repo"], spec["file_pat"])
            dgb = sum(s for _, s in files) / 1e9
            gtot += dgb
            lb = budgets[l]
            y = YIELD_GB.get(l, 0.25)
            if lb is None:
                nfiles = len(files)
                staged_est += nfiles * y
            else:
                nfiles = (min(len(files), max(1, math.ceil(lb / (y * 1e9))))
                          if files else 0)
                staged_est += min(lb / 1e9, nfiles * y)
            edl = sum(s for _, s in files[:nfiles]) / 1e9
            est_dl_total += edl
            tag = "uncapped" if lb is None else f"{lb / 1e9:.2f} GB"
            print(f"{l:5s} {spec['repo']:36s} {len(files):5d} {dgb:8.2f}G "
                  f"{tag:>13s} {nfiles:9d} {edl:7.2f}G")
        print(f"available if uncapped = {gtot:.2f} GB compressed "
              f"(one-time; HF cache after)")
        print(f"this budget: est. download = {est_dl_total:.2f} GB -> "
              f"~{staged_est:.2f} GB staged "
              f"(~{eta_str(int(staged_est * 1e9), speed)} to read at "
              f"{speed:,} B/s)")
        print("yields are per-file estimates; the real fetch stops exactly at "
              "each layer budget.")
        return

    # ---- real fetch ----
    install_ok = not args.no_install
    try:
        import pyarrow  # noqa
    except ImportError:
        if install_ok:
            _auto_install(["pyarrow"])
        else:
            sys.exit("pyarrow missing; run: python -m pip install -q pyarrow")

    writer = CorpusWriter(args.out, args.mb)
    per_layer = {}
    t0 = time.time()
    total_staged = 0
    stop_reasons = []

    for l in layers:
        spec = SPEC[l]
        transform = TRANSFORMS[spec["transform"]]
        lbudget = budgets[l]
        files = match_files(spec["repo"], spec["file_pat"])
        staged, docs = 0, 0
        stop = None
        print(f"\n[fetch] layer {l} {spec['label']} - {len(files)} files "
              f"({spec['note']}); staged budget "
              f"{'uncapped' if lbudget is None else f'{lbudget/1e9:.2f} GB'}",
              flush=True)
        for fname, _fsize in files:
            if lbudget is not None and staged >= lbudget:
                stop = f"budget {lbudget/1e9:.2f} GB"
                break
            if args.limit and writer.docs >= args.limit:
                stop = f"limit {args.limit} docs"
                break
            local = download(spec["repo"], fname, install_ok)
            print(f"  {fname} ({_fsize/1e9:.2f} GB) -> {local}", flush=True)
            for row in _read_parquet(local, spec["cols"]):
                if lbudget is not None and staged >= lbudget:
                    stop = f"budget {lbudget/1e9:.2f} GB"
                    break
                if args.limit and writer.docs >= args.limit:
                    stop = f"limit {args.limit} docs"
                    break
                out = transform(row)
                if out is None:
                    continue
                text, name = out
                writer.add(name, text + "\n")
                staged += len(text.encode("utf-8", "ignore"))
                docs += 1
            if stop:
                break
        per_layer[l] = {"label": spec["label"], "staged_bytes": staged,
                        "docs": docs, "budget_bytes": lbudget, "stop": stop}
        total_staged += staged
        if stop:
            stop_reasons.append(f"{l}:{stop}")
        h = eta_str(total_staged or 1, speed)
        print(f"  -> layer {l}: {staged/1e6:.1f} MB text, {docs} docs "
              f"({stop or 'all files'}; cumulative {total_staged/1e6:.0f} MB "
              f"~ {h} at {speed:,} B/s)", flush=True)

    stopped_by = "; ".join(stop_reasons) or None

    manifest = {
        "spec_version": 1,
        "spec_hash": sh,
        "budgets_gb": {l: (None if budgets[l] is None else budgets[l] / 1e9)
                       for l in layers},
        "args": vars(args),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "staged_bytes_total": total_staged,
        "stop_reason": stopped_by,
        "parts": writer.parts,
        "total_docs": writer.docs,
        "layers": per_layer,
    }
    writer.close(manifest)

    print(f"\n[fetch] done in {time.time()-t0:.0f}s -> {writer.parts} part files, "
          f"{total_staged/1e9:.2f} GB staged, {writer.docs} docs "
          f"({stopped_by or 'all layers'})")
    print(f"[fetch] ETA to read it at {speed:,} B/s: {eta_str(total_staged or 1, speed)}")


if __name__ == "__main__":
    main()