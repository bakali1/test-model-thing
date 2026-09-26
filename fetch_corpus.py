#!/usr/bin/env python3
"""fetch_corpus.py — in-session corpus staging for TMT GPU runs.

Runs INSIDE the Colab/Kaggle session (or locally). Downloads the pinned corpus
layers straight from HuggingFace Hub, strips/normalizes, and interleaves documents
round-robin across requested layers into small plain-text files under ./corpus/.
Documents larger than the 6 MB source-chunk limit are split deterministically
(preferring a newline) and the split is recorded in the manifest. Every output
part therefore contains a mixture rather than one sequential domain block. It
prints an ETA
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
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

USER_AGENT = "tmt-fetch-corpus/1.0"
MAX_PART_BYTES = 8_000_000


def split_text_bytes(text, limit):
    """Split one source document deterministically at <= ``limit`` bytes.

    Prefer the last newline in the second half of the window; otherwise cut
    at a UTF-8 code-point boundary.  The original document is never lost: the
    returned pieces concatenate back to exactly the input bytes.
    """
    raw = text.encode("utf-8", "ignore")
    if limit <= 0:
        raise ValueError(f"split limit must be positive, got {limit}")
    while len(raw) > limit:
        cut = limit
        newline = raw.rfind(b"\n", limit // 2, limit)
        if newline >= 0:
            cut = newline + 1
        while cut > 0 and cut < len(raw) and (raw[cut] & 0xC0) == 0x80:
            cut -= 1
        if cut <= 0:
            raise ValueError("could not find a valid UTF-8 split boundary")
        yield raw[:cut].decode("utf-8", "ignore")
        raw = raw[cut:]
    if raw:
        yield raw.decode("utf-8", "ignore")

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
def _next_link(header):
    """Parse an RFC 5988 Link header for rel="next"."""
    if not header:
        return None
    for part in header.split(","):
        seg = part.split(";")
        if len(seg) < 2:
            continue
        url = seg[0].strip().strip("<>")
        if any('rel="next"' in s for s in seg[1:]):
            return url
    return None


def repo_files(repo, revision="main", max_pages=500):
    """Every file in a dataset repo, FOLLOWING PAGINATION.

    The tree endpoint returns one page and a Link: rel="next" header; a 2 TB
    repo like FineWeb-Edu has thousands of files, so stopping after page one
    would silently shrink the candidate pool (and change the corpus hash).
    """
    from urllib.parse import quote
    url = (f"https://huggingface.co/api/datasets/{repo}/tree/"
           f"{quote(str(revision), safe='')}"
           f"?recursive=true&expand=false")
    out, pages = [], 0
    while url and pages < max_pages:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req) as r:
            tree = json.load(r)
            link = r.headers.get("Link", "")
        out.extend(f for f in tree if f.get("type") == "file")
        url = _next_link(link)
        pages += 1
    if url:
        raise RuntimeError(f"HF tree pagination did not terminate after "
                           f"{max_pages} pages for {repo}")
    return out


def repo_revision(repo):
    """Current commit sha of the dataset repo, recorded in the manifest so a
    rerun can prove it saw the same tree."""
    try:
        url = f"https://huggingface.co/api/datasets/{repo}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req) as r:
            return json.load(r).get("sha")
    except Exception:
        return None


def match_files(repo, pattern, revision="main"):
    """files matching glob pattern, sorted; excludes the README collision."""
    import fnmatch
    out = []
    for f in repo_files(repo, revision=revision):
        p = f["path"]
        if fnmatch.fnmatch(p, pattern) and not p.lower().startswith("readme"):
            out.append((p, f.get("size", 0)))
    out.sort()
    return out


def download(repo, filename, install_ok, revision="main"):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        if install_ok:
            _auto_install(["huggingface_hub"])
            from huggingface_hub import hf_hub_download
        else:
            sys.exit("huggingface_hub missing; run: python -m pip install -q huggingface_hub")
    return hf_hub_download(repo_id=repo, filename=filename,
                           repo_type="dataset", revision=revision)


def _read_parquet(path, cols, batch_size=20000):
    import pyarrow.parquet as pq
    with pq.ParquetFile(path) as pf:
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols,
                                     use_threads=False):
            for row in batch.to_pylist():
                yield row


def round_robin_documents(next_document, max_items=None, is_active=None):
    """Yield ``(key, document)`` from keyed iterators/next-callables.

    ``next_document(key)`` returns the next value for a key or raises
    ``StopIteration`` when that key is exhausted.  Each key is advanced once
    per round, so the earliest output shard is genuinely mixed whenever all
    requested layers have data. ``max_items`` bounds smoke fetches without
    reading the rest of a multi-gigabyte source. ``is_active`` may disable a
    source as soon as its per-layer byte budget is reached.
    """
    getter = next_document.__getitem__
    active = set(next_document.keys())
    emitted = 0
    while active:
        for key in sorted(active):
            if is_active is not None and not is_active(key):
                active.remove(key)
                continue
            try:
                value = getter(key)()
            except StopIteration:
                active.remove(key)
                continue
            yield key, value
            emitted += 1
            if max_items and emitted >= max_items:
                return


class LayerSource:
    """Lazily read transformed docs from one pinned layer's shuffled files."""

    def __init__(self, layer, spec, files, install_ok, revision, log=print,
                 max_document_bytes=MAX_PART_BYTES):
        self.layer = layer
        self.spec = spec
        self.revision = revision
        self.transform = TRANSFORMS[spec["transform"]]
        self.files = list(files)
        self.install_ok = install_ok
        self.log = log
        self.file_index = -1
        self.local = None
        self.rows = None
        self.selected_files = []
        self.exhausted = False
        self.max_document_bytes = int(max_document_bytes)
        self._pending = []
        self.split_documents = 0
        self.split_pieces = 0
        self.split_bytes = 0
        self.split_examples = []

    def _open_next_file(self):
        self.file_index += 1
        if self.file_index >= len(self.files):
            self.exhausted = True
            self.rows = None
            return False
        fname, remote_size = self.files[self.file_index]
        self.local = download(self.spec["repo"], fname, self.install_ok,
                              revision=self.revision)
        self.selected_files.append(fname)
        self.rows = _read_parquet(self.local, self.spec["cols"])
        self.log(f"  {self.layer}: {fname} ({remote_size/1e9:.2f} GB) -> "
                 f"{self.local}", flush=True)
        return True

    def start(self):
        return self._open_next_file()

    def next_document(self):
        """Next (text-piece, provenance-name), or None after all files."""
        if self._pending:
            return self._pending.pop(0)
        while self.file_index < len(self.files):
            if self.rows is None and not self._open_next_file():
                return None
            try:
                row = next(self.rows)
            except StopIteration:
                if not self._open_next_file():
                    return None
                continue
            out = self.transform(row)
            if out is None:
                continue
            text, name = out
            pieces = list(split_text_bytes(text, self.max_document_bytes))
            if len(pieces) > 1:
                self.split_documents += 1
                self.split_pieces += len(pieces)
                self.split_bytes += len(text.encode("utf-8", "ignore"))
                if len(self.split_examples) < 8:
                    self.split_examples.append({
                        "document": name, "bytes": len(text.encode("utf-8", "ignore")),
                        "pieces": len(pieces),
                    })
            self._pending = [(piece, f"{name}_part{i:03d}")
                             for i, piece in enumerate(pieces)]
            return self._pending.pop(0)
        return None


# ----------------------------------------------------------------------------
# Staging writer: round-robin document order, small files, bounded source pieces
# ----------------------------------------------------------------------------
class CorpusWriter:
    """Crash-safe publisher: stage in a hidden versioned dir, rename it into
    place, then flip a symlink atomically.

    The trainer reads `<out>/current/corpus_*.txt`. A crash before the flip
    leaves `current` pointing at the previous complete corpus; a crash after
    it leaves the new one. There is no window where the directory is partial.
    """

    def __init__(self, outdir, mb, expected_layers=None):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.version = self._next_version()
        self.dir = os.path.join(outdir, f".staging-{self.version:04d}")
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir, exist_ok=True)
        # The run contract forbids domain-sized tails. Keep the hard ceiling
        # independent of the requested target so a historical --mb 40 cannot
        # silently reintroduce the oversized-part failure.
        self.max_part_bytes = MAX_PART_BYTES
        self.target = min(int(mb * 1e6), self.max_part_bytes)
        self.parts = 0
        self.buf = []
        self.bufbytes = 0
        self.docs = 0
        self.parts_docs = []          # (name, part) audit trail
        self.expected_layers = (set(expected_layers) if expected_layers
                                else set())
        self.part_layers = set()
        self.first_part_layers = set()
        self.part_sizes = []
        self.order = io.StringIO()

    def _next_version(self):
        v = 0
        for name in os.listdir(self.outdir):
            m = re.match(r"^v(\d{4})$", name)
            if m:
                v = max(v, int(m.group(1)) + 1)
        return v

    def add(self, name, text, layer=None):
        b = text.encode("utf-8", "ignore")
        if len(b) > self.max_part_bytes:
            raise ValueError(
                f"source document is {len(b):,} bytes, over the 8 MB part "
                f"limit (layer={layer or '-'}, document={name}); refusing "
                "to publish an oversized part")
        # Only the first part is held until every requested layer has
        # contributed. This is the data-order guarantee. If that guarantee
        # itself cannot fit under 8 MB, fail loudly rather than publish a pure
        # domain block or a giant tail.
        part_mixed = self.parts > 0 or not self.expected_layers or \
            self.expected_layers.issubset(self.part_layers)
        if not part_mixed and self.bufbytes + len(b) > self.max_part_bytes:
            missing = sorted(self.expected_layers - self.part_layers)
            raise ValueError(
                "cannot make the first mixed part fit under 8 MB before "
                f"adding layer={layer or '-'} document={name}; missing layers "
                f"{missing}")
        if self.bufbytes and part_mixed and self.bufbytes + len(b) > self.target:
            if self.parts == 0:
                self.first_part_layers = set(self.part_layers)
            self.flush()
        self.buf.append(b)
        self.bufbytes += len(b)
        self.docs += 1
        if layer:
            self.part_layers.add(layer)
        self.parts_docs.append((name, self.parts))
        label = f"{layer}\t{name}" if layer else name
        self.order.write(f"{self.parts:05d}\t{label}\n")

    def flush(self):
        if not self.buf:
            return
        with open(os.path.join(self.dir, f"corpus_{self.parts:05d}.txt"), "wb") as f:
            f.write(b"".join(self.buf))
        self.part_sizes.append(self.bufbytes)
        self.parts += 1
        self.buf = []
        self.bufbytes = 0
        self.part_layers = set()

    def close(self, manifest):
        if self.parts == 0:
            self.first_part_layers = set(self.part_layers)
        self.flush()
        manifest["part_sizes"] = list(self.part_sizes)
        manifest["first_part_layers"] = sorted(self.first_part_layers)
        with open(os.path.join(self.dir, "ORDER.txt"), "w") as f:
            f.write(self.order.getvalue())
        manifest["published_as"] = f"v{self.version:04d}"
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        # 1) rename staging -> final version dir (target does not exist yet)
        final = os.path.join(self.outdir, f"v{self.version:04d}")
        os.replace(self.dir, final)
        # 2) flip the pointer atomically: symlink via temp + os.replace
        link = os.path.join(self.outdir, "current")
        tmp = link + ".swap"
        if os.path.lexists(tmp):
            os.remove(tmp)
        os.symlink(os.path.basename(final), tmp)
        if os.path.lexists(link) and not os.path.islink(link):
            # legacy flat layout from before versioning: replace the dir
            shutil.rmtree(link) if os.path.isdir(link) else os.remove(link)
        os.replace(tmp, link)
        # 3) keep the last 2 versions, drop older ones
        vers = sorted(int(m.group(1)) for m in
                      (re.match(r"^v(\d{4})$", n)
                       for n in os.listdir(self.outdir)) if m)
        for v in vers[:-2]:
            shutil.rmtree(os.path.join(self.outdir, f"v{v:04d}"),
                          ignore_errors=True)
        return os.path.join(link, "corpus_00000.txt")


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
            revision = repo_revision(spec["repo"]) or "main"
            files = match_files(spec["repo"], spec["file_pat"],
                                revision=revision)
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
        # yield check: a silently-short candidate pool (pagination change,
        # renamed slice) must fail the dry run, not the 8-hour fetch
        want = gb if gb > 0 else 0
        if want and staged_est < want * 0.9:
            sys.exit(f"dry run: estimated staged yield {staged_est:.2f} GB is "
                     f"below 90% of the requested {want:.2f} GB — the selected "
                     f"file pool changed or the slice is too small; re-check "
                     f"the repo listing before spending quota")
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

    writer = CorpusWriter(args.out, args.mb, expected_layers=layers)
    sources = {}
    revisions = {}
    per_layer = {
        l: {"label": SPEC[l]["label"], "staged_bytes": 0, "docs": 0,
            "budget_bytes": budgets[l], "stop": None, "repo": SPEC[l]["repo"]}
        for l in layers
    }
    t0 = time.time()
    total_staged = 0

    # Build one lazy source per layer before consuming documents. Downloading
    # the first parquet from every layer is intentional: the first 6 MB output
    # shard must contain A+B+C, not whichever layer was listed first.
    for l in layers:
        spec = SPEC[l]
        revision = repo_revision(spec["repo"])
        if not revision:
            raise RuntimeError(f"could not pin dataset revision for {spec['repo']}")
        revisions[l] = revision
        files = match_files(spec["repo"], spec["file_pat"],
                            revision=revision)
        import random as _random
        _random.Random(args.seed).shuffle(files)
        if not files:
            raise RuntimeError(f"no files for layer {l} at revision {revision}")
        source = LayerSource(l, spec, files, install_ok, revision=revision,
                             max_document_bytes=writer.target)
        source.start()
        sources[l] = source
        print(f"\n[fetch] layer {l} {spec['label']} - {len(files)} files "
              f"at revision {revision[:12]} ({spec['note']}); staged budget "
              f"{'uncapped' if budgets[l] is None else f'{budgets[l]/1e9:.2f} GB'}",
              flush=True)

    next_document = {l: sources[l].next_document for l in layers}
    def source_is_active(l):
        budget = budgets[l]
        return (per_layer[l]["stop"] is None and
                (budget is None or per_layer[l]["staged_bytes"] < budget))

    for l, (text, name) in round_robin_documents(
            next_document, max_items=args.limit or None,
            is_active=source_is_active):
        if args.limit and writer.docs >= args.limit:
            per_layer[l]["stop"] = f"limit {args.limit} docs"
            continue
        lbudget = budgets[l]
        if lbudget is not None and per_layer[l]["staged_bytes"] >= lbudget:
            per_layer[l]["stop"] = f"budget {lbudget/1e9:.2f} GB"
            continue
        writer.add(f"{name}_{writer.docs:08d}", text + "\n", layer=l)
        nbytes = len(text.encode("utf-8", "ignore"))
        per_layer[l]["staged_bytes"] += nbytes
        per_layer[l]["docs"] += 1
        total_staged += nbytes
        if writer.docs % 1000 == 0:
            print(f"  [fetch] {writer.docs:,} docs / "
                  f"{total_staged/1e6:.1f} MB staged", flush=True)

    for l in layers:
        source = sources[l]
        budget = budgets[l]
        if per_layer[l]["stop"] is None:
            if args.limit and writer.docs >= args.limit:
                per_layer[l]["stop"] = f"global limit {args.limit} docs"
            elif budget is not None and per_layer[l]["staged_bytes"] >= budget:
                per_layer[l]["stop"] = f"budget {budget/1e9:.2f} GB"
            else:
                per_layer[l]["stop"] = "all files"
        per_layer[l]["revision"] = revisions[l]
        per_layer[l]["selected_files"] = source.selected_files
        per_layer[l]["split_documents"] = source.split_documents
        per_layer[l]["split_pieces"] = source.split_pieces
        per_layer[l]["split_bytes"] = source.split_bytes
        per_layer[l]["split_examples"] = source.split_examples
        print(f"  -> layer {l}: {per_layer[l]['staged_bytes']/1e9:.3f} GB text, "
              f"{per_layer[l]['docs']} docs ({per_layer[l]['stop']})",
              flush=True)

    stop_reasons = [f"{l}:{per_layer[l]['stop']}" for l in layers]

    stopped_by = "; ".join(stop_reasons) or None

    manifest = {
        "spec_version": 1,
        "ordering": "files shuffled with Random(seed); rows within a file in "
                    "source order; transformed documents emitted round-robin "
                    "across requested layers",
        "mixed_parts": "first part guaranteed mixed; later parts follow "
                       "the round-robin stream and remain small",
        "mix": {l: budgets[l] for l in layers},
        "fetch_seed": args.seed,
        "part_target_mb": args.mb,
        "part_max_bytes": writer.max_part_bytes,
        "source_document_split_bytes": writer.target,
        "split_documents_total": sum(v["split_documents"] for v in per_layer.values()),
        "split_pieces_total": sum(v["split_pieces"] for v in per_layer.values()),
        "spec_hash": sh,
        "budgets_gb": {l: (None if budgets[l] is None else budgets[l] / 1e9)
                       for l in layers},
        "layer_budgets_bytes": {l: budgets[l] for l in layers},
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
    print(f"[fetch] published at {args.out}/current/ (versioned v*/; the "
          f"trainer glob is {args.out}/current/**/corpus_*)")
    print(f"[fetch] ETA to read it at {speed:,} B/s: {eta_str(total_staged or 1, speed)}")


if __name__ == "__main__":
    main()