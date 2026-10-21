#!/bin/bash
# fetch_corpus.sh -- rebuild the training corpus ON THE BOX, byte-identically.
#
# The corpus is not shipped from the laptop: it is Range-fetched from the
# HuggingFace dataset, exactly as the project's own record says the original
# was ("64 MB of TinyStories Range-fetched", RESULTS.md). The local shards are
# a contiguous region of TinyStories-train.txt split at fixed boundaries,
# with the file NAMES possibly not in byte order -- so the region is fetched
# once and the boundaries are matched back to the names by sha256.
#
# The oracle (the local files' sha256) travels as 6 lines of text. Nothing
# else crosses the wire.
set -u
cd /content
mkdir -p data/wikipedia_clean/wikistories data/eval
URL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
TOTAL=206568120        # 165259856 (the four training shards) + 41308264 (the held-out one)
OUT=/content/tiny_region.bin

if [ ! -s "$OUT" ] || [ "$(stat -c%s "$OUT")" != "$TOTAL" ]; then
  echo "=== Range-fetching $TOTAL bytes of TinyStories-train.txt ==="
  curl -sL --retry 5 --retry-delay 5 -r 0-$((TOTAL-1)) "$URL" -o "$OUT"
  GOT=$(stat -c%s "$OUT")
  echo "fetched $GOT bytes (expected $TOTAL)"
  [ "$GOT" = "$TOTAL" ] || { echo "FETCH SIZE MISMATCH"; exit 1; }
fi

echo "=== splitting at the recorded boundaries and matching by sha256 ==="
python3 - <<'PY'
import hashlib, os, sys
# the local files' sizes, in the order they were cut, and the sha256 of each
# local file (the oracle). Both travel as text.
sizes = [41300184, 41317091, 41315451, 41327130, 41308264]
oracle = {}
for line in open("/content/corpus_sha256.txt"):
    h, p = line.split()
    oracle[os.path.basename(p)] = h
region = open("/content/tiny_region.bin", "rb").read()
off = 0
chunks = []
for s in sizes:
    chunks.append((off, off + s))
    off += s
assert off == len(region), (off, len(region))
by_hash = {}
for (a, b) in chunks:
    h = hashlib.sha256(region[a:b]).hexdigest()
    by_hash[h] = (a, b)
    print(f"  chunk {a:>10}-{b:<10} sha256 {h[:16]}…")
names = {
    "wiki_tinystories_001.txt": 0,
    "wiki_tinystories_002.txt": 1,
    "wiki_tinystories_003.txt": 2,
    "wiki_tinystories_004.txt": 3,
    "wiki_tinystories_005.txt": 4,
}
dest = {
    "wiki_tinystories_001.txt": "data/wikipedia_clean/wikistories/",
    "wiki_tinystories_002.txt": "data/wikipedia_clean/wikistories/",
    "wiki_tinystories_003.txt": "data/wikipedia_clean/wikistories/",
    "wiki_tinystories_004.txt": "data/wikipedia_clean/wikistories/",
    "wiki_tinystories_005.txt": "data/eval/",
}
ok = True
for name, i in names.items():
    want = oracle.get(name)
    a, b = chunks[i]
    h = hashlib.sha256(region[a:b]).hexdigest()
    if want and h != want:
        # the names are not in byte order -- find the chunk that matches
        match = by_hash.get(want)
        if match is None:
            print(f"  {name}: NO CHUNK MATCHES ITS ORACLE HASH")
            ok = False
            continue
        a, b = match
        h = want
    p = dest[name] + name
    with open(p, "wb") as fh:
        fh.write(region[a:b])
    print(f"  {name}: bytes {a}-{b} sha256 {h[:16]}…  {'MATCHES' if h == want else 'MISMATCH'}")
    ok = ok and (h == want)
print("ALL FOUR TRAINING SHARDS + THE HELD-OUT ONE ARE BYTE-IDENTICAL"
      if ok else "SOME FILE DID NOT MATCH ITS ORACLE HASH")
sys.exit(0 if ok else 2)
PY

echo "=== the canonical eval file (the incumbent's held-out set) ==="
VURL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
if [ ! -s data/eval/tinystories_valid.txt ]; then
  curl -sL --retry 5 --retry-delay 5 "$VURL" -o data/eval/tinystories_valid.txt
fi
sha256sum data/eval/tinystories_valid.txt
echo "expected: $(grep tinystories_valid /content/corpus_sha256.txt | cut -d' ' -f1)"

ls -la data/wikipedia_clean/wikistories/ data/eval/
echo "DATA READY"
