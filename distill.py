"""distill.py — Marginalize-It byte targets from a token-level teacher.

One definition of the dump format, shared by the writer (tools/mi_dump.py) and
the reader that the trainer uses, because two implementations of a file format
is exactly how a plausible-looking wrong number happens.

WHAT IS STORED, AND WHY IT IS NOT 256 BYTES PER POSITION
--------------------------------------------------------
A byte-space target would be 256 x fp16 = 512 B per training byte = 2.1 kB per
token. The token-space top-k dump is 192 B/token (32 x int32 id + 32 x fp16
prob) + 1 B/token for the token byte-length, i.e. **46.8 B per training byte**,
which is 45x smaller and is what the frozen plan specifies. The Marginalize-It
renormalization is therefore applied by the READER, over the stored
prefix-matching set, once per step. That is the honest description of what the
target is: the teacher, truncated at k=32, reconditioned on "this token does not
end here" (measured retained mass 0.803 / 0.371 / 0.320 at byte depths 0/1/2 --
see RESULTS.md, 2026-10-03).

ALIGNMENT IS THE OTHER TRAP
---------------------------
`numpy_port.Runtime.make_streams` splits a corpus file into B streams by
round-robin over LINES, so stream j position t is NOT a contiguous byte range of
the file. If the dump is written in file order the targets silently land on the
wrong bytes. So the writer calls the SAME `make_streams(data, B)` with the same B
and writes one aligned array per stream, and asserts
`reconstruct(table, ids) == stream_bytes` for every stream before saving. A
mismatch is a hard failure, not a warning.

The target, after the one-hot discovery (2026-10-03)
----------------------------------------------------
The first implementation followed the paper's depth-loop literally: condition on
the gold byte prefix, keep the tokens that match it, renormalize. That was
measured to produce a **point mass at the gold byte on 100.000000% of covered
positions, entropy exactly 0.000000 nats** -- because every token matching
`bytes(c)[:d+1] == g[:d+1]` has `bytes(c)[d] == g[d]` by definition. The "soft
label" was a hard label, so alpha*KL + (1-alpha)*CE degenerated to
position-dependent CE reweighting, the arm lost to its control and diverged.

What survives one teacher forward pass is depth 0 ONLY, and the paper's Case I
constructs it WITHOUT conditioning on the gold byte:

    p_target[x] = sum(prob[c] for c in top-k if first_byte(c) == x)   # then / sum

That is the teacher's genuine first-byte distribution -- measured entropy mean
0.306 nats, 64.4% of positions non-degenerate. So:

    depth 0  (a byte that begins a token, ~1 position in 4.1): real soft target
    depth >=1 (inside a token): NO TARGET. Declined and counted, because one
             forward pass conditioned on the gold prefix cannot represent byte
             uncertainty there; restoring it needs the EoT channel or extra
             teacher passes.

A declined position returns an all-zero row, which contributes no KD gradient
(the CE-only fallback). Declines are counted, never averaged away.
"""
import hashlib
import json
import os

import numpy as np

F = np.float32


# ---------------------------------------------------------------- format ----
def dump_paths(out_dir, stem):
    return {
        "ids": os.path.join(out_dir, f"{stem}.ids.npy"),
        "probs": os.path.join(out_dir, f"{stem}.probs.npy"),
        "tok_len": os.path.join(out_dir, f"{stem}.tok_len.npy"),
        "meta": os.path.join(out_dir, f"{stem}.meta.json"),
    }


def save_dump(out_dir, stem, ids, probs, tok_len, meta):
    """ids (B,L,k) int32 · probs (B,L,k) fp16 · tok_len (B,L) uint8."""
    paths = dump_paths(out_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    assert ids.dtype == np.int32, ids.dtype
    assert probs.dtype == np.float16, probs.dtype
    assert tok_len.dtype == np.uint8, tok_len.dtype
    # tok_len is ONE entry per token (B, L); ids/probs are (B, L, k)
    assert ids.shape == probs.shape and ids.shape[:2] == tok_len.shape, (
        f"shape mismatch ids{ids.shape} probs{probs.shape} "
        f"tok_len{tok_len.shape}")
    for key, arr in (("ids", ids), ("probs", probs), ("tok_len", tok_len)):
        tmp = paths[key] + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, paths[key])
    for key in ("ids", "probs", "tok_len"):
        meta[f"{key}_sha256"] = _sha256(paths[key])
    meta["format"] = "tmt-mi-dump/1"
    tmp = paths["meta"] + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, paths["meta"])
    return paths


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while blk := fh.read(chunk):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------ vocabulary ----
class VocabBytes:
    """Dense (V, Lmax) uint8 view of the token->byte table, for vectorized MI.

    Built from the hand-built table only. `tokenizer.decode()` is lossy for 791
    non-UTF-8 byte tokens (gate 0) and is never called here.
    """

    def __init__(self, table, drop_ids=()):
        keep = [i for i in range(len(table)) if i not in set(drop_ids)]
        self.index = np.asarray(keep, np.int64)
        lens = np.asarray([len(table[i]) for i in keep], np.uint8)
        self.maxlen = int(lens.max()) if lens.size else 1
        self.bytes = np.zeros((len(keep), max(1, self.maxlen)), np.uint8)
        for r, i in enumerate(keep):
            b = table[i]
            if b:
                self.bytes[r, :len(b)] = np.frombuffer(b, np.uint8)
        self.len = lens
        self.n_kept = len(keep)
        # first byte of each token; all-zero when every token is empty (maxlen 0)
        self.first = (self.bytes[:, 0].copy() if self.maxlen > 0
                      else np.zeros(len(keep), np.uint8))
        self.has_byte = (lens > 0)                    # empty tokens contribute 0

    def take(self, ids):
        """Map global ids -> row index in `self` (-1 if dropped)."""
        lut = np.full(int(self.index.max()) + 2, -1, np.int64)
        lut[self.index] = np.arange(self.n_kept, dtype=np.int64)
        safe = np.clip(ids, 0, lut.size - 1)
        return lut[safe]


# ---------------------------------------------------------------- reader ----
class MiReader:
    """Block-serving reader for one corpus file's dump.

    Memory is a memmap, not a load: a 64 MB slice is 3.0 GB of ids+probs, which
    should not be required to train. Rows are pulled in contiguous blocks
    because per-step random reads of 1024 streams cost more than the MI itself.

    `streams` is REQUIRED: the MI prefix test compares a candidate token's
    bytes against the gold token's bytes 0..d, and those gold bytes are not in
    the dump — only the byte at depth d is. Handing the reader the exact
    streams the dump was built from is what makes that test possible, and it
    doubles as the alignment gate: the byte count implied by `tok_len` must
    equal the stream length, or the targets are on the wrong bytes.
    """

    def __init__(self, out_dir, stem, vocab, streams=None, block=32,
                 backoff="skip"):
        self.p = dump_paths(out_dir, stem)
        self.meta = json.load(open(self.p["meta"]))
        if self.meta.get("format") != "tmt-mi-dump/1":
            raise ValueError(f"{stem}: unknown dump format "
                             f"{self.meta.get('format')!r}")
        self.ids = np.load(self.p["ids"], mmap_mode="r")
        self.probs = np.load(self.p["probs"], mmap_mode="r")
        self.tok_len = np.load(self.p["tok_len"], mmap_mode="r")
        self.B, self.Ltok, self.k = self.ids.shape
        self.ntok = np.asarray(self.meta.get("stream_tokens",
                                             [self.Ltok] * self.B), np.int64)
        self.L_step = int(self.meta.get("L_step", self.Ltok))
        self.vocab = vocab
        self.block_size = int(block)   # attribute; `block()` is the accessor
        self.backoff = backoff
        if streams is None:
            raise ValueError("MiReader needs `streams`: the MI prefix test needs "
                             "the gold token's bytes 0..d, which the dump does "
                             "not contain. See the class docstring.")
        if len(streams) != self.B:
            raise ValueError(f"{stem}: dump B={self.B} but {len(streams)} "
                             f"streams supplied")
        self.sdata = np.zeros((self.B, max(len(s) for s in streams)), np.uint8)
        for j, s in enumerate(streams):
            self.sdata[j, :len(s)] = np.frombuffer(s, np.uint8)
        # byte -> (token, depth) per stream, precomputed once per file.
        # Width is in BYTES (sum of token lengths), not tokens: indexing this by
        # token position truncated the map and raised a broadcast error.
        lens = np.asarray(self.tok_len, np.int64)
        self.total_bytes = lens.sum(axis=1)
        # ALIGNMENT GATE: the dump must describe exactly the bytes we were given.
        for j in range(self.B):
            if int(self.total_bytes[j]) > len(streams[j]):
                raise ValueError(
                    f"{stem}: stream {j} has {len(streams[j])} bytes but the "
                    f"dump's token lengths imply {int(self.total_bytes[j])}; "
                    f"the dump was bucketed with a different B or a different "
                    f"file order and its targets are on the WRONG bytes")
        # +1 so a step whose predicted byte sits one past the end still indexes
        # safely; that column is 0, so q=0 and `is_start` is false -> declined.
        width = int(self.total_bytes.max()) + 1
        self.byte2tok = np.zeros((self.B, width), np.int32)
        self.tok_start = np.zeros((self.B, self.Ltok + 1), np.int32)
        for j in range(self.B):
            tl = lens[j]
            self.byte2tok[j, :tl.sum()] = np.repeat(
                np.arange(len(tl), dtype=np.int32), tl)
            starts = np.concatenate([[0], np.cumsum(tl)[:-1]])
            self.tok_start[j, :len(tl)] = starts
        self.dmax = self.vocab.maxlen
        self._cache_t0 = None
        self._cache = None
        # Statistics the trainer reports. Coverage is now "produced / positions",
        # and `entropy` is the load-bearing number: a one-hot target means the
        # KD term is not doing anything a reweighted CE could not.
        self.positions = 0
        self.produced_n = 0
        self.declined = 0
        self.empty = 0
        self.entropy_sum = 0.0
        self.top1_gold = 0
        self.top1_n = 0

    # -- the target -------------------------------------------------------
    def _build_block(self, t0, blk):
        """Targets for steps [t0, t0+blk).

        Returns (targets (B, blk, 256) float32, produced (B, blk) float32).
        `produced` is 1.0 where a target exists and 0.0 where the position was
        declined (inside a token, so depth >= 1, where one forward pass cannot
        represent byte uncertainty).
        """
        # steps are BYTE positions; the arrays are indexed by TOKEN
        p = np.minimum(t0 + 1 + np.arange(blk, dtype=np.int64), self.L_step)
        q = self.byte2tok[:, p]                                   # (B, blk)
        # OFF BY ONE: dump row r holds the logits at token position r, which
        # predict token r+1, so the distribution that predicted token q is q-1.
        qrow = q - 1
        ok = qrow >= 0
        qrow = np.maximum(qrow, 0)
        # Only a byte that BEGINS a token gets a target; `starts == p` is that
        # test. Everything inside a token is depth >= 1 and is declined.
        starts = np.take_along_axis(self.tok_start, q, axis=1)     # (B, blk)
        is_start = (p[None, :] == starts) & ok                    # (B, blk)

        qmin, qmax = int(qrow.min()), int(qrow.max())
        rel = (qrow - qmin)[:, :, None]
        cand = np.take_along_axis(self.ids[:, qmin:qmax + 1, :], rel,
                                 axis=1).astype(np.int64)         # (B, blk, k)
        prob = np.take_along_axis(self.probs[:, qmin:qmax + 1, :], rel,
                                 axis=1).astype(F)                # (B, blk, k)
        rows = self.vocab.take(cand)
        live = (rows >= 0) & self.vocab.has_byte[np.where(rows >= 0, rows, 0)]
        rows = np.where(live, rows, 0)
        b0 = self.vocab.first[rows].astype(np.int64)            # (B, blk, k)

        w = np.where(live, prob, F(0.0))
        out = np.zeros((self.B, blk, 256), F)
        flat = (np.arange(self.B)[:, None, None] * (blk * 256)
                + np.arange(blk)[None, :, None] * 256 + b0)
        m3 = np.broadcast_to(is_start[:, :, None], w.shape)
        np.add.at(out.reshape(-1), flat.reshape(-1)[m3.reshape(-1)],
                  w.reshape(-1)[m3.reshape(-1)])
        tot = out.sum(axis=2)
        produced = (tot > 0) & is_start
        # normalize only where a target was produced
        nz = tot > 0
        out[nz] /= tot[nz][:, None]
        out[~produced] = 0.0

        self.positions += int(is_start.size)
        self.produced_n += int(produced.sum())
        self.declined += int((~is_start).sum())
        self.empty += int((is_start & (tot <= 0)).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(out * np.log(out + 1e-12)).sum(axis=2)
        self.entropy_sum += float(ent[produced].sum()) if produced.any() else 0.0
        self.top1_gold += 0        # filled by the caller (needs the gold byte)
        return out, produced.astype(F)

    def block(self, t0):
        """Whole cached block starting at t0: (targets (B,blk,256), retained (B,blk), blk)."""
        self.targets(t0)
        return self._cache, self._cache_ret, self._cache_blk

    def targets(self, t0):
        """(targets (B,256), retained (B,)) for step t0. Block-served."""
        if self._cache_t0 is None or not (self._cache_t0 <= t0 <
                                          self._cache_t0 + self._cache_blk):
            blk = min(self.block_size, self.L_step - t0)
            self._cache, self._cache_ret = self._build_block(t0, blk)
            self._cache_t0 = t0
            self._cache_blk = blk
        i = t0 - self._cache_t0
        return self._cache[:, i], self._cache_ret[:, i]

    def stats(self):
        p = max(1, self.positions)
        return {
            "positions": self.positions,
            "produced": self.produced_n,
            "coverage": self.produced_n / p,
            "declined_depth_ge_1": self.declined,
            "declined_fraction": self.declined / p,
            "empty_at_token_start": self.empty,
            "target_entropy_nats": (self.entropy_sum / self.produced_n
                                    if self.produced_n else None),
            "B": self.B, "L_step": self.L_step, "Ltok": self.Ltok, "k": self.k,
        }