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

The target: the paper's Marginalize-It, at EVERY depth
----------------------------------------------------
Two bugs have lived here, and the second was a misdiagnosis of the first.

Bug 1 (found 2026-10-03, real): the depth loop conditioned on byte d and *then*
aggregated byte d, so every surviving token sat on the gold byte and the target
was a point mass -- entropy exactly 0.000000 nats, argmax == gold on 100% of
positions. Every c with `bytes(c)[:d+1] == gold[:d+1]` has `bytes(c)[d] == gold[d]`
by definition, so the "soft label" was a hard label and `alpha*KL + (1-alpha)*CE`
became position-dependent CE reweighting.

The "fix" applied to it -- emit depth 0 only, decline every depth >= 1 -- was
wrong, and it was justified by a claim the paper contradicts on the same page.
"It takes more than one forward pass to represent byte uncertainty inside a
token" is false: the paper's own worked example gives P(isu)=0.8 / P(isk)=0.2 at
depth 2 from a single pass. The correct fix is to delete ONE TERM from the
match, not to delete the depths.

Bug 2 (fixed 2026-10-04): the paper (section 2.1, Figure 2) conditions on the
bytes ALREADY EMITTED and aggregates over the NEXT one -- "we subset the
vocabulary by the ground-truth byte `i` and aggregate the probabilities over
B2". So at depth d, with `gold` the byte string of the token being predicted:

    S_d   = { c : bytes(c)[:d] == gold[:d]  and  len(c) > d }
    p_tgt = sum_{c in S_d, bytes(c)[d] == x} p(c)  /  sum_{c in S_d} p(c)

Depth 0 of this is exactly the paper's Case I (no gold byte is available yet,
because the first byte of a token IS the byte being predicted). Depth >= 1 is
soft whenever two candidates share the emitted prefix and diverge at byte d --
which is the normal case, not an exception.

Measured on real teacher text, k=32 (RESULTS.md 2026-10-04): coverage goes from
23.1% (depth 0 only) to 89.0%, and the richest band is depth 1 -- 95.5% of its
targets put mass on more than one byte, mean entropy 1.108 nats.

WHAT IS STILL APPROXIMATE, AND IT IS NOT ABOUT FORWARD PASSES
--------------------------------------------------------------
Two real losses of fidelity, both inherent to top-k truncation, both reported:

* Truncation. Under k=32 of a 64,402-token vocabulary a 4-byte gold prefix
  matches ~1 candidate, so depth >= 4 degenerates toward one-hot. `floor` and
  `backoff` exist for this and are counted, never averaged away.
* MI's structural bias. `len(c) > d` drops every candidate that ENDS at byte
  d-1, so the target is the teacher conditioned on "this token does not end
  here". This is the paper's stated approximation, and it is large: top-1 agrees
  with the gold byte at only 44.9% of depth-1 positions. It biases the student
  toward long tokens, which is what the word-length-drift ruler measures, and it
  is precisely what End-Of-Token's `<eot>` channel exists to recover.

A declined position returns an all-zero row and contributes no KD gradient.
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
    bytes against the gold token's bytes 0..d-1, and those gold bytes are not
    in the dump -- only the byte at depth d is. Handing the reader the exact
    streams the dump was built from is what makes that test possible, and it
    doubles as the alignment gate: the byte count implied by `tok_len` must
    equal the stream length, or the targets are on the wrong bytes.
    """

    def __init__(self, out_dir, stem, vocab, streams=None, block=32,
                 backoff="skip", floor=0.05):
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
        self.floor = float(floor)
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
        # safely; that column is 0, so q=0, `ok` is false and the position is
        # declined rather than given a target from the wrong row.
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
        # Statistics the trainer reports. Coverage is "produced / positions";
        # `entropy` and `top1_gold_fraction` are the two that decide whether
        # the KD term is doing anything a reweighted CE could not: entropy says
        # the label is soft, and top1_gold_fraction says how much of that
        # softness contradicts the corpus.
        self.positions = 0
        self.produced_n = 0
        self.declined = 0
        self.empty = 0
        self.entropy_sum = 0.0
        self.depth_sum = 0.0
        self.retained_sum = 0.0
        self.retained_n = 0
        self.retained_below_floor = 0
        self.backoff_n = 0
        self.top1_gold = 0
        self.top1_n = 0

    # -- the target -------------------------------------------------------
    def _build_block(self, t0, blk):
        """Targets for steps [t0, t0+blk).

        Returns (targets (B, blk, 256) float32, produced (B, blk) float32).
        `produced` is 1.0 where a usable target exists and 0.0 where the
        position was declined: the very first byte of a stream (no teacher row
        predicts it), or a position whose surviving top-k mass fell under
        `floor` while `backoff == "skip"`.

        This is the paper's Marginalize-It at EVERY depth. `acc` carries the
        gold-prefix test over the bytes ALREADY EMITTED (0..depth-1); the byte
        being predicted, `depth`, is never conditioned on. That single omission
        is what the 2026-10-03 implementation got wrong, and it is what makes
        every target collapse onto the gold byte.
        """
        # steps are BYTE positions; the arrays are indexed by TOKEN
        p = np.minimum(t0 + 1 + np.arange(blk, dtype=np.int64), self.L_step)
        q = self.byte2tok[:, p]                                   # (B, blk)
        # OFF BY ONE: dump row r holds the logits at token position r, which
        # predict token r+1, so the distribution that predicted token q is q-1.
        qrow = q - 1
        ok = qrow >= 0
        qrow = np.maximum(qrow, 0)
        # depth of the byte being predicted inside its own token, and where
        # that token starts. A byte that BEGINS a token has d == 0, which is
        # the paper's Case I: no gold byte exists yet to condition on.
        starts = np.take_along_axis(self.tok_start, q, axis=1)     # (B, blk)
        d = (p[None, :] - starts).astype(np.int64)                 # (B, blk)

        qmin, qmax = int(qrow.min()), int(qrow.max())
        rel = (qrow - qmin)[:, :, None]
        cand = np.take_along_axis(self.ids[:, qmin:qmax + 1, :], rel,
                                 axis=1).astype(np.int64)         # (B, blk, k)
        prob = np.take_along_axis(self.probs[:, qmin:qmax + 1, :], rel,
                                 axis=1).astype(F)                # (B, blk, k)
        rows = self.vocab.take(cand)
        live = (rows >= 0) & self.vocab.has_byte[np.where(rows >= 0, rows, 0)]
        rows = np.where(live, rows, 0)
        vl = self.vocab.len[rows].astype(np.int64)                # (B, blk, k)
        vb = self.vocab.bytes[rows]                               # (B, blk, k, Lmax)

        # Gold bytes of the token, from its start. The prefix test needs bytes
        # 0..d-1, NOT the byte at d.
        ar = np.arange(self.dmax, dtype=np.int64)
        gidx = np.minimum(starts[:, :, None] + ar[None, None, :],
                          self.sdata.shape[1] - 1)                 # (B, blk, dmax)
        gold_win = np.take_along_axis(self.sdata[:, None, :], gidx,
                                      axis=2).astype(np.int64)[:, :, :, None]

        out = np.zeros((self.B, blk, 256), F)
        acc = np.ones(cand.shape, bool)   # AND_{j < depth} bytes(c)[j]==gold[j]
        maxd = int(min(self.dmax, int(d.max()) + 1))
        for depth in range(maxd):
            # `acc` is the gold prefix EMITTED SO FAR; `vl > depth` keeps only
            # candidates that extend past the byte being predicted (this is
            # MI's structural approximation: candidates that END at depth-1
            # are dropped, which is what End-Of-Token's <eot> recovers).
            keep = acc & (vl > depth) & live & ok[:, :, None]
            sel = d == depth
            if sel.any():
                wsel = np.where(keep, prob, F(0.0))
                mask3 = np.broadcast_to(sel[:, :, None], wsel.shape)
                nb = vb[:, :, :, depth].astype(np.int64)
                flat = (np.arange(self.B)[:, None, None] * (blk * 256)
                        + np.arange(blk)[None, :, None] * 256 + nb)
                np.add.at(out.reshape(-1), flat.reshape(-1)[mask3.reshape(-1)],
                          wsel.reshape(-1)[mask3.reshape(-1)])
            # advance the emitted-prefix test for the NEXT depth. This runs
            # AFTER the aggregation above, never before it.
            acc = acc & (vb[:, :, :, depth] == gold_win[:, :, depth, :])

        tot = out.sum(axis=2)
        self.retained_sum += float(tot.sum())
        self.retained_n += int(tot.size)
        produced = (tot > 0) & ok
        if self.backoff == "skip":
            dead = tot <= self.floor
            self.retained_below_floor += int(dead.sum())
            self.backoff_n += int((dead & ok).sum())
            produced = produced & ~dead
        # normalize only where a target was produced
        out[produced] /= tot[produced][:, None]
        out[~produced] = 0.0

        self.positions += int(produced.size)   # (B, blk), not p.size which is (blk,)
        self.produced_n += int(produced.sum())
        self.declined += int((~produced).sum())
        self.empty += int((ok & (tot <= 0)).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(out * np.log(out + 1e-12)).sum(axis=2)
        if produced.any():
            self.entropy_sum += float(ent[produced].sum())
            self.depth_sum += float(d[produced].sum())
            # p can reach L_step, which a hand-built dump may set one past the
            # stream; clip rather than index off the end.
            gp = np.minimum(p, self.sdata.shape[1] - 1)
            gold_p = self.sdata[np.arange(self.B)[:, None], gp[None, :]]
            self.top1_gold += int((out.argmax(axis=2)[produced]
                                   == gold_p[produced]).sum())
            self.top1_n += int(produced.sum())
        return out, produced.astype(F)

    def block(self, t0):
        """Whole cached block starting at t0: (targets (B,blk,256), produced (B,blk), blk)."""
        self.targets(t0)
        return self._cache, self._cache_ret, self._cache_blk

    def targets(self, t0):
        """(targets (B,256), produced (B,)) for step t0. Block-served."""
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
        n = max(1, self.produced_n)
        return {
            "positions": self.positions,
            "produced": self.produced_n,
            "coverage": self.produced_n / p,
            "declined": self.declined,
            "declined_fraction": self.declined / p,
            "no_surviving_candidate": self.empty,
            "retained_mean": (self.retained_sum / max(1, self.retained_n)),
            "retained_below_floor": self.retained_below_floor,
            "backoff_skipped": self.backoff_n,
            "floor": self.floor,
            "target_entropy_nats": (self.entropy_sum / n
                                    if self.produced_n else None),
            "target_mean_depth": (self.depth_sum / n
                                  if self.produced_n else None),
            # How often the teacher's argmax byte IS the corpus byte. The
            # complement is real label conflict: at depth >= 1 it is MI's
            # not-ending bias, and it is what the word-length-drift ruler is
            # for.
            "top1_gold_fraction": (self.top1_gold / self.top1_n
                                   if self.top1_n else None),
            "B": self.B, "L_step": self.L_step, "Ltok": self.Ltok, "k": self.k,
        }