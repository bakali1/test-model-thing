"""Exact all-lag embedding credit for ring-truncated (hot) dims.

The ring buffer keeps only the last K bytes per stream; a dim whose decay
sigma satisfies sigma**K > HOT_FRAC would have more than 1% of its embed
credit mass dropped every step. For those "hot" dims -- usually a handful --
we maintain the thing the ring was built to avoid: a dense (B, 256, H)
eligibility EMA, but only for the hot dims. It is exact at every lag.

Rhythm per step mirrors the ring's exactly (decay -> credit -> add):
    ring:     w *= sigma;  credit(w * dlds);  w[:, p] = 1.0; ring_b[:, p] = b
    tail:     begin_step;  credit(E * dlds);  add_byte(c_arr)
so the tail's credit and the ring's credit describe the same bytes with the
same weights, and the tail is a strict superset (lags > K included).

Membership is re-evaluated every REBASE_EVERY steps: dims adopt a private
EMA when sigma**K crosses HOT_FRAC (seeded from the ring, which still holds
correct values because ring_w is kept updated for all dims), and drop back
to the ring when sigma sinks again.

No numba? Falls back to numpy (same math, slower). The decay gradient is
untouched (it already used the dense decaytrace EMA) -- this only fixes the
embedding credit path.

FUSED TAIL PLAN (decay+credit, 3 passes -> 2). The tail's per-step rhythm is
    begin_step -> credit -> add_byte   (3 element-passes over (B,256,H))
and the two cheapest to fold are begin_step (E *= sigma, a read+write of E)
and credit (read E again to reduce). Fusing them into ONE pass that reads E
once -- computing (E*sig) in place and feeding the same decayed value into a
left-fold credit sum -- drops the accumulator from 3 passes to 2:
    [fused decay+credit: 1 read of E, 1 write of E, accumulate out]
    [add_byte: 1 read of E, 1 write of E]

BIT-IDENTITY CONTRACT. The fused kernel multiplies in the UNFUSED order --
`d = E[b,v,h] * sig[h]` (E*sig first), THEN `d * g[b,h]` -- and folds the
credit as a left fold over `b` starting from 0.0, with a float64 accumulator,
exactly mirroring `_tail_decay` followed by `_tail_credit`. The property test
in tests/check_fused_tail.py confirms _tail_fused_decay_credit is
bit-identical to (_tail_decay; _tail_credit) at B=32..1024, H up to 512:
max|delta E| = 0.0, max|delta out| = 0.0 -- so fusion does NOT reassociate
here. A naive `einsum("bvh,h,bh->vh", E, sig, g)` (which torch keeps
BIT-identical to `mul_;einsum` on CPU, 0.0 delta) avoids reassociation but
still reads E twice; the numba kernel reads E once and is the real fast path.

CPU vs GPU. On CPU the fused path runs the numba kernel directly (bit-identical
to this reference). On CUDA a 1-read nvrtc kernel (tmt_tail_fused, cc-floor
free) runs; it uses a float32 accumulator, so the per-call value can diverge
from the float64 reference by the same ~1e-6 the ring credit lives with -- it
is never bit-identical across backends, but it stays within the reference's
own run-to-run spread and the CPU path is bit-identical.

SELECT. ``TailCredit(fused=True)`` fuses begin_step+credit: callers skip
begin_step and pass ``sig`` to ``credit``. ``fused=False`` (default) is the
incumbent, unchanged.
"""
import os
import sys

import numpy as np

try:
    from numba import njit as _njit
except Exception:                        # pragma: no cover
    _njit = None

F = np.float32

HOT_FRAC = 0.01        # hot if sigma**ring_k > 1% of credit mass would drop
REBASE_EVERY = 500     # steps between membership re-evaluations
HOT_WARN = 256         # print a warning when the total hot count passes this


def tail_fused_resolve(val, device='cpu'):
    # Resolve a --fused-tail {auto,on,off} value (or TMT_FUSED_TAIL env) to bool.
    # auto = numba on CPU (or bust), an nvrtc CUDA kernel on CUDA (JIT'd lazily,
    # no cc floor -> runs on 6.1+). on/off unconditional. Unknown -> off.
    if val is None:
        return False
    v = str(val).strip().lower()
    if v in ('off', '0', '', 'false'):
        return False
    if v in ('on', '1', 'true'):
        return True
    if v == 'auto':
        if device == 'cpu':
            return _njit is not None
        # GPU: optimistic -- TorchTail's kernel fallback covers compile failure.
        return True
    return False


if _njit is not None:
    @_njit(cache=True)
    def _tail_decay(E, sig):
        """E: (B,256,H) f32, sig: (H,) f32. E[b,v,h] *= sig[h]"""
        B, V, H = E.shape
        for b in range(B):
            eb = E[b]
            for v in range(V):
                ev = eb[v]
                for h in range(H):
                    ev[h] *= sig[h]

    @_njit(cache=True)
    def _tail_credit(E, g, out):
        """out[v,h] += sum_b E[b,v,h] * g[b,h]   (E: (B,256,H), g: (B,H))"""
        B, V, H = E.shape
        for v in range(V):
            for h in range(H):
                acc = 0.0
                for b in range(B):
                    acc += E[b, v, h] * g[b, h]
                out[v, h] = acc

    @_njit(cache=True)
    def _tail_fused_decay_credit(E, sig, g, out):
        # Fused begin_step + credit, ONE pass over E. Bit-identical to
        # _tail_decay(E, sig) then _tail_credit(E, g, out): decay writes
        # E[b,v,h] = E[b,v,h]*sig[h] (E*sig FIRST, never the reassociating
        # E*(sig*g)), and the same decayed value d is multiplied by g[b,h]
        # and left-folded into out[v,h] over b (acc float64 from 0.0, exactly
        # like _tail_credit). Reads E once -> a 2-pass tail, not 3.
        B, V, H = E.shape
        for v in range(V):
            for h in range(H):
                acc = 0.0
                for b in range(B):
                    d = E[b, v, h] * sig[h]     # decay  (E*sig first)
                    E[b, v, h] = d              # writeback decayed EMA
                    acc += d * g[b, h]          # credit: (E*sig)*g, left fold
                out[v, h] = acc

    @_njit(cache=True)
    def _tail_add(E, ids):
        """E[b, ids[b], h] += 1.0  (ids: (B,) int)"""
        B, H = E.shape[0], E.shape[2]
        for b in range(B):
            x = ids[b]
            for h in range(H):
                E[b, x, h] += 1.0
else:                                    # pragma: no cover
    def _tail_decay(E, sig):
        E *= sig[None, None, :]

    def _tail_credit(E, g, out):
        out += np.einsum("bvh,bh->vh", E, g, dtype=np.float64).astype(F)

    def _tail_fused_decay_credit(E, sig, g, out):
        # Scalar fallback (numba absent). NOT a single pass: decays in place
        # then credits, but is bit-identical to _tail_fused_decay_credit above
        # because both do E*sig first then (E*sig)*g with a float64 fold.
        E *= sig[None, None, :]
        out += np.einsum("bvh,bh->vh", E, g, dtype=np.float64).astype(F)

    def _tail_add(E, ids):
        B, H = E.shape[0], E.shape[2]
        # E is (B, 256, H): axis 0 is the STREAM, axis 1 the byte. Indexing axis
        # 0 with `ids` (the byte id) is a real bug -- it raised IndexError
        # whenever a byte id >= B (e.g. 127 with B=32), so this fallback only
        # ever worked for B=256. The numba path above has always been correct.
        bidx = np.arange(B)[:, None]                    # (B, 1) stream index
        hh = np.arange(H)[None, :]                      # (1, H)
        np.add.at(E, (bidx, ids[:, None], hh), 1.0)     # broadcast (B, H)


class TailCredit:
    """Private per-dim eligibility EMAs, exact at every lag."""

    def __init__(self, B, ring_k, hot_frac=HOT_FRAC, fused=False):
        self.B = B
        self.ring_k = ring_k
        self.hot_frac = hot_frac
        # Fused mode drops begin_step's separate decay pass: credit() decays E
        # in place and reads it once. Callers skip begin_step and pass sig to
        # credit; begin_step is kept as a defensive no-op.
        self.fused = bool(fused)
        self.cols = {}       # layer -> int array of hot dim indices
        self.E = {}          # layer -> (B, 256, H) float32
        self._mask = {}      # layer -> bool (dim,) mask, cached from sync

    # ---------- membership ----------
    def hot_from(self, sigmas):
        """sigmas: (L, D) f32 -> bool (L, D) hot mask."""
        return sigmas ** self.ring_k > self.hot_frac

    def sync(self, sigmas, ring_ws, ring_ids):
        """Recompute the hot set; adopt new dims (seeded from the ring),
        drop cold ones. sigmas: (L,D); ring_ws: list of (B,K,D) per layer;
        ring_ids: (B,K). Call at load, after new_states, and every
        REBASE_EVERY steps."""
        L = sigmas.shape[0]
        hot = self.hot_from(sigmas)
        for l in range(L):
            new = np.nonzero(hot[l])[0]
            old = self.cols.get(l, np.array([], int))
            adopt = np.setdiff1d(new, old)
            drop = np.setdiff1d(old, new)
            keep = np.intersect1d(old, new)
            E = self.E.get(l)
            if len(drop) and E is not None:
                pos = np.array([int(np.searchsorted(old, d)) for d in drop])
                E = np.delete(E, pos, axis=2)
                if E.shape[2] == 0:
                    E = None
            if len(adopt):
                seed = self._seed_from_ring(adopt, ring_ws[l], ring_ids)
                E = seed if E is None else np.concatenate([E, seed], axis=2)
            if len(keep) and E is None:
                E = np.zeros((self.B, 256, len(keep)), F)
            if E is not None:
                self.cols[l] = np.concatenate([keep, adopt]).astype(int) \
                    if len(adopt) else keep.astype(int)
                self.E[l] = np.ascontiguousarray(E, dtype=F)
                self._mask[l] = np.zeros(sigmas.shape[1], bool)
                self._mask[l][self.cols[l]] = True
            else:
                self.cols.pop(l, None)
                self.E.pop(l, None)
                self._mask.pop(l, None)
        n = sum(len(c) for c in self.cols.values())
        if n > HOT_WARN:
            print(f"[tail] warning: {n} hot dims; accumulator is "
                  f"{n * self.B * 256 * 4 / 1e6:.0f} MB", file=sys.stderr)
        return n

    def _seed_from_ring(self, dim_cols, ring_w, ring_ids):
        """Adopt a newly-hot dim: its EMA must equal what the ring already
        knew (lags 0..K-1 at weights sigma^0..sigma^{K-1}); the begin_step
        decay turns those into the lags 1..K the next credit sees."""
        B, K, _ = ring_w.shape
        h = len(dim_cols)
        E = np.zeros((B, 256, h), F)
        w = ring_w[:, :, dim_cols]                       # (B, K, h)
        ids = ring_ids.astype(np.int64)                  # (B, K)
        bidx = np.arange(B)[:, None]
        hh = np.arange(h)[None, :]
        for k in range(min(K, self.ring_k)):
            np.add.at(E, (bidx, ids[:, k, None], hh), w[:, k, :])
        return E

    # ---------- per-step (mirror the ring's decay -> credit -> add) ----
    def nhot(self, layer):
        return len(self.cols.get(layer, ()))

    def cols_of(self, layer):
        return self.cols.get(layer, np.array([], int))

    def mask(self, layer):
        if layer in self._mask:
            return self._mask[layer]
        return np.zeros(0, bool)

    def begin_step(self, layer, sig_hot):
        """Decay all of this layer's EMAs by their sigma. Call where the
        ring does `w *= decay`. No-op under fused mode (credit() decays)."""
        if self.fused:
            return
        E = self.E.get(layer)
        if E is not None:
            _tail_decay(E, np.ascontiguousarray(sig_hot, dtype=F))

    def credit(self, layer, dlds_full, sig_hot=None):
        """dlds_full: (B, dim). Returns (256, H) f32, or None if no hot dims.

        Fused mode (self.fused and sig_hot given): decay E and credit in ONE
        pass -- (E*sig) computed in place then left-folded with g, bit-identical
        to begin_step + this method. Callers skip begin_step and pass sig_hot.
        Unfused mode (sig_hot=None): credit from the already-decayed E that
        begin_step produced."""
        E = self.E.get(layer)
        if E is None:
            return None
        cols = self.cols[layer]
        g = np.ascontiguousarray(dlds_full[:, cols], dtype=F)   # (B, H)
        out = np.zeros((256, E.shape[2]), F)
        if self.fused and sig_hot is not None:
            sig = np.ascontiguousarray(sig_hot, dtype=F)       # (H,)
            _tail_fused_decay_credit(E, sig, g, out)
        else:
            _tail_credit(E, g, out)
        return out

    def add_byte(self, layer, c_arr):
        """E[b, c_arr[b], h] += 1.0. Call where the ring writes w[:, p] = 1."""
        E = self.E.get(layer)
        if E is not None:
            _tail_add(E, c_arr.astype(np.int64))

    # ---------- persistence ----------
    def count(self):
        return sum(len(c) for c in self.cols.values())

    def save_dict(self):
        d = {}
        for l, cols in self.cols.items():
            if len(cols):
                d[f"tail_cols.{l}"] = np.asarray(cols, np.int64)
                d[f"tail_E.{l}"] = np.ascontiguousarray(self.E[l], dtype=F)
        return d

    def restore_from(self, d, layers):
        """Rebuild cols/E from a checkpoint dict; B is adapted like the
        state rows in Model.load (row 0 preserved, extra streams padded)."""
        self.cols, self.E, self._mask = {}, {}, {}
        for l in range(layers):
            ck, ek = f"tail_cols.{l}", f"tail_E.{l}"
            if ck not in d:
                continue
            cols = np.asarray(d[ck], int)
            E = d[ek].astype(F)
            if E.shape[0] != self.B:
                resized = np.zeros((self.B, 256, E.shape[2]), F)
                resized[:min(self.B, E.shape[0])] = E[:min(self.B, E.shape[0])]
                E = resized
            self.cols[l] = cols
            self.E[l] = np.ascontiguousarray(E, dtype=F)
        # mask cache rebuilt lazily in sync(); nothing else to do.