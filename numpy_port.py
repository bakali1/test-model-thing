# numpy port of test-model-thing (TMT)
# Preserves the architecture: latent-space prediction, RTU (recurrent trace
# unit) layers, byte I/O, continuous streaming, test-time training.
#
# Why not MLX: MLX only accelerates via Metal (Apple Silicon) or CUDA (NVIDIA).
# On anything else it falls back to an unoptimized CPU path (~2 GFLOPS here).
# numpy + OpenBLAS uses the x86 CPU properly (~50 GFLOPS on this Sandy Bridge),
# so the same architecture runs ~20x faster without any GPU.
#
# Training uses minibatching (B independent byte-streams in lockstep):
#   - matvecs become GEMMs that OpenBLAS multithreads
#   - the dense AdamW update amortizes over B bytes instead of 1
# Per-stream recurrence, traces, and losses are mathematically identical to
# main.py; the optimizer just sees the summed gradient of B bytes per step.
#
# CLI is compatible with main.py; checkpoints are .npz instead of .safetensors.

import argparse
import glob
import itertools
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

import tail_credit
from tail_credit import TailCredit, REBASE_EVERY, HOT_FRAC

# Trace truncation tolerance: ring lag k is kept while its weight (~decay**k)
# is above this. 1e-7 reproduces the dense float32 trace (K=25 at decay 0.5);
# 1e-4 (K=15) sits ~4 orders below the stochastic gradient's own noise and is
# ~25% cheaper in the credit, so it is the default.
RING_TOL = float(os.environ.get("TMT_RING_TOL", "1e-4"))

# The ring's credit is a scatter: dembed[row] += vals[i] for B*K rows. numpy's
# np.add.at costs ~29 ns/element and a one-hot BLAS matmul wastes 256x the FLOPs;
# a compiled tight loop does the minimum work with contiguous row accesses and
# measured ~5-7x faster than the BLAS form. Optional: falls back to BLAS if numba
# is unavailable, so the port still runs on numpy alone.
try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _scatter_credit(dembed, rows, vals):
        n = rows.shape[0]
        dim = vals.shape[1]
        for i in range(n):
            r = rows[i]
            for d in range(dim):
                dembed[r, d] += vals[i, d]
except Exception:                     # pragma: no cover
    _scatter_credit = None
rng = np.random.default_rng()
F = np.float32


def sigmoid(x):
    # overflow-safe: sigmoid(x) = e^x/(1+e^x) for x<0, 1/(1+e^-x) for x>=0
    #
    # Branchless np.where rather than a boolean mask plus two fancy-indexed
    # writes (out[pos] = ..., ex = np.exp(x[~pos])). Fancy indexing on the
    # (B, dim) arrays this is called on is far slower than a branchless select.
    #
    # Same two expressions, same x>=0 split, so the result is BIT-IDENTICAL:
    # verified over 131,072,000 elements spanning N(0,1), N(0,8) and
    # U(-120,120) -- 0 differing bits, max|delta| = 0.0, dtype preserved. The
    # U(-120,120) arm is what exercises the overflow branches.
    #
    # Worth doing because this is ~20% of the CPU step: 16 calls forward and 16
    # backward per step on (B, dim) arrays. Measured 16.3 ms -> 8.7 ms per call
    # at (1024,512), i.e. 1.22x end-to-end on the CPU reference path.
    #
    # exp overflows to +inf outside roughly [-88, 88]. In the DISCARDED branch
    # that yields inf/(1+inf) = nan or 1/inf = 0, which np.where then drops, so
    # errstate only silences the warning -- no selected value is affected.
    # astype(x.dtype) rather than a hard float32: identical for the float32
    # inputs this always sees, but faithful if it is ever handed float64.
    with np.errstate(over="ignore", invalid="ignore"):
        a = 1.0 / (1.0 + np.exp(-x))
        b = np.exp(x) / (1.0 + np.exp(x))
    return np.where(x >= 0, a, b).astype(x.dtype, copy=False)


def batch_layernorm(x, g, b, eps=F(1e-5)):
    """x: (B, dim). Returns normalized (B, dim) + cache."""
    mu = x.mean(axis=1, keepdims=True)
    xv = x.var(axis=1, keepdims=True)
    inv = 1.0 / np.sqrt(xv + eps)
    xhat = (x - mu) * inv
    return g * xhat + b, (xhat, inv, g)


def batch_layernorm_backward(dy, cache):
    """dy: (B, dim). Returns dx (B, dim), dg (dim,), db (dim,)."""
    xhat, inv, g = cache
    dg = (dy * xhat).sum(axis=0)
    db = dy.sum(axis=0)
    dxhat = dy * g
    n = xhat.shape[1]
    dx = inv * (dxhat - dxhat.mean(axis=1, keepdims=True)
                - (xhat * (dxhat * xhat).sum(axis=1, keepdims=True)) / n)
    return dx.astype(F), dg.astype(F), db.astype(F)


class Layer:
    """One recurrent trace unit (RTU) layer, faithful to main.py."""

    def __init__(self, dim):
        self.dim = dim
        lim = 1.0 / np.sqrt(dim)
        self.W = rng.uniform(-lim, lim, (dim, dim)).astype(F)
        self.g = np.ones(dim, F)
        self.b = np.zeros(dim, F)
        self.decay = rng.uniform(-lim, lim, (dim,)).astype(F)
        # P3 gate. `gate` is a per-dim vector of LOGITS added to the decay
        # logit, so a = sigmoid(decay + gate * x_in). It exists only in the
        # gated arm: the default engine's theta layout -- and therefore
        # check_parity's numpy==torch identity -- must stay untouched.
        self.gate = None
        self.gated = False

    def adopt_views(self, W, g, b, decay, gate=None):
        """Rebind params to views into the model's flat buffer."""
        self.W, self.g, self.b, self.decay = W, g, b, decay
        self.gate = gate
        self.gated = gate is not None

    def forward(self, enc, x_in, states):
        """enc, x_in, states: (B, dim). Returns new x, new_states, a, cache."""
        if self.gated:
            # a = sigmoid(decay + gate * x_in): PER-SAMPLE (B, dim), because
            # the gate makes the retention depend on the input. gate=0 at
            # init makes this EXACTLY the incumbent: 0.0 * x_in is an exact
            # zero and decay + 0.0 == decay, so a == sigmoid(decay)
            # bit-for-bit.
            a = sigmoid(self.decay[None, :] + self.gate[None, :] * x_in)
        else:
            a = sigmoid(self.decay)                      # (dim,)
        state = states * a + enc                         # (B, dim)
        h, cache = batch_layernorm(state, self.g, self.b)
        z = h @ self.W.T                                 # GEMM (B, dim)
        sig = sigmoid(z)
        y = z * sig                                      # silu
        # x_in is returned because the gate reaches a_t through gate * x_in:
        # x_in is a factor of the gate's cross-time trace, not just of the
        # local term (see gate_trace_update).
        return x_in + y, state, a, (z, sig, h, cache, x_in)

    def backward(self, dx_out, cache):
        """dx_out: (B, dim). Returns dx_in, dW, dg, db, dstate."""
        z, sig, h, ln_cache = cache[:4]
        dx_in = dx_out
        dz = dx_out * sig * (1.0 + z * (1.0 - sig))      # silu'
        dW = dz.T @ h                                    # GEMM (dim, dim)
        dh = dz @ self.W                                 # GEMM (B, dim)
        dstate, dg, db = batch_layernorm_backward(dh, ln_cache)
        return dx_in, dW, dg, db, dstate


# ---------------------------------------------------------------------------
# The P3 gate gradient -- the numpy mirror of gpu_port's, element for element.
#
# The gate makes the retention input-dependent, a_t = sigmoid(decay + gate*x_t),
# so the gate enters a_t at time t ONLY -- but a_t does not stop at s_t: it
# multiplies s_{t-1} into s_t, and s_t then keeps propagating, multiplied by
# every later a. The gate's derivative therefore carries the SAME cross-time
# chain the decay's `decaytrace` carries, with one extra factor: the gate
# reaches a_t through gate * x_t, so the chain picks up x_t.
#
#     gatetrace_t = a_t * gatetrace_{t-1} + a_t(1-a_t) * x_t * s_{t-1}
#     dL/dgate    = sum_t dL/ds_t * gatetrace_t
#
# dL/ds_t here is the READOUT-ONLY state gradient -- the one the engine's
# per-byte backward produces, because the states are stop_gradient there. The
# trace is what rebuilds the chain the engine does not have. (Feeding this the
# TOTAL state gradient instead would double count the chain: the trace's chain
# and the total gradient's chain are the same chain.)
# ---------------------------------------------------------------------------
def gate_trace_update(gatetrace, a, x_in, s_prev):
    """One step of the gate's cross-time trace (numpy mirror of gpu_port's).

    `a` is the EFFECTIVE retention a_t: per-SAMPLE (B, dim) in the gated arm
    and per-dim (dim,) in the ungated one. The trace is per-sample either
    way, so a per-dim `a` is broadcast over the batch and a per-sample `a` is
    used as it is.
    """
    if a.ndim == 1:
        a = a[None, :]
    return a * gatetrace + a * (1.0 - a) * x_in * s_prev


def gate_gradient(dlds, gatetrace, a, x_in, s_prev):
    """The gate gradient for ONE timestep of ONE layer (numpy mirror).

    dlds is the readout-only state gradient dL/ds_t, (B, dim); x_in is the
    layer's input at this timestep and s_prev the state BEFORE it, both
    (B, dim). Returns (contribution, new_trace): the contribution is the
    (dim,) gate gradient for this timestep -- SUMMED OVER THE BATCH, because
    the gate parameter is per-dim and the batch is the other axis -- and the
    new trace, which the next timestep (or the next window) continues from.
    """
    gt = gate_trace_update(gatetrace, a, x_in, s_prev)
    return (dlds * gt).sum(axis=0), gt


class Model:
    def __init__(self, dim=512, layers=16, temp=0.75, lr=5e-4, decay_init=None,
                 half_life_init=None,
                 grad_clip=0.0, nan_guard=True, wd_decay=0.01, gated=False,
                 fsp=False):
        self.dim = dim
        self.layercount = layers
        self.gated = bool(gated)
        self.temp = temp
        self.lr = F(lr)
        self.b1, self.b2, self.eps, self.wd = F(0.9), F(0.999), F(1e-8), F(0.01)
        # SAFETY (PLAN-2026-10-11 P0 item 3) and P6 (per-group weight decay).
        # These mirror gpu_port exactly, because check_parity.py gates the two
        # engines against each other BIT-EXACTLY: a safety feature that exists
        # on one side only would make the parity gate measure the difference
        # in features rather than in the engine.
        self.nan_guard = bool(nan_guard)
        self.nan_refusals = 0
        self.last_nan_refused = False
        self.grad_clip = float(grad_clip)
        self.wd_decay = float(wd_decay)
        self.t = 0
        self.bytes_seen = 0          # exact training bytes, saved in checkpoints
        # (file index, byte offset within the file) -- the DATA STREAM's
        # position, persisted in the checkpoint so a resume continues the
        # corpus. None = unknown (a fresh run).
        self.data_pos = None

        lim = 1.0 / np.sqrt(dim)
        # Ring-buffer trace: retain lag k while its weight (~decay**k) is above
        # RING_TOL. The dense (B,256,dim) eligibility matrix only has O(K)
        # non-negligible rows, so the ring is the same information in 256/K less
        # memory. RING_TOL=1e-7 reproduces float32 exactly; 1e-4 is still 4 orders
        # below the stochastic gradient's own noise and cuts the GEMM cost ~1.7x.
        dmax = float(sigmoid(np.array([decay_init if decay_init is not None else lim]))[0])
        self.ring_k = int(min(256, np.ceil(np.log(RING_TOL) / np.log(max(dmax, 1e-3)))))
        init = {
            "embed": (rng.standard_normal((256, dim)) * 0.1).astype(F),
            "Wd": rng.uniform(-lim, lim, (256, dim)).astype(F),
            "bd": np.zeros(256, F),
            "Ws": rng.uniform(-lim, lim, (dim,)).astype(F),
            "bs": np.zeros(1, F),
        }
        for i in range(layers):
            init[f"layer{i}.W"] = rng.uniform(-lim, lim, (dim, dim)).astype(F)
            init[f"layer{i}.g"] = np.ones(dim, F)
            init[f"layer{i}.b"] = np.zeros(dim, F)
            # decay_init is pre-sigmoid. The default (uniform ~0) gives
            # sigmoid ~= 0.5, a one-byte memory half-life: the recurrent state
            # is then too short-lived for the decay param to get a coherent
            # gradient, which is why decay sits at 0.5 forever in both this
            # port and main.py. +3 => sigmoid 0.953, ~14-byte half-life.
            # Always draw the uniform first, even when overridden, so that an
            # A/B against the default shares one RNG stream (otherwise np.full
            # skips a draw and every later layer gets different weights).
            d0 = rng.uniform(-lim, lim, (dim,)).astype(F)
            if half_life_init is not None:
                # P4: log-spaced half-life prior, mirroring gpu_port.py
                # EXACTLY (same rng call order, so the two engines' inits are
                # identical and check_parity can gate the P4 build too).
                # half-life h means sigma = 0.5**(1/h), logit = ln(sigma/(1-sigma)).
                hmin, hmax = (float(half_life_init[0]),
                              float(half_life_init[1]))
                if not (0 < hmin <= hmax):
                    raise ValueError(
                        f"half_life_init needs 0 < half_min <= half_max, got "
                        f"{(hmin, hmax)}")
                h = np.geomspace(hmin, hmax, dim)
                sig = np.power(0.5, 1.0 / h)
                logits = np.log(sig / (1.0 - sig)).astype(F)
                rng.shuffle(logits)
                d0[:] = logits
            elif decay_init is not None:
                d0[:] = decay_init
            init[f"layer{i}.decay"] = d0
            if gated:
                # P3 gate logits, ZERO so a == sigmoid(decay) at init. Added to
                # `init` (and therefore to self.shapes and theta) only in the
                # gated arm: the default engine's theta layout must stay
                # byte-comparable with gpu_port's for check_parity. The order
                # mirrors gpu_port's exactly (gate right after the decay).
                init[f"layer{i}.gate"] = np.zeros(dim, F)

        if fsp:
            # The FSP auxiliary head's parameters, in the SAME order as
            # gpu_port.py's (`fsp.g`, `fsp.b0`, `fsp.W`, `fsp.b`), so a
            # checkpoint written by the torch engine loads here with the same
            # layout. The numpy reference does not implement the FSP FORWARD
            # -- the ruler never runs the auxiliary loss, it only needs the
            # parameter vector to line up -- and the FSP path itself is pinned
            # by tests/check_fsp.py against finite differences.
            init["fsp.g"] = np.ones(dim, F)
            init["fsp.b0"] = np.zeros(dim, F)
            init["fsp.W"] = rng.uniform(-lim, lim, (dim, 256)).astype(F)
            init["fsp.b"] = np.zeros(256, F)

        # flat parameter buffer + grads + adam moments (contiguous, in-place ops)
        self.shapes = [(k, v.shape) for k, v in init.items()]
        n = sum(int(np.prod(s)) for _, s in self.shapes)
        self.theta = np.concatenate([init[k].ravel() for k, _ in self.shapes])
        self.grads = np.zeros(n, F)
        self.m = np.zeros(n, F)
        self.v = np.zeros(n, F)

        # bind all params as views into theta
        self.views, self.gviews = {}, {}
        off = 0
        for name, shape in self.shapes:
            size = int(np.prod(shape))
            self.views[name] = self.theta[off:off + size].reshape(shape)
            self.gviews[name] = self.grads[off:off + size].reshape(shape)
            off += size

        self.embed = self.views["embed"]
        self.Wd, self.bd = self.views["Wd"], self.views["bd"]
        self.Ws, self.bs = self.views["Ws"], self.views["bs"]
        self.layers = [Layer(dim) for _ in range(layers)]
        for i, l in enumerate(self.layers):
            l.adopt_views(self.views[f"layer{i}.W"], self.views[f"layer{i}.g"],
                          self.views[f"layer{i}.b"], self.views[f"layer{i}.decay"],
                          self.views.get(f"layer{i}.gate"))

        # recurrent state + traces; stream 0 is the persistent "memory" stream
        self.B = 1
        self.new_states(1)

        # tail-credit (exact long-lag embed credit) is created lazily at the
        # first training step / in load(), sized to the active batch B.
        # Read the env var per-instance so an A/B can flip TMT_TAIL between
        # model constructions in one process.
        self.tail = None
        self.tail_state = {}   # checkpoint keys handed to a fresh tail
        self.tail_enabled = os.environ.get("TMT_TAIL", "1") != "0"

    def new_states(self, B):
        self.B = B
        self.ring_b = np.zeros((B, self.ring_k), np.int64)      # byte per lag
        self.ring_rows = self.ring_b.ravel()                    # view, kept fresh
        # one-hot selector, only needed by the BLAS fallback (unused with numba)
        self.ring_M = (np.zeros((256, B * self.ring_k), F)
                       if _scatter_credit is None else None)
        self.ring_vals = np.zeros((B, self.ring_k, self.dim), F)   # scratch, no per-step alloc
        for l in self.layers:
            l.states = np.zeros((B, self.dim), F)
            l.decaytrace = np.zeros((B, self.dim), F)
            # The gate's cross-time trace, allocated ONLY in the gated arm:
            # the default engine must not pay the buffer. It lives next to
            # the decaytrace because it is the same kind of object: the
            # carrier of the chain the engine's per-byte backward does not
            # have.
            if getattr(l, "gated", False):
                l.gatetrace = np.zeros((B, self.dim), F)
            l.ring_w = np.zeros((B, self.ring_k, self.dim), F)

    # ---------- tail credit (exact long-lag embed credit) ----------
    def _ensure_tail(self):
        """Create/resize the tail accumulator for the current B and reconcile
        hot-dim membership against the current decay. Idempotent; a no-op by
        default but ~microseconds once the tail matches B."""
        if not getattr(self, "tail_enabled", False):
            return
        if self.tail is None or self.tail.B != self.B:
            self.tail = TailCredit(self.B, self.ring_k)
            if self.tail_state:
                self.tail.restore_from(self.tail_state, self.layercount)
            self.tail_state = {}
        self._tail_sync()

    def _tail_sync(self):
        """Re-evaluate hot dims against the current decay and re-seed any
        newly-hot dim from the ring. Already-hot dims keep their exact EMA."""
        if self.tail is None:
            return
        sigs = np.stack([sigmoid(l.decay) for l in self.layers]).astype(F)
        self.tail.sync(sigs, [l.ring_w for l in self.layers], self.ring_b)

    def params(self):
        return self.views

    def reset(self):
        self.ring_b[:] = 0
        for l in self.layers:
            l.states[:] = 0
            l.decaytrace[:] = 0
            if getattr(l, "gated", False):
                l.gatetrace[:] = 0
            l.ring_w[:] = 0

    def count(self):
        per_layer = self.dim * self.dim + 3 * self.dim
        if self.gated:
            per_layer += self.dim          # the P3 gate vector
        return (256 * self.dim + self.layercount * per_layer +
                256 * self.dim + 256 + self.dim + 1)

    # ---------- inference (single stream, frozen) ----------
    def frozen_step(self, c):
        s = 0  # stream 0 carries persistent memory
        enc = self.embed[c]
        x = enc[None, :]
        for l in self.layers:
            x, new_state, _, _ = l.forward(enc.reshape(1, -1), x, l.states[s:s + 1])
            l.states[s:s + 1] = new_state
        output = (x[0] @ self.Wd.T + self.bd)
        stop = sigmoid(x[0] @ self.Ws + self.bs)[0]
        return self.sample(output), stop

    def sample(self, output):
        probs = np.exp(output - output.max())
        probs /= probs.sum()
        entropy = -(probs * np.log(probs + 1e-8)).sum() / np.log(256.0)
        temp = max(0.1, self.temp * (1.0 - self.temp * entropy))
        logits = output / temp
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return int(rng.choice(256, p=p))

    # ---------- training (B streams in lockstep) ----------
    def step_batch(self, cs, ns, ends):
        """cs, ns, ends: lists of length B. One AdamW update for B byte-steps."""
        # P3 gated recurrence: REFUSED, exactly as gpu_port refuses it, and for
        # the same reason. Gating makes the retention factor per-sample
        # (B, dim), while the ring credit, the tail and the ring_k inference
        # all assume a per-dim (dim,) vector. That is a separate, larger piece
        # of work; until it lands, refusing is the honest answer.
        #
        # What is NOT missing is the gradient: the gate gradient is implemented
        # below and in gate_gradient() above, in the traced form, and
        # tests/check_gate_grad.py pins it against torch autograd and against
        # a central finite difference -- and pins this file's copy against
        # gpu_port's, element by element.
        #
        # The refusal comes FIRST, before self.t += 1, so that a refused call
        # leaves theta, the moments and the step counter exactly as they were.
        if self.gated:
            raise NotImplementedError(
                "gated=True (P3) is not yet runnable: gating makes the retention "
                "factor per-sample (B, dim), but the ring-credit, tail and "
                "decaytrace machinery all assume a per-dim (dim,) decay vector, "
                "and ring_k is inferred from sigma_max which is then not a "
                "scalar. Reworking that is a separate, deliberate change -- see "
                "PLAN-2026-10-11-T-AXIS.md P3. Refusing rather than emitting a "
                "plausible wrong credit. (The gate GRADIENT itself is not the "
                "blocker any more: it is the traced form, pinned against "
                "autograd and finite differences by tests/check_gate_grad.py.)")
        self.t += 1
        B = len(cs)
        self.bytes_seen += B
        c_arr = np.asarray(cs, np.int64)
        n_arr = np.asarray(ns, np.int64)
        end_arr = np.asarray(ends, F)

        enc = self.embed[c_arr]                          # (B, dim)

        # ---- forward ----
        x = enc
        caches, states, decays = [], [], []
        # The layer INPUT at each timestep, per layer, for THIS byte. The
        # gate reaches a_t through gate * x_in, so x_in is a factor of the
        # gate's cross-time trace (see gate_trace_update).
        xins = []
        for l in self.layers:
            x, state, decay, cache = l.forward(enc, x, l.states)
            caches.append(cache)
            states.append(state)
            decays.append(decay)
            xins.append(cache[4])

        output = x @ self.Wd.T + self.bd                 # (B, 256)
        stop = sigmoid(x @ self.Ws + self.bs[0])         # (B,)

        # ---- loss components (per stream) ----
        xv = x.var(axis=1) + F(1e-4)
        var_active = (1.0 - np.sqrt(xv)) > 0
        tgt = self.embed[n_arr]                          # (B, dim)
        lse = output.max(axis=1, keepdims=True) + np.log(
            np.exp(output - output.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
        ce = (lse[:, 0] - output[np.arange(B), n_arr])
        stop_t = end_arr
        loss = float((np.where(var_active, 1.0 - np.sqrt(xv), 0.0).sum()
                      + ((x - tgt) ** 2).sum() / self.dim
                      + ce.sum() + ((stop - stop_t) ** 2).sum()) / B)
        # CE component alone (nats/byte), so training logs are comparable to
        # eval_npz's "model" line; the total loss also carries the variance
        # hinge, the embed-MSE and the stop-head MSE.
        self.last_ce = float(ce.mean())

        # ---- backward: decoder ----
        doutput = np.exp(output - lse)                   # softmax
        doutput[np.arange(B), n_arr] -= 1.0
        dWd = doutput.T @ x                              # (256, dim) GEMM
        dbd = doutput.sum(axis=0)
        dx = doutput @ self.Wd                           # (B, dim)

        dstop = 2.0 * (stop - stop_t) * stop * (1.0 - stop)
        dWs = (dstop[:, None] * x).sum(axis=0)           # (dim,)
        dbs = dstop.sum(axis=0, keepdims=True)
        dx += dstop[:, None] * self.Ws[None, :]

        # variance hinge loss grad (masked where inactive)
        dvar = np.where(var_active, -1.0 / (2.0 * np.sqrt(xv)), 0.0)[:, None]
        dx += (dvar * 2.0 * (x - x.mean(axis=1, keepdims=True)) / self.dim).astype(F)

        # MSE grad (per-stream mean over dim)
        dx += (2.0 * (x - tgt) / self.dim).astype(F)

        # ---- propagate through layers ----
        denc = dx.copy()
        lw, lg, lb = [], [], []
        lgrads = []
        for i in reversed(range(self.layercount)):
            l = self.layers[i]
            dx_in, dW, dg, db, dstate = l.backward(dx, caches[i])
            dx = dx_in
            denc += dstate
            lw.append(dW); lg.append(dg); lb.append(db)
            lgrads.append(dstate)                        # == dlds per stream
        lgrads.reverse()
        lw.reverse(); lg.reverse(); lb.reverse()

        # ---- trace updates (ring buffer, BLAS credit) ----
        # dembed[v] += sum over retained lags landing on byte v of w_k * dlds.
        # Doing that with np.add.at costs ~29 ns/element (unbuffered scatter);
        # instead build a (256, B*K) one-hot selector and let BLAS do the
        # reduction (~1 ns/element). The selector's byte indices are shared by all
        # layers, so it is built once per step.
        self._ensure_tail()
        dembed = np.zeros((256, self.dim), F)
        decay_grads = []
        # P3 gate gradients, per layer, accumulated over the window exactly
        # like the decay's.
        gate_grads = []
        K = self.ring_k
        p = (self.t - 1) % K
        scratch = self.ring_vals
        if _scatter_credit is None:                 # BLAS fallback needs the selector
            M = self.ring_M
            M.fill(0)
            M[self.ring_rows, np.arange(B * K)] = 1.0
        for i, l in enumerate(self.layers):
            dlds = lgrads[i]                             # (B, dim)
            w = l.ring_w                                 # (B, K, dim) lag weights
            # main.py credits the trace to the FULL (256, dim) embedding matrix:
            #   grads["encoder"]["embed"]["weight"] += dlds * (embedtrace * decay)
            # using the pre-update trace. Reading only row c_b drops the temporal
            # credit for every byte except the current one (and double-counts it).
            #
            # The (B,256,dim) bookkeeping is memory-bound (~117 MB moved per layer
            # per step), so: compute the decayed trace ONCE and reuse it for both
            # the credit and the update (it used to be computed twice), sum across
            # streams with einsum (bit-identical to the broadcast+sum it replaces,
            # but without materialising a (B,256,dim) product), and add this byte's
            # one-hot in place instead of building a (B,256,1) array.
            # decay first: the dense credit uses (T_{t-1} * decay_t), not T_{t-1}
            w *= decays[i]
            if self.tail is not None and self.tail.nhot(i):
                self.tail.begin_step(i, decays[i][self.tail.cols_of(i)])
            np.multiply(w, dlds[:, None, :], out=scratch)
            if self.tail is not None and self.tail.nhot(i):
                # ring skips hot dims (their credit would be truncated); the
                # tail EMA covers them exactly at every lag
                scratch[:, :, self.tail.mask(i)] = 0.0
                tailc = self.tail.credit(i, dlds)
                if tailc is not None:
                    np.add.at(dembed, (slice(None), self.tail.cols_of(i)), tailc)
            sv = scratch.reshape(B * K, self.dim)
            if _scatter_credit is not None:
                _scatter_credit(dembed, self.ring_rows, sv)
            else:
                np.add(dembed, M @ sv, out=dembed)
            w[:, p] = 1.0        # this byte becomes lag 1 (empty product = 1)
            if self.tail is not None and self.tail.nhot(i):
                self.tail.add_byte(i, c_arr)
            l.decaytrace = (decays[i][None, :] * l.decaytrace +
                            decays[i][None, :] * (1.0 - decays[i][None, :]) * l.states)
            decay_grads.append((dlds * l.decaytrace).sum(axis=0))
            if self.gated:
                # The GATE's chain, which the plain sum did not have -- the
                # numpy mirror of gpu_port's block. The gate enters a_t at
                # time t only, but a_t keeps multiplying s_{t-1} into every
                # later state, so the gate's derivative carries the same
                # cross-time chain the decay's decaytrace carries -- plus the
                # x_t factor, because the gate reaches a_t through gate * x_t.
                #
                # `dlds` is the READOUT-ONLY state gradient (the states are
                # stop_gradient) and the trace rebuilds the chain the engine
                # does not have. `decays[i]` is the EFFECTIVE retention a_t:
                # (B, dim) in the gated arm, (dim,) in the ungated one.
                # `l.states` still holds the state BEFORE this byte.
                contrib, l.gatetrace = gate_gradient(
                    dlds, l.gatetrace, decays[i], xins[i], l.states)
                gate_grads.append(contrib)
            l.states = states[i]                         # stop_gradient
        self.ring_b[:, p] = c_arr

        gv = self.gviews
        np.add.at(dembed, c_arr, denc)                   # autograd part
        gv["embed"][:] = dembed
        gv["Wd"][:] = dWd
        gv["bd"][:] = dbd
        gv["Ws"][:] = dWs
        gv["bs"][:] = dbs
        for i in range(self.layercount):
            gv[f"layer{i}.W"][:] = lw[i]
            gv[f"layer{i}.g"][:] = lg[i]
            gv[f"layer{i}.b"][:] = lb[i]
            gv[f"layer{i}.decay"][:] = decay_grads[i]
            if self.gated:
                gv[f"layer{i}.gate"][:] = gate_grads[i]

        # ---- fused AdamW (in-place, one pass over 4.5M params) ----
        t = self.t
        th, gr, m, v = self.theta, self.grads, self.m, self.v
        # NaN guard: a non-finite gradient is REFUSED (theta, m, v untouched),
        # and counted. See gpu_port.py for the full statement of the contract.
        self.last_nan_refused = False
        finite = True
        if self.nan_guard:
            finite = bool(np.isfinite(gr).all())
            if not finite:
                self.nan_refusals += 1
                self.last_nan_refused = True
        if finite:
            if self.grad_clip > 0.0:
                gn = float(np.linalg.norm(gr))
                coef = self.grad_clip / (gn + 1e-6)
                if coef < 1.0:
                    gr *= coef
            m *= self.b1; m += (1 - self.b1) * gr
            v *= self.b2; v += (1 - self.b2) * (gr * gr)
            np.sqrt(v, out=v)
            th -= self.lr * (m * (1 / (1 - self.b1 ** t)) /
                             (v * (1 / (1 - self.b2 ** t)) + self.eps)
                             + self._wd_coord_vec() * th)

        return loss

    def _wd_coord_vec(self):
        """Per-coordinate weight decay: uniform `wd` except the `.decay` group,
        which gets `wd_decay` (PLAN-2026-10-11 P6). With the default
        wd_decay == wd this is uniform and bit-identical to `wd * th`."""
        if getattr(self, "_wd_vec", None) is None:
            wv = np.full(self.theta.shape, self.wd, dtype=F)
            off = 0
            for name, shape in self.shapes:
                k = int(np.prod(shape)) if shape else 1
                if name.endswith(".decay") or name.endswith(".gate"):
                    wv[off:off + k] = self.wd_decay
                off += k
            self._wd_vec = wv
        return self._wd_vec

    # ---------- persistence ----------
    def save(self, path):
        data = {k: v for k, v in self.params().items()}
        for i, l in enumerate(self.layers):
            data[f"state.{i}"] = l.states
            data[f"decaytrace.{i}"] = l.decaytrace
            # The GATE's trace is NOT persisted (see gpu_port.params_dict):
            # the gated arm refuses to run today, so the buffer is always
            # zero. It must be added here when the per-sample credit
            # plumbing lands, or a resumed gated run would silently lose the
            # chain across the resume.
        data["t"] = np.array(self.t, np.int64)
        data["bytes"] = np.array(self.bytes_seen, np.int64)
        # The DATA STREAM's position, so a resume continues the corpus
        # instead of replaying it (see gpu_port.params_dict for why this
        # matters and what it costs if it is missing).
        if getattr(self, "data_pos", None) is not None:
            data["data_file"] = np.array(self.data_pos[0], np.int64)
            data["data_off"] = np.array(self.data_pos[1], np.int64)
        data["m"] = self.m
        data["v"] = self.v
        data["B"] = np.array(self.B, np.int64)
        data["ring_k"] = np.array(self.ring_k, np.int64)
        data["algorithm_version"] = np.array(0, np.int64)
        if self.tail is not None:
            data.update(self.tail.save_dict())
        tmp = path + ".tmp.npz"
        np.savez(tmp, **data)
        os.replace(tmp, path)

    def load(self, path):
        if not os.path.exists(path):
            return
        d = np.load(path)
        if "algorithm_version" in d and int(np.ravel(d["algorithm_version"])[0]) != 0:
            raise ValueError(
                f"checkpoint {path} uses factored GPU algorithm state; "
                f"numpy_port is the version-0 reference and cannot reinterpret it")
        # ring_k is a property of the checkpoint, not of this process's
        # RING_TOL: the run that wrote it sized its ring for its own decay
        # range, and re-deriving it here (from an unrelated --decay-init or
        # TMT_RING_TOL) silently truncates the embed credit for the hottest
        # dims -- e.g. the live run's K=18 shrinks to 14 at the default tol.
        # Honor the stored K and resize the ring buffers to match before any
        # state is read into them. (The eval tools already follow this rule.)
        if "ring_k" in d:
            ck_k = int(d["ring_k"])
            if ck_k != self.ring_k:
                self.ring_k = ck_k
                self.tail = None      # rebuilt below at the checkpoint's K
                self.new_states(self.B)
        if "t" in d:
            self.t = int(d["t"])
        # Training-byte counter. Checkpoints saved before this field existed
        # don't have it; fall back to t*B, which is exact unless the batch size
        # changed at some point in the model's history.
        self.bytes_seen = int(d["bytes"]) if "bytes" in d else self.t * self.B
        if "data_file" in d and "data_off" in d:
            self.data_pos = (int(d["data_file"]), int(d["data_off"]))
        if "m" in d:
            if d["m"].shape != self.m.shape:
                raise ValueError(
                    f"checkpoint {path} holds {int(d['m'].shape[0]):,} parameters but "
                    f"this model was built with {self.m.size:,} "
                    f"(dim={self.dim}, layers={self.layercount}); pass matching "
                    f"--dim/--layers")
            self.m[:] = d["m"]
            self.v[:] = d["v"]
        # ring_k is sized from the *init* decay. If this checkpoint carries
        # larger decay values, the first dropped lag keeps a nonzero weight
        # (sigma^K) and the embed credit loses that share. Note: only the embed
        # credit is truncated -- the decay gradient uses the dense decaytrace,
        # so timescale learning itself is unaffected. Warn on the absolute
        # magnitude, not the ratio to the requested tolerance.
        # (checked at the END of load, once theta holds the loaded values)

        off = 0
        for name, shape in self.shapes:
            if name in d:
                size = int(np.prod(shape))
                self.theta[off:off + size] = d[name].astype(F).ravel()
            off += size
        # Recurrent buffers are shaped (B, ...). Adapt their row count to the
        # caller's self.B instead of adopting the checkpoint's B: a B=32
        # checkpoint loaded into a B=1 chat step used to broadcast silently and
        # then die in the trace update. Row 0 (the persistent memory stream) is
        # preserved; extra streams are zero-padded.
        for i, l in enumerate(self.layers):
            for key, attr in ((f"state.{i}", "states"),
                              (f"decaytrace.{i}", "decaytrace")):
                if key not in d:
                    continue
                v = d[key].astype(F)
                if v.shape[0] != self.B:
                    resized = np.zeros((self.B,) + v.shape[1:], F)
                    n = min(self.B, v.shape[0])
                    resized[:n] = v[:n]
                    v = resized
                setattr(l, attr, v)

        # tail-credit state (exact long-lag embed credit); handed to a freshly
        # created tail in _ensure_tail (B is adapted there like the state rows)
        self.tail_state = {}
        for i in range(self.layercount):
            for key in (f"tail_cols.{i}", f"tail_E.{i}"):
                if key in d:
                    self.tail_state[key] = d[key]
        if self.tail_enabled:
            self._ensure_tail()

        if self.layers:
            sigs = [sigmoid(l.decay) for l in self.layers]
            if self.tail_enabled:
                hot = np.concatenate([s[s ** self.ring_k > HOT_FRAC] for s in sigs])
                if hot.size:
                    print(f"[tail] {hot.size} hot dim(s) get exact embed credit "
                          f"(K={self.ring_k}, sigma**K > {HOT_FRAC:.0%}); "
                          f"worst would have dropped {hot.max() ** self.ring_k:.1%}")
                cold = np.concatenate([s[s ** self.ring_k <= HOT_FRAC] for s in sigs])
                cut = float(cold.max()) ** self.ring_k if cold.size else 0.0
                if cut > HOT_FRAC:
                    print(f"[warn] cold-dim truncation {cut:.1%} at lag "
                          f"{self.ring_k} (tol {RING_TOL:g})", file=sys.stderr)
            else:
                sig_max = max(float(s.max()) for s in sigs)
                cut = sig_max ** self.ring_k
                if cut > 0.01:
                    print(f"[warn] trace truncation: heaviest dim "
                          f"(sigma={sig_max:.3f}) still carries {cut:.1%} of its "
                          f"gradient at lag {self.ring_k} (tol {RING_TOL:g}); "
                          f"raise TMT_RING_TOL for more lags", file=sys.stderr)


class Runtime:
    def __init__(self, path, threshold=0.35, batch=8, maxtokens=None,
                 save_every=1000, max_steps=0, data_seed=0, tagged=True,
                 keep_k=8, **kw):
        self.model = Model(**kw)
        self.path = path
        self.threshold = threshold
        self.batch = batch
        # Fixes the training file order (and therefore which bytes an arm
        # sees) so two A/B runs share one byte stream. Logged at startup.
        self.data_seed = data_seed
        # Tagged saves: `tmt_t061800.npz` next to the rolling `tmt.npz`, keep
        # the last `keep_k`. The rolling path is a hard link to the newest tag,
        # so chat/eval keep working while no result's checkpoint is destroyed.
        self.tagged = tagged
        self.keep_k = max(1, keep_k)
        # Stop training after this many steps (0 = never). Same as main.py's
        # TMT_STEPS; makes two runs comparable at an exact step budget.
        self.max_steps = max_steps
        # Steps between checkpoint writes. A B=32 checkpoint is ~322 MB (the
        # per-layer (B,256,dim) embedtrace dominates), so saving every 100
        # steps -- the old hardcoded interval -- meant ~900 GB written per day
        # of training. 0 disables periodic saves (exit/^C still saves).
        self.save_every = save_every
        # Same guard as main.py: the stop head is a learned signal and can sit
        # below the threshold forever, so generation also needs a hard cap.
        self.maxtokens = (int(os.environ.get("TMT_MAXTOKENS", "2000") or 2000)
                          if maxtokens is None else maxtokens)
        self.step = 0
        self.prevtime = None

    def save(self):
        self.step += 1
        if self.step % 100 == 0:
            self.model.save(self.path)

    def save_checkpoint(self):
        """Train-path save: durable tagged checkpoint + rolling path.

        Writes `{base}_t{t:06d}.npz` atomically, points the rolling path
        (`tmt.npz`) at it (hard link, falling back to a copy), and prunes all
        but the newest `keep_k` tags. Never deletes the checkpoint a rollout
        was measured on before it ages out of the window.
        """
        m = self.model
        if not self.tagged:
            m.save(self.path)
            return
        directory = os.path.dirname(self.path) or "."
        base = os.path.splitext(os.path.basename(self.path))[0]
        tag = os.path.join(directory, f"{base}_t{m.t:06d}.npz")
        m.save(tag)
        tmp = self.path + ".lnk.tmp"
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
            os.link(tag, tmp)
            os.replace(tmp, self.path)
        except OSError:
            import shutil
            shutil.copyfile(tag, tmp)
            os.replace(tmp, self.path)
        for old in sorted(glob.glob(os.path.join(directory, f"{base}_t*.npz")))[:-self.keep_k]:
            try:
                os.unlink(old)
            except OSError:
                pass

    def write(self, b):
        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def chat(self, save, frozen):
        model = self.model
        model.new_states(1)
        model.load(self.path)
        print()
        while True:
            try:
                text = input(f"\n[{self.now()} | "
                             f"{0 if self.prevtime is None else time.time() - self.prevtime:.4f}s]\nUser >> ")
            except EOFError:
                print("\n[EOF, exiting chat]")
                break
            self.prevtime = time.time()
            data = (text + "\n").encode("utf-8")
            t0 = time.time()
            for i, (c, n) in enumerate(itertools.pairwise(data)):
                self.ingest(c, n, i == len(data) - 2, save, frozen)
            rate = (len(data) - 1) / max(1e-9, time.time() - t0)
            print(f"\n[{self.now()}] (ingest {rate:.0f} B/s)\nModel >> ", end="", flush=True)
            b = data[-1]
            generated = 0
            while True:
                b, stop = model.frozen_step(b)
                self.write(b)
                generated += 1
                if stop > self.threshold:
                    print()
                    break
                if generated >= self.maxtokens:
                    print(f"\n[stopped: max tokens reached ({self.maxtokens})]")
                    break
            if save and not frozen:
                model.save(self.path)

    def ingest(self, c, n, end, save, frozen):
        if frozen:
            model = self.model
            b, stop = model.frozen_step(c)
        else:
            self.model.step_batch([c], [n], [end])
            b, stop = None, None
        if save and not frozen:
            self.save()
        return b, stop

    @staticmethod
    def make_streams(data, B):
        """Split `data` into B byte-streams by round-robining whole lines.

        Returns (streams, ends) where ends[j][t] is the stop-head target for
        step t of stream j: True on the byte-pair that finishes a line. main.py
        streams line-by-line and so flags one end per line; slicing a 40 MB file
        into B contiguous chunks instead flagged exactly one end per stream
        (~6.5M steps), which starves the stop head and teaches it to never
        stop."""
        lines = data.splitlines(keepends=True)
        buckets = [lines[i::B] for i in range(B)]
        if any(not bucket for bucket in buckets):
            # Fewer lines than streams: fall back to contiguous chunks.
            chunk = len(data) // B
            streams = [data[i * chunk:(i + 1) * chunk] for i in range(B)]
            ends = []
            for s in streams:
                e = np.zeros(len(s), bool)
                if len(s) >= 2:
                    e[len(s) - 2] = True
                ends.append(e)
            return streams, ends

        streams, ends = [], []
        for bucket in buckets:
            s = b"".join(bucket)
            e = np.zeros(len(s), bool)
            off = 0
            for line in bucket:
                off += len(line)
                if len(line) >= 2:       # 1-byte lines yield no byte-pair
                    e[off - 2] = True
            streams.append(s)
            ends.append(e)
        return streams, ends

    def train(self, save, frozen, dataset):
        model = self.model
        B = self.batch
        model.new_states(B)   # shape buffers before load, so load() adapts rows
        model.load(self.path)  # to B and keeps row 0's trained memory
        files = glob.glob(dataset, recursive=True)
        if not files:
            raise FileNotFoundError(f"Could not find training files: {dataset!r}")
        random.Random(self.data_seed).shuffle(files)
        print(f"[data] order seed={self.data_seed} | {len(files)} file(s): "
              + ", ".join(os.path.basename(f) for f in files[:10])
              + (" ..." if len(files) > 10 else ""), flush=True)
        total = 0
        loss_win = ce_win = 0.0          # windowed means between progress prints
        n_win = 0
        t0 = time.time()
        while True:
            for file in files:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    data = f.read().encode("utf-8", errors="ignore")
                if len(data) < B * 4:
                    continue
                streams, endss = self.make_streams(data, B)
                L = min(len(s) for s in streams) - 1
                if L < 1:
                    continue
                for t in range(L):
                    cs = [s[t] for s in streams]
                    ns = [s[t + 1] for s in streams]
                    loss_win += model.step_batch(cs, ns, [e[t] for e in endss])
                    ce_win += model.last_ce
                    n_win += 1
                    total += B
                    if self.max_steps and model.t >= self.max_steps:
                        print(f"  [reached step budget {self.max_steps}, stopping]")
                        return
                    if save and self.save_every and model.t % self.save_every == 0:
                        self.save_checkpoint()
                    if total % (B * 100) == 0:
                        dt = time.time() - t0
                        print(f"  {total} bytes | {total / dt:.1f} B/s | step {model.t} | "
                              f"loss {loss_win / n_win:.3f} | ce {ce_win / n_win:.3f}",
                              flush=True)
                        loss_win = ce_win = 0.0
                        n_win = 0

    def now(self):
        return datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    def __call__(self, mode, dataset, save, frozen):
        print()
        crashed = False
        try:
            if mode == "train":
                self.train(save, frozen, dataset)
            elif mode == "chat":
                self.chat(save, frozen)
            elif mode == "bench":
                self.bench(frozen)
        except Exception:
            # A hard error can leave states/traces half-updated: don't overwrite
            # a good checkpoint with that. KeyboardInterrupt is a BaseException,
            # so ^C still saves (that is how a train run is meant to end).
            crashed = True
            raise
        finally:
            # Frozen runs are read-only, matching main.py: never rewrite the
            # checkpoint from a frozen chat (it would also shrink the state
            # rows to B=1).
            if save and not frozen and not crashed:
                if mode == "train":
                    self.save_checkpoint()
                elif mode == "chat":
                    self.model.save(self.path)

    def bench(self, frozen):
        model = self.model
        text = ("The quick brown fox jumps over the lazy dog. "
                "Machine learning models learn patterns from data.\n").encode()
        if frozen:
            N = 300
            model.new_states(1)
            t0 = time.time()
            for i in range(N):
                model.frozen_step(97 + (i % 26))
            dt = time.time() - t0
            print(f"params: {model.count():,}")
            print(f"frozen: {dt / N * 1000:.1f} ms/byte -> {N / dt:.1f} B/s (MLX: 12.1 B/s)")
        else:
            B = self.batch
            model.new_states(B)
            stream = (text * ((B * 300) // len(text) + 1))
            chunk = len(stream) // B
            streams = [stream[i * chunk:(i + 1) * chunk] for i in range(B)]
            N = min(chunk - 1, 300)
            t0 = time.time()
            for t in range(N):
                model.step_batch([s[t] for s in streams], [s[t + 1] for s in streams], [False] * B)
            dt = time.time() - t0
            print(f"params: {model.count():,}")
            print(f"train (batch {B}): {dt / N * 1000:.1f} ms/step -> {N * B / dt:.1f} B/s (MLX: 6.1 B/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test-model-thing (numpy port)")
    parser.add_argument("path")
    parser.add_argument("mode", choices=["train", "chat", "bench"])
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--no-save", action="store_false")
    parser.add_argument("--dataset", default="data/wikipedia_clean/**/wiki_*")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temp", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for init + sampling (default: nondeterministic)")
    parser.add_argument("--decay-init", type=float, default=None,
                        help="pre-sigmoid decay init, e.g. 3 for sigmoid 0.95 "
                             "(default: uniform ~0 => sigmoid ~0.5, 1-byte memory)")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="steps between checkpoint writes; 0 = only on exit/^C")
    parser.add_argument("--data-seed", type=int, default=0,
                        help="seed for the training file shuffle (A/B arms share "
                             "a seed => identical byte streams; default 0)")
    parser.add_argument("--keep-k", type=int, default=8,
                        help="tagged checkpoints to retain (default 8)")
    parser.add_argument("--no-tagged", action="store_false", dest="tagged",
                        help="overwrite the rolling checkpoint instead of "
                             "writing tmt_t{step}.npz tags")
    parser.add_argument("--steps", type=int, default=0,
                        help="stop training after N steps and save; 0 = run forever")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="chat stop-head cutoff (main.py uses 0.25)")
    args = parser.parse_args()

    if args.seed is not None:
        rng = np.random.default_rng(args.seed)

    runtime = Runtime(path=args.path, dim=args.dim, layers=args.layers,
                      temp=args.temp, lr=args.lr, batch=args.batch,
                      decay_init=args.decay_init, save_every=args.save_every,
                      max_steps=args.steps, threshold=args.threshold,
                      data_seed=args.data_seed, tagged=args.tagged,
                      keep_k=args.keep_k)
    print(f"parameters: {runtime.model.count():,}")
    runtime(args.mode, args.dataset, args.no_save, args.frozen)
