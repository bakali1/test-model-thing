"""gpu_port.py — torch translation of numpy_port.step_batch (TMT training engine).

Semantics-preserving transcription of numpy_port.Model / step_batch. Same
parameter set, same flat-buffer + inline-AdamW design, same ring/tail-credit
bookkeeping, same .npz checkpoint format. Runs on CPU (any torch build) or
CUDA (auto device). fp32 everywhere, no autocast, no TF32.

Parity contract (tests/check_parity.py): given the same init (from a numpy
model via `from_numpy`) and the same byte stream, per-step loss/last_ce match
to ~1e-4 relative and per-parameter gradients to ~1e-4 relative, with the tail
engaged. Anything beyond tolerance is a bug here, not a tolerance.

The flat elementwise AdamW below is NOT torch.optim.AdamW: numpy's update
(`th -= lr*(m*bc1/(v*bc2+eps) + wd*th)` with v pre-sqrt'd) is transcribed
elementwise, so every element's arithmetic matches numpy_port's exactly on any
device (matmul accumulation order is the only tolerated difference).

Author: baku (night session 2026-09-23). If another port variant exists,
merge toward THIS file only if it passes tests/check_parity.py.
"""
import argparse
import os

import numpy as np
import torch

try:  # Optional: used only by the measured Phase-C fused-ring candidate.
    import triton
    import triton.language as tl
except ImportError:  # CPU wheels and the laptop correctness gates do not ship Triton.
    triton = None
    tl = None

import tail_credit as _tc  # HOT_FRAC / REBASE_EVERY constants

RING_TOL = float(os.environ.get("TMT_RING_TOL", "1e-4"))
HOT_FRAC = _tc.HOT_FRAC          # 0.01
REBASE_EVERY = _tc.REBASE_EVERY  # 500
F = torch.float32
# Bit 0 stores undecayed ring values with per-slot decay factors; bit 1 stores
# undecayed tail EMAs with per-column decay factors. Version 0 is the live
# numpy-compatible algorithm and remains the production default.
ALGORITHM_VERSION_LEGACY = 0


def _sig(x):
    return torch.sigmoid(x)


# Phase profiling hooks. `record_function` is ~1 us per region and the regions
# below cost ~0.02% of a 48 ms step, so they are on by default; set
# TMT_PROFILE_REGIONS=0 to remove even that.
_PROFILE_REGIONS = os.environ.get("TMT_PROFILE_REGIONS", "1") != "0"
try:
    from torch.profiler import record_function as _record_function

    def _region(name):
        return _record_function(name)
except Exception:                                    # pragma: no cover
    def _region(name):
        return _null_region()


class _null_region:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _layer_norm(x, g, b, eps=1e-5):
    """Mirror of numpy_port.batch_layernorm (biased var, eps=1e-5)."""
    mu = x.mean(dim=1, keepdim=True)
    xv = ((x - mu) ** 2).mean(dim=1, keepdim=True)
    inv = 1.0 / torch.sqrt(xv + eps)
    xhat = (x - mu) * inv
    return g * xhat + b, (xhat, inv, g)


def _layer_norm_backward(dy, cache):
    xhat, inv, g = cache
    dxhat = dy * g
    n = xhat.shape[1]
    dx = inv * (dxhat - dxhat.mean(dim=1, keepdim=True)
                - (xhat * (dxhat * xhat).sum(dim=1, keepdim=True)) / n)
    dg = (dy * xhat).sum(dim=0)
    db = dy.sum(dim=0)
    return dx, dg, db


class TorchTail:
    """per-step tail math on torch tensors, mirroring tail_credit.TailCredit.

    Legacy (numpy-identical) decay is the only supported mode. A factored
    variant was implemented, measured at 0.98x on the T4 (no win) and removed;
    it had also cost four correctness bugs, so the simpler exact form stands
    (see RESULTS.md, "factored tail bugs").
    """

    def __init__(self, B, ring_k, hot_frac=HOT_FRAC, device="cpu", cap_gb=None):
        self.B = B
        self.ring_k = ring_k
        self.hot_frac = hot_frac
        self.device = device
        self.cap_bytes = None if cap_gb is None else float(cap_gb) * 1e9
        self.cols = {}   # layer -> int64 tensor (H,) on device
        self.mask = {}   # layer -> bool (dim,) on device
        self.E = {}      # layer -> (B,256,H) fp32 on device

    @staticmethod
    def hot_from_np(sigmas, ring_k, hot_frac):
        return sigmas ** ring_k > hot_frac

    def nhot(self, layer):
        return self.cols.get(layer, torch.empty(0, dtype=torch.long,
                                                device=self.device)).numel()

    def cols_of(self, layer):
        return self.cols.get(layer, torch.empty(0, dtype=torch.long,
                                                device=self.device))

    def mask_of(self, layer):
        m = self.mask.get(layer)
        if m is None:
            m = torch.zeros(self.dim if hasattr(self, "dim") else 512,
                            dtype=torch.bool, device=self.device)
        return m

    def sync(self, sigmas_np, ring_ws_t, ring_b_t, dim):
        """sigmas_np: (L,D) numpy (small, from decay). ring_ws_t: list of
        (B,K,D) torch tensors, ring_b_t: (B,K) torch long. Rebuilds E for
        layers whose hot set changed; keeps existing columns' EMAs untouched.
        Called every REBASE_EVERY steps (and at creation), NEVER per step: a
        per-step sync would copy the whole ring to CPU on GPU (~0.5 GB/step)."""
        self.dim = dim
        self.mask = {l: torch.from_numpy(sigmas_np[l] ** self.ring_k >
                                         self.hot_frac).to(self.device)
                     for l in range(sigmas_np.shape[0])}
        L = sigmas_np.shape[0]
        for l in range(L):
            new = np.nonzero(self.hot_from_np(sigmas_np[l], self.ring_k,
                                              self.hot_frac))[0]
            old = (self.cols.get(l, torch.empty(0, dtype=torch.long,
                                                device=self.device)).cpu().numpy()
                   if l in self.cols else np.array([], int))
            if len(new) == 0:
                self.cols.pop(l, None)
                self.E.pop(l, None)
                continue
            keep = np.intersect1d(old, new)
            adopt = np.setdiff1d(new, old)
            H = len(new)
            # Cap check BEFORE allocation: a 34 GB request must fail with this
            # actionable message, not an opaque CUDA OOM from torch.zeros. In
            # factored mode the (B,H) factor is tiny, but include it exactly.
            if self.cap_bytes is not None:
                other = sum(int(e.numel()) * 4 for k, e in self.E.items()
                            if k != l)
                old_current = (int(self.E[l].numel()) * 4
                               if l in self.E else 0)
                # The old tensor remains live while the replacement allocates.
                projected = other + old_current + self.B * 256 * H * 4
                if projected > self.cap_bytes:
                    raise RuntimeError(
                        f"tail-credit memory would be "
                        f"{projected/1e9:.3f} GB ({len(new)} hot dims in "
                        f"layer {l}, B={self.B}), over the "
                        f"{self.cap_bytes/1e9:.3f} GB cap. Lower --batch, or "
                        f"TMT_TAIL=0 (ring-only credit), or raise "
                        f"TMT_TAIL_CAP_GB deliberately.")
            E = torch.zeros((self.B, 256, H), dtype=F, device=self.device)
            if len(keep):
                pos = np.array([int(np.searchsorted(old, k)) for k in keep])
                E[:, :, :len(keep)] = self.E[l][:, :, pos]
            if len(adopt):
                h0 = len(keep)
                adopt_t = torch.from_numpy(adopt.astype(np.int64)).to(self.device)
                w = ring_ws_t[l][:, :, adopt_t]            # (B,K,H1) device
                H1 = len(adopt)
                for k in range(min(self.ring_k, w.shape[1])):
                    idx_b = ring_b_t[:, k]                 # (B,) device long
                    for j in range(H1):
                        E[:, :, h0 + j].scatter_add_(
                            1, idx_b[:, None], w[:, k, j:j + 1])
            self.cols[l] = torch.from_numpy(new.astype(np.int64)).to(self.device)
            self.E[l] = E

    def begin_step(self, layer, sig_hot):
        E = self.E.get(layer)
        if E is not None and sig_hot.numel():
            E.mul_(sig_hot[None, None, :])

    def credit(self, layer, dlds):
        """dlds (B,D) -> (256,H) fp32 (decayed E, lags>=1 exact)."""
        E = self.E.get(layer)
        if E is None:
            return None
        cols = self.cols[layer]
        g = dlds[:, cols]                                # (B,H)
        return torch.einsum("bvh,bh->vh", E, g)

    def add_byte(self, layer, c_arr):
        E = self.E.get(layer)
        if E is not None:
            B = c_arr.shape[0]
            idx = c_arr[:, None, None].expand(B, 1, E.shape[2])
            src = torch.ones((B, 1, E.shape[2]), dtype=F, device=self.device)
            E.scatter_add_(1, idx, src)


class TorchLayer:
    def __init__(self, dim):
        self.dim = dim
        # P3 gated recurrence. `gate` is a per-dim vector of LOGITS added to the
        # decay logit, so a = sigmoid(decay + gate * x_in). It is created only
        # when the arm is gated (GpuModel sets self.gate/self.gated): leaving it
        # absent in the default engine keeps theta's layout -- and therefore
        # check_parity's numpy==torch identity -- untouched.
        self.gate = None
        self.gated = False

    def forward(self, enc, x_in, states):
        decay = _sig(self.decay)
        if self.gated:
            # a = sigmoid(decay + gate * x_in).
            #
            # gate init 0 makes this EXACTLY the incumbent: 0.0 * x_in is an
            # exact zero and decay + 0.0 == decay, so a == decay bit-for-bit.
            # That is what lets an incumbent checkpoint be resumed into the
            # gated arm as its own control (the checkpoint carries no
            # layer{i}.gate key, so load leaves the gate at 0).
            #
            # NOTE the form: s = a*s + x, NOT s = a*s + (1-a)*x. The latter is
            # what the plan wrote, but (1-a) is not 1.0, so that form cannot
            # reproduce the incumbent even at gate=0 and the "bit-identical
            # starting point / resumable A/B" property the same paragraph asks
            # for is impossible. Retention is gated; the write stays weight 1.0.
            a = _sig(self.decay + self.gate * x_in)
        else:
            a = decay
        state = states * a + enc
        h, cache = _layer_norm(state, self.g, self.b)
        z = h @ self.W.t()
        sg = torch.sigmoid(z)
        y = z * sg
        # x_in is returned so the caller can form dL/dgate without re-deriving
        # the residual stream: the gate reaches a_t through gate * x_in, so
        # x_in is a factor of the gate's cross-time trace (see
        # gate_trace_update), not just of the local term.
        return x_in + y, state, a, (z, sg, h, cache, x_in)

    def backward(self, dx_out, cache):
        # cache is (z, sg, h, ln_cache, x_in); x_in is unused by the backward
        # proper -- the gate gradient is formed in step_batch, where the
        # per-timestep recurrence lives -- but it is unpacked so the tuple
        # shape stays honest about what forward produced.
        z, sg, h, ln_cache, _x_in = cache
        dz = dx_out * sg * (1.0 + z * (1.0 - sg))
        dW = dz.t() @ h
        dh = dz @ self.W
        dstate, dg, db = _layer_norm_backward(dh, ln_cache)
        return dstate, dW, dg, db


# ---------------------------------------------------------------------------
# The P3 gate gradient.
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
# trace is what rebuilds the chain the engine does not have. (Feeding this
# the TOTAL state gradient instead would double count the chain: the trace's
# chain and the total gradient's chain are the same chain. The four-way truth
# table is measured in tools/gate_grad_probe.py and pinned by
# tests/check_gate_grad.py.)
#
# These two functions are the engine's OWN implementation of the formula:
# step_batch calls them, and the test calls the same two, so the test pins
# the shipped code and not a re-derivation of it.
# ---------------------------------------------------------------------------
def gate_trace_update(gatetrace, a, x_in, s_prev):
    """One step of the gate's cross-time trace.

    `a` is the EFFECTIVE retention a_t. It is per-SAMPLE (B, dim) in the
    gated arm -- the gate makes a depend on the input, so it is not a single
    vector any more -- and per-dim (dim,) in the ungated arm. The trace
    itself is per-sample either way, so a per-dim `a` is broadcast over the
    batch here and a per-sample `a` is used as it is. Getting this wrong is
    silent: a (B, dim) `a` broadcast against a (1, dim) view produces
    (1, B, dim), and the batch axis then survives the sum that is supposed
    to remove it.
    """
    if a.dim() == 1:
        a = a[None, :]
    return a * gatetrace + a * (1.0 - a) * x_in * s_prev


def gate_gradient(dlds, gatetrace, a, x_in, s_prev):
    """The gate gradient for ONE timestep of ONE layer.

    dlds is the readout-only state gradient dL/ds_t, (B, dim); x_in is the
    layer's input at this timestep and s_prev is the state BEFORE it, both
    (B, dim). Returns (contribution, new_trace): the contribution is the
    (dim,) gate gradient for this timestep -- SUMMED OVER THE BATCH, because
    the gate parameter is per-dim and the batch is the other axis -- and the
    new trace, which the next timestep (or the next window) continues from.
    """
    gt = gate_trace_update(gatetrace, a, x_in, s_prev)
    return (dlds * gt).sum(dim=0), gt


def _ring_credit_compiled(dest, w, dlds, hot_mask, rows):
    """Inductor candidate for A2; compilation is warmed outside the timing."""
    vals = w * dlds[:, None, :]
    if hot_mask is not None:
        vals.masked_fill_(hot_mask[None, None, :], 0.0)
    dest.index_add_(0, rows, vals.reshape(-1, w.shape[-1]))
    return dest


# NVRTC ring credit -- the path for GPUs that Triton cannot serve.
#
# `compiled` and `triton` both die on pre-Volta devices: Triton requires
# compute capability >= 7.0, and torch._inductor raises GPUTooOldForTriton on
# anything older (measured on an MX150, cc 6.1, where the shipped `auto`
# default therefore CRASHED rather than falling back). NVRTC has no such floor,
# so this path works on every CUDA device torch itself runs on, with no nvcc,
# no ninja and no build step -- torch.cuda._compile_kernel JITs from source at
# first use (~1 s, cached in-process).
#
# Why fusing is worth it. `full` issues three ops per layer over (B,K,D):
#     w.mul_(dec)          read+write
#     w * dlds   -> vals   read+write
#     index_add_(rows,vals) read + atomic scatter
# i.e. five passes over the ring, of which two are pure re-reads of the same
# data. Measured at B=1024, K=14, D=512 (29 MiB ring) on an MX150, per layer:
#     full   5.60 ms (no hot mask) / 6.83 ms (with)   [39.1 GiB/s = peak copy]
#     nvrtc  2.55 ms / 2.53 ms                         -> 2.2x / 2.7x
# The fused kernel does one pass: read w, multiply by the decay, write the
# decayed value back, multiply by the gradient, atomically add to dest.
#
# Parity (measured, tests/ is the durable gate; see RESULTS.md):
#   * the ring buffer is BIT-IDENTICAL (max|d| = 0.0) -- decay and credit are
#     per-element and the arithmetic order is unchanged;
#   * dembed differs by 5.9e-07 of RMS, which is SMALLER than the `full`
#     path's own run-to-run difference (8.8e-07): index_add_ is an atomic
#     scatter whose summation order was never deterministic, so exact equality
#     was never a property of the reference to begin with.
_NVRTC_RING_SRC = r"""
extern "C" __global__
void tmt_ring_credit(float* __restrict__ w,
                     const float* __restrict__ dec,
                     const float* __restrict__ dlds,
                     const long* __restrict__ rows,
                     float* __restrict__ dest,
                     const unsigned char* __restrict__ hotmask,
                     int B, int K, int D, int has_mask) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)B * K * D) return;
    int d = (int)(idx % D);
    long bk = idx / D;
    int b = (int)(bk / K);
    // dec is (D,), NOT (B,D): the layer decay is a per-dimension sigmoid
    // broadcast over the batch (see TorchLayer.forward / w.mul_(dec[None,None,:])).
    float wv = w[idx] * dec[d];
    w[idx] = wv;
    float v = wv * dlds[(long)b * D + d];       // ring credit
    if (has_mask && hotmask[d]) v = 0.0f;       // hot-dim mask (bool, 1 byte)
    atomicAdd(&dest[rows[bk] * D + d], v);
}
"""

_NVRTC_RING_KERNEL = None
_NVRTC_RING_FAILED = False


def _ensure_cuda_home():
    """torch.cuda._compile_kernel insists on resolving CUDA include paths,
    which raises OSError when CUDA_HOME is unset. Our kernel includes no CUDA
    headers, so a CUDA_HOME is synthesized from the header-only layout the
    `nvidia-*-cu12` pip wheels ship, which is how a torch CUDA build without a
    system toolkit can still JIT. Returns True if CUDA_HOME is resolvable.
    """
    if os.environ.get("CUDA_HOME"):
        return True
    import glob as _glob
    import tempfile
    import site
    roots = []
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        roots.append(base)
    inc = None
    for base in roots:
        hit = _glob.glob(os.path.join(base, "nvidia", "cuda_runtime", "include",
                                     "cuda_runtime.h"))
        if hit:
            inc = os.path.dirname(hit[0])
            break
    if inc is None:
        return False
    home = tempfile.mkdtemp(prefix="tmt-cuda-home-")
    dst = os.path.join(home, "include")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(inc):
        try:
            os.symlink(os.path.join(inc, f), os.path.join(dst, f))
        except OSError:
            pass
    os.environ["CUDA_HOME"] = home
    return True


def _nvrtc_ring_kernel():
    """Compile (once) and return the fused ring kernel, or None if unavailable.

    None is a normal outcome, not an error: torch.cuda._compile_kernel insists
    on resolving CUDA include paths, which raises when CUDA_HOME is unset, and
    any NVRTC failure must degrade to `full` rather than kill a run.
    """
    global _NVRTC_RING_KERNEL, _NVRTC_RING_FAILED
    if _NVRTC_RING_KERNEL is not None or _NVRTC_RING_FAILED:
        return _NVRTC_RING_KERNEL
    try:
        if not hasattr(torch.cuda, "_compile_kernel"):
            raise RuntimeError("torch.cuda._compile_kernel is unavailable")
        if not _ensure_cuda_home():
            raise RuntimeError("no CUDA_HOME and no bundled cuda_runtime headers")
        _NVRTC_RING_KERNEL = torch.cuda._compile_kernel(
            _NVRTC_RING_SRC, "tmt_ring_credit")
    except Exception as exc:                     # pragma: no cover
        _NVRTC_RING_FAILED = True
        print(f"[ring_credit] nvrtc unavailable ({type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:90]}) -> falling back to 'full'",
              flush=True)
        return None
    return _NVRTC_RING_KERNEL


if triton is not None:
    @triton.jit
    def _fused_ring_credit_kernel(dest_ptr, w_ptr, grad_ptr, rows_ptr, factor_ptr,
                                  hot_ptr, B, K, D, HAS_FACTOR: tl.constexpr,
                                  HAS_MASK: tl.constexpr, BLOCK_D: tl.constexpr):
        slot = tl.program_id(0)
        b = slot // K
        k = slot % K
        cols = tl.arange(0, BLOCK_D)
        valid = cols < D
        wp = w_ptr + (b * K + k) * D + cols
        value = tl.load(wp, mask=valid, other=0.0)
        value *= tl.load(grad_ptr + b * D + cols, mask=valid, other=0.0)
        if HAS_FACTOR:
            # factor is (B,K,D): the decay is per (stream, slot, dim), so the
            # load must be strided by D like w. Loading a bare scalar here
            # (factor[b,k,0]) scales the whole row by one dim's decay, which
            # measured 11x RMS state error.
            value *= tl.load(factor_ptr + (b * K + k) * D + cols,
                             mask=valid, other=0.0)
        if HAS_MASK:
            value = tl.where(tl.load(hot_ptr + cols, mask=valid, other=1), 0.0,
                             value)
        byte = tl.load(rows_ptr + slot)
        tl.atomic_add(dest_ptr + byte * D + cols, value, mask=valid)


class GpuModel:
    """Flat-parameter torch model, 1:1 with numpy_port.Model's params/buffers."""

    def __init__(self, dim=512, layers=16, temp=0.75, lr=5e-4, lr_decay=None,
                 decay_init=None, half_life_init=None,
                 device=None, ring_credit="auto", sync_metrics=True,
                 grad_assembly="cat", seed=0, ring_k=None, distill_alpha=0.0,
                 ce_only=False, seqlen=1, gated=False, trace_mode="ring",
                 grad_clip=0.0, nan_guard=True, wd_decay=0.01,
                 fsp_alpha=0.0, fsp_r=16, fsp_tau=64, fsp_pos_weight=50.0,
                 fsp_layer=-1):
        # trace_mode: how per-byte embedding credit is accumulated.
        #   "ring"  -- the incumbent: a K-lag ring buffer plus tail_credit's
        #              private EMA for the "hot" dims where truncation would
        #              cost more than 1% of the credit mass.
        #   "dense" -- the exact (B, 256, dim) eligibility trace per layer,
        #              which is the tail with ALL dims hot. No ring, no tail, no
        #              membership logic, no ring_k inference, and therefore no
        #              ring_k<=256 cap -- which is what currently clips P4's
        #              long half-lives at ~32 B.
        # "ring" stays the default so every existing arm and checkpoint is
        # untouched; "dense" is the P2 arm and cannot load a ring checkpoint.
        if trace_mode not in ("ring", "dense"):
            raise ValueError(f"trace_mode must be 'ring' or 'dense', got "
                             f"{trace_mode!r}")
        self.trace_mode = trace_mode
        self.dense_E = None
        # gated (P3): the per-dim gate on the recurrence, a = sigmoid(decay +
        # gate * x). OFF by default and, when off, the parameter does not exist
        # at all -- so theta's layout is unchanged and check_parity's
        # numpy==torch identity still holds exactly. On, the gate starts at 0,
        # which makes a == sigmoid(decay) bit-for-bit and lets an incumbent
        # checkpoint be resumed into the gated arm as its own control.
        self.gated = bool(gated)
        # seqlen (T): bytes per stream per optimizer step. 1 is the historical
        # behaviour and MUST stay the default -- every existing caller and every
        # checkpoint describes a T=1 run, and T=1 is bit-identical to the
        # pre-T-axis engine (tests/check_parity.py pins that to the bit).
        # Raising it does NOT change B: state, ring and tail are all B-shaped, so
        # the memory is unchanged and only the work per update grows.
        if int(seqlen) < 1:
            raise ValueError(f"seqlen must be >= 1, got {seqlen!r}")
        self.seqlen = int(seqlen)
        self.dim = dim
        self.layercount = layers
        self.temp = temp
        self.lr = float(lr)
        # Distillation (PLAN-2026-10-03-DISTILL.md). alpha = 0 is the default and
        # MUST reproduce the pre-KD engine bit-for-bit (gate 3), so the KD path
        # is skipped entirely rather than multiplied by zero.
        self.distill_alpha = float(distill_alpha)
        # CE-only switch (PLAN-2026-10-06 MEMORY-BENCHMARKS priority 3):
        # L = CE only. The variance-hinge, embed-MSE/JEPA and end-of-text
        # terms are masked to EXACTLY zero in step_batch; KD is incompatible
        # and refusing loudly beats silently disabling (gates in
        # tests/check_ce_only.py).
        self.ce_only = bool(ce_only)
        # ---- SAFETY (PLAN-2026-10-11 P0 item 3) --------------------------
        # `nan_guard`: if the assembled gradient contains a non-finite value,
        # REFUSE the update instead of poisoning the weights. Our own KD arm
        # diverged to 860.68; the CUDA fork refuses a non-finite gradient
        # outright (CHANGES.md "NaN guard"). A refusal leaves theta, m and v
        # untouched and counts itself; `t` still advances, because the DATA
        # was consumed and the schedule must not slip (an update that did not
        # happen should not also rewind the learning-rate schedule).
        #
        # The guard costs one reduction and one host sync per step, and it
        # changes NO arithmetic when the gradient is finite -- that is what
        # tests/check_safety.py gates, bit-exactly, against the same engine
        # with the guard off.
        self.nan_guard = bool(nan_guard)
        self.nan_refusals = 0
        self.last_nan_refused = False
        # `grad_clip`: GLOBAL L2 clip on the assembled gradient, in absolute
        # gradient units. 0 disables it (the default, so every existing arm
        # and checkpoint keeps its exact dynamics).
        #
        # NOTE the units, because they interact with the T axis: the window's
        # gradient is a SUM over T bytes, and the gradient is also a sum over
        # B streams, so an ABSOLUTE threshold means a T=16 arm reaches it
        # 16x sooner than a T=1 arm at the same B. A clip threshold is
        # therefore not a comparable quantity across arms -- it is a safety
        # net, not a knob, and it must be off (or scaled) for a T A/B.
        self.grad_clip = float(grad_clip)
        # `wd_decay`: the decoupled weight decay applied to the `.decay` and
        # `.gate` groups. The default equals the flat `self.wd`, which makes
        # the whole vector uniform and therefore bit-identical to the
        # incumbent's scalar `wd * th` (PLAN-2026-10-11 P6).
        #
        # Why P6 wants it at 0: `sigma(0) = 0.5` is exactly a 1-BYTE half-life,
        # so a weight decay on the decay logits is a systematic pull toward NO
        # MEMORY, anchored at init. It fights the log-spaced half-life init
        # (P4) directly: the init that gives the model a 512 B channel is
        # also the thing weight decay is pulling back to 1 B. The two are
        # therefore a 2x2, not a sequence: measure (wd_decay 0.01 / 0.0) x
        # (default init / P4 init).
        self.wd_decay = float(wd_decay)
        self._wd_vec = None
        # ---- FSP: the Future State Prediction auxiliary loss ---------------
        #
        # From FlashLM (Chang, May 2026), §3.1: at every r-th position the
        # model predicts WHICH tokens appear in the next tau positions, as a
        # binary multi-hot, with BCE (pos_weight 50); the total loss is
        # L_CE + alpha * L_FSP with alpha = 0.1.
        #
        # WHY IT IS THE RIGHT IDEA FOR THIS MODEL. A recurrent byte model has
        # exactly ONE place where context can live: the state. Next-byte CE
        # only ever asks the OUTPUT (one step ahead) to be right, so it never
        # asks the STATE to be USABLE later -- and the measured result is
        # exactly that: the state can HOLD a signal 128 B out (5.5% of the
        # control floor with the P4 init) while every copy/retrieval probe
        # stays at chance. FSP puts a read-out on the state and asks it to
        # predict the future's bag-of-bytes, so the state is pushed to carry
        # forward-looking content in a linearly readable form.
        #
        # The head reads the LAST layer's state (the one closest to the
        # output) through that layer's own LayerNorm, so no extra
        # normalisation parameters are introduced; the projection itself is
        # new (dim -> 256 bytes), and the parameters live in theta ONLY when
        # alpha > 0, so the default engine's layout stays byte-comparable
        # with numpy_port's for check_parity.
        self.fsp_alpha = float(fsp_alpha)
        self.fsp_r = int(fsp_r)
        self.fsp_tau = int(fsp_tau)
        self.fsp_pos_weight = float(fsp_pos_weight)
        self.fsp_layer = int(fsp_layer)
        if self.fsp_alpha < 0:
            raise ValueError(f"fsp_alpha must be >= 0, got {self.fsp_alpha}")
        self.last_fsp = 0.0
        self.fsp_positions = 0
        # (file index, byte offset within the file) -- the DATA STREAM's
        # position, set by the trainer and persisted in the checkpoint so a
        # resume continues the corpus instead of replaying it. None = "no
        # position known" (a fresh run).
        self.data_pos = None
        if self.ce_only and self.distill_alpha > 0.0:
            raise ValueError(
                f"ce_only=True requires distill_alpha == 0, got "
                f"{self.distill_alpha}: a CE-only run strips KD by design "
                f"(PLAN-2026-10-06 MEMORY-BENCHMARKS priority 3). Set "
                f"--distill-alpha 0.")
        self._target_fn = None
        self.kd_positions = 0
        self.kd_covered = 0
        self.last_kd = 0.0
        # Optional per-ROLE learning rate. `lr_decay` applies ONLY to the 16
        # per-layer `.decay` slices (512 coords each = 8,192 total = 0.183% of
        # the 4,481,793 parameters); every other coordinate keeps `lr`. This is
        # the role-specific-LR axis of the planned factorial, kept separate from
        # the optimizer axis because its cost is completely different: this one
        # is RESUMABLE (the AdamW state format is untouched), so arms can branch
        # off the incumbent in one leg, whereas AdamW->SGD changes the stored
        # state and forces from-scratch arms with their own LR grid.
        # None means "one LR everywhere", the historical behaviour.
        self.lr_decay = None if lr_decay is None else float(lr_decay)
        self._lr_vec = None
        self.b1, self.b2, self.eps, self.wd = 0.9, 0.999, 1e-8, 0.01
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._init_seed = int(seed)
        self._ring_k_explicit = ring_k is not None
        if self._ring_k_explicit:
            if not isinstance(ring_k, (int, np.integer)) or int(ring_k) < 1 or int(ring_k) > 256:
                raise ValueError(f"ring_k must be an integer in [1, 256], got {ring_k!r}")
        if ring_credit not in ("auto", "full", "compiled", "triton", "nvrtc"):
            raise ValueError(f"unknown ring_credit={ring_credit!r}")
        if grad_assembly not in ("cat", "loop"):
            raise ValueError(f"unknown grad_assembly={grad_assembly!r}")
        self.grad_assembly = grad_assembly
        # ring_credit: how the ring credit is applied.
        #   full     -- materialize (B,K,D) then index_add_ (portable, CPU-safe)
        #   compiled -- Inductor fuses multiply + mask + scatter: measured +27%
        #               on T4 vs full, with no recompile stall at a tail rebase
        #   triton   -- hand-written fused kernel, +9% on T4; the fallback when
        #               Inductor is unavailable
        #   auto     -- compiled on CUDA, full elsewhere (safe default)
        self.ring_credit = ring_credit
        if self.ring_credit == "auto":
            self.ring_credit = ("compiled"
                                if str(device).startswith("cuda") else "full")
        self._compiled_ring_credit = None
        self._compile_failed = False
        self._ring_resolved = None
        self.sync_metrics = bool(sync_metrics)
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False  # fp32 only
        self.t = 0
        self.bytes_seen = 0
        # same documented switch as numpy_port: TMT_TAIL=0 disables the exact
        # long-lag credit entirely (ring-only embed credit, as pre-tail).
        self.tail_enabled = os.environ.get("TMT_TAIL", "1") != "0"
        # Tail memory = sum_layers B*256*H*4 bytes. At B=4096 with 512 hot
        # dims/layer that is ~34 GB: guard before allocating (PLAN risk list:
        # "hot-dim blowup: abort if tail > cap"). Override with TMT_TAIL_CAP_GB.
        self.tail_cap_gb = float(os.environ.get("TMT_TAIL_CAP_GB", "4.0"))
        with torch.no_grad():
            lim = 1.0 / np.sqrt(dim)
            # ring_k must cover the LONGEST-lived dim. decay_init may be a
            # per-dim array (the log-spaced half-life init), so reduce over
            # dims instead of calling .item() -- an array used to raise
            # "only one element tensors can be converted to Python scalars"
            # right here, at construction, before any training.
            _src = half_life_init if half_life_init is not None else decay_init
            if _src is None:
                dmax = float(_sig(torch.tensor(lim, dtype=torch.float32)).item())
            elif isinstance(_src, (tuple, list)):
                dmax = float(0.5 ** (1.0 / float(_src[1])))
            else:
                _s = _sig(torch.as_tensor(
                    np.asarray(_src, dtype=np.float32)))
                dmax = float(_s.max().item()) if _s.ndim else float(_s.item())
            inferred_ring_k = int(min(256, np.ceil(
                np.log(RING_TOL) / np.log(max(dmax, 1e-3)))))
            self.ring_k = int(ring_k) if self._ring_k_explicit else inferred_ring_k
        self._build(dim, layers, decay_init, half_life_init, gated)
        self.gated = bool(gated)
        if self.gated:
            for i, l in enumerate(self.layers):
                l.gated = True
                l.gate = self.views[f"layer{i}.gate"]

    def set_target_fn(self, fn):
        """Install ``step -> (B,256) target on device`` (or None to disable).

        The provider owns the dump; the engine only consumes a distribution.
        Positions MI could not resolve arrive as all-zero and contribute no KD
        gradient — the CE-only fallback, counted in kd_stats, never averaged
        away.
        """
        self._target_fn = fn
        return self

    def kd_stats(self):
        return {"alpha": self.distill_alpha, "positions": self.kd_positions,
                "covered": self.kd_covered,
                "coverage": (self.kd_covered / self.kd_positions
                             if self.kd_positions else None)}

    def _build(self, dim, layers, decay_init, half_life_init=None,
                gated=False):
        lim = 1.0 / np.sqrt(dim)
        # Seeded init. Seed 0 is the historical default and is what every
        # from-scratch A/B arm must share, so their starting weights are
        # bit-identical and the data is the only axis that differs.
        rng = np.random.default_rng(self._init_seed)
        init = {
            "embed": (rng.standard_normal((256, dim)) * 0.1).astype(np.float32),
            "Wd": rng.uniform(-lim, lim, (256, dim)).astype(np.float32),
            "bd": np.zeros(256, np.float32),
            "Ws": rng.uniform(-lim, lim, (dim,)).astype(np.float32),
            "bs": np.zeros(1, np.float32),
        }
        for i in range(layers):
            init[f"layer{i}.W"] = rng.uniform(-lim, lim, (dim, dim)).astype(np.float32)
            init[f"layer{i}.g"] = np.ones(dim, np.float32)
            init[f"layer{i}.b"] = np.zeros(dim, np.float32)
            d0 = rng.uniform(-lim, lim, (dim,)).astype(np.float32)
            if half_life_init is not None:
                # P4: log-spaced half-life prior. half-life h means
                # sigma = 0.5**(1/h), so logit = ln(sigma/(1-sigma)).
                #
                # The incumbent uniform init puts every sigma within 1% of 0.5
                # (|logit| <= 1/sqrt(dim) = 0.044), i.e. EVERY half-life within
                # a byte of 1 B. The trained checkpoint confirms it has stayed
                # there: sigma_max = 0.9847 -> longest dim forgets in ~45 B, and
                # only 8 of 512 dims hold more than 8 B. So the model has no
                # capacity at all for the retention we want, and no amount of
                # training can build one it was never given a dim for.
                #
                # The SET of half-lives is geometric; the assignment to dims is
                # shuffled by the seeded rng so dim index stays meaningless.
                # Leaving it ordered would make dim 0..dim-1 monotonically
                # longer-lived in every layer, coupling a coordinate to a
                # timescale and handing the model a structure it does not have
                # under the incumbent init.
                hmin, hmax = (float(half_life_init[0]), float(half_life_init[1]))
                if not (0 < hmin <= hmax):
                    raise ValueError(
                        f"half_life_init needs 0 < half_min <= half_max, got "
                        f"{(hmin, hmax)}")
                h = np.geomspace(hmin, hmax, dim)
                sig = np.power(0.5, 1.0 / h)
                logits = np.log(sig / (1.0 - sig)).astype(np.float32)
                rng.shuffle(logits)
                d0[:] = logits
            elif decay_init is not None:
                d0[:] = decay_init
            init[f"layer{i}.decay"] = d0
            if gated:
                # P3 gate logits, ZERO so a == sigmoid(decay) at init. Added to
                # `init` (and therefore to self.shapes and theta) only in the
                # gated arm: the default engine's theta layout must stay
                # byte-comparable with numpy_port's for check_parity.
                init[f"layer{i}.gate"] = np.zeros(dim, np.float32)

        if self.fsp_alpha > 0.0:
            # The FSP head. Appended LAST so the default layout is untouched,
            # and it exists only when the loss is on (same rule as the gate).
            # Init: small random projection (like Wd) and a zero bias, so the
            # head starts as a near-uniform predictor and its first gradient
            # is informative rather than saturated.
            #
            # fsp.g / fsp.b0 are the head's OWN LayerNorm parameters. Borrowing
            # the layer's would make the auxiliary loss a knob on the local
            # read-out: the FSP gradient would move layer{g,b}, and the arm's
            # held-out CE could move for a reason that has nothing to do with
            # memory. With its own normalisation the head can only affect the
            # state, which is the thing being tested.
            init["fsp.g"] = np.ones(dim, np.float32)
            init["fsp.b0"] = np.zeros(dim, np.float32)
            init["fsp.W"] = rng.uniform(-lim, lim, (dim, 256)).astype(np.float32)
            init["fsp.b"] = np.zeros(256, np.float32)

        self.shapes = [(k, v.shape) for k, v in init.items()]
        n = sum(int(np.prod(s)) for _, s in self.shapes)
        self.theta = torch.from_numpy(np.concatenate(
            [init[k].ravel() for k, _ in self.shapes]).astype(np.float32)
        ).to(self.device)
        self.m = torch.zeros(n, dtype=F, device=self.device)
        self.v = torch.zeros(n, dtype=F, device=self.device)
        self.grads = torch.zeros(n, dtype=F, device=self.device)

        self.views = {}
        self.gviews = {}
        off = 0
        for name, shape in self.shapes:
            size = int(np.prod(shape))
            self.views[name] = self.theta[off:off + size].view(shape)
            self.gviews[name] = self.grads[off:off + size].view(shape)
            off += size
        self.embed = self.views["embed"]
        self.Wd, self.bd = self.views["Wd"], self.views["bd"]
        self.Ws, self.bs = self.views["Ws"], self.views["bs"]
        self.fsp_W = self.views["fsp.W"] if "fsp.W" in self.views else None
        self.fsp_b = self.views["fsp.b"] if "fsp.b" in self.views else None
        self.fsp_g = self.views["fsp.g"] if "fsp.g" in self.views else None
        self.fsp_b0 = self.views["fsp.b0"] if "fsp.b0" in self.views else None
        if self.fsp_alpha > 0.0:
            li = self.fsp_layer % max(1, layers)
            self.fsp_layer = li
        self.layers = [TorchLayer(dim) for _ in range(layers)]
        for i, l in enumerate(self.layers):
            l.W = self.views[f"layer{i}.W"]
            l.g = self.views[f"layer{i}.g"]
            l.b = self.views[f"layer{i}.b"]
            l.decay = self.views[f"layer{i}.decay"]
            if gated:
                l.gated = True
                l.gate = self.views[f"layer{i}.gate"]
        self.B = 1
        self.new_states(1)
        self.tail = None
        self.tail_state = {}
        self._tail_rebase_at = -1

    def new_states(self, B):
        self.B = B
        # Replacing ring/states invalidates any tail built against the old
        # buffers: drop it so _ensure_tail rebuilds and _tail_sync re-seeds.
        # (Without this, a B change or a checkpoint ring_k change could keep a
        # fresh rebase marker and leave the new tail unseeded with 0 hot dims.)
        if hasattr(self, "tail"):
            self.tail = None
            self.tail_state = {}
            self._tail_rebase_at = -1
        self.ring_b = torch.zeros((B, self.ring_k), dtype=torch.long,
                                  device=self.device)
        # The dense trace REPLACES the ring, so the ring's per-layer buffer
        # and its scratch are allocated EMPTY on that path. They are never
        # read there (the credit block takes the dense branch, and the tail's
        # rebase -- the only other reader -- is skipped), and at ring_k=256
        # they are the same size as the dense trace itself: 2.15 GB at B=256,
        # 8.6 GB at B=1024. Allocating both is how the P4 arm OOM'd on a 4 GB
        # card while the dense trace was supposed to be the cheaper of the two.
        # `states` and `decaytrace` are needed in BOTH modes -- they are the
        # recurrence and its gradient carrier, not credit.
        dense = (self.trace_mode == "dense")
        self.ring_vals = torch.zeros((B, 0 if dense else self.ring_k,
                                      self.dim), dtype=F, device=self.device)
        for l in self.layers:
            l.states = torch.zeros((B, self.dim), dtype=F, device=self.device)
            l.decaytrace = torch.zeros((B, self.dim), dtype=F, device=self.device)
            # The gate's cross-time trace, allocated ONLY in the gated arm:
            # the default engine must not pay the buffer, and must not grow
            # an extra live tensor (the T=1 bit-identity gate snapshots every
            # live buffer, so a new one would have to be pinned there too).
            # It lives next to the decaytrace because it is the same kind of
            # object: the carrier of the chain the engine's per-byte backward
            # does not have.
            if getattr(l, "gated", False):
                l.gatetrace = torch.zeros((B, self.dim), dtype=F,
                                          device=self.device)
            l.ring_w = torch.zeros((B, 0 if dense else self.ring_k,
                                    self.dim), dtype=F, device=self.device)
        if self.trace_mode == "dense":
            # P2: the exact embedding trace, (B, 256, dim) per layer.
            #
            # This is the tail with ALL dims hot. tail_credit.py already
            # maintains exactly this object for the dims whose
            # sigma**ring_k > 1% -- the ring exists to avoid materialising
            # (B,256,dim) for every layer, and the tail patches the dims where
            # that truncation would cost more than 1% of the credit mass.
            # Materialising it for every dim is therefore not a new mechanism,
            # it is the approximation removed.
            #
            # Cost: B*256*dim*4 bytes per layer. At dim=512, 16 layers:
            #   B=16 -> 134 MB, B=64 -> 537 MB, B=256 -> 2.1 GB.
            # That is the whole trade, and it buys away the ring's hardcoded
            # ring_k<=256 cap -- which is what currently clips P4's long
            # half-lives at ~32 B -- and the tail's membership/rebasing logic.
            self.dense_E = [torch.zeros((B, 256, self.dim), dtype=F,
                                        device=self.device)
                            for _ in self.layers]
        elif getattr(self, "dense_E", None) is not None:
            self.dense_E = None

    # ---------------- tail credit --------------------
    def _ensure_tail(self):
        if not self.tail_enabled:
            return
        if self.trace_mode == "dense":
            # The dense trace REPLACES the tail: it is the tail with every dim
            # hot, so the tail's accumulator would be allocated and then never
            # read. That is not free: with the P4 init's 239 hot dims per
            # layer, the tail's (B, 256, H) accumulator is 1.0 GB at B=256 --
            # a third of a 4 GB card, spent on a buffer the dense path never
            # touches. (Found by an OOM while running the P4 arms: the RING
            # at ring_k=256 is 8.6 GB at B=1024, which is the same bill the
            # plan priced for the dense trace, and the plan's P4 arm was
            # going to be run at B=1024.)
            self.tail = None
            return
        if self.tail is None or self.tail.B != self.B:
            self.tail = TorchTail(self.B, self.ring_k, hot_frac=HOT_FRAC,
                                  device=self.device, cap_gb=self.tail_cap_gb)
            if self.tail_state:
                self._tail_restore_from(self.tail_state)
                self.tail_state = {}
        # Membership re-evaluated on the documented REBASE_EVERY cadence (and
        # once at creation), NOT per step: a per-step sync copies the ring
        # (~0.5 GB/step at B=1024,K=18) and was 99% of the GPU step time.
        if self._tail_rebase_at < 0 or (self.t - self._tail_rebase_at) >= REBASE_EVERY:
            self._tail_sync()

    def _tail_sync(self):
        if self.tail is None:
            return
        # Only the small decay sigmas cross to CPU for the membership test;
        # the ring stays on-device and is used for seeding in torch.
        sigs = torch.stack([_sig(l.decay) for l in self.layers]
                           ).detach().cpu().numpy().astype(np.float32)
        self.tail.sync(sigs, [l.ring_w for l in self.layers],
                       self.ring_b, self.dim)
        nhot = sum(int(c.numel()) for c in self.tail.cols.values())
        e_bytes = sum(int(E.numel()) * 4 for E in self.tail.E.values())
        if e_bytes > self.tail_cap_gb * 1e9:
            raise RuntimeError(
                f"tail-credit memory {e_bytes/1e9:.1f} GB exceeds cap "
                f"{self.tail_cap_gb:.1f} GB ({nhot} hot dims at B={self.B}). "
                f"Lower --batch, or TMT_TAIL=0 (ring-only credit), or raise "
                f"TMT_TAIL_CAP_GB deliberately.")
        if nhot > 256:
            print(f"[tail] warning: {nhot} hot dims; accumulator "
                  f"{e_bytes/1e6:.0f} MB", flush=True)
        self._tail_rebase_at = self.t

    def _tail_restore_from(self, d):
        if self.tail is None:
            return
        self.tail.cols, self.tail.E, self.tail.mask = {}, {}, {}
        self.tail.factor = {}
        # Validate all restored E shapes/bytes before copying any of them to
        # CUDA. The post-load _tail_sync check is too late to prevent a huge
        # checkpoint from allocating past the cap.
        restored = []
        for l in range(self.layercount):
            ck, ek = f"tail_cols.{l}", f"tail_E.{l}"
            if ck not in d or ek not in d:
                continue
            cols = np.asarray(d[ck], np.int64)
            E = np.asarray(d[ek], np.float32)
            if E.ndim != 3 or E.shape[1] != 256:
                raise ValueError(f"tail_E.{l} has invalid shape {E.shape}")
            if E.shape[0] != self.B:
                resized = np.zeros((self.B, 256, E.shape[2]), np.float32)
                resized[:min(self.B, E.shape[0])] = E[:min(self.B, E.shape[0])]
                E = resized
            factor = None
            fk = f"tail_factor.{l}"
            restored.append((l, cols,
                             np.ascontiguousarray(E, dtype=np.float32)))
        restore_bytes = sum(E.size * 4 for _, _, E in restored)
        if self.tail.cap_bytes is not None and restore_bytes > self.tail.cap_bytes:
            raise RuntimeError(
                f"restored tail-credit memory {restore_bytes/1e9:.3f} GB "
                f"exceeds cap {self.tail.cap_bytes/1e9:.3f} GB at B={self.B}; "
                f"lower --batch or set TMT_TAIL=0")
        for l, cols, E in restored:
            self.tail.cols[l] = torch.from_numpy(cols).to(self.device)
            self.tail.E[l] = torch.from_numpy(E).to(self.device)
            if factor is not None:
                self.tail.factor[l] = torch.from_numpy(factor).to(self.device)
        self.tail.dim = self.dim
        for l in range(self.layercount):
            m = torch.zeros(self.dim, dtype=torch.bool, device=self.device)
            if l in self.tail.cols:
                m[self.tail.cols[l]] = True
            self.tail.mask[l] = m

    # ---------------- ring credit variants ----------------
    def _ring_mode(self):
        """The effective ring-credit path, resolved once and cached.

        `auto` means compiled on CUDA, but Inductor/Triton have a hard
        compute-capability floor (7.0) that is NOT reported until the compiled
        function is executed. Checking the capability up front is what lets the
        default degrade instead of crashing: on a pre-Volta GPU (measured: MX150,
        cc 6.1) `auto` used to raise GPUTooOldForTriton on the very first step.
        """
        if self._ring_resolved is not None:
            return self._ring_resolved
        mode = self.ring_credit
        is_cuda = str(self.device).startswith("cuda")
        if not is_cuda:
            # compiled/triton/nvrtc are all CUDA-only; `full` is portable.
            if mode in ("compiled", "triton", "nvrtc"):
                mode = "full"
        elif mode == "auto":
            mode = "compiled"
        if is_cuda and mode in ("compiled", "triton") and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 7:
                fused = "nvrtc" if _nvrtc_ring_kernel() else "full"
                print(f"[ring_credit] cc {major}.x is below Triton's floor (7.0); "
                      f"{mode} unavailable -> {fused}", flush=True)
                mode = fused
        self._ring_resolved = mode
        return mode

    def _scatter_ring_credit(self, layer, w, dlds, rows, hot_mask, dec):
        """Decay this layer's ring in place and credit dembed, via one path.

        The decay lives INSIDE each path, not in the caller, because the fused
        NVRTC path does decay+credit+scatter in a single pass. Every path keeps
        the reference arithmetic order (`w.mul_(dec)`, then `w * dlds`, then the
        atomic scatter), so switching paths cannot change a number.

        Paths that survived measurement on the T4: `full` (portable reference),
        `compiled` (the +27% winner, `auto` on CUDA), `triton` (the +9% fallback
        when Inductor is unavailable), plus `nvrtc` -- fused, and the only fused
        option below cc 7.0 (measured 2.2-2.7x on an MX150). The `cold`,
        `kloop` and `segmented` candidates were removed after measuring 0.95x,
        0.90x and 0.81x respectively -- see RESULTS.md.
        """
        mode = self._ring_mode()
        if mode == "nvrtc":
            kern = _nvrtc_ring_kernel()
            if kern is not None:
                B, K, D = w.shape
                # The hot mask reaches us as EITHER a cuda bool tensor (the
                # rebase path writes one) or a numpy bool array (tail_credit's
                # own _mask dict holds numpy, and masked_fill_ silently accepts
                # both). A raw-pointer kernel accepts neither implicitly, so
                # normalise here rather than fail at the launch.
                if hot_mask is None:
                    marg = self._nvrtc_nomask()
                elif (isinstance(hot_mask, torch.Tensor)
                      and hot_mask.is_cuda and hot_mask.dtype == torch.bool):
                    marg = hot_mask
                else:
                    marg = torch.as_tensor(np.asarray(hot_mask),
                                           dtype=torch.bool,
                                           device=self.device)
                grid = ((B * K * D + 255) // 256, 1, 1)
                kern(grid=grid, block=(256, 1, 1),
                     args=[w, dec, dlds, rows, self._ring_dest, marg,
                           B, K, D, 0 if hot_mask is None else 1])
                return
            mode = "full"        # kernel unavailable: portable reference
        if mode == "compiled":
            if self._compiled_ring_credit is None:
                self._compiled_ring_credit = torch.compile(
                    _ring_credit_compiled, dynamic=False)
            w.mul_(dec[None, None, :])
            self._compiled_ring_credit(self._ring_dest, w, dlds, hot_mask, rows)
            return
        if mode == "triton":
            if triton is None or not w.is_cuda:
                raise RuntimeError("ring_credit=triton requires a CUDA torch "
                                   "build with Triton")
            w.mul_(dec[None, None, :])
            if not hasattr(self, "_triton_dummy"):
                self._triton_dummy = torch.empty(1, dtype=F, device=w.device)
            block_d = triton.next_power_of_2(self.dim)
            _fused_ring_credit_kernel[(w.shape[0] * w.shape[1],)](
                self._ring_dest, w, dlds, rows, self._triton_dummy, hot_mask,
                w.shape[0], w.shape[1], w.shape[2],
                False, hot_mask is not None, block_d, num_warps=4)
            return
        # full: decay, materialize, mask, scatter (the portable reference path)
        w.mul_(dec[None, None, :])
        torch.mul(w, dlds[:, None, :], out=self.ring_vals)
        if hot_mask is not None:
            self.ring_vals.masked_fill_(hot_mask[None, None, :], 0.0)
        self._ring_dest.index_add_(0, rows,
                                  self.ring_vals.reshape(-1, self.dim))

    def _nvrtc_nomask(self):
        if not hasattr(self, "_nvrtc_nomask_t"):
            self._nvrtc_nomask_t = torch.zeros(1, dtype=torch.bool,
                                               device=self.device)
        return self._nvrtc_nomask_t

    # ---------------- training step --------------------
    def step_batch(self, cs, ns, ends, fsp=None):
        """cs/ns: int64 (B,) or (B,T); ends: fp32, same shape — all on
        self.device. Returns the window's loss (python float, or a device
        tensor when sync_metrics=False). Mirrors numpy_port.step_batch at T=1.

        `fsp` is the optional FSP auxiliary: a tuple (mask, targets) where
        `mask` is a host-side tuple of ints (one per byte of the window: 1 =
        this byte is an FSP position) and `targets` is a (B, T, 256) device
        tensor of binary multi-hots (which bytes appear in the next tau
        bytes). The mask is a python tuple on purpose: a device mask would
        need a `.item()` to branch on, and a sync per byte is exactly the
        cost the tail-rebase comment warns about.

        THE T AXIS (PLAN-2026-10-11 P1)
        -------------------------------
        `(B,)` is the historical call and means T=1: it is unsqueezed to
        `(B,1)` and then takes the SAME arithmetic path as a window, so there is
        one implementation and not two that can drift. `(B,T)` is a WINDOW: T
        consecutive bytes per stream, one recurrent pass per byte, ONE AdamW
        update for the whole window.

        What "one update per window" pins down — every one of these is a
        plausible-looking way to be wrong, and none of them raises:

        * `self.t` advances by 1, NOT by T. It is the OPTIMIZER-STEP counter: it
          drives AdamW's bias correction and the LR schedule, and the ring write
          pointer is derived from it. Advancing it per byte would silently train
          on a different schedule with a different bias-correction history while
          every T=1 gate still passed.
        * `self.bytes_seen` advances by B*T. That counter is bytes (numpy_port's
          is B per call, for the same reason).
        * the returned loss is the window SUM `sum_t L_t` — the objective whose
          gradient is actually accumulated below. So the logged loss at T=4 is
          roughly 4x the T=1 line: that is the objective, not a regression. It
          is also why `--batch` and `--seqlen` must be stated together in every
          A/B arm — with the gradient summed over T, the one term in AdamW that
          does NOT scale with the gradient, decoupled weight decay, is
          relatively T times weaker. That is the same shape of interaction
          `--batch` already has: this engine's gradient is a SUM over B while
          the loss is a mean over B (the `/B` asymmetry check_trace_bptt.py
          reports), so a larger B already means relatively weaker decay.
        * `last_ce` is a RATE — nats/byte over the window's B*T positions, the
          same quantity numpy_port documents as "nats/byte" — so training logs
          stay comparable to eval across T. It is deliberately NOT the
          objective; do not read it as the CE component of `loss`.

        EXACTNESS. The residual stream does not cross timesteps: every layer is
        handed the same `enc` and writes only the residual `x`, which is rebuilt
        from `enc` at each byte. So the ONLY cross-time path is the recurrent
        state, and that path is carried exactly — the ring credits every
        retained lag with its true survival product and the tail EMA credits the
        hot dims at every lag. Within the window the gradient is therefore exact
        truncated BPTT; the residual approximation is the one that already
        existed at T=1 (credit older than ring_k lags is dropped, or carried by
        the tail EMA). tests/check_trace_bptt.py measures it at T>1 and reports
        the within-window term and the window-boundary term separately.

        MEMORY. The engine runs with requires_grad=False everywhere — theta is a
        plain `torch.from_numpy` buffer and every gradient is hand-built — so
        there is no autograd graph and nothing is retained across the window:
        each byte's forward cache is dropped when its iteration ends. Activation
        memory stays O(B*D) per layer, NOT O(B*T*D). tools/seqlen_probe.py
        measures it; the PLAN-2026-10-11 §2 P1 note predicting `B*T*D` does not
        hold for this engine, and the missing `requires_grad` is the reason.

        Phase regions (`TMT_PROFILE_REGIONS=0` disables) let the Phase-0
        profiler attribute time to ring / tail / optimizer / layer fwd-bwd
        instead of leaving us to guess from raw kernel names. The regions are
        per-BYTE, so at T>1 each fires T times; that is intended — a window's
        profile is the sum of its bytes.
        """
        rf = _region if _PROFILE_REGIONS else _null_region
        # (B,) -> (B,1). One branch, not two code paths: everything below speaks
        # the 2-D form, so the historical call and a windowed call run identical
        # arithmetic and only the byte index differs.
        if cs.dim() == 1:
            cs = cs.unsqueeze(1)
            ns = ns.unsqueeze(1)
            ends = ends.unsqueeze(1)
        elif cs.dim() != 2:
            raise ValueError(
                f"step_batch wants (B,) or (B,T) tensors; got cs with shape "
                f"{tuple(cs.shape)} (dim {cs.dim()}). The historical 1-D call "
                f"means T=1 and needs no change.")
        B, T = cs.shape
        if T < 1:
            raise ValueError(f"step_batch got an empty window (B,T)=({B},{T})")
        if ns.shape != cs.shape or ends.shape != cs.shape:
            raise ValueError(
                f"cs/ns/ends must share one shape: cs {tuple(cs.shape)}, "
                f"ns {tuple(ns.shape)}, ends {tuple(ends.shape)}")
        # KD is REFUSED at T>1 rather than served wrongly. The target provider
        # is indexed by optimizer step (`step -> (B,256)`), and a window
        # advances that counter by 1 while consuming T bytes per stream, so
        # there is no honest way to ask it for T positions: calling it T times
        # hands every byte of the window the SAME teacher distribution. A
        # silently-wrong KD gradient is worse than a refused run, so refuse,
        # before any state has moved.
        # P3 gated recurrence: REFUSED until the credit machinery is reworked.
        #
        # The blocker is architectural, not a missing line. Gating makes the
        # retention factor input-dependent, a = sigmoid(decay + gate * x), so it
        # becomes PER-SAMPLE (B, dim) instead of the per-dim (dim,) vector every
        # consumer assumes:
        #   - _scatter_ring_credit does w.mul_(dec[None, None, :]) at three
        #     sites, including the fused nvrtc/kloop paths, and w is (B, K, dim);
        #   - TorchTail.begin_step takes sig_hot for the hot dims, i.e. a vector;
        #   - the decaytrace recurrence multiplies by dec[None, :];
        #   - ring_k is INFERRED from sigma_max, which stops being a single
        #     number once sigma depends on the input.
        # Each of those, patched to broadcast a per-sample a, is the kind of
        # change that yields a plausible-looking gradient that is quietly wrong
        # -- the failure class this project is organised around avoiding. A
        # refused run is recoverable; a silently-wrong one is not.
        #
        # The forward, the gate parameter (created only in this arm, so theta's
        # layout is unchanged otherwise) and the GATE GRADIENT are
        # IMPLEMENTED, and the gate gradient is now KNOWN and VERIFIED:
        #
        #   gatetrace_t = a_t*gatetrace_{t-1} + a_t(1-a_t)*x_t*s_{t-1}
        #   dL/dgate    = sum_t dL/ds_t * gatetrace_t        (readout-only dlds)
        #
        # The old formula was a plain sum of the LOCAL effect of a_t on s_t
        # and it was wrong in two ways at once: it dropped the cross-time
        # chain (which the decay's decaytrace already carries) and it summed
        # the wrong axis, so nothing was summed over the batch. The four-way
        # truth table -- plain/trace x readout-only/total dL/ds_t, measured
        # against torch autograd -- is in tools/gate_grad_probe.py, and
        # tests/check_gate_grad.py pins the fixed formula against autograd AND
        # against a central finite difference of the loss.
        #
        # So the REFUSAL is now about the CREDIT machinery only, not about the
        # gradient: the gated arm still cannot run, because the ring credit,
        # the tail and ring_k's inference all assume a per-dim retention. That
        # is a separate, larger piece of work, and until it lands the arm
        # refuses rather than emitting a plausible wrong credit.
        #
        # (An earlier version of this comment said the gate gradient was
        # "finite-difference-pinned in tests/check_gated_recurrence.py". That
        # was false -- the gate file says the opposite -- and a comment that
        # contradicts the gate it points at is worse than no comment, because
        # it is a green light on a known-wrong gradient. The gradient is now
        # pinned, by tests/check_gate_grad.py, which is the file that says so.)
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
        if T > 1 and self.distill_alpha > 0.0:
            raise RuntimeError(
                f"seqlen={T} with distill_alpha={self.distill_alpha}: the KD "
                f"target provider is indexed by OPTIMIZER step, so a window of "
                f"{T} bytes cannot be given {T} distinct teacher targets "
                f"(one call per byte would repeat the same distribution T "
                f"times). Run --seqlen 1 with --distill-alpha, or make the "
                f"provider byte-indexed first.")
        self.t += 1                       # ONE per window — see the docstring
        self.bytes_seen += B * T
        ar = torch.arange(B, device=self.device)   # byte-independent: hoisted
        # alpha = 0 must not perturb a single bit, so the CE weight is not
        # applied at all in that case (gate 3 checks this over 120 steps).
        ce_weight = 1.0 - self.distill_alpha if self.distill_alpha > 0.0 else 1.0
        # CE-only (PLAN-2026-10-06 MEMORY-BENCHMARKS priority 3): the
        # variance-hinge, embed-MSE/JEPA and end-of-text terms are masked by
        # aux_w, which is EXACTLY 0.0 with the switch on and EXACTLY 1.0 off.
        # Off, fp32 x*1.0 == x, so every term keeps its historical value and
        # the loss reduces bit-for-bit to the pre-change expression (checked
        # per-step over 120 steps by tests/check_ce_only.py). The per-term
        # quantities below are exposed MASKED (what enters the loss) and RAW
        # (what would enter without the mask) so that gate can prove the zeroing.
        aux_w = 0.0 if self.ce_only else 1.0

        # ---- window accumulators ------------------------------------------
        # Every gradient piece is a plain SUM over the window's T bytes. That is
        # what makes the "no credit dropped" gate a theorem instead of a
        # tolerance: at T=1 an accumulator receives exactly one term, and
        # 0.0 + x == x bit-exactly in fp32 for every finite x. (Signed zero is
        # the sole exception and is unobservable: `th - (-0.0) == th - 0.0`, and
        # `==` treats them equal, so neither a gate nor AdamW can tell them
        # apart. tests/check_parity.py asserts BIT equality, sign of zero
        # included, so if that ever changes it fails loudly.)
        gv = self.gviews
        dembed = torch.zeros((256, self.dim), dtype=F, device=self.device)
        dWd = torch.zeros_like(gv["Wd"])
        dbd = torch.zeros_like(gv["bd"])
        dWs = torch.zeros_like(gv["Ws"])
        dbs = torch.zeros_like(gv["bs"])
        lw = [torch.zeros_like(gv[f"layer{i}.W"]) for i in range(self.layercount)]
        lg = [torch.zeros_like(gv[f"layer{i}.g"]) for i in range(self.layercount)]
        lb = [torch.zeros_like(gv[f"layer{i}.b"]) for i in range(self.layercount)]
        decay_grads = [torch.zeros_like(gv[f"layer{i}.decay"])
                       for i in range(self.layercount)]
        # P3 gate gradients. Allocated only in the gated arm so the ungated
        # engine pays nothing and, more importantly, so nothing in the default
        # path changes.
        gate_grads = ([torch.zeros_like(gv[f"layer{i}.gate"])
                       for i in range(self.layercount)] if self.gated else None)
        # FSP head gradients. Allocated only when the auxiliary loss is on, so
        # the default path allocates nothing and changes nothing.
        dfsp_W = (torch.zeros_like(gv["fsp.W"]) if self.fsp_alpha > 0.0 else None)
        dfsp_b = (torch.zeros_like(gv["fsp.b"]) if self.fsp_alpha > 0.0 else None)
        dfsp_g = (torch.zeros_like(gv["fsp.g"]) if self.fsp_alpha > 0.0 else None)
        dfsp_b0 = (torch.zeros_like(gv["fsp.b0"]) if self.fsp_alpha > 0.0 else None)
        fsp_sum = torch.zeros((), dtype=F, device=self.device)
        fsp_n = 0
        # The layer INPUT at each timestep, per layer, for THIS byte. The gate
        # reaches a_t through gate * x_t, so x_t is part of the gate's
        # cross-time trace (see gate_trace_update). Stashed here because the
        # backward runs after the whole window and the residual stream has
        # moved on by then.
        gate_xin = ([None] * self.layercount if self.gated else None)
        loss_tensor = torch.zeros((), dtype=F, device=self.device)
        ce_sum = torch.zeros((), dtype=F, device=self.device)
        var_sum = torch.zeros((), dtype=F, device=self.device)
        mse_sum = torch.zeros((), dtype=F, device=self.device)
        end_sum = torch.zeros((), dtype=F, device=self.device)
        # Once per WINDOW, not per byte. The rebase cadence is counted in
        # OPTIMIZER steps (REBASE_EVERY), so a 16-byte window must not trigger
        # 16 membership re-evaluations — a per-step sync was 99% of GPU step
        # time when it ran per step.
        self._ensure_tail()
        self._ring_dest = dembed            # every byte's credit lands here
        K = self.ring_k

        for tt in range(T):
            c_arr, n_arr, end_arr = cs[:, tt], ns[:, tt], ends[:, tt]
            # The ring write pointer is (step, byte) -> exactly one slot per
            # byte, so a window advances it T times. Deriving it from self.t
            # alone — the T=1 form — would make all T bytes of the window write
            # the SAME slot: the last byte would win and T-1 bytes' worth of
            # embedding credit would disappear with no error anywhere.
            p = (self.t - 1 + tt) % K
            # Recomputed PER BYTE. ring_b is rewritten T times inside a window,
            # so a (B*K,) snapshot taken once before the layer loop would still
            # name the PREVIOUS window's bytes for T-1 of the T scatters and
            # move that credit onto the wrong byte ids. (It is a view of ring_b
            # and would track the writes below; re-deriving it makes that
            # explicit instead of load-bearing.)
            rows = self.ring_b.reshape(B * K)            # (B*K,) byte ids

            with rf("fwd_layers"):
                enc = self.embed[c_arr]                  # (B, dim)
                x = enc
                caches, states, decays = [], [], []
                for l in self.layers:
                    x, state, decay, cache = l.forward(enc, x, l.states)
                    caches.append(cache)
                    states.append(state)
                    decays.append(decay)
                if self.gated:
                    # x_in for the gate term, per layer, for THIS byte. Stashed
                    # here because the backward runs after the whole window and
                    # the residual stream has moved on by then.
                    for i, c in enumerate(caches):
                        gate_xin[i] = c[4]

            # ---- FSP: read the STATE and ask it about the future ---------
            fsp_cache = None
            fsp_tgt = None
            if fsp is not None and self.fsp_alpha > 0.0 and fsp[0][tt]:
                fsp_tgt = fsp[1][:, tt, :]              # (B, 256) multi-hot
                fh, fcache = _layer_norm(states[self.fsp_layer],
                                         self.fsp_g, self.fsp_b0)
                fsp_cache = (fh @ self.fsp_W + self.fsp_b, fh, fcache)

            with rf("decoder"):
                output = x @ self.Wd.t() + self.bd       # (B, 256)
                stop = _sig(x @ self.Ws + self.bs[0])    # (B,)

            with rf("loss_metrics"):
                xv = x.var(dim=1, unbiased=False) + 1e-4
                var_active = (1.0 - torch.sqrt(xv)) > 0
                tgt = self.embed[n_arr].detach()
                mx = output.max(dim=1, keepdim=True).values
                lse = mx + torch.log(torch.exp(output - mx).sum(dim=1, keepdim=True))
                ce = lse[:, 0] - output[ar, n_arr]
            # PER-BYTE terms, accumulated separately from the loss expression:
            # folding a running total into the loss would sum prefixes.
            var_t = torch.where(var_active, 1.0 - torch.sqrt(xv), 0.0).sum()
            mse_t = ((x - tgt) ** 2).sum() / self.dim
            end_t = ((stop - end_arr) ** 2).sum()
            ce_t = ce.sum()
            loss_tensor = loss_tensor + (aux_w * var_t + aux_w * mse_t
                                         + ce_weight * ce_t
                                         + aux_w * end_t) / B
            var_sum = var_sum + var_t
            mse_sum = mse_sum + mse_t
            end_sum = end_sum + end_t
            ce_sum = ce_sum + ce_t

            # ---- FSP: the auxiliary loss and its weight -------------------
            if fsp_cache is not None:
                flogits, _, _ = fsp_cache
                # BCE-with-logits against the multi-hot. The positive class is
                # up-weighted (pos_weight) because the future window contains
                # a handful of distinct bytes out of 256: without the weight
                # the loss is dominated by the 200+ absent bytes and the head
                # predicts "nothing is coming" forever.
                wpos = torch.where(fsp_tgt > 0.5,
                                   torch.full_like(fsp_tgt, self.fsp_pos_weight),
                                   torch.ones_like(fsp_tgt))
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    flogits, fsp_tgt, pos_weight=wpos, reduction="none")
                # mean over the 256 classes, then summed over the batch --
                # the same shape as the engine's other per-byte terms, which
                # are all divided by B below.
                fsp_t = bce.mean(dim=1).sum()
                loss_tensor = loss_tensor + self.fsp_alpha * fsp_t / B
                fsp_sum = fsp_sum + fsp_t
                fsp_n = fsp_n + 1

            # ---- distillation: KL(p_teacher_bytes || p_student_bytes) ------
            # L_KD = sum_v p_T(v) (log p_T(v) - log p_S(v)); with tau = 1 frozen
            # its gradient wrt the pre-softmax logits is exactly (p_S - p_T).
            p_t = None
            if self.distill_alpha > 0.0:
                if self._target_fn is None:
                    raise RuntimeError(
                        f"distill_alpha={self.distill_alpha} but no target "
                        f"provider was installed (model.set_target_fn); "
                        f"refusing to train a KD arm with a silently missing "
                        f"target")
                p_t = self._target_fn(self.t)
                if p_t is None:
                    raise RuntimeError(f"target provider returned None at step "
                                       f"{self.t}")
                p_t = p_t.to(output.device, non_blocking=True).float()
                if p_t.shape != output.shape:
                    raise RuntimeError(f"target shape {tuple(p_t.shape)} != "
                                       f"logits {tuple(output.shape)}")
                logp = torch.log_softmax(output, dim=1)
                s_t = (p_t * (torch.log(p_t.clamp_min(1e-9)) - logp)).sum(dim=1)
                covered = p_t.sum(dim=1) > 0
                s_t = torch.where(covered, s_t, torch.zeros_like(s_t))
                loss_tensor = loss_tensor + self.distill_alpha * s_t.sum() / B
                # exposed for telemetry and for tests/check_kd.py: total loss
                # across different alphas is NOT comparable (CE is down-weighted
                # by (1-alpha) by design), so the KD term has to be readable
                # directly.
                self.last_kd = float(s_t.sum()) / B
                self.kd_positions += int(B)
                self.kd_covered += int(covered.sum())
            else:
                self.last_kd = 0.0
            self.last_loss_tensor = loss_tensor.detach()
            # nats/byte over the window's B*T positions. `ce.sum()/n` is
            # bit-identical to `ce.mean()` (checked over 400 random shapes), so
            # the T=1 value is unchanged from the pre-T-axis engine.
            ce_tensor = ce_sum / (B * T)
            self.last_ce_tensor = ce_tensor.detach()
            if self.sync_metrics:
                loss = float(loss_tensor)
                self.last_ce = float(ce_tensor)
            else:
                # Async metrics are the prerequisite for graph/compile
                # experiments: callers sync only when a value is actually logged.
                loss = loss_tensor
                self.last_ce = ce_tensor

            with rf("bwd_decoder"):
                p_s = torch.exp(output - lse)              # softmax
            doutput = p_s.clone()
            doutput[ar, n_arr] -= 1.0
            if ce_weight != 1.0:
                doutput *= ce_weight
            if p_t is not None:
                # The KD gradient must be masked exactly where the KD loss is.
                # `covered` is False at declined positions (inside the token
                # where top-k support fell under `floor`, or the stream's first
                # byte), where p_t is all-zero and the loss above contributes
                # exactly nothing. Without the mask the gradient still received
                # alpha * (p_s - 0) = alpha * p_s there, so the optimiser
                # descended something other than the loss being logged --
                # measured at ~8% of the decoder gradient at 24% coverage, and
                # zero at 100% coverage, which is why check_kd.py never saw it:
                # its synthetic target is non-zero on every row.
                doutput += (self.distill_alpha * (p_s - p_t)
                            * covered[:, None].to(p_s.dtype))
            dWd += doutput.t() @ x
            dbd += doutput.sum(dim=0)
            dx = doutput @ self.Wd

            # The auxiliary terms' gradients are masked with the same aux_w as
            # the loss, so under CE-only the optimizer descends exactly
            # d(CE)/dtheta: the stop-head gradient is exactly zero (its weights
            # then move only by weight decay, which touches a term absent from
            # L), and the var-hinge / embed-MSE contributions to dx are exactly
            # zero. aux_w == 1.0 off-path keeps every product bit-identical to
            # the pre-change engine.
            dstop = 2.0 * (stop - end_arr) * stop * (1.0 - stop)
            dWs += (dstop[:, None] * x).sum(dim=0) * aux_w
            dbs += dstop.sum(dim=0, keepdim=True) * aux_w
            dx = dx + (dstop[:, None] * self.Ws[None, :]) * aux_w

            dvar = torch.where(var_active, -1.0 / (2.0 * torch.sqrt(xv)),
                               0.0)[:, None]
            dx = dx + aux_w * (dvar * 2.0 * (x - x.mean(dim=1, keepdim=True))
                               / self.dim)
            dx = dx + aux_w * (2.0 * (x - tgt) / self.dim)

            with rf("bwd_layers"):
                denc = dx.clone()
                lgrads = [None] * self.layercount
                for i in reversed(range(self.layercount)):
                    l = self.layers[i]
                    dstate, dW, dg, db = l.backward(dx, caches[i])
                    # dx_in == dx_out (residual pass-through); state grad -> denc
                    denc = denc + dstate
                    lw[i] += dW
                    lg[i] += dg
                    lb[i] += db
                    lgrads[i] = dstate
                if fsp_cache is not None:
                    # The FSP head's gradient lands on the STATE it read, and
                    # from there the engine's existing machinery does the
                    # rest: `lgrads[i]` is what the ring credits to the
                    # embedding and what the decaytrace turns into the decay
                    # logits' gradient. That is the whole point of putting
                    # the auxiliary loss on the state -- it reaches the
                    # MEMORY (the decay) and not just the read-out.
                    flogits, fh, fcache = fsp_cache
                    p_fsp = torch.sigmoid(flogits)
                    wpos = torch.where(fsp_tgt > 0.5,
                                       torch.full_like(fsp_tgt,
                                                       self.fsp_pos_weight),
                                       torch.ones_like(fsp_tgt))
                    # d/dz of BCEWithLogits = (sigmoid(z) - y) * w / n_classes,
                    # then alpha.
                    #
                    # NO 1/B HERE, and that is not an oversight: the engine's
                    # gradients are the gradient of the batch SUM, while the
                    # reported loss is the batch MEAN (the `/B` in the loss
                    # expression is for reporting). The CE, the embedding and
                    # the decay rows all follow the SUM convention -- the
                    # repo's P0 gate documents the same asymmetry for the
                    # decay rows. Matching it is what keeps alpha the ratio
                    # the paper means: both terms then scale with the number
                    # of streams, so alpha compares like with like.
                    dz = ((p_fsp - fsp_tgt) * wpos / 256.0 * self.fsp_alpha)
                    dfh = dz @ self.fsp_W.t()
                    dstate_fsp, dg_fsp, db_fsp = _layer_norm_backward(dfh, fcache)
                    lgrads[self.fsp_layer] = lgrads[self.fsp_layer] + dstate_fsp
                    dfsp_g = dfsp_g + dg_fsp
                    dfsp_b0 = dfsp_b0 + db_fsp
                    dfsp_W = dfsp_W + fh.t() @ dz
                    dfsp_b = dfsp_b + dz.sum(dim=0)

            # ---- trace updates (ring + tail credit) — PER BYTE ------------
            # Everything in this block is a per-timestep operation, so each
            # line fires T times inside a window. Leaving any of them as a
            # single per-step write is the silent-failure class this plan is
            # about: each one produces a plausible gradient with T-1 bytes'
            # worth of credit quietly missing.
            for i, l in enumerate(self.layers):
                dlds = lgrads[i]                         # (B, dim)
                w = l.ring_w
                dec = decays[i]
                if self.trace_mode == "dense":
                    # ---- P2: the exact embedding trace -------------------
                    # This is tail_credit.py with every dim hot, and the three
                    # operations are exactly the ones TailCredit performs:
                    #   begin_step  -> E *= sigma            (decay)
                    #   credit      -> dEmb += sum_b E[b,v,:]*dlds[b,:]
                    #   add_byte    -> E[b, c_b, :] += 1.0
                    # Same order (decay, credit, add) as the ring, so the
                    # rhythm -- and therefore which lag each credit term sits
                    # at -- matches.
                    E = self.dense_E[i]
                    with rf("dense_decay"):
                        E.mul_(dec[None, None, :])
                    with rf("dense_credit"):
                        dembed += torch.einsum("bvd,bd->vd", E, dlds)
                    with rf("dense_add"):
                        E[ar, c_arr, :] += 1.0
                else:
                    if self.tail is not None and self.tail.nhot(i):
                        self.tail.begin_step(i, dec[self.tail.cols_of(i)])
                    hot_mask = (self.tail.mask_of(i)
                                if self.tail is not None and self.tail.nhot(i)
                                else None)
                    with rf("ring_credit"):
                        # the decay itself lives in _scatter_ring_credit: the
                        # fused nvrtc path folds it into the same pass as the
                        # credit
                        self._scatter_ring_credit(i, w, dlds, rows, hot_mask,
                                                  dec)
                        w[:, p] = 1.0                    # this byte = lag 1
                    if self.tail is not None and self.tail.nhot(i):
                        with rf("tail_credit"):
                            tailc = self.tail.credit(i, dlds)
                            if tailc is not None:
                                cols = self.tail.cols_of(i)
                                dembed[:, cols] = dembed[:, cols] + tailc
                            self.tail.add_byte(i, c_arr)
                with rf("decaytrace"):
                    # `l.states` still holds the state BEFORE this byte — it is
                    # overwritten at the bottom of this block — which is what
                    # makes the recurrence s_t = a*s_{t-1} + e_t yield
                    # ds_t/ddec = a(1-a)*s_{t-1} + a*ds_{t-1}/ddec exactly.
                    l.decaytrace = (dec[None, :] * l.decaytrace +
                                    dec[None, :] * (1.0 - dec[None, :]) * l.states)
                    decay_grads[i] += (dlds * l.decaytrace).sum(dim=0)
                    if self.gated:
                        # The GATE's chain, which the plain sum did not have.
                        #
                        # The gate enters a_t at time t only, but a_t keeps
                        # multiplying s_{t-1} into every later state, so the
                        # gate's derivative carries the same cross-time chain
                        # the decay's decaytrace carries -- and one extra
                        # factor, because the gate reaches a_t through
                        # gate * x_t:
                        #     gatetrace_t = a_t*gatetrace_{t-1}
                        #                  + a_t(1-a_t)*x_t*s_{t-1}
                        #     dL/dgate    = sum_t dL/ds_t * gatetrace_t
                        #
                        # `dlds` is the READOUT-ONLY state gradient (the states
                        # are stop_gradient), and the trace is what rebuilds
                        # the chain the engine does not have. Feeding this the
                        # TOTAL state gradient instead would double count the
                        # chain -- measured in tools/gate_grad_probe.py.
                        #
                        # `dec` is the EFFECTIVE retention a_t that forward
                        # returned in place of the raw sigmoid(decay): (B, dim)
                        # in the gated arm (a is per-sample there) and (dim,)
                        # in the ungated one. gate_trace_update handles both.
                        contrib, l.gatetrace = gate_gradient(
                            dlds, l.gatetrace, dec, gate_xin[i], l.states)
                        # The contribution is already summed over the BATCH
                        # (gate_gradient does `.sum(dim=0)` on the (B, dim)
                        # product). The old code summed an axis of size 1 --
                        # `dec[None, :]` on a (B, dim) dec broadcasts to
                        # (1, B, dim) -- so nothing was summed over the batch
                        # and the result kept a batch axis it must not have.
                        gate_grads[i] += contrib
                # stop_gradient; after the loop this is the state at the END of
                # the window, which is what the next window must start from.
                l.states = states[i]

            self.ring_b[:, p] = c_arr
            dembed.index_add_(0, c_arr, denc)            # autograd part

        self.last_ce_sum = ce_sum.detach()
        self.last_ce_weight = ce_weight
        if self.fsp_alpha > 0.0:
            # The FSP term is reported SEPARATELY from the loss, exactly like
            # the KD term: a total loss is not comparable across alphas, so
            # the auxiliary term has to be readable on its own.
            #
            # Reported as the MEAN over the batch and over the FSP positions
            # -- the same shape as the CE's per-byte mean -- so the two
            # auxiliary terms can be compared with each other and with the
            # CE without a hidden factor of B.
            self.last_fsp = float(fsp_sum.detach()) / (B * max(1, fsp_n))
            self.fsp_positions += fsp_n
        self.last_aux_masked = ((aux_w * var_sum).detach(),
                                (aux_w * mse_sum).detach(),
                                (aux_w * end_sum).detach())
        self.last_aux_raw = (var_sum.detach(), mse_sum.detach(),
                             end_sum.detach())

        # ---- assemble the flat gradient buffer ----
        # The legacy path issued one copy_ per parameter tensor (69 kernels per
        # step: 5 + 16x4). The profiler charged that to direct_copy at
        # 5.8 ms/step. The pieces are concatenated in self.shapes order -- the
        # exact order the per-view copies wrote -- so this is a pure copy
        # reorganization: bit-identical output, one kernel.
        srcs = {
            "embed": dembed.reshape(self.gviews["embed"].shape),
            "Wd": dWd.reshape(self.gviews["Wd"].shape),
            "bd": dbd.reshape(self.gviews["bd"].shape),
            "Ws": dWs.reshape(self.gviews["Ws"].shape),
            "bs": dbs.reshape(self.gviews["bs"].shape),
        }
        for i in range(self.layercount):
            srcs[f"layer{i}.W"] = lw[i].reshape(self.gviews[f"layer{i}.W"].shape)
            srcs[f"layer{i}.g"] = lg[i].reshape(self.gviews[f"layer{i}.g"].shape)
            srcs[f"layer{i}.b"] = lb[i].reshape(self.gviews[f"layer{i}.b"].shape)
            srcs[f"layer{i}.decay"] = decay_grads[i].reshape(
                self.gviews[f"layer{i}.decay"].shape)
            if self.gated:
                srcs[f"layer{i}.gate"] = gate_grads[i].reshape(
                    self.gviews[f"layer{i}.gate"].shape)
        if self.fsp_alpha > 0.0:
            srcs["fsp.g"] = dfsp_g.reshape(self.gviews["fsp.g"].shape)
            srcs["fsp.b0"] = dfsp_b0.reshape(self.gviews["fsp.b0"].shape)
            srcs["fsp.W"] = dfsp_W.reshape(self.gviews["fsp.W"].shape)
            srcs["fsp.b"] = dfsp_b.reshape(self.gviews["fsp.b"].shape)
        with rf("grad_copy"):
            if self.grad_assembly == "cat":
                torch.cat([srcs[name].reshape(-1) for name, _ in self.shapes],
                          out=self.grads.view(-1))
            else:
                gv = self.gviews
                for name, _ in self.shapes:
                    gv[name].copy_(srcs[name])

        # ---- flat inline AdamW (mirror numpy_port exactly) ----
        # ONE update per window, with `t = self.t` — the optimizer-step counter,
        # not a byte counter. Bias correction and the LR schedule therefore see
        # the same sequence of t values at T>1 as at T=1; only the accumulated
        # gradient differs. (Advancing t by T here would keep every T=1 gate
        # green while changing the schedule and the correction history.)
        with rf("optimizer"):
            t = self.t
            th, gr, m, v = self.theta, self.grads, self.m, self.v
            # ---- NaN guard (PLAN-2026-10-11 P0 item 3) -------------------
            # A non-finite gradient is REFUSED, not applied: theta, m and v
            # are left exactly as they were. `t` still advances (the data was
            # consumed; rewinding the LR schedule would be a second, quieter
            # way to change the run), and the refusal is counted so a run that
            # silently stopped learning is visible in the log.
            #
            # The guard is a no-op on finite gradients, and the check is a
            # single reduction plus one host sync -- the same trade the CUDA
            # fork makes for its global grad-norm sync.
            self.last_nan_refused = False
            finite = True
            if self.nan_guard:
                finite = bool(torch.isfinite(gr).all().item())
                if not finite:
                    self.nan_refusals += 1
                    self.last_nan_refused = True
            if finite:
                # ---- global grad clip ----------------------------------
                # PyTorch's clip_grad_norm_ convention: scale the whole
                # gradient by min(clip / (norm + 1e-6), 1). Applied BEFORE the
                # AdamW moments are updated, so the moments see the gradient
                # that was actually applied.
                if self.grad_clip > 0.0:
                    gn = float(torch.linalg.vector_norm(gr).item())
                    coef = self.grad_clip / (gn + 1e-6)
                    if coef < 1.0:
                        gr.mul_(coef)
                m.mul_(self.b1).add_((1.0 - self.b1) * gr)
                v.mul_(self.b2).add_((1.0 - self.b2) * (gr * gr))
                v.sqrt_()
                # The learning rate is a per-coordinate VECTOR so a role can be
                # scaled independently. Filled with exactly float32(self.lr) when
                # lr_decay is unset, which makes this arithmetically identical to
                # the old scalar form (a Python float promotes to float32 against a
                # float32 tensor) -- tests/check_lr_roles.py gates that bit-exactly.
                lv = self._role_lr_vec()
                # The weight decay is ALSO a per-coordinate vector, so the
                # `.decay`/`.gate` groups can be exempted (P6). With the
                # default wd_decay == wd it is a uniform vector and therefore
                # bit-identical to the incumbent's scalar `wd * th`.
                th.sub_(lv * (m * (1.0 / (1.0 - self.b1 ** t)) /
                              (v * (1.0 / (1.0 - self.b2 ** t)) + self.eps)
                              + self._wd_coord_vec() * th))
            # else: the update is REFUSED. theta, m, v untouched.
        return loss

    def _wd_coord_vec(self):
        """The per-coordinate weight-decay vector, cached.

        Uniform `self.wd` everywhere except the `.decay` and `.gate` groups,
        which get `self.wd_decay`. With the default wd_decay == wd this is a
        uniform vector, and `vec * th` is bit-identical to the incumbent's
        scalar `wd * th` (both multiply a float32(0.01) by a float32 tensor).
        """
        if self._wd_vec is not None:
            return self._wd_vec
        n = self.theta.numel()
        wv = torch.full((n,), self.wd, dtype=F, device=self.device)
        off = 0
        for name, shape in self.shapes:
            k = int(np.prod(shape)) if shape else 1
            if name.endswith(".decay") or name.endswith(".gate"):
                wv[off:off + k] = self.wd_decay
            off += k
        self._wd_vec = wv
        return wv

    def _role_lr_vec(self):
        """The per-coordinate learning-rate vector, cached.

        Per-role semantics follow the usual param-group convention: the role's
        LR scales the WHOLE update bracket for that coordinate, i.e. both the
        adaptive term and the decoupled weight-decay term. A decay coordinate
        with `lr_decay=16*lr` therefore also gets 16x the weight decay --
        which is why P6 exists: `wd_decay=0` removes the weight-decay term
        from the `.decay` group entirely, so the role's LR then scales ONLY
        the adaptive term and the "knob wired to a trap" is gone.

        (The trap is still live for any role that is scaled and still has a
        non-zero weight decay; today `.decay` is the only such role.)
        """
        if self._lr_vec is not None:
            return self._lr_vec
        n = self.theta.numel()
        lv = torch.full((n,), self.lr, dtype=F, device=self.device)
        if self.lr_decay is not None:
            off = 0
            for name, shape in self.shapes:
                k = int(np.prod(shape)) if shape else 1
                if name.endswith(".decay"):
                    lv[off:off + k] = self.lr_decay
                off += k
        self._lr_vec = lv
        return lv

    # ---------------- persistence / bridging -------------
    def params_dict(self):
        out = {k: v.detach().cpu().numpy() for k, v in self.views.items()}
        for i, l in enumerate(self.layers):
            out[f"state.{i}"] = l.states.detach().cpu().numpy()
            out[f"decaytrace.{i}"] = l.decaytrace.detach().cpu().numpy()
            # The GATE's trace is NOT persisted: the gated arm refuses to run
            # today, so the buffer is always zero and there is nothing to
            # lose. When the per-sample credit plumbing lands, the gate trace
            # MUST be added here and to load(): a resumed gated run that
            # started the trace at zero would silently lose the chain across
            # the resume -- i.e. exactly the bug the trace exists to fix.
        out["t"] = np.array(self.t, np.int64)
        out["bytes"] = np.array(self.bytes_seen, np.int64)
        # WHERE THE DATA STREAM IS. The trainer restarts the corpus from the
        # first file on every resume, so without this a resumed run re-reads
        # the bytes it had already trained on: the arm's UNIQUE data would be
        # capped at the checkpoint cadence (10 MB at the 10k-step cadence),
        # and a 165 MB epoch would degenerate into "the same 10 MB, sixteen
        # times". The pair (file index, byte offset within the file) is the
        # whole position, because the file order is a deterministic shuffle of
        # the data seed.
        if self.data_pos is not None:
            out["data_file"] = np.array(self.data_pos[0], np.int64)
            out["data_off"] = np.array(self.data_pos[1], np.int64)
        out["m"] = self.m.detach().cpu().numpy()
        out["v"] = self.v.detach().cpu().numpy()
        out["B"] = np.array(self.B, np.int64)
        out["ring_k"] = np.array(self.ring_k, np.int64)
        if self.trace_mode == "dense" and self.dense_E is not None:
            # P2: persist the exact trace, exactly as the ring persisted ring_w.
            # Same order of size -- B*256*dim*4 per layer vs the ring's
            # B*ring_k*dim*4 with ring_k capped at 256 -- so this is not a new
            # cost, it is the same one moved.
            #
            # Without it load() would silently start the trace at zero. Harmless
            # for cold dims (sigma**50 ~ 1e-15) but NOT for hot ones (sigma=0.98
            # gives sigma**50 = 0.36), so a resumed dense run would quietly lose
            # the history of exactly the dims the trace exists for.
            for i in range(self.layercount):
                out[f"dense_E.{i}"] = self.dense_E[i].detach().cpu().numpy()
            out["trace_mode"] = np.array(0, np.int64)   # 0 == dense
        elif self.tail_enabled and self.tail is not None:
            for l, cols in self.tail.cols.items():
                out[f"tail_cols.{l}"] = cols.cpu().numpy().astype(np.int64)
                out[f"tail_E.{l}"] = self.tail.E[l].cpu().numpy()
        return out

    def save(self, path):
        data = self.params_dict()
        tmp = path + ".tmp.npz"
        # DEFLATE, not stored. np.savez writes a ZIP with STORED members; the
        # B-dependent state is mostly small-magnitude, so DEFLATE collapses it.
        # Measured on real checkpoints, LOSSLESS -- a compressed checkpoint
        # reloads bit-exact (107/107 keys identical) and np.load auto-detects
        # either form, so old and new checkpoints stay interchangeable.
        #
        #   tmt_t094747.npz  B=1024 ring_k=18 115 hot dims
        #       stored  230.3 MiB  ->  DEFLATE  115.9 MiB   (2.0x)
        #   tmt_t101847.npz  B=1024 ring_k=18 123 hot dims
        #       stored  238.4 MiB  ->  DEFLATE  ~119   MiB   (2.0x)
        #   a freshly seeded model (tail_E still all zeros)
        #       stored  115.3 MiB  ->  DEFLATE   15.9 MiB   (7.3x)
        #
        # The win is entirely in the B-dependent state: only ~7% of tail_E
        # entries exceed 1e-4 of their layer max (all 16 layers measured), so
        # most of it is long runs of near-identical bytes. params (theta, m, v,
        # ~51 MiB) are float noise and barely compress.
        #
        # WHAT THIS DOES NOT DO: it does not reduce RAM or VRAM. The live
        # buffers are the same size; only the file shrinks. So it makes a big
        # checkpoint cheap to push/pull, but it does NOT raise the batch
        # ceiling -- VRAM is the limit there, and B=4096 already needs
        # 3,148 MiB of a 3.94 GiB card.
        #
        # Cost: ~8.4s to write a 230 MiB checkpoint vs ~0.2s stored, ~0.2s
        # extra to read. Against a 1.5-2 min network transfer, free.
        np.savez_compressed(tmp, **data)
        os.replace(tmp, path)

    def load(self, path):
        """Load a numpy_port-compatible .npz (params/m/v/states/tail/ring_k)."""
        if not os.path.exists(path):
            return False
        d = np.load(path)
        # Refuse checkpoints written by the removed factored candidates rather
        # than silently reading them with the legacy semantics.
        stale = [k for k in d.files
                 if k.startswith("tail_factor.") or k.startswith("ring_factor.")
                 or k == "algorithm_version"]
        if stale:
            raise ValueError(
                f"checkpoint {path} carries removed factored-candidate keys "
                f"({stale[:3]}...); it cannot be read by the current engine -- "
                f"retrain or convert from a legacy checkpoint")
        if "ring_k" in d:
            stored_ring_k = int(np.ravel(d["ring_k"])[0])
            if self._ring_k_explicit and stored_ring_k != self.ring_k:
                raise ValueError(
                    f"checkpoint {path} stores ring_k={stored_ring_k}, but "
                    f"the requested explicit ring_k={self.ring_k}; refusing "
                    "to silently migrate or overwrite the run contract")
            if stored_ring_k != self.ring_k:
                self.ring_k = stored_ring_k
                # Inferred (non-explicit) loads may restore the checkpoint's
                # historical K. Explicit loads never reach this branch.
                self.new_states(self.B)
        if "t" in d:
            self.t = int(d["t"])
        self.bytes_seen = int(d["bytes"]) if "bytes" in d else self.t * self.B
        if "data_file" in d and "data_off" in d:
            self.data_pos = (int(d["data_file"]), int(d["data_off"]))
        if "m" in d:
            if d["m"].shape[0] != self.m.shape[0]:
                raise ValueError(
                    f"checkpoint {path} holds {int(d['m'].shape[0]):,} params but "
                    f"model has {self.m.shape[0]:,}")
            with torch.no_grad():
                self.m.copy_(torch.from_numpy(d["m"].astype(np.float32)).to(self.device))
                self.v.copy_(torch.from_numpy(d["v"].astype(np.float32)).to(self.device))
        # trace_mode is ABSENT from every legacy (ring) checkpoint -- it is a
        # new key -- so absent means ring. Checking `"trace_mode" in d` and then
        # only acting on the dense case let a ring checkpoint load into a dense
        # engine, which silently starts the trace at zero.
        ck_dense = ("trace_mode" in d
                    and int(np.ravel(d["trace_mode"])[0]) == 0)
        if ck_dense and self.trace_mode != "dense":
            raise ValueError(
                f"checkpoint {path} is a DENSE-trace run but this engine is "
                f"trace_mode={self.trace_mode!r}. Resume with "
                f"trace_mode='dense'; the ring would start cold and silently "
                f"drop the trace history it was saved with.")
        if (not ck_dense) and self.trace_mode == "dense":
            raise ValueError(
                f"checkpoint {path} is a RING run but this engine is "
                f"trace_mode='dense'. The dense trace would start at zero "
                f"rather than at the ring's accumulated credit -- harmless for "
                f"cold dims but not hot ones (sigma**50 = 0.36 at sigma=0.98). "
                f"Convert deliberately, or train the dense arm from scratch.")
        # Restore the dense trace, adapting B the way the state rows do.
        if self.trace_mode == "dense" and self.dense_E is not None:
            for i in range(self.layercount):
                k = f"dense_E.{i}"
                if k not in d:
                    continue
                E = d[k].astype(np.float32)
                if E.shape[0] != self.B:
                    r = np.zeros((self.B, 256, self.dim), np.float32)
                    n = min(self.B, E.shape[0])
                    r[:n] = E[:n]
                    E = r
                with torch.no_grad():
                    self.dense_E[i].copy_(torch.from_numpy(E).to(self.device))
        off = 0
        for name, shape in self.shapes:
            if name in d:
                size = int(np.prod(shape))
                with torch.no_grad():
                    self.theta[off:off + size].copy_(
                        torch.from_numpy(d[name].astype(np.float32).ravel()).to(self.device))
            off += size
        with torch.no_grad():
            for i, l in enumerate(self.layers):
                for key, attr in ((f"state.{i}", "states"),
                                  (f"decaytrace.{i}", "decaytrace")):
                    if key not in d:
                        continue
                    v = d[key].astype(np.float32)
                    if v.shape[0] != self.B:
                        resized = np.zeros((self.B,) + v.shape[1:], np.float32)
                        n = min(self.B, v.shape[0])
                        resized[:n] = v[:n]
                        v = resized
                    t = torch.from_numpy(v).to(self.device)
                    if attr == "states":
                        l.states.copy_(t)
                    else:
                        l.decaytrace.copy_(t)
        self.tail_state = {}
        if self.tail_enabled:
            for i in range(self.layercount):
                for key in (f"tail_cols.{i}", f"tail_E.{i}",
                            f"tail_factor.{i}"):
                    if key in d:
                        self.tail_state[key] = d[key]
        self._ensure_tail()
        return True

    def from_numpy(self, np_model):
        """Adopt every parameter/buffer/moment from a numpy_port.Model."""
        self.new_states(np_model.B)
        d = {}
        for k, v in np_model.params().items():
            d[k] = v
        for i, l in enumerate(np_model.layers):
            d[f"state.{i}"] = l.states
            d[f"decaytrace.{i}"] = l.decaytrace
        d["t"] = np.array(np_model.t, np.int64)
        d["bytes"] = np.array(np_model.bytes_seen, np.int64)
        d["m"] = np_model.m
        d["v"] = np_model.v
        d["B"] = np.array(np_model.B, np.int64)
        d["ring_k"] = np.array(np_model.ring_k, np.int64)
        if np_model.tail is not None:
            d.update(np_model.tail.save_dict())
        tmp = "/tmp/_tmt_bridge.npz"
        np.savez(tmp, **d)
        self.load(tmp)
        return self


if __name__ == "__main__":
    print("gpu_port model devices:",
          "cuda" if torch.cuda.is_available() else "cpu")