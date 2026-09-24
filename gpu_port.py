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

    def forward(self, enc, x_in, states):
        decay = _sig(self.decay)
        state = states * decay + enc
        h, cache = _layer_norm(state, self.g, self.b)
        z = h @ self.W.t()
        sg = torch.sigmoid(z)
        y = z * sg
        return x_in + y, state, decay, (z, sg, h, cache)

    def backward(self, dx_out, cache):
        z, sg, h, ln_cache = cache
        dz = dx_out * sg * (1.0 + z * (1.0 - sg))
        dW = dz.t() @ h
        dh = dz @ self.W
        dstate, dg, db = _layer_norm_backward(dh, ln_cache)
        return dstate, dW, dg, db


def _ring_credit_compiled(dest, w, dlds, hot_mask, rows):
    """Inductor candidate for A2; compilation is warmed outside the timing."""
    vals = w * dlds[:, None, :]
    if hot_mask is not None:
        vals.masked_fill_(hot_mask[None, None, :], 0.0)
    dest.index_add_(0, rows, vals.reshape(-1, w.shape[-1]))
    return dest


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

    def __init__(self, dim=512, layers=16, temp=0.75, lr=5e-4, decay_init=None,
                 device=None, ring_credit="auto", sync_metrics=True,
                 grad_assembly="cat", seed=0):
        self.dim = dim
        self.layercount = layers
        self.temp = temp
        self.lr = float(lr)
        self.b1, self.b2, self.eps, self.wd = 0.9, 0.999, 1e-8, 0.01
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._init_seed = int(seed)
        if ring_credit not in ("auto", "full", "compiled", "triton"):
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
            dmax = float(_sig(torch.tensor(
                decay_init if decay_init is not None else lim,
                dtype=torch.float32)).item())
            self.ring_k = int(min(256, np.ceil(
                np.log(RING_TOL) / np.log(max(dmax, 1e-3)))))
        self._build(dim, layers, decay_init)

    def _build(self, dim, layers, decay_init):
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
            if decay_init is not None:
                d0[:] = decay_init
            init[f"layer{i}.decay"] = d0

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
        self.layers = [TorchLayer(dim) for _ in range(layers)]
        for i, l in enumerate(self.layers):
            l.W = self.views[f"layer{i}.W"]
            l.g = self.views[f"layer{i}.g"]
            l.b = self.views[f"layer{i}.b"]
            l.decay = self.views[f"layer{i}.decay"]
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
        self.ring_vals = torch.zeros((B, self.ring_k, self.dim), dtype=F,
                                     device=self.device)  # reusable scratch
        for l in self.layers:
            l.states = torch.zeros((B, self.dim), dtype=F, device=self.device)
            l.decaytrace = torch.zeros((B, self.dim), dtype=F, device=self.device)
            l.ring_w = torch.zeros((B, self.ring_k, self.dim), dtype=F,
                                   device=self.device)

    # ---------------- tail credit --------------------
    def _ensure_tail(self):
        if not self.tail_enabled:
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
    def _scatter_ring_credit(self, layer, w, dlds, rows, hot_mask):
        """Apply one layer's ring credit to dembed via the selected path.

        Only the three paths that survived measurement on the T4 remain:
        `full` (portable reference), `compiled` (the +27% winner, default on
        CUDA) and `triton` (the +9% fallback when Inductor is unavailable).
        The `cold`, `kloop` and `segmented` candidates were removed after
        measuring 0.95x, 0.90x and 0.81x respectively -- see RESULTS.md.
        """
        mode = self.ring_credit
        if mode == "compiled":
            if self._compiled_ring_credit is None:
                self._compiled_ring_credit = torch.compile(
                    _ring_credit_compiled, dynamic=False)
            self._compiled_ring_credit(self._ring_dest, w, dlds, hot_mask, rows)
            return
        if mode == "triton":
            if triton is None or not w.is_cuda:
                raise RuntimeError("ring_credit=triton requires a CUDA torch "
                                   "build with Triton")
            if not hasattr(self, "_triton_dummy"):
                self._triton_dummy = torch.empty(1, dtype=F, device=w.device)
            block_d = triton.next_power_of_2(self.dim)
            _fused_ring_credit_kernel[(w.shape[0] * w.shape[1],)](
                self._ring_dest, w, dlds, rows, self._triton_dummy, hot_mask,
                w.shape[0], w.shape[1], w.shape[2],
                False, hot_mask is not None, block_d, num_warps=4)
            return
        # full: materialize, mask, scatter (the portable reference path)
        torch.mul(w, dlds[:, None, :], out=self.ring_vals)
        if hot_mask is not None:
            self.ring_vals.masked_fill_(hot_mask[None, None, :], 0.0)
        self._ring_dest.index_add_(0, rows,
                                  self.ring_vals.reshape(-1, self.dim))

    # ---------------- training step --------------------
    def step_batch(self, cs, ns, ends):
        """cs/ns: int64 tensors (B,), ends: fp32 (B,) — all on self.device.
        Returns loss (python float, or a device tensor when sync_metrics=False).
        Mirrors numpy_port.step_batch exactly.

        Phase regions (`TMT_PROFILE_REGIONS=0` disables) let the Phase-0
        profiler attribute time to ring / tail / optimizer / layer fwd-bwd
        instead of leaving us to guess from raw kernel names.
        """
        rf = _region if _PROFILE_REGIONS else _null_region
        self.t += 1
        B = cs.shape[0]
        self.bytes_seen += B
        c_arr, n_arr, end_arr = cs, ns, ends

        with rf("fwd_layers"):
            enc = self.embed[c_arr]                      # (B, dim)
            x = enc
            caches, states, decays = [], [], []
            for l in self.layers:
                x, state, decay, cache = l.forward(enc, x, l.states)
                caches.append(cache)
                states.append(state)
                decays.append(decay)

        with rf("decoder"):
            output = x @ self.Wd.t() + self.bd           # (B, 256)
            stop = _sig(x @ self.Ws + self.bs[0])        # (B,)

        with rf("loss_metrics"):
            xv = x.var(dim=1, unbiased=False) + 1e-4
            var_active = (1.0 - torch.sqrt(xv)) > 0
            tgt = self.embed[n_arr].detach()
            mx = output.max(dim=1, keepdim=True).values
            lse = mx + torch.log(torch.exp(output - mx).sum(dim=1, keepdim=True))
            ar = torch.arange(B, device=self.device)
            ce = lse[:, 0] - output[ar, n_arr]
        loss_tensor = (torch.where(var_active, 1.0 - torch.sqrt(xv), 0.0).sum()
                       + ((x - tgt) ** 2).sum() / self.dim
                       + ce.sum() + ((stop - end_arr) ** 2).sum()) / B
        ce_tensor = ce.mean()
        self.last_loss_tensor = loss_tensor.detach()
        self.last_ce_tensor = ce_tensor.detach()
        if self.sync_metrics:
            loss = float(loss_tensor)
            self.last_ce = float(ce_tensor)
        else:
            # Async metrics are the prerequisite for graph/compile experiments:
            # callers sync only when a value is actually logged.
            loss = loss_tensor
            self.last_ce = ce_tensor

        with rf("bwd_decoder"):
            doutput = torch.exp(output - lse)            # softmax
        doutput[ar, n_arr] -= 1.0
        dWd = doutput.t() @ x
        dbd = doutput.sum(dim=0)
        dx = doutput @ self.Wd

        dstop = 2.0 * (stop - end_arr) * stop * (1.0 - stop)
        dWs = (dstop[:, None] * x).sum(dim=0)
        dbs = dstop.sum(dim=0, keepdim=True)
        dx = dx + dstop[:, None] * self.Ws[None, :]

        dvar = torch.where(var_active, -1.0 / (2.0 * torch.sqrt(xv)), 0.0)[:, None]
        dx = dx + (dvar * 2.0 * (x - x.mean(dim=1, keepdim=True)) / self.dim)
        dx = dx + (2.0 * (x - tgt) / self.dim)

        with rf("bwd_layers"):
            denc = dx.clone()
            lw, lg, lb, lgrads = [], [], [], []
            for i in reversed(range(self.layercount)):
                l = self.layers[i]
                dstate, dW, dg, db = l.backward(dx, caches[i])
                # dx_in == dx_out (residual pass-through); state grad -> denc
                denc = denc + dstate
                lw.append(dW); lg.append(dg); lb.append(db)
                lgrads.append(dstate)
            lgrads.reverse(); lw.reverse(); lg.reverse(); lb.reverse()

        # ---- trace updates (ring + tail credit) ----
        self._ensure_tail()
        dembed = torch.zeros((256, self.dim), dtype=F, device=self.device)
        decay_grads = []
        K = self.ring_k
        p = (self.t - 1) % K
        rows = self.ring_b.reshape(B * K)                # (B*K,) byte ids
        self._ring_dest = dembed
        for i, l in enumerate(self.layers):
            dlds = lgrads[i]                             # (B, dim)
            w = l.ring_w
            dec = decays[i]
            w.mul_(dec[None, None, :])               # decay the ring in place
            if self.tail is not None and self.tail.nhot(i):
                self.tail.begin_step(i, dec[self.tail.cols_of(i)])
            hot_mask = (self.tail.mask_of(i)
                        if self.tail is not None and self.tail.nhot(i) else None)
            with rf("ring_credit"):
                self._scatter_ring_credit(i, w, dlds, rows, hot_mask)
                w[:, p] = 1.0                            # this byte = lag 1
            if self.tail is not None and self.tail.nhot(i):
                with rf("tail_credit"):
                    tailc = self.tail.credit(i, dlds)
                    if tailc is not None:
                        cols = self.tail.cols_of(i)
                        dembed[:, cols] = dembed[:, cols] + tailc
                    self.tail.add_byte(i, c_arr)
            with rf("decaytrace"):
                l.decaytrace = (dec[None, :] * l.decaytrace +
                                dec[None, :] * (1.0 - dec[None, :]) * l.states)
                decay_grads.append((dlds * l.decaytrace).sum(dim=0))
            l.states = states[i]                         # stop_gradient

        self.ring_b[:, p] = c_arr
        dembed.index_add_(0, c_arr, denc)                # autograd part

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
        with rf("grad_copy"):
            if self.grad_assembly == "cat":
                torch.cat([srcs[name].reshape(-1) for name, _ in self.shapes],
                          out=self.grads.view(-1))
            else:
                gv = self.gviews
                for name, _ in self.shapes:
                    gv[name].copy_(srcs[name])

        # ---- flat inline AdamW (mirror numpy_port exactly) ----
        with rf("optimizer"):
            t = self.t
            th, gr, m, v = self.theta, self.grads, self.m, self.v
            m.mul_(self.b1).add_((1.0 - self.b1) * gr)
            v.mul_(self.b2).add_((1.0 - self.b2) * (gr * gr))
            v.sqrt_()
            th.sub_(self.lr * (m * (1.0 / (1.0 - self.b1 ** t)) /
                               (v * (1.0 / (1.0 - self.b2 ** t)) + self.eps)
                               + self.wd * th))
        return loss

    # ---------------- persistence / bridging -------------
    def params_dict(self):
        out = {k: v.detach().cpu().numpy() for k, v in self.views.items()}
        for i, l in enumerate(self.layers):
            out[f"state.{i}"] = l.states.detach().cpu().numpy()
            out[f"decaytrace.{i}"] = l.decaytrace.detach().cpu().numpy()
        out["t"] = np.array(self.t, np.int64)
        out["bytes"] = np.array(self.bytes_seen, np.int64)
        out["m"] = self.m.detach().cpu().numpy()
        out["v"] = self.v.detach().cpu().numpy()
        out["B"] = np.array(self.B, np.int64)
        out["ring_k"] = np.array(self.ring_k, np.int64)
        if self.tail_enabled and self.tail is not None:
            for l, cols in self.tail.cols.items():
                out[f"tail_cols.{l}"] = cols.cpu().numpy().astype(np.int64)
                out[f"tail_E.{l}"] = self.tail.E[l].cpu().numpy()
        return out

    def save(self, path):
        data = self.params_dict()
        tmp = path + ".tmp.npz"
        np.savez(tmp, **data)
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
        if "ring_k" in d and int(np.ravel(d["ring_k"])[0]) != self.ring_k:
            self.ring_k = int(np.ravel(d["ring_k"])[0])
            # new_states() drops the tail built against the old ring, so the
            # newly sized ring/tail gets re-seeded by _ensure_tail below.
            self.new_states(self.B)
        if "t" in d:
            self.t = int(d["t"])
        self.bytes_seen = int(d["bytes"]) if "bytes" in d else self.t * self.B
        if "m" in d:
            if d["m"].shape[0] != self.m.shape[0]:
                raise ValueError(
                    f"checkpoint {path} holds {int(d['m'].shape[0]):,} params but "
                    f"model has {self.m.shape[0]:,}")
            with torch.no_grad():
                self.m.copy_(torch.from_numpy(d["m"].astype(np.float32)).to(self.device))
                self.v.copy_(torch.from_numpy(d["v"].astype(np.float32)).to(self.device))
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