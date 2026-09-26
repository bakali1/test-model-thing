#!/usr/bin/env python3
"""Observation-only training telemetry for TMT.

The trainer calls :meth:`RunTelemetry.observe` after each training step and
:meth:`RunTelemetry.maybe_emit` only at an absolute step interval.  No value is
fed back into the model, optimizer, data order, or checkpoint state.

Two JSONL event types are emitted:

* ``interval``: loss/CE windows, warm throughput, parameter movement and
  gradient SNR, per-layer decay/state health, and tail/ring summaries;
* ``nonfinite_loss``: a loud immediate failure record written before training
  aborts.

The telemetry deliberately aggregates vectors on device and synchronizes only
at the configured boundary.  It is disabled by default in ``train_gpu.py``.
"""
import json
import math
import os
import time
from collections import deque

import torch


def update_gradient_ema(grad, ema_grad, ema_grad_sq, decay=0.9):
    """Update fixed-size gradient EMAs in place and return (mean, snr).

    ``snr = ||EMA(g)||^2 / EMA(||g||^2)`` for a flat group.  A zero-gradient
    group returns exactly zero; a constant-direction vector approaches one
    after the EMA has accumulated history.
    """
    with torch.no_grad():
        g2 = grad * grad
        ema_grad.mul_(decay).add_(grad, alpha=1.0 - decay)
        ema_grad_sq.mul_(decay).add_(g2, alpha=1.0 - decay)
        numerator = ema_grad.square().sum()
        denominator = ema_grad_sq.sum().clamp_min(torch.finfo(grad.dtype).tiny)
        return float(grad.abs().mean()), float(numerator / denominator)


def _finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def _vector_rms(x):
    return float(torch.sqrt(torch.sum(x.float() * x.float()) /
                           max(1, x.numel())).clamp_min(0.0))


def _vector_stats(x):
    xf = x.detach().float()
    return {
        "rms": _finite_float(torch.sqrt(torch.mean(xf * xf))),
        "mean_abs": _finite_float(xf.abs().mean()),
        "activity_ge_1e-3": _finite_float((xf.abs() >= 1.0e-3).float().mean()),
    }


class RunTelemetry:
    """Fixed-size, interval-only observation state for one training run."""

    def __init__(self, path, batch, measure_after=0, ema_decay=0.9):
        self.path = os.path.abspath(path)
        self.batch = int(batch)
        self.measure_after = max(0, int(measure_after))
        self.ema_decay = float(ema_decay)
        self.enabled = True
        self._loss = deque(maxlen=200000)
        self._ce = deque(maxlen=200000)
        self._prev_theta = None
        self._ema_grad = None
        self._ema_grad_sq = None
        self._measure_clock = None
        self._measure_t = None
        self._measure_bytes = None
        self._last_rebase_at = -1
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    @property
    def is_active(self):
        return self.enabled

    def observe(self, loss, ce):
        # The production path already returns Python floats.  ``float`` is a
        # no-op there; it also supports an async-metrics smoke experiment.
        self._loss.append(float(loss))
        self._ce.append(float(ce))

    def nonfinite_loss(self, model, loss, ce):
        event = {
            "event": "nonfinite_loss",
            "t": int(model.t),
            "bytes_seen": int(model.bytes_seen),
            "B": self.batch,
            "loss": str(loss),
            "ce": str(ce),
        }
        self._write(event)
        print(f"[telemetry] FATAL non-finite loss at t={model.t}: "
              f"loss={loss} ce={ce}", flush=True)
        raise FloatingPointError(f"non-finite training loss at t={model.t}")

    def _write(self, event):
        line = json.dumps(event, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def mark_measurement(self, model, now=None):
        """Start the throughput clock without touching model state."""
        if self._measure_clock is None:
            self._measure_clock = time.time() if now is None else now
            self._measure_t = int(model.t)
            self._measure_bytes = int(model.bytes_seen)
        return self._measure_clock

    def _throughput(self, model, now):
        if self._measure_clock is None:
            if model.t < self.measure_after:
                return None, None, None
            self.mark_measurement(model, now)
            return None, None, None
        elapsed = max(now - self._measure_clock, 1.0e-9)
        bps = (int(model.bytes_seen) - self._measure_bytes) / elapsed
        steps_per_s = (int(model.t) - self._measure_t) / elapsed
        return _finite_float(bps), _finite_float(steps_per_s), elapsed

    def _prepare_emas(self, model):
        if self._ema_grad is None:
            n = model.grads.numel()
            self._ema_grad = torch.zeros_like(model.grads)
            self._ema_grad_sq = torch.zeros_like(model.grads)

    def _groups(self, model):
        """[(name, start, end)] in the canonical flat parameter layout."""
        groups = []
        offset = 0
        trunk_ranges = []
        for name, shape in model.shapes:
            n = math.prod(shape) if shape else 1
            if not name.startswith("layer") or not name.endswith(".decay"):
                trunk_ranges.append((offset, offset + n))
            else:
                groups.append((name, offset, offset + n))
            offset += n
        merged = []
        for start, end in trunk_ranges:
            if merged and start == merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        out = [("shared_trunk", a, b) for a, b in merged]
        out.extend(groups)
        return out

    def _parameter_group(self, model, start, end, update_em_as=True):
        grad = model.grads[start:end]
        mean_abs, snr = update_gradient_ema(
            grad, self._ema_grad[start:end], self._ema_grad_sq[start:end],
            self.ema_decay)
        now = model.theta[start:end].detach()
        if self._prev_theta is None:
            movement = 0.0
        else:
            delta = now - self._prev_theta[start:end]
            movement = _finite_float(torch.sqrt(torch.mean(delta * delta)))
        theta_rms = _vector_rms(now)
        return {
            "grad_mean_abs": _finite_float(mean_abs),
            "grad_snr": _finite_float(snr),
            "movement_rms": movement,
            "update_to_param": _finite_float(movement / max(theta_rms, 1.0e-12)),
        }

    def maybe_emit(self, model, interval, checkpoint_digest=None):
        if not self.enabled or interval <= 0 or model.t % interval != 0:
            return None
        now = time.time()
        self._prepare_emas(model)
        if self._prev_theta is None:
            self._prev_theta = model.theta.detach().clone()

        bps, sps, measure_elapsed = self._throughput(model, now)
        loss = sum(self._loss) / max(1, len(self._loss))
        ce = sum(self._ce) / max(1, len(self._ce))
        self._loss.clear()
        self._ce.clear()

        params = {}
        for name, start, end in self._groups(model):
            params[name] = self._parameter_group(model, start, end)

        decay_layers = []
        hot_by_layer = {}
        state_rms_sq = 0.0
        state_count = 0
        state_active = 0.0
        for i, layer in enumerate(model.layers):
            sig = torch.sigmoid(layer.decay.detach())
            half_life = math.log(0.5) / torch.log(sig.clamp_min(1.0e-12))
            hl = half_life.cpu()
            ss = _vector_stats(layer.states)
            state_rms_sq += ss["rms"] ** 2 * layer.states.numel()
            state_count += layer.states.numel()
            state_active += ss["activity_ge_1e-3"] * layer.states.numel()
            nhot = int(model.tail.nhot(i)) if model.tail is not None else 0
            hot_by_layer[str(i)] = nhot
            decay_layers.append({
                "layer": i,
                "sigma_mean": _finite_float(sig.mean()),
                "sigma_min": _finite_float(sig.min()),
                "sigma_max": _finite_float(sig.max()),
                "half_life_mean_bytes": _finite_float(hl.mean()),
                "half_life_gt4": int((hl > 4).sum()),
                "half_life_gt8": int((hl > 8).sum()),
                "half_life_gt16": int((hl > 16).sum()),
                "half_life_gt32": int((hl > 32).sum()),
                "state": ss,
                "tail_members": nhot,
            })

        sig_all = torch.cat([torch.sigmoid(l.decay.detach()) for l in model.layers])
        hl_all = (math.log(0.5) /
                  torch.log(sig_all.clamp_min(1.0e-12))).cpu()
        # Ring/eligibility reductions happen only at this boundary, never in
        # the step hot path.  ring_w is the actual delayed-credit use mass.
        ring_sq, ring_count = 0.0, 0
        for layer in model.layers:
            w = layer.ring_w.detach()
            ring_sq += float(torch.sum(w * w).item()) * w.numel()
            ring_count += w.numel()
        ring_rms = math.sqrt(ring_sq / max(1, ring_count))
        tail_e_sq, tail_e_count = 0.0, 0
        if model.tail is not None:
            for e in model.tail.E.values():
                tail_e_sq += float(torch.sum(e * e).item()) * e.numel()
                tail_e_count += e.numel()
        tail_rms = math.sqrt(tail_e_sq / max(1, tail_e_count))
        rebase_at = int(getattr(model, "_tail_rebase_at", -1))
        crossed = rebase_at != self._last_rebase_at
        self._last_rebase_at = rebase_at

        event = {
            "event": "interval",
            "t": int(model.t),
            "bytes_seen": int(model.bytes_seen),
            "B": self.batch,
            "ring_k": int(model.ring_k),
            "tail_enabled": bool(model.tail_enabled),
            "loss": _finite_float(loss),
            "ce": _finite_float(ce),
            "throughput_B_per_s": bps,
            "steps_per_s": sps,
            "throughput_measurement_seconds": (
                _finite_float(measure_elapsed) if measure_elapsed is not None else None),
            "parameter_groups": params,
            "decay_layers": decay_layers,
            "hot_dimensions": int(sum(hot_by_layer.values())),
            "hot_by_layer": hot_by_layer,
            "half_life_gt4": int((hl_all > 4).sum()),
            "half_life_gt8": int((hl_all > 8).sum()),
            "half_life_gt16": int((hl_all > 16).sum()),
            "half_life_gt32": int((hl_all > 32).sum()),
            "state_rms": _finite_float(math.sqrt(state_rms_sq /
                                                  max(1, state_count))),
            "state_activity_ge_1e-3": _finite_float(
                state_active / max(1, state_count)),
            "ring_weight_rms": _finite_float(ring_rms),
            "tail_eligibility_rms": _finite_float(tail_rms),
            "tail_rebase_at": rebase_at,
            "crossed_rebase_boundary": bool(crossed),
            "checkpoint_digest": checkpoint_digest,
            "run_id": os.environ.get("TMT_RUN_ID"),
        }
        self._write(event)
        self._prev_theta = model.theta.detach().clone()
        return event
