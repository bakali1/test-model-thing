#!/usr/bin/env python3
"""Observation-only telemetry gates: SNR, shape, parity, and loud failure."""
import json
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_port  # noqa: E402
from telemetry import RunTelemetry, update_gradient_ema  # noqa: E402


def t_snr():
    ema = torch.zeros(8)
    ema2 = torch.zeros(8)
    mean, snr = update_gradient_ema(torch.zeros(8), ema, ema2)
    assert mean == 0.0 and snr == 0.0, (mean, snr)

    ema.zero_()
    ema2.zero_()
    g = torch.ones(8)
    mean, snr = update_gradient_ema(g, ema, ema2)
    # The first EMA sample includes zero history; repeated samples establish
    # the constant-direction SNR contract.
    for _ in range(100):
        mean, snr = update_gradient_ema(g, ema, ema2)
    assert abs(snr - 1.0) < 1.0e-4, snr
    assert abs(mean - g.abs().mean()) < 1.0e-6, (mean, g.abs().mean())
    print("  zero gradient SNR=0; constant direction SNR=1: PASS")
    return True


def _run(path, telemetry_every=0):
    m = gpu_port.GpuModel(device="cpu", dim=16, layers=2, seed=0,
                         ring_k=18, ring_credit="full")
    m.new_states(4)
    tr = None
    if telemetry_every:
        tr = RunTelemetry(path, 4, measure_after=1)
    g = torch.Generator().manual_seed(7)
    for _ in range(4):
        c = torch.randint(0, 256, (4,), generator=g)
        n = torch.randint(0, 256, (4,), generator=g)
        e = torch.zeros(4)
        loss = m.step_batch(c, n, e)
        if tr:
            tr.observe(loss, m.last_ce)
            if m.t == 1:
                tr.mark_measurement(m)
            tr.maybe_emit(m, telemetry_every)
    return m


def t_shape_and_parity():
    d = tempfile.mkdtemp()
    try:
        with_t = _run(os.path.join(d, "telemetry.jsonl"), telemetry_every=2)
        without_t = _run(os.path.join(d, "unused.jsonl"))
        assert torch.equal(with_t.theta, without_t.theta), "theta trajectory changed"
        for a, b in zip(with_t.layers, without_t.layers):
            assert torch.equal(a.states, b.states), "state trajectory changed"
            assert torch.equal(a.decaytrace, b.decaytrace), "trace changed"

        rows = [json.loads(line) for line in
                open(os.path.join(d, "telemetry.jsonl"), encoding="utf-8")]
        assert [r["t"] for r in rows] == [2, 4], [r["t"] for r in rows]
        for row in rows:
            assert row["B"] == 4 and row["ring_k"] == 18, row
            assert math.isfinite(row["loss"]) and math.isfinite(row["ce"]), row
            assert "shared_trunk" in row["parameter_groups"], row
            assert "layer0.decay" in row["parameter_groups"], row
            assert len(row["decay_layers"]) == 2, row
            assert "throughput_B_per_s" in row, row
        assert rows[0]["throughput_B_per_s"] > 0, rows[0]
        assert rows[1]["throughput_B_per_s"] > 0, rows[1]
        print("  fixed shape + telemetry-on/off trajectory parity: PASS")
        return True
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def t_checkpoint_reload_identity():
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "k18.npz")
        m = gpu_port.GpuModel(device="cpu", dim=16, layers=2, seed=0,
                             ring_k=18, ring_credit="full")
        m.new_states(4)
        m.t = 7
        m.save(path)
        probe = gpu_port.GpuModel(device="cpu", dim=16, layers=2, seed=99,
                                  ring_k=18, ring_credit="full")
        probe.new_states(4)
        probe.load(path)
        assert (probe.t, probe.B, probe.ring_k) == (7, 4, 18)
        print("  explicit K=18 checkpoint reload preserves t/B/K: PASS")
        return True
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def t_nonfinite_is_loud():
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "bad.jsonl")
        tr = RunTelemetry(path, 4)
        m = gpu_port.GpuModel(device="cpu", dim=16, layers=1)
        m.new_states(4)
        m.t = 7
        try:
            tr.nonfinite_loss(m, float("nan"), float("inf"))
            raise AssertionError("non-finite loss did not raise")
        except FloatingPointError:
            pass
        row = json.loads(open(path, encoding="utf-8").readline())
        assert row["event"] == "nonfinite_loss" and row["loss"] == "nan"
        print("  non-finite loss writes a loud event and aborts: PASS")
        return True
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def main():
    ok = True
    for fn in (t_snr, t_shape_and_parity, t_checkpoint_reload_identity,
               t_nonfinite_is_loud):
        try:
            ok &= bool(fn())
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")
            ok = False
    print("\nCHECK_TELEMETRY:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
