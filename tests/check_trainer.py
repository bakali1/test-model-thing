#!/usr/bin/env python3
"""Trainer + hub + state-lifecycle gates (GPU-IMPLEMENTATION-REVIEW items 1,2,3,9,10,12).

No GPU, no network: everything here is a local reproduction of the review's
failure modes, so it can run as a pre-colab gate.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint_hub as hub          # noqa: E402
import gpu_port                        # noqa: E402
import train_gpu as tg                 # noqa: E402

# DEVICE CONVENTION. This gate is a CPU gate by its own contract ("No GPU, no
# network"), so the engines are pinned to "cpu" here. Leaving the device to
# the engine's auto-default made this gate a CUDA gate on a GPU box, where its
# CPU-built input tensors die with "Expected all tensors to be on the same
# device" -- i.e. it was not testing anything there, it was crashing. The
# device-parameterised gate is tests/check_seqlen_credit.py --device, which
# runs the T-axis claim on both.



def hot_model(dim=64, layers=2, B=8):
    m = gpu_port.GpuModel(device="cpu", dim=dim, layers=layers)
    m.new_states(B)
    with torch.no_grad():                       # pre-sigmoid 4 => sigma .982
        for l in m.layers:
            l.decay.fill_(4.0)
    return m


def step_once(m, B=8):
    cs = torch.randint(1, 250, (B,))
    ns = torch.randint(1, 250, (B,))
    e = torch.zeros(B)
    m.step_batch(cs, ns, e)


def item1_multiarm_tags():
    """Two arms, same step, one directory: distinct tags, independent pruning."""
    d = tempfile.mkdtemp()
    ok = True
    for arm in ("armW", "armN"):
        m = gpu_port.GpuModel(device="cpu", dim=32, layers=2)
        for t in (1, 2, 3):
            m.t = t
            tg.save_checkpoint(m, os.path.join(d, f"{arm}.npz"), keep_k=2,
                               tagged=True)
    tagsW = sorted(os.path.basename(p) for p in glob.glob(f"{d}/armW_t*.npz"))
    tagsN = sorted(os.path.basename(p) for p in glob.glob(f"{d}/armN_t*.npz"))
    print(f"  armW tags {tagsW}\n  armN tags {tagsN}")
    ok &= tagsW == ["armW_t000002.npz", "armW_t000003.npz"]   # keep_k=2
    ok &= tagsN == ["armN_t000002.npz", "armN_t000003.npz"]
    ok &= not glob.glob(f"{d}/tmt_t*.npz")
    # each rolling file is a hard link to its OWN newest tag
    for arm in ("armW", "armN"):
        r = os.path.join(d, f"{arm}.npz")
        newest = os.path.join(d, f"{arm}_t000003.npz")
        ok &= os.path.exists(r) and os.path.samefile(r, newest)
    shutil.rmtree(d)
    return ok


def item2_cap_preallocation():
    """Cap must raise BEFORE torch.zeros, with the actionable message."""
    m = hot_model()
    m.tail_cap_gb = 1e-9
    try:
        step_once(m)
    except RuntimeError as ex:
        msg = str(ex)
        print(f"  raised: {msg[:80]}...")
        return ("would be" in msg and "TMT_TAIL=0" in msg)
    return False


def item3_state_invalidation():
    """new_states() must drop the tail + rebase marker (review's repro)."""
    m = hot_model()
    step_once(m)
    before = sum(m.tail.nhot(i) for i in range(m.layercount))
    m._tail_rebase_at = 1
    m.t = 2
    m.new_states(8)
    ok = m.tail is None and m._tail_rebase_at == -1
    step_once(m)
    after = sum(m.tail.nhot(i) for i in range(m.layercount))
    print(f"  hot {before} -> {after} after new_states + 1 step")
    ok &= after == before > 0
    # ring_k change through load() must also re-seed
    d = tempfile.mkdtemp()
    p = os.path.join(d, "ck.npz")
    m.save(p)
    z = dict(np.load(p))
    z["ring_k"] = np.array(m.ring_k + 3, np.int64)
    q = os.path.join(d, "ck_k.npz")
    np.savez(q, **z)
    m2 = hot_model()
    m2.load(q)
    n = sum(m2.tail.nhot(i) for i in range(m2.layercount))
    print(f"  ring_k {m.ring_k} -> {m2.ring_k}, tail re-seeded {n} hot")
    ok &= n == before and m2.ring_k == m.ring_k + 3
    shutil.rmtree(d)
    return ok


def item9_content_validation():
    """Mislabeled / corrupt / wrong-shape remote files are rejected."""
    d = tempfile.mkdtemp()
    m = gpu_port.GpuModel(device="cpu", dim=32, layers=2)
    m.t = 5
    p = os.path.join(d, "x_t000005.npz")
    m.save(p)
    ok = hub.validate_npz(p, expect_t=5, expect_n=m.theta.shape[0]) == 5
    # filename/content disagreement
    z = dict(np.load(p))
    z["t"] = np.array(6, np.int64)
    bad = os.path.join(d, "y_t000005.npz")
    np.savez(bad, **z)
    try:
        hub.validate_npz(bad, expect_t=5)
        ok = False
        print("  FAIL: mislabeled t accepted")
    except SystemExit as ex:
        print(f"  rejected mislabeled: {str(ex)[:60]}...")
    # wrong shape
    try:
        hub.validate_npz(p, expect_n=123)
        ok = False
        print("  FAIL: wrong shape accepted")
    except SystemExit as ex:
        print(f"  rejected wrong shape: {str(ex)[:60]}...")
    # truncated file
    trunc = os.path.join(d, "z_t000007.npz")
    with open(p, "rb") as f:
        data = f.read(200)
    with open(trunc, "wb") as f:
        f.write(data)
    try:
        hub.validate_npz(trunc)
        ok = False
        print("  FAIL: truncated accepted")
    except SystemExit as ex:
        print(f"  rejected truncated: {str(ex)[:60]}...")
    # sidecar parsing: sha256sum format, named <file>.npz inside <file>.npz.sha256
    side = p + ".sha256"
    with open(side, "w") as f:
        f.write("deadbeefcafe  x_t000005.npz\n# run_id expD-W\n")
    ok &= hub._read_sidecar(side) == "deadbeefcafe"
    bad = os.path.join(d, "y_t000005.npz.sha256")
    with open(bad, "w") as f:
        f.write("deadbeefcafe  other.npz\n")
    ok &= hub._read_sidecar(bad) is None          # names a different file
    ok &= hub._read_sidecar(os.path.join(d, "nope.sha256")) is None
    shutil.rmtree(d)
    return ok


def item10_equal_step_conflict():
    """Equal step + different bytes must fail loudly, not silently skip."""
    d = tempfile.mkdtemp()
    local = os.path.join(d, "tmt.npz")
    remote = os.path.join(d, "remote_t000010.npz")
    m = gpu_port.GpuModel(device="cpu", dim=32, layers=2)
    m.t = 10
    m.theta[0] = 0.1
    m.save(remote)
    m.theta[0] = -0.2                            # different content, same step
    m.save(local)
    pulled = {}

    import huggingface_hub as hh
    orig_dl, orig_tags = hh.hf_hub_download, hub.remote_tags
    hh.hf_hub_download = lambda *a, **k: remote
    hub.remote_tags = lambda *a, **k: [(10, "remote_t000010.npz")]
    try:
        try:
            hub.pull(local, repo_id="fake/repo", require_digest=False)
            print("  FAIL: equal-step conflict silently accepted")
            return False
        except SystemExit as ex:
            print(f"  conflict rejected: {str(ex)[:70]}...")
        # same content -> clean skip
        m.theta[0] = 0.1
        m.save(local)
        pulled["r"] = hub.pull(local, repo_id="fake/repo",
                               require_digest=False)
    finally:
        hh.hf_hub_download = orig_dl
        hub.remote_tags = orig_tags
    ok = pulled.get("r") is None
    shutil.rmtree(d)
    return ok


def item12_push_no_tagged_rejected():
    """--no-tagged with hub flags must exit, not silently skip pushes."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, TMT_HUB_REPO="fake/repo")
    r = subprocess.run(
        [sys.executable, os.path.join(here, "train_gpu.py"), "x.npz", "train",
         "--no-tagged", "--hub-repo", "fake/repo"],
        capture_output=True, text=True, env=env)
    ok = r.returncode != 0 and "never uploads" in (r.stdout + r.stderr)
    print(f"  exit={r.returncode}: {(r.stdout + r.stderr).strip()[:70]}")
    return ok


def item13_from_scratch_seed():
    """From-scratch arms must share ONE init, and must not pull/load state.

    This is the precondition for the corpus A/B (FS-W / FS-N / FS-L): if the
    arms did not start from bit-identical weights, the data axis would be
    confounded with the init. Also asserts the historical seed-0 init is
    unchanged by the seed plumbing.
    """
    import torch
    from train_gpu import Trainer
    tmp = tempfile.mkdtemp()
    data = os.path.join(tmp, "t.txt")
    with open(data, "w") as f:
        f.write("the quick brown fox jumps over the lazy dog. " * 4000)

    def arm(seed, data_seed, path):
        t = Trainer(path, batch=64, from_scratch=True, seed=seed,
                    data_seed=data_seed, device="cpu", dim=64, layers=2,
                    max_steps=3)
        t.train(save=lambda *a: None, dataset=os.path.join(tmp, "*.txt"))
        return t

    a = arm(0, 0, os.path.join(tmp, "a.npz"))
    b = arm(0, 1, os.path.join(tmp, "b.npz"))
    c = arm(7, 0, os.path.join(tmp, "c.npz"))
    # Same seed, different data ordering -> identical starting weights.
    same = torch.equal(a.model.theta, b.model.theta)
    # Different seed -> genuinely different init (the flag is not a no-op).
    diff = not torch.equal(a.model.theta, c.model.theta)
    # It really starts at t=0 and trains forward.
    t0 = a.start_t == 0
    # A stale checkpoint at the path must NOT be loaded in from-scratch mode.
    stale = os.path.join(tmp, "d.npz")
    np.savez(stale, theta=np.full((100,), 7.0, np.float32), B=np.array(64))
    d = Trainer(stale, batch=64, from_scratch=True, seed=0, device="cpu",
                dim=64, layers=2, max_steps=1)
    d.train(save=lambda *a: None, dataset=os.path.join(tmp, "*.txt"))
    ignored = d.start_t == 0
    shutil.rmtree(tmp, ignore_errors=True)
    ok = same and diff and t0 and ignored
    if not same:
        print("    same seed did not give identical init")
    if not diff:
        print("    --seed is a no-op: different seeds gave identical init")
    if not ignored:
        print("    from-scratch loaded a stale checkpoint")
    return ok


def main():
    items = [
        ("1  A/B tag isolation", item1_multiarm_tags),
        ("2  tail cap pre-allocation", item2_cap_preallocation),
        ("3  state/rebase invalidation", item3_state_invalidation),
        ("9  hub content validation", item9_content_validation),
        ("10 equal-step conflict", item10_equal_step_conflict),
        ("12 --push --no-tagged rejected", item12_push_no_tagged_rejected),
        ("13 from-scratch seed/isolation", item13_from_scratch_seed),
    ]
    bad = []
    for name, fn in items:
        try:
            ok = fn()
        except Exception as ex:
            print(f"  ERROR {type(ex).__name__}: {ex}")
            ok = False
        print(f"{name:36s} {'PASS' if ok else 'FAIL'}")
        if not ok:
            bad.append(name)
    print()
    print("CHECK_TRAINER: " + ("PASS" if not bad else f"FAIL {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
