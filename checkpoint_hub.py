#!/usr/bin/env python3
"""Checkpoint transport for the TMT GPU runs (PLAN-2026-09-23-GPU §2).

Design rule: **whoever pushes last wins; every push is atomic; a pull refuses
the past.** Checkpoints are 40-170 MB .npz, including Adam moments, recurrent
states and the tail EMA, so they go to a private HuggingFace Hub repo, not git.

Conventions (match numpy_port's Phase 0.3 tagged saves):

  * files are named ``<base>_t<step padded>.npz`` (e.g. ``tmt_t064747.npz``);
    a half-finished ``.tmp.npz`` write is never uploaded.
  * ``pull()`` lists the repo, picks the max-``t`` file, and downloads it only
    if ``t > local t`` -- a stale mirror can never win.
  * ``push()`` uploads one tagged file (resumable via huggingface_hub).

Environment:

  TMT_HUB_URL        repo id or full URL (default ``bakali1/tmt-checkpoints``)
  TMT_TOKEN          write token; falls back to the cached ``hf auth login``
  TMT_HUB_REPO_TYPE  ``model`` (default) or ``dataset``

CLI:

  python checkpoint_hub.py list
  python checkpoint_hub.py pull --dest tmt/tmt.npz
  python checkpoint_hub.py push tmt/tmt_t064747.npz
"""
import argparse
import os
import re
import shutil
import sys

DEFAULT_REPO = "bakali1/tmt-checkpoints"
TAG_RE = re.compile(r"_t(\d+)\.npz$")


def _repo():
    raw = os.environ.get("TMT_HUB_URL", DEFAULT_REPO).strip().rstrip("/")
    for prefix in ("https://huggingface.co/", "http://huggingface.co/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    # a browser URL ends with /tree/main; keep only owner/name
    parts = [p for p in raw.split("/") if p and p not in ("tree", "main", "datasets")]
    return "/".join(parts[:2]) if len(parts) >= 2 else raw


def _repo_type():
    return os.environ.get("TMT_HUB_REPO_TYPE", "model")


def _token():
    return os.environ.get("TMT_TOKEN") or None


def _api(token=None):
    from huggingface_hub import HfApi
    return HfApi(token=token or _token())


def tag_of(name):
    """Step number encoded in a filename, or None."""
    m = TAG_RE.search(os.path.basename(name))
    return int(m.group(1)) if m else None


def remote_tags(repo_id=None, token=None, repo_type=None):
    """[(step, filename)] for every tagged checkpoint in the repo, ascending."""
    api = _api(token)
    files = api.list_repo_files(repo_id or _repo(), repo_type=repo_type or _repo_type())
    out = [(tag_of(f), f) for f in files if f.endswith(".npz")]
    return sorted((t, f) for t, f in out if t is not None)


def local_t(path):
    """Best-known step for a local checkpoint: filename tag, else the npz ``t``."""
    if not os.path.exists(path):
        return -1
    t = tag_of(path)
    if t is not None:
        return t
    try:
        import numpy as np
        with np.load(path) as d:
            return int(d["t"]) if "t" in d else -1
    except Exception:
        return -1


def pull(dest, repo_id=None, token=None, repo_type=None, force=False):
    """Download the newest checkpoint if it is newer than ``dest``.

    Writes the tagged file next to ``dest`` and points ``dest`` at it (hard
    link, copy fallback). Returns the step pulled, or None if skipped.
    """
    from huggingface_hub import hf_hub_download
    repo_id = repo_id or _repo()
    repo_type = repo_type or _repo_type()
    tagged = remote_tags(repo_id, token, repo_type)
    if not tagged:
        print(f"[hub] {repo_id}: no tagged checkpoints yet")
        return None
    best_t, best_f = tagged[-1]
    dest = os.path.abspath(dest)
    have = local_t(dest)
    if not force and best_t <= have:
        print(f"[hub] {repo_id}: remote t={best_t} <= local t={have}, skipping pull")
        return None
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    got = hf_hub_download(repo_id, best_f, repo_type=repo_type,
                          token=token or _token(), local_dir=os.path.dirname(dest))
    base = os.path.splitext(os.path.basename(dest))[0]
    tagged_local = os.path.join(os.path.dirname(dest), f"{base}_t{best_t:06d}.npz")
    if os.path.abspath(got) != os.path.abspath(tagged_local):
        _atomic_copy(got, tagged_local)
    _atomic_link(tagged_local, dest)
    print(f"[hub] pulled t={best_t} ({best_f}) -> {dest}")
    return best_t


def push(path, repo_id=None, token=None, repo_type=None):
    """Upload one tagged checkpoint (never a ``.tmp``). Returns the step."""
    from huggingface_hub import HfApi
    if not os.path.exists(path):
        sys.exit(f"no such checkpoint: {path}")
    name = os.path.basename(path)
    if ".tmp" in name or not name.endswith(".npz"):
        sys.exit(f"refusing to push a non-atomic name: {name} "
                 f"(expected <base>_t<step>.npz)")
    repo_id = repo_id or _repo()
    repo_type = repo_type or _repo_type()
    HfApi(token=token or _token()).upload_file(
        path_or_fileobj=path, path_in_repo=name, repo_id=repo_id,
        repo_type=repo_type)
    t = tag_of(name)
    print(f"[hub] pushed {name} (t={t}) -> {repo_id}")
    return t


def _atomic_copy(src, dst):
    tmp = dst + ".part"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _atomic_link(src, dst):
    tmp = dst + ".lnk.part"
    try:
        if os.path.exists(tmp):
            os.unlink(tmp)
        os.link(src, tmp)
        os.replace(tmp, dst)
    except OSError:
        _atomic_copy(src, dst)


def push_latest(directory, base="tmt", **kw):
    """Push the newest local ``<base>_t*.npz`` in ``directory``."""
    import glob
    cands = []
    for f in glob.glob(os.path.join(directory, f"{base}_t*.npz")):
        t = tag_of(f)
        if t is not None:
            cands.append((t, f))
    if not cands:
        print(f"[hub] no local tagged checkpoints in {directory}")
        return None
    return push(max(cands)[1], **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("pull")
    p.add_argument("--dest", default="tmt.npz")
    p.add_argument("--force", action="store_true")
    q = sub.add_parser("push")
    q.add_argument("path")
    q = sub.add_parser("push-latest")
    q.add_argument("--dir", default=".")
    q.add_argument("--base", default="tmt")
    args = ap.parse_args()

    if args.cmd == "list":
        tags = remote_tags()
        print(f"[hub] {_repo()} ({_repo_type()}): {len(tags)} tagged checkpoint(s)")
        for t, f in tags:
            print(f"  t={t:<8} {f}")
    elif args.cmd == "pull":
        pull(args.dest, force=args.force)
    elif args.cmd == "push":
        push(args.path)
    elif args.cmd == "push-latest":
        push_latest(args.dir, args.base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
