#!/usr/bin/env python3
"""Checkpoint transport for the TMT GPU runs (PLAN-2026-09-23-GPU §2).

Design rule: **content-verified transport; a pull refuses the past and refuses
to guess at equal steps.** Checkpoints are 40-170 MB .npz, including Adam
moments, recurrent states and the tail EMA, so they go to a private HuggingFace
Hub repo, not git.

Conventions (match numpy_port's Phase 0.3 tagged saves):

  * files are named ``<base>_t<step padded>.npz`` (e.g. ``tmt_t064747.npz``);
    a half-finished ``.tmp.npz`` write is never uploaded. The ``<base>`` is the
    rolling checkpoint's basename, so A/B arms in one directory never collide.
  * every push validates content (readable npz, internal ``t`` == filename
    ``t``) and uploads a ``<name>.sha256`` sidecar with the digest and the
    optional ``$TMT_RUN_ID``; every pull re-validates and checks the sidecar.
  * ``pull()`` picks the max-``t`` file and downloads it only if ``t > local
    t``. An equal-step remote with different bytes is a two-writers conflict
    and fails loudly (pass ``force=True`` to override).

Environment:

  TMT_HUB_URL        repo id or full URL (default ``bakali1/tmt-checkpoints``)
  TMT_TOKEN          write token; falls back to the cached ``hf auth login``
  TMT_HUB_REPO_TYPE  ``model`` (default) or ``dataset``
  TMT_RUN_ID         optional arm/run label recorded in push sidecars

CLI:

  python checkpoint_hub.py list
  python checkpoint_hub.py pull --dest tmt/tmt.npz
  python checkpoint_hub.py push tmt/tmt_t064747.npz
  python checkpoint_hub.py prune-remote --keep 8 [--yes]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

DEFAULT_REPO = "bakali1/tmt-checkpoints"
TAG_RE = re.compile(r"_t(\d+)\.npz$")
SIDECAR_RE = re.compile(r"_t(\d+)\.npz\.sha256$")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def validate_npz(path, expect_t=None, expect_n=None):
    """Content validation for a checkpoint file.

    Rejects unreadable/corrupt npz, filename/internal-t disagreement, and
    (when given) a parameter count that disagrees with the receiving model.
    Returns the internal t. Raises SystemExit with the reason otherwise.
    A mislabeled or truncated remote file can never become the loaded truth.
    """
    import numpy as np
    name_t = tag_of(path)
    if expect_t is not None and name_t is not None and name_t != expect_t:
        sys.exit(f"checkpoint {path}: filename says t={name_t} but caller "
                 f"expects t={expect_t}")
    try:
        with np.load(path) as d:
            if "t" not in d:
                sys.exit(f"checkpoint {path}: no internal step 't'")
            t = int(np.ravel(d["t"])[0])
            if name_t is not None and t != name_t:
                sys.exit(f"checkpoint {path}: filename says t={name_t} but "
                         f"content says t={t} — refusing mislabeled file")
            if "m" not in d or "v" not in d:
                sys.exit(f"checkpoint {path}: missing Adam moments m/v")
            n = int(d["m"].shape[0])
            if expect_n is not None and n != expect_n:
                sys.exit(f"checkpoint {path}: holds {n:,} params but the "
                         f"model has {expect_n:,} — refusing wrong-shape file")
            # touch one big array so truncation is caught, not just headers
            float(d["m"][:1].sum())
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"checkpoint {path}: unreadable ({type(e).__name__}: {e})")
    return t


def _read_sidecar(path):
    """Parse a '<file>.sha256' sidecar, or None.

    Written in sha256sum format ("<hex>  <name>", optional "# run_id ...").
    The file is named "<name>.sha256" and names the npz inside, so compare
    against the basename with the .sha256 suffix stripped.
    """
    try:
        with open(path) as f:
            txt = f.read().split()
        named = os.path.basename(path)
        if named.endswith(".sha256"):
            named = named[:-len(".sha256")]
        if len(txt) >= 2 and txt[1].lstrip("*") == named:
            return txt[0]
    except OSError:
        pass
    return None


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


def pull(dest, repo_id=None, token=None, repo_type=None, force=False,
         expect_n=None):
    """Download the newest checkpoint if it is newer than ``dest``.

    Writes the tagged file next to ``dest`` and points ``dest`` at it (hard
    link, copy fallback). Returns the step pulled, or None if skipped.

    Content rules: the chosen remote is validated (readable, internal t ==
    filename t, sidecar hash match when present, param count vs. the model)
    BEFORE it can replace ``dest``. An equal-step remote with different
    content is a conflict and fails loudly unless ``force``.
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
    if not force and best_t < have:
        print(f"[hub] {repo_id}: remote t={best_t} < local t={have}, skipping pull")
        return None
    if not force and best_t == have:
        # Equal step: compare content. Same bytes = nothing to do; different
        # bytes = two writers at the same step (concurrent arms/platforms) and
        # we refuse to guess which is truth.
        if os.path.exists(dest):
            try:
                from huggingface_hub import hf_hub_download as _dl
                got = _dl(repo_id, best_f, repo_type=repo_type,
                          token=token or _token())
                if sha256_file(got) == sha256_file(dest):
                    print(f"[hub] {repo_id}: t={best_t} already present "
                          f"(identical content), skipping")
                    return None
                sys.exit(
                    f"[hub] equal-step conflict at t={best_t}: remote "
                    f"{best_f} differs from local {os.path.basename(dest)}. "
                    f"Two writers pushed the same step (e.g. concurrent "
                    f"arms/platforms). Refusing to guess: rename one arm's "
                    f"rolling file / re-tag, or re-run with force=True.")
            except SystemExit:
                raise
            except Exception as e:
                sys.exit(f"[hub] equal-step check at t={best_t} failed "
                         f"({type(e).__name__}: {e}); re-run with force=True "
                         f"to override")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    got = hf_hub_download(repo_id, best_f, repo_type=repo_type,
                          token=token or _token(), local_dir=os.path.dirname(dest))
    validate_npz(got, expect_t=best_t, expect_n=expect_n)
    # Fetch the sidecar (105 bytes) and verify the digest. It is NOT part of
    # the npz download, so it must be requested explicitly -- otherwise this
    # check silently never runs.
    side_path = None
    try:
        side_path = hf_hub_download(repo_id, best_f + ".sha256",
                                    repo_type=repo_type, token=token or _token(),
                                    local_dir=os.path.dirname(dest))
    except Exception as e:
        print(f"[hub] note: no sidecar for {best_f} ({type(e).__name__}); "
              f"content validated, digest NOT verified")
    remote_sha = _read_sidecar(side_path) if side_path else None
    if remote_sha is None and side_path:
        print(f"[hub] note: {best_f}.sha256 unreadable; digest NOT verified")
    if remote_sha and remote_sha != sha256_file(got):
        sys.exit(f"[hub] {best_f}: content hash does not match its sidecar "
                 f"— refusing corrupted download")
    base = os.path.splitext(os.path.basename(dest))[0]
    tagged_local = os.path.join(os.path.dirname(dest), f"{base}_t{best_t:06d}.npz")
    if os.path.abspath(got) != os.path.abspath(tagged_local):
        _atomic_copy(got, tagged_local)
    _atomic_link(tagged_local, dest)
    print(f"[hub] pulled t={best_t} ({best_f}) -> {dest}")
    return best_t


def push(path, repo_id=None, token=None, repo_type=None, run_id=None):
    """Upload one tagged checkpoint (never a ``.tmp``). Returns the step.

    Validates content first (readable, internal t == filename t) and uploads a
    ``.sha256`` sidecar beside it, so pulls can verify what they received.
    ``run_id`` (default $TMT_RUN_ID) is recorded in the sidecar for
    multi-arm provenance.
    """
    from huggingface_hub import HfApi
    if not os.path.exists(path):
        sys.exit(f"no such checkpoint: {path}")
    name = os.path.basename(path)
    if ".tmp" in name or not name.endswith(".npz"):
        sys.exit(f"refusing to push a non-atomic name: {name} "
                 f"(expected <base>_t<step>.npz)")
    t = validate_npz(path, expect_t=tag_of(path))
    repo_id = repo_id or _repo()
    repo_type = repo_type or _repo_type()
    api = HfApi(token=token or _token())
    digest = sha256_file(path)
    side_local = path + ".sha256"
    run = run_id or os.environ.get("TMT_RUN_ID")
    with open(side_local, "w") as f:
        f.write(f"{digest}  {name}\n")
        if run:
            f.write(f"# run_id {run}\n")
    api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo_id,
                    repo_type=repo_type)
    api.upload_file(path_or_fileobj=side_local, path_in_repo=name + ".sha256",
                    repo_id=repo_id, repo_type=repo_type)
    print(f"[hub] pushed {name} (t={t}, sha={digest[:12]}"
          f"{f', run={run}' if run else ''}) -> {repo_id}")
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


def prune_remote(keep, repo_id=None, token=None, repo_type=None, dry_run=True):
    """Keep the newest ``keep`` tagged checkpoints remotely; delete the rest.

    Never runs implicitly. Requires --yes to actually delete, and prints every
    victim first. Checkpoints that may anchor an A/B result should be copied
    out of the repo before pruning.
    """
    from huggingface_hub import HfApi
    repo_id = repo_id or _repo()
    repo_type = repo_type or _repo_type()
    tagged = remote_tags(repo_id, token, repo_type)
    victims = [f for _t, f in tagged[:-keep]] if keep > 0 else [f for _t, f in tagged]
    if not victims:
        print(f"[hub] nothing to prune ({len(tagged)} tagged, keep={keep})")
        return []
    print(f"[hub] would delete {len(victims)} of {len(tagged)} remote tags "
          f"(keep newest {keep}):")
    for v in victims:
        print(f"  - {v}")
    if dry_run:
        print("[hub] dry run; pass --yes to delete")
        return victims
    api = HfApi(token=token or _token())
    for v in victims:
        api.delete_file(v, repo_id=repo_id, repo_type=repo_type)
    print(f"[hub] deleted {len(victims)} remote tag(s)")
    return victims


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
    q.add_argument("--run-id", default=None)
    r = sub.add_parser("prune-remote")
    r.add_argument("--keep", type=int, default=8)
    r.add_argument("--yes", action="store_true", help="actually delete")
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
        push_latest(args.dir, args.base, run_id=args.run_id)
    elif args.cmd == "prune-remote":
        prune_remote(args.keep, dry_run=not args.yes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
