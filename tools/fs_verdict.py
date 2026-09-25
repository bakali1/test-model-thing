#!/usr/bin/env python3
"""Score the from-scratch arms side by side and render the verdict.

    .venv/bin/python fs_verdict.py tmt_fs-w_t*.npz tmt_fs-n_t*.npz tmt_fs-l_t*.npz
    .venv/bin/python fs_verdict.py --pull            # fetch the tags from the hub

Every arm is scored with IDENTICAL eval settings (same slices, same byte
counts, same seed), because the 0.03 CE bar is only meaningful if the ruler
was applied the same way to each.

The comparison is arm-vs-arm, NOT arm-vs-incumbent: a from-scratch run has the
incumbent's entire 64,747 steps of history missing, so an absolute comparison
to tmt_t094747 at equal wall clock would be unfair. What the arms answer is
whether the CORPUS changes the trajectory, which is the question continuation
A/Bs could not ask.

Verdict rules (PLAN §6 / RESULTS.md ruler):
  * bar = 0.03 CE nats/byte on the 3-file held-out mean
  * in-domain separation uses the pooled SE across arms
  * a winner must clear the bar on the in-domain mean AND not be worse on
    cross-domain novels by more than the bar
"""
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, HERE)

BAR = float(os.environ.get("TMT_FS_BAR", "0.03"))
HUB_REPO = os.environ.get("TMT_HUB_REPO", "bakali1/tmt-checkpoints")
ARMS = ("W", "N", "L")
# Identical for every arm. --slices/--bytes fix the ruler's sample size, and
# they must match whatever produced the incumbent baseline in RESULTS.md
# (the eval.py default: 3 slices x 20,000 B). TMT_FS_EVAL_ARGS exists only to
# smoke-test THIS script quickly; never use it for a real verdict.
EVAL_ARGS = os.environ.get("TMT_FS_EVAL_ARGS", "").split() or [
    "--slices", "3", "--bytes", "20000", "--xd-bytes", "8000"]


def pull_exact(repo, filename):
    """Download ONE exact tag. Deliberately NOT checkpoint_hub.pull().

    pull(dest) is a "get the newest remote" API: it ignores the name you give
    it and downloads whatever is latest. Calling it for an arm tag silently
    fetched tmt_t094747.npz (the incumbent) into fs_fs-w_t060000.npz, which
    would have compared the incumbent against itself. Verify by filename here.
    """
    import hashlib
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # 28x faster on this host
    path = hf_hub_download(repo_id=repo, filename=filename, repo_type="model")
    # hf_hub_download returns a snapshot path whose RELATIVE symlink points
    # into the cache's blob store. Moving it elsewhere leaves a dangling link
    # (np.load then fails with FileNotFoundError on a path that clearly
    # exists), so resolve to the real blob and copy that.
    real = os.path.realpath(path)
    side = filename + ".sha256"
    try:
        s = hf_hub_download(repo_id=repo, filename=side, repo_type="model")
        want = open(s).read().split()[0].lower()
        h = hashlib.sha256()
        with open(real, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        got = h.hexdigest()
        if got != want:
            sys.exit(f"digest mismatch for {filename}: {got} != {want}")
        print(f"    digest verified {got[:12]}")
    except Exception as e:
        print(f"    [warn] no sidecar for {filename} ({type(e).__name__})")
    return real


def arm_tags(arm):
    """Local tags for an arm. Tolerates the old 'fs_fs-w_*' naming (arms run
    before the double-prefix fix) alongside the current 'fs_w_*'."""
    pats = [f"fs_{arm.lower()}_t*.npz", f"fs_fs-{arm.lower()}_t*.npz"]
    out = []
    for p in pats:
        out.extend(glob.glob(os.path.join(HERE, p)))
    return sorted(set(out))


def arm_matches(name, arm):
    n = name.lower()
    return f"fs_{arm.lower()}_" in n or f"fs_fs-{arm.lower()}_" in n


def pick_at_step(tags, target):
    """Choose the newest tag at or before `target` steps.

    The arms are NOT byte-matched by construction: each derives its own step
    count from its own in-session calibration, and those rates differ by up to
    1.6x (FS-W calibrated 20,500 B/s -> 60,058 steps; FS-N 34,000 B/s ->
    99,603). Comparing final-to-final would confound corpus with compute, so
    the verdict is taken at a COMMON step count, where every arm has a tag.
    """
    best = None
    for t in tags:
        try:
            ts = int(os.path.basename(t).split("_t")[-1].split(".")[0])
        except ValueError:
            continue
        if ts <= target and (best is None or ts > best[0]):
            best = (ts, t)
    return best[1] if best else None


def pull_tags(at_step=None):
    """Fetch the final tagged checkpoint for each arm from the hub."""
    import checkpoint_hub as hub
    for arm in ARMS:
        tags = hub.remote_tags(HUB_REPO, prefix=f"fs_{arm.lower()}")
        if not tags:
            print(f"  [warn] no remote tag for FS-{arm}")
            continue
        newest = tags[-1]
        out = os.path.join(HERE, newest)
        print(f"  pulling FS-{arm}: {newest} ...", flush=True)
        try:
            hub.pull(newest, repo_id=HUB_REPO,
                     expect_n=4481793, require_digest=True)
        except SystemExit as e:
            print(f"  [warn] pull refused: {e}")
    return None


def score(path):
    """Run eval.py on one checkpoint and pull the ruler numbers out.

    Caches the raw transcript next to the checkpoint so re-rendering the table
    (or fixing a regex) costs nothing: one CPU eval at the ruler setting is
    ~10 minutes.
    """
    # The eval args are part of the cache key: without them a transcript
    # produced at --slices 1 --bytes 2000 gets served for a ruler-grade
    # request, silently reporting the wrong numbers under the right heading.
    key = hashlib.sha1(" ".join(EVAL_ARGS).encode()).hexdigest()[:8]
    # Cached transcripts live in cache/ so the repo root stays readable; they
    # are ~40 KB each and cost ~10 min of CPU to regenerate.
    cache_dir = os.path.join(HERE, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir,
                         f"{os.path.basename(path)}.evaltxt.{key}")
    if os.path.exists(cache) and os.environ.get("TMT_FS_NO_CACHE") != "1":
        out = open(cache).read()
    else:
        p = subprocess.run([sys.executable, os.path.join(HERE, "eval.py"),
                            path] + EVAL_ARGS,
                           capture_output=True, text=True, cwd=HERE)
        if p.returncode != 0:
            print(f"  [eval.py rc={p.returncode} on {os.path.basename(path)}]")
            print((p.stdout[-1500:] or "") + (p.stderr[-1500:] or ""))
            return None
        out = p.stdout
        with open(cache, "w") as f:
            f.write(out)
    m = re.search(r"mean\s+([0-9.]+) \+/- ([0-9.]+)", out)
    ind = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r"cross-domain CE.*?mean\s+([0-9.]+) \+/- ([0-9.]+)",
                  out, re.S)
    xd = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r"shift-distribution CE.*?mean\s+([0-9.]+) \+/- ([0-9.]+)",
                  out, re.S)
    sh = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r"hot dims \(sigma\^K>1%\):\s*(\d+)", out)
    hot = int(m.group(1)) if m else None
    m = re.search(r"half-life\s+mean=([0-9.]+) B", out)
    hl = float(m.group(1)) if m else None
    m = re.search(r"steps \(t\)\s*:\s*([\d,]+)\s+bytes seen\s*([\d,]+)", out)
    # The ruler prints thousands separators ("60,000"), which a plain \d+
    # silently fails to match and leaves the table as None.
    t = int(m.group(1).replace(",", "")) if m else None
    seen = int(m.group(2).replace(",", "")) if m else None
    return dict(ind=ind, xd=xd, shift=sh, hot=hot, hl=hl, t=t, seen=seen)


def fmt_pair(p):
    if p is None or p[0] is None:
        return "n/a"
    return f"{p[0]:.4f} +/- {p[1]:.4f}"


def num(v, spec="{:>9,}", scale=1.0):
    """Format a possibly-missing number; a None must never crash the table."""
    return spec.format("n/a") if v is None else spec.format(int(v * scale))


def main():
    argv = sys.argv[1:]
    # Parse --at-step ONCE, before any block consumes argv, so both the pull
    # and the selection use the same target.
    at_step = None
    if "--at-step" in argv:
        i = argv.index("--at-step")
        val = argv[i + 1] if i + 1 < len(argv) else ""
        at_step = int(val.split("=")[1] if "=" in val else val)
        del argv[i:i + 2]

    if argv and argv[0] == "--pull":
        import checkpoint_hub as hub
        argv = argv[1:]
        for arm in ARMS:
            # Union of local and remote tags. Using local-only when any file
            # happens to exist means a second run silently re-picks the
            # already-downloaded step instead of the one requested: --at-step
            # 86000 returned the 60,000 tag because that file was local.
            tags = set(arm_tags(arm))
            try:
                remote = [fn for _st, fn in hub.remote_tags(HUB_REPO)
                          if arm_matches(fn, arm) and fn.endswith(".npz")]
            except Exception as e:
                print(f"  [warn] remote tag list failed for FS-{arm}: {e}")
                remote = []
            have = {os.path.basename(t) for t in tags}
            tags = sorted(tags | {r for r in remote
                                  if os.path.basename(r) not in have})
            if not tags:
                print(f"  [warn] no tag for FS-{arm}")
                continue
            if at_step:
                chosen = pick_at_step(tags, at_step)
                if not chosen:
                    print(f"  [warn] FS-{arm} has no tag at or before "
                          f"t={at_step:,}")
                    continue
                tags = [chosen]
            newest = tags[-1]
            print(f"  pulling FS-{arm}: {os.path.basename(newest)} ...",
                  flush=True)
            try:
                got = pull_exact(HUB_REPO, os.path.basename(newest))
                dst = os.path.join(HERE, os.path.basename(newest))
                if os.path.abspath(got) != os.path.abspath(dst):
                    if os.path.lexists(dst):
                        os.remove(dst)
                    shutil.copy2(got, dst)   # real file, not a cache symlink
            except SystemExit:
                raise
            except Exception as e:
                print(f"  [warn] pull failed: {type(e).__name__}: {e}")
    if not argv:
        argv = sorted(glob.glob(os.path.join(HERE, "fs_*_t*.npz")))
    paths = []
    for a in argv:
        paths.extend(sorted(glob.glob(a)))
    if at_step:
        paths = []
        for a in ARMS:
            # Same union rule as the pull, so the selection sees every tag
            # that is actually available, not just the ones on disk.
            tags = set(arm_tags(a))
            try:
                import checkpoint_hub as _h
                remote = [fn for _st, fn in _h.remote_tags(HUB_REPO)
                          if arm_matches(fn, a) and fn.endswith(".npz")]
                have = {os.path.basename(t) for t in tags}
                tags = sorted(tags | {r for r in remote
                                      if os.path.basename(r) not in have})
            except Exception:
                pass
            chosen = pick_at_step(tags, at_step) if tags else None
            if chosen:
                print(f"  FS-{a}: matched tag {os.path.basename(chosen)}")
                paths.append(chosen)
            else:
                print(f"  FS-{a}: NO tag at or before t={at_step:,} -- "
                      f"this arm is excluded from the matched comparison")
    if not paths:
        sys.exit("no checkpoints given and none found; pass paths or --pull")

    # LOUD mismatch guard. pick_at_step returns the newest tag at or before the
    # target, so an arm that stopped short silently lands on a SMALLER step
    # (FS-W maxes at 60,058, so at --at-step 86000 it lands there while N and L
    # sit at 86,000). A table mixing 60,058 with 86,000 invites a comparison
    # that is not byte-matched, so say so before anyone reads the numbers.
    def step_of(p):
        try:
            return int(os.path.basename(p).split("_t")[-1].split(".")[0])
        except (ValueError, IndexError):
            return None
    chosen_steps = {p: step_of(p) for p in paths}
    uniq = set(chosen_steps.values())
    MISMATCH = len(uniq) > 1
    if MISMATCH:
        print("\n" + "!" * 74)
        print("WARNING: the selected checkpoints are NOT at the same step.")
        for p, s in chosen_steps.items():
            print(f"    t={s if s else '?':>8}  {os.path.basename(p)}")
        print("Rows at different step counts are NOT byte-matched. Compare only")
        print("rows that share a step, or the differences conflate corpus with")
        print("amount of training.")
        print("!" * 74 + "\n")

    jobs = []
    for p in paths:
        arm = None
        for a in ARMS:
            if arm_matches(os.path.basename(p), a):
                arm = a
        if arm is None:
            continue
        jobs.append((arm, p))
    if not jobs:
        print("no FS-* checkpoints recognised in", paths)
        return

    # Each CPU eval at the ruler setting takes ~10 min, so score the arms
    # concurrently; three arms then cost one eval's wall clock, not three.
    from concurrent.futures import ThreadPoolExecutor
    print(f"scoring {len(jobs)} arm(s) with eval {' '.join(EVAL_ARGS)} "
          f"(identical for every arm)...", flush=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as ex:
        futs = {ex.submit(score, p): (a, p) for a, p in jobs}
        for fut, (a, p) in futs.items():
            try:
                s = fut.result()
            except Exception as e:
                s = None
                print(f"  FS-{a}: scoring raised {type(e).__name__}: {e}")
            if s:
                rows[a] = (p, s)
                print(f"  FS-{a}: in-domain {fmt_pair(s['ind'])}", flush=True)
            else:
                print(f"  FS-{a}: FAILED to score", flush=True)

    if len(rows) < 2:
        print(f"\nonly {len(rows)} arm(s) scored; need >= 2 to compare.")
        for a, (p, s) in rows.items():
            print(f"  FS-{a}: t={s['t']:,} bytes={s['seen']:,} "
                  f"in-domain {fmt_pair(s['ind'])}")
        return

    print("\n" + "=" * 74)
    print(f"{'arm':6s} {'t':>9s} {'bytes seen':>13s} {'in-domain CE':>20s} "
          f"{'novels':>8s} {'hot':>5s} {'half-life':>10s}")
    print("-" * 74)
    for a in ARMS:
        if a not in rows:
            continue
        p, s = rows[a]
        xd = f"{s['xd'][0]:.3f}" if s['xd'] and s['xd'][0] is not None else "n/a"
        hl = f"{s['hl']:.2f}" if s['hl'] is not None else "n/a"
        hot = s['hot'] if s['hot'] is not None else "n/a"
        print(f"FS-{a:3s} {num(s['t']):>9s} {num(s['seen']):>13s} "
              f"{fmt_pair(s['ind']):>20s} {xd:>8s} {str(hot):>5s} {hl:>10s}")
    print("=" * 74)

    # ---- verdict ---------------------------------------------------------
    # Pairwise verdicts only between arms that share a step count.
    pairable = {a for a, p in zip([j[0] for j in jobs], paths)
                if paths.count(p) == 1}
    step_of_arm = {a: chosen_steps.get(p) for a, p in jobs if a in rows}
    print(f"\nbar = {BAR} CE nats/byte (the real-difference line from RESULTS.md)")
    best = min((a for a in rows if rows[a][1]["ind"]
                and rows[a][1]["ind"][0] is not None),
               key=lambda a: rows[a][1]["ind"][0], default=None)
    if best is None:
        sys.exit("no in-domain mean to compare")
    print(f"best in-domain: FS-{best} at {rows[best][1]['ind'][0]:.4f}\n")

    for a in rows:
        if a == best:
            continue
        if MISMATCH and step_of_arm.get(a) != step_of_arm.get(best):
            print(f"  FS-{best} vs FS-{a}: SKIPPED -- not at a common step "
                  f"(t={step_of_arm.get(best)} vs t={step_of_arm.get(a)}); "
                  f"the difference would confound corpus with training.")
            continue
        # lower CE is better, and `best` is the MINIMUM, so the margin is
        # other - best (positive => best wins by that much). The first version
        # computed best - other, which is always negative and therefore
        # reported a 0.52 CE gap as "within noise".
        d = rows[a][1]["ind"][0] - rows[best][1]["ind"][0]
        sep = d > BAR
        # Cross-domain guard: a winner that wrecks novel-domain transfer has
        # not really won. d_xd positive => the in-domain winner is WORSE on
        # novels by that margin.
        d_xd = None
        if rows[best][1]["xd"] and rows[a][1]["xd"] \
                and rows[best][1]["xd"][0] is not None \
                and rows[a][1]["xd"][0] is not None:
            d_xd = rows[best][1]["xd"][0] - rows[a][1]["xd"][0]
        verdict = "SEPARATED" if sep else "within noise"
        if d_xd is not None and d_xd > BAR:
            verdict += f" (but worse on novels by {d_xd:+.3f})"
        line = (f"  FS-{best} vs FS-{a}: FS-{best} wins in-domain by "
                f"{d:+.4f} -> {verdict}")
        if d_xd is not None:
            line += f" | FS-{best} minus FS-{a} on novels = {d_xd:+.4f}"
        print(line)

    hot = {a: rows[a][1]["hot"] for a in rows}
    if len(set(hot.values())) > 1:
        print("\nhot dims (long-memory growth) differ: "
              + ", ".join(f"FS-{a}={hot[a]}" for a in ARMS if a in rows))
    print("\nNOTE: this is arm-vs-arm. It is NOT a claim that a from-scratch "
          "run beats\nthe incumbent at t=94,747, which has 64,747 more steps "
          "of history.")


if __name__ == "__main__":
    main()
