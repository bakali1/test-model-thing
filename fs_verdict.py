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
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
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


def pull_tags():
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

    A non-zero exit prints the tail of both streams: a silent failure here
    would otherwise look like a missing number in the verdict table.
    """
    p = subprocess.run([sys.executable, os.path.join(HERE, "eval.py"), path]
                       + EVAL_ARGS,
                       capture_output=True, text=True, cwd=HERE)
    if p.returncode != 0:
        print(f"  [eval.py rc={p.returncode} on {os.path.basename(path)}]")
        print((p.stdout[-1500:] or "") + (p.stderr[-1500:] or ""))
        return None
    out = p.stdout
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
    m = re.search(r"steps \(t\)\s*:\s*(\d+)\s+bytes seen\s*(\d+)", out)
    t = int(m.group(1)) if m else None
    seen = int(m.group(2)) if m else None
    return dict(ind=ind, xd=xd, shift=sh, hot=hot, hl=hl, t=t, seen=seen)


def fmt_pair(p):
    return f"{p[0]:.4f} +/- {p[1]:.4f}" if p[0] is not None else "n/a"


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--pull":
        pull_tags()
        argv = argv[1:]
    if not argv:
        argv = sorted(glob.glob(os.path.join(HERE, "fs_*_t*.npz")))
    paths = []
    for a in argv:
        paths.extend(sorted(glob.glob(a)))
    if not paths:
        sys.exit("no checkpoints given and none found; pass paths or --pull")

    jobs = []
    for p in paths:
        arm = None
        for a in ARMS:
            if f"fs_{a.lower()}_" in os.path.basename(p).lower():
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
        print(f"FS-{a:3s} {s['t']:>9,} {s['seen']:>13,} "
              f"{fmt_pair(s['ind']):>20s} "
              f"{(f'{s['xd'][0]:.3f}' if s['xd'][0] is not None else 'n/a'):>8s} "
              f"{s['hot']:>5d} "
              f"{(f'{s['hl']:.2f}' if s['hl'] is not None else 'n/a'):>10s}")
    print("=" * 74)

    # ---- verdict ---------------------------------------------------------
    print(f"\nbar = {BAR} CE nats/byte (the real-difference line from RESULTS.md)")
    best = min((a for a in rows if rows[a][1]["ind"][0] is not None),
               key=lambda a: rows[a][1]["ind"][0], default=None)
    if best is None:
        sys.exit("no in-domain mean to compare")
    print(f"best in-domain: FS-{best} at {rows[best][1]['ind'][0]:.4f}\n")

    for a in rows:
        if a == best:
            continue
        d = rows[best][1]["ind"][0] - rows[a][1]["ind"][0]
        sep = (d > BAR)
        # cross-domain guard: the winner must not wreck novel-domain transfer
        xd_gap = None
        if rows[best][1]["xd"][0] is not None and rows[a][1]["xd"][0] is not None:
            xd_gap = rows[best][1]["xd"][0] - rows[a][1]["xd"][0]
        verdict = "SEPARATED" if sep else "within noise"
        if xd_gap is not None and xd_gap < -BAR:
            verdict += " (but worse on novels)"
        print(f"  FS-{best} vs FS-{a}: d(indomain) = {d:+.4f} -> {verdict}"
              + (f" | d(novels) = {xd_gap:+.4f}" if xd_gap is not None else ""))

    hot = {a: rows[a][1]["hot"] for a in rows}
    if len(set(hot.values())) > 1:
        print("\nhot dims (long-memory growth) differ: "
              + ", ".join(f"FS-{a}={hot[a]}" for a in ARMS if a in hot))
    print("\nNOTE: this is arm-vs-arm. It is NOT a claim that a from-scratch "
          "run beats\nthe incumbent at t=94,747, which has 64,747 more steps "
          "of history.")


if __name__ == "__main__":
    main()
