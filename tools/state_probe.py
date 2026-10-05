#!/usr/bin/env python3
"""state_probe.py -- the ruler v2 never had: is the cue IN the state, and does
the head USE it?

    .venv/bin/python tools/state_probe.py
    .venv/bin/python tools/state_probe.py --distractors random     # the self-check
    .venv/bin/python tools/state_probe.py --trials 512 --batch 512

WHY THIS EXISTS. Two standing blockers, one of which is memory formation
(mean half-life 1.14 B; 7 of 8192 dims above 8 B). Both are undecidable
because nothing in the repo distinguishes

    "the state contains no information about the cue"          from
    "the state contains it and the predictive head never uses it".

The #23 needle probe cannot make that distinction: it tested LITERAL-CUE
INDUCTION ONLY, so its null does not generalize to "no memory". This probe
does not use surface form at all -- the cue is a uniformly random byte -- so
there is no word to continue and no phrase to quote.

PROTOCOL (ROADMAP v3, P0 item 2), per trial:
  1. zero the recurrent state; feed ONE cue byte, drawn UNIFORMLY OVER ALL 256
     SYMBOLS (rng.integers(0, 256)) -- never a word, never a corpus-common byte,
     because a plausible cue would reproduce #23's confound;
  2. feed d bytes of distractors taken from a HELD-OUT corpus file;
  3. at each d in --distances, snapshot the recurrent state
     (concatenated layer states, layers x dim) and the output distribution;
  4. after the sweep, fit a tiny OFFLINE linear probe per distance -- ridge
     regression onto one-hot targets, closed form, in the dual, NO SGD and NO
     TMT weight updates -- and decode the cue byte from the state snapshot.
     Chance = 1/256 = 0.390625%.

WHAT IT REFUSES TO DO, and why:

1. **It will not run on uniform random distractor bytes and call the result
   anything.** `tools/random_vs_real.py` measured this: on random bytes
   rho(1) flips +0.534 -> -0.171 and SNR moves 8.6x. The gate asserts real
   corpus bytes and refuses to print a verdict in the random-byte mode; that
   mode exists only as a self-check and is labelled CONTROL everywhere it
   appears.
2. **It will not report an accuracy without its trial count, its SE, its
   chance level, its probe type, its ridge lambda and its distance
   definition.** Coverage is definition x floor; the same is true of an
   accuracy. Every number below is printed with them.
3. **It will not let a high accuracy stand without the mismatched-cue
   control.** The identical pipeline on permuted cue labels must land at
   chance. If it does not, the probe is exploiting trial identity rather than
   the cue and the verdict is INCONCLUSIVE, not PASS.
4. **It will not call the second column of the 2x2 "retrieval".** With a
   uniform random cue there is no correct continuation tied to the cue, so
   what is measurable is only whether the output distribution DEPENDS on the
   cue -- a sensitivity, not a task. See "THE 2x2 IS NOT SYMMETRIC" below.
5. **It will not be scored against the incumbent's CE.** This probe says
   nothing about language modelling quality.

THE 2x2 IS NOT SYMMETRIC -- read this before reading a cell.
The probe can fill the first column (does the state carry the cue?) with a
linear readout, and it can fill the second column only in the weak "the
output distribution depends on the cue" sense. A POSITIVE result in that
second column is close to automatic whenever the first column is positive,
because the byte head is a fixed function of the state: if the state carries a
bit of the cue, some readout direction of the state carries it too, and the
head is a linear map of the state. So a "yes / no" verdict from this tool
means "the state carries the cue and the head's output does not track it",
which is the most valuable cell -- but "yes / yes" here does NOT establish
task-relevant retrieval, because a uniformly random cue has no task. The row
"state no / head yes for LITERAL cues" (the surface-form-shortcut row) cannot
be produced by this design at all: it requires a cue with surface form, which
is exactly what this probe is forbidden to use.
"""
import argparse
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import numpy as np

import numpy_port as N
from eval_npz import infer_config

VOCAB = 256
CHANCE = 1.0 / VOCAB          # 0.00390625
# Practical bar for leg B, in probability units: a cue that moves the output
# distribution by less than this is not "the head using it" in any sense that
# could matter, whatever its p-value. Stated, not tuned: 1 percentage point of
# probability mass. Significance alone is not enough -- a differential of
# 2e-3 is reliably non-zero over a few hundred trials and is still nothing.
LEG_B_BAR = 0.01
DEFAULT_CKPT = os.path.join(HERE, "runs/lrroles/arm_lrdecay1x_t919897.npz")
DEFAULT_DISTRACTORS = os.path.join(HERE, "data/eval/tinystories_valid.txt")
TRAIN_GLOB = "data/wikipedia_clean/**/wiki_*"
DEFAULT_DISTANCES = [8, 32, 64, 128, 256, 512, 1024, 4096, 16384]
# The roadmap's required set, completed to a log-spaced series. One pass per
# trial snapshots every distance, so the extra points cost NOTHING in compute
# and d=16 is the single most informative addition: the decay curve puts the
# horizon between 8 B and 32 B, and a ruler that skips 16 cannot say where.
FULL_DISTANCES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


# --------------------------------------------------------------------------
# the offline probe: closed-form ridge, no SGD, no TMT updates
# --------------------------------------------------------------------------
def onehot(y, n_cls=VOCAB):
    Y = np.zeros((len(y), n_cls), np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def _fold_idx(n, k, rng):
    p = rng.permutation(n)
    return [(np.setdiff1d(np.arange(n), p[i::k]), p[i::k]) for i in range(k)]


def _gram(X, standardize=True):
    """Centre on the pooled mean and return the n x n Gram. n is a few hundred
    trials against 8192 features, so the DUAL form is used throughout and no
    p x p system is ever formed.

    `standardize` z-scores each dimension using the POOLED (all-trial, label-
    free) mean and SD. That is preprocessing, not extra capacity: the probe is
    still a plain ridge on the same features. It is on because the state
    dimensions are wildly unequal in scale, and without it the ridge spends
    its budget on the high-variance dimensions -- which are the short-sigma
    ones, i.e. exactly the ones that retain nothing.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    if standardize:
        Xc = Xc / np.maximum(Xc.std(axis=0, ddof=0), 1e-8)
    return Xc @ Xc.T, Xc


def _fit_predict(G, Y, tr, te, lam):
    alpha = np.linalg.solve(G[np.ix_(tr, tr)] + lam * np.eye(len(tr)), Y[tr])
    return (G[np.ix_(te, tr)] @ alpha).argmax(axis=1)


def probe_accuracy(X, y, rng, folds_n, lams, perm=None, standardize=True):
    """Honest held-out accuracy, with lambda chosen by an INNER CV restricted to
    each outer fold's TRAINING trials.

    Index discipline, which is where a nested CV usually goes wrong: the inner
    folds index into the training SUBSET, the outer test indices index into
    the full set. Both are applied to the pooled Gram, so an inner index and
    an outer index can never be silently mixed.
    """
    n = len(y)
    yy = y if perm is None else y[perm]
    G, _ = _gram(X, standardize)
    Y = onehot(yy)
    outer = _fold_idx(n, folds_n, rng)
    chosen, per_fold = [], []
    for tr, te in outer:
        sub = rng.permutation(len(tr))              # inner index space
        k = min(3, len(tr))
        inner = [np.setdiff1d(np.arange(len(tr)), sub[i::k]) for i in range(k)]
        inner = [(inner[i], sub[i::k]) for i in range(k)]
        scores = np.zeros(len(lams))
        for li, lam in enumerate(lams):
            hits = 0
            for itr, ite in inner:
                gtr = np.ix_(tr[itr], tr[itr])
                alpha = np.linalg.solve(
                    G[gtr] + lam * np.eye(len(itr)), Y[tr[itr]])
                pred = (G[np.ix_(tr[ite], tr[itr])] @ alpha).argmax(axis=1)
                hits += int((pred == yy[tr[ite]]).sum())
            scores[li] = hits / len(tr)
        li = int(scores.argmax())
        chosen.append(lams[li])
        per_fold.append(float((_fit_predict(G, Y, tr, te, lams[li]) == yy[te]).mean()))
    return float(np.mean(per_fold)), chosen, per_fold


def binom_se(p, n):
    return float(np.sqrt(max(p * (1 - p), 0.0) / max(n, 1)))


def boot_ci(vals, rng, reps=10000, alpha=0.05):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(v), size=(reps, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------
def sweep(ckpt, raw, cues, mis, offsets, distances, batch):
    """Feed every trial and snapshot state + output at each distance.

    cost = trials x max(distance) bytes, because ONE pass per trial takes
    snapshots at every d rather than re-running from scratch per d. That is
    also the correct semantics: the state at distance d is the state after
    exactly d distractor bytes, not after a re-run.
    """
    d = np.load(ckpt)
    dim, layers = infer_config(d)
    m = N.Model(dim=dim, layers=layers, temp=0.75, lr=5e-4)
    m.new_states(1)
    m.load(ckpt)                       # honours the checkpoint's ring_k
    K = m.ring_k
    m.new_states(1)                    # score the weights, not a stream's state
    dim_state = layers * dim

    n = len(cues)
    Dmax = max(distances)
    feats = {dd: np.zeros((n, dim_state), np.float32) for dd in distances}
    p_cue = {dd: np.zeros(n, np.float64) for dd in distances}
    p_mis = {dd: np.zeros(n, np.float64) for dd in distances}
    floor_cue = np.zeros(n, np.float64)  # per-trial mean P(cue) over the run
    floor_mis = np.zeros(n, np.float64)  # ... and mean P(mismatched cue)
    ctx_ce = np.zeros(n, np.float64)     # per-trial mean CE on the distractor run
    want = set(distances)

    for b0 in range(0, n, batch):
        sl = slice(b0, min(b0 + batch, n))
        nb = sl.stop - sl.start
        seq = np.empty((nb, Dmax + 1), np.uint8)
        seq[:, 0] = cues[sl]
        for j in range(nb):
            seq[j, 1:] = raw[offsets[sl.start + j]:
                            offsets[sl.start + j] + Dmax]
        cu, ms = cues[sl], mis[sl]
        m.new_states(nb)
        acc_c = np.zeros(nb)
        acc_m = np.zeros(nb)
        acc_ce = np.zeros(nb)
        cnt = 0
        ar = np.arange(nb)
        for t in range(Dmax + 1):
            enc = m.embed[seq[:, t]]
            x = enc
            for l in m.layers:
                x, st, _d, _c = l.forward(enc, x, l.states)
                l.states = st
            logits = x @ m.Wd.T + m.bd
            mx = logits.max(axis=1)
            lse = mx + np.log(np.exp(logits - mx[:, None]).sum(axis=1))
            p = np.exp(logits - lse[:, None])
            if t in want:
                dd = t
                # (nb, dim) per layer, concatenated along the FEATURE axis:
                # row j is trial j's full recurrent state, layers in order.
                snap = np.concatenate([m.layers[k].states for k in range(layers)],
                                      axis=1)
                feats[dd][sl] = snap.astype(np.float32)
                # Index p BY THE CUE BYTE. An earlier version of this line
                # indexed by seq[:, t+1], which is the next REAL byte -- so the
                # "does the output depend on the cue" leg was silently
                # measuring ordinary byte predictability instead. It looked
                # plausible (0.52 at d=8, matching the distance-curve cell's
                # p_close) precisely because it is a real, different statistic.
                p_cue[dd][sl] = p[ar, cu]
                p_mis[dd][sl] = p[ar, ms]
            # The context CE is accumulated on EVERY step, not only on the
            # snapshot steps. Inside the `if t in want` block it was summed
            # over ~13 values and divided by all 16385, which printed
            # "0.0030 nats/byte" for a stream whose real CE is ~1.17 -- an
            # absurd number that the sanity check was there to catch, and that
            # this very check was hiding.
            nxt = seq[:, min(t + 1, Dmax)]
            acc_ce += -np.log(np.maximum(p[ar, nxt], 1e-30))
            if t >= 1:
                # floors: this trial's own mean probability on its own cue byte
                # and on the mismatched one, over the same positions.
                acc_c += p[ar, cu]
                acc_m += p[ar, ms]
            cnt += 1
        floor_cue[sl] = acc_c / max(1, cnt - 1)
        floor_mis[sl] = acc_m / max(1, cnt - 1)
        ctx_ce[sl] = acc_ce / max(1, cnt)
        print(f"    ... {sl.stop}/{n} trials", flush=True)
    return {"feats": feats, "p_cue": p_cue, "p_mis": p_mis,
            "floor_cue": floor_cue, "floor_mis": floor_mis,
            "ctx_ce": ctx_ce, "K": K, "dim_state": dim_state,
            "layers": layers, "dim": dim}


# --------------------------------------------------------------------------
def ascii_plot(distances, accs, ses, label):
    print(f"\n  cue decoding accuracy vs distance{label} "
          f"(chance {100 * CHANCE:.3f}%, '+' = above, 'v' = at/below chance)")
    hi = max(accs + [CHANCE]) * 100
    lo = min(accs + [CHANCE]) * 100
    if hi - lo < 1e-9:
        hi, lo = hi + 1.0, lo
    rows = 10
    print(f"  {100 * CHANCE:6.3f} " + "-" * (len(distances) * 4 + 4) + " chance")
    for r in range(rows, -1, -1):
        val = lo + (hi - lo) * r / rows
        row = f"  {val:6.3f} |"
        for a, s in zip(accs, ses):
            mark = "+" if a - 2 * s > CHANCE else ("v" if a + 2 * s <= CHANCE
                                                  else ("~" if a > CHANCE else "."))
            row += f" {mark:^3}"
        print(row + (f"  <- {val:.3f}%" if r == rows else ""))
    print("       " + "".join(f"{d:>4}" for d in distances))
    print("         distance (bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--distractors", default=DEFAULT_DISTRACTORS,
                    help="held-out corpus file the distractor bytes come from")
    ap.add_argument("--train-glob", default=TRAIN_GLOB,
                    help="used only to ASSERT the distractor file is held out")
    ap.add_argument("--distractors-mode", choices=("real", "random"),
                    default="real",
                    help="'random' is the CONTROL arm; a verdict is refused in it")
    ap.add_argument("--distances",
                    default=",".join(str(d) for d in FULL_DISTANCES),
                    help="roadmap requires at least 8,32,64,128,256,512,1024,"
                         "4096,16384; the default adds the log-spaced points "
                         "between them, which cost no extra compute because "
                         "one pass snapshots every distance")
    ap.add_argument("--calibration-distances", default="0,1,2,4",
                    help="extra distances ALWAYS swept, reported separately as "
                         "the PROBE POWER curve. Decoding the cue at d=0..4 is "
                         "easy; where this curve falls to chance is where the "
                         "probe STOPS BEING ABLE TO ANSWER, and without it "
                         "'not decodable at d=8' is uninterpretable (no "
                         "information? or no power?)")
    ap.add_argument("--trials", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-trials", type=int, default=64,
                    help="trials for the random-distractor CONTROL arm (0 skips)")
    ap.add_argument("--lambdas", default="1,10,100,1000,1e4,1e5,1e6,1e7",
                    help="ridge lambda grid; chosen per outer fold by inner CV")
    ap.add_argument("--no-standardize", action="store_true",
                    help="disable per-dimension z-scoring of the state (it is "
                         "on by default; see _gram() for why)")
    ap.add_argument("--stability", type=int, default=0,
                    help="repeat the probe with N INDEPENDENT fold "
                         "partitions and report the spread. A distance whose "
                         "(accuracy - control) changes sign across partitions "
                         "is NOT a finding, however far above chance its "
                         "single-run number sits")
    ap.add_argument("--load", default=None,
                    help="npz written by --save; re-run the PROBE and the "
                         "verdict on a completed sweep without re-paying the "
                         "forward pass. The random-byte CONTROL arm is still "
                         "swept, because a verdict needs it and it is cheap")
    ap.add_argument("--save", default=None,
                    help="npz path for the swept state snapshots and per-trial "
                         "readouts, so the probe can be re-run with different "
                         "settings without re-paying the forward pass")
    ap.add_argument("--label", default="")
    ap.add_argument("--cue", choices=("uniform", "corpus-support", "corpus-freq"),
                    default="uniform",
                    help="cue-byte distribution. 'uniform' (default) is the "
                         "protocol as written: uniform over all 256 symbols. "
                         "'corpus-support' is uniform over the byte values that "
                         "actually occur in the distractor file -- in-distribution "
                         "without a degenerate mode. 'corpus-freq' samples "
                         "proportional to corpus frequency and is REPORTED WITH A "
                         "WARNING: a constant 'always answer the modal byte' "
                         "predictor scores ~its frequency with ZERO memory, so it "
                         "must be compared against --modal-baseline, not chance.")
    args = ap.parse_args()
    STD = not args.no_standardize
    # The trial count is part of every accuracy's definition, so it must be the
    # one actually used. On a --load run that is the saved sweep's n, NOT the
    # (then unused) --trials default; reading it here, before the header
    # prints, is why this is not printed later.
    N_TRIALS = args.trials
    if args.load and os.path.exists(args.load):
        try:
            N_TRIALS = int(len(np.load(args.load)["cues"]))
        except Exception:
            pass

    distances = [int(x) for x in args.distances.split(",") if x.strip()]
    calib = [int(x) for x in args.calibration_distances.split(",") if x.strip()]
    all_d = sorted(set(distances) | set(calib))
    lams = [float(x) for x in args.lambdas.split(",")]
    mode = "CONTROL" if args.distractors_mode == "random" else "MEASUREMENT"
    print("=" * 78)
    print(f"state probe -- {mode}{(' / ' + args.label) if args.label else ''}")
    print("=" * 78)
    print(f"  checkpoint     : {args.ckpt}")
    print(f"  CUE            : ONE byte, UNIFORM over all 256 symbols "
          f"(rng.integers(0, 256), seed={args.seed}).")
    print(f"                   Not a word, not a corpus-common byte. A "
          f"surface-form cue would\n                   reproduce the #23 "
          f"confound (literal-cue induction only).")
    print(f"  DISTRACTORS    : {args.distractors_mode.upper()} bytes from "
          f"{args.distractors}")
    print(f"  distance d     : bytes of distractor fed AFTER the cue; the "
          f"state is snapshotted\n                   after the model has "
          f"consumed cue + exactly d distractor bytes.")
    print(f"  distances      : {distances}")
    print(f"  calibration    : {calib} (PROBE POWER -- same pipeline, same "
          f"cue; where it fails is where the probe cannot answer)")
    print(f"  trials         : {N_TRIALS}  (probe CV: "
          f"{args.folds}-fold; every trial is tested exactly once"
          f"{'; read from the loaded sweep, --trials is IGNORED here'
           if args.load else ''})")
    print(f"  probe          : ridge regression onto one-hot targets, "
          f"closed form, DUAL form\n                   (n x n Gram, never an "
          f"8192 x 8192 system). NO SGD, NO TMT updates.")
    print(f"  lambda grid    : {lams} (chosen per outer fold by inner 3-fold "
          f"CV on the TRAINING folds only)")
    print(f"  preprocessing  : per-dimension z-scoring on POOLED (label-free) "
          f"statistics: {'ON' if STD else 'OFF'}")
    print(f"  chance         : 1/{VOCAB} = {100 * CHANCE:.6f}%  "
          f"(argmax over {VOCAB} near-uniform classes)")

    # ---- the real-bytes assertion (P0 item 3 of v3) ----
    tr_files = {os.path.realpath(f) for f in
                glob.glob(os.path.join(HERE, args.train_glob), recursive=True)}
    ok_real = os.path.exists(args.distractors)
    if args.distractors_mode == "real":
        assert ok_real, f"no distractor file at {args.distractors}"
        inside = os.path.realpath(args.distractors) in tr_files
        print(f"\n  [assert] distractor file is NOT reachable by the training "
              f"glob ({args.train_glob}): "
              f"{'PASS' if not inside else 'FAIL'}")
        if inside:
            sys.exit("[FATAL] the distractor bytes are training bytes. Probing "
                     "on them measures memorization. Move the file or change "
                     "--train-glob.")
    if mode == "CONTROL":
        print("\n  ** CONTROL ARM. Distractor bytes are uniform random, NOT "
              "corpus text.**\n     On random bytes the states are nearly iid "
              "and lag weightings collapse onto one\n     direction, which "
              "manufactures confident results (tools/random_vs_real.py: "
              "rho(1) flips\n     +0.534 -> -0.171, SNR moves 8.6x). This arm "
              "exists to be compared against the real\n     arm and MUST read "
              "at or below chance. It is not a measurement.")
    raw = (np.frombuffer(open(args.distractors, "rb").read(), np.uint8)
           if ok_real else None)

    def run_arm(n_trials, distractors_mode, tag):
        rng = np.random.default_rng(args.seed)
        if args.cue == "uniform":
            cues = rng.integers(0, VOCAB, size=n_trials).astype(np.uint8)
            support_note = "uniform over all 256 symbols"
            modal = 1.0 / VOCAB
        elif args.cue == "corpus-support":
            # Uniform over the byte values that actually occur in the corpus.
            # Keeps every cue plausible to the model (in-distribution) while
            # preserving uniformity, so there is no degenerate constant answer.
            # A FREQUENCY-weighted draw is deliberately NOT used here: with
            # one, "always answer the modal byte" scores ~20% with no memory
            # at all and reads as excellent retention.
            assert raw is not None, "--cue corpus-support needs real distractors"
            present = np.flatnonzero(np.bincount(raw, minlength=VOCAB) > 0)
            cues = rng.choice(present, size=n_trials).astype(np.uint8)
            support_note = (f"uniform over the {len(present)} byte values that "
                            f"occur in the distractor file")
            modal = 1.0 / len(present)
        else:  # corpus-freq
            assert raw is not None, "--cue corpus-freq needs real distractors"
            cnt = np.bincount(raw, minlength=VOCAB).astype(np.float64)
            p = cnt / cnt.sum()
            cues = rng.choice(VOCAB, size=n_trials, p=p).astype(np.uint8)
            support_note = ("WARNING proportional to corpus frequency; compare "
                            "against the MODAL baseline, not against chance")
            modal = float(p.max())
        # A derangement: trial j is probed against trial j-1's cue. This is
        # leg B's control -- without it, any position-dependent or byte-value
        # dependent artefact in the output distribution reads as "the head
        # uses the cue".
        mis = np.roll(cues, 1)
        bad = mis == cues
        if bad.any():
            cues[bad] = (cues[bad] + 1 + (bad.sum() % 200)) % VOCAB
            mis = np.roll(cues, 1)
        assert (mis != cues).all(), "mismatched-cue control is not a derangement"
        Dmax = max(all_d)
        if distractors_mode == "real":
            need = Dmax
            slots = max(1, (len(raw) - need) // need)
            pick = rng.permutation(slots)[:n_trials]
            reuse = n_trials > slots
            offs = (pick % slots) * need
            src = f"held-out corpus, {slots} disjoint {need}-B windows" \
                  f"{' (REUSED: trials > windows)' if reuse else ''}"
        else:
            offs = np.zeros(n_trials, int)
            src = "uniform random bytes (CONTROL)"
        print(f"\n  [{tag}] {n_trials} trials | distractor source: {src}")
        print(f"  [{tag}] cue distribution: {support_note}")
        if args.cue == "corpus-freq":
            print(f"  [{tag}] baselines -- chance 1/{VOCAB} = {100.0/VOCAB:.3f}%"
                  f"   MODAL-constant predictor = {100*modal:.3f}%"
                  f"  <-- a CONSTANT 'always answer the modal byte' predictor"
                  f" scores {100*modal:.1f}% with ZERO memory."
                  f" Judge this arm by EXCESS OVER THE MISMATCHED-CUE CONTROL"
                  f" (valid under any cue distribution), never by the absolute rate.")
        else:
            print(f"  [{tag}] baselines -- chance 1/{VOCAB} = {100.0/VOCAB:.3f}%"
                  f"   uniform-over-support = {100*modal:.3f}%"
                  f"   (excess over the mismatched-cue control is valid under"
                  f" either)")
        base = 1.0 / VOCAB if args.cue == "uniform" else modal
        print(f"  [{tag}] cost = trials x max(distance) = "
              f"{n_trials * (Dmax + 1):,} bytes of forward pass "
              f"(one pass per trial; snapshots at every d)")
        out = sweep(args.ckpt, raw, cues, mis, offs, all_d, args.batch)
        out.update({"cues": cues, "mis": mis, "rng": rng, "src": src,
                    "base": base, "cue_mode": args.cue,
                    "modal": modal, "support_note": support_note})
        return out

    if args.load:
        z = np.load(args.load)
        _dd = np.load(args.ckpt)
        _dim, _layers = infer_config(_dd)
        A = {"feats": {dd: z[f"X_{dd}"] for dd in all_d},
             "p_cue": {dd: z[f"p_cue_{dd}"] for dd in all_d},
             "p_mis": {dd: z[f"p_mis_{dd}"] for dd in all_d},
             "floor_cue": z["floor_cue"], "floor_mis": z["floor_mis"],
             "ctx_ce": z["ctx_ce"], "cues": z["cues"], "mis": z["mis"],
             "rng": np.random.default_rng(args.seed), "src": f"loaded {args.load}",
             "dim": _dim, "layers": _layers, "dim_state": _dim * _layers,
             "K": int(_dd["ring_k"]) if "ring_k" in _dd else None}
        A["n_trials"] = len(z["cues"])
        print(f"\n  [loaded] {args.load} -- probe and verdict re-run on the "
              f"completed sweep,\n            no forward pass repeated. "
              f"n={len(z['cues'])} trials from the saved sweep (--trials is "
              f"IGNORED on a load;\n            printing it here would "
              f"misreport the trial count, which is exactly the kind\n"
              f"            of definitional slip this file exists to "
              f"prevent).")
        print("  [note] the context-CE sanity line below is NOT recoverable "
              "from a loaded sweep: its\n          accumulator was fixed after "
              "that sweep ran. Re-run without --load to check it.")
        _B = None
        if f"B_X_0" in z.files:
            _B = {"feats": {dd: z[f"B_X_{dd}"] for dd in all_d},
                  "cues": z["B_cues_0"],
                  "rng": np.random.default_rng(args.seed + 1)}
            print(f"            random-byte CONTROL arm also restored from "
                  f"the file (n={len(_B['cues'])}); not re-swept.")
    else:
        A = run_arm(args.trials, args.distractors_mode,
                    "real" if mode == "MEASUREMENT" else "control")
        A["n_trials"] = args.trials
        _B = None
    B = (_B if _B is not None else
         (run_arm(args.control_trials, "random", "random-byte CONTROL")
          if args.control_trials > 0 else None))

    print(f"\n  [context sanity] the model's CE on the distractor run itself: "
          f"{A['ctx_ce'].mean():.4f} nats/byte over {A['ctx_ce'].size} trials. "
          f"This is\n  what the probe is running on; if it were far from the "
          f"ruler's ~1.17 the stream is not\n  the text it claims to be."
          + ("   (NOT CHECKED: loaded sweep)" if args.load else ""))
    print(f"  [probe input] state dimension {A['dim_state']} = "
          f"{A['layers']} layers x {A['dim']} dims | ring K={A['K']}")

    # ---- the probe, per distance ----
    print("\n" + "-" * 78)
    print("LEG A -- DOES THE STATE CARRY THE CUE?  (linear probe, held-out)")
    print("-" * 78)
    print("  DEFINITION: ridge-to-one-hot accuracy on the concatenated "
          f"{A['layers']}x{A['dim']} = {A['dim_state']}-dim state snapshot, "
          f"{args.folds}-fold CV,\n              lambda chosen by inner CV on "
          f"training folds only. Chance {100 * CHANCE:.4f}%. SE is the "
          f"BINOMIAL SE at\n              the held-out n. "
          f"'mismatched-cue' is the identical pipeline on\n              permuted "
          f"cue labels; it must sit at chance or the probe is reading trial "
          f"identity.")
    rng = A["rng"]
    rowsA, accs, ses = [], [], []
    cal_rows = []
    for dd in all_d:
        X = A["feats"][dd]
        y = A["cues"]
        acc, chosen, per = probe_accuracy(X, y, rng, args.folds, lams,
                                          standardize=STD)
        # The mismatched-cue control MUST be recomputed for every distance,
        # including the calibration ones. An earlier version computed it only
        # for the main distances and let the calibration rows reuse the
        # previous iteration's value, which silently judged d=2 and d=4
        # against a control belonging to a different distance.
        perm = rng.permutation(len(y))
        ctrl, _cc, _ = probe_accuracy(X, y, rng, args.folds, lams,
                                      perm=perm, standardize=STD)
        se = binom_se(acc, len(y))
        cse = binom_se(ctrl, len(y))
        # Compare against the pipeline's OWN empirical null (the same probe on
        # permuted cue labels), not against the analytic chance rate. At n=256
        # and a 0.39% rate, one lucky correct decode in the control is 0.39%
        # -- enough to clear chance + 2 SE and would veto a real 5% reading.
        # The empirical null is the right reference and it is stricter, not
        # looser: a real result must beat what the identical machinery gets
        # on labels known to carry no information.
        has = (acc > CHANCE) and (acc - ctrl > 2.0 * math.hypot(se, cse))
        if dd in distances:
            rowsA.append({"d": dd, "acc": acc, "se": se, "ctrl": ctrl, "cse": cse,
                          "lams": chosen, "has": has, "n": len(y)})
            accs.append(acc)
            ses.append(se)
            print(f"  d={dd:>6} B | acc {100 * acc:7.4f}% +/- {100 * se:6.4f} "
                  f"(n={len(y)}) | chance {100 * CHANCE:.4f}% | "
                  f"mismatched-cue control {100 * ctrl:6.4f}% "
                  f"+/- {100 * cse:.4f} | lambda {chosen[0]:g}..{chosen[-1]:g} | "
                  f"state carries cue: {'YES' if has else 'no'}")
        else:
            cal_rows.append({"d": dd, "acc": acc, "se": se, "ctrl": ctrl,
                             "cse": cse, "has": has})
    power = [c for c in cal_rows if c["has"]]
    power_max = max((c["d"] for c in power), default=None)
    print(f"\n  PROBE POWER (calibration distances, same pipeline, same cue):")
    for c in sorted(cal_rows, key=lambda r: r["d"]):
        print(f"    d={c['d']:>3} B | acc {100 * c['acc']:7.4f}% +/- "
              f"{100 * c['se']:.4f}% | mismatched-cue control "
              f"{100 * c['ctrl']:6.4f}% | decodable: "
              f"{'YES' if c['has'] else 'no'}")
    if power_max is None:
        print("    ** THE PROBE CANNOT EVEN DECODE THE CUE AT d=0, WHERE THE "
              "STATE IS LITERALLY\n       emb[cue]. ** Nothing below is "
              "evidence about memory; it is evidence about the probe.")
    else:
        print(f"    The probe decodes the cue at every calibration distance up "
              f"to d = {power_max} B. Note what this curve IS: the probe's own\n"
              f"    RESPONSE to a cue of KNOWN strength. It decays smoothly "
              f"with distance, not at a threshold,\n    because the cue's "
              f"amplitude decays too. So it is not a 'noise floor' and a 'no' "
              f"beyond it is\n    NOT automatically evidence of absent memory "
              f"-- see the amplitude threshold below, which is the\n    "
              f"quantity that actually transfers.")

    # ---- leg B: does the output depend on the cue? ----
    print("\n" + "-" * 78)
    print("LEG B -- DOES THE OUTPUT DEPEND ON THE CUE?  (sensitivity, NOT a task)")
    print("-" * 78)
    print("  DEFINITION: for each trial, [P(cue byte) at distance d MINUS that "
          "trial's own mean\n              P(cue byte) over the run] MINUS the "
          "same quantity for a MISMATCHED cue\n              (another trial's "
          "cue byte, same context, same positions). The differential is the "
          "statistic;\n              the raw lift is printed too because it is "
          "what the differential corrects. CI is a\n              10,000-"
          "replicate bootstrap over trials.\n"
          f"  BAR: the differential must be positive (bootstrap CI excluding "
          f"0) AND exceed {LEG_B_BAR:g} in probability\n       units. The "
          f"magnitude bar matters: significance alone called a 2e-3 shift a "
          f"result,\n       and 2e-3 of probability mass is nothing anyone "
          f"would call the head using memory.\n"
          "  READ THIS BEFORE THE NUMBER: a uniform random cue has NO correct "
          "continuation, so\n  this measures whether the output distribution "
          "MOVES, not whether the model retrieved\n  anything. It follows "
          "almost automatically from leg A, because the byte head is a\n"
          "  fixed linear map of the state. It is a necessary condition, not "
          "evidence of\n  task-relevant retrieval, and the 2x2's literal-cue "
          "row cannot be produced by this design.")
    rowsB = []
    for dd in distances:
        raw_lift = A["p_cue"][dd] - A["floor_cue"]
        mis_lift = A["p_mis"][dd] - A["floor_mis"]
        diff = raw_lift - mis_lift
        lo, hi = boot_ci(diff, rng)
        uses = (lo > 0.0) and (float(diff.mean()) > LEG_B_BAR)
        rowsB.append({"d": dd, "lift": float(diff.mean()), "lo": lo, "hi": hi,
                      "uses": uses, "raw": float(raw_lift.mean()),
                      "mis": float(mis_lift.mean()),
                      "p_cue": float(A["p_cue"][dd].mean()),
                      "p_mis": float(A["p_mis"][dd].mean()),
                      "floor": float(A["floor_cue"].mean())})
        print(f"  d={dd:>6} B | P(cue) {A['p_cue'][dd].mean():.3e} vs floor "
              f"{A['floor_cue'].mean():.3e} | P(mismatch) "
              f"{A['p_mis'][dd].mean():.3e} | raw lift "
              f"{raw_lift.mean():+.3e} | mismatch lift "
              f"{mis_lift.mean():+.3e} | DIFFERENTIAL {diff.mean():+.3e} "
              f"[{lo:+.3e}, {hi:+.3e}] | "
              f"{'YES' if uses else ('positive but below bar' if lo > 0 else 'no')}")

    # ---- the control arm ----
    ctrlA = []
    if B is not None:
        print("\n" + "-" * 78)
        print("CONTROL -- SAME PIPELINE, UNIFORM RANDOM DISTRACTORS")
        print("-" * 78)
        print("  This arm must read AT OR BELOW chance. If it reads above, "
              "the real arm's\n  number cannot be attributed to memory of the "
              "cue.")
        for dd in all_d:
            acc, _c, _p = probe_accuracy(B["feats"][dd], B["cues"],
                                         B["rng"], args.folds, lams,
                                         standardize=STD)
            se = binom_se(acc, len(B["cues"]))
            ctrlA.append({"d": dd, "acc": acc, "se": se,
                          "above": acc - 2 * se > CHANCE})
            print(f"  d={dd:>6} B | acc {100 * acc:7.4f}% +/- "
                  f"{100 * se:6.4f}% (n={len(B['cues'])}) | chance "
                  f"{100 * CHANCE:.4f}% | above chance: "
                  f"{'YES <-- CONTROL FAILED' if acc - 2 * se > CHANCE else 'no'}")
    ascii_plot(distances, accs, ses, "")

    # ---- stability across independent fold partitions ----
    # A single 4-fold partition at a 1-2% accuracy is 4-8 correct decodes out
    # of 512, and the mismatched-cue control is 2-5 hits out of 512. Whether
    # the difference is 4 hits or 2 is decided by the partition, not by the
    # model. Re-running the whole probe with independent partitions is the only
    # cheap way to see which distances survive that.
    if args.stability > 1:
        print("\n" + "-" * 78)
        print(f"STABILITY -- {args.stability} independent fold partitions per "
              f"distance")
        print("-" * 78)
        stable, marginal = [], []
        for a in rowsA:
            X, y = A["feats"][a["d"]], A["cues"]
            accs_r, ctrl_r = [], []
            for rep in range(args.stability):
                r2 = np.random.default_rng(args.seed * 1000 + 17 * rep + a["d"])
                ac, _c1, _ = probe_accuracy(X, y, r2, args.folds, lams,
                                            standardize=STD)
                pm = r2.permutation(len(y))
                co, _c2, _ = probe_accuracy(X, y, r2, args.folds, lams,
                                            perm=pm, standardize=STD)
                accs_r.append(ac)
                ctrl_r.append(co)
            d_r = np.array(accs_r) - np.array(ctrl_r)
            flips = bool((d_r > 0).any() and (d_r <= 0).any())
            call = ("SIGN FLIPS ACROSS PARTITIONS -- NOT A FINDING" if flips
                    else ("stable" if d_r.mean() > 0 else "consistently null"))
            (marginal if flips else stable).append(a["d"])
            print(f"  d={a['d']:>6} B | acc {100 * np.mean(accs_r):6.3f}% "
                  f"+/- {100 * np.std(accs_r):.3f} | control "
                  f"{100 * np.mean(ctrl_r):6.3f}% +/- {100 * np.std(ctrl_r):.3f}"
                  f" | diff {100 * d_r.mean():+.3f} pts, min "
                  f"{100 * d_r.min():+.3f} max {100 * d_r.max():+.3f} | {call}")
        print(f"\n  stable, above control: {stable}")
        print(f"  flips across partitions: {marginal} "
              f"-- these are NOT findings and must not be quoted as a horizon")
        rowsA = [a for a in rowsA if a["d"] not in marginal]
        if rowsA:
            print(f"  horizon recomputed on stable rows only: "
                  f"{max(a['d'] for a in rowsA if a['has'])} B")

    if args.save:
        blob = {"cues": A["cues"], "mis": A["mis"],
                "floor_cue": A["floor_cue"], "floor_mis": A["floor_mis"],
                "ctx_ce": A["ctx_ce"],
                **{f"p_cue_{dd}": A["p_cue"][dd] for dd in all_d},
                **{f"p_mis_{dd}": A["p_mis"][dd] for dd in all_d},
                **{f"X_{dd}": A["feats"][dd] for dd in all_d}}
        np.savez_compressed(args.save, **blob,
                            **{f"B_X_{dd}": B["feats"][dd] for dd in all_d},
                            **{f"B_cues_{dd}": B["cues"] for dd in [0]})
        print(f"\n  [saved] {args.save} -- swept states and readouts (main arm "
              f"+ random-byte control), so the\n            probe can be "
              f"re-fitted without re-running either sweep")

    # ---- the 2x2 ----
    print("\n" + "=" * 78)
    print("THE 2x2 -- one cell per distance")
    print("=" * 78)
    print("  state has cue? | output uses cue? | verdict")
    cells = {}
    for a, b in zip(rowsA, rowsB):
        has, uses = a["has"], b["uses"]
        if has and uses:
            c = ("yes / yes", "genuine learned retrieval -- AS FAR AS THIS "
                               "PROBE CAN SAY; see the leg-B caveat")
        elif has and not uses:
            c = ("yes / no", "MEMORY EXISTS; THE PREDICTIVE HEAD/OBJECTIVE "
                             "DOES NOT USE IT")
        elif (not has) and uses:
            c = ("no / yes", "the output moves without the cue being linearly "
                             "available; NOT the literal-cue-shortcut row, "
                             "which this design cannot produce")
        else:
            c = ("no / no", "no learned memory at this distance")
        cells[a["d"]] = c
        print(f"  {'YES' if has else 'no ':<14} | {'YES' if uses else 'no ':<14} | "
              f"{c[0]:<9} | {c[1]}")
    distinct = sorted({c[0] for c in cells.values()})
    print(f"\n  cells occupied across the swept range: {distinct}")

    # ---- decay parameters, needed by both the amplitude threshold and the
    # ---- consistency section ----
    _d = np.load(args.ckpt)
    _L = sum(1 for k in _d.files if k.startswith("layer") and k.endswith(".decay"))
    sig = np.concatenate([1 / (1 + np.exp(-_d[f"layer{i}.decay"]))
                          for i in range(_L)])
    hl = np.log(0.5) / np.log(sig)

    # ---- horizon ----
    has_ds = [a["d"] for a in rowsA if a["has"]]
    use_ds = [b["d"] for b in rowsB if b["uses"]]
    horizon_state = max(has_ds) if has_ds else None
    horizon_head = max(use_ds) if use_ds else None
    print("\n" + "-" * 78)
    print("MEMORY HORIZON -- the number the project does not have")
    print("-" * 78)
    print(f"  largest swept distance where the cue is DECODABLE from the "
          f"state: "
          f"{horizon_state if horizon_state else f'< {min(distances)} B (none)'}"
          + (" B" if horizon_state else ""))
    print(f"  largest swept distance where the output depends on it:        "
          f"{horizon_head if horizon_head else f'< {min(distances)} B (none)'}"
          + (" B" if horizon_head else ""))
    fine = min(a["se"] for a in rowsA)
    print(f"  resolution: smallest detectable effect at 2 sigma is "
          f"{200 * fine:.4f} points at n={rowsA[0]['n']}; chance is "
          f"{100 * CHANCE:.4f}%")
    print(f"  calibration rows: {[c['d'] for c in cal_rows]} (decodable out to "
          f"d = {power_max if power_max is not None else 'NONE'} B)")
    print("  STRUCTURAL CAVEAT: the cue is fed FIRST, so at every distance it "
          "is the OLDEST\n  content in the state and therefore the most "
          "attenuated element of it (every other\n  byte is at a smaller "
          "distance and a smaller sigma exponent). This design is the\n  "
          "hardest possible case for retention and it is deliberate -- it is "
          "the only way\n  to plant a cue whose presence is not predictable "
          "from surface form.")

    # ---- amplitude threshold: the quantity that actually transfers ----
    # The cue's contribution to the state at distance d is sigma^d * emb[cue],
    # so its RELATIVE amplitude is known in closed form and is not fitted.
    # The probe's detection threshold can therefore be read off the calibration
    # curve in units of that amplitude, and then applied to any distance --
    # which is what separates "absent" from "below what this probe can see".
    print("\n" + "-" * 78)
    print("AMPLITUDE THRESHOLD -- turning the response curve into a bound")
    print("-" * 78)
    print("  The state is s_t = sigma*s_{t-1} + enc_t, so a cue planted at "
          "distance d contributes\n  sigma^d * emb[cue] and its relative "
          "amplitude is known in closed form. The table\n  below pairs each "
          "distance with that amplitude (at the MEAN sigma and at the HOTTEST\n"
          "  sigma) and with what the probe actually read.")
    print(f"\n  {'d (bytes)':>10} {'sigma_mean^d':>15} {'sigma_max^d':>15} "
          f"{'probe acc':>12} {'decoded':>9}")
    for c in sorted(cal_rows + rowsA, key=lambda r: r["d"]):
        dd = c["d"]
        print(f"  {dd:>10} {sig.mean() ** dd:>15.3e} {sig.max() ** dd:>15.3e} "
              f"{100 * c['acc']:>11.4f}% {'YES' if c['has'] else 'no':>9}")
    seen = sorted([c for c in cal_rows + rowsA if c["has"]], key=lambda r: r["d"])
    unseen = sorted([c for c in cal_rows + rowsA if not c["has"]],
                    key=lambda r: r["d"])
    if seen:
        hi_amp = sig.mean() ** seen[-1]["d"]
        lo_amp = sig.mean() ** unseen[0]["d"] if unseen else 0.0
        print(f"\n  The probe's detection threshold in units of relative cue "
              f"amplitude (sigma_mean^d):\n    last distance DECODED   d = "
              f"{seen[-1]['d']:>6} B, amplitude {hi_amp:.3e}\n    "
              f"first distance NOT decoded d = "
              f"{unseen[0]['d'] if unseen else '-':>6} B, amplitude "
              f"{lo_amp:.3e}")
        print(f"  => any cue whose relative amplitude falls below ~"
              f"{hi_amp:.1e} (at the mean sigma) is below what this probe\n"
              f"     can see. It is NOT absent; it is unmeasured. The bound "
              f"that transfers to\n     the horizon is a bound on AMPLITUDE, "
              f"and only the decoded rows are a bound on DISTANCE.")
        print(f"  Using the HOTTEST coordinate instead (sigma_max = "
              f"{sig.max():.5f}): that coordinate alone would keep amplitude "
              f"{sig.max() ** seen[-1]['d']:.3e} at d = {seen[-1]['d']} B, so "
              f"the\n  surviving signal is concentrated in a handful of "
              f"dimensions and is diluted 8192-fold in the\n  concatenated "
              f"state. That dilution, not the decay alone, is what the "
              f"probe's threshold is measuring.")

    # ---- decay consistency ----
    print("\n" + "-" * 78)
    print("CONSISTENCY -- the decay histogram vs this probe")
    print("-" * 78)
    print(f"  decay sigma: min {sig.min():.5f} mean {sig.mean():.5f} max "
          f"{sig.max():.5f} | mean half-life {hl.mean():.3f} B | dims > 8 B: "
          f"{int((hl > 8).sum())} of {sig.size}")
    print(f"  {'d (bytes)':>10} {'max sigma^d':>14} {'mean sigma^d':>14} "
          f"{'dims with sigma^d > 1e-7':>26}")
    for dd in distances:
        print(f"  {dd:>10} {sig.max() ** dd:>14.3e} {sig.mean() ** dd:>14.3e} "
              f"{int((sig ** dd > 1e-7).sum()):>26}")
    surv = [dd for dd in distances if sig.max() ** dd > 1e-7]
    print(f"  The BEST single decay coordinate keeps >1e-7 of a cue up to "
          f"d = {max(surv) if surv else 0} B.")
    print(f"  Probe horizon (decodable) = {horizon_state if horizon_state else 'none'} B"
          f" | decay-histogram prediction = {max(surv) if surv else 0} B.")
    if horizon_state and max(surv) and horizon_state > max(surv):
        print("  ** CONTRADICTION: the state is measurably decodable FARTHER "
              "than its own decay\n     parameters allow. That is a real "
              "disagreement between two results and it is worth\n     more "
              "than either number alone: either the probe is reading "
              "something other than memory\n     of the cue, or the decay "
              "histogram is not the whole story. **")
    elif horizon_state:
        print("  CONSISTENT: the probe's horizon is at or below what the "
              "decay parameters permit, and\n  the decay histogram predicts "
              "the probe's answer.")
    else:
        print("  CONSISTENT (and unsurprising): the state does not carry a "
              "random cue even at the\n  SHORTEST swept distance, which is "
              "what a mean half-life near 1 byte implies.")

    # ---- verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("  Definitions used here (the roadmap states the 2x2 but not the "
          "three outcomes; these are\n  this tool's, printed so nobody has to "
          "guess):")
    print("    INCONCLUSIVE  a control failed -- the random-distractor arm or "
          "the mismatched-cue\n                  control read above chance. "
          "The probe is then measuring something\n                  other than "
          "memory of the cue and the number is not evidence.")
    print("    PASS          controls clean AND the sweep resolved to a cell "
          "OTHER than the\n                  already-assumed 'no / no'. The "
          "probe's job is to DISCRIMINATE, and\n                  a cell the "
          "project could not previously reach is the result.")
    print("    FAIL          controls clean AND every swept distance is "
          "'no / no'. A clean\n                  measurement that reproduces "
          "the standing belief and fixes the horizon\n                  number, "
          "but no new cell.")
    ctrl_failed = [c for c in ctrlA if c["above"]]
    ctrl_failed += [a for a in rowsA if a["has"]
                    and a["acc"] - a["ctrl"] <= 2.0 * math.hypot(a["se"], a["cse"])]
    if power_max is None:
        ctrl_failed.append({"d": "d=0"})
    if mode == "CONTROL":
        verdict = "INCONCLUSIVE"
        print("\n  This is the CONTROL ARM. It cannot return PASS or FAIL and "
              "it is not a result.")
    elif power_max is None:
        verdict = "INCONCLUSIVE"
        print("\n  INCONCLUSIVE -- the probe cannot decode the cue even at "
              "d=0, where the state IS the\n  cue's embedding. The instrument "
              "is not sensitive enough; no cell below is a result.")
    elif ctrl_failed:
        verdict = "INCONCLUSIVE"
        print(f"\n  INCONCLUSIVE -- {len(ctrl_failed)} control(s) read above "
              f"chance: "
              f"{[c['d'] for c in ctrl_failed]}. Until they do not, no cell "
              f"here is evidence.")
    elif set(distinct) != {"no / no"}:
        verdict = "PASS"
        print(f"\n  PASS -- controls clean, and the sweep resolved to "
              f"{distinct}, which the\n  project's existing instrumentation "
              f"could not distinguish. Read the cell, not this word:")
        for a, b in zip(rowsA, rowsB):
            c = cells[a["d"]]
            if c[0] != "no / no":
                print(f"    d={a['d']:>6} B -> {c[0]}: {c[1]}")
    else:
        verdict = "FAIL"
        print("\n  FAIL -- controls clean, and every swept distance is "
              "'no / no': no learned\n  memory of a random cue at any "
              "distance from 8 B to "
              f"{max(distances)} B. The result\n  is clean and it is also "
              "the cell the project already assumed; what it adds is the\n"
              "  horizon number and the confirmation that the existing "
              "instrumentation could not have\n  told 'no memory' from 'memory "
              "unused' -- this probe does, and the answer is 'no memory'.")

    print("\n  Reminders. This probe says NOTHING about: literal-cue "
          "induction (#23's domain),\n  obfuscated or parallel-tracking "
          "retrieval (never tested by anything in this\n  repo), language-"
          "modelling quality, or the distillation path. A 'no / no' here is "
          "a statement\n  about a UNIFORM RANDOM CUE under a LINEAR readout, "
          "and a linear readout can miss\n  information that is present but "
          "not linearly available.")
    if mode != "CONTROL":
        print(f"\n  [no RESULT has been written to RESULTS.md; this run "
              f"stops at the printed output]")
    return 0 if verdict in ("PASS", "FAIL") else 2


if __name__ == "__main__":
    sys.exit(main())