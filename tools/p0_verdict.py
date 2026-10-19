#!/usr/bin/env python3
"""P0 item 3 -- the credit-quality gate. Read three legs, render one verdict.

    .venv/bin/python tools/p0_verdict.py \
        --arm trace  runs/p0/trace_t010000.npz  runs/p0/trace.telemetry.jsonl  620.4 \
        --arm cand   runs/p0/cand_t010000.npz   runs/p0/cand.telemetry.jsonl   618.9

    .venv/bin/python tools/p0_verdict.py --eval-args "--slices 1 --bytes 800" \
        --dc-args "--dc-bytes 4000"    # smoke only: not a ruler-grade setting

WHAT THIS IS. The harness for ROADMAP P0 item 3: a matched from-scratch
A/B, judged at EQUAL WALL-CLOCK, scored on three legs -- gradient SNR, the
decay histogram, and a memory ruler. All three are required.

WHAT IT REFUSES TO DO, because each refusal is a result this repo already
had to learn:

1. **It will not score an arm against the incumbent.** Arm-vs-arm only. A
   from-scratch arm is missing the incumbent's 64,747 steps of history, so an
   absolute comparison is not a verdict, it is a subtraction of two different
   experiments. (Same discipline as tools/fs_verdict.py.)
2. **It will not call PASS if the two arms were not judged at equal
   wall-clock.** Equal steps is not equal compute: #23's A/B style exists
   precisely because at matched steps a slower arm gets more bytes. The arm
   seconds are printed with every verdict, and a gap over --wall-tol is a
   hard FAIL, not a caveat.
3. **It will not accept a cosine as evidence.** #23 measured that gradient
   cosines do not predict training outcome ("a cosine of 0.97 against full
   BPTT is apparently not the quantity that decides the outcome"), and the
   2026-10-04 trace gate measured cosine 1.000000 with no SNR consequence.
   There is no --vs-bptt flag here and there will not be one.
4. **It will not print a bare number.** Every figure carries its estimator
   definition and its floor. Two rigs disagreed about MI coverage for a day
   (85.2% at floor 0 vs 64.38% at floor 0.05, byte-identical input) because
   coverage is definition x floor. The same trap applies to SNR (which
   depends on the telemetry interval and the EMA warm-up), to the decay
   histogram (which depends on ring-K and the 8 B threshold) and to the ruler
   (which depends on slice count and bytes per slice). Any of those printed
   without its definition is a defect in this file.
5. **It cannot be talked into PASS.** Each leg is a boolean with an
   explicit, printed bar. If a leg's inputs are missing the leg is
   UNMEASURED, and UNMEASURED is never PASS. The tri-state verdict is
   PASS / FAIL / PARTIAL(ruler leg absent) and they are NOT collapsed.

THE THREE LEGS (ROADMAP P0 item 3 and S6(a)/(c)):

  SNR    candidate's gradient SNR is measurably above the trace trainer's at
         the same wall-clock, on BOTH the shared trunk and the decay role.
         "Measurably" is a PAIRED test across telemetry emissions
         (mean delta > 2 x SE of the paired deltas), not a magic constant.
  DECAY  the decay histogram has a real long tail. Incumbent reference:
         mean half-life 1.14 B and 7 of 8192 dims above 8 B -- the "~24 dims"
         figure sometimes quoted is NOT supported by RESULTS.md.
  RULER  no CE regression beyond the 0.03 bar and no distance-curve
         regression. 0.03 is the repo's existing real-difference line
         (fs_verdict.BAR), not one invented here.

THE NON-SUCCESS BRANCH IS PRE-COMMITTED (ROADMAP S1): if SNR moves but CE
and the ruler do not, P0 is FALSIFIED, estimator archaeology stops, and the
roadmap proceeds to P1. That is a defined outcome of this gate, not a failure
of the run, and this script prints it as a first-class result.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, HERE)

import numpy as np

# The 0.03 CE bar is the repo's existing real-difference line. It is READ OUT
# OF tools/fs_verdict.py rather than retyped, so the arm-vs-arm scorer and
# this gate cannot drift apart by one digit without anyone noticing.
def _fs_bar():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_fs_verdict", os.path.join(HERE, "tools", "fs_verdict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return float(mod.BAR)


try:
    CE_BAR = _fs_bar()
except Exception as _e:                        # pragma: no cover
    print(f"[warn] could not read the CE bar from tools/fs_verdict.py ({_e}); "
          f"using 0.03, which is what it holds today", file=sys.stderr)
    CE_BAR = 0.03
SNR_SIGMAS = 2.0          # paired-delta bar, stated in the printed definition
WALL_TOL = 0.02           # +/-2% of the slower arm; beyond that it is not a match
HL_LONG = 8.0             # "long tail" threshold, in bytes, from ROADMAP S6(c)
INCUMBENT_LONG_DIMS = 7   # RESULTS.md long run, of 8192. The "~24" is not supported.
INCUMBENT_MEAN_HL = 1.14

# Identical for every arm, like tools/fs_verdict.py. The eval args are part
# of the transcript cache key, so a transcript produced at a smaller setting
# can never be served for a ruler-grade request.
EVAL_ARGS = os.environ.get("TMT_FS_EVAL_ARGS", "").split() or [
    "--slices", "3", "--bytes", "20000", "--xd-bytes", "8000", "--xd-files", "4"]
DC_ARGS = os.environ.get("TMT_DC_ARGS", "").split() or [
    "--dc-bytes", "200000", "--dc-batch", "64"]


# --------------------------------------------------------------------------
# SNR: read from a telemetry JSONL, with the definition attached
# --------------------------------------------------------------------------
SNR_DEFN = (
    "snr = ||EMA(g)||^2 / EMA(||g||^2) over the flattened group, EMA decay 0.9\n"
    "        (telemetry.update_gradient_ema). Two floors bound it: 0 for a\n"
    "        zero-magnitude gradient group, 1 for a constant-direction one.\n"
    "        The EMA is updated ONCE PER EMISSION, i.e. once every\n"
    "        --telemetry-every STEPS -- not once per step. A number from a\n"
    "        100-step interval and a number from a 1000-step interval are\n"
    "        different quantities that both print as 'snr'. The EMA starts at\n"
    "        zero, so the first emissions are biased LOW by construction and\n"
    "        the warm-up emissions are discarded (count printed with the value).")


def read_telemetry(path, warmup):
    """Emission-indexed SNR series per parameter group, plus the wall clock."""
    if not path or not os.path.exists(path):
        return None
    out = {"t": [], "snr": {}, "seconds": None, "bytes": None, "every": None,
           "B": None, "ema_decay": 0.9, "n_discarded": warmup, "emissions": 0}
    first = last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("event") != "interval":
                continue
            out["t"].append(int(ev["t"]))
            out["bytes"] = int(ev["bytes_seen"])
            out["B"] = int(ev.get("B", 0)) or None
            if ev.get("throughput_measurement_seconds"):
                out["seconds"] = float(ev["throughput_measurement_seconds"])
            groups = ev.get("parameter_groups", {})
            for g, row in groups.items():
                if "grad_snr" in row:
                    out["snr"].setdefault(g, []).append(float(row["grad_snr"]))
            if out["every"] is None and len(out["t"]) > 1:
                out["every"] = out["t"][1] - out["t"][0]
            if first is None:
                first = (int(ev["t"]), ev["throughput_measurement_seconds"])
            last = (int(ev["t"]), ev["throughput_measurement_seconds"])
    if not out["t"]:
        return None
    out["emissions"] = len(out["t"])
    # Trunk = every non-decay group; decay = the 16 layer*.decay groups.
    out["trunk"] = np.array(out["snr"].get("shared_trunk", []), float)
    dec = [v for k, v in out["snr"].items() if k.endswith(".decay")]
    out["decay"] = np.array(np.mean(dec, axis=0), float) if dec else None
    return out


def snr_leg(cand, base, warmup):
    """Paired arm-vs-arm SNR test. Returns (verdict_bool, rows, notes)."""
    rows, ok = [], True
    for role, key in (("shared_trunk", "trunk"), ("decay role", "decay")):
        a, b = cand.get(key), base.get(key)
        if a is None or b is None or len(a) != len(b) or len(a) <= warmup:
            rows.append((role, None, None, None, None, "UNMEASURED"))
            ok = False
            continue
        d = a[warmup:] - b[warmup:]
        n = len(d)
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        mean = float(d.mean())
        passed = bool(n > 1 and mean > SNR_SIGMAS * se)
        rows.append((role, mean, se, n, passed,
                     None if passed else "not above the paired bar"))
        ok = ok and passed
    return ok, rows


# --------------------------------------------------------------------------
# decay histogram
# --------------------------------------------------------------------------
DECAY_DEFN = (
    "half-life(bytes) = ln(2) / -ln(sigmoid(decay)), one value per state\n"
    "        dimension; the population is the SAME 8,192 coordinates (16 layers\n"
    "        x 512 dims) in every arm, because --dim/--layers are matched.\n"
    "        FLOOR / definition to quote with every count: ring-K, the 8 B\n"
    "        edge, and the dimension total. 'hot dims' is a DIFFERENT\n"
    "        statistic -- it is sigma^K > 1%, i.e. depends on K, and K is read\n"
    "        from each checkpoint because the ring is sized from the init.")


def decay_leg(cand_ckpt, base_ckpt):
    def load(p):
        d = np.load(p)
        layers = sum(1 for k in d.files
                     if k.startswith("layer") and k.endswith(".decay"))
        sig = np.concatenate([1 / (1 + np.exp(-d[f"layer{i}.decay"]))
                              for i in range(layers)])
        hl = np.log(0.5) / np.log(sig)
        K = int(d["ring_k"]) if "ring_k" in d else None
        B = int(d["B"]) if "B" in d else None
        return {"hl": hl, "sig": sig, "K": K, "B": B, "dims": int(sig.size),
                "mean": float(hl.mean()),
                "gt4": int((hl > 4).sum()), "gt8": int((hl > 8).sum()),
                "gt16": int((hl > 16).sum()), "gt32": int((hl > 32).sum()),
                "hot": (int((sig ** K > 0.01).sum()) if K else None)}
    try:
        a, b = load(cand_ckpt), load(base_ckpt)
    except Exception as e:
        return None, [("decay histogram", f"UNMEASURED: {type(e).__name__}: {e}")]
    mism = [k for k in ("dims", "K", "B") if a[k] != b[k]]
    if mism:
        return None, [("dims > 8 B (the long tail)", None, None,
                       f"UNMEASURED: arms differ on {mism} "
                       f"({[(k, a[k], b[k]) for k in mism]}) -- the population "
                       f"or the ring is not the same, so a histogram "
                       f"comparison would be comparing two different "
                       f"estimators")]
    # A real long tail: strictly more dims past 8 B than the trace arm has.
    # "Meaningfully more than 7" (ROADMAP S6(c)) is checked against the
    # incumbent's 7 as well, and both numbers are printed.
    passed = a["gt8"] > b["gt8"]
    detail = []
    if a["gt8"] == b["gt8"]:
        detail.append(f"identical tail count ({a['gt8']} dims > {HL_LONG:.0f} B)")
    if a["gt8"] <= INCUMBENT_LONG_DIMS:
        detail.append(f"still at or below the incumbent's {INCUMBENT_LONG_DIMS}")
    return passed, [("dims > 8 B (the long tail)", a["gt8"], b["gt8"],
                     "; ".join(detail) or "strictly more than the trace arm")]


# --------------------------------------------------------------------------
# ruler: CE + the distance curve
# --------------------------------------------------------------------------
RULER_DEFN = (
    "held-out CE is the mean over --slices disjoint windows of --bytes bytes\n"
    "        in each held-out file, +/- SE across those slices. The bar is the\n"
    "        repo's existing 0.03 nats/byte real-difference line, applied\n"
    "        ARM-VS-ARM. Quote the slice count and the byte count with the\n"
    "        mean: 1 slice x 1000 B is a different measurement from\n"
    "        3 slices x 20000 B even when both print a mean.\n"
    "        distance curve: P(correctly closing a byte) vs the distance back\n"
    "        to its matching open byte, teacher-forced, against the model's own\n"
    "        mean probability of that closing byte as the no-memory FLOOR.")


def score(path):
    """Run tools/eval.py once per (checkpoint, eval args, dc args); cache it."""
    args = EVAL_ARGS + DC_ARGS
    key = hashlib.sha1((" ".join(args) + "|" + os.path.basename(path))
                       .encode()).hexdigest()[:8]
    cache_dir = os.path.join(HERE, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir,
                         f"{os.path.basename(path)}.p0verdict.{key}")
    if os.path.exists(cache) and os.environ.get("TMT_FS_NO_CACHE") != "1":
        out = open(cache).read()
    else:
        p = subprocess.run([sys.executable, os.path.join(HERE, "tools", "eval.py"),
                            path] + args, capture_output=True, text=True, cwd=HERE)
        if p.returncode != 0:
            return None, (p.stdout or "") + (p.stderr or "")
        out = p.stdout
        with open(cache, "w") as f:
            f.write(out)
    return out, None


def parse_eval(out):
    r = {}
    m = re.search(r"mean\s+(\S+) \+/- (\S+)\s+\(SE across (\d+) slices\)", out)
    if m:
        r["ind"] = (float(m.group(1)), float(m.group(2)), int(m.group(3)))
    m = re.search(r"held-out CE \(nats/byte; (\d+) slices, ([\d,]+) B\)", out)
    if m:
        r["n_slices"], r["n_bytes"] = int(m.group(1)), int(m.group(2).replace(",", ""))
    m = re.search(r"shift-distribution CE.*?mean\s+(\S+) \+/- (\S+)", out, re.S)
    if m:
        r["shift"] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"cross-domain CE.*?mean\s+(\S+) \+/- (\S+)", out, re.S)
    if m:
        r["xd"] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"steps \(t\)\s*:\s*([\d,]+)", out)
    r["t"] = int(m.group(1).replace(",", "")) if m else None
    m = re.search(r"half-life   mean=([0-9.]+) B \| hot dims \(sigma\^K>1%\): (\d+)", out)
    if m:
        r["mean_hl"], r["hot"] = float(m.group(1)), int(m.group(2))
    m = re.search(r"ring       : K=(\d+)", out)
    r["K"] = int(m.group(1)) if m else None
    # distance curve, per class
    dc = {}
    cls = None
    for line in out.splitlines():
        m = re.match(r"\s*--- class (\w+)(?: \| no-memory floor .* = ([\d.eE+-]+))?", line)
        if m:
            cls = m.group(1)
            dc[cls] = {"floor": float(m.group(2)) if m.group(2) else None,
                       "bins": [], "xover": None, "pairs": None}
            continue
        if cls is None:
            continue
        m = re.match(r"\s*\[(\d+),(inf|\d+)\)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", line)
        if m:
            hi = m.group(2)
            dc[cls]["bins"].append({
                "lo": int(m.group(1)), "hi": (math.inf if hi == "inf" else int(hi)),
                "n": int(m.group(3)),
                "mean": (float(m.group(4)) if m.group(4) != "no" else None),
                "se": (float(m.group(5)) if m.group(5) not in ("n/a", "no") else None)})
            continue
        m = re.search(r"DC-XOVER (\w+): .*?-> up to d < (\S+)", line)
        if m and m.group(1) in dc:
            v = m.group(2)
            dc[m.group(1)]["xover"] = (math.inf if v.startswith("inf") else int(v))
        m = re.search(r"DC-PAIRS (\w+): (\d+)", line)
        if m and m.group(1) in dc:
            dc[m.group(1)]["pairs"] = int(m.group(2))
    r["dc"] = dc
    return r


def ruler_leg(cand, base):
    """(bool, rows). None rows -> the leg is UNMEASURED, never PASS."""
    rows, ok = [], True
    # --- CE
    for key, label in (("ind", "held-out CE (in-domain)"),
                       ("shift", "shift-distribution CE"),
                       ("xd", "cross-domain CE (novels)")):
        a, b = cand.get(key), base.get(key)
        if a is None or b is None:
            rows.append((label, None, None, None, None, "UNMEASURED"))
            ok = False
            continue
        d = a[0] - b[0]
        passed = d <= CE_BAR
        rows.append((label, d, a[0], b[0], passed,
                     "" if passed else f"regresses the 0.03 bar by {d - CE_BAR:+.4f}"))
        ok = ok and passed
    # --- distance curve
    a, b = cand.get("dc", {}), base.get("dc", {})
    shared = sorted(set(a) & set(b))
    if not shared:
        rows.append(("distance curve", None, None, None, None,
                     "UNMEASURED: no class in both arms (was --dc-bytes > 0?)"))
        return False, rows
    for cls in shared:
        for rb, ba in zip(a[cls]["bins"], b[cls]["bins"]):
            if rb["lo"] != ba["lo"] or rb["n"] == 0 or ba["n"] == 0 \
                    or rb["mean"] is None or ba["mean"] is None:
                continue
            d = rb["mean"] - ba["mean"]
            # Independent-means SE of the difference. The two arms are scored
            # on the SAME pairs, so a paired SE would be tighter; this is the
            # conservative choice and is labelled as such below.
            sed = math.sqrt((rb["se"] or 0) ** 2 / max(1, rb["n"])
                            + (ba["se"] or 0) ** 2 / max(1, ba["n"]))
            passed = d >= -SNR_SIGMAS * sed
            ok = ok and passed
            rows.append((f"p_close[{rb['lo']},{rb['hi']:g}) {cls} (n={rb['n']})",
                         d, rb["mean"], ba["mean"], passed,
                         "" if passed else "regression beyond 2x SE of the difference"))
        xa, xb = a[cls]["xover"], b[cls]["xover"]
        if xa is not None and xb is not None:
            passed = xa >= xb
            ok = ok and passed
            rows.append((f"DC-XOVER {cls} (distance, bytes)", xa - xb, xa, xb, passed,
                         "" if passed else "candidate's curve reaches less far"))
    return ok, rows


# --------------------------------------------------------------------------
def fmt(v, spec="{:+.4f}"):
    if v is None:
        return "     n/a"
    if isinstance(v, float) and math.isinf(v):
        return "     inf"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    return spec.format(v)


def wall_row(name, secs, last_t, nbytes, B, t):
    return (f"  {name:<10s} {secs:>10.1f} s  last emission t={last_t:>9,}  "
            f"{nbytes:>13,} B  B={B if B else '?':<5} checkpoint t={t if t else '?':>9,}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="+", action="append",
                    metavar="NAME CKPT [TELEMETRY_JSONL] [WALL_SECONDS]",
                    help="one arm; >= 2 required. WALL_SECONDS is the arm's own "
                         "measured training time and is REQUIRED for an "
                         "equal-wall-clock verdict")
    ap.add_argument("--snr-warmup", type=int, default=10,
                    help="telemetry emissions discarded as EMA warm-up (the "
                         "EMA starts at zero, so early SNR is biased low)")
    ap.add_argument("--baseline", default=None,
                    help="arm NAME to compare against; default = first --arm")
    ap.add_argument("--eval-args", default=None,
                    help="override the ruler settings (must be identical for "
                         "every arm; this is the same contract as "
                         "tools/fs_verdict.py's EVAL_ARGS)")
    ap.add_argument("--dc-args", default=None)
    ap.add_argument("--label", default="P0 item 3 -- the credit-quality gate")
    args = ap.parse_args()

    global EVAL_ARGS, DC_ARGS
    if args.eval_args:
        EVAL_ARGS = args.eval_args.split()
    if args.dc_args:
        DC_ARGS = args.dc_args.split()

    arms = {}
    order = []
    for spec in args.arm:
        if len(spec) < 2:
            sys.exit("--arm needs at least NAME and CKPT")
        name, ckpt = spec[0], spec[1]
        tel = spec[2] if len(spec) > 2 and not spec[2].replace(".", "").isdigit() else None
        wall = None
        for tok in spec[2:]:
            if tok.replace(".", "").replace("-", "").isdigit():
                wall = float(tok)
        arms[name] = {"ckpt": ckpt, "tel": tel, "wall": wall}
        order.append(name)
    if len(arms) < 2:
        sys.exit(f"need >= 2 arms to compare; got {len(arms)}. The gate is "
                 f"arm-vs-arm by construction -- it does not score a "
                 f"from-scratch arm against the incumbent, which has 64,747 "
                 f"more steps of history.")
    base_name = args.baseline or order[0]
    cand_name = [n for n in order if n != base_name][0]
    cand, base = arms[cand_name], arms[base_name]

    # ---- refusals that are checks, not warnings ----
    for n, a in arms.items():
        if not os.path.exists(a["ckpt"]):
            sys.exit(f"arm {n}: no such checkpoint {a['ckpt']}")
    for tok in sys.argv:
        if tok.startswith("--") and ("cos" in tok or "bptt" in tok):
            sys.exit("this gate has no cosine input and never had: #23 "
                     "measured that gradient cosines do not predict training "
                     "outcome, and the 2026-10-04 trace gate measured cosine "
                     "1.000000 with no SNR consequence. Do not reintroduce it.")

    print("=" * 78)
    print(args.label)
    print("=" * 78)
    print(f"  baseline  : {base_name}  ({os.path.basename(base['ckpt'])})")
    print(f"  candidate : {cand_name}  ({os.path.basename(cand['ckpt'])})")
    print(f"  ruler     : identical for every arm: tools/eval.py "
          f"{' '.join(EVAL_ARGS)} {' '.join(DC_ARGS)}")
    print(f"  scoring rule: ARM-VS-ARM only. Not arm-vs-incumbent: a "
          f"from-scratch arm is\n                missing the incumbent's "
          f"64,747 steps of history, so that subtraction would be a "
          f"comparison of two\n                different experiments, not a "
          f"verdict.")
    print(f"  cosine    : NOT an input to any leg below (see the refusal note "
          f"in the docstring).")

    # ---- equal wall-clock ----
    print("\n" + "-" * 78)
    print("MATCH DISCIPLINE -- equal WALL-CLOCK, the #23 A/B style")
    print("-" * 78)
    wall_rows, walls = [], {}
    for n in order:
        a = arms[n]
        t = read_telemetry(a["tel"], args.snr_warmup)
        walls[n] = a["wall"] if a["wall"] is not None else (
            t["seconds"] if t else None)
        wall_rows.append(wall_row(
            n, walls[n] or 0.0, t["t"][-1] if t else 0,
            t["bytes"] if t else 0, t["B"] if t else None,
            int(np.load(a["ckpt"])["t"]) if os.path.exists(a["ckpt"]) else None))
    for line in wall_rows:
        print(line)
    have_wall = all(walls[n] for n in order)
    if not have_wall:
        print("  [warn] at least one arm has no measured wall-clock; the "
              "equal-time claim is UNVERIFIED for it")
    else:
        lo, hi = min(walls.values()), max(walls.values())
        gap = (hi - lo) / hi
        print(f"  spread: {hi - lo:.1f} s = {100 * gap:.2f}% of the slower arm "
              f"(tolerance +/-{100 * WALL_TOL:.0f}%)")
    wall_ok = have_wall and \
        (max(walls.values()) - min(walls.values())) <= WALL_TOL * max(walls.values())

    # ---- leg 1: SNR ----
    print("\n" + "-" * 78)
    print("LEG 1 of 3 -- GRADIENT SNR  (required)")
    print("-" * 78)
    print("  DEFINITION + FLOORS:")
    for line in SNR_DEFN.splitlines():
        print("    " + line)
    print(f"  BAR: paired across emissions, mean(candidate - baseline) must "
          f"exceed {SNR_SIGMAS:g} x SE(paired deltas),\n       on BOTH the "
          f"shared trunk and the decay role. Both are required; the standing\n"
          f"       blocker (SNR ~ 0.05-0.07) has never been reported on the "
          f"trunk alone.")
    tc = read_telemetry(cand["tel"], args.snr_warmup)
    tb = read_telemetry(base["tel"], args.snr_warmup)
    snr_rows, snr_ok = [], False
    if tc is None or tb is None:
        missing = [n for n in order
                   if read_telemetry(arms[n]["tel"], args.snr_warmup) is None]
        print(f"  UNMEASURED: no usable telemetry for {missing}. Gradient SNR "
              f"is a training-time\n              quantity; it cannot be "
              f"recovered from a checkpoint after the fact, and an\n"
              f"              UNMEASURED leg is not a zero.")
        un = "UNMEASURED: no telemetry"
        snr_rows = [("shared_trunk", None, None, None, None, un),
                    ("decay role", None, None, None, None, un)]
    else:
        if tc["t"] != tb["t"]:
            print(f"  [warn] the two arms did not emit at the same step "
                  f"counts; pairing is by emission INDEX, not by t, and the "
                  f"paired SE below is then optimistic")
        print(f"  emissions: {tc['emissions']} per arm, interval "
              f"{tc['every']} steps, {args.snr_warmup} discarded as EMA "
              f"warm-up -> {max(0, tc['emissions'] - args.snr_warmup)} used")
        print(f"  B/s and total seconds come from telemetry's own clock: "
              f"{tc['seconds']} s (candidate) vs {tb['seconds']} s (baseline)")
        snr_ok, snr_rows = snr_leg(tc, tb, args.snr_warmup)
    print(f"\n  {'group':<16s} {'delta':>10s} {'SE(paired)':>12s} {'n':>5s} "
          f"{'pass':>6s}   note")
    for role, mean, se, n, passed, note in snr_rows:
        nn = n if isinstance(n, int) else (n if n is not None else "-")
        print(f"  {role:<16s} {fmt(mean):>10s} {fmt(se):>12s} {str(nn):>5s} "
              f"{fmt(passed):>6s}   {note or ''}")
    for role in ("trunk", "decay"):
        if tc and role == "trunk" and tc.get("trunk") is not None:
            a, b = tc["trunk"][args.snr_warmup:], tb["trunk"][args.snr_warmup:]
            print(f"  {role} level: candidate {a.mean():.4f} vs baseline "
                  f"{b.mean():.4f} (n={len(a)} emissions, warm-up "
                  f"{args.snr_warmup} discarded)")

    # ---- leg 2: decay histogram ----
    print("\n" + "-" * 78)
    print("LEG 2 of 3 -- DECAY HISTOGRAM  (required)")
    print("-" * 78)
    print("  DEFINITION + FLOORS:")
    for line in DECAY_DEFN.splitlines():
        print("    " + line)
    print(f"  BAR: strictly more dims past {HL_LONG:.0f} B than the trace arm. "
          f"Incumbent reference: mean half-life\n"
          f"       {INCUMBENT_MEAN_HL} B and {INCUMBENT_LONG_DIMS} of 8192 dims "
          f"above 8 B. The '~24 dims' figure sometimes quoted is NOT "
          f"supported\n       by RESULTS.md and is not used here.")
    dec_ok, dec_rows = decay_leg(cand["ckpt"], base["ckpt"])
    print(f"\n  {'statistic':<34s} {'cand':>8s} {'trace':>8s}   note")
    for label, a, b, note in dec_rows:
        print(f"  {label:<34s} {str(a):>8s} {str(b):>8s}   {note}")
    for side, p in (("cand", cand["ckpt"]), ("trace", base["ckpt"])):
        d = np.load(p)
        layers = sum(1 for k in d.files
                     if k.startswith("layer") and k.endswith(".decay"))
        sig = np.concatenate([1 / (1 + np.exp(-d[f"layer{i}.decay"]))
                              for i in range(layers)])
        hl = np.log(0.5) / np.log(sig)
        K = int(d["ring_k"]) if "ring_k" in d else 0
        B = int(d["B"]) if "B" in d else 0
        hist = " | ".join(f"{int(((hl > a) & (hl <= b)).sum())}"
                          for a, b in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16),
                                       (16, 32), (32, math.inf)))
        print(f"  {side:<6} mean half-life {hl.mean():.3f} B | >4B "
              f"{int((hl > 4).sum())} | >8B {int((hl > 8).sum())} | >16B "
              f"{int((hl > 16).sum())} | >32B {int((hl > 32).sum())} | "
              f"hot(sigma^{K}>1%) {int((sig ** K > 0.01).sum())} | K={K} "
              f"B={B} | {sig.size} dims")
        print(f"         half-life hist (<=1,<=2,<=4,<=8,<=16,<=32,>32 B): {hist}")

    # ---- leg 3: the ruler ----
    print("\n" + "-" * 78)
    print("LEG 3 of 3 -- MEMORY RULER (CE + distance curve)  (required)")
    print("-" * 78)
    print("  DEFINITION + FLOORS:")
    for line in RULER_DEFN.splitlines():
        print("    " + line)
    print(f"  BAR: CE delta <= {CE_BAR} nats/byte arm-vs-arm (the repo's "
          f"existing real-difference line), and\n       no distance-curve bin "
          f"regressing by more than {SNR_SIGMAS:g} x SE of the difference.")
    print("  Running the ruler on every arm (this is the slow part: the "
          "distance curve is opt-in CPU)...")
    from concurrent.futures import ThreadPoolExecutor
    scored = {}
    with ThreadPoolExecutor(max_workers=min(4, len(arms))) as ex:
        futs = {ex.submit(score, arms[n]["ckpt"]): n for n in order}
        for fut, n in futs.items():
            try:
                out, err = fut.result()
            except Exception as e:
                out, err = None, f"{type(e).__name__}: {e}"
            if out is None:
                print(f"  {n}: ruler FAILED: {err}")
            else:
                scored[n] = parse_eval(out)
    ruler_ok, ruler_rows = None, []
    if len(scored) == len(arms):
        ruler_ok, ruler_rows = ruler_leg(scored[cand_name], scored[base_name])
    else:
        ruler_rows = [("memory ruler", None, None, None, None,
                       "UNMEASURED: at least one arm could not be scored")]

    if scored:
        print(f"  sample size: {scored[base_name].get('n_slices')} windows x the "
              f"requested byte count = {scored[base_name].get('n_bytes'):,} B "
              f"scored in-domain\n  (settings: {' '.join(EVAL_ARGS)} "
              f"{' '.join(DC_ARGS)}; identical for every arm and part of the "
              f"transcript cache key)")
    print(f"\n  {'ruler cell':<40s} {'delta':>10s} {'cand':>10s} "
          f"{'trace':>10s} {'pass':>6s}   note")
    for label, d, a, b, passed, note in ruler_rows:
        print(f"  {label:<40s} {fmt(d):>10s} {fmt(a, '{:>10.4f}'):>10s} "
              f"{fmt(b, '{:>10.4f}'):>10s} {fmt(passed):>6s}   {note or ''}")
    if not scored:
        print("  NOTE: the distance-curve leg is what makes this a memory "
              "ruler rather\n        than a loss readout. If the run was made "
              "without --dc-bytes, the ruler\n        is PARTIAL, not PASS.")

    # ---- verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    legs = [("SNR (gradient SNR improved)", snr_ok, bool(tc and tb)),
            (f"DECAY (a real long tail, >{HL_LONG:.0f} B)", dec_ok, dec_ok is not None),
            ("RULER (no CE or distance-curve regression)", ruler_ok, ruler_ok is not None)]
    for name, ok, measured in legs:
        state = ("PASS" if ok else "FAIL") if measured else "UNMEASURED"
        print(f"  {name:<46s} {state}")
    if not wall_ok:
        print(f"  {'EQUAL WALL-CLOCK (a precondition, not a leg)':<46s} "
              f"{'FAIL' if have_wall else 'UNMEASURED'}")
    else:
        print(f"  {'EQUAL WALL-CLOCK (a precondition, not a leg)':<46s} PASS")

    unmeasured = [n for n, _ok, m in legs if not m]
    ruler_name = "RULER (no CE or distance-curve regression)"
    if unmeasured:
        verdict = "PARTIAL"
        head = ("PARTIAL -- ruler leg absent" if unmeasured == [ruler_name]
                else "PARTIAL -- " + ", ".join(unmeasured) + " absent")
    elif all(ok for _n, ok, _m in legs) and wall_ok:
        verdict, head = "PASS", "PASS"
    else:
        verdict, head = "FAIL", "FAIL"
    print(f"\n  OVERALL: {head}")
    print(f"  This is one of three distinct outcomes (PASS / FAIL / "
          f"PARTIAL-ruler-absent) and is\n  not to be read as any other.")

    if verdict == "PASS":
        print("\n  P0 item 3 PASSES: the candidate buys gradient SNR at equal "
              "wall-clock without\n  regressing the rulers, and the decay "
              "histogram has a real tail. ROADMAP S6(a)/(c)\n  are met. What "
              "this does NOT establish: that the gain comes from credit "
              "quality\n  rather than from anything else the two arms "
              "differ in -- a single paired run\n  separates the arms, not "
              "the mechanism.")
    else:
        if snr_ok and ruler_ok is False:
            print("\n  PRE-COMMITTED BRANCH FIRES (ROADMAP S1, 'a defined "
                  "non-success'):\n    SNR moved but the rulers did NOT "
                  "follow. This is the shape #23 observed. Per the roadmap:\n"
                  "      - P0 IS FALSIFIED as the explanation for the memory "
                  "wall;\n      - STOP estimator archaeology. Do not add more "
                  "credit-assignment work;\n      - proceed to P1 and record "
                  "the falsification in RESULTS.md as a first-class\n"
                  "        result, not as a failure of the run.\n"
                  "  This is a defined outcome of the gate, not a broken run.")
        elif not snr_ok:
            print("\n  SNR did not move measurably. The gate is FAIL, and the "
                  "roadmap's\n  pre-committed branch does not fire (it is "
                  "conditioned on SNR moving and the\n  rulers not). "
                  "Estimator work stays closed; P1 is unblocked either way.")
        if unmeasured:
            print("  Any UNMEASURED leg is not a zero. It is a missing "
                  "measurement, and it cannot\n  be reported as a pass.")
    print("\n  Reminder of what this gate is NOT: a cosine (measured useless "
          "for predicting\n  outcome, #23), an arm-vs-incumbent comparison "
          "(the from-scratch arms are\n  64,747 steps behind), or a claim "
          "about obfuscated / parallel-tracking\n  retrieval (never tested; "
          "see the scope note printed by the distance-curve cell).")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
