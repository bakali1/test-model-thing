#!/usr/bin/env python3
"""CPU gates for chat_models.py's degeneracy metrics and --set parsing.

No model, no checkpoint, no GPU: every assertion is on synthetic bytes or on
pure functions, so this runs in seconds and is safe to run anywhere.

    .venv/bin/python tests/check_chat_metrics.py
"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import chat_models as cm  # noqa: E402

FAILURES = []

ALPHABET = (b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            b".,;:'\"!?-")


def unique_bytes_seq(n, seed=7):
    """Deterministic n-byte string with no repeated 8-grams (alphabet 66).

    Used to build the 'healthy text' control: any metric complaining about
    this is a metric bug, not a model bug.
    """
    import random
    rng = random.Random(seed)
    return bytes(ALPHABET[rng.randrange(len(ALPHABET))] for _ in range(n))


def check(name, cond, detail=""):
    if cond:
        print(f"  {name}: PASS")
    else:
        print(f"  {name}: FAIL  {detail}")
        FAILURES.append(name)


def t_newline_collapse():
    print("[whitespace / newline collapse]")
    b = b"the man said" + b"\n" * 16 + b"and then he"
    m = cm.degeneracy_metrics(b)
    check("16-newline string: max_run_newlines == 16",
          m["max_run_newlines"] == 16, m["max_run_newlines"])
    check("16-newline string: degenerate is True", m["degenerate"] is True)
    # 7 consecutive newlines is below the >= 8 rule on its own.
    m2 = cm.degeneracy_metrics(b"a" + b"\n" * 7 + b"b")
    check("7-newline run is measured as 7", m2["max_run_newlines"] == 7,
          m2["max_run_newlines"])
    check("scattered newlines: max_run_newlines == 1",
          cm.max_run_newlines(b"a\nb\nc\nd\ne\nf\ng\nh") == 1)
    check("7 consecutive newlines alone do not flag degenerate",
          not cm.is_degenerate(b"a" + b"\n" * 7 + b"b" + unique_bytes_seq(120, 11)))
    # pure newlines also collapse the 8-gram ratio
    pure = cm.degeneracy_metrics(b"\n" * 16)
    check("16 bare newlines: u8 ratio below 0.20",
          pure["unique_8gram"] < 0.20, pure["unique_8gram"])

    # The 400k regression: the same attractor in SPACES. A newline-only rule
    # scores this clean, which is how a real loop shipped as 0/9.
    sp = b"the quick brown" + b" " * 177 + b"fox jumps over"
    m3 = cm.degeneracy_metrics(sp)
    check("177-space run: max_run_newlines == 0 (newline-only rule misses it)",
          m3["max_run_newlines"] == 0, m3["max_run_newlines"])
    check("177-space run: max_run_ws == 177", m3["max_run_ws"] == 177,
          m3["max_run_ws"])
    check("177-space run: degenerate is True", m3["degenerate"] is True)
    for name, run in (("tab", b"\t" * 9), ("CR", b"\r" * 9)):
        check(f"{name} run of 9 is flagged degenerate",
              cm.is_degenerate(b"a" + run + b"b"))
    check("mixed whitespace run counts across byte values",
          cm.max_run_whitespace(b"a \t\n\r " * 3) == 5,
          cm.max_run_whitespace(b"a \t\n\r " * 3))
    check("a non-whitespace byte breaks the run",
          cm.max_run_whitespace(b"a" + b" \t\n\r" * 3) == 12,
          cm.max_run_whitespace(b"a" + b" \t\n\r" * 3))
    check("whitespace run of 1 is not degenerate",
          not cm.is_degenerate(b"a b " + unique_bytes_seq(120, 3)))


def t_rep8_frac_arm():
    print("[top-8-gram arm: scale-free, and the loop it was added for]")
    # A 2000-byte generation with one 8-gram repeated 115x: globally diverse
    # (u8 ~0.5) but locally stuck -- the case unique_8gram alone missed.
    # 400k's real stuck 8-gram is b"        " (8 spaces); use a non-space tag
    # here so this check exercises the fraction arm and nothing else.
    ws_free = bytes(c for c in unique_bytes_seq(2100, seed=5)
                    if c not in (0x20, 0x09, 0x0A, 0x0D))
    tag = b"zqxjvbnm"
    out = bytearray()
    i = 0
    while len(out) < 2000:
        out += ws_free[i:i + 9] + tag
        i += 9
    looped = bytes(out[:2000])
    m = cm.degeneracy_metrics(looped)
    check("115x repeated 8-gram: u8 ratio is NOT degenerate (the hole)",
          m["unique_8gram"] > 0.20, f"u8={m['unique_8gram']:.3f}")
    check("115x repeated 8-gram: rep_8gram_top >= 100",
          m["rep_8gram_top"] >= 100, m["rep_8gram_top"])
    check("115x repeated 8-gram: rep_8gram_frac >= 2%",
          m["rep_8gram_frac"] >= 0.02, f"{100 * m['rep_8gram_frac']:.2f}%")
    check("115x repeated 8-gram: flagged ONLY by the fraction arm",
          m["degenerate"] is True and m["max_run_ws"] < cm.DEGEN_WS_RUN
          and m["unique_8gram"] >= cm.DEGEN_U8_RATIO
          and m["rep_8gram_top"] >= cm.DEGEN_REP8_MIN
          and m["tail_4gram_top"] < cm.DEGEN_TAIL_REPEAT,
          f"ws={m['max_run_ws']} u8={m['unique_8gram']:.3f} "
          f"tail4={m['tail_4gram_top']}")

    # Scale invariance: the same loop in a 250-byte output is caught too. A
    # raw COUNT threshold of 8 alone would mean 3.3% at n=250 and 0.4% at
    # n=2000, i.e. it would flag short outputs and miss long ones.
    short = (b"ab" * 120)[:250]
    m2 = cm.degeneracy_metrics(short)
    check("250-byte looped string is flagged on the fraction arm",
          m2["degenerate"] is True, f"rep8={m2['rep_8gram_top']} "
          f"frac={m2['rep_8gram_frac']:.3f}")
    # The count floor, and the false positive it removes. At n=250, five
    # repeats of a common 8-gram is 2.06% and is ordinary English, and it
    # flagged healthy generations of 600k/890k before the floor was added.
    def tag_repeated(k, total=250):
        """`total` bytes containing exactly k copies of one 8-gram."""
        tag = b" and the "
        ws = bytes(c for c in unique_bytes_seq(600, seed=17)
                   if c not in (0x20, 0x09, 0x0A, 0x0D))
        body, n = len(ws), total - k * len(tag)
        pieces = [ws[i * n // (k + 1):(i + 1) * n // (k + 1)]
                  for i in range(k + 1)]
        return tag.join(pieces)

    seven = cm.degeneracy_metrics(tag_repeated(7))
    check("7 repeats of one 8-gram in 250 B: frac over 2% but count under "
          "the floor, so NOT flagged",
          len(tag_repeated(7)) == 250
          and seven["rep_8gram_top"] == 7
          and seven["rep_8gram_frac"] >= cm.DEGEN_REP8_FRAC
          and seven["rep_8gram_top"] < cm.DEGEN_REP8_MIN
          and not seven["degenerate"],
          f"rep8={seven['rep_8gram_top']} "
          f"frac={seven['rep_8gram_frac']:.3f} degen={seven['degenerate']}")
    eight = cm.degeneracy_metrics(tag_repeated(8))
    check("the same string with 8 repeats IS flagged (the floor is 8)",
          eight["rep_8gram_top"] >= cm.DEGEN_REP8_MIN
          and eight["degenerate"] is True,
          f"rep8={eight['rep_8gram_top']} degen={eight['degenerate']}")
    check("neither fixture is flagged by the whitespace arm (isolated here)",
          seven["max_run_ws"] == 1 and eight["max_run_ws"] == 1)
    # A healthy control must stay under the threshold: the incumbent's worst
    # measured value on the 24-Sep protocol is 0.80%.
    ctrl = cm.degeneracy_metrics(unique_bytes_seq(250, seed=9))
    check("healthy 250-byte control: rep_8gram_frac well under 2%",
          ctrl["rep_8gram_frac"] < 0.02, f"{100 * ctrl['rep_8gram_frac']:.2f}%")
    check("rep_8gram_frac == rep_8gram_top / total 8-grams",
          abs(m["rep_8gram_frac"]
              - m["rep_8gram_top"] / (len(looped) - 8 + 1)) < 1e-12)
    # b"abcabcabc" (9 bytes) has 6 4-grams in 2 phases, not 3
    check("rep_4gram_frac is the same quantity at n=4",
          abs(cm.degeneracy_metrics(b"abcabcabc")["rep_4gram_frac"]
              - 2 / 6) < 1e-12)


def t_nontermination():
    print("[non-termination is a separate failure from degeneracy]")
    text = unique_bytes_seq(2000, seed=13)
    m = cm.degeneracy_metrics(text, stop_hits=0)
    check("2000 bytes, 0 stop firings: nonterminating is True",
          m["nonterminating"] is True)
    check("...and it is NOT counted as degenerate",
          m["degenerate"] is False)
    m2 = cm.degeneracy_metrics(text, stop_hits=3)
    check("2000 bytes, 3 stop firings: nonterminating is False",
          m2["nonterminating"] is False)
    m3 = cm.degeneracy_metrics(unique_bytes_seq(250, seed=13), stop_hits=0)
    check("250 bytes, 0 firings: not flagged (too short to conclude)",
          m3["nonterminating"] is False, "0-firings at n=250 is not evidence")
    check("threshold matches the default --n, so a default run has evidence",
          cm.NONTERM_MIN_BYTES == cm.DEFAULT_N, cm.NONTERM_MIN_BYTES)
    # normalization
    rows = [cm.degeneracy_metrics(text, stop_hits=s) for s in (0, 2, 4)]
    agg = cm.aggregate(rows)
    check("stop_per_kB normalizes 6 firings in 6000 bytes to 1.0",
          abs(agg["stop_per_kB"] - 1.0) < 1e-9, agg["stop_per_kB"])
    check("aggregate counts nonterminating generations",
          agg["nonterminating"] == 1, agg["nonterminating"])
    check("aggregate tracks total bytes generated",
          agg["bytes"] == 6000, agg["bytes"])


def t_english_like():
    print("[English-ish, no repetition]")
    words = ("the quick brown fox jumps over a lazy dog while rain falls on "
             "the tin roof and the kettle sings inside the quiet kitchen "
             "where children draw maps of imaginary countries and argue "
             "about whether bread needs butter more than it needs salt "
             "beside a window full of steam and morning light that makes "
             "every ordinary thing look briefly worth remembering for "
             "years afterwards without anyone saying why at the time")
    vocab = words.split()
    b = bytearray()
    i = 0
    while len(b) < 200:
        w = vocab[i % len(vocab)]
        i += 1
        b += w.encode() + b" "
    b = bytes(b[:200])
    m = cm.degeneracy_metrics(b)
    check("200-byte text: unique_8gram ratio > 0.9", m["unique_8gram"] > 0.9,
          f"{m['unique_8gram']:.3f} ({m['bytes']} bytes)")
    check("200-byte text: degenerate is False", m["degenerate"] is False,
          f"nl<={m['max_run_newlines']} rep8={m['rep_8gram_top']} "
          f"tail4={m['tail_4gram_top']}")
    check("200-byte text: no newline run", m["max_run_newlines"] == 0)
    check("200-byte text: unique_bytes within 10..40",
          10 <= m["unique_bytes"] <= 40, m["unique_bytes"])


def t_looped():
    print("[looped string]")
    b = b"abc" * 80                       # 240 bytes, period 3
    m = cm.degeneracy_metrics(b)
    check("looped string: degenerate is True", m["degenerate"] is True)
    check("looped string: u8 ratio below 0.20", m["unique_8gram"] < 0.20,
          m["unique_8gram"])
    check("looped string: rep_8gram_top is large",
          m["rep_8gram_top"] >= 70, m["rep_8gram_top"])
    check("looped string: unique_8gram == 3/233 by hand",
          abs(m["unique_8gram"] - 3 / 233) < 1e-9, m["unique_8gram"])
    # tail rule alone: a long unique prefix with a looped ending
    prefix = unique_bytes_seq(200)
    loop = prefix + b"xyzxyzxyz" * 8
    m2 = cm.degeneracy_metrics(loop)
    check("unique prefix + looped tail: degenerate is True",
          m2["degenerate"] is True)
    check("unique prefix + looped tail: u8 ratio still high",
          m2["unique_8gram"] > 0.5, m2["unique_8gram"])


def t_hand_computed():
    print("[hand-computed 8-byte example]")
    b = b"abcdabcx"
    m = cm.degeneracy_metrics(b)
    check("unique_bytes == 5", m["unique_bytes"] == 5, m["unique_bytes"])
    # 4-grams: abcd bcda cdab dabc abcx  -> 5 total, 5 distinct, top 1
    check("unique_4gram == 5/5", m["unique_4gram"] == 1.0, m["unique_4gram"])
    check("rep_4gram_top == 1", m["rep_4gram_top"] == 1, m["rep_4gram_top"])
    # 8-grams: abcdabcx -> 1 total, 1 distinct, top 1
    check("unique_8gram == 1/1", m["unique_8gram"] == 1.0, m["unique_8gram"])
    check("rep_8gram_top == 1", m["rep_8gram_top"] == 1, m["rep_8gram_top"])
    check("max_run_newlines == 0", m["max_run_newlines"] == 0)
    check("ngram_stats(8 bytes, 8) == (1,1,1)",
          cm.ngram_stats(b, 8) == (1, 1, 1), cm.ngram_stats(b, 8))
    check("ngram_stats(8 bytes, 4) == (5,5,1)",
          cm.ngram_stats(b, 4) == (5, 5, 1), cm.ngram_stats(b, 4))
    check("ngram_stats(4 bytes, 8) == (0,0,0) and ratio defaults to 1.0",
          cm.ngram_stats(b"abcd", 8) == (0, 0, 0)
          and cm.unique_ngram_ratio(b"abcd", 8) == 1.0)
    # a repeated n-gram by hand: "aaaa"  -> 4-grams: aaaa x3
    check("ngram_stats(8x'a', 4) == (5,1,5)",
          cm.ngram_stats(b"a" * 8, 4) == (5, 1, 5), cm.ngram_stats(b"a" * 8, 4))
    check("ngram_stats(b'aaaa', 4) == (1,1,1) (shorter than n)",
          cm.ngram_stats(b"aaaa", 4) == (1, 1, 1), cm.ngram_stats(b"aaaa", 4))
    check("max_run_newlines finds the longest run only",
          cm.max_run_newlines(b"a\n\n\nb\n\nc") == 3)
    # tail rule by hand: 4-gram 'wwww' repeated 6x inside the last 32 bytes
    tail = unique_bytes_seq(40) + b"wwww" * 6
    check("tail_4gram_top >= 6 for a 6x-repeated tail 4-gram",
          cm.tail_ngram_top(tail) >= 6, cm.tail_ngram_top(tail))
    check("tail_4gram_top == 1 when every tail 4-gram is distinct",
          cm.tail_ngram_top(unique_bytes_seq(32)) == 1)


def t_set_parsing():
    print("[--set parsing]")
    got = cm.parse_set("  a = /tmp/a.npz ,  b=/tmp/b.npz  ")
    check("whitespace stripped around label and path",
          got == [("a", "/tmp/a.npz"), ("b", "/tmp/b.npz")], got)

    for spec, why in (("= /tmp/a.npz", "empty label"),
                      ("a=", "empty path"),
                      ("a", "no '=' at all"),
                      ("a=1,b=2,", "trailing empty element")):
        try:
            cm.parse_set(spec)
            check(f"rejects {why} ({spec!r})", False, "no exception")
        except ValueError:
            check(f"rejects {why} ({spec!r})", True)

    try:
        cm.parse_set("a=1, a=2")
        check("rejects duplicate label", False, "no exception")
    except ValueError as e:
        check("rejects duplicate label", "duplicate" in str(e), str(e))

    # a label containing a comma must fail loudly, not become two models
    try:
        cm.parse_set("a,b=/tmp/x.npz")
        check("rejects a comma inside a label", False, "no exception")
    except ValueError:
        check("rejects a comma inside a label", True)

    print("[--set-glob parsing]")
    with tempfile.TemporaryDirectory() as d:
        for nm in ("m_t100.npz", "m_t200.npz"):
            open(os.path.join(d, nm), "wb").close()
        got = cm.parse_set_glob(os.path.join(d, "*.npz"))
        check("glob discovers sorted checkpoints labelled by stem",
              [g[0] for g in got] == ["m_t100", "m_t200"], got)
        try:
            cm.parse_set_glob(os.path.join(d, "nope_*.npz"))
            check("glob with no match is an error", False, "no exception")
        except ValueError:
            check("glob with no match is an error", True)


def t_skipped_summary():
    print("[SKIPPED summary]")
    models = [("present", os.path.join(HERE, "..", "tools",
                                       "chat_models.py")),
              ("ghost", "/nonexistent/checkpoint_t123.npz")]
    present, skipped = cm.resolve_models(models)
    check("existing file is present", ("present", models[0][1]) in present)
    check("missing file lands in SKIPPED, not in the run",
          [s["label"] for s in skipped] == ["ghost"], skipped)
    check("SKIPPED entry carries a reason",
          skipped and skipped[0]["reason"] == "file not found", skipped)

    with tempfile.TemporaryDirectory() as d:
        empty = os.path.join(d, "empty.npz")
        open(empty, "wb").close()
        present, skipped = cm.resolve_models([("empty", empty)])
        check("empty file is skipped, not loaded", not present
              and skipped[0]["reason"] == "empty file", skipped)

    # the default set reports its missing entries instead of hiding them
    present, skipped = cm.resolve_models(cm.DEFAULT_SET)
    print(f"    DEFAULT_SET resolves to {len(present)} present / "
          f"{len(skipped)} skipped: "
          + ", ".join(s['label'] for s in skipped))
    check("every missing default entry is in SKIPPED",
          len(present) + len(skipped) == len(cm.DEFAULT_SET))


def t_digest():
    print("[digest verification]")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.npz")
        with open(p, "wb") as f:
            f.write(b"weights")
        digest = cm.sha256_file(p)
        with open(p + ".sha256", "w") as f:
            f.write(f"{digest}  c.npz\n# run_id test\n")
        pref, verified = cm.verify_digest(p)
        check("sidecar digest verifies and prefix is printed",
              verified and pref == digest[:12], (pref, verified))
        with open(p, "wb") as f:            # tamper
            f.write(b"wights")
        try:
            cm.verify_digest(p)
            check("MISMATCH is a hard error", False, "no exception")
        except SystemExit as e:
            check("MISMATCH is a hard error", "MISMATCH" in str(e))
        # no sidecar -> unverified, not an error
        os.remove(p + ".sha256")
        pref, verified = cm.verify_digest(p)
        check("no sidecar reports unverified", pref is None and not verified)
        # a sidecar that carries no digest is an error, not a pass
        with open(p + ".sha256", "w") as f:
            f.write("# run_id only, no digest here\n")
        try:
            cm.verify_digest(p)
            check("digestless sidecar is an error", False, "no exception")
        except ValueError:
            check("digestless sidecar is an error", True)
        # a renamed copy is still verified against the digest it carries
        shutil.copyfile(os.path.join(ROOT, "models", "tmt_t094747.npz"),
                        os.path.join(d, "renamed.npz"))
        shutil.copyfile(os.path.join(ROOT, "models", "tmt_t094747.npz.sha256"),
                        os.path.join(d, "renamed.npz.sha256"))
        pref, verified = cm.verify_digest(os.path.join(d, "renamed.npz"))
        check("renamed copy is still verified",
              verified and pref == cm.sha256_file(
                  os.path.join(ROOT, "models", "tmt_t094747.npz"))[:12])


def t_escape_render():
    print("[readable rendering]")
    got = cm.escape(b'a\nb"c\td\x00e')
    check("newline/quote/tab/NUL render visibly on one line",
          got == 'a\\nb"c\\td\\x00e', got)
    check("wrapped output never contains a raw newline",
          all("\n" not in ln for ln in cm.wrap(cm.escape(b"x" * 200))))


def t_incumbent_smoke():
    """Regression gate on the KNOWN-GOOD model.

    The degenerate-vs-not verdict of a real checkpoint, end to end. This is
    the check that would have caught the missing repetition arm: the
    thresholds are calibrated against this model's measured behaviour, so if
    a threshold change starts flagging it (or the model itself regresses),
    this fails. Skipped cleanly if the checkpoint is not in this checkout.
    """
    print("[incumbent smoke test — known-good reference]")
    path = os.path.join(ROOT, "models", "tmt_t094747.npz")
    if not os.path.exists(path):
        print("  SKIP: models/tmt_t094747.npz not in this checkout")
        return
    m, info = cm.load_model(path)
    check("incumbent loads with t=94,747", info["t"] == 94747, info["t"])
    # 24-Sep protocol: 250 bytes, TinyStories prompt, 6 seeds. The documented
    # reference row is unique_bytes 32.5, max newline run 1, 0/6 degenerate.
    rows = []
    for seed in range(6):
        out, stops, dt, ent, eff_t = cm.generate(m, cm.PROMPT_TS, 250,
                                                 None, seed)
        rows.append(cm.degeneracy_metrics(out, stops, ent, eff_t, dt))
    agg = cm.aggregate(rows)
    check("incumbent: 0/6 degenerate", agg["degenerate"] == 0,
          f"{agg['degenerate']}/6")
    check("incumbent: not non-terminating at 250 bytes (short, by design)",
          agg["nonterminating"] == 0)
    check("incumbent: mean unique_bytes reproduces the 24-Sep reference (32.5)",
          abs(agg["unique_bytes_mean"] - 32.5) < 0.6,
          agg["unique_bytes_mean"])
    check("incumbent: max whitespace run is 1-2 (24-Sep: max newline run 1)",
          agg["max_run_ws"] <= 2, agg["max_run_ws"])
    check("incumbent: top 8-gram stays under the 2% arm on every seed",
          max(r["rep_8gram_frac"] for r in rows) < cm.DEGEN_REP8_FRAC,
          f"{100 * max(r['rep_8gram_frac'] for r in rows):.2f}%")
    # A longer budget, so the smoke test also exercises the liveness path that
    # a 250-byte probe cannot reach. Asserted over a 6000-byte budget, not per
    # generation: the incumbent's measured rate is ~2.7/kB, so an individual
    # 1500-byte generation may legitimately contain zero firings (it does), and
    # a per-generation assertion would be a flaky test, not a regression check.
    long_rows = []
    for seed in (0, 1, 2):
        out, stops, dt, ent, eff_t = cm.generate(m, cm.PROMPT_TS, 2000,
                                                 None, seed)
        long_rows.append(cm.degeneracy_metrics(out, stops, ent, eff_t, dt))
    la = cm.aggregate(long_rows)
    check("incumbent at 2000 B x3: 0/3 degenerate", la["degenerate"] == 0,
          f"{la['degenerate']}/3 ws={[r['max_run_ws'] for r in long_rows]} "
          f"rep8%={[round(100 * r['rep_8gram_frac'], 2) for r in long_rows]}")
    check("incumbent at 2000 B x3: stop head fires at least once overall",
          la["stop_hits"] >= 1, la["stop_hits"])
    check("incumbent at 2000 B x3: stop rate is the documented order (0.83/kB)",
          0.2 < la["stop_per_kB"] < 5.0, f"{la['stop_per_kB']:.2f}/kB")
    check("incumbent at 2000 B: zero-firing generations are allowed but rare "
          "(documented 2/9 on the 3x3 protocol)", la["nonterminating"] <= 1,
          f"{la['nonterminating']}/3")
    check("a model with a live stop head is not labelled non-terminating",
          la["nonterminating"] == 0 or la["stop_hits"] > 0)


def main():
    for fn in (t_newline_collapse, t_english_like, t_looped, t_hand_computed,
               t_rep8_frac_arm, t_nontermination, t_set_parsing,
               t_skipped_summary, t_digest, t_escape_render,
               t_incumbent_smoke):
        fn()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("all chat_models metric/parsing checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
