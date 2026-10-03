"""check_distill.py — GATE 0 for the distillation plan.

    .venv/bin/python tests/check_distill.py [--bytes 2000000]

Free (CPU, ~10 s). Run this before any GPU hour: the token->byte table is the
artifact most able to produce a *plausible-looking* wrong result, because a
corrupted table still trains fine and still lowers the loss.

Contract proven here:

  t0  the table builds from tokenizer.json with zero deps, and every
      base-vocab id has a non-empty byte string
  t1  TRAP: `tokenizer.decode()` is LOSSY for non-UTF-8 byte tokens.
      Assert the hand-built table disagrees with decode() exactly where the
      decode is wrong (U+FFFD), so this trap cannot be reintroduced silently.
  t2  the table reconstructs the ruler's real held-out text BYTE FOR BYTE
      (data/eval/, data/eval_shift/) and reports bytes/token
  t3  TRAP: the default post-processor injects BOS == the 15-byte literal
      "<|startoftext|>". With it disabled no special id is ever emitted.
  t4  control ids (added tokens carrying literal "<|...|>" text) are
      identified and are never emitted by the encoder; they cannot be
      byte-level targets
  t5  DECONTAMINATION: a story copied verbatim from an EVAL file is dropped by
      the widened membership set, and is NOT dropped by the old
      training-only set (which is the bug this gate exists to prevent). The
      filter's limit is demonstrated too: line-level matching does not catch a
      paraphrase.

Exit 0 = PASS. No GPU, no network beyond one tokenizer.json download.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import teacher_table as tt  # noqa: E402
import decontaminate as dc  # noqa: E402

REPO = "LiquidAI/LFM2.5-230M-Base"
EVAL_FILES = [
    "data/eval/tinystories_valid.txt",
    "data/eval/wiki_tinystories_005.txt",
    "data/eval_shift/tinystories_v2gpt4_valid.txt",
]


def _t(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def _is_utf8(b):
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=2_000_000)
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()
    ok = True

    print("== t0/t4: table construction ==")
    path = tt.fetch_tokenizer_json(args.repo)
    table, meta = tt.build_table(path)
    ok &= _t("table built", True,
             f"{meta['n_base']} base + {len(meta['added_ids'])} added, "
             f"{meta['n_ids']} ids, tokenizer sha256 {meta['sha256'][:12]}")
    text_ids = [i for i in range(meta["n_ids"])
                if i not in set(meta["control_ids"]) | set(meta["nonbyte_ids"])]
    empty_text = [i for i in text_ids if not table[i]]
    ok &= _t("every non-special id has bytes", not empty_text,
             f"{len(empty_text)} empty" if empty_text else f"{len(text_ids)} text ids")
    ok &= _t("added_tokens SHADOW same-id base entries (the bug this gate caught)",
             len(meta["shadowed_ids"]) > 0,
             f"{len(meta['shadowed_ids'])} of {len(meta['added_ids'])} added ids "
             f"also exist in the base vocab; a base-vocab-only table would map "
             f"them to the wrong bytes")
    ok &= _t("no empty byte strings among text ids", not empty_text,
             f"{len(meta['nonbyte_ids'])} non-byte ids in this tokenizer")
    ok &= _t("control ids are the added tokens, none a corpus byte",
             len(meta["control_ids"]) == len(meta["added_ids"]) and len(meta["control_ids"]) > 0,
             f"{len(meta['marker_ids'])} chat/FIM markers + "
             f"{len(meta['other_added_ids'])} other (e.g. modality placeholders); "
             f"all excluded from byte targets")

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("  [FAIL] `tokenizers` is required for the encode-side checks")
        return 1
    tk = Tokenizer.from_file(path)

    print("== t1: decode() is lossy (the trap) ==")
    lossy = []
    for i in text_ids:
        s = tk.decode([i], skip_special_tokens=False)
        lib = s.encode("utf-8") if s else b""
        if lib != table[i]:
            lossy.append(i)
    ok &= _t("lossy decode() ids are exactly the non-UTF-8 byte tokens",
             len(lossy) > 0 and all(not _is_utf8(table[i]) for i in lossy),
             f"{len(lossy)} ids; e.g. id {lossy[0] if lossy else '-'} -> "
             f"decode gives {tk.decode([lossy[0]])!r} (U+FFFD) but the byte is "
             f"{table[lossy[0]]!r}")

    print("== t2: exact byte reconstruction of real ruler text ==")
    tk.post_processor = None
    for f in EVAL_FILES:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            print(f"  [skip] {f} missing")
            continue
        raw = open(p, "rb").read()[:args.bytes]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            ok &= _t(f"{f} decodes", False, str(e))
            continue
        ids = tk.encode(text).ids
        recon = tt.reconstruct(table, ids)
        ok &= _t(f"{f} reconstructs exactly", recon == raw,
                 f"{len(raw):,} B -> {len(ids):,} tok, "
                 f"{len(raw) / max(1, len(ids)):.3f} bytes/token"
                 + ("" if recon == raw else f" (len {len(recon):,} vs {len(raw):,})"))

    print("== t3: post-processor trap ==")
    raw = open(os.path.join(ROOT, EVAL_FILES[0]), "rb").read()[:200_000]
    text = raw.decode("utf-8")
    default_ids = Tokenizer.from_file(path).encode(text).ids
    ok &= _t("default encoding DOES inject BOS (trap is real)",
             default_ids[0] == 1 and table[1] == b"<|startoftext|>",
             f"first id {default_ids[0]} = {table[default_ids[0]]!r} = "
             f"{len(table[1])} phantom bytes")
    clean_ids = tk.encode(text).ids
    special = [i for i in clean_ids if i in set(meta["added_ids"])]
    ok &= _t("no special/added token is emitted with post-processor disabled",
             not special, f"{len(special)} emissions in {len(clean_ids):,} ids")
    ok &= _t("reconstruction still exact after disabling it",
             tt.reconstruct(table, clean_ids) == raw)

    print("== t5: decontamination of teacher-generated text ==")
    eval_path = os.path.join(ROOT, "data/eval/tinystories_valid.txt")
    if not os.path.exists(eval_path):
        print("  [skip] eval file missing")
    else:
        blob = open(eval_path, "rb").read()
        # Need >=3 long lines: the drop rule is hit/len(long) >= 0.6, so a
        # 2-line story can only ever be dropped if BOTH lines match.
        story = None
        for seg in blob.split(b"\n\n"):
            cand = seg.strip(b"\n")
            if len([l for l in cand.split(b"\n") if len(l) >= 40]) >= 5:
                story = cand + b"\n\n"
                break
        long_lines = [] if story is None else [l for l in story.split(b"\n") if len(l) >= 40]
        ok &= _t("picked an eval story with >=5 long lines to reuse",
                 len(long_lines) >= 5, f"{len(long_lines)} long lines")
        if story is None:
            print("  [skip] no suitable story found")
            return 0 if ok else 1

        lines = dc.membership_lines(verbose=False)
        kept, dropped, _ = dc.filter_corpus(story, lines, verbose=False)
        ok &= _t("widened set (train + eval + shift) DROPS a copied eval story",
                 dropped == 1 and len(kept.strip()) == 0,
                 f"dropped={dropped}")

        train_only = dc.membership_lines(
            ("data/wikipedia_clean/**/wiki_*",), verbose=False)
        kept2, dropped2, _ = dc.filter_corpus(story, train_only, verbose=False)
        ok &= _t("training-only set does NOT drop it (the gap being fixed)",
                 dropped2 == 0 and len(kept2) == len(story),
                 f"dropped={dropped2}; eval lines are invisible to the "
                 f"training glob, so an eval-copied story would train")

        # The honest limit: the rule is line-exact. A rewrite with the same
        # beats is NOT caught, and TinyStories is heavily templated.
        para = b"\n".join(b" ".join(w[::-1] for w in l.split(b" "))
                          for l in story.strip().split(b"\n")) + b"\n\n"
        kept3, dropped3, _ = dc.filter_corpus(para, lines, verbose=False)
        ok &= _t("KNOWN LIMIT demonstrated: a rewrite is NOT dropped",
                 dropped3 == 0,
                 "line-exact matching misses paraphrases — every generated-"
                 "corpus result must carry that caveat")

    print("\n== verdict:", "PASS" if ok else "FAIL", "==")
    if ok:
        print("gate 0 cleared: the token->byte mapping is exact, so Marginalize-It")
        print("conversion cannot silently corrupt targets. Storage math at")
        print("  k=32, fp16 values + int32 indices = 192 B/token")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())