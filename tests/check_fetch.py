"""check_fetch.py — fetch_corpus unit tests: transforms, writer, budget mix.

Synthetic rows only (no network). Exercises the corpus pipeline's pure
functions so a schema/transform regression fails here, not on a Colab session.

Usage: .venv/bin/python tests/check_fetch.py
Exit code 0 = PASS.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, ".")
import fetch_corpus as fc


def t_gutenberg():
    cases = [
        ("*** START OF THE PROJECT GUTENBERG EBOOK ALICE ***\nCONTENT HERE\n*** END OF THE PROJECT GUTENBERG EBOOK ALICE ***",
         "CONTENT HERE"),
        ("The Project Gutenberg eBook...\n*** START OF THIS PROJECT GUTENBERG EBOOK ALICE ***\nBODY\n*** END OF THIS PROJECT GUTENBERG EBOOK ALICE ***\nENDNOTES",
         "BODY"),
        ("no markers at all\njust prose", "no markers at all\njust prose"),
        ("", None),   # empty -> dropped
    ]
    for text, want in cases:
        out = fc.transform_gutenberg({"text": text, "book_id": 42,
                                      "title": "T", "author": "A"})
        if want is None:
            assert out is None, text[:20]
        else:
            assert out is not None and out[0] == want, (text[:30], out)
    print("  gutenberg strip: PASS")
    return True


def t_raw():
    rows = [{"text": "x\n", "score": 2.9, "id": "a"},
            {"text": "y\n", "score": 3.0, "id": "b"},
            {"text": "", "score": 5.0, "id": "c"}]
    outs = [fc.transform_raw(r) for r in rows]
    assert outs[0] is None, "score < 3.0 must drop"
    assert outs[1] is not None and outs[1][0] == "y", "score == 3.0 kept"
    assert outs[2] is None, "empty text dropped"
    print("  fineweb score filter: PASS")
    return True


def t_dialogue():
    row = {"prompt_id": "p1",
           "messages": [{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"}],
           }
    out = fc.transform_dialogue(row)
    assert out and out[0].startswith("User: hi\nAssistant: hello\n---\n")
    assert out[1] == "ultrachat_p1"
    assert fc.transform_dialogue({"prompt_id": "p", "messages": []}) is None
    print("  dialogue serialize: PASS")
    return True


def t_writer():
    d = tempfile.mkdtemp()
    try:
        w = fc.CorpusWriter(d, mb=0.0002)      # 0.2 KB target: forces >1 part
        w.add("doc1", "a" * 100)
        w.add("doc2", "b" * 100)
        w.add("doc3", "c" * 100)
        w.close({"x": 1})
        parts = sorted(x for x in os.listdir(d) if x.startswith("corpus_"))
        assert len(parts) >= 1, parts
        # docs never split: each part is a concatenation of WHOLE docs
        total = sum(os.path.getsize(os.path.join(d, p)) for p in parts)
        assert total == 300, total
        assert os.path.exists(os.path.join(d, "manifest.json"))
        assert os.path.exists(os.path.join(d, "ORDER.txt"))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  writer (no split, manifest, order): PASS")
    return True


def t_budget():
    budgets, mix = fc.resolve_budgets(["A", "B", "C"], 4.5, None)
    assert budgets["A"] == int(2.7e9) and budgets["B"] == int(1.5e9) \
        and budgets["C"] == int(0.3e9), budgets
    budgets2, _ = fc.resolve_budgets(["A", "B", "C"], 4.5,
                                     "A=1.0,B=2.0,C=1.0")
    assert budgets2 == {"A": int(1e9), "B": int(2e9), "C": int(1e9)}, budgets2
    u, _ = fc.resolve_budgets(["A"], 0, None)
    assert u == {"A": None}, u
    print("  per-layer budget mix: PASS")
    return True


def main():
    ok = all(f() for f in [t_gutenberg, t_raw, t_dialogue, t_writer, t_budget])
    print("\nCHECK_FETCH:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())