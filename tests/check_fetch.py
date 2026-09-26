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


def t_oversized_document_split():
    text = ("line-%04d\n" % 0) * 1000
    text += "z" * 12_000
    pieces = list(fc.split_text_bytes(text, 4_000))
    assert len(pieces) > 1
    assert all(len(p.encode("utf-8")) <= 4_000 for p in pieces)
    assert "".join(pieces).encode("utf-8") == text.encode("utf-8")
    assert pieces[0].endswith("\n")
    print("  oversized source document splits deterministically: PASS")
    return True


def t_round_robin():
    values = {
        "A": iter(["a1", "a2"]),
        "B": iter(["b1"]),
        "C": iter(["c1", "c2", "c3"]),
    }

    getters = {k: (lambda k=k: next(values[k])) for k in values}
    got = list(fc.round_robin_documents(getters, max_items=5))
    assert got == [("A", "a1"), ("B", "b1"), ("C", "c1"),
                   ("A", "a2"), ("C", "c2")], got
    print("  round-robin mixed document order: PASS")
    return True


def t_writer():
    d = tempfile.mkdtemp()
    try:
        w = fc.CorpusWriter(d, mb=0.0002)      # 0.2 KB target: forces >1 part
        w.add("doc1", "a" * 100, layer="A")
        w.add("doc2", "b" * 100, layer="B")
        w.add("doc3", "c" * 100, layer="C")
        w.close({"x": 1})
        # published through a versioned dir + atomic `current` symlink
        cur = os.path.join(d, "current")
        assert os.path.islink(cur), os.listdir(d)
        target = os.path.realpath(cur)
        assert os.path.basename(target) == "v0000", target
        parts = sorted(x for x in os.listdir(target) if x.startswith("corpus_"))
        assert len(parts) >= 1, parts
        # docs never split: each part is a concatenation of WHOLE docs
        total = sum(os.path.getsize(os.path.join(target, p)) for p in parts)
        assert total == 300, total
        assert os.path.exists(os.path.join(target, "manifest.json"))
        order = open(os.path.join(target, "ORDER.txt")).read()
        assert order.splitlines()[0].split("\t", 1)[1].startswith("A\t")
        assert [line.split("\t", 1)[1].split("\t", 1)[0]
                for line in order.splitlines()] == ["A", "B", "C"]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  writer (versioned publish, no split, manifest, order): PASS")
    return True


def t_mixed_writer_parts():
    d = tempfile.mkdtemp()
    try:
        w = fc.CorpusWriter(d, mb=0.0002, expected_layers=("A", "B", "C"))
        for i in range(6):
            for layer in ("A", "B", "C"):
                w.add(f"{layer}{i}", "x" * 100, layer=layer)
        w.close({"mixed": True, "mix": {"A": 1, "B": 1, "C": 1},
                 "layer_budgets_bytes": {"A": 1, "B": 1, "C": 1},
                 "fetch_seed": 42, "spec_hash": "test-spec"})
        target = os.path.realpath(os.path.join(d, "current"))
        parts = sorted(p for p in os.listdir(target)
                       if p.startswith("corpus_"))
        assert len(parts) >= 2, parts
        order = open(os.path.join(target, "ORDER.txt")).read().splitlines()
        first_layers = {line.split("\t", 2)[1]
                        for line in order if line.startswith("00000\t")}
        assert first_layers == {"A", "B", "C"}, first_layers
        assert all(os.path.getsize(os.path.join(target, p)) < 1000
                   for p in parts), [(p, os.path.getsize(os.path.join(target, p)))
                                     for p in parts]
        import json
        manifest = json.load(open(os.path.join(target, "manifest.json")))
        assert set(manifest["first_part_layers"]) == {"A", "B", "C"}
        assert manifest["part_sizes"] == [
            os.path.getsize(os.path.join(target, p)) for p in parts]
        assert "mix" in manifest and "layer_budgets_bytes" in manifest
        assert "fetch_seed" in manifest and "spec_hash" in manifest
        print("  first mixed part + bounded later parts + manifest: PASS")
        return True
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_oversized_document_rejected():
    d = tempfile.mkdtemp()
    try:
        w = fc.CorpusWriter(d, mb=0.001, expected_layers=("A",))
        try:
            w.add("huge-doc", "x" * 8_000_001, layer="A")
            raise AssertionError("oversized source document was accepted")
        except ValueError as e:
            msg = str(e)
            assert "over the 8 MB part limit" in msg
            assert "layer=A" in msg and "huge-doc" in msg
        print("  oversized source document fails loudly: PASS")
        return True
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_publish_crash_safety():
    """Review #13: a crash before the pointer flip must leave the PREVIOUS
    corpus intact and complete; a second publish advances the version."""
    d = tempfile.mkdtemp()
    try:
        w1 = fc.CorpusWriter(d, mb=1)
        w1.add("a", "first corpus")
        w1.close({"v": 1})
        first = os.path.realpath(os.path.join(d, "current"))

        # crash injection: build a staging dir then "crash" before close()
        w2 = fc.CorpusWriter(d, mb=1)
        w2.add("b", "half written")
        # no close(): simulate the process dying here
        staging = [n for n in os.listdir(d) if n.startswith(".staging-")]
        assert staging, os.listdir(d)
        assert os.path.realpath(os.path.join(d, "current")) == first, \
            "pointer moved before publish completed"
        data = open(os.path.join(first, "corpus_00000.txt")).read()
        assert "first corpus" in data and "half written" not in data

        # recovery: the next publish ignores the abandoned staging dir
        w3 = fc.CorpusWriter(d, mb=1)
        w3.add("c", "second corpus")
        w3.close({"v": 2})
        now = os.path.realpath(os.path.join(d, "current"))
        assert now != first, now
        txt = open(os.path.join(now, "corpus_00000.txt")).read()
        assert "second corpus" in txt, txt
        # manifest records which version was published
        import json
        m = json.load(open(os.path.join(now, "manifest.json")))
        assert m["published_as"] == os.path.basename(now), m
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  publish is crash-safe (pointer flips last): PASS")
    return True


def t_pagination():
    """Review #14: Link-header rel=next parsing (the tree API paginates)."""
    h = '<https://huggingface.co/api/datasets/x/tree/main?cursor=abc>; rel="next"'
    assert fc._next_link(h) == \
        "https://huggingface.co/api/datasets/x/tree/main?cursor=abc"
    assert fc._next_link('<u>; rel="prev"') is None
    assert fc._next_link("") is None
    assert fc._next_link(None) is None
    print("  HF tree pagination (Link rel=next): PASS")
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
    ok = all(f() for f in [t_gutenberg, t_raw, t_dialogue, t_round_robin,
                           t_writer, t_mixed_writer_parts,
                           t_oversized_document_split,
                           t_oversized_document_rejected,
                           t_publish_crash_safety, t_pagination, t_budget])
    print("\nCHECK_FETCH:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())