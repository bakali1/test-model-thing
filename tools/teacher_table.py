"""teacher_table.py — token -> byte table for distilling an open LLM into TMT.

TMT's vocabulary is 256 bytes. A teacher is a token model. To turn the
teacher's next-token distribution into a next-byte distribution (the paper's
Marginalize-It step, `docs/ideas/papers/2609.12303v1.IDEA.md`) you need, for
every teacher token id, the exact bytes that token decodes to. That table is
the single most load-bearing artifact in the distillation pipeline and the
easiest to build wrongly in a way that produces plausible-looking numbers.

Why the table is built by hand instead of with `tokenizer.decode()`
---------------------------------------------------------------------
For LFM2.5 (and every byte-level BPE) `decode()` is LOSSY. 793 of the 64,400
base-vocab ids are single non-UTF-8 bytes (0x80-0xFF), which the ByteLevel
decoder round-trips into U+FFFD:

    id 604:  pure table -> b'\\xa1'      tokenizer.decode -> b'\\xef\\xbf\\xbd'

That is a silent corruption of 1.2% of the vocabulary's byte targets: it would
shrink those bytes' credit mass to zero and nobody would notice, because the
model still trains and the loss still goes down. `tests/check_distill.py` pins
this as a trap and asserts the hand-built table is the correct one.

Two more traps, both asserted by that gate:

* The tokenizer's post-processor prepends BOS, whose content is the literal
  15-byte control string ``<|startoftext|>``. Dump with the post-processor
  DISABLED, or every sequence gains 15 phantom bytes and the first target is
  a prediction of control text.
* 509 ids are `added_tokens` carrying literal control text (``<|pad|>``,
  ``<|startoftext|>``, ``<|endoftext|>``, ``<|im_start|>``, ``<|im_end|>``,
  the FIM and tool-call markers). NONE of them is ever emitted by the encoder
  on real text (gate t3), and none may be a byte-level target.

Measured on this tokenizer (see the gate's output): 4.153 bytes/token on the
ruler's held-out text;791 lossy `decode()` ids; 507 of the 509 added ids also
exist in the base BPE vocab and shadow it.

This module builds the table with **zero dependencies** (json + hashlib) so it
can be verified anywhere; `tokenizers` is imported only by the encoder helper,
never by the table.
"""
import hashlib
import json
import os
import urllib.request

HF_BASE = "https://huggingface.co/{repo}/resolve/main/{name}"


def bytes_to_unicode_decoder():
    """Inverse of the GPT-2 bytes<->unicode alphabet used by ByteLevel BPE.

    Returns {unicode_char: byte}. Every vocab string in a byte-level BPE is a
    sequence of these characters, so `bytes(dec[c] for c in tok)` recovers the
    token's exact byte string.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def fetch_tokenizer_json(repo, dest_dir=".cache/teacher", filename="tokenizer.json"):
    """Download the teacher's tokenizer.json once and cache it.

    The digest is printed because this file defines the whole byte mapping:
    a silently updated tokenizer on the Hub would change every distillation
    target while every existing number stayed valid-looking.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{repo.replace('/', '__')}__{filename}")
    if not os.path.exists(dest):
        url = HF_BASE.format(repo=repo, name=filename)
        req = urllib.request.Request(url, headers={"User-Agent": "tmt-teacher/1"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest + ".tmp", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(dest + ".tmp", dest)
    return dest


def build_table(tokenizer_json_path):
    """Return (table, meta).

    table[i] -> the exact bytes teacher id i stands for.

    NOTE the shadowing rule, which is easy to get wrong and which
    tests/check_distill.py caught during development: `added_tokens`
    OVERRIDE same-id entries of the base BPE vocab in the real tokenizer
    (509 of them here). Building the table from `model.vocab` alone leaves
    those 509 ids mapped to the wrong bytes -- including ids 2, 6 and 7, whose
    effective byte string is EMPTY (unobservable in byte space) while the
    shadowed base entry is ordinary text. So: base vocab first, added_tokens
    on top.
    """
    with open(tokenizer_json_path, encoding="utf-8") as fh:
        d = json.load(fh)
    model = d.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"expected a BPE tokenizer, got {model.get('type')!r}")
    if d.get("normalizer") is not None:
        # A normalizer would mean token bytes are NOT the source bytes, and
        # the exact-reconstruction contract below cannot hold.
        raise ValueError("tokenizer has a normalizer; byte reconstruction is "
                         "not guaranteed — re-derive the table by hand")
    pretok = json.dumps(d.get("pre_tokenizer"))
    if "ByteLevel" not in pretok:
        raise ValueError(f"pre_tokenizer is not ByteLevel: {pretok[:120]}")

    dec = bytes_to_unicode_decoder()
    vocab = model["vocab"]
    added = {int(t["id"]): t["content"] for t in d.get("added_tokens", [])}
    n = max([len(vocab)] and [max(vocab.values()) + 1] or [0])
    n = max(n, (max(added) + 1) if added else 0)
    table = [b""] * n
    for tok, idx in vocab.items():
        try:
            table[idx] = bytes(dec[c] for c in tok)
        except KeyError as exc:
            raise ValueError(
                f"vocab token {tok!r} (id {idx}) uses character {exc} outside "
                f"the ByteLevel alphabet; this is not a byte-level BPE") from exc
    shadowed = sorted(set(added) & set(vocab.values()))
    for idx, content in added.items():
        table[idx] = content.encode("utf-8")

    # Classification of the EFFECTIVE table:
    #   nonbyte  -- empty byte string: cannot be a byte-level target at all
    #               (this tokenizer has none; kept because other teachers do)
    #   control  -- an added token carrying literal control text ("<|pad|>")
    #   text     -- ordinary corpus bytes
    nonbyte = sorted(i for i, b in enumerate(table) if b == b"")
    control = sorted(i for i in added if table[i] != b"")
    # Most control tokens are chat/FIM markers ("<|pad|>"); a few are modality
    # placeholders ("<image>"). Either way they are not corpus bytes, so all of
    # them are excluded from byte targets. The split is reported, not assumed.
    markers = sorted(i for i in control if table[i].startswith(b"<|"))
    other = sorted(i for i in control if i not in set(markers))
    meta = {
        "n_ids": n,
        "n_base": len(vocab),
        "added_ids": sorted(added),
        "shadowed_ids": shadowed,
        "control_ids": control,
        "marker_ids": markers,
        "other_added_ids": other,
        "nonbyte_ids": nonbyte,
        "sha256": hashlib.sha256(open(tokenizer_json_path, "rb").read()).hexdigest(),
        "post_processor": json.dumps(d.get("post_processor"))[:200],
    }
    return table, meta


def is_ascii_printable(b):
    return all(32 <= c < 127 or c in (9, 10, 13) for c in b)


def encode_without_post(tokenizer, text):
    """Encode with the post-processor disabled (see module docstring)."""
    tokenizer.post_processor = None
    return tokenizer.encode(text).ids


def reconstruct(table, ids):
    return b"".join(table[i] for i in ids)