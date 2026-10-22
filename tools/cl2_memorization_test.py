#!/usr/bin/env python3
"""Is the head's 0.975 TRAINING accuracy a LEARNED key->payload mechanism, or
MEMORISATION of the training streams?

The armA_f2_end checkpoint's OWN end-of-phase eval says:
    train_board    head_per_byte 0.1045  head_exact4 0.0
    heldout_board  head_per_byte 0.0967  head_exact4 0.0
    chance         head_per_byte 0.0625
while the per-step training log says  head 0.975/0.910  on the training
batch.  Same board, same builder, same layout code -- so the two numbers
describe two different things, and WHICH is the truth decides the programme.

THREE CANDIDATE EXPLANATIONS, and the test that separates them:
  (A) MEMORISATION -- the model+head fit the exact training streams and do
      not generalise to fresh streams of the SAME facts.
  (B) A CHECKPOINTING ARTEFACT -- training runs the forward with
      use_checkpoint=True, the eval with False.  (The engine has NO dropout,
      so train/eval MODE cannot explain anything.)
  (C) A LAYOUT MISMATCH -- the eval draws streams differently.

Plus the COPY CONTROL: the payload is in the INPUT (in the fact region AND
as the target tail), so replace the payload bytes IN THE INPUT with fresh
random bytes, leaving the cue and the target unchanged.  A memoriser of the
training streams still outputs the OLD payload (it memorised the whole
stream) and so keeps its score; a genuine state-held fact must COLLAPSE to
chance, because the state now holds the NEW random payload.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools.cl2_train as C  # noqa: E402


class A:
    """the subset of the trainer's args that the builders need"""
    dim = 512
    layers = 16
    device = "cpu"
    lr = 5e-4
    seed = 0
    half_min, half_max = 2.0, 512.0
    facts = 4
    gap = 32
    text_frac = 0.5
    batch = 32
    eval_batch = 32
    _text_pool = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--board-seed", type=int, default=202)
    ap.add_argument("--board-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text-files", default="data/*.txt")
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    os.chdir(args.repo)
    a = A()
    a.batch = a.eval_batch = args.batch

    alpha = C.alphabet_symbols("16")
    board = C.make_board(args.board_seed, args.board_size, alpha)
    tfiles = sorted(glob.glob(args.text_files))
    a._text_pool = C.TextPool(tfiles, seed=a.seed)

    model = C.build_model(a)
    heads = list(C.build_head(model)) + list(C.build_head(model))
    opt = C.AdamW([], lr=a.lr)
    C.load_checkpoint(model, opt, heads, args.ckpt)
    # (GpuModel has no train/eval mode -- the engine has NO dropout at all)
    print(f"[loaded] {args.ckpt}")
    print(f"[board] seed {args.board_seed} ({board['n_pairs']} facts), "
          f"alphabet 16, facts {a.facts}, gap {a.gap}, text_frac {a.text_frac}")

    lay = C.sample_layout(a.facts, a.gap)

    def score(seq, pay_new, use_ckpt=False, tag=""):
        with torch.no_grad():
            logits, state_cue = C.engine_forward(model, seq, lay["cue_last"],
                                                  use_checkpoint=use_ckpt)
            ml = state_cue @ heads[0] + heads[1]
            hp = ml.reshape(-1, 256).argmax(-1).reshape(pay_new.shape)
            pb = float((hp == pay_new).float().mean())
            e4 = float((hp == pay_new).all(dim=1).float().mean())
            b0, b1 = lay["cue1"] - 1, lay["tgt1"] - 1
            bp = logits[:, b0:b1].argmax(-1)
            bt = torch.from_numpy(seq[:, b0 + 1:b1 + 1].astype(np.int64))
            bb = float((bp == bt).float().mean())
            b4 = float((bp == bt).all(dim=1).float().mean())
        print(f"  {tag:44} head {pb:.4f}/{e4:.4f}   behav {bb:.4f}/{b4:.4f}")
        return pb, e4

    rng = np.random.default_rng(12345)
    b = C.build_fact_batch(a.batch, board, alpha, a._text_pool, a.text_frac,
                            a.gap, rng, n_facts=a.facts)
    seq, pay = b["seq"], b["pay_new"]

    print("\n[1] FRESH streams, training board (the eval's own condition):")
    score(seq, pay, False, "fresh / use_checkpoint=False")

    print("\n[2] FRESH streams, use_checkpoint=True (artefact test (B)):")
    score(seq, pay, True, "fresh / use_checkpoint=True")

    print("\n[3] THE SAME streams, but the payload bytes IN THE INPUT are")
    print("    replaced by fresh random bytes (the cue and target unchanged):")
    seq2 = seq.copy()
    for i in range(a.facts):
        s = lay["facts0"] + i * C.FACT_LEN + C.KEY_LEN
        seq2[:, s:s + C.PAYLOAD_LEN] = rng.choice(alpha,
                                                   size=(seq2.shape[0],
                                                         C.PAYLOAD_LEN))
    score(seq2, pay, False, "input payload replaced (copy control)")

    print("\n[4] and the same, with the TARGET TAIL also replaced (so NOTHING")
    print("    in the input carries the payload):")
    seq3 = seq2.copy()
    seq3[:, lay["tgt0"]:lay["tgt1"]] = rng.choice(
        alpha, size=(seq3.shape[0], C.PAYLOAD_LEN))
    score(seq3, pay, False, "input payload AND target tail replaced")

    print(f"\n[reference] chance per-byte {1.0/16:.4f}, "
          f"chance exact-4 {(1.0/16)**4:.3e}")
    print("[note] 'behav' predicts the target bytes, which are IN THE INPUT at "
          "the next position -- so behav is a COPY by construction.")


if __name__ == "__main__":
    main()
