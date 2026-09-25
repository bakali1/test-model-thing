#!/usr/bin/env python3
"""Generate text from several checkpoints side by side, same seed prompt.

    .venv/bin/python chat_models.py --n 400

CE can hide degenerate output: a model can hold a respectable mean
cross-entropy while emitting repetition, looped fragments, or a single token
forever. Reading the samples is the only way to see that class of failure, and
it is the cheapest check there is -- no GPU, no training.

All models get the SAME prompt bytes and the same generation length, so the
outputs are directly comparable. Each model starts from a zeroed state, so
what you see is what the weights produce cold, not a stream it was trained on.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy_port  # noqa: E402

# (label, checkpoint path). Checkpoints with different t are reported as such.
DEFAULT_SET = [
    ("incumbent tmt", "tmt_t094747.npz"),
    ("FS-W TinyStories", "fs_fs-w_t060000.npz"),
    ("FS-N Gutenberg", "fs_fs-n_t086000.npz"),
    ("FS-L mix", "fs_l_t086000.npz"),
]

# A real opening from the held-out TinyStories distribution: the in-domain case.
PROMPT_TS = b"Once there was a little girl who loved cocoa."
# A neutral English opening, in no arm's training set: the transfer case.
PROMPT_NE = b"Dear reader,\n\nI am writing to you today because"
# A narrative-prose opening: the case FS-N was actually trained for. Judging
# FS-N only on a TinyStories prompt would be judging it off-manifold.
PROMPT_NOVEL = (b"It was a bright cold day in April, and the clocks were "
                b"striking thirteen. Winstone Smith, his chin nuzzled into "
                b"his breast in an effort to escape the vile wind, slipped "
                b"quickly through the glass doors of Victory Mansions.")


def show(b):
    """Render bytes as text without letting junk break the terminal."""
    return b.decode("utf-8", errors="replace").replace("\x00", "·")


def generate(path, prompt, n, temp=None, seed=0):
    with np.load(path) as d:
        dim = int(d["embed"].shape[1])
        # keys look like "layer0.W", so the index is BEFORE the dot
        layers = 1 + max((int(k.split(".")[0].replace("layer", ""))
                          for k in d.files
                          if k.startswith("layer") and k.endswith(".W")),
                         default=0)
    m = numpy_port.Model(dim=dim, layers=layers)
    m.load(path)
    if temp is not None:
        m.temp = temp
    m.new_states(1)          # zero memory: show the weights cold
    numpy_port.rng = np.random.default_rng(seed)   # same draws for every model
    out = bytearray(prompt)
    stops = 0
    ent_sum = 0.0
    eff_t = float("nan")
    t0 = time.time()
    for _ in range(n):
        # Mirror sample()'s entropy->temperature schedule so the report can
        # state the regime: when entropy is HIGH the temperature COLLAPSES
        # (0.75 * (1 - 0.75*1.0) = 0.19), i.e. near-greedy. Saying so is the
        # difference between "this is what the model believes" and "this is
        # what the sampler did".
        enc = m.embed[out[-1]]
        x = enc[None, :]
        for l in m.layers:
            x, ns, _, _ = l.forward(enc.reshape(1, -1), x, l.states[0:1])
            l.states[0:1] = ns
        logits = x[0] @ m.Wd.T + m.bd
        pr = np.exp(logits - logits.max())
        pr /= pr.sum()
        ent = float(-(pr * np.log(pr + 1e-8)).sum() / np.log(256.0))
        ent_sum += ent
        eff_t = max(0.1, m.temp * (1.0 - m.temp * ent))
        c = m.sample(logits)
        stop = float(np.asarray(x[0] @ m.Ws + m.bs).ravel()[0])
        stop = 1.0 / (1.0 + np.exp(-stop))
        if stop > 0.5:
            stops += 1
        out.append(c)
    return (bytes(out[len(prompt):]), stops, time.time() - t0,
            ent_sum / max(n, 1), eff_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="bytes to generate")
    ap.add_argument("--temp", type=float, default=None,
                    help="override sampling temperature (default: model's)")
    ap.add_argument("--prompt", choices=("ts", "ne", "novel", "all"),
                    default="all")
    ap.add_argument("--set", default=None,
                    help="comma-separated label=path pairs (default: the 4 arms)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; the SAME seed is used for every model "
                         "so the draws line up")
    args = ap.parse_args()

    if args.set:
        models = []
        for part in args.set.split(","):
            lab, _, p = part.partition("=")
            models.append((lab, p))
    else:
        models = DEFAULT_SET

    prompts = []
    if args.prompt in ("ts", "all"):
        prompts.append(("in-domain (TinyStories opening)", PROMPT_TS))
    if args.prompt in ("ne", "all"):
        prompts.append(("transfer (neutral English opening)", PROMPT_NE))
    if args.prompt in ("novel", "all"):
        prompts.append(("narrative prose opening (FS-N's own domain)",
                        PROMPT_NOVEL))

    for title, prompt in prompts:
        print("\n" + "=" * 78)
        print(f"{title}")
        print(f"prompt: {show(prompt)!r}")
        print("=" * 78)
        for label, path in models:
            if not os.path.exists(path):
                print(f"\n--- {label}: MISSING {path}")
                continue
            out, stops, dt, ent, eff_t = generate(path, prompt, args.n,
                                                  args.temp, args.seed)
            with np.load(path) as d:
                t = int(d["t"]) if "t" in d.files else -1
            print(f"\n--- {label}  (t={t:,}, {len(out)} bytes, {dt:.1f}s, "
                  f"stop-head {stops}x, entropy {ent:.2f} -> effective temp "
                  f"{eff_t:.2f}{', NEAR-GREEDY' if eff_t < 0.3 else ''})")
            # Wrap long output without mangling it.
            txt = show(out)
            for i in range(0, len(txt), 74):
                print("    " + txt[i:i + 74].replace("\n", "\n    "))
    print()


if __name__ == "__main__":
    main()
