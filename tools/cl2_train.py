#!/usr/bin/env python3
"""tools/cl2_train.py -- CL-2: the FIRST TRAINED continual-learning run.

PLAN-2026-10-21-CONTINUAL-LEARNING.md §5, step CL-2.  This is the decisive
experiment of the programme and it has never been run: until now the "fact in
the loss" existed only as a refusal message (tools/cl_ruler.py --mode train).

WHAT IS BEING ASKED
-------------------
The engine's recurrent state is a per-coordinate leaky integrator of the byte
embedding (`state = states*a + enc`, gpu_port.py:325).  The incumbent loss is
next-byte CE, which never asks the state to be USABLE.  CL-2 asks:

    can the state be TRAINED to hold a fact?

THE CORPUS (a corpus, not probe trials)
----------------------------------------
Every training sample is one fact STREAM over a real-text background:

    [F_1][F_2]...[F_N][GAP d][KEY_cue][PAY_target]
     \\_________/  \\____/  \\______/  \\________/
      N facts        distractors  the cue   the target

    F_i      = KEY_i(4B) || PAY_i(4B)      (keys uniform over 256, payloads
                                              from the MATCHED alphabet)
    GAP d    = d distractor bytes: real text (a fraction --text-frac of the
               samples) or uniform random bytes
    KEY_cue  = the key of the newest fact: the CUE
    PAY_target = the newest fact's payload: the thing to be recalled

The layout is the CL-RULER's own fixed-gap cell (tools/cl_ruler.py
build_sequences "stream"), so the state at the cue has the SAME distribution
at training time and at scoring time.  The trailing PAY_target is the
BEHAVIOURAL target: the model is asked (by the next-byte CE) to EMIT the
payload right after the cue key.

THE MEMORY LOSS (the new thing)
--------------------------------
An auxiliary linear head on the STATE AT THE CUE (the concatenated 16x512
recurrent state after the last cue byte), predicting the 4 payload bytes with
a per-byte cross-entropy, added to the CE with a small weight
(--memory-alpha).  That term is what puts the gradient on the state, and
therefore on the memory: the model must WRITE the payload into the state, hold
it across the gap, and READ it back at the cue.

The objective must not be satisfiable without the state.  The payloads are
random, so the head cannot guess them: the prior baseline (the global byte
prior of the training corpus) is reported next to the head's accuracy, and
the evaluation also runs the head on a NUISANCE stream whose target payload
is provably absent from the state.  (The FSP arm of PLAN §0 failed exactly
here: F1 0.685 against a prior of 0.817 -- a null.)

THE PHASES (RETAIN needs a "before")
-------------------------------------
  pre   text-only CE (optional): gives the model a general ability, so that
        NO REGRESSION has a non-degenerate baseline
  f1    the PRE-STREAM facts (board A): the facts the model knows before
  f2    the STREAM facts (board B), with NO replay of board A: the new facts
        the model must acquire, while board A must be retained

GRADIENTS
---------
Autograd (torch) over the engine's own forward, NOT the engine's ring credit:
the ring credit is an approximation of the cross-time path, and CL-2 needs the
exact one.  The per-timestep graph is checkpointed, so the memory cost is
O(L * B * dim * layers) rather than O(L^2).

RESUMABLE
---------
The checkpoint stores theta, the optimizer moments, the memory head, the
phase, the step counter and the corpus position (the step counter IS the
corpus position: the corpus is regenerated deterministically from --seed).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import gpu_port  # noqa: E402

# ---------------------------------------------------------------------------
# the fact geometry (the same constants the CL-RULER uses)
# ---------------------------------------------------------------------------
KEY_LEN = 4
PAYLOAD_LEN = 4
FACT_LEN = KEY_LEN + PAYLOAD_LEN      # 8
CUE_LEN = KEY_LEN                     # 4
TARGET_LEN = PAYLOAD_LEN              # 4


# ---------------------------------------------------------------------------
# the payload alphabet (the CL-RULER's, imported -- never re-implemented)
# ---------------------------------------------------------------------------
def alphabet_symbols(kind):
    """'full' -> None (256 symbols); '16'/'4' -> evenly spread byte values."""
    if kind in (None, "full"):
        return None
    k = int(kind)
    if k <= 1 or 256 % k:
        raise ValueError(f"alphabet size must divide 256; got {k}")
    return np.array([(i * 256) // k for i in range(k)], dtype=np.uint8)


def draw(rng, alpha, size):
    if alpha is None:
        return rng.integers(0, 256, size=size).astype(np.uint8)
    return alpha[rng.integers(0, len(alpha), size=size)].astype(np.uint8)


# ---------------------------------------------------------------------------
# the fact boards (the "facts" the model is told)
# ---------------------------------------------------------------------------
def make_board(seed, n_pairs, alpha):
    """A board of n_pairs (key -> payload) facts.  The board IS the fact set:
    board A is the pre-stream knowledge, board B is the stream."""
    rng = np.random.default_rng(int(seed))
    keys = rng.integers(0, 256, size=(n_pairs, KEY_LEN)).astype(np.uint8)
    pays = draw(rng, alpha, (n_pairs, PAYLOAD_LEN))
    return {"keys": keys, "payloads": pays, "seed": int(seed),
            "n_pairs": int(n_pairs)}


# ---------------------------------------------------------------------------
# the real-text background
# ---------------------------------------------------------------------------
class TextPool:
    """The real-text distractor pool: a byte array, sampled in random windows.

    The books in data/*.txt are used on purpose: they are NOT the held-out
    evaluation corpus (data/eval/), so the NO REGRESSION cell measures a
    genuine generalisation and not a memorisation of the scored bytes.
    """

    def __init__(self, files, seed=0):
        chunks = []
        for f in files:
            with open(f, "rb") as fh:
                chunks.append(np.frombuffer(fh.read(), dtype=np.uint8))
        if not chunks:
            raise SystemExit("no text files for the background")
        self.data = np.concatenate(chunks)
        self.n = len(self.data)

    def windows(self, k, width, rng):
        """k random windows of `width` bytes."""
        if width <= 0 or k == 0:
            return np.zeros((k, max(width, 0)), np.uint8)
        hi = max(1, self.n - width)
        starts = rng.integers(0, hi, size=k)
        out = np.empty((k, width), np.uint8)
        for i, s in enumerate(starts):
            out[i] = self.data[s:s + width]
        return out


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------
def sample_layout(n_facts, d, target_len=TARGET_LEN):
    """The byte layout of one stream (the CL-RULER's fixed-gap cell + the
    behavioural target tail)."""
    n_facts = int(n_facts)
    d = int(d)
    g0 = FACT_LEN * n_facts                 # the gap starts here
    cue0 = g0 + d                          # the cue (KEY) starts here
    tgt0 = cue0 + CUE_LEN                  # the behavioural target starts here
    length = tgt0 + target_len
    return {"facts0": 0, "gap0": g0, "gap1": g0 + d, "cue0": cue0,
            "cue1": cue0 + CUE_LEN, "tgt0": tgt0, "tgt1": tgt0 + target_len,
            "length": length, "cue_last": cue0 + CUE_LEN - 1,
            "n_facts": n_facts, "d": d}


def build_fact_batch(B, board, alpha, text_pool, text_frac, d, rng,
                     n_facts=4):
    """One batch of fact streams.  Returns the sequence and the labels."""
    lay = sample_layout(n_facts, d)
    L = lay["length"]
    idx = rng.integers(0, board["keys"].shape[0], size=(B, n_facts))
    keys = board["keys"][idx]  # (B, n_facts, KEY_LEN)
    pays = board["payloads"][idx]  # (B, n_facts, PAYLOAD_LEN)
    seq = np.empty((B, L), np.uint8)
    facts = np.concatenate([keys, pays], axis=2)          # (B, N, 8)
    seq[:, lay["facts0"]:lay["gap0"]] = facts.reshape(B, -1)

    # the gap: real text for a fraction of the rows, uniform random otherwise
    gap = np.empty((B, d), np.uint8)
    use_text = rng.random(B) < float(text_frac)
    n_text = int(use_text.sum())
    if n_text:
        gap[use_text] = text_pool.windows(n_text, d, rng)
    n_rand = B - n_text
    if n_rand:
        gap[~use_text] = rng.integers(0, 256, size=(n_rand, d)).astype(np.uint8)
    seq[:, lay["gap0"]:lay["gap1"]] = gap

    # the cue = the newest fact's key; the behavioural target = its payload
    seq[:, lay["cue0"]:lay["cue1"]] = keys[:, -1]
    seq[:, lay["tgt0"]:lay["tgt1"]] = pays[:, -1]

    # the CE mask: the next-byte CE counts ONLY where the next byte is a
    # TEXT byte (the general language signal) or one of the 4 behavioural
    # target bytes (the recall signal).  The random fact bytes are
    # unpredictable by construction, so putting CE on them would train the
    # model to spread mass over the alphabet.
    is_text = np.zeros((B, L), bool)
    is_text[:, lay["gap0"]:lay["gap1"]] = use_text[:, None]
    ce_mask = np.zeros((B, L - 1), bool)        # ce_mask[:, t] -> byte t+1
    ce_mask[:, lay["gap0"] - 1:lay["gap1"] - 1] = is_text[:, lay["gap0"] - 1:lay["gap1"] - 1]
    ce_mask[:, lay["cue1"] - 1:lay["tgt1"] - 1] = True

    return {
        "seq": seq,
        "lay": lay,
        "pay_new": pays[:, -1].astype(np.int64),     # the ACQUIRE target
        "pay_old": pays[:, 0].astype(np.int64),      # the RETAIN target
        "key_new": keys[:, -1].astype(np.int64),
        "ce_mask": ce_mask,
        "behav_slice": (lay["cue1"] - 1, lay["tgt1"] - 1),   # the 4 CE slots
        "use_text": use_text,
    }


def build_text_batch(B, L, text_pool, rng):
    """A pure-text batch (the pre-training phase: CE on everything)."""
    seq = text_pool.windows(B, L, rng)
    ce_mask = np.ones((B, L - 1), bool)
    return {"seq": seq, "lay": sample_layout(1, max(L - FACT_LEN - CUE_LEN - 4, 1)),
            "pay_new": np.zeros((B, PAYLOAD_LEN), np.int64),
            "pay_old": np.zeros((B, PAYLOAD_LEN), np.int64),
            "key_new": np.zeros((B, KEY_LEN), np.int64),
            "ce_mask": ce_mask, "behav_slice": (0, 0), "use_text": None}


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
def rebind_views(model):
    """REBIND the parameter views so that autograd can reach theta.

    BUG (found 2026-10-22, and it is the reason the first CL-2 arm was "at
    UNIFORM"):  GpuModel builds its parameter VIEWS (model.views) while
    theta does NOT require grad, and build_model then calls
    theta.requires_grad_(True).  A view of a tensor that did not require
    grad is a DETACHED leaf: after the base is switched it reports
    requires_grad=True, but it has no grad_fn, so the autograd engine
    accumulates NOTHING into it -- and because the base's own .grad stays
    None, the gradient is SILENTLY LOST.  loss.backward() raises no error,
    the optimizer sees p.grad is None and skips the parameter, and the arm
    trains NOTHING: the weights stay at their initialisation and the
    held-out CE stays at ln(256) = 5.5452 forever.

    The fix is to create the views AFTER theta requires grad: then they are
    ordinary differentiable views (grad_fn=ViewBackward0) and the gradient
    lands in theta.grad, which the optimizer actually uses.
    """
    off = 0
    views = {}
    for name, shape in model.shapes:
        size = int(np.prod(shape))
        views[name] = model.theta[off:off + size].view(shape)
        off += size
    model.views = views
    model.embed = views["embed"]
    model.Wd, model.bd = views["Wd"], views["bd"]
    model.Ws, model.bs = views["Ws"], views["bs"]
    if "fsp.W" in views:
        model.fsp_W, model.fsp_b = views["fsp.W"], views["fsp.b"]
        model.fsp_g, model.fsp_b0 = views["fsp.g"], views["fsp.b0"]
    for i, l in enumerate(model.layers):
        l.W = views[f"layer{i}.W"]
        l.g = views[f"layer{i}.g"]
        l.b = views[f"layer{i}.b"]
        l.decay = views[f"layer{i}.decay"]
        if getattr(l, "gated", False) and f"layer{i}.gate" in views:
            l.gate = views[f"layer{i}.gate"]
    return model


def build_model(args):
    """The engine, from scratch, with the P4 multi-scale half-life init.

    P4 (half_life_init) is used because the DEFAULT init puts every half-life
    within a byte of 1 B (PLAN §2.4): there is no multi-scale memory to hold a
    fact, so a from-scratch CL-2 run on the default init would be testing an
    architecture that cannot hold anything by construction.  The P4 arm is
    the one the D5 measurement was made on (runs/taxis/A_p4_wd0_c.npz).

    ring_k is set SMALL on purpose: CL-2's gradients come from autograd, not
    from the engine's ring credit, so the ring is never used -- and a small
    ring keeps the numpy evaluation from allocating GBs per layer.
    """
    model = gpu_port.GpuModel(dim=args.dim, layers=args.layers, lr=args.lr,
                              seed=args.seed, device=args.device,
                              half_life_init=(args.half_min, args.half_max),
                              ring_k=8, ring_credit="full",
                              gated=bool(getattr(args, "gated", False)))
    model.theta.requires_grad_(True)
    # THE FIX: the views must be re-made AFTER theta requires grad, or the
    # gradient is silently dropped (see rebind_views).
    rebind_views(model)
    return model


def build_head(model, n_out=PAYLOAD_LEN * 256):
    """The MEMORY HEAD: a linear map from the concatenated state (layers*dim)
    to 4*256 logits (a 256-way head per payload byte, as the plan asks).

    NOTE: the head's parameters must REQUIRE GRAD.  A freshly allocated
    torch tensor does not, and a parameter that does not require grad gets
    no gradient -- so the head would stay at its random initialisation and
    the "memory loss" would be a no-op (the same silent failure as the
    views, and it is checked by the selftest's "the weights move" gate).
    """
    n_state = model.layercount * model.dim
    Wm = torch.empty(n_state, n_out, dtype=torch.float32, device=model.device)
    torch.nn.init.xavier_normal_(Wm, gain=1.0)
    bm = torch.zeros(n_out, dtype=torch.float32, device=model.device)
    Wm.requires_grad_(True)
    bm.requires_grad_(True)
    return Wm, bm


# ---------------------------------------------------------------------------
# the GLOBAL BYTE PRIOR -- the comparison the FSP arm was judged on
# ---------------------------------------------------------------------------
def byte_prior(text_pool, alpha):
    """The GLOBAL BYTE PRIOR of the training corpus.

    This is the baseline the memory head has to beat: a CONSTANT predictor
    ("always predict the most frequent byte").  The FSP arm of PLAN §0 was
    declared a null because its head scored F1 0.685 against a prior of
    0.817 -- the objective was satisfied without the state.  The same
    comparison is reported here, per byte and exact-4.

    Two constants are reported:
      argmax        the most frequent byte of the corpus over all 256
      argmax_alpha  the most frequent byte of the corpus WITHIN the payload
                    alphabet -- the only constant that can be scored on the
                    payload targets at all
    """
    counts = np.bincount(text_pool.data, minlength=256).astype(np.float64)
    total = float(counts.sum())
    p = counts / max(total, 1.0)
    out = {"argmax": int(counts.argmax()),
           "p_argmax": float(p[counts.argmax()]),
           "top5": [[int(i), float(p[i])] for i in np.argsort(-p)[:5]],
           "corpus_bytes": int(total)}
    if alpha is not None:
        sub = counts[alpha]
        out["argmax_alpha"] = int(alpha[int(sub.argmax())])
        out["p_argmax_alpha"] = float(p[out["argmax_alpha"]])
    else:
        out["argmax_alpha"] = out["argmax"]
        out["p_argmax_alpha"] = out["p_argmax"]
    return out


# ---------------------------------------------------------------------------
# the held-out CE (the trajectory the arm must be watched on)
# ---------------------------------------------------------------------------
def eval_slices(files, n_slices, n_bytes):
    """n_slices disjoint windows per file -- the same slicing as
    tools/membench/model_backend.py `_slices` (the CE cell of the ruler)."""
    out = []
    for f in files:
        with open(f, "rb") as fh:
            raw = np.frombuffer(fh.read(), np.uint8)
        if len(raw) <= n_bytes + 1:
            continue
        stride = (len(raw) - n_bytes) // n_slices
        for i in range(n_slices):
            out.append(raw[i * stride:i * stride + n_bytes + 1])
    return out


def held_out_ce(model, slices):
    """The next-byte CE on held-out bytes, with the TORCH engine.

    Same bytes and same estimator as the ruler's CE cell (mb._eval_corpus ->
    eval_npz.cross_entropy), but computed on the GPU so it can be run DURING
    training: this is the trajectory that says whether the arm is learning
    anything at all (a model at UNIFORM sits at ln(256) = 5.5452).

    The slices are batched into ONE forward (the same number of kernel
    launches, more rows per launch), because the forward is launch-bound."""
    if not slices:
        return None
    x = np.stack([s[:min(len(s) for s in slices)] for s in slices])
    with torch.no_grad():
        xin = x.astype(np.int64)
        logits, _ = engine_forward(model, xin, xin.shape[1] - 1,
                                    use_checkpoint=False)
        tgt = torch.from_numpy(xin[:, 1:]).to(logits.device)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, 256),
                             tgt.reshape(-1))
    return float(ce)


# ---------------------------------------------------------------------------
# the memory evaluation: the head, the behaviour, and the prior
# ---------------------------------------------------------------------------
def eval_memory(model, heads, board, alpha, args, rng, prior, tag):
    """Evaluate the MEMORY HEAD and the BEHAVIOURAL recall on a board.

    No gradients and no weight update: this is a measurement, not a step.
    The head is scored on the state AT THE CUE (the same read the training
    loss uses) and the prior baselines are reported next to it, so "the head
    learned something" is never confused with "the head is at the prior".
    """
    alpha_size = 256 if alpha is None else len(alpha)
    text_pool = args._text_pool
    batch = build_fact_batch(args.eval_batch, board, alpha, text_pool,
                             args.text_frac, args.gap, rng, n_facts=args.facts)
    lay = batch["lay"]
    # the EVAL-TIME SWITCH: a fresh random head for the measurement, the
    # trained head restored afterwards (the training is untouched).
    saved = None
    if getattr(args, "head_eval_fresh", False) and heads[0] is not None:
        saved = (heads[0].detach().clone(), heads[1].detach().clone())
        with torch.no_grad():
            torch.nn.init.xavier_normal_(heads[0], gain=1.0)
            heads[1].zero_()
    with torch.no_grad():
        logits, state_cue = engine_forward(model, batch["seq"], lay["cue_last"],
                                           use_checkpoint=False)
        pay = torch.from_numpy(batch["pay_new"]).to(logits.device)
        out = {"tag": tag, "n": int(args.eval_batch)}
        if heads[0] is not None:
            ml = state_cue @ heads[0] + heads[1]
            hp = ml.reshape(-1, 256).argmax(-1).reshape(pay.shape)
            out["head_per_byte"] = float((hp == pay).float().mean())
            out["head_exact4"] = float((hp == pay).all(dim=1).float().mean())
            out["head_ce"] = float(F.cross_entropy(ml.reshape(-1, 256),
                                                   pay.reshape(-1)))
        b0, b1 = batch["behav_slice"]
        if b1 > b0:
            bp = logits[:, b0:b1].argmax(-1)
            bt = torch.from_numpy(
                batch["seq"][:, b0 + 1:b1 + 1].astype(np.int64)).to(logits.device)
            out["behav_per_byte"] = float((bp == bt).float().mean())
            out["behav_exact4"] = float((bp == bt).all(dim=1).float().mean())
    if saved is not None:
        with torch.no_grad():
            heads[0].copy_(saved[0])
            heads[1].copy_(saved[1])
    # the prior baselines on the SAME targets
    tgt = batch["pay_new"]
    for name, byte in (("prior", prior["argmax_alpha"]),
                       ("prior_global", prior["argmax"])):
        hit = (tgt == byte)
        out[f"{name}_per_byte"] = float(hit.mean())
        out[f"{name}_exact4"] = float(hit.all(axis=1).mean())
    out["chance_per_byte"] = 1.0 / alpha_size
    out["chance_exact4"] = (1.0 / alpha_size) ** PAYLOAD_LEN
    return out


class AdamW:
    """A plain, decoupled AdamW over a LIST of tensors, with explicit moment
    buffers (so the checkpoint can carry them as numpy arrays and a resume
    continues the run exactly).  wd is 0 by default: the engine's P6 finding
    is that weight decay erodes the decay logits, i.e. it shortens the
    half-lives -- the one thing a memory run cannot afford (PLAN §0, P4/P6).
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, wd=0.0,
                 wd_decay=0.0, decay_masks=None):
        self.params = list(params)
        self.lr = float(lr)
        self.b1, self.b2, self.eps, self.wd = betas[0], betas[1], eps, wd
        # P6: the decoupled decay on the DECAY LOGITS only.  The engine's
        # finding is that a weight decay on the decay logits is a systematic
        # pull toward sigma = 0.5, i.e. toward a ONE-BYTE half-life, which
        # destroys exactly the multi-scale memory a fact run needs.  So the
        # decay is selectable and OFF by default, and it can be aimed at the
        # decay logits alone (decay_masks: one bool tensor per param, or None).
        self.wd_decay = float(wd_decay)
        self.decay_masks = decay_masks
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        with torch.no_grad():
            for i, (p, m, v) in enumerate(zip(self.params, self.m, self.v)):
                if p.grad is None:
                    continue
                g = p.grad
                m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
                upd = (m / bc1) / ((v / bc2).sqrt() + self.eps)
                if self.wd:
                    upd = upd + self.wd * p
                if self.wd_decay:
                    mask = (self.decay_masks[i] if self.decay_masks else None)
                    if mask is not None:
                        upd = upd + self.wd_decay * p * mask
                p.add_(upd, alpha=-lr)
        return self.t

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_arrays(self):
        out = {"opt_t": np.array(self.t, np.int64)}
        for i, (m, v) in enumerate(zip(self.m, self.v)):
            out[f"opt_m{i}"] = m.detach().cpu().numpy()
            out[f"opt_v{i}"] = v.detach().cpu().numpy()
        return out

    def load_state_arrays(self, d, n_params):
        if "opt_t" in d:
            self.t = int(d["opt_t"])
        for i in range(n_params):
            if f"opt_m{i}" in d:
                self.m[i].copy_(torch.from_numpy(d[f"opt_m{i}"]).to(self.m[i].device))
                self.v[i].copy_(torch.from_numpy(d[f"opt_v{i}"]).to(self.v[i].device))


# ---------------------------------------------------------------------------
# the forward (autograd over the engine's own layers)
# ---------------------------------------------------------------------------
def engine_forward(model, seq, capture_idx, use_checkpoint=True):
    """The engine's forward, under autograd.

    seq        (B, L) uint8
    capture_idx the byte index whose state is returned (the CUE)
    returns    (logits (B, L, 256), state_at_capture (B, layers*dim))

    The per-timestep graph is checkpointed: without it, L timesteps x 16
    layers of (B, 512) activations is ~4 GB at B=256, L=72.  The checkpoint
    keeps only the inputs of each timestep (enc + the 16 layer states) and
    recomputes the rest in the backward -- the same arithmetic, ~2x the
    forward cost, ~L times less memory.
    """
    B, L = seq.shape
    dev = model.device
    seq_f = torch.from_numpy(seq.astype(np.int64)).to(dev)
    layers = model.layers

    def step_fn(enc, *states):
        x = enc
        new_states = []
        for i, l in enumerate(layers):
            x, st, _a, _c = l.forward(enc, x, states[i])
            new_states.append(st)
        logits = x @ model.Wd.t() + model.bd
        return logits, tuple(new_states)

    states = [torch.zeros((B, model.dim), dtype=torch.float32, device=dev)
              for _ in layers]
    logits = []
    captured = None
    for t in range(L):
        enc = model.embed[seq_f[:, t]]
        if use_checkpoint and torch.is_grad_enabled():
            lg, states = checkpoint(step_fn, enc, *states, use_reentrant=False)
        else:
            lg, states = step_fn(enc, *states)
        logits.append(lg)
        if t == capture_idx:
            captured = torch.cat(states, dim=1)
    return torch.stack(logits, dim=1), captured


# ---------------------------------------------------------------------------
# one training step
# ---------------------------------------------------------------------------
def train_step(model, opt, heads, batch, args, wm=None, wm_old=None):
    """One optimizer step: next-byte CE (text + the behavioural target) plus
    the MEMORY LOSS on the state at the cue."""
    lay = batch["lay"]
    seq = batch["seq"]
    B, L = seq.shape
    logits, state_cue = engine_forward(model, seq, lay["cue_last"],
                                       use_checkpoint=not args.no_checkpoint)

    tgt = torch.from_numpy(seq[:, 1:].astype(np.int64)).to(logits.device)
    flat = F.cross_entropy(logits[:, :-1].reshape(-1, 256), tgt.reshape(-1),
                            reduction="none").view(B, L - 1)
    mask = torch.from_numpy(batch["ce_mask"]).to(logits.device)
    n_text = int(mask.sum().item())
    ce_text = (flat * mask).sum() / max(n_text, 1)

    b0, b1 = batch["behav_slice"]
    ce_behav = flat[:, b0:b1].mean() if b1 > b0 else torch.zeros((), device=logits.device)

    pay_new = torch.from_numpy(batch["pay_new"]).to(logits.device)
    mem = torch.zeros((), device=logits.device)
    head_per_byte = 0.0
    head_exact4 = 0.0
    if wm is not None:
        ml = state_cue @ wm + heads[1]
        mem = F.cross_entropy(ml.reshape(-1, 256), pay_new.reshape(-1))
        # the head's ACCURACY on the training batch (the board the phase is
        # training on).  The CE alone does not say whether the head is at the
        # prior; the accuracy does.
        with torch.no_grad():
            hp = ml.detach().reshape(-1, 256).argmax(-1).reshape(pay_new.shape)
            head_per_byte = float((hp == pay_new).float().mean())
            head_exact4 = float((hp == pay_new).all(dim=1).float().mean())
    mem_old = torch.zeros((), device=logits.device)
    if wm_old is not None:
        pay_old = torch.from_numpy(batch["pay_old"]).to(logits.device)
        ml = state_cue @ wm_old + heads[3]
        mem_old = F.cross_entropy(ml.reshape(-1, 256), pay_old.reshape(-1))

    # the BEHAVIOURAL recall: the model's OWN output at the four slots after
    # the cue, against the payload it was asked to emit.  This is the plan's
    # notion of "recalling" (a behaviour), and it is free here -- the logits
    # are already computed.
    behav_per_byte = 0.0
    behav_exact4 = 0.0
    if b1 > b0:
        with torch.no_grad():
            bp = logits[:, b0:b1].detach().argmax(-1)
            bt = torch.from_numpy(seq[:, b0 + 1:b1 + 1].astype(np.int64)).to(
                logits.device)
            behav_per_byte = float((bp == bt).float().mean())
            behav_exact4 = float((bp == bt).all(dim=1).float().mean())

    loss = (ce_text + args.behaviour_weight * ce_behav
            + args.memory_alpha * mem + args.memory_alpha_old * mem_old)
    opt.zero_grad()
    loss.backward()
    # M3 (the matched control): the gate is PRESENT but FROZEN at exactly 0.
    # Its gradient is zeroed here, so AdamW's update for those entries is
    # 0/(0+eps) = 0 and the gate stays bit-exactly 0 for the whole run: the
    # arm then differs from the GATED arm in exactly one thing -- whether the
    # gate is allowed to move.  (Without this, the "frozen" control would be
    # the ungated arm, which also uses the ring, and the comparison would
    # confound the gate with the trace.)
    if getattr(args, "gate_frozen", False) and getattr(model, "gated", False):
        g = model.theta.grad
        if g is not None:
            off = 0
            for name, shape in model.shapes:
                size = int(np.prod(shape))
                if name.endswith(".gate"):
                    g[off:off + size] = 0.0
                off += size
    opt.step()
    return {"loss": float(loss.detach()),
            "ce_text": float(ce_text.detach()),
            "ce_behav": float(ce_behav.detach()),
            "mem": float(mem.detach()),
            "mem_old": float(mem_old.detach()),
            "head_per_byte": head_per_byte,
            "head_exact4": head_exact4,
            "behav_per_byte": behav_per_byte,
            "behav_exact4": behav_exact4}


# ---------------------------------------------------------------------------
# checkpoints (resumable)
# ---------------------------------------------------------------------------
def save_checkpoint(model, opt, heads, path, meta):
    """A numpy_port-compatible checkpoint (so tools/membench/model_backend.py
    can load it and the CL-RULER can score it) PLUS the CL-2 state: the
    memory head, the optimizer moments, the phase and the corpus position."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {}
    # BUG FIX (2026-10-22): this used to save model.GVIEWS -- the views into
    # the GRADIENT buffer (model.grads) -- under the PARAMETER names.  So
    # every "checkpoint" written by the CL-2 trainer held the GRADIENTS, not
    # the WEIGHTS, and loading one installed gradients as weights.  The
    # parameter views (model.views) are the ones that share theta's storage.
    for name, view in model.views.items():
        data[name] = view.detach().cpu().numpy()
    data["m"] = model.m.detach().cpu().numpy()
    data["v"] = model.v.detach().cpu().numpy()
    data["t"] = np.array(model.t, np.int64)
    data["bytes"] = np.array(model.bytes_seen, np.int64)
    data["B"] = np.array(model.B, np.int64)
    data["ring_k"] = np.array(model.ring_k, np.int64)
    for i, l in enumerate(model.layers):
        data[f"state.{i}"] = l.states.detach().cpu().numpy()
        data[f"decaytrace.{i}"] = l.decaytrace.detach().cpu().numpy()
    # the CL-2 extras
    data["Wm"] = heads[0].detach().cpu().numpy()
    data["bm"] = heads[1].detach().cpu().numpy()
    data["Wm_old"] = heads[2].detach().cpu().numpy()
    data["bm_old"] = heads[3].detach().cpu().numpy()
    data.update(opt.state_arrays())
    for k, v in meta.items():
        data[f"cl2_{k}"] = np.array(v)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **data)
    os.replace(tmp, path)
    return path


def load_checkpoint(model, opt, heads, path):
    d = np.load(path, allow_pickle=False)
    for name, view in model.views.items():
        if name in d.files:
            with torch.no_grad():
                view.copy_(torch.from_numpy(d[name]))
    for i, key in enumerate(("Wm", "bm", "Wm_old", "bm_old")):
        if key in d.files:
            with torch.no_grad():
                heads[i].copy_(torch.from_numpy(d[key]).to(heads[i].device))
    opt.load_state_arrays(d, len(opt.params))
    meta = {}
    for k in d.files:
        if k.startswith("cl2_"):
            meta[k[4:]] = d[k]
    return meta


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre-steps", type=int, default=0,
                    help="phase 'pre': text-only CE steps (the general ability)")
    ap.add_argument("--phase1-steps", type=int, default=0,
                    help="phase 'f1': the PRE-STREAM facts (board A)")
    ap.add_argument("--phase2-steps", type=int, default=0,
                    help="phase 'f2': the STREAM facts (board B), no replay of A")
    ap.add_argument("--batch", type=int, default=256, help="streams per step")
    ap.add_argument("--facts", type=int, default=4,
                    help="facts per stream (the ruler's --facts)")
    ap.add_argument("--gap", type=int, default=32,
                    help="the gap d (the pre-registered ACQUIRE lag is 32)")
    ap.add_argument("--alphabet", default="16", choices=("full", "16", "4"),
                    help="the payload alphabet (the matched one)")
    ap.add_argument("--board-size", type=int, default=512,
                    help="facts in each board")
    ap.add_argument("--board-seed-a", type=int, default=101,
                    help="the seed of board A (the pre-stream facts)")
    ap.add_argument("--board-seed-b", type=int, default=202,
                    help="the seed of board B (the stream facts)")
    ap.add_argument("--board-seed-c", type=int, default=303,
                    help="the seed of board C -- the HELD-OUT board, never "
                         "trained on; the ruler scores the acquire cell on it "
                         "so that a memorisation of board B cannot pass as a "
                         "memory")
    ap.add_argument("--text-frac", type=float, default=0.5,
                    help="fraction of the samples whose gap is real text")
    ap.add_argument("--text-files", default="data/*.txt",
                    help="the real-text background (NOT data/eval)")
    ap.add_argument("--memory-alpha", type=float, default=0.25,
                    help="the weight of the memory loss (the NEW thing)")
    ap.add_argument("--head-frozen", action="store_true",
                    help="ARM H1: FREEZE the memory head's weights (and bias) "
                         "at their random initialisation.  The head is then a "
                         "FIXED random readout of the state at the cue: it "
                         "cannot accumulate a key->payload TABLE, so the ONLY "
                         "way to reduce the memory loss is to make the STATE "
                         "carry the payload.  (The model's own weights could "
                         "still carry a table -- the RETAIN cell and the copy "
                         "control are the instruments that expose that.)")
    ap.add_argument("--memory-alpha-old", type=float, default=0.0,
                    help="the weight of the oldest-fact memory head (the "
                         "ruler's RETAIN cell target)")
    ap.add_argument("--behaviour-weight", type=float, default=1.0,
                    help="the weight of the behavioural CE (emit the payload "
                         "after the cue)")
    ap.add_argument("--half-min", type=float, default=2.0)
    ap.add_argument("--half-max", type=float, default=512.0)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0,
                    help="weight decay on EVERY parameter (0: the P4/P6 "
                         "finding is that decay erodes the decay logits)")
    ap.add_argument("--wd-decay", type=float, default=0.0,
                    help="P6: the decoupled weight decay on the DECAY LOGITS "
                         "ONLY (0 by default: a decay on the decay logits is a "
                         "systematic pull toward a ONE-BYTE half-life, i.e. "
                         "toward no memory at all)")
    ap.add_argument("--head-eval-fresh", action="store_true",
                    help="EVAL-TIME SWITCH: at every memory evaluation, use a "
                         "FRESHLY re-initialised (random) head instead of the "
                         "trained one, and restore it afterwards.  This is the "
                         "FALSIFIER for a table: if the head's accuracy "
                         "collapses to the prior with a fresh head, the "
                         "answer was in the head's WEIGHTS, not in the state. "
                         "The default path is untouched.")
    ap.add_argument("--eval-batch", type=int, default=256,
                    help="streams per evaluation batch (the memory head's "
                         "accuracy, the behavioural recall and the prior)")
    ap.add_argument("--ce-eval-every", type=int, default=0,
                    help="every N steps, measure the HELD-OUT CE on "
                         "data/eval (0: only at the phase boundaries).  This is "
                         "the trajectory that says whether the arm is learning "
                         "anything at all: a model at UNIFORM sits at "
                         "ln(256) = 5.5452")
    ap.add_argument("--ce-eval-bytes", type=int, default=2000,
                    help="bytes per held-out slice for the CE trajectory")
    ap.add_argument("--ce-eval-slices", type=int, default=2,
                    help="held-out slices per file for the CE trajectory")
    ap.add_argument("--ce-eval-files", default="data/eval/*.txt",
                    help="the held-out corpus of the CE trajectory (the same "
                         "corpus the ruler's CE cell scores)")
    ap.add_argument("--ce-log", default=None,
                    help="a JSONL file for the CE trajectory "
                         "(default: <ckpt-dir>/<tag>_ce.jsonl)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gated", action="store_true",
                    help="P3: the GATED recurrence, a = sigma(decay + "
                         "gate*x_in). The gate is a per-dim vector in theta, "
                         "it starts at EXACTLY 0 (so the arm's starting point "
                         "is bit-identical to the incumbent), and it is "
                         "trainable. The CL-2 trainer is autograd-based, so "
                         "the gate's gradient comes from the forward pass "
                         "itself -- the engine's credit path is not used by "
                         "this trainer at all.")
    ap.add_argument("--gate-frozen", action="store_true",
                    help="M3, the MATCHED CONTROL: the gate is present but "
                         "frozen at exactly 0 (its gradient is zeroed before "
                         "the optimizer step). The arm then differs from the "
                         "--gated arm in exactly one thing: whether the gate "
                         "moves. Requires --gated.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt-dir", default="runs/cl2")
    ap.add_argument("--resume", default=None,
                    help="a checkpoint to resume from")
    ap.add_argument("--resume-phase", default=None,
                    help="the phase to resume (pre|f1|f2)")
    ap.add_argument("--resume-step", type=int, default=None,
                    help="the step within the phase to resume at")
    ap.add_argument("--tag", default="cl2", help="a tag for the checkpoint names")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="do not checkpoint (the memory head is lost)")
    ap.add_argument("--no-checkpointing", action="store_true",
                    help="disable the per-timestep graph checkpointing")
    ap.add_argument("--selftest", action="store_true",
                    help="parity + gradient-path self test")
    return ap


# ---------------------------------------------------------------------------
# the self test: (i) the autograd CE matches the engine's own CE on the same
# bytes; (ii) the memory loss's gradient REACHES the payload bytes (the
# cross-time path is real); (iii) the state at the cue is the state at the
# cue (the D4 check).
# ---------------------------------------------------------------------------
def selftest(args):
    ok = True
    print("== CL-2 trainer selftest ==")
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.device = dev
    model = build_model(args)
    heads = build_head(model)
    rng = np.random.default_rng(0)
    B, L = 4, 24
    seq = rng.integers(0, 256, size=(B, L)).astype(np.uint8)
    # (i) the autograd forward's CE vs the engine's own step_batch CE
    logits, _ = engine_forward(model, seq, L - 1, use_checkpoint=False)
    _tgt = torch.from_numpy(seq[:, 1:].astype(np.int64)).to(logits.device)
    ce_auto = float(F.cross_entropy(logits[:, :-1].reshape(-1, 256),
                                    _tgt.reshape(-1)).detach())
    with torch.no_grad():
        logits2, _ = engine_forward(model, seq, L - 1, use_checkpoint=False)
    ce_auto2 = float(F.cross_entropy(logits2[:, :-1].reshape(-1, 256),
                                     _tgt.reshape(-1)).detach())
    print(f"  (i) autograd CE {ce_auto:.6f} | no_grad CE {ce_auto2:.6f} "
          f"(must agree to 1e-4) -> "
          f"{'OK' if abs(ce_auto - ce_auto2) < 1e-4 else 'FAIL'}")
    ok &= abs(ce_auto - ce_auto2) < 1e-4

    # (ii) the memory loss's gradient reaches the payload bytes
    lay = sample_layout(2, 8)
    board = make_board(7, 16, alphabet_symbols("16"))
    batch = build_fact_batch(B, board, alphabet_symbols("16"),
                             TextPool(glob.glob(os.path.join(_REPO, "data/*.txt"))[:2], 0),
                             0.5, 8, rng, n_facts=2)
    wm, bm = heads
    opt = AdamW([model.theta, wm, bm], args.lr, wd=args.wd)
    logits, state_cue = engine_forward(model, batch["seq"], lay["cue_last"],
                                       use_checkpoint=False)
    ml = state_cue @ wm + bm
    mem = F.cross_entropy(ml.reshape(-1, 256),
                          torch.from_numpy(batch["pay_new"]).to(
                              ml.device).reshape(-1))
    opt.zero_grad()
    mem.backward()
    # the gradient lands on THETA (the flat parameter vector); the embedding
    # table is a VIEW into it, so the view's own .grad is None -- the old
    # selftest read model.embed.grad and crashed (the selftest had never
    # been run).  The embed's slice of theta's gradient is what carries the
    # cross-time path.
    off = 0
    for name, shape in model.shapes:
        if name == "embed":
            break
        off += int(np.prod(shape))
    g = model.theta.grad[off:off + 256 * model.dim].reshape(model.dim, 256)
    pay_pos = lay["gap0"] - PAYLOAD_LEN        # the payload of the newest fact
    # the embed table is (dim, 256): emb[b] is a COLUMN, so the gradient of
    # the byte b is the column b.
    gpay = float(g[:, batch["seq"][0, pay_pos]].abs().sum())
    gcue = float(g[:, batch["seq"][0, lay["cue0"]]].abs().sum())
    print(f"  (ii) |d mem_loss / d embed| at the payload byte {gpay:.3e}, "
          f"at the cue byte {gcue:.3e} -> "
          f"{'OK (the gradient reaches the payload)' if gpay > 0 else 'FAIL'}")
    ok &= gpay > 0

    # (iii) the state at the cue does not depend on the bytes AFTER it.
    # The capture index is L-4: the state AFTER byte L-4, i.e. BEFORE the
    # three bytes that differ.  (The old check captured at L-1, i.e. AFTER
    # the changed bytes, so it compared two different states and could only
    # ever report a difference -- it had never been run.)
    s1 = np.random.default_rng(1).integers(0, 256, size=(B, L)).astype(np.uint8)
    s2 = s1.copy()
    s2[:, L - 3:] = np.random.default_rng(2).integers(0, 256, size=(B, 3))
    with torch.no_grad():
        _, st1 = engine_forward(model, s1, L - 4, use_checkpoint=False)
        _, st2 = engine_forward(model, s2, L - 4, use_checkpoint=False)
    d = float((st1 - st2).abs().max())
    print(f"  (iii) bytes after the cue changed the cue state by {d:.3e} "
          f"-> {'OK' if d == 0.0 else 'FAIL'}")
    ok &= d == 0.0

    # (iv) THE GATE THAT WAS MISSING: after ONE optimizer step the WEIGHTS
    # must MOVE.  The first CL-2 arm scored ln(256) -- "the model never
    # learned anything" -- and the reason was not the budget: the parameter
    # views were DETACHED leaves (GpuModel builds them before theta requires
    # grad), so the gradient was silently dropped and the optimizer updated
    # nothing.  This gate fails on that code and passes on the fixed code.
    th0 = model.theta.detach().clone()
    wm0 = wm.detach().clone()
    opt2 = AdamW([model.theta, wm, bm], args.lr, wd=args.wd)
    st = train_step(model, opt2, [wm, bm, wm, bm], batch, args, wm=wm)
    dth = float((model.theta.detach() - th0).abs().max())
    dhd = float((wm.detach() - wm0).abs().max())
    print(f"  (iv) after ONE step: |d theta| = {dth:.3e}, |d head| = {dhd:.3e} "
          f"-> {'OK (the weights move)' if dth > 0 and dhd > 0 else 'FAIL (the '
             'gradient is being dropped -- the arm cannot train)'}")
    ok &= bool(dth > 0 and dhd > 0)

    # (v) the checkpoint round-trip: what is written must be the WEIGHTS
    tmpdir = args.ckpt_dir or "runs/cl2"
    os.makedirs(tmpdir, exist_ok=True)
    tmp = os.path.join(tmpdir, f"cl2_selftest_{os.getpid()}.npz")
    save_checkpoint(model, opt2, [wm, bm, wm, bm], tmp,
                    {"phase": "selftest", "step": 1})
    d = np.load(tmp)
    w_err = float(np.abs(d["embed"] - model.embed.detach().cpu().numpy()).max())
    g_err = float(np.abs(d["embed"] - model.gviews["embed"].detach().cpu().numpy()).max())
    os.remove(tmp)
    print(f"  (v) the checkpoint holds the WEIGHTS (max|d| = {w_err:.3e}) and "
          f"not the gradients (max|d| = {g_err:.3e}) -> "
          f"{'OK' if w_err == 0.0 and g_err > 0 else 'FAIL'}")
    ok &= bool(w_err == 0.0 and g_err > 0)

    print(f"== selftest {'PASS' if ok else 'FAIL'} ==")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# the training loop
# ---------------------------------------------------------------------------
def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.selftest:
        return selftest(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    alpha = alphabet_symbols(args.alphabet)
    alpha_size = 256 if alpha is None else len(alpha)
    board_a = make_board(args.board_seed_a, args.board_size, alpha)
    board_b = make_board(args.board_seed_b, args.board_size, alpha)
    board_c = make_board(args.board_seed_c, args.board_size, alpha)
    text_files = sorted(glob.glob(os.path.join(_REPO, args.text_files)))
    text_pool = TextPool(text_files, seed=args.seed)
    args._text_pool = text_pool
    prior = byte_prior(text_pool, alpha)
    print(f"[prior] the global byte prior of the training corpus: "
          f"argmax {prior['argmax']} (p={prior['p_argmax']:.4f}) | "
          f"within the payload alphabet: {prior['argmax_alpha']} "
          f"(p={prior['p_argmax_alpha']:.4f})", flush=True)

    model = build_model(args)
    # the memory head: [Wm, bm] for the newest fact, [Wm_old, bm_old] for the
    # oldest fact (the ruler's RETAIN cell target).
    heads = list(build_head(model)) + list(build_head(model))
    if args.head_frozen:
        # ARM H1: the head is a FIXED random readout of the state.  It cannot
        # learn, so it cannot carry the key->payload table; the memory loss
        # can then only be reduced by writing the payload into the STATE.
        for h in heads:
            h.requires_grad_(False)
        print("[head] FROZEN at the random init (ARM H1): the memory loss "
              "can now only be reduced through the STATE.", flush=True)
    # P6: the decay mask -- the DECAY LOGITS only, and only if --wd-decay is
    # not 0.  The mask is a bool vector over theta, aligned with model.shapes.
    decay_mask = None
    if args.wd_decay:
        decay_mask = torch.zeros_like(model.theta, dtype=torch.bool)
        off = 0
        for name, shape in model.shapes:
            size = int(np.prod(shape))
            if name.endswith(".decay"):
                decay_mask[off:off + size] = True
            off += size
    opt = AdamW([model.theta] + heads, args.lr, wd=args.wd,
                wd_decay=args.wd_decay,
                decay_masks=[decay_mask] + [None] * len(heads) if decay_mask
                is not None else None)

    # the resume
    start_phase = "pre"
    start_step = 0
    if args.resume:
        meta = load_checkpoint(model, opt, heads, args.resume)
        p = meta.get("phase", "pre")
        start_phase = str(p)
        start_step = args.resume_step if args.resume_step is not None else int(
            meta.get("step", 0))
        print(f"[resume] {args.resume}: phase={start_phase} step={start_step} "
              f"opt_t={opt.t}", flush=True)
    else:
        # the step-0 checkpoint: the PRE-STREAM baseline for every number
        save_checkpoint(model, opt, heads,
                        os.path.join(args.ckpt_dir, f"{args.tag}_t0.npz"),
                        {"phase": "init", "step": 0, "seed": args.seed,
                         "alphabet": args.alphabet, "facts": args.facts,
                         "gap": args.gap, "board_size": args.board_size,
                         "board_seed_a": args.board_seed_a,
                         "board_seed_b": args.board_seed_b,
                         "board_seed_c": args.board_seed_c,
                         "head_frozen": bool(args.head_frozen),
                         "prior": json.dumps(prior)})
        print(f"[init] from scratch, seed={args.seed}, P4 half-life "
              f"{args.half_min}-{args.half_max} B, dim={args.dim}, "
              f"layers={args.layers}, device={args.device}", flush=True)

    lay = sample_layout(args.facts, args.gap)
    L = lay["length"]
    print(f"[corpus] L={L} B (facts={args.facts} x 8 B + gap {args.gap} B + "
          f"cue 4 B + target 4 B) | alphabet {alpha_size} symbols | "
          f"text {len(text_files)} files, {text_pool.n:,} B | "
          f"boards A/B = {args.board_size} facts each", flush=True)

    phases = [("pre", args.pre_steps, None), ("f1", args.phase1_steps, board_a),
              ("f2", args.phase2_steps, board_b)]
    phase_id = {"pre": 1, "f1": 2, "f2": 3}
    start_idx = 0
    for i, (p, _n, _b) in enumerate(phases):
        if p == start_phase:
            start_idx = i

    # -- the held-out CE trajectory -----------------------------------------
    ce_log_path = (args.ce_log or
                   os.path.join(args.ckpt_dir, f"{args.tag}_ce.jsonl"))
    ce_files = sorted(glob.glob(os.path.join(_REPO, args.ce_eval_files)))
    ce_slices = eval_slices(ce_files, args.ce_eval_slices,
                            args.ce_eval_bytes)
    print(f"[ce] the held-out CE trajectory: {len(ce_slices)} slices x "
          f"{args.ce_eval_bytes} B from {len(ce_files)} files -> "
          f"{ce_log_path}", flush=True)
    ce_fh = open(ce_log_path, "a")

    def log_row(phase, step, stats, held_ce):
        row = {"phase": phase, "step": int(step),
               "global_step": int(model.t), "held_out_ce": held_ce,
               "ce_text": stats.get("ce_text"),
               "ce_behav": stats.get("ce_behav"), "mem": stats.get("mem"),
               "head_per_byte": stats.get("head_per_byte"),
               "head_exact4": stats.get("head_exact4"),
               "behav_per_byte": stats.get("behav_per_byte"),
               "behav_exact4": stats.get("behav_exact4")}
        ce_fh.write(json.dumps(row) + "\n")
        ce_fh.flush()
        return row

    t_wall = time.time()
    for pi, (phase, n_steps, board) in enumerate(phases):
        if n_steps <= 0 or pi < start_idx:
            continue
        step0 = start_step if pi == start_idx else 0
        # the phase's own memory evaluation (on the board it trains on) and the
        # HELD-OUT board C (never trained on): the difference between the two
        # is memorisation vs a mechanism.
        eval_rng = np.random.default_rng(
            (args.seed * 7_919 + phase_id[phase] * 104_729) % (2 ** 32))
        evals = {}
        for bname, brd in (("train_board", board), ("heldout_board", board_c)):
            if brd is None:
                continue
            evals[bname] = eval_memory(model, heads, brd, alpha, args,
                                       eval_rng, prior, f"{phase}/{bname}")
            e = evals[bname]
            print(f"  [eval {phase}/{bname}] head {e.get('head_per_byte', 0):.4f}"
                  f"/byte, exact4 {e.get('head_exact4', 0):.4f} | behaviour "
                  f"{e.get('behav_per_byte', 0):.4f}/byte, exact4 "
                  f"{e.get('behav_exact4', 0):.4f} | prior "
                  f"{e['prior_per_byte']:.4f}/byte, exact4 "
                  f"{e['prior_exact4']:.4f} | chance "
                  f"{e['chance_per_byte']:.4f}/{e['chance_exact4']:.2e}",
                  flush=True)
        held_ce0 = held_out_ce(model, ce_slices)
        log_row(phase, 0, {}, held_ce0)
        print(f"  [ce {phase} 0/{n_steps}] held-out CE {held_ce0}", flush=True)
        for step in range(step0, n_steps):
            # the corpus is a deterministic function of (seed, phase, step):
            # the step counter IS the data position.
            rng = np.random.default_rng(
                (args.seed * 1_000_003 + phase_id[phase] * 10_007 + step)
                % (2 ** 32))
            if board is None:
                batch = build_text_batch(args.batch, L, text_pool, rng)
            else:
                batch = build_fact_batch(args.batch, board, alpha, text_pool,
                                         args.text_frac, args.gap, rng,
                                         n_facts=args.facts)
            stats = train_step(model, opt, heads, batch, args,
                               # the memory loss is a FACT loss: it must not
                               # run in the text-only 'pre' phase, where the
                               # "target" is a constant zero and the head
                               # would be trained to always predict it.
                               wm=(heads[0]
                                   if (args.memory_alpha > 0 and board is not None)
                                   else None),
                               wm_old=heads[2] if args.memory_alpha_old > 0 else None)
            model.t += 1
            model.bytes_seen += args.batch * L
            if getattr(model, "gated", False):
                gs = [float(l.gate.detach().pow(2).mean().sqrt())
                      for l in model.layers]
                gmax = max(float(l.gate.detach().abs().max())
                           for l in model.layers)
                stats["gate_rms_mean"] = float(np.mean(gs))
                stats["gate_rms_max"] = float(np.max(gs))
                stats["gate_abs_max"] = gmax
            else:
                stats["gate_rms_mean"] = 0.0
                stats["gate_rms_max"] = 0.0
                stats["gate_abs_max"] = 0.0
            do_ce = (args.ce_eval_every and
                     ((step + 1) % args.ce_eval_every == 0 or step == n_steps - 1))
            held_ce = held_out_ce(model, ce_slices) if do_ce else None
            if (step + 1) % args.log_every == 0 or step == 0:
                el = time.time() - t_wall
                print(f"  [{phase} {step + 1}/{n_steps}] loss {stats['loss']:.4f} "
                      f"| ce_text {stats['ce_text']:.4f} "
                      f"| ce_behav {stats['ce_behav']:.4f} "
                      f"| mem {stats['mem']:.4f} | mem_old {stats['mem_old']:.4f} "
                      f"| head {stats['head_per_byte']:.3f}/"
                      f"{stats['head_exact4']:.3f} "
                      f"| behav {stats['behav_per_byte']:.3f}/"
                      f"{stats['behav_exact4']:.3f} "
                      f"| held_ce {held_ce if held_ce is None else round(held_ce, 4)} "
                      f"| gate {stats['gate_rms_mean']:.4f}/"
                      f"{stats['gate_abs_max']:.4f} "
                      f"| {el:.0f}s", flush=True)
            if do_ce:
                log_row(phase, step + 1, stats, held_ce)
            if (not args.no_checkpoint) and \
                    ((step + 1) % args.save_every == 0 or step == n_steps - 1):
                path = os.path.join(args.ckpt_dir,
                                    f"{args.tag}_{phase}_t{step + 1:06d}.npz")
                save_checkpoint(model, opt, heads, path, {"phase": phase, "step": step + 1,
                                       "seed": args.seed, "alphabet": args.alphabet,
                                       "facts": args.facts, "gap": args.gap,
                                       "board_size": args.board_size,
                                       "board_seed_a": args.board_seed_a,
                                       "board_seed_b": args.board_seed_b,
                                       "board_seed_c": args.board_seed_c,
                                       "head_frozen": bool(args.head_frozen),
                                       "prior": json.dumps(prior),
                                       "memory_eval": json.dumps(evals)})
                print(f"    [save] {path}", flush=True)
        # the phase boundary
        held_ce_end = held_out_ce(model, ce_slices)
        log_row(phase, n_steps, {}, held_ce_end)
        print(f"  [ce {phase} {n_steps}/{n_steps}] held-out CE {held_ce_end}",
              flush=True)
        # the memory evaluation at the END of the phase (the phase-start one
        # above is measured BEFORE the phase trains, so it is the previous
        # phase's state; this one is the phase's own result).
        end_evals = {}
        for bname, brd in (("train_board", board), ("heldout_board", board_c)):
            if brd is None:
                continue
            end_evals[bname] = eval_memory(model, heads, brd, alpha, args,
                                           eval_rng, prior,
                                           f"{phase}-end/{bname}")
            e = end_evals[bname]
            print(f"  [eval {phase}-end/{bname}] head "
                  f"{e.get('head_per_byte', 0):.4f}/byte, exact4 "
                  f"{e.get('head_exact4', 0):.4f} | behaviour "
                  f"{e.get('behav_per_byte', 0):.4f}/byte, exact4 "
                  f"{e.get('behav_exact4', 0):.4f} | prior "
                  f"{e['prior_per_byte']:.4f}/{e['prior_exact4']:.4f} | chance "
                  f"{e['chance_per_byte']:.4f}/{e['chance_exact4']:.2e}",
                  flush=True)
        evals = {**evals, **{f"end_{k}": v for k, v in end_evals.items()}}
        if not args.no_checkpoint:
            path = os.path.join(args.ckpt_dir, f"{args.tag}_{phase}_end.npz")
            save_checkpoint(model, opt, heads, path, {"phase": phase, "step": n_steps, "seed": args.seed,
                                   "alphabet": args.alphabet, "facts": args.facts,
                                   "gap": args.gap, "board_size": args.board_size,
                                   "board_seed_a": args.board_seed_a,
                                   "board_seed_b": args.board_seed_b,
                                   "board_seed_c": args.board_seed_c,
                                   "prior": json.dumps(prior),
                                   "gated": int(bool(args.gated)),
                                   "gate_frozen": int(bool(args.gate_frozen)),
                                   "memory_eval": json.dumps(evals),
                                   "held_out_ce": float(held_ce_end or 0.0)})
            print(f"    [phase end] {path}", flush=True)
        start_phase = ""
        start_step = 0
    ce_fh.close()
    print(f"[done] {time.time() - t_wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
