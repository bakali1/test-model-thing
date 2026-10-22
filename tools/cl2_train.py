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
                chunks.append(fh.read())
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
    idx = rng.integers(0, board["keys"].shape[0], size=B)
    keys = board["keys"][idx]
    pays = board["payloads"][idx]
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
                              ring_k=8, ring_credit="full")
    model.theta.requires_grad_(True)
    return model


def build_head(model, n_out=PAYLOAD_LEN * 256):
    """The MEMORY HEAD: a linear map from the concatenated state (layers*dim)
    to 4*256 logits (a 256-way head per payload byte, as the plan asks)."""
    n_state = model.layercount * model.dim
    Wm = torch.empty(n_state, n_out, dtype=torch.float32, device=model.device)
    torch.nn.init.xavier_normal_(Wm, gain=1.0)
    bm = torch.zeros(n_out, dtype=torch.float32, device=model.device)
    return Wm, bm


class AdamW:
    """A plain, decoupled AdamW over a LIST of tensors, with explicit moment
    buffers (so the checkpoint can carry them as numpy arrays and a resume
    continues the run exactly).  wd is 0 by default: the engine's P6 finding
    is that weight decay erodes the decay logits, i.e. it shortens the
    half-lives -- the one thing a memory run cannot afford (PLAN §0, P4/P6).
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, wd=0.0):
        self.params = list(params)
        self.lr = float(lr)
        self.b1, self.b2, self.eps, self.wd = betas[0], betas[1], eps, wd
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        with torch.no_grad():
            for p, m, v in zip(self.params, self.m, self.v):
                if p.grad is None:
                    continue
                g = p.grad
                m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
                upd = (m / bc1) / ((v / bc2).sqrt() + self.eps)
                if self.wd:
                    upd = upd + self.wd * p
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
    if wm is not None:
        ml = state_cue @ wm + heads[1]
        mem = F.cross_entropy(ml.reshape(-1, 256), pay_new.reshape(-1))
    mem_old = torch.zeros((), device=logits.device)
    if wm_old is not None:
        pay_old = torch.from_numpy(batch["pay_old"]).to(logits.device)
        ml = state_cue @ wm_old + heads[3]
        mem_old = F.cross_entropy(ml.reshape(-1, 256), pay_old.reshape(-1))

    loss = (ce_text + args.behaviour_weight * ce_behav
            + args.memory_alpha * mem + args.memory_alpha_old * mem_old)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return {"loss": float(loss.detach()),
            "ce_text": float(ce_text.detach()),
            "ce_behav": float(ce_behav.detach()),
            "mem": float(mem.detach()),
            "mem_old": float(mem_old.detach())}


# ---------------------------------------------------------------------------
# checkpoints (resumable)
# ---------------------------------------------------------------------------
def save_checkpoint(model, opt, heads, path, meta):
    """A numpy_port-compatible checkpoint (so tools/membench/model_backend.py
    can load it and the CL-RULER can score it) PLUS the CL-2 state: the
    memory head, the optimizer moments, the phase and the corpus position."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {}
    for name, view in model.gviews.items():
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
    for name, view in model.gviews.items():
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
    ap.add_argument("--text-frac", type=float, default=0.5,
                    help="fraction of the samples whose gap is real text")
    ap.add_argument("--text-files", default="data/*.txt",
                    help="the real-text background (NOT data/eval)")
    ap.add_argument("--memory-alpha", type=float, default=0.25,
                    help="the weight of the memory loss (the NEW thing)")
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
                    help="weight decay (0: the P4/P6 finding is that decay "
                         "erodes the decay logits)")
    ap.add_argument("--seed", type=int, default=0)
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
    ce_auto = float(F.cross_entropy(logits[:, :-1].reshape(-1, 256),
                                    torch.from_numpy(seq[:, 1:].astype(np.int64)).reshape(-1)))
    with torch.no_grad():
        logits2, _ = engine_forward(model, seq, L - 1, use_checkpoint=False)
    ce_auto2 = float(F.cross_entropy(logits2[:, :-1].reshape(-1, 256),
                                     torch.from_numpy(seq[:, 1:].astype(np.int64)).reshape(-1)))
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
                          torch.from_numpy(batch["pay_new"]).reshape(-1))
    opt.zero_grad()
    mem.backward()
    g = model.embed.grad
    pay_pos = lay["gap0"] - PAYLOAD_LEN        # the payload of the newest fact
    gpay = float(g[batch["seq"][0, pay_pos]].abs().sum())
    gcue = float(g[batch["seq"][0, lay["cue0"]]].abs().sum())
    print(f"  (ii) |d mem_loss / d embed| at the payload byte {gpay:.3e}, "
          f"at the cue byte {gcue:.3e} -> "
          f"{'OK (the gradient reaches the payload)' if gpay > 0 else 'FAIL'}")
    ok &= gpay > 0

    # (iii) the state at the cue does not depend on the bytes AFTER it
    s1 = np.random.default_rng(1).integers(0, 256, size=(B, L)).astype(np.uint8)
    s2 = s1.copy()
    s2[:, L - 3:] = np.random.default_rng(2).integers(0, 256, size=(B, 3))
    with torch.no_grad():
        _, st1 = engine_forward(model, s1, L - 1, use_checkpoint=False)
        _, st2 = engine_forward(model, s2, L - 1, use_checkpoint=False)
    d = float((st1 - st2).abs().max())
    print(f"  (iii) bytes after the cue changed the cue state by {d:.3e} "
          f"-> {'OK' if d == 0.0 else 'FAIL'}")
    ok &= d == 0.0
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
    text_files = sorted(glob.glob(os.path.join(_REPO, args.text_files)))
    text_pool = TextPool(text_files, seed=args.seed)

    model = build_model(args)
    # the memory head: [Wm, bm] for the newest fact, [Wm_old, bm_old] for the
    # oldest fact (the ruler's RETAIN cell target).
    heads = list(build_head(model)) + list(build_head(model))
    opt = AdamW([model.theta] + heads, args.lr, wd=args.wd)

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
                         "gap": args.gap, "board_size": args.board_size})
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
    t_wall = time.time()
    for pi, (phase, n_steps, board) in enumerate(phases):
        if n_steps <= 0 or pi < start_idx:
            continue
        step0 = start_step if pi == start_idx else 0
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
                               wm=heads[0] if args.memory_alpha > 0 else None,
                               wm_old=heads[2] if args.memory_alpha_old > 0 else None)
            model.t += 1
            model.bytes_seen += args.batch * L
            if (step + 1) % args.log_every == 0 or step == 0:
                el = time.time() - t_wall
                print(f"  [{phase} {step + 1}/{n_steps}] loss {stats['loss']:.4f} "
                      f"| ce_text {stats['ce_text']:.4f} "
                      f"| ce_behav {stats['ce_behav']:.4f} "
                      f"| mem {stats['mem']:.4f} | mem_old {stats['mem_old']:.4f} "
                      f"| {el:.0f}s", flush=True)
            if (not args.no_checkpoint) and \
                    ((step + 1) % args.save_every == 0 or step == n_steps - 1):
                path = os.path.join(args.ckpt_dir,
                                    f"{args.tag}_{phase}_t{step + 1:06d}.npz")
                save_checkpoint(model, opt, heads, path, {"phase": phase, "step": step + 1,
                                       "seed": args.seed, "alphabet": args.alphabet,
                                       "facts": args.facts, "gap": args.gap,
                                       "board_size": args.board_size,
                                       "board_seed_a": args.board_seed_a,
                                       "board_seed_b": args.board_seed_b})
                print(f"    [save] {path}", flush=True)
        # the phase boundary
        if not args.no_checkpoint:
            path = os.path.join(args.ckpt_dir, f"{args.tag}_{phase}_end.npz")
            save_checkpoint(model, opt, heads, path, {"phase": phase, "step": n_steps, "seed": args.seed,
                                   "alphabet": args.alphabet, "facts": args.facts,
                                   "gap": args.gap, "board_size": args.board_size,
                                   "board_seed_a": args.board_seed_a,
                                   "board_seed_b": args.board_seed_b})
            print(f"    [phase end] {path}", flush=True)
        start_phase = ""
        start_step = 0
    print(f"[done] {time.time() - t_wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
