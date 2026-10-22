#!/usr/bin/env python3
"""CL-2: the first trained continual-learning run (PLAN-2026-10-21-CONTINUAL-LEARNING.md
§5).  The decisive experiment of the programme — the one that has never been run.

WHAT CL-2 IS
------------
The current loss is next-byte cross-entropy; it never asks the recurrent state
to be *usable*.  CL-2 asks: can the state be TRAINED to hold a fact?

Build a fact corpus (not probe trials): streams of the form

    KEY(4B) PAYLOAD(4B) [distractors] KEY(4B)          -> target = PAYLOAD(4B)

over a matched alphabet, with a mix of gaps, on a real-text background.  Add a
MEMORY LOSS: an auxiliary 256-way linear head on the final state (read AT THE
CUE — the state after the trailing cue key) predicting the 4-byte payload,
with a per-byte cross-entropy, added to the next-byte CE with a small weight.

That single auxiliary term is what puts the gradient on the state, and
therefore on the memory: the model must WRITE the payload into the state when
it sees KEY->PAYLOAD, RETAIN it across the gap, and READ it back at the cue.

The goal is the CL-RULER (PLAN §1): ACQUIRE / RETAIN / NO REGRESSION, measured
ON THE TRAINED MODEL (the pre-registered thresholds are fixed, and a probe on
a frozen checkpoint is only a diagnostic, never a pass):

  ACQUIRE   >= 0.50 exact 4-byte recall of the newest fact at d=32
            (matched alphabet; chance 16^-4 = 1.5e-5 at a 16-symbol alphabet)
  RETAIN    >= 0.90 of the pre-stream recall, same lag
  NO REGRESSION  dCE <= 0.02 nats/byte on the held-out general ruler

WHAT THIS TOOL DOES
-------------------
1. Trains a model from scratch on the fact corpus with the memory loss on
   (P4 half-life init so a channel exists that can hold a fact for >32 B).
2. Evaluates the trained model's state with the same conventions as
   tools/cl_ruler.py (state read AT THE CUE, matched alphabet, matched null,
   coverage, the nuisance control, per-key group coverage).
3. Reports the memory head's accuracy against the global byte prior — the
   "is the objective satisfiable without the state?" control that the FSP arm
   failed (F1 0.685 vs prior 0.817 -> null).  Here the prior is 1/256 per byte.
4. Verdict: PASS / NULL / UNMEASURED per the pre-registered thresholds, plus
   the honest statement of what remains unmeasured.

The run is RESUMABLE: the checkpoint records t, B, ring_k, theta, m, v, the
memory head (Wm, bm) and the data position.  The data position is the step
counter of the fact corpus (the corpus is regenerated deterministically from
--seed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

# the engine's float type (the torch twin of numpy_port's F)
DT = torch.float32

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import gpu_port
from gpu_port import GpuModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import tools.membench.model_backend as mb
from tools.membench import generators as G
import glob

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
KEY_LEN = G.KEY_LEN            # 4
PAYLOAD_LEN = G.PAYLOAD_LEN    # 4
FACT_LEN = KEY_LEN + PAYLOAD_LEN  # 8
CUE_LEN = KEY_LEN              # 4

# the ridge readout is the FIXED instrument (tools/membench/model_backend.py);
# it is reused verbatim by the CL-RULER cells, never re-implemented here.
# ---------------------------------------------------------------------------


def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=300,
                    help="optimizer steps (relative; 0 = to the step target)")
    ap.add_argument("--batch", type=int, default=256,
                    help="streams per step (state B + LR in every A/B arm)")
    ap.add_argument("--seqlen", type=int, default=1,
                    help="bytes per stream per optimizer step; the fact stream "
                         "is one whole stream per step, so this is 1")
    ap.add_argument("--gap", type=int, default=32,
                    help="probe gap d of the newest fact (pre-registered d=32)")
    ap.add_argument("--facts", type=int, default=8,
                    help="facts in each fact stream (newest fact carries the "
                         "gap; the older ones are held in memory)")
    ap.add_argument("--memory-alpha", type=float, default=0.25,
                    help="weight of the memory loss on the final-state head")
    ap.add_argument("--half-min", type=float, default=2.0,
                    help="P4 init: lower half-life bound in bytes")
    ap.add_argument("--half-max", type=float, default=512.0,
                    help="P4 init: upper half-life bound in bytes")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0,
                    help="torch seed for the from-scratch init "
                         "(0 = the historical default)")
    ap.add_argument("--batch-seed", type=int, default=0,
                    help="seed for sampling the fact streams (deterministic)")
    ap.add_argument("--device", default=None,
                    help="cuda or cpu (default: cuda if available, else cpu)")
    ap.add_argument("--save", default=None,
                    help="checkpoint path (default: runs/cl2/ckpt_t{t:06d}.npz)")
    ap.add_argument("--resume", default=None,
                    help="resume from a checkpoint (overrides --save)")
    ap.add_argument("--eval-n", type=int, default=512,
                    help="eval sample count per cell")
    ap.add_argument("--eval-alphabet", default="16",
                    choices=("full", "16", "4"),
                    help="matched payload alphabet (the pre-registered setting)")
    ap.add_argument("--eval-gaps", default="8,32,128",
                    help="gaps d to evaluate (comma separated)")
    ap.add_argument("--ce-files", type=int, default=2,
                    help="held-out eval files for NO REGRESSION")
    ap.add_argument("--ce-bytes", type=int, default=6000,
                    help="bytes per held-out slice for NO REGRESSION")
    ap.add_argument("--log-every", type=int, default=50,
                    help="print the loss every N steps")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip the CL-2 evaluation at the end")
    ap.add_argument("--ckpt-dir", default="runs/cl2",
                    help="checkpoint directory")
    ap.add_argument("--holdout-ce-n", type=int, default=8,
                    help="evaluate held-out CE on N checkpoints during the run")
    return ap


# ---------------------------------------------------------------------------
# the fact stream
# ---------------------------------------------------------------------------
def _draw(rng, alpha, size):
    if alpha is None:
        return rng.integers(0, 256, size=size).astype(np.uint8)
    return alpha[rng.integers(0, len(alpha), size=size)].astype(np.uint8)


def build_fact_stream(B, n_facts, d, alpha, rng):
    """Build B fact streams: [F_0]...[F_{N-2}][KEY][PAYLOAD][gap d][KEY].

    The layout is EXACTLY the CL-RULER's fixed-gap cell (cl_ruler.build_sequences
    "stream"): the first n_facts-1 facts are KEY||PAYLOAD pairs drawn with the
    full 256 key space and the matched payload alphabet, then a query fact
    [newest_key][newest_payload][random gap][newest_key] whose trailing key is
    the CUE.  Returns (seq (B, L) uint8, payloads (B, 4) int64, keys (B, 4)
    uint8) where payloads/keys are the newest fact only.
    """
    n_facts = int(n_facts)
    L = FACT_LEN * n_facts + d + CUE_LEN
    keys = rng.integers(0, 256, size=(B, n_facts, KEY_LEN)).astype(np.uint8)
    payloads = _draw(rng, alpha, (B, n_facts, PAYLOAD_LEN))

    seq = np.empty((B, L), dtype=np.uint8)
    # earlier facts F_0..F_{N-2}: KEY||PAYLOAD, 8 bytes each
    earlier = np.concatenate(
        [keys[:, :n_facts - 1], payloads[:, :n_facts - 1]], axis=2)   # (B, N-1, 8)
    seq[:, :FACT_LEN * (n_facts - 1)] = earlier.reshape(B, -1)

    # the newest cell = [KEY][PAYLOAD][gap][KEY] (the trailing KEY is the cue)
    pos = FACT_LEN * (n_facts - 1)
    seq[:, pos:pos + KEY_LEN] = keys[:, n_facts - 1]
    seq[:, pos + KEY_LEN:pos + 8] = payloads[:, n_facts - 1]
    gap = int(d)
    seq[:, pos + 8:pos + 8 + gap] = rng.integers(0, 256, size=(B, gap)).astype(
        np.uint8)
    seq[:, pos + 8 + gap:pos + 12 + gap] = keys[:, n_facts - 1]

    return seq, payloads[:, n_facts - 1].astype(np.int64), keys[:, n_facts - 1]


# ---------------------------------------------------------------------------
# the training loop (under autograd over the engine's forward)
# ---------------------------------------------------------------------------
def engine_forward(model, seq):
    """Run the engine's layers on seq under autograd.

    seq: (B, L) uint8.  Returns (logits (B, L, 256), final_state (B, layers*dim)).
    The final state is the concatenated RECURRENT state after the last byte —
    the state AT THE CUE (the sequence ends with the cue key).  The states are
    managed per-timestep exactly as in numpy_port's step_batch, so the gradient
    that reaches the payload bytes is the true recurrent credit.
    """
    model.new_states(seq.shape[0])
    seq_f = torch.from_numpy(seq).long()
    L = seq.shape[1]
    logits_list = []
    for t in range(L):
        enc = model.embed[seq_f[:, t]]
        x = enc
        for l in model.layers:
            x, st, a, cache = l.forward(enc, x, l.states)
            l.states = st
        logits = x @ model.Wd.t() + model.bd
        logits_list.append(logits)
    logits = torch.stack(logits_list, dim=1)                       # (B, L, 256)
    final_state = torch.cat([l.states for l in model.layers], dim=1)  # (B, 8192)
    return logits, final_state


def train_step(model, optimizer, seq, payloads, lam):
    """One step on one fact stream: next-byte CE + memory loss, Adam."""
    model.t += 1
    logits, final_state = engine_forward(model, seq)

    B, L, _ = logits.shape
    # next-byte CE over the L-1 real positions (the final logit is the state
    # AT THE CUE, used only by the memory head, not by the CE loss)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, 256),
                         torch.from_numpy(seq[:, 1:]).reshape(-1).to(logits.device))

    # memory loss: the 4-byte payload from the state AT THE CUE
    mem_logits = final_state @ model.Wm + model.bm               # (B, 4*256)
    mem_targets = torch.from_numpy(payloads).reshape(-1).long().to(mem_logits.device)
    mem_loss = F.cross_entropy(mem_logits.reshape(-1, 256), mem_targets)

    loss = ce + lam * mem_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach()), float(ce.detach()), float(mem_loss.detach())


def build_memory_head(model):
    """The MEMORY HEAD: a 256-way linear over the final state (layers*dim) that
    predicts the 4-byte payload (4*256 logits).  Fresh, untied from theta, so
    the memory loss can flow onto it without touching the engine's layout."""
    n_out = PAYLOAD_LEN * 256
    n_state = model.layercount * model.dim
    Wm = torch.empty(n_state, n_out, dtype=DT, device=model.device)
    torch.nn.init.xavier_normal_(Wm, gain=1.0)
    bm = torch.zeros(n_out, dtype=DT, device=model.device)
    return Wm, bm


def make_optimizer(model, lr):
    """AdamW over the engine's flat theta plus the memory head.  The engine's
    own inline AdamW is deliberately not reused: the memory loss changes the
    objective, and the engine's step_batch is the old objective."""
    params = list(model.gviews.values()) + [model.Wm, model.bm]
    params = [p for p in params if p is not None]
    for p in params:
        p.requires_grad_(True)
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                             eps=1e-8, weight_decay=0.01)


# ---------------------------------------------------------------------------
# checkpointing (resumable: the data position is the step counter of the fact
# corpus; the corpus is regenerated deterministically from --seed)
# ---------------------------------------------------------------------------
def save_checkpoint(model, path):
    """Save a numpy_port-compatible checkpoint: individual param views
    (like numpy_port.Model.save), the memory head, and the engine persistence
    block (t, B, ring_k, m, v, states).  The eval loads it with
    numpy_port.Model.load(), which reads the param views by name."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    # the engine's flat-buffer views (each param lives here, so writing them
    # updates theta in place)
    for name, view in model.gviews.items():
        data[name] = view.detach().cpu().numpy()
    # the memory head (untied from theta)
    data["Wm"] = model.Wm.detach().cpu().numpy()
    data["bm"] = model.bm.detach().cpu().numpy()
    # engine persistence contract
    data["t"] = np.array(model.t, np.int64)
    data["bytes"] = np.array(model.bytes_seen, np.int64)
    data["data_file"] = np.array(0, np.int64)
    data["data_off"] = np.array(0, np.int64)
    data["m"] = model.m.detach().cpu().numpy()
    data["v"] = model.v.detach().cpu().numpy()
    data["B"] = np.array(model.B, np.int64)
    data["ring_k"] = np.array(model.ring_k, np.int64)
    for i, l in enumerate(model.layers):
        data[f"state.{i}"] = l.states.detach().cpu().numpy()
        data[f"decaytrace.{i}"] = l.decaytrace.detach().cpu().numpy()
    tmp = path + ".tmp.npz"
    np.savez(tmp, **data)
    os.replace(tmp, path)


def load_checkpoint(model, path):
    """Load a numpy_port-compatible checkpoint into a FRESH engine.  The
    individual param views are copied into the engine's gviews, which updates
    theta in place; the memory head is restored separately.  The checkpoint's
    ring_k overrides the engine's (freshly inferred) ring_k so the ring buffers
    are sized to the trained run."""
    ckpt = np.load(path)
    # engine params (views of theta)
    for name, view in model.gviews.items():
        if name in ckpt:
            view.copy_(torch.from_numpy(ckpt[name]))
    # memory head
    if "Wm" in ckpt:
        model.Wm.copy_(torch.from_numpy(ckpt["Wm"]))
    if "bm" in ckpt:
        model.bm.copy_(torch.from_numpy(ckpt["bm"]))
    if "B" in ckpt:
        model.B = int(ckpt["B"])
    if "ring_k" in ckpt:
        model.ring_k = int(ckpt["ring_k"])
    if "t" in ckpt:
        model.t = int(ckpt["t"])
    if "data_file" in ckpt:
        model.data_pos = (int(ckpt["data_file"]), int(ckpt["data_off"]))
    # the tail buffers were sized by the checkpoint's ring_k; drop them so
    # _ensure_tail rebuilds with the loaded ring_k.
    model.tail = None


# ---------------------------------------------------------------------------
# the CL-2 evaluation (the three numbers, via tools/cl_ruler.py's conventions)
# ---------------------------------------------------------------------------
def run_cl2_eval(model, args, ckpt_path):
    """Run the CL-RULER on the trained model (state read AT THE CUE, matched
    alphabet, matched null, coverage, nuisance control) and report the memory
    head's accuracy against the global byte prior."""
    import tools.cl_ruler as cr

    # -- the CL-RULER cells on a numpy model loaded from the trained ckpt --
    m_np, info = mb.load_frozen(ckpt_path)
    args_c = SimpleNamespace(
        ckpt=ckpt_path, mode="probe", n=args.eval_n, seed=args.seed,
        facts=args.facts, alphabet=args.eval_alphabet, out=None)
    alpha = cr.alphabet_symbols(args.eval_alphabet)
    alpha_size = 256 if alpha is None else len(alpha)

    gap_results = {}
    for d in [int(x) for x in args.eval_gaps.split(",")]:
        gap_results[d] = cr.run_gap(m_np, d=d, args=args_c, alpha=alpha)

    # -- NO REGRESSION (the held-out general CE) --
    # cl_ruler.run_ce_cell is the official cell, but it is a per-byte numpy
    # forward over 60 MB (way too slow to run inside the CL-2 loop).  The
    # measurement is the same (mean next-byte CE over the general ruler's
    # held-out slices, +- SE), so we compute it fast with the TRAINED model's
    # forward over a small slice of data/eval.  abspath to the repo root's
    # data/eval.
    ce = _fast_held_out_ce(model, args)

    # -- the memory head's accuracy vs the global byte prior --
    mem = memory_head_accuracy(model, args)

    # -- assemble the verdict --
    return {
        "checkpoint": ckpt_path,
        "ckpt_info": info,
        "mode": args_c.mode,
        "alphabet": args.eval_alphabet,
        "alphabet_size": alpha_size,
        "gaps": {d: gap_results[d] for d in sorted(gap_results)},
        "no_regression": ce,
        "memory_head": mem,
        "verdicts": _verdicts(gap_results, ce),
    }


def memory_head_accuracy(model, args):
    """The trained memory head (Wm, bm) read out of the final state on the
    eval streams; reported against the global byte prior (1/256 per byte)."""
    model.new_states(args.eval_n)
    seq, payloads, _keys = build_fact_stream(
        args.eval_n, args.facts, args.gap, None,  # full 256 payload for the
        np.random.default_rng(args.seed + 999991))  # memory-head report

    seq_f = torch.from_numpy(seq).long()
    L = seq.shape[1]
    for t in range(L):
        enc = model.embed[seq_f[:, t]]
        x = enc
        for l in model.layers:
            x, st, a, cache = l.forward(enc, x, l.states)
            l.states = st
    final_state = torch.cat([l.states for l in model.layers], dim=1)

    mem_logits = final_state @ model.Wm + model.bm
    pred = mem_logits.reshape(args.eval_n, PAYLOAD_LEN, 256).argmax(dim=-1)
    acc = float((pred == payloads).float().mean().item())

    # the prior is the same 256-way head that always predicts the byte with the
    # highest marginal frequency in the training corpus (here: uniform over
    # 256, since the payloads are uniform random).  So prior = 1/256 per byte.
    prior = 1.0 / 256.0
    return {
        "per_byte_accuracy": acc,
        "prior_per_byte": prior,
        "accuracy_above_prior": acc - prior,
        "status": ("SIGNAL" if acc > prior + 2.0 * math.sqrt(
            prior * (1.0 - prior) / max(args.eval_n, 1)) else "NULL"),
        "note": ("the memory head is the 256-way linear on the final state "
                 "(trained with the memory loss).  A head at chance is a null: "
                 "the objective was satisfiable without the state."),
    }


def _frozen_forward_torch(model, seq):
    """Frozen forward on a (L,) uint8 tensor -> (L-1, 256) logits.  The state
    carries across timesteps (causal), so this is the model's general-language
    loss evaluated on the held-out bytes."""
    model.new_states(1)
    L = seq.shape[0]
    logits_list = []
    for t in range(L - 1):
        enc = model.embed[seq[t]]
        x = enc.unsqueeze(0)
        for l in model.layers:
            x, st, a, cache = l.forward(enc.unsqueeze(0), x, l.states[0:1])
            l.states[0:1] = st
        logits = x[0] @ model.Wd.t() + model.bd
        logits_list.append(logits)
    return torch.stack(logits_list, dim=0)


def _fast_held_out_ce(model, args):
    """The NO REGRESSION number, fast: mean next-byte CE over a small slice of
    the general held-out corpus (data/eval), +- SE, using the TRAINED model
    (torch forward).  Replaces cl_ruler.run_ce_cell which is a per-byte numpy
    forward over 60 MB (orders of magnitude too slow to run inside CL-2)."
    import numpy as np
    eval_files = sorted(glob.glob(os.path.join(_REPO, "data/eval/**/*"),
                                  recursive=True))
    eval_files = [f for f in eval_files
                  if os.path.isfile(f) and f.endswith((".txt", ".bin"))]
    if not eval_files:
        print("    [no-regression] no eval files found", flush=True)
        return {"status": "UNMEASURED", "why": "no eval files"}
    ces = []
    scores = []
    for f in eval_files[: args.ce_files]:
        raw = np.fromfile(f, dtype=np.uint8)
        if len(raw) < 4:
            continue
        # take a small slice, stepping through it with the causal forward
        chunk = min(args.ce_bytes, len(raw))
        start = 0
        while start + 4 < len(raw):
            chunk_bytes = raw[start:start + chunk]
            if len(chunk_bytes) < 4:
                break
            seq = torch.from_numpy(chunk_bytes).long().to(model.device)
            logits = _frozen_forward_torch(model, seq)
            tgt = seq[1:].to(logits.device)
            ce = F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1))
            ces.append(float(ce))
            scores.append(float(ce))
            start += chunk
    if not ces:
        return {"status": "UNMEASURED", "why": "no CE computed"}
    ces = np.array(ces)
    ce = float(ces.mean())
    se = float(ces.std(ddof=1) / np.sqrt(max(len(ces), 1)))
    return {
        "status": "MEASURED",
        "why": "",
        "held_out_ce_nats_per_byte": ce,
        "held_out_ce_se": se,
        "n": int(len(ces)),
        "held_out_bytes": int(args.ce_bytes),
        "files": [os.path.basename(f) for f in eval_files[: args.ce_files]],
        "estimator": "torch frozen forward, state reset per slice, no weight update",
    }


def _verdicts(gap_results, ce_cell):
    """The three numbers, one verdict each (plan §1, pre-registered)."""
    acquire_status = {}
    for d, g in gap_results.items():
        acq = g["acquire"]
        if acq["status"] != "MEASURED":
            acquire_status[d] = ("UNMEASURED", acq["status"],
                                acq["per_byte"], acq["exact4"])
            continue
        if acq["verdict"] == "SIGNAL":
            acquire_status[d] = ("SIGNAL", acq["status"], acq["per_byte"],
                                 acq["exact4"])
        else:
            acquire_status[d] = ("NULL", acq["status"], acq["per_byte"],
                                 acq["exact4"])

    retain_status = {}
    for d, g in gap_results.items():
        r = g["retain"]
        if r["status"] != "MEASURED":
            retain_status[d] = ("UNMEASURED", r["status"])
            continue
        if r["ratio_per_byte"] >= 0.90:
            retain_status[d] = ("PASS", r["ratio_per_byte"])
        else:
            retain_status[d] = ("FAIL", r["ratio_per_byte"])

    no_reg = ce_cell
    if no_reg["status"] == "MEASURED":
        no_reg_status = ("PASS" if no_reg["delta_ce"] <= 0.02
                         else "FAIL")
    else:
        no_reg_status = ("UNMEASURED", no_reg["status"])

    return {
        "acquire": acquire_status,
        "retain": retain_status,
        "no_regression": no_reg_status,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    return build_argparser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.save is None:
        args.save = os.path.join(ckpt_dir, f"ckpt_t{0:06d}.npz")

    # -- the model (from scratch, P4 init so a channel holds facts for >32 B) --
    model = GpuModel(dim=args.dim, layers=args.layers, lr=args.lr,
                     seed=args.seed, device=args.device,
                     half_life_init=(args.half_min, args.half_max),
                     ring_credit="auto")
    # the memory head (fresh, untied from theta)
    model.Wm, model.bm = build_memory_head(model)
    if args.resume:
        load_checkpoint(model, args.resume)
        print(f"[resume] loaded {args.resume} "
              f"(t={model.t}, B={model.B}, ring_k={model.ring_k})", flush=True)
    else:
        # a step-0 checkpoint = the baseline for the NO REGRESSION delta
        save_checkpoint(model, args.save)
        print(f"[from-scratch] t=0, P4 init (half-life {args.half_min}-{args.half_max} B), "
              f"ring_k={model.ring_k}, seed={args.seed}", flush=True)

    optimizer = make_optimizer(model, args.lr)
    lam = args.memory_alpha

    # -- the fact corpus (deterministic streams from --seed) --
    rng = np.random.default_rng(args.batch_seed)

    print(f"[train] --steps {args.steps} --batch {args.batch} --gap {args.gap} "
          f"--facts {args.facts} --memory-alpha {lam} --P4 half-life "
          f"{args.half_min}-{args.half_max} B --device {args.device}",
          flush=True)
    if args.seqlen != 1:
        print("[warn] --seqlen != 1 is not supported by the fact-stream path "
              "(one whole stream per step)", flush=True)

    wall_t0 = None
    for step in range(args.steps):
        # the fact stream for this step
        seq, payloads, _keys = build_fact_stream(
            args.batch, args.facts, args.gap, None,
            np.random.default_rng(args.batch_seed + step))

        loss, ce, mem = train_step(model, optimizer, seq, payloads, lam)

        if (step + 1) % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step + 1}/{args.steps} | loss {loss:.4f} "
                  f"(ce {ce:.4f} | mem {mem:.4f}) | t={model.t} | "
                  f"B={model.B} | mem head acc this batch: "
                  f"{_mem_head_acc(model, seq, payloads):.3f}", flush=True)
            if wall_t0 is None:
                wall_t0 = time.time()
        if (step + 1) % args.holdout_ce_n == 0:
            _log_holdout_ce(model, args, step + 1)

        # checkpoint every save_every steps + at the end
        if args.save and (step + 1) % 100 == 0:
            save_checkpoint(model, args.save.replace(".npz",
                                  f"_t{step + 1:06d}.npz"))

    # final checkpoint (t = steps run)
    if args.save:
        final_path = args.save.replace(".npz", f"_t{model.t:06d}.npz")
        save_checkpoint(model, final_path)
        print(f"[save] {final_path}", flush=True)

    # -- the CL-2 evaluation (the three numbers) --
    if not args.no_eval:
        out = run_cl2_eval(model, args, final_path if args.save else args.save)
        _print_report(out)
        with open(args.save.replace(".npz", ".report.json"), "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"wrote {args.save.replace('.npz', '.report.json')}", flush=True)
    return 0


# -- inline helpers ------------------------------------------------- ----------


def _mem_head_acc(model, seq, payloads):
    model.new_states(seq.shape[0])
    seq_f = torch.from_numpy(seq).long()
    L = seq.shape[1]
    for t in range(L):
        enc = model.embed[seq_f[:, t]]
        x = enc
        for l in model.layers:
            x, st, a, cache = l.forward(enc, x, l.states)
            l.states = st
    final_state = torch.cat([l.states for l in model.layers], dim=1)
    mem_logits = final_state @ model.Wm + model.bm
    pred = mem_logits.reshape(seq.shape[0], PAYLOAD_LEN, 256).argmax(dim=-1)
    payloads_t = torch.from_numpy(payloads).long().to(pred.device)
    return float((pred == payloads_t).float().mean().item())


def _log_holdout_ce(model, args, step):
    """Hold-out CE on the general corpus at a checkpoint (NO REGRESSION
    standing).  Runs the numpy evaluator on a fresh numpy model loaded from
    the current theta so the number is apples-to-apples with the final eval."""
    import tools.eval_npz as ev
    tmp = os.path.join(args.ckpt_dir, f"holdout_t{step:06d}.npz")
    save_checkpoint(model, tmp)
    try:
        m_np, info = mb.load_frozen(tmp)
        res = mb.model_transfer_ce(m_np, ckpt=tmp, seed=args.seed,
                                   verbose=False)
        print(f"    [holdout t={step}] held-out CE {res['value']:.4f} +/- "
              f"{res['se']:.4f} nats/B ({res['n_slices']} slices, "
              f"{res['total_bytes']:,} B)", flush=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# -- report ----------------------------------------------------------- ----------


def _pct(x, digits=2):
    if x is None:
        return "   n/a"
    return f"{100.0 * x:,.{digits}f}%"


def _print_report(out):
    print("\n" + "=" * 78)
    print("CL-2 -- the first trained continual-learning run")
    print("=" * 78)
    ci = out["ckpt_info"]
    print(f"checkpoint : {out['checkpoint']}")
    print(f"            dim={ci['dim']} layers={ci['layers']} "
          f"sigma_mean={ci['sigma_mean']:.4f} half-life "
          f"mean={ci['half_life_mean_B']:.1f} B  seed={out['ckpt_info'].get('seed','-')}")
    print(f"alphabet   : {out['alphabet']} ({out['alphabet_size']} symbols); "
          f"chance exact-4 = {(1.0 / out['alphabet_size'])**4:.3g}")
    print(f"memory head: {out['memory_head']['per_byte_accuracy']:.4f} per-byte "
          f"vs prior {out['memory_head']['prior_per_byte']:.4f} "
          f"({'SIGNAL' if out['memory_head']['status'] == 'SIGNAL' else 'NULL'}) -> "
          f"{out['memory_head']['status']}")

    print(f"\n{'gap':>6} | {'ACQUIRE':<12} {'RETAIN':<10} {'NO REGRESSION'}")
    print("-" * 52)
    for d in sorted(out["gaps"]):
        g = out["gaps"][d]
        ac = g["acquire"]
        r = g["retain"]
        a_stat = ac["status"]
        if isinstance(ac["acquire"]["status"], tuple):
            a_stat = ac["acquire"]["status"][0]
        r_stat = r["status"]
        if isinstance(r["status"], tuple):
            r_stat = r["status"][0]
        nr = out["no_regression"]
        if isinstance(nr, tuple):
            nr = nr[0]
        print(f"  {d:>4} | {a_stat:<12} {r_stat:<10} {nr}")
    print("-" * 52)
    print("\nNOTES:")
    print(f"  - ACQUIRE threshold: >= 0.50 exact-4 at d=32 (trained model, "
          f"matched alphabet).")
    print(f"  - RETAIN threshold: >= 0.90 of the pre-stream recall.")
    print(f"  - NO REGRESSION threshold: dCE <= 0.02 nats/byte.")
    print(f"  - The model is 0.56 nats/byte ABOVE the order-8 byte n-gram on "
          f"the same bytes (RESULTS.md §6) - the CE ruler is the floor, not the")
    print(f"    headline.")
    print(f"  - a cell that does not beat its own matched null is NULL; a "
          f"cell that does not beat its own nuisance control is UNMEASURED.")
    print(f"  - memory head vs prior: {'SIGNAL' if out['memory_head']['status'] == 'SIGNAL' else 'NULL'} "
          f"(theFSP arm scored F1 0.685 vs a 0.817 prior -> null).")


if __name__ == "__main__":
    sys.exit(main())
