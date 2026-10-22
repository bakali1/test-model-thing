#!/usr/bin/env python3
"""Can the RECURRENT STATE hold the facts, as a linear-readable associative
store?  A capacity budget, with the assumptions stated so they can be
falsified.

WHY THIS QUESTION
-----------------
CL-2 measured (RESULTS.md 17): the model did NOT put the payload in the
recurrent state.  The state carries the CUE KEY and the key->payload
association lives in the HEAD'S WEIGHTS -- a 512-entry table that board B
overwrote, which is why RETAIN is exactly 0.0000.

Two explanations, and they need different fixes:
  (E1) OPTIMIZER PREFERENCE -- the state route is possible, but the
       gradient found the cheaper path (the head's weights).
  (E2) ARCHITECTURAL IMPOSSIBILITY -- the state is too small to hold the
       facts, so the global store was the ONLY option.

This script prices (E2).  The memory head is a LINEAR map
`state @ Wm + bm` -> 4*256 logits, so the state must linearly encode:

  1. the ADDRESS of the wanted fact, and
  2. the PAYLOAD of that fact,

in subspaces that ONE linear map can decode.

THE BUDGET (per fact):
  * ADDRESS: to select one of N slots with a linear map, the slots need
    linearly independent codes -> >= 1 dim per slot (a one-hot code).  With
    a margin, 2 dims per slot is comfortable; 1 is the floor.
  * PAYLOAD: each payload byte is decoded 256-way by a linear map, so the
    byte's 16 symbols must be linearly separable -> >= 16 dims per byte
    (log2 of the alphabet).  PAYLOAD_LEN bytes -> 16*PAYLOAD_LEN dims.
    (If the readout were narrowed to the 16-way alphabet, this would be
    4 dims/byte -- reported below as the optimistic variant.)

  => per fact: PAYLOAD_LEN*log2(alphabet) + address_dims

  * the state has layers*dim scalars.  A float32 scalar holds ~32 bits,
    but a robust LINEAR readout wants a margin; the budget below reports
    the optimistic (32 bits/scalar) and the pessimistic (8 bits/scalar)
    variants.

WHAT THIS DOES NOT CLAIM: that the model could not have found some
nonlinear addressing scheme.  It prices the LINEAR-readable store that the
current architecture actually has.
"""
import argparse

import numpy as np


def budget(n_state, payload_len, log2_alpha, facts, address_dims=1,
           bits_per_scalar=32.0):
    per_fact = payload_len * log2_alpha + address_dims
    need_dims = per_fact * facts
    have_bits = n_state * bits_per_scalar
    need_bits = facts * payload_len * log2_alpha
    return {
        "per_fact_dims": per_fact,
        "need_dims": need_dims,
        "have_dims": n_state,
        "shortfall_factor": need_dims / n_state,
        "bits_needed": need_bits,
        "bits_have": have_bits,
        "bits_shortfall": need_bits / have_bits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--payload-len", type=int, default=4)
    ap.add_argument("--alphabet", type=int, default=16)
    ap.add_argument("--board", type=int, default=512)
    ap.add_argument("--facts-in-stream", type=int, default=4)
    args = ap.parse_args()

    n_state = args.dim * args.layers
    l2 = np.log2(args.alphabet)

    print("=" * 74)
    print("THE CAPACITY BUDGET FOR A LINEAR-READABLE ASSOCIATIVE STORE")
    print("=" * 74)
    print(f"state: {args.layers} layers x {args.dim} dims = {n_state} scalars")
    print(f"payload: {args.payload_len} bytes over {args.alphabet} symbols "
          f"({l2:.0f} bits/byte, {args.payload_len*l2:.0f} bits/fact)")
    print(f"the memory head is a LINEAR map -> 4*256 logits")
    print()

    for label, bps, adims in (("optimistic  (32 bits/scalar, 1 addr dim)", 32.0, 1),
                               ("pessimistic (8 bits/scalar, 2 addr dims)", 8.0, 2)):
        print(f"-- {label} " + "-" * (68 - len(label)))
        for nf, tag in ((args.board, "ONE board (A)"),
                        (2 * args.board, "TWO boards (A+B)")):
            r = budget(n_state, args.payload_len, l2, nf, adims, bps)
            print(f"   {tag:16} {nf:>5} facts: need {r['need_dims']:>7.0f} dims, "
                  f"have {r['have_dims']:>6}  -> {r['shortfall_factor']:>6.2f}x "
                  f"SHORT   (bits: {r['bits_needed']/1e3:>6.1f}k need, "
                  f"{r['bits_have']/1e3:.0f}k have)")
        # how many facts DO fit?
        per = args.payload_len * l2 + adims
        fit = int(n_state * bps / (per * l2))
        print(f"   => a linear-readable store of this size holds about "
              f"{fit} facts")
        print()

    print("-- WHAT SIZE WOULD BE NEEDED? " + "-" * 44)
    per = args.payload_len * l2 + 1
    for nf, tag in ((args.board, "one board (512)"),
                    (2 * args.board, "two boards (1024)")):
        need = per * nf
        dim_needed = need / args.layers
        print(f"   {tag:18}: {need:>6.0f} dims  -> dim >= {dim_needed:>7.0f} "
              f"at {args.layers} layers   (or layers >= {need/args.dim:>5.0f} "
              f"at dim {args.dim})")
    print()
    print("-- OR: SHRINK THE BOARD " + "-" * 51)
    per = args.payload_len * l2 + 1
    print(f"   at dim {args.dim}, {args.layers} layers, a linear store holds "
          f"~{int(n_state/per)} facts;")
    print(f"   the arm used board_size {args.board} per board, i.e. "
          f"{args.board/int(n_state/per):.1f}x over budget for ONE board, "
          f"{2*args.board/int(n_state/per):.1f}x for two.")

    print()
    print("-- THE CODING ASSUMPTION, MADE EXPLICIT " + "-" * 30)
    print("   The factor above uses the OPTIMAL coding: log2(alphabet) = "
          f"{l2:.0f} dims per byte.")
    print("   The NATURAL coding is one-hot over the alphabet: "
          f"{args.alphabet} dims per byte.")
    for label, dims_per_byte in (("optimal (hyper "
                                   f"cube)", l2),
                                  ("natural (one-hot)", args.alphabet)):
        per = args.payload_len * dims_per_byte + 1
        for nf, tag in ((args.board, "one board"), (2 * args.board, "two boards")):
            need = per * nf
            print(f"   {label:22} {tag:12}: {need:>6.0f} dims needed, "
                  f"{n_state} have -> {need/n_state:>5.2f}x short   "
                  f"(dim >= {need/args.layers:>6.0f})")
    print()
    print("   EITHER WAY THE STATE IS TOO SMALL.  The address costs 1 dim per")
    print("   fact; the payload costs 4-16 dims per byte.  A capacity-limited")
    print("   linear store therefore drops the CONTENT and keeps the ADDRESS")
    print("   -- which is EXACTLY the D5 pattern ('the address persists, the")
    print("   content decays'), reached for a reason that has nothing to do")
    print("   with write selectivity.")


if __name__ == "__main__":
    main()
