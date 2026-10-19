# Documentation index

Start with **`../README.md`** (what this is, how to run it) and
**`../RESULTS.md`** (every measured number, including the ruler and today's
from-scratch experiment).

## Where things are

| doc | what it is | status |
|---|---|---|
| `../README.md` | project entry point, how to run | **current** |
| `../RESULTS.md` | the ruler and all measured results | **current — append-only** |
| `plans/PLAN-2026-09-23-GPU.md` | the GPU port plan: engine, hub, correctness gate, week-1 order | **current** |
| ~~`plans/PLAN-2026-09-25.md`~~ | next-day execution plan | **archived** → `plans/superseded/` |
| ~~`plans/PLAN-2026-09-22.md`~~ | the original phase plan | **archived** → `plans/superseded/` |
| `plans/PLAN-2026-10-04-LR-ROLES.md` | role-specific-LR factorial (the cheap half of the SNR blocker's two-axis plan); gate + Colab bundle built, run not launched | **current** |
| `plans/PLAN-2026-10-03-DISTILL.md` | distil an open model into TMT (teacher `LFM2.5-230M-Base`, Marginalize-It, teacher-generated corpus); gate 0 cleared | **current — read before optimizing** |
| `plans/PLAN-2026-10-04-ROADMAP.md` | **roadmap v3** — ruler → step size → memory → SSM. v3 reorders v2 after the SNR blocker was found to be ~12.6× pessimistic; v2's evidence chains and rejected-ideas table are retained below the v3 header | **current — read before new experiments** |
| `plans/PLAN-2026-10-11-T-AXIS.md` | **the T axis** — `seqlen` as a knob with `T=1` as the floor; then gated recurrence → MTP. Synthesizes this repo's `RESULTS.md` with upstream issues `#17/#20/#23/#25/#28/#30/#34` and the CUDA fork's converged config. **Executed through P0 + P6 (2026-10-12): CE-only, the repaired optimization gate, the NaN guard + grad clip, per-group weight decay, `seqlen`, the dense trace, the log-spaced half-life init, and a 0-training P4 capacity gate. Four corrections applied — see §7.** | **current — supersedes the *ordering* of the 10-04 roadmap and the 10-06 plan; read before new experiments** |
| `ideas/ROUTER-4.5M-INTENT-JSON.md` | 4.5M intent→JSON tool router (SFT, short-horizon, teacher-as-labeler) | **design — nothing measured.** moved out of `plans/`: it was never a plan |
| `optimization/OPTIMIZATION_TEST_STATUS.md` | what was measured, what was **not**, what to try next | **current — read before optimizing** |
| `optimization/GPU_OPTIMIZATION_PLAN.md` | the optimization plan and its measured outcome (§0) | historical + measured |
| `optimization/GPU_OPTIMIZATION_PLAN-REVIEW.md` | review of that plan | review, closed |
| `reviews/PLAN-2026-09-23-GPU-IMPLEMENTATION-REVIEW.md` | implementation review + responses | review, closed |
| `runbooks/MORNING-RUNBOOK.md` | how to run the GPU legs | **current** |
| `runbooks/SCHEDULE.md` | quotas and weekly schedule | **current** |
| `ideas/ARCHITECTURE_IDEAS.md`, `ideas/ARCHITECTURE_IDEAS2.md` | speculative directions | ideas, not plans |
| `ideas/papers/README.md`, `ideas/papers/*` | external papers with the notes taken from them, source text beside the notes | ideas, not plans |

## Reading order

1. **`../README.md`** — orientation.
2. **`../RESULTS.md`** — what is actually true, with numbers.
3. **`optimization/OPTIMIZATION_TEST_STATUS.md`** — what has *not* been tested.
   Read this before proposing an optimization; it names the candidates that
   were already measured and lost.
4. **`plans/PLAN-2026-09-23-GPU.md` §7** — the live status and next steps.

## Archived

`docs/plans/superseded/` holds plans that were executed or superseded. Nothing is
deleted — `RESULTS.md` is append-only for a reason, and so is this:

- `superseded/PLAN-2026-09-22.md` — the original phase plan (superseded by the GPU plan)
- `superseded/PLAN-2026-09-25.md` — next-day execution plan (executed)

## Conventions that matter

- **`RESULTS.md` is append-only.** Corrections are added as new sections that
  explicitly supersede the earlier claim, never by editing history. There is a
  worked example: the "layered mix is dominated" verdict was written, then
  overturned by its own dilution control, and both are recorded.
- **Every number needs its ruler setting.** Sample size, seed and bar travel
  with the number. A CE without its SE is not a result.
- **`OPTIMIZATION_TEST_STATUS.md` has a "NOT tested" section.** Listing only
  successes is how a plan ends up shipping predictions that measurement
  already refuted.
- **Every gradient-quality number prints its estimator convention.** `SNR` is
  not one quantity: `||EMA_b(g)||² / EMA_b(||g||²)` measured *while θ moves* reads
  0.05, and the same estimator at *frozen* θ reads 0.633. An unlabelled SNR is
  not a result.
- **Gradient gates assert real corpus bytes.** On uniform random bytes the states
  are iid and every lag weighting collapses to one direction, so the gate passes
  confidently for free. Measured: `ρ(1)` flips `+0.534 → −0.171` on that one
  variable. Scripts: `tools/random_vs_real.py`, `tools/reconcile_snr.py`,
  `tools/rho_probe.py`.

## Repo layout

The **engine stays flat at the root** on purpose: 24 files import it and the
Colab bundles embed it by flat module name, so a package layout would buy
tidiness at the cost of a fragile refactor across the whole transport layer.

```
# engine (root) — imported by everything, embedded into the Colab bundles
gpu_port.py  numpy_port.py  tail_credit.py       model + tail credit
train_gpu.py                                       the GPU trainer (CPU or CUDA)
checkpoint_hub.py  fetch_corpus.py  fetch_eval_data.py   transport + in-session data
test_tail.py  tmt.npz                              the rolling working checkpoint

tools/     eval.py  eval_npz.py  bench_variants.py        the ruler + benchmark board
           chat_models.py  fs_verdict.py  memory_report.py  sampling + experiment analysis
           teacher_table.py  teacher_eval.py                 distillation: token->byte table
           decontaminate.py                                  (zero deps) + teacher quality
                                                           + eval-copy filter (train u eval)
bundles/   make_*.py                                       Colab bundle generators
           colab_lr_roles.py                               GENERATED (gitignored): the
                                                           role-specific-LR factorial on a T4
           colab_*.py                                      GENERATED (gitignored)
legacy/    main.py  main_torch.py  benchmark*.py          the original per-byte line
           smaller-4.5m.safetensors  tinystories_bench.txt
tests/     check_*.py                                      free CPU gates; run these first
           check_distill.py                                GATE 0: exact token->byte map +
                                                           target fidelity + decontamination
           check_lr_roles.py                               per-role LR: bit-exact at equal LR,
                                                           and provably confined to the decay role
docs/      INDEX.md  plans/  optimization/  reviews/  runbooks/  ideas/  ideas/papers/
cache/     cached eval transcripts (gitignored)            lets tables re-render free
models/    banked + historical checkpoints (gitignored)
kaggle/    tmt_gpu_train.py                               the Kaggle mirror
data/  corpus/                                              corpora, staged in-session
```

**Every tool runs from any directory.** Each one inserts the repo root on
`sys.path`, so `python tools/eval.py` works from `/tmp`.

### Rebuilding the Colab bundles

```bash
.venv/bin/python bundles/make_fs_bundle.py       # -> bundles/colab_fs_*.py
colab run --gpu T4 --timeout 4200 bundles/colab_fs_w.py
```

The generators read the engine from the repo root and write into `bundles/`.
Each generated script embeds the engine as gzip+base64 and unpacks it on the
VM, because `colab run` uploads only the one script. They are gitignored: they
embed the HF token.

### Where checkpoints live

| path | what it is | why there |
|---|---|---|
| `tmt.npz` (root) | the **rolling working checkpoint** | `checkpoint_hub.py --dest` and `make_leg_bundle.py` default to this exact path, and save/keep-k pruning writes its tags beside it |
| `models/tmt_t094747.npz` (+ `.sha256`) | the banked incumbent (t=94,747) | versioned, so it belongs with the other named models |
| `models/nlt0.1.npz`, `models/nlt-fix-0.1.npz` | earlier prototypes | historical; no current code references them |
| `fs_*.npz` | from-scratch arm checkpoints | **not kept locally.** All verified on the private hub; re-pull with `fs_verdict.py --pull --at-step N`. Cached transcripts in `cache/` let the tables re-render without re-pulling. |
