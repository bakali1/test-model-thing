# PLAN — the role-specific-LR factorial (2026-10-04)

**Date:** 2026-10-04
**Status:** engine support + gate + Colab bundle **built and rehearsed**; the
Colab run has not been launched (blocked on one prerequisite, §6)
**Predecessors:** `PLAN-2026-10-03-DISTILL.md`, `PLAN-2026-09-23-GPU.md`,
`../optimization/OPTIMIZATION_TEST_STATUS.md`, `../../RESULTS.md`
(`RESULTS.md` §"Gradient SNR was ~0.05 for the entire run").

---

## 0. What this is, in one paragraph

TMT's standing blocker is **gradient SNR ≈ 0.05**, unchanged across 889,897
steps and low in *both* the shared and decay parameter groups. The already-planned
factorial has two axes: **per-role LR** (shared vs decay) and **optimizer**
(AdamW vs SGD). This plan runs **the per-role-LR axis only**, because it is
roughly an order of magnitude cheaper than the optimizer axis, and it is
**resumable** — `--lr-decay` changes a learning *rate*, not the optimizer
*state format*, so every arm branches off the incumbent with AdamW's m/v
carried over instead of starting from scratch.

## 1. Why this axis first, not the optimizer

| axis | scope | resumable? | cost |
|---|---|---|---|
| **per-role LR** | **8,192 params = 0.183 %** | **yes** — AdamW state unchanged | 3 arms × 1 Colab slot |
| AdamW → SGD | all 4,481,793 params | **no** — stored state changes, so from-scratch arms with their own LR grid | separate plan |

The two axes are also not equally informative. SGD's claim (a commenter argued
"AdamW generalizes less well than SGD … at best break-even on small models") is
a *generalization* claim from mostly small-scale vision work, and RESULTS.md's
own convention refuses to carry that kind of claim without a matched arm. The
per-role-LR axis asks a sharper question about *this* model: **the 8,192 decay
coordinates set memory half-life; are they LR-limited?**

## 2. The premise, measured (not assumed)

Reproduced from the real trained checkpoint (`runs/mx150_nvrtc/tmt_t101847.npz`,
CPU, 30 steps, ruler text). Script: `../../tools/`-external scratch,
`~/programing/opensource/new-llm/scratch-optimizer-study/per_coord_grad.py`.

| measurement | value |
|---|---:|
| gradient SNR, flat over all 4,481,793 coords (**bias-corrected**) | **0.0695** |
| SNR, shared_trunk groups | 0.060 – 0.082 |
| SNR, the 16 decay groups (512 each) | 0.0765 mean |
| per-coordinate gradient spread, max/min over non-zero | 7.5 × 10²¹× |
| **top 1 % of coords carry** | **50.7 % of gradient energy** |
| top 5 % / top 10 % carry | 76.5 % / 87.6 % |
| coords with gradient RMS < 1e-8 (effectively dead) | 2.28 % |
| Adam per-coord update spread across the middle 98 % (p50→p99) | ~4× |

This reproduces the standing blocker (RESULTS: ~0.05 for the whole run) from a
different checkpoint at a different batch size, so ~0.05–0.07 is the regime both
groups sit in.

**Caveat that keeps this honest:** heavy-tailed per-coordinate gradients are
normal and are exactly what Adam is *designed* to reweight. So the table shows
the two optimizers will behave very differently — it does **not** show Adam is
hurting. The per-role axis is testable without that leap, which is why it goes
first.

**A telemetry trap found while measuring this:** `telemetry.update_gradient_ema`
uses β₁=0.9 for `m` and β₂=0.999 for `v`. On any run shorter than ~1000 steps
`v` has only `(1−0.999^t)` of its mass, so the raw ratio `‖m‖²/Σv` is **not**
bounded by 1 and reported **2.15** in a first attempt — impossible by
Cauchy–Schwarz. Real legs (10⁵ steps) never see it. Anyone measuring SNR on a
short arm must bias-correct both EMAs.

## 3. What was built for this (and is gated)

| file | what |
|---|---|
| `gpu_port.py` | `--lr-decay`: per-coordinate LR vector, `lr` everywhere + `lr_decay` on the 16 `.decay` slices. Cached; role's LR scales the whole update bracket (adaptive **and** decoupled wd). |
| `train_gpu.py` | `--lr-decay` on the CLI, threaded into the model. |
| `tests/check_lr_roles.py` | **GATE, free, ~1 min.** |
| `bundles/make_lr_roles_bundle.py` | generates `colab_lr_roles.py` (self-contained T4 factorial). |

The gate asserts three things, the first being the one that matters:

- **L1** `lr_decay == lr` reproduces the historical scalar-LR engine
  **bit-exactly** — theta, both AdamW moments, and every per-step loss, 0.0
  max |Δloss| over 120 steps. The LR is now a vector; a float-reassociation
  here would silently invalidate every arm compared against a stored
  checkpoint, so "close enough" is not the bar.
- **L2** `lr_decay != lr` actually moves parameters (max |Δθ| 2.385e-03).
- **L3** the movement is confined to the decay role — all 8,192 decay coords
  move, **all 4,473,601 trunk coords bit-identical**. Guards the index
  arithmetic in `_role_lr_vec`.

`tests/check_parity.py` (gpu_port vs the numpy reference) also still passes, so
the default single-LR path is untouched. Gate result, 2026-10-04:
**PASS** (L1/L2/L3 + uniform-vector control).

## 4. The experiment

**Arms** — `--lr 5e-4` everywhere; only the decay role's LR varies:

| arm | `--lr-decay` | role |
|---|---|---|
| `lrdecay1x` | unset | **CONTROL** — identical to today's single-LR engine |
| `lrdecay4x` | 2e-3 | 4× |
| `lrdecay16x` | 8e-3 | 16× |

Matched on everything else: same pulled checkpoint bytes (each arm gets its own
copy), same data seed 0, same 30,000 steps, same B=1024, same corpus slice
fetched once in-session.

**Why 16× is the important arm — it is a sensitivity probe.** 16× on 0.183 % of
the parameters is a large perturbation to a small, isolated group. Predictions,
in order of strength:

1. If `sigma_mean` does **not** visibly rise from 1× → 16×, the decay role is
   **not LR-limited**, this whole axis is dead, and the blocker is upstream
   (gradient signal / architecture). That is a real, cheap falsifier.
2. If it does rise: `half_life_mean_bytes` and the `half_life_gt4` count should
   rise with it (σ is the per-byte decay, so more LR on it ⇒ slower decay ⇒
   longer memory), and decay-group SNR should decouple from trunk SNR.
3. `sigma` rising is **not** automatically good. Longer memory is only useful if
   the credit actually uses it; the ruler and the generation probe decide that,
   not σ.

**Where the arms are run.** Colab T4, not the local box: a T4 sustains
~22 kB/s against the MX150's **4.2 kB/s** on a mature `ring_k=18` checkpoint
(measured 2026-10-03), so one slot buys ~5× the steps per arm. 3 arms ×
30,000 steps ≈ 23 min/arm ≈ 70 min of training, ~80 min wall with startup.

## 5. Judging — never CE alone

Pull each arm back and ruler it at the plan's standard setting:

```bash
.venv/bin/python tools/eval.py CKPT --slices 3 --bytes 20000 --xd-bytes 8000 --xd-files 4
.venv/bin/python tools/chat_models.py --prompt all --seeds 0,1,2 --set-glob 'runs/lrroles/*.npz'
```

- **Compare every arm to `lrdecay1x` at matched steps** — *not* to the starting
  checkpoint. Continued training on this corpus raises in-domain CE along this
  family (1.1741 → 1.1902 → 1.6999 → 1.7438), so absolute CE rises for every arm
  including the control; only the **difference vs control** means anything.
- **Bar:** an arm is only interesting if held-out CE is within **0.03** of the
  control **and** no regression in the 6-seed degeneracy metrics **and** no
  `User:`/`Assistant:` leakage.
- A decay-telemetry move with a CE regression is a **partial** result, reported
  as such — not a win.
- Only one arm per multiplier; one seed, one α. A sweep is out of scope here.

## 6. How to run it, and the one prerequisite

```bash
.venv/bin/python bundles/make_lr_roles_bundle.py
colab run --gpu T4 --timeout 7200 bundles/colab_lr_roles.py
```

Overrides: `TMT_MULTIPLIERS`, `TMT_STEPS`, `TMT_HUB_REPO`, `TMT_PUSH=0`.

**PREREQUISITE — a Colab secret named `TMT_TOKEN` must exist** for the private
hub repo (`bakali1/tmt-checkpoints`). No HF token is present in this laptop's
environment, and the bundle never embeds credentials by design. Without the
secret the script exits at the pull step and burns the slot. Either configure
the secret, or point `TMT_HUB_REPO` at a public repo.

**Rehearsed locally on CPU** before any slot was spent: 2 arms × 20 steps from
the real checkpoint, `OVERALL = PASS`, both arms ending at the same `t`, only
`lr_decay` differing. The rehearsal also caught that the first draft passed
`hub_repo=None` to the trainer, so **no checkpoint would have left the VM** —
the exact tag loss of the 2026-09-24 live leg. The bundle now pushes each arm
explicitly (`hub.push`, `run_id=lrroles-<arm>`) and fails the run if a push
fails.

## 7. Budget and quota

Colab free tier is usage-capped and evictible (`runbooks/SCHEDULE.md`); this is
~1.3 h of a tier that has used ~1 h so far. No Kaggle quota — the 30 h/weekly
Kaggle budget is untouched and stays reserved for the distillation plan's
T4 steps. Local ruler cost is 3 × 11–22 min on the laptop CPU.

## 8. NOT in scope

- **AdamW → SGD.** Deliberately deferred: from-scratch only, needs its own LR
  grid, and its motivating claim is a small-scale generalization result. Run it
  *after* this axis, and only to answer "did SNR move".
- **ParaRNN / time-axis parallelization.** Parked, and for a stronger reason
  than "doesn't fix SNR": the ring credit is **45 % of the step** and is already
  bandwidth-bound at the measured 39.1 GiB/s copy ceiling, so parallelizing the
  time axis cannot speed it up; and chunking needs a ring snapshot per step in
  the chunk, i.e. ring memory × T, where the ring is already 576 MiB of a
  926 MiB peak. T is capped at ~2–4 on this hardware. Read the paper, don't
  build it.
- No k-sweep, no α sweep, no extra seeds.

## 9. Results

_(to be appended when the arms land — decay telemetry first, then the ruler, with
each arm's ruler setting stated next to its number)_