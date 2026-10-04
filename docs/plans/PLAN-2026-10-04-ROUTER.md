# PLAN — the tool router: 4.5M byte-recurrent intent→JSON (2026-10-04)

**Date:** 2026-10-04
**Status:** **DESIGN, not execution — nothing on this page is measured yet.**
Execution path decided: **warm-start from pinned `t889,897`, train on Colab T4,
all validation on laptop CPU** (§4). This
plan exists because the last three measured verdicts point the same way. It does
not claim the direction works; it names the two measurements that would tell us
in < 3 h CPU whether it does.
**Predecessors:** `PLAN-2026-10-04-LR-ROLES.md` (axis LIVE, buys nothing, closed),
`PLAN-2026-10-04-ROADMAP.md` (credit → SSM → memory ruler), `PLAN-2026-10-03-DISTILL.md`
(KD on stories measured ~one-hot), `../../RESULTS.md` (SNR ≈ 0.05, needle < 64 B).

---

## 0. What this is, in one paragraph

TMT is a 4.5 M-parameter byte-level recurrent model whose measured blockers are
all on the *long-horizon* axis: gradient SNR ≈ 0.05 across a 890k-step run, no
retrieval past ~64 bytes, decay-role LR saturated and worthless (the closed
factorial), KD targets on the story register effectively one-hot. The honest
reading of those three verdicts: **stop asking the 4.5 M to do what it has
measuredly failed at, and find the task where the load-bearing recurrence is not
load-bearing.** That task is a short, closed-vocabulary map — natural-language
command → structured JSON — where input and output are both ≤ ~80 bytes, the
credit path is short-horizon, and the output grammar is bracket-structured ASCII
that this architecture demonstrably already emits. This plan converts the router
idea from a vibe into a pilot with two named gates: *does the trunk learn a
short-horizon supervised objective at all*, and *where does the 64-byte memory
wall bite*.

## 1. Why this task, and why now

| reason | evidence, not opinion |
|---|---|
| The blocker is credit/SNR; routing is short-horizon | SNR 0.05 + needle < 64 B were measured on the LM objective; a command → JSON map does not need the recurrence to carry signal across 100+ bytes. **Claim, not fact: the pilot exists to measure it.** |
| The output is bracket-structured ASCII | the model's one demonstrated capability is closing quotes/brackets (`README.md`); a JSON grammar is exactly that, at short length |
| Data cost is already budgeted | 10–50 MB text ≈ the 16 MB arm this repo already costed (≈ 13 h CPU / 1–2 h T4) |
| The teacher is finally in-register | `LFM2.5-230M` (instruction/RL sibling) is advertised for tool use and data extraction; tool-call JSON is the one register where the non-Base checkpoint is the *right* teacher — the swap question from the distillation thread resolves cleanly here |
| It sidesteps the closed axes | no LR/optimizer knob, no decay role, no MI targets — nothing to re-litigate |

**Why not the alternatives (each was considered, each was rejected):**

| alternative | why not |
|---|---|
| general chat | capacity wall; and it is the LR-factorial-shaped dead end with extra steps |
| short technical Q&A | open-ended generation is the broken axis; a ★★★★★ routing task does not need it |
| prose/story distillation | targets measured ~one-hot; teacher-register mismatch |
| code generation | teacher says it is bad at it; output length > 64 B wall |

## 2. What this is NOT (recorded so it does not come back)

- **This is not a distillation problem.** The pipeline below uses the teacher as a
  **labeler and validator**, not as a next-byte distribution. The MI/KD apparatus
  — Marginalize-It dumps, retained-prefix-mass gates, α = 0.5 — **does not apply;
  do not drag it in.** Dragging it in would re-raise the measured ~one-hot target
  and the +1.5 % (free) cost for zero signal.
- **The 8–16k-tokenizer + tied-embeddings advice is refused.** It comes from a
  token-transformer mindset. TMT has a 256-byte vocabulary and no embedding
  matrix eating the parameter budget; the entire 4.5 M is trunk/RTU/heads. The
  advice does not transfer, and this sentence exists so nobody re-proposes it.
- **No teacher-generated free text.** That is the withdrawn corpus plan
  (`PLAN-2026-10-03-DISTILL.md` §4 steps 1–2, 0.11–0.23 diversity gap, measured).

## 3. The task spec (frozen for the pilot)

- **Input:** one natural-language command, UTF-8 bytes, **length cap L = 80 bytes**
  (the needle wall is ~64 B; the cap is a control variable, tested in §4).
- **Output:** a JSON object `{"intent", "tool", "arguments": {...}}` in a closed
  grammar — 12 categories from the Linux/technical domain: process, networking,
  filesystem, systemd, docker, git, ssh, package, disk, hardware/logs,
  permissions, services. Each intent has a canonical `tool` + slot schema.
- **The JSON must be parse-valid.** Validity is a gate, not a hope: a
  "plausible-looking wrong label" is this project's recurring failure mode (the
  gold-prefix bug of the MI dump). Every dataset example is validated by a
  parser, not by eyeball.
- **Paraphrase discipline:** 5–10 paraphrases per intent/template, drawn from a
  held-out paraphrase pool at train/eval split time — otherwise "intent
  accuracy" measures template memorization, not intent separation.

## 4. Pilot A — the two measurements that decide everything (train T4, validate CPU)

**Execution path (decided 2026-10-04): warm-start from the 900k, not from scratch.**
Both arms branch from the **pinned start** `tmt_mixed_long889897_t889897.npz`
(sha256 recorded, data seed 0 — the exact file the LR-ROLES factorial used, so the
start bytes are already matched and proven). Rationale: the 20k-example router
corpus is an SFT-scale nudge, and the trunk already carries English byte/register
statistics from 890k steps; this is also the deployment-shaped answer — "make the
900k model do this" is the actual product question. The cost of warm-starting is
that *two* things move in every arm (trunk continuation + the router objective),
which is why there is a second arm (§4a).

**Compute:** Colab T4, `ring_credit=auto` (compiled), B = 1024, ~30k steps per arm
≈ 23 min at the measured 22–24 kB/s. **All validation runs on the laptop CPU**
(router_eval, distance curve, ruler — 3 × 11–22 min each) and can overlap the
Colab slots.

### 4a. Two matched arms (this is what warm start forces)

| arm | start | objective | answer |
|---|---|---|---|
| A | pinned t=889,897 bytes | CE on command→JSON as-SFT (seed 0, closed grammar) | does solving the router objective on a warm trunk improve intent acc vs arm B? |
| B | same pinned bytes | **scrambled-label control** — same corpus, same byte stream, intent labels permuted | isolates the objective from trunk continuation: anything arm A gains over arm B is the SFT landing, not the 900k getting better at bytes |

Compare at **matched steps** (both t=889,897 + 30k), same data seed, same B.
Arm B is the W-vs-N-style control reframed for warm start — without it, a
positive arm A proves nothing about the objective.

The two numbers that decide everything are measured on the **val set, on CPU**:

1. **Does the warm trunk learn short-horizon SFT at all?** Intent accuracy (arm A)
   vs the scrambled-label (arm B) and vs a length-matched order-2 n-gram baseline
   (chance = 1/12). This is the first-ever measurement of SNR on a *supervised
   short-horizon* objective in this architecture. If A − B is within noise, the
   objective did not land and no amount of training time fixes it.
2. **The distance curve, for free.** Arm-A intent accuracy bucketed by command
   length (≤ 40 / 41–60 / 61–80 / > 80 B). If it collapses past ~64 B, the needle
   wall reappears on a supervised task — that is the roadmap's P2 distance-curve
   cell, obtained cheaply, on CPU, and it decides the production length cap.

**Gates (all must pass to proceed; val measured on CPU):**

| gate | bar | cost |
|---|---:|---|
| intent acc (val) | ≥ 0.55, and > 2× the n-gram baseline | from the same run |
| slot F1 (val) | ≥ 0.50 | from the same run |
| JSON-validity | ≥ 0.90 | from the same run |
| shift intent acc | ≥ 0.30 (out-of-family; zero would mean memorization) | from the same run |
| degeneracy | 0/9 degenerate, 0/9 non-terminating (existing probe protocol) | 6 seeds |

**Pre-committed branches (written before the run, not after):**

- intent acc ≈ chance at all lengths → the router axis is dead at 4.5 M without
  credit work; record as a first-class result, return to ROADMAP P0. Do not
  scale data.
- flat ≤ 64 B, collapse above → adopt the 64 B cap for production, add the
  distance curve to `tools/eval.py` as a permanent cell, scale the dataset.
- flat at all lengths → worth scaling: proceed to §6.

Dataset: 20k examples ≈ 3–6 MB — 16k train / 2k val / 2k shift, where **shift = two
categories entirely absent from train** (e.g., docker and hardware), so
generalization is measured out-of-family, not in-vocabulary.

## 5. What gets built (and the gates that ship with it)

| file | what |
|---|---|
| `tools/make_router_data.py` | label spec + paraphrase generator + teacher verification loop (teacher = `LiquidAI/LFM2.5-230M`, used as labeler/validator only) |
| `tests/check_router_labels.py` | **GATE, free.** parser-validates 100 % of JSON; slot-vs-text consistency (the argument occurs in the command, not hallucinated); 200 examples hand-verified, disagreement rate recorded |
| `tools/router_eval.py` | the two eval cells: intent acc + slot F1 + JSON-validity, and the command-length curve |
| `tools/router_data_digests.py` | sha256 per shard, so both arms are provably byte-matched (the 889k-run reproducibility lesson) |

**Hand-verification is a deliverable, not a nicety.** The teacher's labels are
synthetic; the repo's convention is that a label a human cannot reproduce is
not a label. Record the disagreement rate in `RESULTS.md` with the dataset.

## 6. The streaming / continual twist (v2 — only after the pilot gates pass)

The static command→JSON router is buildable by a 4.5 M transformer; nothing about
it uses this architecture. The version that does: **a stream router.** Input is a
*stream* (syslog, telemetry, SSH session, chat), not a sanitized command. TMT
detects the command-shaped window, routes it, and — per the README's own design —
**trains on its own corrected outputs at test time**. O(1) memory means per-
command learning is a handful of steps, not a context window; continual learning
was the architecture's founding claim and this is the first task where it is the
*product*, not a liability. **Out of scope for the pilot**, named here so the
pilot's data format (windowed commands with stable digests) does not rule it out.

## 7. Budget and quota

**Colab:** two arms × ~30k steps ≈ ~46 min total T4 (>23 min/arm at the measured
22–24 kB/s), i.e. **one slot, well under the ~1.3 h free tier used so far**
(`runbooks/SCHEDULE.md`). Push/pull cost per arm ≈ hot tail size 180–232 MB
(hub transport, measured class from LR-ROLES). Kaggle's 30 h/weekly quota stays
reserved for distillation's T4 steps, untouched.

**CPU (laptop, overlaps the Colab slots):** corpus build + label gates (free),
`router_eval` on val × 2 arms × 2 slices × the distance curve — 3 × 11–22 min
per ruler class. Total wall ≈ 2–3 h of Colab + rule nights on the laptop.
Nothing larger before the pilot gates pass — the pipeline has been burned twice
by scaling a measurement that never happened.

## 8. NOT in scope (any of these kills the pilot)

- Short technical Q&A generation (open-ended → the broken axis).
- General chat. Ever, at this size.
- The MI/KD pipeline. (§2)
- Any LR / optimizer / decay knob. The per-param axis is closed; touch nothing.
- Reasoning workloads. The teacher itself refuses them; the student cannot.
- L > 80 B commands, until the distance curve says otherwise.

## 9. What success looks like (and what falsifies it)

Success: val intent acc ≥ 0.7 at ≤ 64 B, distance curve flat to the cap,
JSON-validity ≥ 0.95, slot F1 ≥ 0.75 *on the shift set*, zero degeneracy
regression. That is the "4.5 M byte-recurrent model holds a narrow, closed
skill" result — publishable, honest, and the strongest claim this size supports.

**Defined non-success (pre-committed):** pilot A's intent acc within noise of the
n-gram baseline. If the trunk cannot separate intents on a short-horizon
supervised objective, no amount of data fixes it and the router direction is
false — recorded as a first-class result, per `RESULTS.md`'s convention that
corrections outrank conclusions.

**Confounds, stated in advance:** label/template leakage at eval (held-out
paraphrase pool); contamination between train and shift categories (disjoint
topic subsets, digests recorded — *by construction*, the same trick that made
`wiki_tinystories_001..004` usable); mean intent accuracy hiding per-class
failures (report the full per-class table); and the sampler's n-gram baseline
must match the student's corpus (same seed, same byte streams), or "beats
baseline" is a register artifact, not a result.

**Warm-start specifics, stated in advance:** the 900k trunk carries the mixed-run
register (and its measured CE-drift history); the router corpus is closed and
carries no ChatML/`User:`/`Assistant:` markers, so register leakage risk into the
*head* is low — but the final checkpoint still runs the existing leakage probe
(`tools/chat_models.py` inspection) and the 6-seed degeneracy protocol, because
"the start checkpoint was clean" is a fact about the start, not about the finetune.