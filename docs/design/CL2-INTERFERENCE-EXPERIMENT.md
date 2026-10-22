# CL-2b — THE STORE (design)

Plan: `docs/plans/PLAN-2026-10-21-CONTINUAL-LEARNING.md` §CL-2b  
Ruler: `tools/cl_ruler.py` (criterion wiring fixed)  
Budget: `tools/state_capacity_budget.py`

## Two failures, kept apart

1. **TEXT REGRESSION (separate cause).** The behavioural loss rides on the
   SAME logits as the text CE (`ce_behav` computed from shared `flat` over
   payload positions). Evidence: in f2 held_ce rose 4.19→4.59 while
   `ce_behav` fell 0.054→0.018 and fact accuracies climbed.  
   Pre-registered shape test for decoupled head: held-out CE should be
   FLAT-TO-FALLING across f2 (step 500→1000) rather than rising.
2. **RETAIN FAILURE (urgent).** From `RESULTS.md` §17,§20,§21 and
   `docs/results/cl2/armA_ruler.json` (ruler in arm regime facts=4,gaps=8/32/64):
   - ACQUIRE (board B, stream facts), head: **0.9697/0.8906** per-byte/exact-4
     (null ~0.0020) — **PASS** (threshold 0.50).
   - RETAIN: head pre 0.9644/0.8691 → post 0.1978/0.0020, **ratio 0.0022**
     vs threshold 0.90 — **FAIL**.
   - NO REGRESSION: held_ce 3.0695 (base) → 5.2037 (post); post within
     **0.34 nats of unigram baseline 5.5452** — **FAIL** (counterfactual A5
     can only make regression larger; failure certain, A5 quantifies).

**Synthesis (§21):** the recurrent state holds a **COPY** of the input's
payload (a fitted probe reads board C at 0.1953 exact-4; collapses to NULL
when key AND payload replaced). The trained head holds a **TABLE** of
key→payload in the **weights** (survives replacing input payload). **Neither
is a store written once and read many times.** P3 (write selectivity) is
retired as the lever for RETAIN (overwriting a table, not decaying a copy).

**A PASSING RETAIN DOES NOT SHOW THE STATE IS DOING THE WORK.** A bigger
head holding more facts is the global-table route, not the state route.
**The copy control (payload in input replaced, key kept) plus the
zeroed-head/randomised-head eval are what tell store kinds apart** — both
are **mandatory in every cell/arm**.

## Capacity budget
`tools/state_capacity_budget.py`: state 16×512 = **8192 scalars**. Linear-
readable store: optimal coding needs **8704 dims** for 512 facts (1.06× short)
or **17408** for 1024 (2.12× short); one-hot needs 4.06×/8.12×. For two
boards at dim512 need ~**241 facts/board** feasible (optimal). Growing model
alone is not obviously the fix; **kind of store must change**.

## Criterion wiring (fixed)
- **ACQUIRE = board B** (stream's NEW facts), read with the model's **own
  trained head**. **Board C** (never trained) reported separately as the
  **in-state route / stronger claim**. Verdict object states both; headline
  cannot be mis-wired to board C.
- **Ridge** is REFITTED at each measurement → measures **state COPYABILITY,
  never retention**. Head is the model's parameter → honest measure.
- Ruler reads regime from checkpoint by default (**cl2_facts/cl2_gap/cl2_alphabet**)
  with explicit override.

## Arms (pre-registered), in priority order

### (B) RETAIN / STORE — urgent

| Arm | Matched control | Matched null | Nuisance controls | Threshold (pre-reg) | Min effect size | GPU-hours (est.) |
|---|---|---|---|---|---|---|
| **B0: smaller board** `--board-size 64` (128 facts total). Cheapest decisive test. | `board_size 512` (armA) same model/training. | Text/geometry-matched null at `board_size 64` with same schedule. | Copy control (payload in input replaced, key kept); zeroed/randomised head eval. | Head exact-4 on board A (post) >= **0.05** (from 0.0020); ACQUIRE drop <= 0.05; held_ce within **0.05** nats/B. | RETAIN ratio >= **0.10** or board-A post>=0.05 exact-4 with state-route evidence. | ~1-2h (20× cheaper than scaling model) |
| **B1: content-addressed slot store** (`--head-kind slots`) | B0 or baseline linear head. | Same with standard linear head at matched capacity. | Copy control; zeroed-head eval (must force state/path). | RETAIN ratio >= **0.90** OR board-A post exact-4 >= **0.50** with copy-control showing state involvement. | >=0.50 exact-4 on board A post, beats null. | ~2-4h |
| **B2: protected table** (`--head-kind` variant / protected slots) | Linear head, matched slots. | Unprotected table. | Copy control; zeroed-head eval. | Same as B1, pre-registered. | Evidence store kind changed vs capacity alone. | ~2-4h |
| **B3: replay** (`--replay-frac`/`--replay-board`) | No-replay CL-2b. | No-replay matched. | Copy control. | RETAIN ratio >= **0.90**. Must be measured on same three numbers. | Reference baseline established. | ~2-4h (depends on buffer) |
| **B4: bigger state** (dim>=1088 optimal / >=4160 one-hot for 2 boards) or nonlinear/sparse. | dim512. | dim512 matched. | Copy control; zeroed-head eval. | RETAIN improvement with state evidence. | Distinguish from bigger-head (B5). | ~4-8h |
| **B5: bigger head** (capacity route) | Baseline head. | Baseline. | **Mandatory** copy control + zeroed-head eval to show it's table route. | May pass RETAIN by capacity — **not** proof of state route. | Demonstrate distinction explicitly. | ~1-2h |

**Kill criterion:** if `board_size 64` at dim512 gives RETAIN ~0 and copy control shows payload not in state, capacity hypothesis refuted; optimizer preferred head (E1) → force state path via zeroed/randomised-head eval (head cannot memorise, only state can reduce memory loss).

**Memory-capacity probe (required):** retention-vs-board-size curve at fixed model (e.g. board_size 32,64,128,256,512), facts per stream matched to regime, report head exact-4 on board A post at real lags. Tests budget prediction (~241 facts/board for 2 boards at dim512). Pre-registered shape: drops to ~0 past budget; if curve near prediction budget model validated, else wrong.

### (A) TEXT REGRESSION — separate

| Arm | Matched control | Matched null | Nuisance controls | Threshold (pre-reg) | Min effect size | GPU-hours |
|---|---|---|---|---|---|---|
| **A0** `--memory-alpha 0.25 --behaviour-weight 1.0` (baseline) | — | — | — | Baseline. | — | ~as before |
| **A1** `--behaviour-weight 0.1` | A0. | Matched. | Check text path. | held_ce shape: less rise vs A0 in f2; end within reason. | Shape prediction holds. | ~same as A0 |
| **A2** fact-diluted batch (e.g. 1/8 facts) | A0. | Matched. | — | Regression scales with fact load if interference. | Shape consistent with dilution. | ~same or less |
| **A3** decoupled head (payload via separate readout, not shared logits) | A0. | Matched. | Ensure text logits unchanged. | **held_ce FLAT-TO-FALLING** f2 step 500→1000 (not rising). End within 0.05 nats/B of text-only trajectory. | Sharp shape change. | ~same |
| **A4** `--memory-alpha 0` | A0. | Matched. | — | Answers: does MEMORY loss cost text? (separate from fact training). | As measured. | ~same |
| **A5 (required)** text-only counterfactual, same model/budget/steps, no fact streams | A0. | **Null for regression cell** (`dCE = CE(A5)-CE(A)`). | — | Quantifies regression (failure certain to be >= measured). | Makes NO-REGRESSION well-posed. | ~cheaper (no fact overhead) |

Pre-register all thresholds before runs. A cell that does not beat its null is NULL, never PASS.

## Tooling changes (opt-in, default path bit-identical)

- `tools/cl2_train.py`: `--head-kind {linear,mlp,slots}` (default `linear`), `--replay-frac`, `--replay-board`, eval-time switch to **zero or randomise head**, **memory-capacity probe** reporting, **RETAIN-at-a-real-lag** (lags 8/32/64, `8*(facts-1)+d` = 32/56/88 for facts=4). `--behaviour-weight`, `--fact-fraction`/dilution, `--decoupled-head` as needed. Default behavior bit-identical; verify with `--selftest` before/after.
- `tools/cl_ruler.py`: read regime from checkpoint by default; ACQUIRE wired to board B, board C reported separately; head is primary readout (ridge copyability only).
- `tests/check_cl2_arms.py`: must FAIL on pre-change code and PASS after — asserts flags exist, default path unchanged (tiny run reproduces old numbers exactly), zeroed-head eval actually zeroes head.
- `tests/check_ruler_criterion.py`: must FAIL on pre-change and PASS after — asserts ACQUIRE is board B, regime read from checkpoint, wiring prevents wrong-cell headline FAIL.

## Audit notes (behavioural/copy)
Earlier "behavioural recall" numbers where target bytes appear in input are **copies**.
- Training log `behav 0.999/0.994` = copy of input (proved). Judge by **head readout of state/weights**.
- Copy control (payload in input replaced, key kept/target handled) mandatory for every recall number.
Audit `RESULTS.md` and `docs/design/P3-GATE-*.md` for copy-based numbers (list in this doc; do not auto-edit RESULTS.md).

