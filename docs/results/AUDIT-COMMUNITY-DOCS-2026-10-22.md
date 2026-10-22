# AUDIT — the community-facing documents (2026-10-22)

**Scope.** The documents a *stranger* reads: `README-FORK.md`, `docs/HF-RELEASE.md`,
`docs/UPSTREAM-NOTE.md`, `docs/NIGHT-LOG-2026-10-21.md`,
`docs/design/CL2-INTERFERENCE-EXPERIMENT.md`, plus a repo-wide grep for the two
lessons tonight erased ("the state holds the payload" / "write selectivity is the
lever") and for the ridge readout presented as evidence of memory.
**Excluded:** `RESULTS.md` (append-only, carries its own corrections) and the
revised plan (`docs/plans/PLAN-2026-10-21-CONTINUAL-LEARNING.md`).
**Reference truth:** `RESULTS.md` §17–§23, `docs/REVIEW-2026-10-22.md`.

**Verdict in one line:** the *internal* record is honest and precise; the *public*
record is a description of the project as of 2026-10-21 — **before** the programme's
first end-to-end measurement — and it asserts, in three places, the two conclusions
tonight erased.

---

## 1. `README-FORK.md` — the fork's landing page. 8 findings.

| # | line | what is wrong | exact fix |
|---|---|---|---|
| R1 | 49 | **FALSE AND THE MOST DAMAGING SENTENCE IN THE FILE.** "**No measured continual-learning win yet:** `grep -c forgetting RESULTS.md` → **0**." The grep now returns 1; worse, it tells a reader *nothing has been measured*. A trained arm exists and has a complete verdict — and it FAILS 2 of 3 numbers. | Replace the bullet with: **"Continual learning is MEASURED, and it does not work yet. The first trained arm (the CL-RULER's three pre-registered numbers) scores ACQUIRE PASS (0.8906 exact-4 on the stream's facts, null 0.0020), RETAIN FAIL (ratio 0.0022 — every pre-stream fact forgotten), NO REGRESSION FAIL (held-out CE 3.0695 → 5.2037 nats/B, within 0.34 nats of a constant-byte predictor). The cause is measured too (§17–§23): the state carries the ADDRESS, the weights carry the CONTENT, and the two stores this architecture has are a copy and a table — neither is written once and read many times."** |
| R2 | — | **The CL-2 arm is not mentioned anywhere in the README**, although "continual/test-time learning" is the model's headline upstream claim. | Add R1 as a new bullet under "Honest caveats" *and* a row in the "Measured numbers" table: `Continual-learning arm (CL-2) | ACQUIRE PASS 0.8906 exact-4 / RETAIN FAIL ratio 0.0022 / NO REGRESSION FAIL 5.2037 nats/B | RESULTS.md §20` |
| R3 | 46 | "the state holds an addressable persistent **address** and a decaying **content**; decodable payload is real but short-horizon" — true only of the **D5 delayed-copy model, whose loss asked the state to carry the payload**. On the CL-2 arm the state carries the address and **not** the content: a fresh ridge on the state, split over **board facts**, is exact-4 = 0.0000 in every cell (§22). | Replace with: "**Readout is limited, and objective-dependent.** The D5 delayed-copy model (whose loss asked the state to carry the payload) shows a persistent **address** (65–165× over null) and a decaying **content** (4.6× over null at 8–12 B). On the CL-2 fact arm the same probe, split over **board facts** rather than rows, is **exact-4 = 0.0000 in every cell**: the state carries the address, the weights carry the content. A ridge fitted on a *row* split can memorise a table over the board and look like generalisation." |
| R4 | 10 | "**27+ executable gates**" — there are **35** (`tests/`), incl. `check_ruler_criterion.py`, `check_state_capacity.py`. | "**35 executable gates** (`tests/`)" |
| R5 | 6, 12 | The 1.26× is right, but it was realised **only on 2026-10-21** (when `auto` was made to resolve to `nvrtc`); the fused kernel was dead code until then and no earlier result was produced with it. | Append to L12: "(the fused kernel shipped 2026-10-03 but was **not the default until 2026-10-21**; every measurement before that was produced without it)" |
| R6 | 38, 45 | "0.56 nats/byte worse" describes the **incumbent checkpoint at 1.2094**. The continual-learning arm's own base was **3.0695** (2.42 nats above the same n-gram) and its post-stream model is **5.2037** (4.55 above). A reader who sees only "0.56" will badly underestimate the arm. | Add: "**The continual-learning arm is far worse than the 0.56-nat headline:** base 3.0695 nats/B (2.42 nats above the same order-8 n-gram), post-stream 5.2037 (4.55 above, within 0.34 of a constant byte)." |
| R7 | 13, 39 | "15× incumbent **retention**" / "Memory retention (P4+P6)" is the **Δh capacity statistic, not recall**. The trained head's own retention was **0.0022**. | Relabel: "**Δh retention probe (a capacity statistic, not recall** — the trained head's own retention on a fact stream was 0.0022**)" |
| R8 | 5 | "The model has **never been trained above 4.5M params** in this repo" — the CL-2 arm adds a memory head of 8192×1024+1024 = **8.4 M** params, so the arm scored for continual learning is **~12.9 M**, and the fact table it learned lives in that head. | "The base model is 4.5 M; the **continual-learning arm adds an 8.4 M-parameter memory head** (~12.9 M total), and the key→payload table the arm learned lives in that head." |

## 2. `docs/HF-RELEASE.md` — the public release plan / model card. 5 findings.

| # | line | what is wrong | exact fix |
|---|---|---|---|
| H1 | 39 | Model card: "Training: Continual/test-time capable **by design** (see caveats)" — the one line a reader can take as "this works". | "Training: continual/test-time training is **implemented** (weight updates on new bytes) and is **not demonstrated to work** — see Limitations" |
| H2 | 48 | "Memory readout decodable over ~12–128B depending on metric" — unqualified; it is the D5 model, and the fact-split caveat applies. | Use R3's replacement text verbatim. |
| H3 | 44 | "Memory retention (P4+P6): 37.5%/17.1%/5.3% … half-life 95.4B" — a Δh statistic in a public model card, labelled as retention. | "Δh retention probe (capacity statistic, not recall): 37.5%/17.1%/5.3% at 8/32/128 B; half-life 95.4 B. **The trained head's own retention on a fact stream was 0.0022.**" |
| H4 | 46-50 | The model card's Limitations have **no mention of the CL-2 verdict**. A public card claiming a memory would never say the memory was measured and failed. | Add three bullets: "(1) the first trained continual-learning arm fails 2 of 3 pre-registered numbers (ACQUIRE PASS 0.8906 exact-4; RETAIN ratio 0.0022; held-out CE 3.0695→5.2037 nats/B); (2) the recurrent state carries the **address**, the weights carry the **content** — a table that a new stream overwrites; (3) every recall number here is reported with a copy control (the payload in the *input* replaced) and a fact-level split, because both changed a conclusion during development." |
| H5 | 113 | Release checklist has no CL-2 item. | Add: "- [ ] **CL-2 verdict stated in the model card** (ACQUIRE/RETAIN/NO REGRESSION), including the state-vs-weights finding" |

## 3. `docs/UPSTREAM-NOTE.md` — the note to upstream. 6 findings. **This is the worst document in the repo: it teaches the lesson tonight killed.**

| # | line | what is wrong | exact fix |
|---|---|---|---|
| U1 | 33 | "**Conclusion:** Write is partly order-losing (content mixed) but readout is not key-blind; **next lever is write selectivity (e.g. P3 gate)**, not read wiring." — **SUPERSEDED.** The P3 gate engages (max 0.110) but its between-sample spread is **2.4e-05** of its within-sample spread: a global per-dimension re-scaling, not selectivity. And a selective *state* write could not have fixed the failure, because the failure was a **table in the weights** being overwritten. | Replace the conclusion bullet with: "**Conclusion (revised 2026-10-22):** D5's model does hold a payload — but the **loss** is what makes that true: D5's objective *asked* the state to carry the payload. On our continual-learning arm, where nothing asked it to, the state carries only the **address** and the payload lives in the **weights**, as a table (a fresh ridge on the state, split over board facts, is exact-4 = 0.0000 in every cell). So the D5 reading is a property of the **objective**, not of the architecture. We have closed 'write selectivity' as the lever: the gate engages but is a global re-scaling, not selectivity — and a selective state write could not have fixed a table being overwritten." |
| U2 | 29 | Heading "D5: state holds an addressable, persistent address and a decaying content" — needs the model qualifier. | "## 5) D5 (**on the D5 delayed-copy model**): the state holds an addressable, persistent address and a decaying content" |
| U3 | 34 | Refs "PLAN §0/§4" — the plan's §4 D5 update is now marked superseded in place. | Cite `RESULTS.md` §22/§23 (and the plan's CURRENT STATE box) instead of the plan's §4. |
| U4 | 12 | "with a fixed instrument the readout **does** work" — true of D5 only. | "with a fixed instrument the readout **does** work **on the D5 delayed-copy task** (see §6 for what happens when the objective does not ask the state to carry the payload)" |
| U5 | 14-16 | "P4+P6 capacity result (15× incumbent) … **retention**" — the Δh statistic, same as R7. | Relabel as "Δh retention probe (capacity statistic, not recall)". |
| U6 | — | **The note omits the programme's most relevant finding.** Upstream's README advertises "Test-time training"; the fork has now *measured* continual learning and found the failure, with a measured cause. Withholding that from the author whose README asks for outside review is the material omission. | Add "## 6) Continual learning: measured — and it fails, and we know why" with the three-number verdict table, the address-vs-content synthesis, the two failing numbers' separate causes, and an offer of the ruler (`tools/cl_ruler.py`, `tests/check_ruler_criterion.py`) to upstream. |

## 4. `docs/NIGHT-LOG-2026-10-21.md` — the recovery point. 4 findings.

| # | line | what is wrong | exact fix |
|---|---|---|---|
| N1 | 6 | "`main` and `t-axis` are **in sync** at `db7adf8`" — false. HEAD is `d47d4e7` on `t-axis`; `main` is `9a78bb3` (`[origin/main: ahead 38, behind 40]`); `t-axis` is ahead 1 of `fork/t-axis`. A recovery point that points at the wrong SHA is worse than none. | Replace with the current SHAs and the branch relation, and date-stamp the line. |
| N2 | 7 | "the plan: … **§CL-2b**" — the plan has been revised; the current step is now **CL-6** (CL-0/CL-1 done, CL-2 run, CL-3 superseded, CL-4 deferred, CL-5 closed, CL-6 added). | "the plan: `docs/plans/PLAN-2026-10-21-CONTINUAL-LEARNING.md` (current step: **CL-6**; the log's §CL-2b was the 2026-10-22 version)" |
| N3 | 69-73 | "the ruler's criterion wiring (ACQUIRE must be board B …) — the gate is `tests/check_ruler_criterion.py`" — **DONE**: the gate exists, passes, and the ruler reads the regime from the checkpoint. | Delete that item; keep A5 (text-only counterfactual) and the `--memory-alpha 0` control arm. |
| N4 | 8-11, 49 | "the MX150 … is IDLE" and "THE FIRST THREE THINGS TO DO NEXT" are a **session snapshot**, not a live state; the plan is now the single source of truth. | Date-stamp ("as of the end of the 2026-10-21/22 session"); retitle the list "the session's three next things" and add "→ now tracked as the plan's CL-6." |

## 5. `docs/design/CL2-INTERFERENCE-EXPERIMENT.md` — the design the plan points at. 5 findings.

| # | line | what is wrong | exact fix |
|---|---|---|---|
| D1 | 25-30 | "**Synthesis (§21):** the recurrent state holds a **COPY** of the input's payload (a fitted probe reads board C at 0.1953 exact-4; collapses to NULL when key AND payload replaced)" — **SUPERSEDED by §22**: the 0.1953 was a **table over the board** (the split was over rows); with the split over **board facts** it is exact-4 = 0.0000 in every cell. The state holds the **address**; the weights hold the **content**. This doc currently teaches the wrong synthesis to the next reader. | Replace the paragraph with §22's corrected synthesis ("the recurrent state carries the ADDRESS — the key — durably and generically, even for facts the model was never taught; the CONTENT lives only in the WEIGHTS, in a table") and add a banner: "**§25-30 superseded by `RESULTS.md` §22** (the ridge's 'generalisation' was a table over the board)." |
| D2 | 32-36 | "**A PASSING RETAIN DOES NOT SHOW THE STATE IS DOING THE WORK** … the copy control plus the zeroed-head/randomised-head eval are what tell store kinds apart" — **correct, and the most valuable paragraph in the doc; protect it verbatim.** It is incomplete by one control. | Add: "…**and every fitted readout must be re-run with the fact-level split** (§22), or a row split can let it memorise the board." |
| D3 | 61-66, 87 | Arms B1/B2/B3/B4/B5 and the tooling list reference flags that **do not exist**: `--head-kind slots`, `--replay-frac`, `--replay-board`, `--decoupled-head`, `--fact-fraction`; and `tests/check_cl2_arms.py` is **not in `tests/`**. The doc reads as if they are available. | Head the arm table "**PROPOSED — NOT IMPLEMENTED**" and add a status line: "no `--head-kind`/`--replay-*` flag exists today (`tools/cl2_train.py`); `tests/check_cl2_arms.py` does not exist yet. The only implemented arms are those in the plan's CL-6." |
| D4 | 30 | "P3 (write selectivity) is retired as the lever for RETAIN" — **correct**; keep, and add the measured reason: "the gate engages (max 0.110) but is a global per-dimension re-scaling (between/within spread 2.4e-05), and a selective state write could not have fixed a **table** being overwritten (§23)." |
| D5 | 39-43 | The capacity budget ("~241 facts/board for two boards at dim 512") is consistent with `RESULTS.md` §18 (481 facts total) — keep. | no edit |

## 6. Repo-wide grep — remaining "state holds the content" / "selectivity is the lever" / ridge-as-memory.

| file:line | what is wrong | exact fix |
|---|---|---|
| `tools/state_fact_capacity.py:4` | The docstring cites **§21** ("the programme's synthesis … says the recurrent state holds a COPY … generic to unseen facts") — the superseded synthesis, and this docstring is *why* the board-64 arm was designed. | "The programme's **corrected** synthesis (`RESULTS.md` §22, superseding §21) says the recurrent state carries the **ADDRESS** and the weights carry the **CONTENT** (a table). This tool measures how many facts the state can hold *at all* — a capacity question that is independent of tonight's result." |
| `tools/measure_m6_selectivity.py:36` | "if the ridge's answer follows the replacement, the state holds a COPY of the input" — true only for a readout fitted with a **fact-level split**; with a row split a ridge can memorise a table over the board, so the inference does not hold (§22). | Append: "…**only if the readout was fitted with the split over BOARD FACTS**; with a row split a ridge can memorise a table over the board, and 'the state holds a copy' is not inferable (`RESULTS.md` §22)." |
| `tools/d5_probe.py:10-20` | The framing ("the delta-h probe says the state HOLDS a lot"; "the WRITE is order-losing") is D5-specific and still true of D5's model, but the docstring does not say the result is **objective-dependent**. | Add one line to the docstring: "NOTE (2026-10-22): this result is a property of D5's **objective** (its loss asked the state to carry the payload). On an objective that does not ask, the state's usable content is 0 — `RESULTS.md` §22." |
| `tools/cl_ruler.py:138` | "The state's USABLE HORIZON (D5, PLAN §0): the payload decodes to ~128 B" — true of D5; unqualified in the current instrument. | Append: "(D5's delayed-copy objective; where the objective does not ask the state to carry the payload, usable content is 0 — `RESULTS.md` §22)." |
| `docs/papers/RELEVANCE-NOTE.md:188` | "the state holds a signal it cannot *use*" — the superseded picture, in a paper-facing note. | Add a footer: "(2026-10-22 update: the state-carries-content reading is superseded — `RESULTS.md` §22.)" |
| `docs/reviews/STRATEGIC-REVIEW-2026-10-05.md:252-258` | "The testable form of the objection is **selectivity**" — and `docs/INDEX.md:35` still marks this file "**current**". Selectivity is closed as a lever (§23). | Mark the file **superseded in part** in `docs/INDEX.md` ("§3's selectivity axis is closed — `RESULTS.md` §23") and add a banner to the file. |
| `docs/reviews/RESPONSE-2026-10-06-EXTERNAL-CRITIQUE.md:169` | "TMT may have enough passive temporal persistence but insufficient **selectivity**" — same closed axis, and it is the argument the programme's own write-selectivity lever was built on. | Same: banner + INDEX status. |
| `docs/plans/PLAN-2026-10-04-ROADMAP.md:287`, `docs/plans/PLAN-2026-10-06-MEMORY-BENCHMARKS.md:315` | The old queue still lists "Selective write gate" as item 7. | No edit (superseded plans), **but** confirm `docs/INDEX.md` marks both superseded and that no index line resurrects the gate. |
| `README.md` (repo root) | This is **upstream's own README** ("there are very promising results", "Test-time training") and it is what a stranger sees first on the fork's public repo. | Add a 3-line **fork banner** at the top: pointer to `README-FORK.md`, the CL-2 verdict, and `RESULTS.md`. Do not edit upstream's prose. |
| `docs/INDEX.md` | No pointer to `README-FORK.md`, `UPSTREAM-NOTE.md`, `HF-RELEASE.md`, `NIGHT-LOG-2026-10-21.md`, `REVIEW-2026-10-22.md`, `design/CL2-INTERFERENCE-EXPERIMENT.md`, or this audit. | Add a "COMMUNITY / RELEASE / AUDIT" block listing them with one-line statuses. |

---

## PRIORITY — the three that must be fixed before anything is sent or published

1. **U1** (`UPSTREAM-NOTE.md:33`) — it would tell upstream that write selectivity is the next lever. It is not; it is closed, with a measured cause.
2. **D1** (`design/CL2-INTERFERENCE-EXPERIMENT.md:25-30`) — the design the plan's current step points at teaches the superseded synthesis.
3. **R1 + R2** (`README-FORK.md:49`) — the fork's landing page says the programme has measured nothing about continual learning, when it has measured it and it fails.

Everything else in this file is accuracy polish on top of those three.

## BOTTOM LINE — is the repository's public record honest right now?

**No, and the gap is not subtle.** The internal record — `RESULTS.md` §17–§23, the revised
plan, `docs/REVIEW-2026-10-22.md` — is honest to the point of severity: a trained arm,
three pre-registered numbers, two FAILs, a measured cause, and a list of the programme's
own instrument errors. The public record is a **snapshot of 2026-10-21, before the
programme's first end-to-end measurement**, and it still asserts both conclusions the
night erased: that the state holds a decaying content (true only of the D5 model, whose
loss asked for it) and that write selectivity is the next lever (closed, with a measured
cause). A stranger who reads `README-FORK.md` and `docs/UPSTREAM-NOTE.md` today would
conclude the fork has a working memory with a clear next step; the truth is that the fork
has a **working instrument**, a **failing first measurement**, and a next step that is a
new **kind** of store. The single most load-bearing omission is that **no
community-facing document contains the verdict at all** — the project's headline upstream
claim is "continual/test-time learning", the fork has finally measured it, and the README,
the model card and the note to upstream are all silent about the failure.
