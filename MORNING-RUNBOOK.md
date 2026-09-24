# MORNING RUNBOOK — first GPU training legs (2026-09-24)

Everything below is already tested. No discovery needed. Kaggle quota is the
scarce resource: Colab first (fast answers), Kaggle only for the long haul.

## What is proven (night session, 2026-09-23/24)

| gate | result |
|---|---|
| numpy ≡ torch parity, 150 live steps (tail engaged) | loss rel ≤ 2.9e-07, grads ≤ 3.6e-03 |
| live checkpoint continuation (t=64,747, K=18, 57 hot dims) | rel 1.1e-07 |
| npz round trip (torch → numpy → torch) | bit-exact |
| **Colab T4 speed** | **B=1024: 32,353 B/s · B=2048: 32,806 · B=4096: 34,259** (33× the CPU ~1,000 B/s) |
| CUDA vs CPU torch loss (192 hot dims, tail on GPU) | rel 1.10e-07 |
| GPU save → reload → continue | exact (0.0) |
| train_gpu on GPU + tagged saves | PASS |
| corpus fetch (download→stage) + trainer consumption | PASS |

**Next optimization (not done, deliberately):** the ring credit scatter
(`index_add_`, ~600M atomics/step) now dominates GPU step time. Sorting the
byte ids + segment-sum should cut step time several-fold. Do it as its own
measured step, not silently.

---

## 1. Colab — quick answers (use `colab run`, dev loop)

`colab run` takes a local script, runs it on a fresh VM, releases it. For
anything that needs the repo modules, regenerate the self-contained bundle:

```bash
.venv/bin/python make_colab_bundle.py     # packs modules into colab_final_test.py
```

### 1a. A short real training leg (recommended first)

Write `colab_leg.py` (or adapt): the bundle pattern — unpack modules, then:

```python
# 10k steps from the LIVE checkpoint, B=1024, save every 500
import gpu_port, train_gpu as tg, os
# 1. pull tmt.npz to /content (upload it once, or fetch from the hub)
tr = tg.Trainer(path="tmt.npz", batch=1024, save_every=500, keep_k=4,
                max_steps=10_000, device="cuda", dim=512, layers=16)
tr.train(True, "corpus/corpus_*.txt")     # corpus via fetch_corpus.py
```

Run: `colab run --gpu T4 --timeout 3600 colab_leg.py`, then
`colab download` the `tmt_t*.npz` back. Judge the result with the CPU ruler
(`eval.py`) — the numbers must be comparable to the baseline row in RESULTS.md.

### 1b. Only if a leg looks wrong: `colab run --gpu T4 colab_final_test.py`
(the full gate; ~4 min; regenerable with `make_colab_bundle.py`).

## 2. Corpus — in-session, sized by measured speed

```bash
python fetch_corpus.py --layers A,B,C --gb 4.5 --out ./corpus   # prints ETA
```

It prints the ETA from this session's measured B/s. Sizing rule of thumb at the
**measured 34 kB/s**: 1 h ≈ 123 MB · a 4.5 GB one-epoch pass ≈ 37 h → too big
for one 12 h Kaggle session, so either raise throughput first (scatter
optimization) or run 1.2 GB per session and resume. The fetch is idempotent
per layer; Kaggle can save `./corpus` as a dataset version once so later
sessions mount instead of re-downloading.

## 3. Checkpoint transport

`checkpoint_hub.py` is wired into `train_gpu.py`:

```bash
export TMT_HUB_URL=bakali1/tmt-checkpoints    # repo from the other model
export TMT_TOKEN=...                           # or `hf auth login` once
python train_gpu.py tmt.npz train --hub-repo bakali1/tmt-checkpoints ...
```

Pull refuses the past (remote t ≤ local t is skipped), pushes are tagged +
atomic. Verify with `python checkpoint_hub.py list`.

## 4. Kaggle — long haul only (after a Colab leg is proven)

Mirror the Colab flow: same `train_gpu.py`, same hub repo, `--save-every 500`
(checkpoint every few minutes; a 12 h kill costs ≤ the last interval), and log
the session in `SCHEDULE.md` (hours used / 30).

## 5. What to do with the speed, in order

1. **Experiment D** (corpus verdict) — now ~20 min/arm instead of 5–8 h.
2. **Phase 2/3** spread A/B — same day.
3. **Scatter optimization** if throughput binds (the only known hot spot).
4. Corpus volume + Phase 4 scale after the verdict.

## 6. Rules that keep comparisons honest

- One arm = one platform + one `--data-seed` (never mix a CPU leg into a GPU
  A/B; never compare B=1024 vs B=4096 arms without the ruler).
- fp32 everywhere (no AMP, `allow_tf32=False`) — the port enforces it.
- Judge with `eval.py`'s 3-file mean ± SE; ≈0.03 CE is the real-difference bar.
