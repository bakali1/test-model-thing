# GPU plan implementation review — third full review

**Reviewed:** current worktree at commit `53db379` (`leg bundle: embed the hub repo id, not just the token`). The previous P0/P1 findings were rechecked against the current code rather than copied from the earlier report.

**Plan reviewed:** `PLAN-2026-09-23-GPU.md`  
**Overall verdict:** **The implementation fixes are now present and the local gates pass, but the system is not yet clean for production use.** There is one critical secret-management issue in the generated live-leg artifact, several integration/reproducibility gaps, and the real live-checkpoint training leg plus Kaggle run are still not recorded.

## Current gate results

| Check | Result |
|---|---|
| `tests/check_parity.py` | **PASS**; loss `2.87e-7`, gradient `3.63e-3`, ring/tail accumulator checks pass, post-resume state checks pass, ring warmup and membership cadence tests pass. |
| `tests/check_trainer.py` | **PASS (6/6)**; A/B tag isolation, pre-allocation tail cap, state invalidation, hub validation, equal-step conflict, and incompatible CLI handling. |
| `tests/check_fetch.py` | **PASS**; transforms, versioned publication, crash-before-publish recovery, pagination parsing, and budget mix. |
| `tests/check_ring_k.py` | **PASS**. |
| `test_tail.py` | **PASS**. |
| Python compile / `git diff --check` | **PASS**. |
| `kaggle/tmt_gpu_train.py --dry-run` | **PASS operationally**, but it exposed a corpus path mismatch described below. |
| Real live-checkpoint GPU training | **Not yet recorded**. `SCHEDULE.md` still says this is the next action. |
| Real Kaggle leg | **Not yet run**. |

The previous findings about A/B tag collisions, late tail-cap enforcement, stale rebase state, CUDA loss-only gating, ungated post-resume state, corpus publication, HF pagination, hub filename trust, equal-step conflicts, and missing Kaggle code are fixed or covered by current tests.

## Critical remaining issue

### 1. A live Hugging Face token is embedded in the generated Colab leg — P0 security

The ignored generated file `colab_leg.py` contains a literal `TMT_TOKEN` value. `make_leg_bundle.py` deliberately reads the token from the environment or the local Hugging Face cache and embeds it into the generated script. The generated file is ignored by git, but that does not make the secret safe:

- it exists in the local workspace;
- it can be uploaded to Colab, cloud storage, logs, crash reports, or backups;
- it can be accidentally copied into a shared repository or archive;
- a token with repository write access can compromise the checkpoint repository.

The generator's current message only avoids printing token characters; it does not remove the secret from the generated file.

**Required action:** revoke/rotate the exposed token. Change the workflow so the generated bundle does not contain credentials. Use Colab/Kaggle secrets, a short-lived scoped token supplied at runtime, or an interactive/manual secret injection step. Keep generated bundles out of shared artifacts and add a secret-scanning test that fails if `hf_` token patterns appear in generated files.

This is more urgent than the remaining training-execution gaps.

## Remaining correctness and integration issues

### 2. The live-leg script does not pull when a local checkpoint already exists — P1

`colab_leg.py` chooses the local checkpoint before considering the Hub:

```python
if os.path.exists(CKPT) ...:
    src = f"local file {CKPT}"
elif HUB_REPO:
    hub.pull(...)
```

This differs from `Trainer.train()`, which always pulls first when a Hub is configured. A rerun with a stale local `tmt.npz` can silently train the stale checkpoint instead of the newest validated remote checkpoint. A fresh Colab VM often avoids this, but the protocol is still incorrect for retries and reused sessions.

**Fix:** when `HUB_REPO` is configured, call `hub.pull()` first, then inspect the resulting local path. Make the live-leg integration test cover stale-local plus newer-remote.

### 3. Kaggle fetch and training default to different corpus paths — P1

`kaggle/tmt_gpu_train.py` defaults the training glob to:

```text
data/corpus/current/**/corpus_*
```

but when `TMT_FETCH=1`, it invokes `fetch_corpus.py` without adding `--out`. The fetch default is `./corpus`, so the generated corpus is published at `corpus/current`, while training searches under `data/corpus/current`.

A dry run demonstrates the mismatch:

```text
fetch: ... fetch_corpus.py --layers C --limit 1 --gb 0.001
train: ... train_gpu.py ... --dataset data/corpus/current/**/corpus_*
```

With the default `TMT_FETCH_ARGS`, the fetch will succeed but training will fail with no matching files.

**Fix:** make the paths a single shared constant, or set the fetch output to `data/corpus` by default. Add a local end-to-end test that executes a tiny fetch and then a tiny training call using the generated `current` glob.

### 4. Corpus revisions are recorded but not pinned during download — P1 reproducibility

`fetch_corpus.py` records `repo_revision()` in the manifest, but `download()` calls `hf_hub_download()` without `revision=...`. The file listing and each download therefore resolve `main` independently. If a dataset repository changes during a long fetch, the manifest can claim one revision while some files came from another revision.

**Fix:** capture each layer revision before listing, pass it to `hf_hub_download`, and record it once. If the revision changes, fail rather than mixing trees.

### 5. Hub validation is optional in the normal trainer path — P1

`checkpoint_hub.validate_npz()` supports `expect_n`, and the hub unit test calls it, but `Trainer.train()` calls `hub.pull(self.path)` without `expect_n=self.model.theta.numel()`. `colab_leg.py` also calls `hub.pull()` without the expected parameter count. Thus the claimed “wrong-shape remote file is rejected” check is not active in the real training path; a later `GpuModel.load()` may fail instead of rejecting the file before it becomes the rolling truth.

**Fix:** pass the expected parameter count from every caller, and validate all required arrays/shapes, not just `m.shape[0]`.

### 6. SHA-256 sidecars are optional, and push is not atomic as a pair — P1

`pull()` warns and continues when a sidecar is absent. `push()` uploads the `.npz` first and the `.sha256` sidecar second. If the sidecar upload fails, the remote has a selectable checkpoint with no digest; the current pull accepts it with only a warning. This is weaker than the plan's “every push/pull is content-verified” wording.

**Fix:** use a two-phase remote publication protocol or require the sidecar for newly pushed checkpoints. At minimum distinguish “legacy/unverified” checkpoints and refuse them for production arms.

### 7. Remote pruning deletes checkpoints but not their sidecars — P2

`prune_remote()` selects and deletes only `.npz` victims. Their `.sha256` sidecars remain in the Hub indefinitely. This is not a training correctness failure, but it causes storage leaks and leaves confusing stale metadata.

**Fix:** delete each victim’s sidecar in the same explicit retention operation, with a dry-run listing of both files.

## Performance and resource issues

### 8. The live-leg HTTP range download has no hard byte limit — P1 resource risk

`colab_leg.py` requests `Range: bytes=0-N` but reads until EOF and never truncates the stream at `want`. If the Hugging Face endpoint ignores the range and returns `200 OK` with the full file, the script can download the entire approximately 1.9 GB TinyStories file despite the comment that it only needs about 64 MB.

**Fix:** check the response status/`Content-Range`, stop writing after `want`, and fail if the server returns more data than requested. This is especially important in Colab where disk and session time are limited.

### 9. The live-leg and runbook step counts disagree — P2

The runbook describes a 10k-step live leg, but the generated `colab_leg.py` defaults to `STEPS=30000` and its output text says “30k steps.” A command that does not set `TMT_STEPS` will run a substantially longer leg than the runbook’s stated validation task.

**Fix:** use one default everywhere, preferably 10,000 for the first disposable integration leg, and have the runbook print the exact effective value.

### 10. The pre-allocation tail cap does not account for peak replacement memory — P1

`TorchTail.sync()` checks projected bytes before allocating the new layer tensor, but the old tensor for the same layer remains alive until the new tensor is assigned. The peak is therefore approximately old current-layer E + new current-layer E, while the check counts only the other layers plus new E. The cap can pass while actual peak memory exceeds the cap.

**Fix:** include the current layer’s old E in the peak projection, and explicitly release or stage it before the new allocation if necessary. Add a same-H rebuild test, not only a fresh-tail cap test.

### 11. Checkpoint tail restoration still bypasses a true pre-allocation check — P1

`_tail_restore_from()` creates torch tensors from checkpoint E arrays before the subsequent `_tail_sync()` cap check. A large or malicious tail checkpoint can therefore allocate beyond the cap or OOM before the intended guard runs.

**Fix:** validate every restored `(B, 256, H)` shape and aggregate bytes before copying to CUDA, and apply the same cap policy used by `TorchTail.sync()`.

## Accepted design limitations

These are documented, but should remain visible:

- `ring_b` and `ring_w` are not persisted. A resumed run is recoverable, not bit-identical; cold-dimension credit warms over up to `ring_k` steps. The parity test now explicitly verifies this shape.
- GPU tail membership is re-evaluated every 500 steps while NumPy re-evaluates every step. The test intentionally demonstrates the lag and forced reconciliation. This is a deliberate GPU contract difference, not exact cross-engine membership parity.
- `float(loss)` and `float(ce)` synchronize with CUDA. This is accepted for now because the measured sync is small relative to the step, but it remains a performance optimization candidate.
- The GPU benchmark is not the same as a long live-checkpoint throughput run. The recorded T4 numbers are useful smoke measurements, not a guarantee for every corpus/arm configuration.

## Documentation and process gaps

1. The plan now describes a 10k live leg, but the generated default is 30k.
2. The Kaggle default dataset path and fetch default output path must be unified.
3. The plan says every pull is digest-verified, while missing sidecars are currently accepted with a warning.
4. The plan says the live checkpoint is “crash-proof”; the implementation correctly calls it recoverable because ring state is omitted. Keep that wording consistent everywhere.
5. `SCHEDULE.md` logs the short Colab gates but still says the live leg and Kaggle leg are not run.
6. `colab_leg.py` is ignored, but the presence of an embedded secret makes ignoring it insufficient protection.

## Recommended order

1. **Immediately revoke/rotate the exposed token and redesign bundle authentication.**
2. Fix Kaggle’s corpus output/input path and add an end-to-end local fetch→train smoke test.
3. Make the live-leg pull-on-boot behavior match `Trainer` and test stale-local/newer-remote behavior.
4. Pin corpus revisions and require/pass checkpoint shape validation in the real pull path.
5. Make checkpoint sidecar publication/verification a defined protocol; clean up sidecars during remote pruning.
6. Bound the Colab range download and align 10k/30k defaults.
7. Include replacement-tail peak memory and checkpoint-restored tail memory in the cap guard.
8. Run the disposable live checkpoint leg, then the first Kaggle leg, and record both in `RESULTS.md` and `SCHEDULE.md`.

## Final assessment

The numerical and infrastructure fixes are now real and the local gates pass. I would not call the plan fully complete until the exposed token is revoked, the Kaggle corpus path is fixed, and the live/cloud execution path is actually run and recorded. The remaining issues are mostly integration and operational risks rather than the original GPU math defects.
