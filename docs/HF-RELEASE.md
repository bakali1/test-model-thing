# HF Release Plan (Do NOT publish yet)

This is a plan only. **No artifacts are published, no tokens are committed/pushed.**

## 1) Checkpoint to release
| Item | Value | Notes |
| --- | --- | --- |
| **Banked incumbent (reference)** | `models/tmt_t094747.npz` | CE **1.1741** per roadmap/baseline (`RESULTS.md` 2026-09-23). Has `.sha256` alongside. |
| **Fork's own arms** | On private hub **`bakali1/tmt-checkpoints`** | Colab/HF durability layer; arms may be superseded by from-scratch runs. Salvaged/local copies remain historical. |
| **Default rolling checkpoint** | `tmt.npz` (root) | Hard-linked to newest tagged save; not necessarily the cleanest public candidate. |

**Action:** Prefer a clean, tagged checkpoint (e.g. `tmt_t*.npz`) with known ruler scores. Record its git SHA and training config.

## 2) Corpus
- **Training corpus:** As used for the baseline (wiki+TinyStories mix under `data/wikipedia_clean/` per upstream). 
- **Held-out/eval:** `data/eval/tinystories_valid.txt`, `data/eval/wiki_tinystories_005.txt`, `data/eval_shift/tinystories_v2gpt4_valid.txt`, Gutenberg novels for cross-domain — **not necessarily bundled**; document provenance via `fetch_eval_data.py`.
- **Note:** Corpora are **gitignored and local-only**; do not re-host full training corpora unless rights/size allow. Reference provenance/sha256s and provide fetch scripts instead.

## 3) Flags / training config (to document)
Capture at release time:
- Architecture: `dim=512, layers=16` (4.5M params)
- Optimizer: AdamW (engine form with per-role LR/WD as used)
- Init: note P4 init if used (geomspace time-scales) and P6 (WD on decay logits) impact
- Precision: fp32 (torch port)
- Key features: ring/tail/trace credit (inference has **zero** extra params), P4+P6 settings if released checkpoint uses them
- Ruler settings: slices/bytes used for reported CE (e.g. `tools/eval.py ... --slices 3 --bytes 20000`)

## 4) License
**MIT** (© 2026 jrz97619761) — preserve upstream license; note fork modifications. Include `LICENSE.md`.

## 5) Model card text (draft)
```markdown
# Model Card: TMT (Test-Model-Thing) — fork release candidate

## Model
- Type: Byte-level recurrent LM (JEPA-style latent prediction + RTUs)
- Params: 4.5M (dim=512, layers=16)
- Context/memory: Recurrent state (EMA-like) with decay; no fixed context window
- Training: Continual/test-time capable by design (see caveats)

## Performance (ruler)
- Held-out (tinystories_valid): CE 1.1741 ± 0.0415 nats/byte (baseline)
- Order-8 byte n-gram (byte-matched): 0.6512 nats/byte → **model is 0.56 nats/byte worse**
- Memory retention (P4+P6): 37.5%/17.1%/5.3%/8B/32B/128B; half-life 95.4B

## Limitations
- Coherent long-form generation limited at 4.5M / ~10^5 steps
- Memory readout decodable over ~12–128B depending on metric, **measured on the delayed-copy model**; earlier nulls were instrument artifacts. **The model card must not state this as a general property of the architecture**: on a continual-learning objective the state holds only the address and the content moves into the weights (`RESULTS.md` §22).
- Trained on local corpora (not included); see provenance
- Never trained above 4.5M params

## Usage
See README-FORK.md; run via `tools/eval.py` for ruler metrics.
```

## 6) Eval numbers & sha256
- **Eval numbers:** Report CE ± SE on held-out per `tools/eval.py` (in-domain, shift, cross-domain), plus n-gram comparison and memory readout summary if desired.
- **sha256:** Compute `sha256sum models/tmt_t*.npz` (and any released `.npz`). Publish alongside checkpoint. Verify with `checkpoint_hub.py` content checks.

## 7) Checkpoint format — weights-only conversion REQUIRED
**Critical:** TMT checkpoints (`.npz`) carry **Adam moments, ring buffers, tail EMA, and the decaytrace** in addition to weights. A public release **must not** require exposing optimizer state unnecessarily and must be unambiguous about contents.

- **Requirement:** Ship a **WEIGHTS-ONLY** checkpoint (model parameters only) **OR** ship a **conversion script** that strips optimizer/training state to weights-only and verifies parity/structure.
- **What to strip:** Adam `m/v` (or optimizer state), ring buffers, tail EMA traces, decaytrace/telemetry state, any hub metadata not needed for inference.
- **Verification:** After conversion, load in inference mode and confirm architecture matches; if CE/eval differs unexpectedly, document it. The bit-identical engine expects specific key structure — test loading.
- **Implementation note:** Use existing save/load paths (`train_gpu.py`, `checkpoint_hub.py`) as reference; add a `tools/convert_weights_only.py` if one does not exist.

## 8) Credential handling (CRITICAL — past incident)
**Past incident:** Two bundle generators wrote a live HF token to disk in plaintext. **This must never happen.**

- **Resolve at RUNTIME only.** Token must come from runtime environment/login, never written to files in repo, never embedded in committed/generated scripts in plaintext form persisted to disk as a secret, never passed on command line in ways that leak to logs unnecessarily.
- **Sources (preferred order):**
  1. `huggingface_hub` cached login from `hf auth login` on the machine doing the push (recommended) — `checkpoint_hub` should use cached creds.
  2. Environment variable read at runtime (e.g. `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`) injected by the orchestrator/CI secret store, **not** written to disk by our tools.
- **Rules:**
  - No code change may write tokens to disk.
  - Bundle generators must not bake tokens into committed artifacts; any generated Colab scripts that need auth must obtain tokens at runtime (or instruct user to login) — treat as secret.
  - Never log tokens.
  - Verify `.gitignore` covers any local token files; add guards if needed.
- **Reference:** `docs/plans/PLAN-2026-10-13-COLAB-HF-RUNS.md` §6.

## 9) Exact commands (do NOT execute)
```bash
# 1) Verify checkpoint and get sha256
sha256sum models/tmt_t094747.npz
.venv/bin/python tools/eval.py models/tmt_t094747.npz --slices 3 --bytes 20000

# 2) Create weights-only checkpoint (required) OR use conversion script
#    .venv/bin/python tools/convert_weights_only.py \
#      --input models/tmt_t094747.npz --output models/tmt_t094747_weights_only.npz

# 3) Authenticate at runtime (on pushing machine) — DO NOT write token to repo
#    hf auth login  # interactive, credentials cached by huggingface_hub

# 4) Push (example) — use huggingface_hub from env/cached login only
#    .venv/bin/python -c "
#    from huggingface_hub import HfApi
#    api = HfApi()  # uses cached token or HF_TOKEN env var at runtime
#    api.upload_file(path_or_fileobj='models/tmt_t094747_weights_only.npz',
#                    path_in_repo='tmt_t094747_weights_only.npz',
#                    repo_id='jrz97619761/test-model-thing' or '<org/repo>',
#                    repo_type='model')
#    "

# 5) Upload model card README.md and LICENSE.md as appropriate
```
**Note:** Replace repo_id with the agreed public repo. Do **not** hardcode tokens. If using Colab, inject tokens as session env vars only.

## 10) Release checklist
- [ ] Weights-only checkpoint created and tested for inference load
- [ ] `sha256sum` recorded in model card/release notes
- [ ] Ruler numbers (±SE) included
- [ ] n-gram comparison included (honest about 0.56 nats/byte gap)
- [ ] License and attribution preserved
- [ ] Credential handling verified (no tokens in repo/history)
- [ ] Corpora provenance documented (fetch scripts, not raw dumps)
- [ ] Limitations/caveats stated clearly
