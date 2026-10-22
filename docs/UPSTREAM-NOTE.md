# Note to the original author (polite, concrete)

Hi [jrz97619761](https://github.com/jrz97619761),

Thanks for publishing TMT under MIT and for saying "I really need other people to review and verify my work!" Your design (byte-level, latent prediction + RTUs/recurrent state, continual/test-time training) is exactly what we've been trying to understand and measure rigorously in this fork.

Below are concrete, checkable findings from our measurements. All numbers/locations come from this repo's append-only log (`RESULTS.md`) and code.

## 1) The read-path instrument bug and its fix
- **Bug:** In `tools/membench/model_backend.py:_ridge_score`, lambda selection had a **swapped tuple assignment** (`best_acc, best = lam, acc`), so only `lam=1.0` was ever considered and `chosen = best` became a **training accuracy** (effectively unregularized solve on a rank-deficient Gram matrix). Also `y = targets.astype(np.int64).ravel()` flattened `(n,4)` payloads so **only payload byte 0** was decoded while metric claimed "exact 4-byte payload matches" vs 2⁻³² chance. A third issue: read happened **after zero-padding** in `model_copy_readout` (`D4`).
- **Fix:** Correct assignment, choose λ on an **inner hold-out carved from training half**, fit **k independent readouts** for k-byte payloads, permute **samples** (not bytes) for the veto, and read **at the cue** (padding-free). Gate: `tests/check_ridge_readout.py` fails on pre-fix code, passes after.
- **Impact:** The claim *"the readout never beats its own control"* was an **artifact**; with a fixed instrument the readout **does** work. (`RESULTS.md` §D1/D4; `tools/membench/model_backend.py`, `tests/check_ridge_readout.py`)

## 2) P4+P6 capacity result (15× incumbent)
- **Setup:** P4 init (`np.geomspace` time-scales, reshuffled per layer) + P6 (weight decay on the **decay logits**).
- **Result:** **95.4 B half-life** vs incumbent **1.05 B**; retention **37.5% / 17.1% / 5.3% / 0.78%** at **8 / 32 / 128 / 512 B** vs incumbent **4.8% / 1.7% / 0.36% / 0.0007%** — **~15×** the incumbent's retention.
- **Eroder:** Weight decay on the decay logits is the eroder (**92.30 → 39.07 B** without P6), not training alone.
- **Refs:** `RESULTS.md` §0 and §D5 context; `docs/plans/PLAN-2026-10-21-CONTINUAL-LEARNING.md` §0; `gpu_port.py:1017-1021` (P4 init).

## 3) 1.26× ring credit speedup
- **Result:** Fused `nvrtc` ring-credit path is **1.26× end-to-end, bit-identical** (**26.72 vs 33.68 ms/step**, loss identical, `max|Δ|=0`). Default on CUDA; measured-and-losing candidates removed.
- **Refs:** `RESULTS.md`, `docs/optimization/OPTIMIZATION_TEST_STATUS.md`, `tail_credit.py`, `gpu_port.py`.

## 4) Order-8 byte n-gram beats the model by 0.56 nats/byte
- **Byte-matched rebuild:** `tools/ngram_ce.py` (original `lag5.py` lost) trains on first 8M bytes of `data/eval/wiki_tinystories_005.txt`, scored on the same ruler windows as the model (3×20k per eval file in `data/eval/`, 120k positions).
- **Result:** Order-8 byte n-gram (0 params) scores **0.6512** nats/byte; model scores **1.2094** nats/byte on same bytes — **model is 0.56 nats/byte worse (×1.74 perplexity)**. Optimum around r≈0.85–0.9; robust across most settings.
- **Refs:** `RESULTS.md` §6 (D2), `tools/ngram_ce.py`.

## 5) D5: state holds an addressable, persistent address and a decaying content
With fixed instruments (`tools/d5_probe.py`, padding-free, correct ridge, matched 16-symbol alphabet, per-sample veto):
- **Cue/address:** Trailing key decodes **65–165× over null**, holds **~27% at 128 B** (persistent address).
- **Content/payload:** Decodes **4.6× over null at 8–12 B**, falls to **1.5× at 128 B** (decaying but real). Δh retention curve and decodable payload agree in shape.
- **Conclusion:** Write is partly order-losing (content mixed) but **readout is not key-blind**; next lever is **write selectivity** (e.g. P3 gate), not read wiring. Capacity vs usable content paradox resolved.
- **Refs:** `RESULTS.md` §7 (D5), `tools/d5_probe.py`, `docs/plans/PLAN-2026-10-21-CONTINUAL-LEARNING.md` §0/§4.

**Note on architecture:** We verified the structure matches your design — `state = states*a + enc` with `enc` from a single embedding per byte and 16 layers reading one multi-scale EMA bank (512 distinct time-scales under P4; ~0.18% of params control decay). This matches `gpu_port.py:228-256`, `:1701-1707`, `numpy_port.py`.

Again, thank you for the upstream design — it's a great object of study. If anything above looks off or you'd like the exact commands to reproduce any of these measurements, I'm happy to share them.

— Fork maintainers
