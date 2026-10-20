# PAPER-NOTE — FlashLM: CPU-Native Language Models Trained From Scratch on Free-Tier Hardware

Source file: `/home/baku/Downloads/output.pdf` (200,298 bytes, 8 pages, PDF 1.5, produced by pdfTeX-1.40.27).
Read with the `pdf-reader` skill (`pdf_info.py`, `pdf_extract.py --pages all`, `pdf_render.py --pages all --dpi 170`, and visual reading of all 8 rendered pages). No code was modified; no GPU job was run; no network access was used.

---

## 1. Identification

| Field | Value |
|---|---|
| Title | **FlashLM: CPU-Native Language Models Trained From Scratch on Free-Tier Hardware** |
| Author | **Cheng Chang** (single author; no email, no affiliation block) |
| Affiliation | **NOT STATED** — no institution, laboratory, or department appears anywhere on the title page or in the body |
| Venue / publication status | **NOT STATED** — no journal, conference, workshop, arXiv id, DOI, or license-in-venue line. The paper reads as a self-published preprint/project report |
| Date (on paper) | **May 2026** (title page, under the GitHub URL) |
| Date (PDF metadata) | `creationDate = D:20260510205904Z`, `modDate = D:20260510205904Z` → 2026-05-10 20:59:04 UTC — consistent with "May 2026" |
| PDF metadata `title` | **empty string** |
| PDF metadata `author` | **empty string** |
| PDF metadata `subject` / `keywords` | **empty string** |
| PDF `creator` / `producer` | `LaTeX with hyperref` / `pdfTeX-1.40.27` |
| Code URL | `https://github.com/changcheng967/FlashLM` (stated as MIT-licensed; not fetched — no network access) |
| Page count | 8 (1 title/abstract + 2 body + 2 method/architecture + 1 experiments/results + 1 discussion/conclusion + 1 references) |
| Figures | **0** — every page reports `image_count = 0`; the paper contains no figures at all, only three tables |

Explicit note on metadata: the PDF's document metadata is **empty** (no title, no author, no subject, no keywords). The author name, date, and repository URL above are transcribed from the rendered title page, not from metadata.

---

## 2. The problem the paper attacks, in its own terms

The paper's framing, quoted from the abstract and §1:

- "The training of language models is overwhelmingly associated with GPUs. From GPT-2 to LLaMA, virtually every open-source model has been trained on CUDA hardware. This creates a barrier for researchers and hobbyists who lack access to GPU resources."
- The question the paper poses: *"what can be learned when training language models from scratch on free-tier CPUs?"*
- Constraints of the setting: **4 virtual CPU cores, 16 GB RAM, at zero cost**; the dataset is **TinyStories** (~550M tokens), "designed to test whether tiny models can produce coherent text."
- The paper's central puzzle, from §3.1.1: "After 21 architecture experiments failed to produce coherent text at 3–5M parameters, we analyzed the shared structure of these failures. Every experiment used token-level cross-entropy as the sole training objective—predict the next token, nothing else. This local objective gives the model no incentive to maintain representations useful for long-range coherence."
- Second sub-problem (§3.2.1): standard self-attention is O(T²) and on CPU "the overhead of these operations dominates at small model sizes"; the claim is that "CPU efficiency comes from large contiguous matrix operations, not from reducing operation count," so the fix must be an O(n) algorithm that MKL can vectorize, not a hand-written kernel.
- Third sub-problem (§7.2): perplexity and coherence diverge — "Our best-perplexity model at small scale, v7.4 (PPL 2.33), generates repetitive text… Conversely, v5 Thunderbolt (PPL 1.36, 29.7M params, 40h) is the only model that generates coherent text."

Framed as: *at 3–5M parameters and a 2-hour CPU budget, what training objective (not what architecture) gets a language model to coherent text?*

---

## 3. The method — full technical detail

### 3.1 Future Sentence Prediction (FSP), §3.1

**Motivation.** Token-level cross-entropy alone gives a "strong local gradient but no global planning incentive" (§7.1). FSP adds a sentence-level planning signal.

**Definition (§3.1.2).** At every r-th position (**r = 16**) the model predicts which tokens will appear in the next τ positions (**τ = 64**):

$$\mathcal{L}_{\text{FSP}} = \frac{1}{|\mathcal{P}|}\sum_{p \in \mathcal{P}} \mathrm{BCE}\big(f(h_p),\ \mathrm{BoW}(y_{p+1:p+\tau})\big) \tag{1}$$

where:
- $\mathcal{P} = \{0, r, 2r, \ldots\}$ is the set of prediction positions,
- $h_p$ is the hidden state at position $p$,
- $f$ is "a linear projection into vocabulary space",
- $\mathrm{BoW}(\cdot)$ creates a **binary multi-hot vector** indicating which tokens appear in the future window,
- **BCE uses a positive weight of 50** ("to handle the extreme class imbalance").

**Total loss (§3.1.2):**

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \alpha \cdot \mathcal{L}_{\text{FSP}}, \qquad \alpha = 0.1 \tag{2}$$

**FSP head.** "The FSP head consists of one $d \times d$ linear projection (**65K parameters for $d = 256$**), sharing the output embedding matrix. This is 1.7% of the total 3.74M parameter model."
(Check: $256 \times 256 = 65{,}536$ params; $65{,}536/3.74\text{M} = 1.75\%$, consistent with the stated 1.7%.)

### 3.2 CumMix: CPU-native token mixing, §3.2

**Motivation (§3.2.1).** "We benchmarked a single attention layer at 2,062 µs on AMD EPYC 7B13, compared to a single MKL-optimized matrix multiply of comparable size at 218 µs." ⇒ design rule: "CPU efficiency comes from large contiguous matrix operations, not from reducing operation count."

**Mechanism (§3.2.2) — four steps, per token $t$:**

1. **Compress** (project $d \to k$, ReLU):
   $$c_t = \mathrm{ReLU}\!\left(W_c \cdot \mathrm{Norm}(x_t)\right)$$
2. **Accumulate** (cumulative sum over the sequence):
   $$g_t = \sum_{i=1}^{t} c_i$$
3. **Mix** (learned mixing matrix in code space):
   $$m_t = W_m \cdot \mathrm{Norm}(g_t)$$
4. **Expand** (project back to full dimension, with residual):
   $$\hat{x}_t = x_t + W_e \cdot m_t$$

**Cost.** "The computational cost per layer is **O(Tdk)** for $k \ll d$, compared to **O(Td² + T²d)** for attention. For our configuration (**d = 256, k = 32, T = 256**), CumMix costs **136 µs** per layer versus **2,062 µs** for attention—a **15× reduction**."

**Why cumsum is cheap.** "The cumulative sum is fully parallelizable (prefix sum), highly optimized in MKL, and costs only **11 µs for T = 256**. Each position sees the sum of all previous compressed representations, creating a running summary of the sequence. The compression bottleneck ($d \to k$) forces the model to learn *what* to aggregate."

**Incremental generation (§3.2.3).** During autoregressive generation CumMix keeps a running sum instead of recomputing:

$$g_{t+1} = g_t + \mathrm{ReLU}\!\left(W_c \cdot \mathrm{Norm}(x_{t+1})\right) \tag{3}$$

"This gives **O(dk) per token per layer, independent of context length**." (Note: the per-token cost of the full layer also includes the $W_m$ ($k\times k$) and $W_e$ ($k\times d$) products; the paper's O(dk) omits the $k^2$ term, which is dominated when $d \gg k$.)

### 3.3 The v11 architecture (§4, Table 1)

Table 1 is captioned "**v11 CumMix architecture: all components are novel.**" Standard vs. FlashLM v11:

| Component | Standard | FlashLM v11 |
|---|---|---|
| Normalization | RMSNorm ($p=2$) | **PowerNorm**: learnable $L_p$ norm, $p$ is trainable |
| Position encoding | RoPE / sinusoidal | **CumStepPos**: positions as cumulative random walk |
| Token mixing | Self-attention | **CumMix**: compress→cumsum→mix→expand |
| Feed-forward | SwiGLU (3 matmuls) | **HarmonicFFN**: $h+\sin(\omega h+\phi)$, 2 matmuls |
| Loss | Cross-entropy | **TACE**: frequency-weighted CE + learned temperature + FSP |
| Optimizer | AdamW | **DualMomAdam**: dual momentum + MACD crossover |

**PowerNorm (§4.1).** Generalizes RMSNorm to a learnable $L_p$ norm:

$$\mathrm{PowerNorm}(x) = \frac{x}{\left(\frac{1}{d}\sum_{i=1}^{d} |x_i|^p + \epsilon\right)^{1/p}} \cdot w \tag{4}$$

"where $p$ is parameterized as $p = \exp(\log p)$ with **$\log p$ initialized to 0.35 (near $p \approx 1.4$)** and **clamped to [0.5, 3.0]**."
($e^{0.35} = 1.4191$, matching the stated "$p \approx 1.4$".)

**CumStepPos (§4.2).** "Instead of sinusoidal curves or learned lookup tables, positions are a cumulative random walk":

$$\mathrm{pos}_t = \sum_{i=1}^{t} \theta_i \tag{5}$$

"where $\theta_i \in \mathbb{R}^d$ are learned step vectors initialized with **$\sigma = 0.02$**."

**HarmonicFFN (§4.3).** "Standard SwiGLU uses three matrix multiplications. HarmonicFFN uses two":

$$\mathrm{FFN}(x) = x + W_{\text{down}}\big(h + \sin(\omega \odot h + \phi)\big) \tag{6}$$

"where $h = W_{\text{up}}(\mathrm{Norm}(x))$ and $\omega, \phi \in \mathbb{R}^{d_{ff}}$ are learned frequency and phase. The identity shortcut $h + \sin(\cdot)$ ensures gradient flow is never blocked. With 2 matmuls instead of 3, we can use a wider **$d_{ff} = 768$** at similar parameter count."

**TACE — Token-Adaptive Cross-Entropy (§4.4).** Two modifications to CE.

*Frequency weighting* — "Rare tokens receive higher loss weight":

$$w_i = \frac{1/(\sqrt{f_i} + 0.01)}{\mathrm{mean}\big(1/(\sqrt{f_j} + 0.01)\big)} \tag{7}$$

"where $f_i$ is the token frequency in the training corpus."

*Learned temperature*:

$$\mathcal{L}_{\text{TACE}} = \frac{1}{N}\sum_i w_{y_i} \cdot \mathrm{CE}\big(\mathrm{logits}_i/\tau,\ y_i\big), \qquad \tau = \exp(\log \tau) \tag{8}$$

**DualMomAdam (§4.5).** "Standard Adam uses a single first-moment estimate. DualMomAdam maintains two momentum buffers with different decay rates (**$\beta_{\text{fast}} = 0.9$, $\beta_{\text{slow}} = 0.99$**) and uses their crossover as an amplification signal":

$$\mathrm{update} = \Big(1 + 0.5 \cdot \tanh\big(5(\hat{m}_{\text{fast}} - \hat{m}_{\text{slow}})\big)\Big) \cdot \hat{m}_{\text{fast}} \tag{9}$$

"When fast momentum exceeds slow momentum (positive gradient trend), updates are amplified up to **1.5×**. When trends reverse, updates are dampened to **0.5×**."

### 3.4 Experiments setup (§5)

**Data (§5.1).** "All experiments use **TinyStories V2** [Eldan and Li, 2023] with a **4,096-token BPE vocabulary trained on the corpus**. The dataset contains **~550M training tokens**. Sequences of length **256** are sampled randomly."

**Hardware (§5.2).** "Free-tier **Lightning AI studio**: **4 vCPU (AMD EPYC 7B13, Zen 3, SMT), 16 GB RAM**. **PyTorch 2.8.0 with MKL**. Setting **MKL_DEBUG_CPU_TYPE=5** provides **~28% matmul speedup**."

**Training configuration (§5.3).** "All 2-hour experiments use: **batch size 4 with gradient accumulation 8 (effective batch 32)**, **learning rate 5 × 10⁻⁴ with cosine decay and 200-step warmup**, **weight decay 0.1**, **gradient clipping 1.0**, **dropout 0.1**."

**Evaluation (§5.4).** "Validation perplexity is computed on **20 random batches** from the held-out validation split. Generation samples use **nucleus sampling ($p = 0.9$, temperature = 0.8$)** with **three fixed prompts**."

**Compute accounting (derived, not stated).** The time budget is wall-clock, not steps: at the reported 2,750 tok/s, 2 hours ≈ **19.8M tokens ≈ 2,417 optimizer steps** (effective batch 32 × seq len 256 = 8,192 tokens/step) ≈ **0.036 epochs** of the 550M-token corpus. The paper states neither the step count, the token count, nor the seeds — see §6/§7.

### 3.5 What is NOT specified (implementation-critical gaps)

- **v11 is never evaluated.** Table 1 defines v11, but **no v11 row exists in Table 2**; no perplexity, no coherence, no tok/s is reported for the "fully novel architecture" the abstract advertises.
- FSP head shape: "one $d\times d$ linear projection (65K parameters for $d=256$), sharing the output embedding matrix" is internally ambiguous — a $d\times d$ projection cannot itself produce a 4,096-dim vocabulary logit vector unless it is followed by (or composed with) the shared embedding matrix. The exact composition is **NOT STATED**.
- $f_i$ in Eq. (7): whether it is a raw count, a probability, or a normalized frequency is **NOT STATED** (this changes $w_i$ by a constant, absorbed by the mean normalizer, but changes nothing else — however the interpretation of "rare tokens get higher weight" depends on it).
- Initialization of $\log\tau$ in Eq. (8) is **NOT STATED**.
- $\epsilon$ in Eq. (4) is **NOT STATED** numerically.
- $\beta_2$, the second-moment/denominator term, bias-correction procedure, gradient-clipping/weight-decay interaction, and how Eq. (9) composes with a learning rate are **NOT STATED** — Eq. (9) is a momentum scaler, not a complete Adam update.
- Whether the $p$ clamp in §4.1 is applied to $p$ or to $\log p$ (and how it interacts with the exponential reparameterization) is **NOT STATED**.
- How $\mathrm{pos}_t$ is combined with token embeddings (added? concatenated? inside Norm?) is **NOT STATED**; how positions beyond the trained length (T=256) are produced at generation time is **NOT STATED**.
- Which `Norm` is used inside CumMix (RMSNorm? PowerNorm? LayerNorm?), and over which axis, is **NOT STATED**; likewise the Norm inside HarmonicFFN.
- Number of layers, $d_{ff}$ for the attention-based v10 runs, attention head count, pre- vs post-norm placement, and dropout placement for the v10 architecture: **NOT STATED** (only "identical 4-layer transformer architecture" in §6.2, and $d=256$ implied by the FSP head and the latency table).
- Evaluation batch size, number of validation tokens, and whether PPL is computed as exp(mean NLL) over pooled tokens or a mean of per-batch exponentials: **NOT STATED**.
- BPE tokenizer implementation, merges count, special tokens, and normalization: **NOT STATED**.
- Whether TinyStories validation is the official split or a held-out slice: **NOT STATED**.
- Seeds, number of repeats, and error bars: **NOT STATED**.

---

## 4. Every quantitative claim, with baseline and exact metric

Metric key: **Val PPL** = validation perplexity (unitless, lower is better) on the TinyStories validation split, 20 random batches. **tok/s** = training throughput in tokens per second on 4 vCPU. **Latency** in microseconds (µs) per layer/block on AMD EPYC 7B13. **Coherent?** = subjective yes/no/partial judgment by the authors (no rubric stated).

### 4.1 Headline claims

| # | Claim (exact) | Baseline compared against | Metric + unit | Where |
|---|---|---|---|---|
| 1 | "improved perplexity by **2.5×** (**25.08 to 10.24**)" | v10 base (Attention, no FSP), 3.74M params, 2 h | Val PPL (25.08 → 10.24) | Abstract; §6.2; §8 |
| 2 | "with only **1.7% parameter overhead**" | 3.74M total model | % of params (65,536 FSP head params = 1.75% of 3.74M) | Abstract; §3.1.2 |
| 3 | "**21 architecture-only changes failed to match this gain**" / "never achieved more than marginal improvements over the baseline" | v10 base, 25.08 | count of experiments; Val PPL | Abstract; §1; §7.1 |
| 4 | "**15× lower latency per layer**" | Attention layer, 2,062 µs | µs per layer (136 µs vs 2,062 µs; ratio 15.16) | Abstract; §3.2.2; §6.3 |
| 5 | "**3.6× lower latency per full block**" | Attention + FFN, 2,672 µs | µs per block (745 µs vs 2,672 µs; ratio 3.59) | §6.3 |
| 6 | "cumsum over T=256 costs only **11 µs**" | MKL matmul 256×256, 218 µs | µs | §3.2.2; Table 3 |
| 7 | "replaces O(n²) attention with O(n) cumulative sum" | attention O(Td² + T²d) | asymptotic complexity; CumMix O(Tdk) | Abstract; §3.2.2 |
| 8 | "O(dk) per token per layer, independent of context length" | — | asymptotic per-token cost | §3.2.3 (Eq. 3) |
| 9 | "This allows **6 CumMix layers** in the same compute budget as **3 attention layers**" | 3 attention full blocks = 3 × 2,672 = 8,016 µs | µs of wall-clock (6 CumMix blocks = 6 × 745 = 4,470 µs) | §6.3 |
| 10 | "A fully novel architecture (**3.66M parameters**)" | — | parameter count (abstract's number) | Abstract |
| 11 | "This is **1.7%** of the total **3.74M** parameter model" | — | params (note: 3.66M in abstract vs 3.74M here) | §3.1.2 |
| 12 | "the **~28% matmul speedup**" from `MKL_DEBUG_CPU_TYPE=5` | default MKL dispatch | % matmul speedup | §5.2 |
| 13 | "hand-optimized code was **2× slower** than Python calling MKL" (v6.1 custom C kernels) | Python + MKL | relative latency | §7.3 |
| 14 | "**30+ experiments**" / "Across 30+ experiments" | — | experiment count | Abstract; §1; §8 |
| 15 | "**~550M tokens**" TinyStories training corpus | — | token count | §1; §5.1 |
| 16 | "**4 virtual CPU cores, 16 GB RAM**, at zero cost" | — | hardware | §1; §5.2 |
| 17 | "MKL_DEBUG_CPU_TYPE=5 … provides ~28% matmul speedup" | — | % | §5.2 (dup. of #12) |
| 18 | "the dominant factor in small-scale language model training is not the architecture but the training objective" | — | qualitative, supported by #1 and #3 | §1; §7.1 |
| 19 | "Below **~10M parameters**, models lack the capacity for this [narrative state]" | — | parameter threshold (hypothesis, unquantified) | §7.2 |
| 20 | "our only coherent model (perplexity **1.36**) required **29.7M parameters and 40 hours**" | v10 base 3.74M / 2 h | Val PPL, params, hours | Abstract; §7.2; §8 |

### 4.2 Full results table (Table 2) — every number

| Version | Architecture | Params | Time | Val PPL | tok/s | Coherent? |
|---|---|---|---|---|---|---|
| v5† | Ternary recurrence | 29.7M | 40h | **1.36** | — | Yes |
| v4-Large* | Ternary Bolt | 16.8M | 9h | 6.11 | — | Yes |
| v7.4 | Gated DeltaNet + SWA | 6.6M | 2h | 2.33 | 2,000 | No |
| v10 FSP | Attention + FSP | 3.74M | 2h | 10.24 | 2,750 | Partial |
| v5.2 | Attention + RoPE | 5.0M | 2h | 10.56 | 2,750 | No |
| v4 | Ternary Bolt | 4.3M | varies | 15.05 | — | No |
| v10 base | Attention (no FSP) | 3.74M | 2h | 25.08 | 2,750 | No |

Footnotes as printed: "† indicates training beyond the 2-hour budget." / "*v4-Large trained by community on 24-core/256 GB RAM machine."

### 4.3 Latency table (Table 3) — every number

| Component | Latency (µs) | Notes (as printed) |
|---|---|---|
| MKL matmul (256×256) | 218 | Baseline |
| cumsum over T=256 | 11 | Nearly free |
| CumMix layer (d=256, k=32) | 136 | Our replacement |
| Attention layer (d=256) | 2,062 | Standard approach |
| CumMix + FFN | 745 | Full block |
| Attention + FFN | 2,672 | Full block |

---

## 5. Ablations — which component carries which claim

| Component | Ablation actually performed in the paper | Claim it carries | Verdict |
|---|---|---|---|
| **FSP auxiliary loss** | **Yes — the only real ablation in the paper.** v10 base (25.08) vs v10 FSP (10.24), "identical 4-layer transformer architecture, differing only in the addition of the FSP auxiliary loss" | The headline "loss matters more than architecture"; the 2.5× PPL claim | Supported *as a paired comparison*, but single-run and unseeded; confounded with everything else that differs between runs unless truly identical |
| **CumMix** | **No perplexity ablation at all.** Only a latency benchmark vs. an attention layer | The 15×/3.6× speed claim only | **Speed claim supported by the micro-benchmark; the quality claim is entirely unevidenced** — CumMix never appears in Table 2 |
| **PowerNorm** | **None** | Part of "all components are novel" | Unevidenced |
| **CumStepPos** | **None** | ditto | Unevidenced |
| **HarmonicFFN** | **None** | ditto | Unevidenced |
| **TACE (freq weighting + learned temperature)** | **None** — bundled with FSP in the same "TACE" row of Table 1 | ditto | Unevidenced; TACE's individual contributions are not separable from each other or from FSP |
| **DualMomAdam** | **None** | ditto | Unevidenced |
| **v11 as a whole** | **None** — no v11 row in Table 2 | Contribution #3 ("A fully novel CPU-native architecture") | Unevidenced; the paper's third contribution has no result attached |
| **r = 16, τ = 64, α = 0.1, BCE pos_weight = 50** | **None** | FSP's hyperparameters | Unevidenced; no sensitivity study |
| **The "21 architecture-only changes"** | Reported only as a count; **7 of 30+ experiments are tabulated**; no per-experiment PPL list | Contribution #4 ("Systematic negative results") | Largely unevidenced; the "21 failures" table is not in the paper |
| **v5.2 (Attention + RoPE)** | Present in Table 2 but never discussed | — | Read as an architecture-only change that *does* match the FSP gain (see §7) |
| **v7.4 (Gated DeltaNet + SWA)** | Present in Table 2 but never discussed in the ablation context | — | Read as an architecture-only change that *beats* the FSP gain on PPL (see §7) |

---

## 6. Limitations

### 6.1 Stated by the authors

1. **Scale dominates at this regime.** "Our only coherent model (perplexity 1.36) required 29.7M parameters and 40 hours, suggesting that model scale dominates architecture choice at this regime." (§Abstract, §8) / "At 3–5M parameters in 2 hours, coherence remains elusive."
2. **Perplexity ≠ coherence.** "Our best-perplexity model at small scale, v7.4 (PPL 2.33), generates repetitive text. It has learned to predict common story patterns accurately but cannot construct novel narratives." (§7.2)
3. **A capacity hypothesis, explicitly labelled a hypothesis.** "We hypothesize that coherence requires sufficient model capacity to maintain a 'narrative state'… Below ~10M parameters, models lack the capacity for this state, regardless of architecture or training objective." (§7.2) — note this "regardless of architecture or training objective" is asserted, not tested at the ~10M boundary (the only model above ~10M is v5 at 29.7M/40 h and v4-Large at 16.8M/9 h on different hardware).
4. **The 10M threshold is untested.** No run exists between 6.6M (v7.4) and 16.8M (v4-Large), and both of those are flagged as not-coherent / community-trained.
5. **Ternary quantization loses at small scale.** "Earlier versions of FlashLM explored ternary architectures (v4–v6), but we found that at 3–5M parameters, the quantization overhead outweighed the efficiency gains." (§2.4)
6. **Hand-written kernels lose.** "Our experiments with custom C kernels (v6.1) revealed that hand-optimized code was 2× slower than Python calling MKL." (§7.3) — leading to their three principles: maximize MKL utilization; replace O(n²) with O(n); reduce algorithmic complexity, not implementation.
7. **Partial coherence for the best small model.** v10 FSP is marked "Partial" for coherence, not "Yes".
8. **Mixed comparability is disclosed for v4-Large only** (trained by the community on a 24-core/256 GB machine) and for v5† (beyond the 2-hour budget).

### 6.2 Visible from the numbers but NOT stated by the authors

1. **The paper's own Table 2 contradicts its headline.** v7.4 (Gated DeltaNet + SWA, 6.6M, 2 h) reaches **PPL 2.33** versus v10 base's **25.08** — a **10.8×** perplexity improvement from an *architecture-only* change, larger than FSP's 2.45×. The claim that 21 architecture-only experiments "never achieved more than marginal improvements over the baseline" is not supported by the table printed two pages earlier.
2. **v5.2 (Attention + RoPE, 5.0M, 2 h) = 10.56 PPL**, i.e. adding RoPE alone gets essentially the same PPL as the FSP model (10.24) — but with 1.33M more parameters, and it is *never mentioned in the discussion*. Adding RoPE is exactly an architecture change, so the "loss > architecture" ordering rests on parameter count, not on a controlled comparison.
3. **The v10 base / v10 FSP pairing is not clean.** v10 base is "Attention (no FSP)" while v5.2 is "Attention + RoPE" — the paper never states whether v10 base and v10 FSP use RoPE. If v10 base is the no-RoPE variant, the 2.5× conflates "adding RoPE" with "adding FSP".
4. **The comparison runs are ~2,417 steps / ~19.8M tokens (0.036 epochs).** At that budget a 4-layer 3.74M model is nowhere near converged; PPL 25.08 vs 10.24 at ~2.4k steps is a very early-training measurement, and single-run, single-seed, with **no error bars and no repeats** — yet the paper draws a 2.5× headline from it. (The v7.4 and v5.2 numbers in the same table are equally early.)
5. **The FSP head is 1.7% of params but the loss is 4,096-way BCE at 1/16 of positions** — a per-token compute overhead not quantified; the paper claims only "negligible training overhead", unsupported by a measurement, and the reported tok/s is *identical* (2,750) for v10 base, v10 FSP and v5.2, which is implausible for three different models (a 5.0M model cannot match a 3.74M model's throughput exactly).
6. **The "6 CumMix layers for 3 attention layers" claim is not what the numbers say.** 3 attention blocks = 8,016 µs; 6 CumMix blocks = 4,470 µs (56% of the budget) — the table implies ~10.8 CumMix blocks fit in the budget of 3 attention blocks. The claim is conservative in the paper's favour but is arithmetically inconsistent with Table 3.
7. **The 15× speedup is measured against a *non-cached* attention layer and a single T=256 point.** No scaling curve in T is given (which is where O(T²) vs O(T) actually bites), and no comparison against a KV-cached attention layer, which is the only comparison that matters for the incremental-generation claim in §3.2.3.
8. **CumMix's "running summary" is a *sum*, not a decayed state.** With ReLU-compressed $c_t \ge 0$ and no decay term, $g_t$ grows without bound in magnitude (only the subsequent Norm rescales it) and, critically, **order information is destroyed**: $\sum_{i\le t} c_i$ is permutation-invariant over the prefix. A plain cumulative sum of compressed tokens is therefore much weaker than attention/recurrent models with order-sensitive updates, and the paper never discusses this. (Related: no $W_m$ nonlinearity is specified between "mix" and "expand" — if $W_m$ is linear, the whole mix step is a linear function of the order-free prefix sum.)
9. **A byte-level caveat for the reader:** the token-mixing story is byte-agnostic, but with a 4,096-token BPE vocabulary on ~550M tokens the vocabulary covers the corpus; at byte level (256 symbols) the "compression bottleneck forces the model to learn what to aggregate" argument changes character, and BoW-over-future-bytes is a much weaker target than BoW-over-future-words.
10. **No test-set results, no downstream benchmarks, no generation samples.** Only validation PPL, subjective coherence, and latency. The three prompts and their text are not given, so the coherence judgments are not independently checkable.
11. **`MKL_DEBUG_CPU_TYPE=5` is a debugging override**, not a supported performance knob; it forces MKL to assume a fixed CPU type and can produce illegal instructions on machines lacking the assumed ISA, and it also masks the real dispatch behavior of a reproduction. Presenting it as a ~28% speedup method is a reproducibility hazard.
12. **The abstract's parameter count (3.66M) contradicts §3.1.2 and Table 2 (3.74M)** for what appears to be the same model family, with no explanation of the difference.

---

## 7. Red flags for a reproduction

1. **The third contribution has no experiment.** "A fully novel CPU-native architecture where every component is designed from scratch" (contribution #3) and Table 1's v11 — no v11 result appears anywhere in the paper. Every one of PowerNorm, CumStepPos, HarmonicFFN, TACE, DualMomAdam is *proposed but never measured*.
2. **The "30+ experiments / 21 failures" claim is unverifiable from the paper.** Only 7 experiments are tabulated, and 4 of the 7 (v5, v4-Large, v4, v5.2) are from *earlier versions* of the project (§2.4 says v4–v6 were the ternary line), not from the v11 line. The failure list — the claimed contribution #4 — is not published.
3. **The headline 2.5× is a single unseeded run of an under-trained model, and the paper's own table contains better architecture-only results.** See §6.2 items 1–4. No seed, no variance, no repeats, no steps, no tokens seen.
4. **Three rows in Table 2 share the exact same tok/s (2,750) across different model sizes** (3.74M base, 3.74M FSP, 5.0M RoPE) — a number that is implausibly identical and suggests a copied/rounded/defaulted value rather than a measurement.
5. **The FSP head description is self-contradictory** ("one $d\times d$ linear projection … sharing the output embedding matrix", while $f$ must map into a 4,096-dim vocabulary space). The composition that actually determines what is trained is unstated.
6. **The optimizer is not fully specified.** Eq. (9) is a momentum *scaler*; there is no second-moment term, no $\epsilon$, no bias-correction definition for $\hat m$, no learning-rate application, no statement of how weight decay 0.1 and clipping 1.0 combine with it. The word "Adam" in the name is not supported by the equation.
7. **Clipping of $p$ is ambiguous** (clamp $p$ or clamp $\log p$?) and the clamp would make the gradient of $p$ zero at the boundary — a numerically delicate detail left undefined.
8. **"MACD crossover" is a misnomer.** Eq. (9) is a $\tanh$ of the *difference* of two momentum estimates, not a crossover detector (which would require a sign change and hysteresis). A reader following the prose will implement something different from the equation.
9. **Hidden-state / Norm ambiguity throughout CumMix, PowerNorm, and HarmonicFFN** (which Norm, over which axis, pre- vs post-). The $\epsilon$ in Eq. (4) is unvalued.
10. **Latency claims are single-point, methodologically thin.** No warmup, no iteration count, no thread-count statement, no measurement of the *training* step (forward+backward) latency, and no sweep over $T$ — which is the only axis on which the O(T²)→O(T) claim is meaningful. The 2,062 µs attention-layer figure and the 218 µs "comparable size" matmul are not tied to an identical FLOP count, so the 15× is a comparison of two differently-sized operations, not of two algorithms at equal work.
11. **The "sum of previous compressed representations" mixer is order-blind** (§6.2 item 8) — an implementer following the text will build a bag-of-prefix-tokens model and may be surprised by the quality.
12. **Mixed hardware in the results table** (v4-Large on a 24-core/256 GB machine by a third party) with a "Coherent? Yes" that is then contradicted in §7.2 ("v5 … is the only model that generates coherent text") — the paper simultaneously marks v4-Large coherent and calls v5 the only coherent model.
13. **Abstract/body parameter-count mismatch** (3.66M vs 3.74M).
14. **"2-hour budget" is wall-clock, so different architectures get different numbers of optimizer steps.** Comparing PPL across rows of Table 2 therefore compares different amounts of training, and the paper does not report steps per row. This is a direct confound for the "architecture doesn't matter / loss matters" conclusion.
15. **`MKL_DEBUG_CPU_TYPE=5`** — a debug-only env var used as a performance recommendation (§5.2) is unsafe for a general reproduction and likely to break on other CPUs.
16. **No released artefacts are verifiable from the paper**: no checkpoints, no logs, no config files, no sample generations, no tokenizer, no prompt list. The GitHub link is the only pointer and was not fetched (no network access).

---

## 8. Figures and tables — captions transcribed verbatim

**There are no figures in this paper.** The PDF has zero raster images on all 8 pages (`image_count = 0` everywhere); the only vector drawings are table rules. Therefore no axes, legends, or panel labels exist to transcribe, and no figure is essential to the argument — the entire argument is carried by three tables and by inline equations. The "essential" visual content is Table 2 (results) plus Table 3 (latency); the FSP argument itself has no diagram at all, which is itself a presentational gap for a mechanism this central.

### Table 1 (p. 4)
Caption: "**Table 1: v11 CumMix architecture: all components are novel.**"
Columns: `Component` | `Standard` | `FlashLM v11`. Rows: Normalization / Position encoding / Token mixing / Feed-forward / Loss / Optimizer. (Full contents in §3.3 above.) No axes — a categorical comparison table.

### Table 2 (p. 6)
Caption: "**Table 2: FlashLM experiment results on TinyStories. All models trained from scratch. † indicates training beyond the 2-hour budget.**"
Columns: `Version` | `Architecture` | `Params` | `Time` | `Val PPL` | `tok/s` | `Coherent?`.
Rows (as printed, with the paper's own bolding preserved in words: bold = v5†, v10 FSP, and their Params/Time/Val PPL/tok/s cells):
v5† Ternary recurrence 29.7M 40h 1.36 — Yes;
v4-Large* Ternary Bolt 16.8M 9h 6.11 — Yes;
v7.4 Gated DeltaNet + SWA 6.6M 2h 2.33 2,000 No;
**v10 FSP** Attention + FSP **3.74M** **2h** **10.24** **2,750** Partial;
v5.2 Attention + RoPE 5.0M 2h 10.56 2,750 No;
v4 Ternary Bolt 4.3M varies 15.05 — No;
v10 base Attention (no FSP) 3.74M 2h 25.08 2,750 No.
Footnote: "*v4-Large trained by community on 24-core/256 GB RAM machine."
No axes; the "axes" are the table columns (params, time, Val PPL, tok/s). Note the sort order is not monotone in any column (PPL ascends 1.36, 6.11, 2.33, 10.24, 10.56, 15.05, 25.08 — i.e. it is *not* sorted by PPL).

### Table 3 (p. 6)
Caption: "**Table 3: Per-layer latency on AMD EPYC 7B13 (PyTorch + MKL, d=256, T=256).**"
Columns: `Component` | `Latency (µs)` | `Notes`. Rows: MKL matmul (256×256) 218 "Baseline"; cumsum over T=256 11 "Nearly free"; CumMix layer (d=256, k=32) 136 "Our replacement"; Attention layer (d=256) 2,062 "Standard approach"; CumMix + FFN 745 "Full block"; Attention + FFN 2,672 "Full block".
No axes; the only implied independent variable is the component identity. There is **no** T-sweep, no d-sweep, and no k-sweep, which is the minimum needed to substantiate the O(T²)→O(T) claim.

---

## 9. What I would need to implement this

**Data pipeline**
- `data/tinystories.txt` (TinyStories V2) + a 4,096-token BPE tokenizer trained on that corpus (tokenizer impl/merges/specials must be chosen — not specified).
- A random 256-token sequence sampler over the 550M-token corpus; a held-out validation split.
- A token-frequency table $f_i$ over the 4,096-symbol vocabulary (for TACE Eq. 7).

**Model modules**
- `norm/powernorm.py` — Eq. (4) with learnable $p=\exp(\log p)$, $\log p$ init 0.35, clamp [0.5, 3.0], learnable scale $w$, an $\epsilon$ (value mine to pick), and an explicit decision on clamp-target.
- `pos/cumsteppos.py` — Eq. (5): a learnable step table $\theta \in \mathbb{R}^{T\times d}$ init N(0, 0.02²), a prefix-sum to produce `pos[t]`, and a rule for combining `pos` with the token embedding and for positions $t \ge T$.
- `mix/cummix.py` — Eqs. (compress/accumulate/mix/expand): `W_c (d→k)`, `W_m (k→k)`, `W_e (k→d)`, ReLU on $c_t$, a `cumsum` over the time axis, a `Norm` in code space, and the residual $\hat x_t = x_t + W_e m_t$. Needs a **training path** (parallel, `torch.cumsum` over T) and a **generation path** (Eq. 3, a persistent `(d → k)` running-sum buffer per layer, so per-token cost is O(dk) with no recompute).
- `ffn/harmonic_ffn.py` — Eq. (6): `W_up (d→768)`, learnable $\omega,\phi \in \mathbb{R}^{768}$, elementwise `sin(ω⊙h+φ)`, `W_down (768→d)`, outer residual `x +`.
- `model/flashlm.py` — the block stack (pre- or post-norm is my choice; the paper doesn't say), the `f` FSP head (ambiguity to resolve: a $d\times d$ projection composed with the tied output embedding), and the 3.74M/3.66M parameter target.

**Loss / training**
- `loss/tace.py` — Eq. (7) frequency weights, Eq. (8) frequency-weighted CE with a learnable temperature $\tau=\exp(\log\tau)$ (init mine to pick), plus the FSP branch Eq. (1) with a 4,096-way multi-hot target, BCE with `pos_weight = 50`, positions $\mathcal{P} = \{0,16,32,\dots\}$ and $\tau_{\text{window}} = 64$; combine with Eq. (2) $\mathcal{L} = \mathcal{L}_{\text{CE}} + 0.1\,\mathcal{L}_{\text{FSP}}$; decide whether TACE's weighted CE replaces $\mathcal{L}_{\text{CE}}$ or multiplies it.
- `optim/dual_mom_adam.py` — two Adam-style momentum buffers (β 0.9 / 0.99), bias-corrected $\hat m_{\text{fast}}, \hat m_{\text{slow}}$, the Eq. (9) `tanh` amplifier (range 0.5×–1.5×), a second-moment denominator, $\beta_2$, $\epsilon$, learning rate, decoupled weight decay 0.1, and grad-norm clipping 1.0 (all beyond Eq. 9 are my invention).
- `train/trainer.py` — batch 4, grad-accum 8 (effective 32), seq len 256, Adam-family optimizer, lr 5e-4 cosine with 200-step warmup, weight decay 0.1, clip 1.0, dropout 0.1, PyTorch 2.8.0 + MKL, and a **wall-clock 2-hour budget** (which means the LR schedule horizon and the number of steps are a function of throughput — a design decision the paper hides).

**Evaluation / benchmarking**
- `eval/ppl.py` — validation perplexity over 20 random batches (batch size and pooling rule to be chosen).
- `eval/generate.py` — nucleus sampling $p = 0.9$, temperature 0.8, three fixed prompts (prompts must be written by me), and a **coherence rubric** — the paper has none, and this is where reproduction will diverge most.
- `bench/latency.py` — per-layer latency on AMD EPYC 7B13 (or whatever CPU is available), d=256, T=256, k=32, PyTorch+MKL, with warmup and iteration counts (the paper gives none), covering: MKL matmul 256×256, cumsum over T=256, CumMix layer, attention layer, CumMix+FFN, attention+FFN. A T-sweep (T = 128/256/512/1024/2048/4096) is needed to actually show the O(T²)→O(T) claim, and a KV-cached attention comparison is needed to test the §3.2.3 generation claim.

**Artefacts to write, concretely**: 1 tokenizer file; 1 frequency-count file; 1 model file; 1 loss file; 1 optimizer file; 1 trainer; 1 eval script; 1 benchmark script; 1 prompt file; and (crucially) a **config per experiment version** (v10 base, v10 FSP, v11) plus a seed-control mechanism the paper entirely lacks.

---

## 10. 15-line summary of the core claim and the single most useful idea for a byte-level recurrent LM

1. FlashLM (Cheng Chang, May 2026, self-published preprint, MIT code at github.com/changcheng967/FlashLM) asks what can be learned training LMs from scratch on free-tier CPUs: 4 vCPU (AMD EPYC 7B13), 16 GB RAM, TinyStories V2 (~550M tokens, 4,096-token BPE), 2-hour budget.
2. Its headline claim: the **training objective matters more than the architecture** at 3–5M parameters.
3. The single supporting experiment: adding an auxiliary **Future Sentence Prediction (FSP)** loss to a 4-layer 3.74M-param attention model moved validation PPL from **25.08 to 10.24** (2.5× as claimed; 2.45× as computed) at a cost of **65K extra params (1.7%)**.
4. FSP, in full: at every r=16-th position, predict a **binary bag-of-words multi-hot** over the next τ=64 tokens (BCE, pos_weight 50), added to CE with α=0.1 (Eqs. 1–2).
5. The second contribution, **CumMix**, replaces attention with compress(d→k=32) → cumsum → mix → expand(k→d), claimed **15× lower per-layer latency** (136 µs vs 2,062 µs) and **3.6× per full block** (745 vs 2,672 µs) on EPYC 7B13, with O(dk) per-token generation via a running sum (Eq. 3).
6. The third contribution, a "fully novel" v11 architecture (PowerNorm, CumStepPos, HarmonicFFN, TACE, DualMomAdam; 3.66M or 3.74M params depending on where you read) is **specified in Table 1 but never evaluated** — no v11 result exists.
7. The paper's own Table 2 undercuts its thesis: architecture-only v7.4 (Gated DeltaNet+SWA) reaches PPL **2.33** and v5.2 (Attention+RoPE) **10.56**, both from the same 2-hour baseline of 25.08.
8. The strongest reproducible claim is therefore *not* "loss > architecture" but: **a cheap, orthogonal auxiliary prediction target is worth real perplexity at tiny scale and tiny compute**.
9. Coherence is the paper's real currency, and it is measured subjectively (three prompts, no rubric, no samples); only the 29.7M-param / 40 h v5 (PPL 1.36) and the community-trained 16.8M v4-Large (PPL 6.11) are marked coherent, and §7.2 calls only v5 coherent.
10. Everything is one unseeded run at ~2,417 steps / ~19.8M tokens (0.036 epochs) with no error bars, no step counts, and a wall-clock (not step) budget — the 2.5× is a single noisy early-training measurement.
11. Reproduction hazards: the FSP head's shape is self-contradictory, the optimizer equation lacks a second-moment term, "MACD crossover" is a $\tanh$ of a momentum difference, CumMix's prefix sum is order-blind, and `MKL_DEBUG_CPU_TYPE=5` is a debug override presented as a speed knob.
12. **The single most useful idea for a byte-level recurrent LM is FSP itself: an auxiliary "what will appear next" loss read directly out of the recurrent state.**
13. Concretely, for a byte-level RWKV/Mamba/DeltaNet-style model: at every r-th position, project the recurrent hidden state to a **256-way multi-hot vector** predicting which of the next τ bytes will occur (BCE, pos_weight to counter imbalance), and add it to next-byte CE with a small α (0.1 in the paper).
14. Why it suits a recurrent model specifically: a recurrent LM has exactly **one** place to store context — the state — and next-token CE never asks that state to be *usable later*; the FSP gradient forces the state to carry forward-looking content, which is precisely the "narrative state" the paper hypothesizes (and never demonstrates) as the capacity bottleneck.
15. Cheap, architecture-agnostic, and orthogonal to the state update — it needs no new mixing machinery, no per-position extra matmuls beyond one small head, and it costs ~one extra forward+backward over 1/16 of positions, so it can be dropped at inference; the paper's caution to carry over is that PPL gains did **not** translate into coherence at 3–5M params (v10 FSP = "Partial"), so measure coherence separately from perplexity.
