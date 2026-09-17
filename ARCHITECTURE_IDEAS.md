# TMT Architecture Ideas

Based on the project's core goals:
- JEPA-style latent-space prediction
- Internal state + recurrent trace units (RTUs)
- Byte input/output
- Continuous data streaming
- Test-time training / continual learning

---

## The Core Problem

The model's flagship demo is **closing quotes/brackets** — but the current architecture struggles because:
- A single state vector mixes everything: "open quote", grammar context, last few bytes — all jumbled together
- One decay parameter controls forgetting for the *entire* state — can't selectively remember the quote while forgetting prose
- There's no explicit "I'm inside a quote" flag

---

## Proposed Architecture: "Tagged RTU" (T-RTU)

Keep the JEPA + streaming + O(1) memory foundation. Add three targeted changes.

### Change 1: Multi-Slot RTU with Content Tags

Instead of one `states` vector per layer, use **K memory slots** (e.g., K=4), each a `dim`-dimensional vector.

```
Current:
  states:      [dim]
  decaytrace:  [dim]
  decay:       [dim]

T-RTU:
  states:      [K, dim]
  decaytrace:  [K, dim]
  decay:       [K]
```

Each slot has its own decay rate, so the model can **independently control** how fast each memory fades. One slot might hold "open quote", another holds "current grammar context" — they decay at different rates.

**O(1) preserved:** K is a fixed constant, not dependent on input length.

```python
# In Layer.__init__:
self.K = 4
self.states = nn.Parameter(torch.zeros(K, dim))
self.decaytrace = nn.Parameter(torch.zeros(K, dim))
self.decay = nn.Parameter(torch.zeros(K))  # per-slot decay

# In Layer.__call__:
decay = torch.sigmoid(self.decay)  # now shape [K]
state = decay[:, None] * self.states + x + dummy  # broadcast over slots
# output: states shape [K, dim], each slot evolves independently
```

### Change 2: Write Gates (Controlled Memory Updates)

Currently, `state = decay * states + x + dummy` always writes the new input into every slot. Add a **write gate** per slot:

```python
self.write_gate = nn.Parameter(torch.zeros(K))  # learned per-slot

# In forward:
write = torch.sigmoid(self.write_gate)  # [K]
state = decay[:, None] * self.states + (write[:, None] * (x + dummy))
```

The model can learn: "slot 0 gets every new byte, slot 1 only writes on quotes, slot 2 only writes on newlines." This is how you get **selective remembering** without attention.

### Change 3: Explicit Tag Projections

Add a small learned projection that detects structural markers *before* they enter the RTU:

```python
# In Layer:
self.tag_proj = nn.Linear(dim, K)  # maps current input to K tag scores

# In forward:
tags = torch.sigmoid(self.tag_proj(dummies[i]))  # [K] — what each slot should track
# Modulate write gate:
write = write * tags  # only write to slot k if tag k is active
```

This gives the model a mechanism to learn: "when you see `"` → activate tag 0 → write to slot 0 → slot 0 remembers until decay slowly fades it → decoder reads slot 0 and outputs `"`."

### T-RTU Forward Pass Summary

```
Input byte → Encoder → x (dim)
                ↓
            For each layer:
              tags   = sigmoid(tag_proj(dummy))     # [K]
              write  = sigmoid(write_gate) * tags   # [K]
              decay  = sigmoid(decay)               # [K]
              state  = decay[:,None] * states + write[:,None] * (x + dummy)
              output = sum over slots (or pass all slots)
```

---

## Why This Is Better for Each Project Goal

| Goal | Current | T-RTU |
|------|---------|-------|
| JEPA latent prediction | Single latent mixes everything | Slots specialize (syntax, semantics, structure) → cleaner latent |
| RTU memory | One decay for all | Per-slot decay → selective forgetting |
| Byte I/O | Unchanged | Unchanged |
| Streaming / O(1) | O(1) with fixed dim | O(1) with fixed K×dim — same asymptotic cost |
| Test-time training | All params update equally | Write gates teach model *what* to remember during chat |

---

## Other Architectural Ideas (Lower Priority)

### Adaptive Decay
Make decay depend on the input rather than a fixed per-layer parameter. When the model sees punctuation or structure markers, it could remember longer; when processing noise, forget faster.

### Hierarchical JEPA (Multi-Scale Prediction)
Predict at multiple time scales:
- Fast: next byte's embedding (current)
- Slow: "what will the latent be in N bytes" (forces the model to build forward-looking internal representations)

### Latent Dictionary / Codebook
Replace the continuous latent space with a discrete codebook (like VQ-VAE). The model picks from K latent codes. This forces compression and may lead to more interpretable, stable representations.

### Separate Memory Types
Two sets of trace arrays:
- **Facts**: episodic memory (what happened at position X) — decays quickly
- **Skills**: abstracted patterns (general rules) — persists

Facts decay; skills persist. Supports continual learning better.

### Latent Bookmarks
Explicit mechanism where the model can flag important moments and write to a protected memory region that resists decay and overwriting. Directly targets the quote/bracket closure problem.

### Self-Modifying Decay
Let the model adjust its own decay parameters during inference. If it detects a pattern (like an opening quote), it can reduce decay to remember longer. True test-time adaptation.

---

## Implementation Plan

### Minimum Viable Change
Modify only the `Layer` class in `main_torch.py`:
1. Reshape `states`, `decaytrace`, `decay` to `[K, dim]` or `[K]`
2. Add `write_gate: nn.Parameter(torch.zeros(K))`
3. Add `tag_proj: nn.Linear(dim, K)`
4. Update forward pass per T-RTU formula above

Everything else (Model, Encoder, Decoder, AdamW, save/load) stays the same — just adjust shapes.

**Estimated change: ~50 lines in `main_torch.py`**

### Testing Order
1. Verify T-RTU trains at all (loss should decrease)
2. Test quote/bracket closure in chat mode
3. Run benchmark to compare MCC
4. Compare memory usage vs current architecture
5. Try different K values (2, 4, 8, 16)

---

## Open Questions

- Should the decoder read all K slots or just one?
- How should `dummy` inputs be shaped with multi-slot? (Repeat K times, or project?)
- Does multi-slot actually help or does it just add parameters the model can't learn to use?
- Should write gates and tag projections share parameters?
