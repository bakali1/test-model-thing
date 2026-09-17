Here is a complete plan you can keep as the project roadmap. I organized it so you can implement and evaluate one architectural change at a time without losing track of what each experiment proves.

# TMT Tagged RTU Architecture and Experiment Plan

## 1. Project Goal

Improve TMT's persistent memory mechanism while preserving its main properties:

* JEPA-style latent-space prediction
* Persistent recurrent state
* RTU-based memory
* Byte-level input/output
* Continuous streaming
* O(1) memory with respect to sequence length
* Test-time training / continual learning

The main problem to solve is long-range structural information.

Examples:

```text
"hello world
           ↓
model must remember that a quote is still open

(
  lots of content
)
           ↓
model must remember that a bracket was opened
```

The current single-state RTU mixes:

* recent input
* syntax
* semantic context
* structural information
* short-term and long-term information

into one representation with limited control over what is retained.

The new architecture should allow the model to learn:

```text
what to remember
where to remember it
how long to remember it
what memory to read
```

---

# 2. Main Architectural Idea

Introduce multiple persistent latent registers instead of one RTU state.

Current:

```text
state: [dim]
decay: [dim]
```

Proposed:

```text
states: [K, dim]
decay:  [K, dim]        # preferred experimental version
```

where `K` is a small fixed constant such as:

```text
K = 2
K = 4
K = 8
K = 16
```

Because `K` is fixed:

```text
memory complexity = O(K × dim)
```

and therefore remains:

```text
O(1)
```

with respect to sequence length.

The slots are not explicitly assigned meanings.

The network should discover their roles.

Possible emergent behavior:

```text
slot 0 → structural state
slot 1 → local context
slot 2 → long-term state
slot 3 → prediction-related state
```

but this should not be hard-coded.

---

# 3. T-RTU Components

The new RTU should contain five main mechanisms.

## 3.1 Multiple latent registers

```text
state[0]
state[1]
...
state[K-1]
```

Every register contains a full latent vector:

```text
state[k] ∈ R^dim
```

---

## 3.2 Input-dependent write routing

Instead of always writing the current input to every register, calculate a write probability for every register.

```text
write = sigmoid(W_write(x))
```

Shape:

```text
[K]
```

This lets the model decide:

```text
write strongly to slot 0
write weakly to slot 1
don't write to slot 2
write moderately to slot 3
```

The gate must depend on the current input.

Do not use only a learned static scalar such as:

```python
self.write_gate = nn.Parameter(...)
```

for the main version.

A static gate only learns whether a slot is generally active.

An input-dependent gate can learn:

```text
ordinary text → low write
opening quote → high write
newline → high write
punctuation → different routing
```

---

# 4. Slot-Specific Write Projections

Do not send exactly the same representation to every slot.

Use a projection producing a candidate representation for every slot:

```text
candidate = W_write(x)
```

with output:

```text
[K, dim]
```

So:

```text
candidate[0] = W0(x)
candidate[1] = W1(x)
...
candidate[K-1] = WK(x)
```

This allows the slots to learn different representations.

The resulting update is:

```text
state_k =
    decay_k * state_k
    +
    write_k * candidate_k
```

---

# 5. Adaptive Decay

Each slot needs its own memory timescale.

Preferred first experimental version:

```text
decay: [K, dim]
```

rather than:

```text
decay: [K]
```

This preserves the ability for different latent dimensions inside the same slot to have different timescales.

For example:

```text
slot 0:
dimension 1 → fast
dimension 2 → fast
dimension 3 → medium
dimension 4 → slow

slot 1:
mostly long-term

slot 2:
mixed

slot 3:
mostly short-term
```

Use:

```python
decay = torch.sigmoid(self.decay)
```

so the effective decay remains between 0 and 1.

---

# 6. Read Routing

Writing is only half of the memory problem.

The model also needs to decide:

> Which memory should I use right now?

Calculate read scores:

```text
read_logits = W_read(x)
```

then:

```text
read = softmax(read_logits)
```

with:

```text
read.shape = [K]
```

The final latent representation becomes a weighted combination of the slots:

```text
latent =
    read_0 * state_0
    +
    read_1 * state_1
    +
    ...
    +
    read_K * state_K
```

This prevents all memory slots from being mixed together permanently.

The model can learn:

```text
current input → read structural memory
current input → read recent memory
current input → read long-term memory
```

---

# 7. T-RTU Forward Pass

The conceptual pipeline:

```text
input byte
    │
    ▼
encoder
    │
    ▼
x
    │
    ├───────────────┐
    │               │
    ▼               ▼
write router     read router
    │               │
    ▼               │
candidate states    │
    │               │
    ▼               │
┌──────────────┐    │
│ slot 0       │    │
│ slot 1       │    │
│ ...          │    │
│ slot K-1     │    │
└──────────────┘    │
    │               │
    └───────┬───────┘
            ▼
       weighted latent
            │
            ▼
     JEPA prediction
            │
            ▼
       byte decoder
```

Mathematically:

```text
candidate_k = W_k(x)

write = sigmoid(W_gate(x))

decay = sigmoid(D)

state_k =
    decay_k ⊙ state_k
    +
    write_k * candidate_k

read = softmax(W_read(x))

latent =
    Σ read_k * state_k
```

---

# 8. Do Not Add Everything at Once

The architecture should be developed through controlled experiments.

The experiment sequence is critical.

Do not immediately add:

* multi-slot memory
* adaptive decay
* bookmarks
* codebooks
* hierarchical JEPA
* self-modifying parameters

at the same time.

Otherwise, an improvement or failure becomes impossible to attribute.

---

# 9. Experiment 0: Baseline

Freeze the current architecture.

Record:

```text
parameter count
training loss
validation loss
training speed
inference speed
memory usage
quote closure accuracy
bracket closure accuracy
long-range recall accuracy
benchmark MCC
```

Save the exact model checkpoint and training configuration.

This becomes the control.

---

# 10. Experiment 1: Multi-Slot RTU

Change only:

```text
state → [K, dim]
```

Start with:

```text
K = 4
```

Keep everything else as similar as possible.

Initially use simple slot aggregation if necessary.

Goal:

Determine whether multiple persistent states improve memory without additional routing.

Questions:

```text
Does loss still decrease?
Does training remain stable?
Does quote closure improve?
Does long-range recall improve?
```

---

# 11. Experiment 2: Input-Dependent Write Gates

Add:

```text
write = sigmoid(W_gate(x))
```

Each slot receives a dynamic write probability.

Keep the rest of the architecture unchanged.

Goal:

Test whether selective writing improves memory retention.

Especially test:

```text
opening quote
closing quote
brackets
parentheses
newlines
punctuation
ordinary prose
```

---

# 12. Experiment 3: Slot-Specific Write Projections

Add:

```text
candidate = W_write(x)
```

with output:

```text
[K, dim]
```

Now each slot receives a different candidate latent representation.

Goal:

Test whether slots become more specialized.

---

# 13. Experiment 4: Read Routing

Add:

```text
read = softmax(W_read(x))
```

and use weighted slot aggregation.

Goal:

Allow the model to dynamically select which persistent memory is relevant.

This is the first version that gives the model both:

```text
WRITE CONTROL
+
READ CONTROL
```

---

# 14. Experiment 5: Per-Dimension Slot Decay

Compare:

```text
version A:
decay = [K]

version B:
decay = [K, dim]
```

Goal:

Determine whether allowing multiple timescales inside each slot produces better memory behavior.

Analyze both:

```text
performance
parameter count
stability
learned timescales
```

---

# 15. Experiment 6: Different K Values

Run:

```text
K = 1
K = 2
K = 4
K = 8
K = 16
```

Treat:

```text
K = 1
```

as a particularly important control because it approximates the original architecture.

Measure:

```text
accuracy
loss
parameters
VRAM/RAM
training time
inference time
long-range memory
```

Do not assume a larger K is better.

The goal is to determine whether there is an efficient memory capacity.

---

# 16. Memory Stress Benchmark

Create a benchmark specifically for persistent memory.

Normal language-model loss is not enough.

The benchmark should measure whether information survives over long distances.

---

## 16.1 Delayed Symbol Recall

Example:

```text
The secret symbol is XQ7P.

[random distracting data]

What was the secret symbol?
```

Distances:

```text
10 bytes
100
1,000
10,000
100,000
1,000,000
```

Record accuracy.

---

## 16.2 Quote Closure

Example:

```text
"hello
```

followed by a long stream.

Measure whether the model predicts:

```text
"
```

at the appropriate point.

Test increasing distance.

---

## 16.3 Bracket Closure

Examples:

```text
(
[
{
```

followed by increasingly long content.

Measure closing delimiter prediction.

---

## 16.4 Nested Brackets

Example:

```text
{
[
(
content
)
]
}
```

Increase:

```text
nesting depth
distance
distractor content
```

This tests whether the architecture can retain multiple structural states.

---

## 16.5 Distractor Test

Store:

```text
important information
```

then provide huge amounts of irrelevant information.

Measure whether the important information survives.

---

# 17. Memory Retention Curves

For every architecture, generate:

```text
distance → recall accuracy
```

Example:

```text
10         99%
100        97%
1,000      93%
10,000     81%
100,000    57%
1,000,000  21%
```

Do not focus only on a single score.

The entire curve is important.

You want to see how memory degrades with distance.

---

# 18. Slot Specialization Analysis

One of the most interesting scientific questions is whether the slots actually specialize.

Log:

```text
average write probability per slot
average read probability per slot
decay distribution per slot
```

for different input types.

For example:

```text
                 slot0 slot1 slot2 slot3

letters            .2    .7    .3    .4
spaces             .1    .6    .2    .2
quotes             .9    .2    .1    .3
brackets           .8    .1    .7    .2
newline            .1    .2    .8    .6
numbers            .3    .5    .4    .4
```

Do not force these meanings.

The purpose is to discover whether specialization emerges naturally.

---

# 19. Visualize Learned Timescales

Convert learned decay values into approximate memory timescales.

For each slot, visualize:

```text
fast dimensions
medium dimensions
slow dimensions
very slow dimensions
```

Compare layers.

Look for patterns such as:

```text
early layers → short-term memory
middle layers → medium-term memory
deep layers → long-term memory
```

This could reveal whether the model naturally creates temporal hierarchy.

---

# 20. JEPA Evaluation

The memory architecture should not only improve byte prediction.

Check the latent-space objective directly.

Measure:

```text
latent prediction loss
byte prediction loss
state consistency
representation quality
```

Compare:

```text
baseline RTU
multi-slot RTU
T-RTU
```

The goal is to determine whether multiple memory registers produce a cleaner and more predictive latent state.

---

# 21. Continual Learning Evaluation

TMT also includes test-time training.

This creates two different forms of memory:

```text
1. recurrent state
2. learned parameters
```

Test them separately.

---

## 21.1 State Memory Test

Reset the model state.

Ask whether previously observed information disappears.

This measures information stored in:

```text
recurrent state
```

---

## 21.2 Parameter Memory Test

Allow test-time learning.

Then reset recurrent state.

Check whether the model still retains learned information.

This measures:

```text
weight-based memory
```

---

## 21.3 Contradiction Test

Teach the model:

```text
A = 123
```

then later:

```text
A = 999
```

Measure:

```text
which information is retained
how quickly old information is overwritten
whether the model becomes unstable
```

This is important for continual learning.

---

# 22. Separate Facts and Skills Later

Only after T-RTU works reliably, experiment with separate memory types.

Possible design:

```text
episodic memory
    ↓
specific observations / events
```

and:

```text
semantic or skill memory
    ↓
generalized patterns
```

For example:

```text
FACT:
"This user said X."

SKILL:
"This type of structure usually behaves like Y."
```

This should be a later experiment, not part of the first T-RTU implementation.

---

# 23. Latent Bookmarks

After the basic architecture works, add a protected memory mechanism.

Concept:

```text
normal slots
+
protected slot
```

The protected memory has much slower decay or another learned retention mechanism.

The model can decide when to write into it.

For example:

```text
important structural event
        ↓
bookmark gate
        ↓
protected memory
```

Do not hard-code quotes or brackets.

The model should learn which events are worth bookmarking.

---

# 24. Adaptive Decay

Later, experiment with:

```text
decay = f(x)
```

instead of only:

```text
decay = learned parameter
```

This allows the model to dynamically change its memory horizon.

For example:

```text
normal content
    ↓
short memory

important event
    ↓
long memory
```

Compare against static decay.

Watch carefully for instability because adaptive decay can allow the model to retain too much information.

---

# 25. Hierarchical JEPA

Later introduce multiple prediction horizons.

Current:

```text
predict next latent
```

Future:

```text
fast prediction:
predict next byte / latent

medium prediction:
predict latent N steps ahead

slow prediction:
predict future representation at larger horizon
```

Possible structure:

```text
fast latent
medium latent
slow latent
```

This should encourage the model to learn representations useful at different temporal scales.

---

# 26. Latent Dictionary / Codebook

Treat this as a separate research branch.

Instead of a completely continuous latent representation:

```text
latent ∈ R^dim
```

experiment with:

```text
latent → nearest learned code
```

This could encourage:

```text
compression
discrete concepts
stable representations
```

But it may also make optimization harder.

Do not introduce it until the continuous architecture is well understood.

---

# 27. Self-Modifying Decay

This is another later experiment.

Allow the model to modify its own memory behavior during inference.

Conceptually:

```text
important event
       ↓
model detects it
       ↓
reduce decay
       ↓
keep memory longer
```

This is potentially powerful but should be implemented only after the ordinary adaptive-decay experiment.

---

# 28. Implementation Strategy

For the first implementation, modify only:

```text
main_torch.py
```

Primarily the `Layer` implementation.

Avoid changing:

```text
Model
Encoder
Decoder
Optimizer
Checkpoint format
Training loop
```

unless necessary.

The objective is to make T-RTU a drop-in replacement for the existing layer.

---

# 29. Initial Implementation

Start with:

```text
K = 4
```

State:

```python
states.shape == [K, dim]
```

Decay:

```python
decay.shape == [K, dim]
```

Candidate projection:

```python
write_proj → [K, dim]
```

Write gate:

```python
write_gate → [K]
```

Read gate:

```python
read_gate → [K]
```

Basic flow:

```python
candidate = write_proj(x)

write = torch.sigmoid(write_gate(x))

decay = torch.sigmoid(self.decay)

state = decay * state + write[:, None] * candidate

read = torch.softmax(read_gate(x), dim=-1)

latent = (read[:, None] * state).sum(dim=0)
```

The exact handling of `dummy` and trace variables should follow the existing TMT implementation rather than blindly copying the simplified equations above.

---

# 30. Dummy / Trace Handling

Before implementing, inspect exactly what the existing:

```text
states
decaytrace
dummy
```

represent in `main_torch.py`.

Do not simply repeat `dummy` K times without understanding its role.

Preferred approach:

```text
existing dummy representation
        ↓
small projection
        ↓
K slot-specific contributions
```

For example:

```text
dummy → slot_write_proj → [K, dim]
```

This preserves the role of the existing mechanism while allowing slot specialization.

---

# 31. Checkpoint Compatibility

Keep old and new checkpoints distinguishable.

Example:

```text
model_baseline/
model_trtu_k4/
model_trtu_k8/
```

Do not silently load an old checkpoint into a changed layer shape.

Record architecture metadata:

```text
architecture = baseline
architecture = trtu

K = 4

decay_type = per_dimension

read_routing = true
write_routing = true
```

---

# 32. Logging

Every training run should record:

```text
model architecture
K
dimension
layer count
parameter count
training dataset
epochs
batch size
learning rate
seed
training loss
validation loss
latent loss
byte loss
memory benchmark
```

For T-RTU specifically also record:

```text
mean write gate
mean read gate
mean decay
slot activation statistics
```

---

# 33. Reproducibility

Every experiment should have:

```text
fixed random seed
fixed dataset
fixed training budget
fixed evaluation set
```

When comparing architectures, keep everything else unchanged.

The ideal comparison is:

```text
same data
same number of epochs
same optimizer
same learning rate
same seed
same evaluation
```

with only the architecture changed.

---

# 34. Main Comparison Table

Eventually maintain one table:

| Model        | Params |  K | Write routing | Read routing | Decay   | Memory accuracy | Byte loss | MCC | Speed |
| ------------ | -----: | -: | ------------- | ------------ | ------- | --------------: | --------: | --: | ----: |
| Baseline     |    ... |  1 | No            | No           | ...     |             ... |       ... | ... |   ... |
| Multi-slot   |    ... |  4 | No            | No           | ...     |             ... |       ... | ... |   ... |
| T-RTU        |    ... |  4 | Yes           | No           | ...     |             ... |       ... | ... |   ... |
| T-RTU + read |    ... |  4 | Yes           | Yes          | ...     |             ... |       ... | ... |   ... |
| T-RTU full   |    ... |  4 | Yes           | Yes          | per-dim |             ... |       ... | ... |   ... |

This becomes the main experiment record.

---

# 35. Success Criteria

Do not define success only as lower training loss.

The architecture should ideally demonstrate improvements in several independent areas:

```text
1. Long-range memory
2. Quote/bracket closure
3. Delayed recall
4. Resistance to distractors
5. Latent prediction
6. Continual learning
7. Stable training
8. Reasonable parameter growth
9. O(1) sequence memory
```

A change that improves one metric while severely hurting the others should be investigated rather than automatically considered an improvement.

---

# 36. Priority Order

## Phase 1 — Baseline

* Freeze current model
* Record all metrics
* Build memory benchmark

## Phase 2 — Multi-Slot RTU

* K=2
* K=4
* K=8
* compare against baseline

## Phase 3 — Dynamic Write

* input-dependent write gates
* measure slot specialization

## Phase 4 — Slot-Specific Candidates

* separate write projection per slot
* compare against shared projection

## Phase 5 — Read Routing

* dynamic read selection
* analyze read behavior

## Phase 6 — Per-Dimension Decay

* `[K]`
* `[K, dim]`
* compare performance and stability

## Phase 7 — Full T-RTU

Combine:

```text
multi-slot
+
dynamic writing
+
slot-specific candidates
+
dynamic reading
+
per-dimension decay
```

## Phase 8 — Analysis

* memory curves
* slot specialization
* learned timescales
* JEPA representation analysis

## Phase 9 — Continual Learning

* state memory
* parameter memory
* contradiction tests
* forgetting tests

## Phase 10 — Advanced Research

Only now investigate:

```text
protected bookmarks
adaptive decay
hierarchical JEPA
separate fact/skill memory
latent codebooks
self-modifying decay
```

---

# 37. Final Target Architecture

The long-term version should conceptually look like:

```text
                         INPUT BYTE
                             │
                             ▼
                         ENCODER
                             │
                             ▼
                             x
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        WRITE ROUTER    DECAY CONTROL   READ ROUTER
              │              │              │
              ▼              ▼              │
       ┌─────────────────────────────┐      │
       │        MEMORY REGISTERS     │      │
       │                             │      │
       │  Slot 0  ────────────────   │      │
       │  Slot 1  ────────────────   │      │
       │  Slot 2  ────────────────   │      │
       │  ...                        │      │
       │  Slot K-1 ────────────────  │      │
       └──────────────┬──────────────┘      │
                      │                     │
                      └──────────┬──────────┘
                                 ▼
                         WEIGHTED LATENT
                                 │
                      ┌──────────┴──────────┐
                      ▼                     ▼
              JEPA PREDICTION          BYTE DECODER
                      │                     │
                      ▼                     ▼
                future latent          next byte
```

The core principle is:

```text
WRITE SELECTIVELY
STORE AT MULTIPLE TIMESCALES
READ SELECTIVELY
```

while maintaining:

```text
streaming
+
constant memory
+
latent prediction
+
continual learning
```

---

# 38. Research Questions

The project should ultimately answer these questions:

### Memory

Can multiple persistent registers retain information over much longer streams than one RTU state?

### Specialization

Do different registers naturally develop different roles?

### Timescales

Does the model learn a hierarchy of memory timescales?

### Routing

Does input-dependent write/read routing improve useful memory rather than merely increasing capacity?

### JEPA

Does better memory produce more predictive latent representations?

### Continual learning

Can the system separate temporary recurrent memory from persistent learned knowledge?

### Efficiency

Can these improvements be achieved while maintaining constant sequence-memory complexity?

### Interpretability

Can the behavior of the learned memory registers be understood through their:

```text
decay
write
read
activation
```

patterns?

---

# 39. Immediate Next Step

Do not implement the entire roadmap yet.

The first concrete milestone should be:

```text
Baseline benchmark
        ↓
K=4 multi-slot RTU
        ↓
dynamic write gates
        ↓
benchmark comparison
```

Only if that shows a measurable improvement should the project move to:

```text
slot-specific candidates
        ↓
read routing
        ↓
per-dimension decay
```

This keeps the research clean and makes every architectural change experimentally meaningful.
