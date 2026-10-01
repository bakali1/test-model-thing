# Test-Model-Thing (TMT)

[YouTube Video](https://youtu.be/9UERVVwpNew)

This is a small proof-of-concept language model (not an LLM) that incorporates the following (and some smaller features as well):
* Latent-space prediction
* Internal state + recurrent trace units (RTUs)
* Byte input/output
* Continuous data streaming
* Test-time training

The model is built with MLX, so it should run fine on all Apple Silicon devices. MLX on Linux has not been tested, but feel free to try it.

Being a proof of concept I have only trained a 4.5-million parameter model (keep in mind, GPT-1 was ~117m) for about 12 hours, but there are very promising results. The model tends to misspell characters (since it outputs byte-by-byte, rather than token-by-token) but it is able to close quotes/brackets and such. Given further training and scaling up the hyperparameters this could become much more powerful. My dataset is also tiny (only a few hundred MB), so there's a lot more world knowledge that can be fed into the model.

This model architecture was designed in about a month by me (a solo high school dev) and some Gemini (only pair programming, no agents). I wrote about a dozen prototypes before creating this architecture. I write READMEs myself without AI.

Feel free to fork the training and benchmark code (everything is under MIT). I really encourage you to try things out, submit issues, and fork the repo.

<img width="499" height="497" alt="3f7f1530-c0c7-43c4-9981-30e9023a19fb" src="https://github.com/user-attachments/assets/eb7e5a97-09b5-4a7b-9484-eb898042e9dc" />

## Running on Linux

MLX ships CPU wheels for Linux x86_64, so this runs on any modern Linux box (no GPU needed). On Linux you must install the ```mlx[cpu]``` extra so the CPU backend library (```libmlx.so```) is pulled in:

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'mlx[cpu]'
python main.py
```

The default entry point now uses the 4.5m config from the README (```dim = 512```, ```layers = 16```) and saves/loads weights to/from ```model.safetensors``` (created from scratch on first run). If ```data/wikipedia_clean/``` doesn't exist, dataset mode automatically falls back to any ```*.txt``` files in ```data/``` (e.g. ```data/sample.txt```) or a small built-in sample text.

A few environment variables are supported for non-interactive/CI use:
* ```TMT_MODE=0``` (dataset) or ```TMT_MODE=1``` (chat) skips the mode prompt.
* ```TMT_STEPS=N``` stops training cleanly after N steps (weights are still saved).
* ```TMT_LOG=1``` prints the running loss every 100 steps.

Example smoke test: ```TMT_MODE=0 TMT_STEPS=50 TMT_LOG=1 .venv/bin/python main.py > /dev/null```

## Training your own model

Model weights (in ```.safetensors```) are not provided because GitHub doesn't like very large files. But, you can train your own model simply by initializing a ```venv``` and installing ```mlx```, no other libraries needed, then running ```main.py```. When you run it, you will be prompted with the mode, ```0``` being train on dataset and ```1``` being chat. You will have to configure your own dataset by modifying the code (to run dataset mode), but you should be able to run chat mode without modifying anything if you have weights already.

Once it begins training, you can safely ^C the program and it will save weights. It should also periodically save weights if I'm not mistaken. The saved weights include the internal memory so the model will remember that the next time it runs. You can launch into chat mode and the memory should carry on from whatever it was learning in training.

## How it works

In detail, here are some of the main capabilities of the model that differ from LLMs:
* JEPA-style latent space prediction, as the decoder can be removed/disabled and the model still rolls out forward as is. The model is not trained explicitly on predicting the next byte, but rather on two separate goals (predicting the next 'thing' in latent space, and translating the current latent space vector to a byte).
* Theoretically infinite memory, as it does not have a context window and instead relies on RTUs to store internal state/memory. However it does decay old memories over time. Also I think this should be O(1) memory based on my implementation but I'm not 100% sure.
* Built-in multimodality, as the model outputs bytes (and thus should theoretically be capable of handling any binary data).
* Streaming data live, since the model only processes one byte at once at rapid pace. In fact it is completely 'blind' to everything that came before the current byte, only relying on the current processed byte and its internal memory to decide the next byte. This confirms the model is definitely learning to remember things.
* Continual training, as it keeps training on user input, training data, and its own output to improve its predictions automatically. Keep in mind the model can only output a byte (0-255) each pass anyways (in addition to updating its own state).

The two important hyperparameters are the size of the latent vector (dim) and the amount of individual state layers the latent passes through before decoding (layers). For my 4.5m test these are ```dim = 512``` and ```layers = 16```. There are some other configurations you can change but I think they are less important.

I think this probably will contribute significantly to solving continual learning and memory but I still need other people to review and verify my work! Please feel free to open GitHub issues to tell me what's wrong. If you have compute (e.g. you are a lab or just have GPUs lying around), feel free to fork my code and train larger models as well, with credit. I personally don't have enough compute and as such I can't really train very large models.

Below is an approximate flow chart of the model architecture, made in Apple's Freeform app (excluding the wrapper for dataset cleaning and input/output handling) for reference. Note that the arrow connecting the target latent to the CE loss should instead be the target byte to the CE loss.

<img width="1653" height="1161" alt="JEPA thing" src="https://github.com/user-attachments/assets/2d3a34ff-ba6a-44b8-b361-6c73da9216c0" />

---

## GPU port and experimental work (added 2026-09-24)

The description above is the original MLX proof of concept. Everything since
has been a batched **torch port** (`gpu_port.py`, fp32, CPU/CUDA parity gated)
plus a documented experiment record. Start here:

| I want to… | go to |
|---|---|
| know what is actually true, with numbers | **[`RESULTS.md`](RESULTS.md)** — the ruler and every measured result |
| see the documentation map | **[`docs/INDEX.md`](docs/INDEX.md)** |
| talk to the model, watch its memory | **`serve.py`** — see [Chat and memory page](#chat-and-memory-page) |
| know what has *not* been tested | [`docs/optimization/OPTIMIZATION_TEST_STATUS.md`](docs/optimization/OPTIMIZATION_TEST_STATUS.md) |
| run a GPU leg | [`docs/runbooks/MORNING-RUNBOOK.md`](docs/runbooks/MORNING-RUNBOOK.md) |
| read the current plan | [`docs/plans/PLAN-2026-09-23-GPU.md`](docs/plans/PLAN-2026-09-23-GPU.md) |

Quick facts: the shipped default is a torch port that is **bit-identical** to
the original engine and runs **~27.5 kB/s on a T4** (vs ~1 kB/s on CPU); the
`+27%` Inductor-fused ring-credit path is the default on CUDA, with every
measured-and-losing candidate removed rather than left in the tree.

The model still produces largely incoherent text — it is 4.5 M parameters
trained for ~10^5 steps. See the generated samples in `RESULTS.md`
("Qualitative inspection") for an honest look at what it currently does.

## Chat and memory page

```bash
.venv/bin/python serve.py                 # then open http://127.0.0.1:8080
```

A chat window, plus the three things this model actually does to itself while
it talks. Ported from `../mini-AGI/serve.py`, which is the same idea against a
different model — same four routes, same SSE byte stream, same single lock,
same one-page-no-build-step HTML.

The middle panel is the part that is specific to TMT:

* **test-time training** — the conversation is fed back through `step_batch`,
  the same call the trainer makes on a corpus, one AdamW update per `--learn-chunk`
  bytes. Talking to the model *is* training it, and it says so on the page.
  `--no-learn` serves read-only.
* **long memory** — one chip per dimension whose decay gives it a half-life
  past `--long-bytes`, each showing its half-life and, in the tooltip, the RMS
  of the state it is actually holding. That pairing is deliberate: both plans
  in `docs/plans/` say not to infer liveness from sigma or half-life alone,
  because a dim can have a 32-byte half-life and carry nothing. A dim with a
  long half-life and zero state is drawn dim, not lit.
* **gradient SNR** — `||EMA(g)||² / EMA(||g||²)`, read off the gradient
  `step_batch` already computed. The measured value over the whole 889,897-step
  run was ~0.05 and never recovered.

Generation streams one byte at a time, so what you watch is the model writing.
A byte that is not a printable character is sent with its value and drawn as a
placeholder, so the reply on screen is never shorter than the number of bytes
the model produced. When the stop head never fires and `--max-tokens` ends the
reply, the page says so rather than letting the cap pass for a decision.

Two deliberate departures from mini-AGI, both because TMT's memory is the
recurrent state rather than a context window:

* **A turn is fed alone, not the conversation.** mini-AGI rebuilds its prompt
  from the whole message list every request, because a fresh KV cache has no
  memory between requests. Replaying the history here would apply every earlier
  reply to the state a *second* time.
* **The prime is fed into the state once, at load.** Prepending it to each
  prompt, as mini-AGI does, would re-apply ~1 kB of corpus to the memory on
  every turn and bury the conversation. `/api/prime` still shows you what it
  was, because the model is answering from state you cannot otherwise see.

Turn markers are **off** by default, so the page feeds exactly what
`numpy_port.Runtime.chat` feeds and stays comparable to the chat results
recorded in `RESULTS.md`. `--markers` opts into `<user>`/`<bot>` turns; the
current model tends to leak the tags back, which `RESULTS.md` also records.

Learning writes through `Runtime.save_checkpoint` — the tagged, atomic, keep-k
path — so talking to the model cannot destroy a checkpoint a result was
measured on. Point `--path` somewhere else to keep a served model off
`tmt.npz` entirely. `flask` is used if installed and `http.server` if not;
this project ships with numpy and torch and nothing else. Binds to localhost.

Gated by `tests/check_web.py` (free, CPU): the memory math against the
half-lives recorded in `RESULTS.md`, the prime's line-boundary cut, the
prompt format in both modes, all four routes on **both** servers, the SSE
frame contract, and that learning writes a tagged checkpoint while
`--no-learn` writes nothing.
