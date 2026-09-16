import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Pure-PyTorch CPU port of main.py (the MLX byte-level recurrent LM).
#
# Semantics-preserving port: every leaf that MLX registers on the module
# (including the recurrent/trace arrays states/decaytrace/embedtrace/decay,
# which MLX's value_and_grad differentiates and AdamW updates) is a torch
# nn.Parameter here, and AdamW updates ALL of them exactly like MLX does.
#
# AdamW parity note: MLX's opt.AdamW defaults to bias_correction=False
# (update = w*(1-lr*wd) - lr*m/(sqrt(v)+eps)); torch.optim.AdamW always
# applies bias correction (torch.optim._functional.adam divides by
# 1-b1^step/1-b2^step), which measurably changes the trajectory (~10x
# larger effective first steps). So this file implements AdamW with the
# exact MLX formulas (the same ones main.py's compiled _adamwc uses), and
# additionally reproduces main.py's quirk that the AdamW step is a pure
# function whose *input values* for the trace arrays are the freshly
# bookkept raw trace values (mirrored into the cached tree before the
# optimizer step), not the values the forward pass read.
# ---------------------------------------------------------------------------

# Per-byte one-hot column, precomputed once (mirrors ONEHOT in main.py).
ONEHOT = torch.eye(256)
ONEHOT_COLS = [ONEHOT[:, i][:, None] for i in range(256)]
# log(256), precomputed once (sample() recomputed this tensor every call).
LN256 = torch.log(torch.tensor(256.0))


class AdamW:
    """MLX-semantics AdamW (bias_correction=False), same update as
    main.py's compiled _adamwc:

        m' = b1*m + (1-b1)*g
        v' = b2*v + (1-b2)*g^2
        w' = w*(1-lr*wd) - lr*m'/(sqrt(v')+eps)

    torch.optim.AdamW always applies bias correction (torch.optim.
    _functional.adam divides by 1-b1^step and 1-b2^step), which would
    change the trajectory, so we implement the MLX formulas directly.

    state: flat dict {name: {"m": tensor, "v": tensor}} plus
    {"step": int, "learning_rate": float} - mirrors the per-leaf
    optimizer state MLX saves under "o.<name>.m"/"o.<name>.v".
    """

    def __init__(self, names, lr, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, bias_correction=False):
        assert bias_correction is False
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.lr = float(lr)
        self.names = list(names)
        self.state = {"step": 0, "learning_rate": self.lr}
        for n in self.names:
            self.state[n] = {"m": None, "v": None}

        # torch.compile'd leaf math (see _step_math). The step counter is
        # deliberately kept OUTSIDE the compiled region (in _apply_pure):
        # a Python int that changes every call would guard-trigger a
        # recompile per step until dynamo's cache overflows. Tensors are
        # guarded by shape/dtype only, so the compiled graph is reused.
        self._step_math_c = torch.compile(self._step_math)

    def init_state(self, values):
        """Allocate m/v from concrete values (must be called once, with the
        real tensors, before the first _apply_pure)."""
        for n in self.names:
            self.state[n]["m"] = torch.zeros_like(values[n])
            self.state[n]["v"] = torch.zeros_like(values[n])

    def _step_math(self, grads, values, state):
        """Pure AdamW leaf math for one step (no step counter): the same
        formulas as before, over all leaves. This is the hot 73% of the
        training step - the eager loop did ~8 small memory passes over
        ~4.5M params per leaf (~800 kernel launches). torch.compile fuses
        the elementwise chain per leaf; measured 45.0 -> 16.9 ms/step on
        the target machine, allclose (rtol 1e-5) vs the eager loop on
        params and m/v. Reads only tensors from `state` (the "step" int
        is never touched, see __init__ note)."""
        new_values, new_mv = {}, {}
        for n, g in grads.items():
            w = values[n]
            m = state[n]["m"]
            v = state[n]["v"]
            m_ = self.b1 * m + (1.0 - self.b1) * g
            v_ = self.b2 * v + (1.0 - self.b2) * g * g
            lr = self.lr
            w_ = w * (1.0 - lr * self.wd) - lr * m_ / (torch.sqrt(v_) + self.eps)
            new_values[n] = w_
            new_mv[n] = (m_, v_)
        return new_values, new_mv

    def _apply_pure(self, grads, values, state):
        """Pure AdamW step (mirrors main.py._adamwc): given the grads tree
        and the *input values* the update should apply to (which for the
        trace arrays are the freshly bookkept values, NOT the values the
        forward read), return new values + new m/v + incremented step."""
        new_values, new_mv = {}, {}
        with torch.no_grad():
            # Compiled leaf math: run under no_grad so no input/output can
            # be autograd-tracked (keeps dynamo guards stable across steps).
            new_values, new_mv = self._step_math_c(grads, values, state)
        return new_values, new_mv, state["step"] + 1

    def step_inplace(self, grads, param_map, values=None):
        """Eager convenience path (benchmark head training): update in place,
        reading current values by default."""
        if values is None:
            values = {n: param_map[n].detach() for n in self.names}
        new_values, new_mv, new_step = self._apply_pure(grads, values, self.state)
        with torch.no_grad():
            for n, w_ in new_values.items():
                param_map[n].copy_(w_)
        for n, (m_, v_) in new_mv.items():
            self.state[n]["m"] = m_
            self.state[n]["v"] = v_
        self.state["step"] = new_step


class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # MLX init: Embedding weight ~ Normal(0, sqrt(1/dim)).
        self.embed = nn.Embedding(256, dim)
        nn.init.normal_(self.embed.weight, 0.0, (1.0 / dim) ** 0.5)
        self.embedtrace = nn.Parameter(torch.zeros(256, dim))

    def __call__(self, c: int):
        return self.embed.weight[c]


class Decoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # MLX Linear init: uniform(-k, k), k = sqrt(1/input_dims).
        k = (1.0 / dim) ** 0.5
        self.decode = nn.Linear(dim, 256)
        nn.init.uniform_(self.decode.weight, -k, k)
        nn.init.uniform_(self.decode.bias, -k, k)
        self.stop = nn.Linear(dim, 1)
        nn.init.uniform_(self.stop.weight, -k, k)
        nn.init.uniform_(self.stop.bias, -k, k)

    def __call__(self, x: torch.Tensor):
        return self.decode(x), torch.sigmoid(self.stop(x))


class Layer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.decay = nn.Parameter(torch.zeros(dim))
        self.states = nn.Parameter(torch.zeros(dim))
        self.decaytrace = nn.Parameter(torch.zeros(dim))

        # main.py uses nn.LayerNorm(dim) (MLX default eps=1e-5, affine) -
        # the recurrent LM itself uses LayerNorm, not RMSNorm.
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias=False)
        k = (1.0 / dim) ** 0.5
        nn.init.uniform_(self.weights.weight, -k, k)

    def __call__(self, x: torch.Tensor, dummy: torch.Tensor):
        decay = torch.sigmoid(self.decay)
        state = (decay * self.states) + x + dummy
        return x + F.silu(self.weights(self.norm(state))), state, decay


def _names(params: dict):
    return {p: n for n, p in params.items()}


class Model(nn.Module):
    def __init__(self, dim: int, layers: int, temp: float, lr: float):
        super().__init__()
        self.dim = dim
        self.layercount = layers
        self.temp = temp

        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)
        self.layers = nn.ModuleList([Layer(dim) for _ in range(layers)])

        # Flat name->Parameter map for ALL leaves (including the trace
        # arrays), mirroring the MLX parameter tree keys
        # (encoder.embed.weight, layers.0.states, ...).
        self._pn = {n: p for n, p in self.named_parameters()}

        self.optimizer = AdamW(list(self._pn), lr=lr)
        self.optimizer.init_state(self._pn)

        # Same dummies list reused every step (never mutated); kept out of
        # the module's parameter registry via object.__setattr__.
        object.__setattr__(self, "_dummies",
                           [torch.zeros(dim, requires_grad=True) for _ in range(layers)])
        self.frozen = False

        # Hot-path caches: the per-leaf name list and parameter list are
        # loop-invariant but were rebuilt with list(self._pn) every step.
        self._names = list(self._pn)
        self._params = [self._pn[n] for n in self._names]

    # -- bookkeeping helpers -------------------------------------------------

    def freeze(self):
        """Feature-extraction mode (mirrors main.py): stop the training
        path and make __call__ return the final latent state instead of
        sampling (used by benchmark_torch.py)."""
        self.frozen = True

    def reset(self):
        """Zero the recurrent/trace buffers (embedtrace and each layer's
        states/decaytrace) in place so every example starts clean. Used by
        benchmark_torch.py between examples; parameters untouched."""
        with torch.no_grad():
            self.encoder.embedtrace.zero_()
            for layer in self.layers:
                layer.states.zero_()
                layer.decaytrace.zero_()

    def sample(self, output: torch.Tensor):
        # Mirrors main.py.sample: entropy via logsumexp, adaptive temp,
        # then a categorical draw. (RNG differs from mx.random.categorical
        # by design; the sampled byte only feeds the chat loop, not the
        # training stream.)
        probs = torch.softmax(output, dim=0)
        entropy = (torch.logsumexp(output, dim=0) - torch.sum(probs * output)) / LN256
        temp = max(0.1, self.temp * (1.0 - self.temp * float(entropy)))
        return torch.multinomial(torch.softmax(output / temp, dim=0), 1).item()

    def generate(self, currb: int):
        """No-grad streaming forward for read-only chat (TMT_FROZEN=1):
        mirrors main.py.generate - evolve only layer.states, no trace
        bookkeeping, no optimizer, no save."""
        with torch.no_grad():
            x = self.encoder(currb)
            for i, layer in enumerate(self.layers):
                x, state, _ = layer(x, self._dummies[i])
                layer.states.copy_(state)
            output, stop = self.decoder(x)
            b = self.sample(output)
        return b, float(stop)

    # -- training step -------------------------------------------------------

    def __call__(self, currb: int, nextb: int | None, end: bool):
        if self.frozen:
            # Mirrors main.py frozen path: stream the byte through and
            # return the last layer's COMPUTED state (the latent the
            # benchmark head consumes). No gradient tracking, no writes
            # to the stored recurrent buffers.
            with torch.no_grad():
                x = self.encoder(currb)
                for layer in self.layers:
                    x, state, _ = layer(x, torch.zeros(self.dim))
                final = state.detach()
            return final

        c = currb
        n = nextb if nextb is not None else 0
        hasnext = 1.0 if nextb is not None else 0.0
        endf = 1.0 if end else 0.0

        dummies = self._dummies

        # ---- forward (mirrors the compiled vgc in main.py) ----
        x = self.encoder(c)
        states, decays = [], []
        for i, layer in enumerate(self.layers):
            x, state, decay = layer(x, dummies[i])
            states.append(state)
            decays.append(decay)
        output, stop = self.decoder(x)

        variance = torch.clamp(1.0 - torch.sqrt(torch.var(x, correction=0) + 1e-4), min=0.0)
        tgt = self.encoder.embed.weight[n].detach()
        pred = torch.mean(torch.square(x - tgt))
        xent = -output[n] + torch.logsumexp(output, dim=0)
        stopmse = torch.mean(torch.square(stop - endf))
        loss = variance + hasnext * (pred + xent + stopmse)

        # ---- grads wrt ALL leaves and the dummies (one backward, like
        # value_and_grad with argnums=(0,1)) ----
        names, params = self._names, self._params
        gout = torch.autograd.grad(
            loss, params + dummies, allow_unused=True
        )
        glist, dlds = gout[: len(names)], gout[len(names):]
        grads = {
            n: (g if g is not None else torch.zeros_like(p))
            for n, g, p in zip(names, glist, params)
        }

        # ---- manual grafts (main.py's post-vgc bookkeeping, using the
        # values the forward read - params are untouched so far) ----
        # .detach() on the grafts: they are computed from requires_grad
        # parameters inside the graph, so without it the resulting "grad"
        # tensors carry requires_grad=True (a latent wart the eager loop
        # tolerated). Every tensor fed to the compiled AdamW must be
        # requires_grad=False or dynamo's requires_grad guard fails and
        # the graph recompiles every step (~50 MB/step of retained
        # artifacts - verified leak). Values are bit-identical either way
        # (MLX grads are plain arrays; detaching is the exact analogue).
        decays0 = decays[0]
        embedtrace_graft = dlds[0] * (self.encoder.embedtrace * decays0)
        grads["encoder.embed.weight"] = grads["encoder.embed.weight"] + embedtrace_graft.detach()

        new_trace = {}  # freshly bookkept values the AdamW step must consume
        for i, layer in enumerate(self.layers):
            decaytrace = (decays[i] * layer.decaytrace) + (
                decays[i] * (1.0 - decays[i]) * layer.states
            )
            grads[f"layers.{i}.decay"] = (dlds[i] * decaytrace).detach()
            new_trace[f"layers.{i}.states"] = states[i].detach()
            new_trace[f"layers.{i}.decaytrace"] = decaytrace.detach()
        new_trace["encoder.embedtrace"] = (
            self.encoder.embedtrace * decays0 + ONEHOT_COLS[c]
        ).detach()

        # ---- AdamW step (pure; trace arrays update FROM their fresh
        # bookkept values, exactly like main.py mirrors p before _adamwc) ----
        values = {n: p.detach() for n, p in self._pn.items()}
        values.update(new_trace)
        new_values, new_mv, new_step = self.optimizer._apply_pure(
            grads, values, self.optimizer.state
        )
        with torch.no_grad():
            # Single multi-tensor copy kernel per side instead of 102
            # individual copy_ calls (measured 5.0 -> ~1.5 ms/step).
            torch._foreach_copy_(params, [new_values[n] for n in names])
        for n, (m_, v_) in new_mv.items():
            self.optimizer.state[n]["m"] = m_
            self.optimizer.state[n]["v"] = v_
        self.optimizer.state["step"] = new_step

        self.lastloss = float(loss.detach())
        return self.sample(output.detach()), float(stop.detach())

    # -- persistence ---------------------------------------------------------

    def save(self, path: str):
        """Atomic save: write a genuine safetensors file (safetensors.torch
        save_file, same on-disk format the MLX version writes) to a temp
        file next to the destination, then os.replace. Key layout mirrors
        main.py (m.<name> params, o.<name>.[m|v] optimizer state) and the
        trace arrays are ordinary params here, so they live under
        m.encoder.embedtrace / m.layers.N.[states|decaytrace]. (torch.save
        pickles would also work, but safetensors keeps the .safetensors
        extension meaningful and interoperable.)"""
        import safetensors.torch as st

        data = {}
        for n, p in self._pn.items():
            data[f"m.{n}"] = p.detach().cpu().contiguous()
        for n in self._pn:
            stt = self.optimizer.state[n]
            data[f"o.{n}.m"] = stt["m"].detach().cpu().contiguous()
            data[f"o.{n}.v"] = stt["v"].detach().cpu().contiguous()
        data["o.step"] = torch.tensor(self.optimizer.state["step"], dtype=torch.int64)
        data["o.learning_rate"] = torch.tensor(self.optimizer.state["learning_rate"], dtype=torch.float64)

        tmp = f"{path}.tmp.safetensors"
        st.save_file(data, tmp)
        os.replace(tmp, path)

    def load(self, path: str):
        import os as _os
        if not _os.path.exists(path) or _os.path.getsize(path) == 0:
            return
        try:
            import safetensors.torch as st
            data = st.load_file(path)
        except Exception:
            # Not a readable checkpoint (e.g. an MLX-written one with arrays
            # the torch shapes disagree on, or a truncated file): start
            # fresh, mirroring main.py's silent skip.
            print("\n[checkpoint unreadable by torch port, starting fresh]", flush=True)
            return

        with torch.no_grad():
            for n, p in self._pn.items():
                if f"m.{n}" in data:
                    p.copy_(data[f"m.{n}"])
        for n in self._pn:
            if f"o.{n}.m" in data and f"o.{n}.v" in data:
                stt = self.optimizer.state[n]
                stt["m"] = data[f"o.{n}.m"].to(self._pn[n].dtype)
                stt["v"] = data[f"o.{n}.v"].to(self._pn[n].dtype)
        if "o.step" in data:
            self.optimizer.state["step"] = int(data["o.step"])
        if "o.learning_rate" in data:
            self.optimizer.state["learning_rate"] = float(data["o.learning_rate"])


class Runtime:
    def __init__(self, path: str, threshold: float, **kwargs):
        self.model = Model(**kwargs)
        self.path = path
        self.threshold = threshold
        self.step = 0
        self.limit = int(os.environ.get('TMT_STEPS', '0') or 0)
        self.log = os.environ.get('TMT_LOG', '') == '1'
        self.frozen = os.environ.get('TMT_FROZEN', '') == '1'
        self.maxtokens = int(os.environ.get('TMT_MAXTOKENS', '2000') or 2000)

    def save(self):
        self.step += 1
        if self.step % 500 == 0:
            self.model.save(self.path)

    def call(self, c: int, n: int | None, end: bool):
        outputs = self.model(c, n, end)
        self.save()

        if self.log and self.step % 100 == 0:
            avg = self.model.lastloss
            print(f'\r[step {self.step}] loss = {avg:.4f}', end='', flush=True)

        if self.limit and self.step >= self.limit:
            raise KeyboardInterrupt
        return outputs

    def write(self, b: int):
        import sys
        sys.stdout.write(bytes([b]).decode('utf-8', errors='replace'))
        sys.stdout.flush()

    def chat(self):
        import itertools

        if self.frozen:
            print('\n[frozen] read-only chat: no training, no checkpoint writes', flush=True)

        while True:
            try:
                text = input(f'\n[{self.now()}]\nUser >> ')
            except EOFError:
                print('\n[EOF, exiting chat]', flush=True)
                break
            data = (text + '\n').encode('utf-8')

            if self.frozen:
                for c, _ in itertools.pairwise(data):
                    self.model.generate(c)
            else:
                for i, (c, n) in enumerate(itertools.pairwise(data)):
                    b, _ = self.call(c, n, i == len(data) - 2)

            print(f'\n[{self.now()}]\nModel >> ', end='', flush=True)

            b = data[-1]
            generated = 0
            while True:
                if self.frozen:
                    b, stop = self.model.generate(b)
                else:
                    b, stop = self.call(b, None, False)
                self.write(b)
                generated += 1
                if stop > self.threshold:
                    break
                if generated >= self.maxtokens:
                    print('\n[stopped: max tokens reached]', flush=True)
                    break

    def dataset(self):
        import glob, itertools
        files = glob.glob('wikipedia_clean/**/wiki_*', recursive=True)

        if not files:
            files = glob.glob('*.txt')
            print(f'\n[wikipedia_clean/ not found, falling back to local .txt files: {files}]', flush=True)

        if not files:
            print('\n[no dataset found, training on built-in sample text]', flush=True)
            files = None

        while True:
            if files is None:
                lines = iter(SAMPLE_TEXT.splitlines(keepends=True))
            else:
                lines = (line for file in files for line in open(file, 'r', encoding='utf-8', errors='ignore'))

            for line in lines:
                data = line.encode('utf-8')
                for i, (c, n) in enumerate(itertools.pairwise(data)):
                    b, _ = self.call(c, n, i == len(data) - 2)
                    self.write(b)

    def now(self):
        from datetime import datetime
        return datetime.now().strftime('%d/%m/%Y, %H:%M:%S')

    def __call__(self):
        envmode = os.environ.get('TMT_MODE')
        if envmode is not None:
            mode = bool(int(envmode))
            print(f'\n[{self.now()}]\nmode >> {int(mode)} (from TMT_MODE)')
        else:
            try:
                mode = bool(int(input(f'\n[{self.now()}]\nmode >> ')))
            except ValueError:
                print('\nInvalid mode.')
                return

        self.model.load(self.path)
        print()

        try:
            match mode:
                case False:
                    self.dataset()
                case True:
                    self.chat()
        except KeyboardInterrupt:
            pass
        finally:
            if not self.frozen:
                self.model.save(self.path)


SAMPLE_TEXT = '''The quick brown fox jumps over the lazy dog. The dog barks, and the fox runs away into the forest.
Deep in the forest there lived an old owl who knew every path, every stream, and every tree by its name.
To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune.
All that glitters is not gold; often have you heard that told.
The rain in Spain stays mainly in the plain. In Hertford, Hereford, and Hampshire, hurricanes hardly ever happen.
It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.
Once upon a midnight dreary, while I pondered, weak and weary, over many a quaint and curious volume of forgotten lore.
A journey of a thousand miles begins with a single step. Practice makes perfect, and patience pays in the end.
The sun rises in the east and sets in the west, and the stars come out to watch over us all through the night.
Hello, world! This is a tiny sample used to smoke-test training when no real dataset is available (a bit like CoLA).
'''

if __name__ == '__main__':
    Runtime(path='model.safetensors', threshold=0.25, dim=512, layers=16, temp=0.6, lr=5e-4)()
