import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util
import os

# Per-byte one-hot column, precomputed once: onehot[c][:, None] replaces
# building (mx.arange(256) == c)[:, None] on every training step.
ONEHOT = mx.eye(256)
# Cache for mx.array(byte) conversions (one shared scalar array per byte 0..255).
BYTE_ARRAYS = [mx.array(i) for i in range(256)]
# 0-d has-next / stop-target flags for the compiled training step.
FLAGS = (mx.array(0.0), mx.array(1.0))

class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)
        self.embedtrace = mx.zeros((256, dim))

    def __call__(self, x: mx.array): return self.embed(x)

class Decoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def __call__(self, x: mx.array): return self.decode(x), mx.sigmoid(self.stop(x))

class Layer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        
        self.decay = mx.zeros((dim, ))
        self.states = mx.zeros((dim, ))
        self.decaytrace = mx.zeros((dim, ))
        
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias = False)
        self.silu = nn.SiLU()

    def __call__(self, x: mx.array, dummy: mx.array):
        decay = mx.sigmoid(self.decay)
        state = (decay * self.states) + x + dummy

        return x + self.silu(self.weights(self.norm(state))), state, decay

def _prune_empty(tree):
    """Drop empty dict/list subtrees (e.g. the parameterless nn.SiLU child
    modules, which appear as `layers.N.silu = {}` in trainable_parameters()).
    They carry no arrays, but save()/load() round-trips flatten them away, so
    an unpruned tree goes out of structural sync with a reloaded optimizer
    state and tree_map raises KeyError on resume."""
    if isinstance(tree, dict):
        pruned = {k: _prune_empty(v) for k, v in tree.items()}
        return {k: v for k, v in pruned.items()
                if not (isinstance(v, (dict, list)) and not v)}
    if isinstance(tree, list):
        return [_prune_empty(v) for v in tree]
    return tree


class Model(nn.Module):
    def __init__(self, dim: int, layers: int, temp: float, lr: float):
        super().__init__()
        self.dim = dim
        self.layercount = layers
        self.temp = temp

        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)

        self.layers = [Layer(dim) for _ in range(layers)]
        self.optimizer = opt.AdamW(learning_rate = lr)

        # Same dummies list reused every step (it is never mutated) instead of
        # reallocating layercount zero arrays per call; the trainable parameter
        # tree is cached lazily on first step (trainable_parameters() walks the
        # whole module tree otherwise, every single step). object.__setattr__
        # keeps these out of nn.Module's parameter registry.
        object.__setattr__(self, "_dummies", [mx.zeros((self.dim, )) for _ in range(self.layercount)])
        object.__setattr__(self, "_trainable", None)

        # The two hot inner loops of a training step (value_and_grad of the
        # forward pass, and the AdamW parameter update) are mx.compile'd into
        # single fused graphs. Both are written as pure functions of their
        # arguments (byte scalars, has-next/end flags and the optimizer state
        # are passed in explicitly, so nothing changes between calls except
        # input values); the surrounding mutation/trace bookkeeping stays
        # eager. Measured: optimizer 70.7ms -> 36.7ms, vjp 19.5ms -> 17.0ms,
        # with bit-exact results over chained steps.
        self._vgc = self._build_vgc()
        self._adamwc = self._build_adamw()
        self.frozen = False

    def _build_vgc(self):
        """mx.compile(mx.value_and_grad(fwd)) for one training step.

        fwd is pure: every per-step varying quantity (input byte c, target
        byte n, has-next flag, stop-target flag) is a compile input. The
        next-byte terms are gated with hasnext so a single graph covers both
        the streaming path (hasnext=1.0) and generation (hasnext=0.0, loss =
        variance term only, exactly like the eager `if nextb is not None`).
        The module update inside is only plumbing to route the traced params
        into the layer reads (same trick as the eager value_and_grad path)."""
        model = self

        def fwd(params, dummies: list[mx.array], c, n, hasnext, endf):
            model.update(params)
            x = model.encoder(c)

            states, decays = [], []

            for i, layer in enumerate(model.layers):
                x, state, decay = layer(x, dummies[i])

                states.append(state)
                decays.append(decay)

            output, stop = model.decoder(x)

            loss = mx.maximum(0.0, 1.0 - mx.sqrt(mx.var(x) + 1e-4))
            tgt = mx.stop_gradient(model.encoder(n))
            pred = mx.mean(mx.square(x - tgt))
            xent = -output[n] + mx.logsumexp(output)
            stopmse = mx.mean(mx.square(stop - endf))

            # loss = variance loss + pred mse loss + crossentropy loss + stop mse loss
            return loss + hasnext * (pred + xent + stopmse), (states, decays, output, stop)

        return mx.compile(mx.value_and_grad(fwd, argnums = (0, 1)))

    def _build_adamw(self):
        """mx.compile'd AdamW step, bit-exact reproduction of
        opt.AdamW.apply_gradients (bias_correction=False default: update =
        w*(1-lr*decay) - lr*m/(sqrt(v)+eps); per-leaf m/v returned instead of
        mutated so the function is pure)."""
        o = self.optimizer
        b1, b2 = o.betas
        eps = o.eps
        wd = o.weight_decay

        def adamw(grads, params, state, lr_arr):
            step = state["step"] + 1

            def single(g, w, s):
                lr = lr_arr.astype(g.dtype)
                m_ = b1 * s["m"] + (1 - b1) * g
                v_ = b2 * s["v"] + (1 - b2) * mx.square(g)
                return w * (1 - lr * wd) - lr * m_ / (mx.sqrt(v_) + eps), m_, v_

            combined = util.tree_map(single, grads, params, state)
            split = lambda i: util.tree_map(
                lambda t: t[i], combined, is_leaf = lambda x: isinstance(x, tuple)
            )
            return split(0), split(1), split(2), step

        return mx.compile(adamw)

    def freeze(self):
        """Feature-extraction mode: stop all parameter updates and make
        __call__ return the final latent state instead of sampling (used by
        benchmark.py, which trains only a linear head on frozen features)."""
        self.frozen = True
        object.__setattr__(self, "_trainable", None)

    def sample(self, output: mx.array):
        # Entropy via logsumexp: H = -sum(softmax(o) * log_softmax(o)) equals
        # (logsumexp(o) - sum(probs * o)) / ln(256), but avoids log(probs + eps).
        probs = mx.softmax(output)
        entropy = (mx.logsumexp(output) - mx.sum(probs * output)) / mx.log(mx.array(256))

        # Keep temp as an array (no .item() mid-step): mx.random.categorical
        # accepts array logits, so the sync can wait until the caller needs
        # the sampled byte. Same scale, same distribution.
        temp = mx.maximum(0.1, self.temp * (1.0 - self.temp * entropy))
        return mx.random.categorical(output / temp)

    def __call__(self, currb: int, nextb: int | None, end: bool):
        c = BYTE_ARRAYS[currb]

        if self.frozen:
            # No gradients, no optimizer updates, no trace/state bookkeeping:
            # just stream the byte through and return the last layer's state
            # (the latent benchmark.py's head consumes).
            x = self.encoder(c)
            states = []
            for layer in self.layers:
                x, state, _ = layer(x, mx.zeros((self.dim, )))
                states.append(state)

            final = mx.stop_gradient(states[-1])
            mx.eval(final)
            return final

        p = self._trainable
        if p is None:
            p = object.__setattr__(self, "_trainable", _prune_empty(self.trainable_parameters())) or self._trainable

        dummies = self._dummies
        c_arg = c
        n_arg = BYTE_ARRAYS[nextb] if nextb is not None else BYTE_ARRAYS[0]
        hasnext = FLAGS[1 if nextb is not None else 0]
        endf = FLAGS[1 if end else 0]

        (lossval, (states, decays, output, stop)), (grads, dlds_s) = self._vgc(
            p, dummies, c_arg, n_arg, hasnext, endf
        )

        self.update(p)

        embedtrace = (self.encoder.embedtrace * decays[0]) + ONEHOT[c][:, None]
        grads["encoder"]["embed"]["weight"] += dlds_s[0] * (self.encoder.embedtrace * decays[0])
        self.encoder.embedtrace = mx.stop_gradient(embedtrace)

        for i, layer in enumerate(self.layers):
            dlds = dlds_s[i]

            decaytrace = (decays[i] * layer.decaytrace) + (decays[i] * (1.0 - decays[i]) * layer.states)

            grads["layers"][i]["decay"] = dlds * decaytrace

            layer.states = mx.stop_gradient(states[i])
            layer.decaytrace = mx.stop_gradient(decaytrace)

        # The trace arrays were just reassigned on the module; the optimizer
        # must update from these fresh values (the old code did this implicitly
        # by reading self.parameters() after the assignments), so mirror them
        # into the cached tree before the AdamW step.
        p["encoder"]["embedtrace"] = self.encoder.embedtrace
        for i, layer in enumerate(self.layers):
            p["layers"][i]["states"] = layer.states
            p["layers"][i]["decaytrace"] = layer.decaytrace

        # Graft the post-update parameter arrays back into the cached tree (the
        # optimizer replaces leaves by identity, so the cache would go stale;
        # tree_map keeps the original structure and swaps in the new arrays).
        # Compiled AdamW step: pure function returning the new parameters, the
        # new m/v moment trees and the incremented step counter.
        if "encoder" not in self.optimizer.state:
            st = {"step": mx.array(0, mx.uint64), "learning_rate": self.optimizer.learning_rate}
            st.update(util.tree_map(lambda g: {"m": mx.zeros_like(g), "v": mx.zeros_like(g)}, grads))
            self.optimizer.state = st

        new_params, new_m, new_v, new_step = self._adamwc(
            grads, p, self.optimizer.state, self.optimizer.learning_rate
        )
        self.update(new_params)
        self.optimizer.state["step"] = new_step
        util.tree_map(lambda a, s: s.__setitem__("m", a), new_m, self.optimizer.state)
        util.tree_map(lambda a, s: s.__setitem__("v", a), new_v, self.optimizer.state)
        self._trainable = util.tree_map(lambda _, new: new, self._trainable, new_params)

        # ONE eval per step: the trace bookkeeping, gradient grafts and optimizer
        # update above are pure lazy graph construction. Each extra mx.eval is a
        # full dispatch barrier (syncs the stream), so fusing them into a single
        # eval of everything the step produced (including lastloss/sampled byte,
        # so the sample temp needs no mid-step .item()) is materially faster.
        mx.eval(
            self.encoder.embedtrace,
            [layer.states for layer in self.layers],
            [layer.decaytrace for layer in self.layers],
            self.parameters(),
            self.optimizer.state,
        )

        self.lastloss = lossval.item()
        return self.sample(output).item(), stop.item()

    def save(self, path: str):
        data = {}
        for k, v in util.tree_flatten(self.parameters()): data[f"m.{k}"] = v
        for k, v in util.tree_flatten(self.optimizer.state): data[f"o.{k}"] = v

        data[f"embedtrace"] = self.encoder.embedtrace
        for i, layer in enumerate(self.layers):
            data[f"state.{i}"] = layer.states
            data[f"decaytrace.{i}"] = layer.decaytrace

        # Atomic save: write to a temp file next to the destination and rename.
        # A crash/interrupt mid-save must never truncate an existing checkpoint
        # (mx.save_safetensors opens the destination directly and can leave a
        # 0-byte file behind if it fails partway). Note mx.save_safetensors
        # appends '.safetensors' if missing, so the temp name must end in it.
        import os
        tmp = f"{path}.tmp.safetensors"
        mx.save_safetensors(tmp, data)
        mx.synchronize()  # save_safetensors writes lazily on the stream
        os.replace(tmp, path)

    def load(self, path: str):
        import os
        if not os.path.exists(path) or os.path.getsize(path) == 0: return

        data, model, opts = mx.load(path), {}, {}
        
        for k, v in data.items():
            if k.startswith("m."): model[k[2:]] = v
            elif k.startswith("o."): opts[k[2:]] = v
            elif k.startswith("state."): self.layers[int(k.split('.')[1])].states = v
            elif k.startswith("decaytrace."): self.layers[int(k.split('.')[1])].decaytrace = v
            elif k == "embedtrace": self.encoder.embedtrace = v
            
        if model: self.update(util.tree_unflatten([(k, v) for k, v in model.items() if k in util.tree_flatten(self.parameters())]))
        if opts: self.optimizer.state = util.tree_unflatten(list(opts.items()))
        # Loaded arrays replace the module's parameter leaves by identity, so
        # any cached trainable tree is now stale.
        object.__setattr__(self, "_trainable", None)

class Runtime:
    def __init__(self, path: str, threshold: float, **kwargs):
        self.model = Model(**kwargs)
        self.path = path
        self.threshold = threshold
        self.step = 0
        self.limit = int(os.environ.get('TMT_STEPS', '0') or 0)
        self.log = os.environ.get('TMT_LOG', '') == '1'

    def save(self):
        self.step += 1
        if self.step % 500 == 0: self.model.save(self.path)

    def call(self, c: int, n: int | None, end: bool):
        outputs = self.model(c, n, end)
        self.save()

        if self.log and self.step % 100 == 0:
            avg = self.model.lastloss
            print(f'\r[step {self.step}] loss = {avg:.4f}', end = '', flush = True)

        if self.limit and self.step >= self.limit: raise KeyboardInterrupt
        return outputs

    def write(self, b: int):
        import sys
        # Decoding with 'replace' keeps the terminal clean: raw model bytes can
        # be invalid/partial UTF-8 (especially early in training).
        sys.stdout.write(bytes([b]).decode('utf-8', errors = 'replace'))
        sys.stdout.flush()

    def chat(self):
        import itertools

        while True:
            text = input(f'\n[{self.now()}]\nUser >> ')
            data = (text + '\n').encode('utf-8')
            
            for i, (c, n) in enumerate(itertools.pairwise(data)):
                b, _ = self.call(c, n, i == len(data) - 2)

            print(f'\n[{self.now()}]\nModel >> ', end = '', flush = True)

            b = data[-1]
            while True:
                b, stop = self.call(b, None, False)
                self.write(b)
                if stop > self.threshold: break

    def dataset(self):
        import glob, itertools
        files = glob.glob('wikipedia_clean/**/wiki_*', recursive = True)

        if not files:
            files = glob.glob('*.txt')
            print(f'\n[wikipedia_clean/ not found, falling back to local .txt files: {files}]', flush = True)

        if not files:
            print('\n[no dataset found, training on built-in sample text]', flush = True)
            files = None

        while True:
            if files is None:
                lines = iter(SAMPLE_TEXT.splitlines(keepends = True))
            else:
                lines = (line for file in files for line in open(file, 'r', encoding = 'utf-8', errors = 'ignore'))

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
            try: mode = bool(int(input(f'\n[{self.now()}]\nmode >> ')))
            except ValueError:
                print('\nInvalid mode.')
                return

        self.model.load(self.path)
        print()

        try:
            match mode:
                case False: self.dataset()
                case True: self.chat()
        except KeyboardInterrupt: pass
        finally: self.model.save(self.path)

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
    # 4.5m parameter config (see README). Set TMT_MODE=0/1 to skip the mode
    # prompt, TMT_STEPS=N to stop after N steps, TMT_LOG=1 to print the loss.
    # threshold = stop-head cutoff for chat (valid range ~0..1, higher = longer
    # responses; 0.25 chosen to stop whitespace rambling on under-trained models)
    # temp = sampling temperature (lower = more deterministic; 0.6 chosen for
    # the same reason). Both are easy tweaks if chat feels too stingy/random.
    Runtime(path = 'model.safetensors', threshold = 0.25, dim = 512, layers = 16, temp = 0.6, lr = 5e-4)()
    # param count = (256 * dim) + (dim * dim + dim * 2 + dim) + (256 * dim + dim + 1)