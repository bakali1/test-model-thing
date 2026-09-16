import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util
import os

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
        self.frozen = False

    def freeze(self):
        """Feature-extraction mode: stop all parameter updates and make
        __call__ return the final latent state instead of sampling (used by
        benchmark.py, which trains only a linear head on frozen features)."""
        self.frozen = True

    def sample(self, output: mx.array):
        probs = mx.softmax(output)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8)) / mx.log(mx.array(256))

        temp = mx.maximum(0.1, self.temp * (1.0 - self.temp * entropy)).item()
        return mx.random.categorical(output / temp)

    def __call__(self, currb: int, nextb: int | None, end: bool):
        c = mx.array(currb)

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

        p = self.trainable_parameters()

        def fwd(params, dummies: list[mx.array]):
            self.update(params)
            x = self.encoder(c)

            states, decays = [], []

            for i, layer in enumerate(self.layers):
                x, state, decay = layer(x, dummies[i])

                states.append(state)
                decays.append(decay)

            output, stop = self.decoder(x)

            loss = mx.maximum(0.0, 1.0 - mx.sqrt(mx.var(x) + 1e-4))
            if nextb is not None:
                n = mx.array(nextb)
                tgt = mx.stop_gradient(self.encoder(n))

                loss = loss + mx.mean(mx.square(x - tgt))
                loss = loss - output[n] + mx.logsumexp(output)

                loss = loss + mx.mean(mx.square(stop - mx.array([1.0 if end else 0.0])))

            return loss, (states, decays, output, stop) # loss = variance loss + pred mse loss + crossentropy loss + stop mse loss

        (lossval, (states, decays, output, stop)), (grads, dlds_s) = mx.value_and_grad(
            fwd, argnums = (0, 1)
        )(p, [mx.zeros((self.dim, )) for _ in range(self.layercount)])

        self.update(p)

        embedtrace = (self.encoder.embedtrace * decays[0]) + (mx.arange(256) == c)[:, None]
        grads["encoder"]["embed"]["weight"] += dlds_s[0] * (self.encoder.embedtrace * decays[0])
        self.encoder.embedtrace = mx.stop_gradient(embedtrace)
        mx.eval(self.encoder.embedtrace)

        for i, layer in enumerate(self.layers):
            dlds = dlds_s[i]
            
            decaytrace = (decays[i] * layer.decaytrace) + (decays[i] * (1.0 - decays[i]) * layer.states)

            grads["layers"][i]["decay"] = dlds * decaytrace

            layer.states = mx.stop_gradient(states[i])
            layer.decaytrace = mx.stop_gradient(decaytrace)

            mx.eval(layer.states, layer.decaytrace)

        self.optimizer.update(self, grads)
        mx.eval(self.parameters(), self.optimizer.state)

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
            
        mx.save_safetensors(path, data)

    def load(self, path: str):
        import os
        if not os.path.exists(path): return

        data, model, opts = mx.load(path), {}, {}
        
        for k, v in data.items():
            if k.startswith("m."): model[k[2:]] = v
            elif k.startswith("o."): opts[k[2:]] = v
            elif k.startswith("state."): self.layers[int(k.split('.')[1])].states = v
            elif k.startswith("decaytrace."): self.layers[int(k.split('.')[1])].decaytrace = v
            elif k == "embedtrace": self.encoder.embedtrace = v
            
        if model: self.update(util.tree_unflatten([(k, v) for k, v in model.items() if k in util.tree_flatten(self.parameters())]))
        if opts: self.optimizer.state = util.tree_unflatten(list(opts.items()))

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
    Runtime(path = 'model.safetensors', threshold = 0.35, dim = 512, layers = 16, temp = 0.75, lr = 5e-4)()
    # param count = (256 * dim) + (dim * dim + dim * 2 + dim) + (256 * dim + dim + 1)