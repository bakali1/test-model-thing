import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util

from main import Model

class Classification(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2)

    def __call__(self, x: mx.array): return self.proj(x)

# Tiny built-in stand-in for CoLA (grammatical=1, ungrammatical=0) so the
# benchmark can run when CoLA/original/raw/in_domain_train.tsv is missing.
BUILTIN_COLA = [
    (b'The plot of the novel was surprisingly clever.', 1),
    (b'The professor gave a fascinating lecture yesterday.', 1),
    (b'She sings beautifully and plays the piano well.', 1),
    (b'The children played happily in the park all afternoon.', 1),
    (b'This is the book that I told you about last week.', 1),
    (b'Every student finished their assignment on time.', 1),
    (b'The movie was so entertaining that we watched it twice.', 1),
    (b'Ran quickly the dog down street the.', 0),
    (b'The seems cat sleeping under table the.', 0),
    (b'Bought yesterday I a new book interesting.', 0),
    (b'Their laughing baby the loudly was.', 0),
    (b'Is more this box heavy than that one much.', 0),
    (b'Went she to store the morning early.', 0),
    (b'Cake delicious the ate slowly dog the.', 0),
]

def cola(filepath: str):
    import os
    if not os.path.exists(filepath):
        print(f"\n[{filepath} not found, using tiny built-in labeled sample ({len(BUILTIN_COLA)} examples)]", flush = True)
        return BUILTIN_COLA
    data = []
    with open(filepath, 'r', encoding = 'utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4: data.append((parts[3].encode('utf-8'), int(parts[1])))
    return data

def mcc(tp, tn, fp, fn):
    import math
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0

def run():
    model = Model(dim = 512, layers = 16, temp = 0.75, lr = 5e-4)
    model.load('smaller-4.5m.safetensors')
    model.freeze()

    head = Classification(model.dim)
    headopt = opt.AdamW(learning_rate = 1e-3)

    data = cola('CoLA/original/raw/in_domain_train.tsv')

    def l(params, state: mx.array, target: int):
        head.update(params)
        choice = head(state)

        loss = nn.losses.cross_entropy(choice[None, :], mx.array([target])).mean()
        return loss, choice

    for epoch in range(3):
        print(f'\nEpoch {epoch + 1}')

        dummies = [mx.zeros((model.dim, )) for _ in range(model.layercount)]
        tp, tn, fp, fn = 0, 0, 0, 0
        
        for i, (bytes, label) in enumerate(data):
            final = None
            for b in bytes:
                x = model.encoder(mx.array(b))
                for j, layer in enumerate(model.layers):
                    x, state, _ = layer(x, dummies[j])
                    layer.states = mx.stop_gradient(state)
                
                final = model.layers[-1].states

            (_, choice), grads = mx.value_and_grad(l, argnums = 0)(head.trainable_parameters(), final, label)

            headopt.update(head, grads)
            mx.eval(head.parameters(), headopt.state)

            predicted_class = mx.argmax(choice).item()
            if predicted_class == 1 and label == 1: tp += 1
            elif predicted_class == 0 and label == 0: tn += 1
            elif predicted_class == 1 and label == 0: fp += 1
            elif predicted_class == 0 and label == 1: fn += 1

            score = mcc(tp, tn, fp, fn)

            if i > 0 and i % 500 == 0: print(f'{i + 1}: TP, TN, FP, FN | {tp}, {tn}, {fp}, {fn} ({score})')

        print(f'{i + 1}: TP, TN, FP, FN | {tp}, {tn}, {fp}, {fn} ({score})')

if __name__ == '__main__':
    run()