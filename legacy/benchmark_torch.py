import torch
import torch.nn as nn
import torch.nn.functional as F

from main_torch import Model, AdamW   # renamed from main_mixed in a98f3ca


class Classification(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2)
        k = (1.0 / dim) ** 0.5  # MLX Linear init
        nn.init.uniform_(self.proj.weight, -k, k)
        nn.init.uniform_(self.proj.bias, -k, k)

    def __call__(self, x: torch.Tensor):
        return self.proj(x)


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
        print(f"\n[{filepath} not found, using tiny built-in labeled sample ({len(BUILTIN_COLA)} examples)]", flush=True)
        return BUILTIN_COLA
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                data.append((parts[3].encode('utf-8'), int(parts[1])))
    return data


def mcc(tp, tn, fp, fn):
    import math
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0


def encode(model, sentence, dummies):
    """Stream one example through the frozen model and return the final
    latent, mirroring benchmark.py's encode (no grad, recurrent-state
    writes detached, last layer's state is the latent)."""
    with torch.no_grad():
        for b in sentence:
            x = model.encoder(b)
            for j, layer in enumerate(model.layers):
                x, state, _ = layer(x, dummies[j])
                layer.states.copy_(state)
        final = model.layers[-1].states.detach()
    return final


def run():
    model = Model(dim=512, layers=16, temp=0.75, lr=5e-4)
    model.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smaller-4.5m.safetensors'))
    model.freeze()

    head = Classification(model.dim)
    headopt = AdamW([n for n, _ in head.named_parameters()], lr=1e-3)
    headopt.init_state({n: p.detach() for n, p in head.named_parameters()})

    data = cola('CoLA/original/raw/in_domain_train.tsv')

    if not data:
        print('\n[no CoLA examples found, nothing to benchmark]', flush=True)
        return

    split = int(len(data) * 0.8)
    train, dev = data[:split], data[split:]

    for epoch in range(3):
        print(f'\nEpoch {epoch + 1}')

        dummies = [torch.zeros(model.dim) for _ in range(model.layercount)]
        tp, tn, fp, fn = 0, 0, 0, 0

        for i, (sentence, label) in enumerate(train):
            model.reset()
            final = encode(model, sentence, dummies)

            choice = head(final)
            loss = F.cross_entropy(choice[None, :], torch.tensor([label]))

            glist = torch.autograd.grad(loss, [p for _, p in head.named_parameters()])
            headopt.step_inplace(
                {n: g for (n, _), g in zip(head.named_parameters(), glist)},
                dict(head.named_parameters()),
            )

            predicted_class = int(choice.argmax())
            if predicted_class == 1 and label == 1:
                tp += 1
            elif predicted_class == 0 and label == 0:
                tn += 1
            elif predicted_class == 1 and label == 0:
                fp += 1
            elif predicted_class == 0 and label == 1:
                fn += 1

            if i > 0 and i % 500 == 0:
                print(f'train {i + 1}: TP, TN, FP, FN | {tp}, {tn}, {fp}, {fn} ({mcc(tp, tn, fp, fn)})')

        if train:
            print(f'train {len(train)}: TP, TN, FP, FN | {tp}, {tn}, {fp}, {fn} ({mcc(tp, tn, fp, fn)})')
        else:
            print('\n[no training examples]')

        if dev:
            dev_tp, dev_tn, dev_fp, dev_fn = 0, 0, 0, 0
            with torch.no_grad():
                for sentence, label in dev:
                    model.reset()
                    final = encode(model, sentence, dummies)
                    choice = head(final)

                    predicted_class = int(choice.argmax())
                    if predicted_class == 1 and label == 1:
                        dev_tp += 1
                    elif predicted_class == 0 and label == 0:
                        dev_tn += 1
                    elif predicted_class == 1 and label == 0:
                        dev_fp += 1
                    elif predicted_class == 0 and label == 1:
                        dev_fn += 1

            dev_score = mcc(dev_tp, dev_tn, dev_fp, dev_fn)
            print(f'dev  {len(dev)}: TP, TN, FP, FN | {dev_tp}, {dev_tn}, {dev_fp}, {dev_fn} (MCC {dev_score})')
        else:
            print('\n[no dev examples]')


if __name__ == '__main__':
    run()