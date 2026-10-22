import json
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_eval_set(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['samples'])} samples from {filepath}")
    return data['samples']


def get_eval_letters(samples) -> list:
    """Answer letters used by an eval set, read from each sample's `options` keys.

    Falls back to A/B/C/D when samples carry no `options` (original MCQ sets, which
    always have exactly four options, so behaviour there is unchanged).
    """
    letters = None
    for s in samples:
        opts = s.get('options')
        if not opts or not isinstance(opts, dict):
            # no options, or a plain list of option texts (ARC / FaithEval sets): A-D, as before
            return ['A', 'B', 'C', 'D']
        keys = sorted(opts.keys())
        if letters is None:
            letters = keys
        elif keys != letters:
            raise ValueError(f"Inconsistent option letters in eval set: {letters} vs {keys}")
    return letters or ['A', 'B', 'C', 'D']


def parse_layer_list(spec: str) -> list:
    layers = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)
