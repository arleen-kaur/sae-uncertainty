"""Quick combined suppression - all layers at once, no per-layer passes."""

import csv
import torch
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import gc

from src.utils import seed_everything, load_eval_set, get_eval_letters
from src.model import load_model, get_letter_token_ids
from src.sae import SAESuppressionHook, load_sae
from src.evaluation import run_eval, summarise, compute_flips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--individual-csv', type=str, required=True)
    parser.add_argument('--eval-set', type=str, required=True)
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    # Select beneficial features (handle both column name formats)
    selected = defaultdict(list)
    with open(args.individual_csv) as f:
        for row in csv.DictReader(f):
            acc_col = 'acc_delta_selection' if 'acc_delta_selection' in row else 'acc_delta'
            ent_col = 'ent_delta_selection' if 'ent_delta_selection' in row else 'ent_delta'
            if float(row[acc_col]) >= 0 and float(row[ent_col]) < 0:
                selected[int(row['layer'])].append(int(row['feature_idx']))
    selected = dict(selected)

    total = sum(len(v) for v in selected.values())
    print(f"Beneficial features: {total} across {len(selected)} layers")

    if total == 0:
        print("No beneficial features. Exiting.")
        return

    # Load model
    model, tokenizer, model_config = load_model(args.model, device=args.device)
    samples = load_eval_set(args.eval_set)
    letter_token_ids = get_letter_token_ids(tokenizer, get_eval_letters(samples))

    # Baseline
    print(f"\nBaseline")
    baseline = run_eval(model, tokenizer, samples, letter_token_ids, args.device, "Baseline")
    base_acc, base_ent = summarise(baseline)
    print(f"Baseline: {base_acc*100:.2f}%  entropy: {base_ent:.4f}")

    # All layers combined
    hooks = []
    saes = {}
    for layer_idx, feats in selected.items():
        sae = load_sae(model_config, layer_idx, device=args.device)
        saes[layer_idx] = sae
        h = SAESuppressionHook(sae, feats, scale=args.scale)
        h.register(model, model_config, layer_idx)
        hooks.append(h)

    res = run_eval(model, tokenizer, samples, letter_token_ids, args.device, "All layers")
    for h in hooks:
        h.remove()

    acc, ent = summarise(res)
    flipped_correct, flipped_incorrect = compute_flips(baseline, res)
    acc_delta = (acc - base_acc) * 100
    ent_delta = ent - base_ent

    print(f"\n{'='*60}")
    print(f"{args.label} ALL LAYERS ({total} feats):")
    print(f"  Acc: {acc*100:.2f}% ({acc_delta:+.2f}%)")
    print(f"  Ent: {ent:.4f} ({ent_delta:+.4f})")
    print(f"  Flipped: ->correct={flipped_correct} ->wrong={flipped_incorrect}")
    print(f"{'='*60}")

    for sae in saes.values():
        del sae
    del saes
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
