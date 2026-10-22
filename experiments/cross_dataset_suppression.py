"""
Cross-Dataset Suppression Validation

Take features selected on one dataset (e.g. MMLU) and suppress them on a
different dataset (e.g. ARC-Challenge) to test generalization.

Usage:
    python experiments/cross_dataset_suppression.py \
        --selection-pkl results/sae_both_feature_selection_all_layers_p25/both_feature_selection.pkl \
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_1094.json \
        --output-dir results/sae_cross_dataset_arc/
"""

import csv
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import gc

from src.utils import seed_everything, load_eval_set, get_eval_letters
from src.model import load_model, get_letter_token_ids
from src.sae import SAESuppressionHook, load_sae
from src.evaluation import run_eval, summarise, compute_flips


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Suppression Validation")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--selection-pkl', type=str, required=True,
                        help='Pickle from feature_selection.py with selected_features')
    parser.add_argument('--eval-set', type=str, required=True,
                        help='Dataset to test suppression on')
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load selected features
    with open(args.selection_pkl, 'rb') as f:
        selection_data = pickle.load(f)
    selected_features = selection_data['selected_features']
    n_total = sum(len(v) for v in selected_features.values())
    print(f"Loaded {n_total} selected features across {len(selected_features)} layers")
    for layer, feats in sorted(selected_features.items()):
        print(f"  L{layer}: {feats}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, model_config = load_model(args.model, device=args.device)

    # Load eval set
    samples = load_eval_set(args.eval_set)
    letter_token_ids = get_letter_token_ids(tokenizer, get_eval_letters(samples))

    # Baseline
    print(f"\nBaseline (no suppression)")
    baseline = run_eval(model, tokenizer, samples, letter_token_ids,
                        args.device, "Baseline")
    base_acc, base_ent = summarise(baseline)
    print(f"Baseline accuracy: {base_acc*100:.2f}%  mean entropy: {base_ent:.4f}")

    all_rows = []

    # Per-layer suppression
    for layer_idx, feats in sorted(selected_features.items()):
        sae = load_sae(model_config, layer_idx, device=args.device)
        h = SAESuppressionHook(sae, feats, scale=args.scale)
        h.register(model, model_config, layer_idx)

        res = run_eval(model, tokenizer, samples, letter_token_ids,
                       args.device, f"L{layer_idx}")
        h.remove()
        del sae
        torch.cuda.empty_cache()
        gc.collect()

        acc, ent = summarise(res)
        flipped_correct, flipped_incorrect = compute_flips(baseline, res)
        acc_delta = (acc - base_acc) * 100
        ent_delta = ent - base_ent

        print(f"  L{layer_idx} ({len(feats)} feats): "
              f"acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
              f"ent={ent:.4f} ({ent_delta:+.4f})  "
              f"->correct={flipped_correct} ->wrong={flipped_incorrect}")

        all_rows.append({
            'condition': f'L{layer_idx}_combined',
            'layer': layer_idx,
            'features': str(feats),
            'n_features': len(feats),
            'accuracy': acc, 'acc_delta': acc_delta,
            'mean_entropy': ent, 'ent_delta': ent_delta,
            'n_flipped_to_correct': flipped_correct,
            'n_flipped_to_incorrect': flipped_incorrect,
        })

    # All layers combined
    saes = {}
    hooks = []
    for layer_idx, feats in selected_features.items():
        sae = load_sae(model_config, layer_idx, device=args.device)
        saes[layer_idx] = sae
        h = SAESuppressionHook(sae, feats, scale=args.scale)
        h.register(model, model_config, layer_idx)
        hooks.append(h)

    res = run_eval(model, tokenizer, samples, letter_token_ids,
                   args.device, "All layers")
    for h in hooks:
        h.remove()

    for sae in saes.values():
        del sae
    del saes
    torch.cuda.empty_cache()
    gc.collect()

    acc, ent = summarise(res)
    flipped_correct, flipped_incorrect = compute_flips(baseline, res)
    acc_delta = (acc - base_acc) * 100
    ent_delta = ent - base_ent

    print(f"\n  ALL LAYERS ({n_total} feats): "
          f"acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
          f"ent={ent:.4f} ({ent_delta:+.4f})  "
          f"->correct={flipped_correct} ->wrong={flipped_incorrect}")

    all_rows.append({
        'condition': 'all_layers_combined',
        'layer': 'all',
        'features': str(selected_features),
        'n_features': n_total,
        'accuracy': acc, 'acc_delta': acc_delta,
        'mean_entropy': ent, 'ent_delta': ent_delta,
        'n_flipped_to_correct': flipped_correct,
        'n_flipped_to_incorrect': flipped_incorrect,
    })

    # Save
    csv_file = output_dir / "cross_dataset_suppression.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved: {csv_file}")

    pkl_file = output_dir / "cross_dataset_suppression.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': vars(args),
            'source_selection': args.selection_pkl,
            'selected_features': selected_features,
            'baseline': {'accuracy': base_acc, 'mean_entropy': base_ent},
            'results': all_rows,
            'timestamp': datetime.now().isoformat(),
        }, f)
    print(f"Saved: {pkl_file}")
    print(f"\nDONE")


if __name__ == '__main__':
    main()
