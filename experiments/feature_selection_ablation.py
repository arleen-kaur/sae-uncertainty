"""
Feature Selection Criteria Ablation

Reuses individual suppression results from feature_selection.py and tests
different selection criteria on the held-out validation set:
  1. acc >= baseline AND ent < baseline  (original, strict)
  2. ent < baseline only                 (entropy-only)
  3. acc >= baseline only                (accuracy-only)

This avoids re-running the expensive individual suppression (102 features x 7K questions).
Only the combined validation passes are run (~20 layers x 7K questions per criterion).

Usage:
    python experiments/feature_selection_ablation.py \
        --individual-csv results/sae_both_feature_selection_all_layers_p25/individual_suppression_selection.csv \
        --validation-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
        --output-dir results/sae_feature_selection_ablation_p25/
"""

import csv
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse
import gc

from src.utils import seed_everything, load_eval_set, get_eval_letters
from src.model import load_model, get_letter_token_ids
from src.sae import SAESuppressionHook, load_sae
from src.evaluation import run_eval, summarise, compute_flips


CRITERIA = {
    'acc_and_ent': lambda row: row['acc_delta_selection'] >= 0 and row['ent_delta_selection'] < 0,
    'ent_only':    lambda row: row['ent_delta_selection'] < 0,
    'acc_only':    lambda row: row['acc_delta_selection'] >= 0,
}


def load_individual_results(csv_path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            row['layer'] = int(row['layer'])
            row['feature_idx'] = int(row['feature_idx'])
            row['acc_delta_selection'] = float(row['acc_delta_selection'])
            row['ent_delta_selection'] = float(row['ent_delta_selection'])
            rows.append(row)
    return rows


def select_features(individual_results, criterion_fn):
    selected = defaultdict(list)
    for row in individual_results:
        if criterion_fn(row):
            selected[row['layer']].append(row['feature_idx'])
    return dict(selected)


def main():
    parser = argparse.ArgumentParser(description="Feature Selection Criteria Ablation")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--individual-csv', type=str, required=True,
                        help='Path to individual_suppression_selection.csv from feature_selection.py')
    parser.add_argument('--validation-set', type=str,
                        default='data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json')
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--output-dir', type=str,
                        default='results/sae_feature_selection_ablation_p25/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load individual results
    individual_results = load_individual_results(args.individual_csv)
    print(f"Loaded {len(individual_results)} individual results from {args.individual_csv}")

    # Apply each criterion
    for name, fn in CRITERIA.items():
        selected = select_features(individual_results, fn)
        n = sum(len(v) for v in selected.values())
        print(f"\n  {name}: {n} features across {len(selected)} layers")
        for layer, feats in sorted(selected.items()):
            print(f"    L{layer}: {feats}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, model_config = load_model(args.model, device=args.device)

    # Load validation set
    validation_set = load_eval_set(args.validation_set)
    letter_token_ids = get_letter_token_ids(tokenizer, get_eval_letters(validation_set))

    # Baseline
    print(f"\nBaseline (validation set, no suppression)")
    baseline_val = run_eval(model, tokenizer, validation_set, letter_token_ids,
                            args.device, "Baseline (validation)")
    base_acc, base_ent = summarise(baseline_val)
    print(f"Baseline accuracy: {base_acc*100:.2f}%  mean entropy: {base_ent:.4f}")

    all_rows = []
    csv_file = output_dir / "selection_ablation.csv"

    def save_partial():
        if all_rows:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"  Saved partial results: {csv_file}")

    for crit_name, crit_fn in CRITERIA.items():
        print(f"\n{'='*60}")
        print(f"Criterion: {crit_name}")
        print(f"{'='*60}")

        selected = select_features(individual_results, crit_fn)
        n_selected = sum(len(v) for v in selected.values())

        if n_selected == 0:
            print("  No features selected, skipping.")
            continue

        # Per-layer suppression (load one SAE at a time)
        for layer_idx, feats in sorted(selected.items()):
            sae = load_sae(model_config, layer_idx, device=args.device)
            h = SAESuppressionHook(sae, feats, scale=args.scale)
            h.register(model, model_config, layer_idx)

            res = run_eval(model, tokenizer, validation_set, letter_token_ids,
                           args.device, f"{crit_name} L{layer_idx}")
            h.remove()
            del sae
            torch.cuda.empty_cache()
            gc.collect()

            acc, ent = summarise(res)
            flipped_correct, flipped_incorrect = compute_flips(baseline_val, res)
            acc_delta = (acc - base_acc) * 100
            ent_delta = ent - base_ent

            print(f"  L{layer_idx} ({len(feats)} feats): "
                  f"acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
                  f"ent={ent:.4f} ({ent_delta:+.4f})  "
                  f"->correct={flipped_correct} ->wrong={flipped_incorrect}")

            all_rows.append({
                'criterion': crit_name,
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
        hooks = []
        saes = {}
        for layer_idx, feats in selected.items():
            sae = load_sae(model_config, layer_idx, device=args.device)
            saes[layer_idx] = sae
            h = SAESuppressionHook(sae, feats, scale=args.scale)
            h.register(model, model_config, layer_idx)
            hooks.append(h)

        res = run_eval(model, tokenizer, validation_set, letter_token_ids,
                       args.device, f"{crit_name} all layers")
        for hook in hooks:
            hook.remove()

        for sae in saes.values():
            del sae
        del saes
        torch.cuda.empty_cache()
        gc.collect()

        acc, ent = summarise(res)
        flipped_correct, flipped_incorrect = compute_flips(baseline_val, res)
        acc_delta = (acc - base_acc) * 100
        ent_delta = ent - base_ent

        print(f"\n  ALL LAYERS ({n_selected} feats): "
              f"acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
              f"ent={ent:.4f} ({ent_delta:+.4f})  "
              f"->correct={flipped_correct} ->wrong={flipped_incorrect}")

        all_rows.append({
            'criterion': crit_name,
            'condition': 'all_layers_combined',
            'layer': 'all',
            'features': str(selected),
            'n_features': n_selected,
            'accuracy': acc, 'acc_delta': acc_delta,
            'mean_entropy': ent, 'ent_delta': ent_delta,
            'n_flipped_to_correct': flipped_correct,
            'n_flipped_to_incorrect': flipped_incorrect,
        })
        save_partial()

    # Final save
    save_partial()

    pkl_file = output_dir / "selection_ablation.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': vars(args),
            'baseline': {'accuracy': base_acc, 'mean_entropy': base_ent},
            'results': all_rows,
            'timestamp': datetime.now().isoformat(),
        }, f)
    print(f"Saved: {pkl_file}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY (all_layers_combined)")
    print(f"{'='*60}")
    print(f"{'Criterion':<15} {'#Feats':>6} {'Acc Delta':>10} {'Entropy':>8} {'Ent Delta':>10}")
    print("-" * 55)
    for row in all_rows:
        if row['condition'] == 'all_layers_combined':
            print(f"{row['criterion']:<15} {row['n_features']:>6} "
                  f"{row['acc_delta']:>+9.2f}% {row['mean_entropy']:>8.4f} "
                  f"{row['ent_delta']:>+9.4f}")

    print(f"\nDONE")


if __name__ == '__main__':
    main()
