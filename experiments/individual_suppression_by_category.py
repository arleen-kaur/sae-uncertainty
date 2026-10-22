"""
Individual Feature Suppression for Pure Uncertainty and Pure Incorrectness Categories

Suppress top-N features one-by-one from pure_uncertainty and pure_incorrectness
categories (from quadrant analysis) and measure accuracy/entropy impact.

Usage:
    python experiments/individual_suppression_by_category.py \
        --quadrant-pkl results/sae_quadrant_mmlu_discovery_all_layers_p25/quadrant_analysis.pkl \
        --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
        --output-dir results/sae_individual_suppression_by_category_p25/
"""

import csv
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import gc

from src.utils import seed_everything, load_eval_set, parse_layer_list, get_eval_letters
from src.model import load_model, get_letter_token_ids
from src.sae import SAESuppressionHook, load_sae
from src.evaluation import run_eval, summarise


def load_top_features_by_category(quadrant_pkl, layers, top_n=5):
    """Load top-N features per category per layer."""
    with open(quadrant_pkl, 'rb') as f:
        data = pickle.load(f)

    p_threshold = data['config']['p_threshold']
    features = {}  # {layer: {category: [(feat_idx, effect_size), ...]}}

    for layer in layers:
        if layer not in data['all_comparisons']:
            continue

        comp = data['all_comparisons'][layer]

        sig_unc = {r['feature_idx'] for r in comp['pure_uncertainty']
                   if r['p_value'] < p_threshold and r['effect_size'] > 0}
        sig_inc = {r['feature_idx'] for r in comp['pure_incorrectness']
                   if r['p_value'] < p_threshold and r['effect_size'] > 0}

        pure_unc = sig_unc - sig_inc
        pure_inc = sig_inc - sig_unc
        both_set = sig_unc & sig_inc

        unc_ranked = sorted(
            [r for r in comp['pure_uncertainty'] if r['feature_idx'] in pure_unc],
            key=lambda x: x['effect_size'], reverse=True
        )[:top_n]

        inc_ranked = sorted(
            [r for r in comp['pure_incorrectness'] if r['feature_idx'] in pure_inc],
            key=lambda x: x['effect_size'], reverse=True
        )[:top_n]

        # For "both", rank by min(unc_effect, inc_effect)
        inc_lookup = {r['feature_idx']: r['effect_size'] for r in comp['pure_incorrectness']}
        both_ranked = sorted(
            [r for r in comp['pure_uncertainty'] if r['feature_idx'] in both_set],
            key=lambda x: min(x['effect_size'], inc_lookup.get(x['feature_idx'], 0)),
            reverse=True
        )[:top_n]

        features[layer] = {
            'pure_uncertainty': [(r['feature_idx'], r['effect_size']) for r in unc_ranked],
            'pure_incorrectness': [(r['feature_idx'], r['effect_size']) for r in inc_ranked],
            'both': [(r['feature_idx'], min(r['effect_size'], inc_lookup.get(r['feature_idx'], 0)))
                     for r in both_ranked],
        }

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Individual Feature Suppression by Category"
    )
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--eval-set', type=str,
                        default='data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json')
    parser.add_argument('--quadrant-pkl', type=str,
                        default='results/sae_quadrant_mmlu_discovery_all_layers_p25/quadrant_analysis.pkl')
    parser.add_argument('--suppress-layers', type=str, default=None,
                        help='Layers to test (default: all layers in quadrant pkl)')
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--output-dir', type=str,
                        default='results/sae_individual_suppression_by_category_p25/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--category', type=str, default='all',
                        choices=['all', 'pure_uncertainty', 'pure_incorrectness', 'both'],
                        help='Which category to run (all=unc+inc, or a single category)')
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine layers
    if args.suppress_layers:
        suppress_layers = parse_layer_list(args.suppress_layers)
    else:
        with open(args.quadrant_pkl, 'rb') as f:
            data = pickle.load(f)
        suppress_layers = sorted(data['all_comparisons'].keys())

    # Load features
    print(f"Loading top-{args.top_n} features per category...")
    features = load_top_features_by_category(args.quadrant_pkl, suppress_layers, args.top_n)

    if args.category == 'all':
        categories = ['pure_uncertainty', 'pure_incorrectness', 'both']
    else:
        categories = [args.category]

    total = 0
    for cat in categories:
        n = sum(len(features[l][cat]) for l in features)
        print(f"  {cat}: {n} features")
        total += n
    print(f"  Total: {total} features to test")

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

    # Individual suppression
    all_rows = []
    csv_file = output_dir / "individual_suppression_by_category.csv"
    fieldnames = ['category', 'layer', 'feature_idx', 'effect_size',
                  'accuracy', 'acc_delta', 'mean_entropy', 'ent_delta']

    def save_partial():
        if all_rows:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

    for cat in categories:
        print(f"\n{'='*60}")
        print(f"Category: {cat}")
        print(f"{'='*60}")

        for layer_idx in suppress_layers:
            if layer_idx not in features or not features[layer_idx][cat]:
                continue

            sae = load_sae(model_config, layer_idx, device=args.device)

            for feat_idx, effect_size in features[layer_idx][cat]:
                hook = SAESuppressionHook(sae, [feat_idx], scale=args.scale)
                hook.register(model, model_config, layer_idx)

                res = run_eval(model, tokenizer, samples, letter_token_ids,
                               args.device, f"{cat} L{layer_idx} feat {feat_idx}")
                hook.remove()

                acc, ent = summarise(res)
                acc_delta = (acc - base_acc) * 100
                ent_delta = ent - base_ent

                print(f"  L{layer_idx} feat {feat_idx:>5} (d={effect_size:.2f}): "
                      f"acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
                      f"ent={ent:.4f} ({ent_delta:+.4f})")

                all_rows.append({
                    'category': cat,
                    'layer': layer_idx,
                    'feature_idx': feat_idx,
                    'effect_size': effect_size,
                    'accuracy': acc,
                    'acc_delta': acc_delta,
                    'mean_entropy': ent,
                    'ent_delta': ent_delta,
                })

            del sae
            torch.cuda.empty_cache()
            gc.collect()
            save_partial()

    # Final save
    save_partial()
    print(f"\nSaved: {csv_file}")

    pkl_file = output_dir / "individual_suppression_by_category.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': vars(args),
            'baseline': {'accuracy': base_acc, 'mean_entropy': base_ent},
            'results': all_rows,
            'timestamp': datetime.now().isoformat(),
        }, f)
    print(f"Saved: {pkl_file}")
    print(f"\nDONE")


if __name__ == '__main__':
    main()
