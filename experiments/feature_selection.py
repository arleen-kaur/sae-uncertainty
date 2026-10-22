"""
SAE "Both" Feature Selection via Individual Suppression

Test suppressing "both" category features one-by-one on a discovery set,
select beneficial ones (acc >= baseline AND ent < baseline), then suppress
selected features together on a held-out validation set.

Usage:
    python experiments/feature_selection.py \
        --model meta-llama/Llama-3.1-8B \
        --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
        --validation-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
        --quadrant-pkl results/sae_quadrant_mmlu_discovery/quadrant_analysis.pkl \
        --output-dir results/sae_both_feature_selection/
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
from src.model import load_model, get_letter_token_ids, get_default_sae_layers
from src.sae import SAESuppressionHook, load_sae
from src.quadrant import load_both_features
from src.evaluation import run_eval, summarise, compute_flips


def main():
    parser = argparse.ArgumentParser(
        description="SAE 'Both' Feature Selection via Individual Suppression"
    )
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--eval-set', type=str,
                        default='data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json',
                        help='Discovery set for individual feature suppression.')
    parser.add_argument('--validation-set', type=str,
                        default='data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json',
                        help='Held-out validation set for final combined suppression.')
    parser.add_argument('--quadrant-pkl', type=str,
                        default='results/sae_quadrant_mmlu_discovery/quadrant_analysis.pkl')
    parser.add_argument('--suppress-layers', type=str, default=None,
                        help='Layers to test (default: from model config, last 4)')
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--output-dir', type=str,
                        default='results/sae_both_feature_selection/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, model_config = load_model(
        args.model, device=args.device
    )

    # Determine suppress layers
    if args.suppress_layers:
        suppress_layers = [int(x) for x in args.suppress_layers.split(',')]
    else:
        default_layers = get_default_sae_layers(model_config)
        suppress_layers = default_layers[-4:]
    print(f"Suppress layers: {suppress_layers}")

    # Load data
    selection_set = load_eval_set(args.eval_set)
    validation_set = load_eval_set(args.validation_set)
    print(f"Discovery set:  {len(selection_set)} samples")
    print(f"Validation set: {len(validation_set)} samples")

    # Load both features
    print("\nLoading 'both' category features...")
    both_features = load_both_features(args.quadrant_pkl, suppress_layers)

    total_both = sum(len(v) for v in both_features.values())
    print(f"\nTotal 'both' features across all layers: {total_both}")
    if total_both == 0:
        print("No 'both' features found. Exiting.")
        return

    letter_token_ids = get_letter_token_ids(
        tokenizer, get_eval_letters(selection_set + validation_set))
    print(f"Letter token IDs: {letter_token_ids}")

    # =========================================================================
    # Baseline on selection set
    # =========================================================================
    print(f"\n{'='*60}")
    print("Baseline (selection set, no suppression)")
    print(f"{'='*60}")
    baseline_sel = run_eval(model, tokenizer, selection_set, letter_token_ids,
                            args.device, "Baseline (selection)")
    base_sel_acc, base_sel_ent = summarise(baseline_sel)
    print(f"Baseline accuracy: {base_sel_acc*100:.2f}%  mean entropy: {base_sel_ent:.4f}")

    # =========================================================================
    # Individual feature suppression on selection set
    # =========================================================================
    individual_results = []
    selected_features = {}

    for layer_idx in suppress_layers:
        if layer_idx not in both_features or not both_features[layer_idx]:
            print(f"\nLayer {layer_idx}: no 'both' features, skipping")
            continue

        print(f"\nLoading SAE for layer {layer_idx}...")
        sae = load_sae(model_config, layer_idx, device=args.device)

        layer_selected = []

        for feat_idx, unc_effect, inc_effect in both_features[layer_idx]:
            desc = f"L{layer_idx} feat {feat_idx}"
            hook = SAESuppressionHook(sae, [feat_idx], scale=args.scale)
            hook.register(model, model_config, layer_idx)

            res = run_eval(model, tokenizer, selection_set, letter_token_ids,
                           args.device, desc)
            hook.remove()

            acc, ent = summarise(res)
            acc_delta = (acc - base_sel_acc) * 100
            ent_delta = ent - base_sel_ent
            beneficial = (acc >= base_sel_acc) and (ent < base_sel_ent)

            print(f"  L{layer_idx} feat {feat_idx:>5}: "
                  f"acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
                  f"ent={ent:.4f} ({ent_delta:+.4f})  "
                  f"{'SELECTED' if beneficial else ''}")

            individual_results.append({
                'layer': layer_idx,
                'feature_idx': feat_idx,
                'unc_effect': unc_effect,
                'inc_effect': inc_effect,
                'acc_selection': acc,
                'acc_delta_selection': acc_delta,
                'ent_selection': ent,
                'ent_delta_selection': ent_delta,
                'beneficial': beneficial,
            })

            if beneficial:
                layer_selected.append(feat_idx)

        del sae
        torch.cuda.empty_cache()
        gc.collect()

        if layer_selected:
            selected_features[layer_idx] = layer_selected

    # Save individual results
    ind_csv = output_dir / "individual_suppression_selection.csv"
    if individual_results:
        with open(ind_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=individual_results[0].keys())
            writer.writeheader()
            writer.writerows(individual_results)
        print(f"\nSaved individual results: {ind_csv}")

    n_total = sum(len(v) for v in both_features.values())
    n_selected = sum(len(v) for v in selected_features.values())
    print(f"\n{'='*60}")
    print(f"Feature selection summary: {n_selected}/{n_total} features selected")
    for layer_idx, feats in selected_features.items():
        print(f"  Layer {layer_idx}: {feats}")
    print(f"{'='*60}")

    if n_selected == 0:
        print("No beneficial features found. Skipping validation run.")
        pkl_file = output_dir / "both_feature_selection.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump({
                'config': vars(args),
                'both_features': both_features,
                'selected_features': selected_features,
                'individual_results': individual_results,
                'validation_results': None,
                'timestamp': datetime.now().isoformat(),
            }, f)
        print(f"Saved: {pkl_file}")
        return

    # =========================================================================
    # Validation
    # =========================================================================
    print(f"\n{'='*60}")
    print("Baseline (validation set, no suppression)")
    print(f"{'='*60}")
    baseline_val = run_eval(model, tokenizer, validation_set, letter_token_ids,
                            args.device, "Baseline (validation)")
    base_val_acc, base_val_ent = summarise(baseline_val)
    print(f"Baseline accuracy: {base_val_acc*100:.2f}%  mean entropy: {base_val_ent:.4f}")

    # Load SAEs for selected layers
    saes = {}
    for layer_idx in selected_features:
        print(f"  Loading SAE for layer {layer_idx}...")
        saes[layer_idx] = load_sae(model_config, layer_idx, device=args.device)

    def run_suppression(layer_feats_dict, desc):
        hooks = []
        for lidx, feats in layer_feats_dict.items():
            h = SAESuppressionHook(saes[lidx], feats, scale=args.scale)
            h.register(model, model_config, lidx)
            hooks.append(h)
        res = run_eval(model, tokenizer, validation_set, letter_token_ids, args.device, desc)
        for h in hooks:
            h.remove()
        acc, ent = summarise(res)
        return acc, ent, res

    validation_rows = []

    def record(condition, layer, feats, acc, ent, results):
        acc_delta = (acc - base_val_acc) * 100
        ent_delta = ent - base_val_ent
        flipped_correct, flipped_incorrect = compute_flips(baseline_val, results)
        print(f"  {condition}: acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
              f"ent={ent:.4f} ({ent_delta:+.4f})  "
              f"->correct={flipped_correct}  ->wrong={flipped_incorrect}")
        validation_rows.append({
            'condition': condition, 'layer': layer,
            'features': str(feats),
            'n_features': len(feats) if isinstance(feats, list) else sum(len(v) for v in feats.values()),
            'accuracy': acc, 'acc_delta': acc_delta,
            'mean_entropy': ent, 'ent_delta': ent_delta,
            'n_flipped_to_correct': flipped_correct,
            'n_flipped_to_incorrect': flipped_incorrect,
        })

    # Per-layer
    for layer_idx, feats in selected_features.items():
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx} selected features: {feats}")
        acc, ent, res = run_suppression({layer_idx: feats}, f"L{layer_idx} combined")
        record(f"L{layer_idx}_combined", layer_idx, feats, acc, ent, res)

    # All layers combined
    print(f"\n{'='*60}")
    print(f"All layers combined: {selected_features}")
    acc, ent, res = run_suppression(selected_features, "All layers combined")
    record("all_layers_combined", "all", selected_features, acc, ent, res)

    for sae in saes.values():
        del sae
    del saes
    torch.cuda.empty_cache()
    gc.collect()

    # Save
    val_csv = output_dir / "validation_suppression.csv"
    with open(val_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=validation_rows[0].keys())
        writer.writeheader()
        writer.writerows(validation_rows)
    print(f"\nSaved: {val_csv}")

    pkl_file = output_dir / "both_feature_selection.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': vars(args),
            'both_features': both_features,
            'selected_features': selected_features,
            'baseline_selection': {'accuracy': base_sel_acc, 'mean_entropy': base_sel_ent},
            'baseline_validation': {'accuracy': base_val_acc, 'mean_entropy': base_val_ent},
            'individual_results': individual_results,
            'validation_rows': validation_rows,
            'timestamp': datetime.now().isoformat(),
        }, f)
    print(f"Saved: {pkl_file}")
    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == '__main__':
    main()
