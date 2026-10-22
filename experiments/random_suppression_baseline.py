"""
Random Feature Suppression Baseline (with screening)

Fair comparison against our method: randomly select the same number of features
per layer as the real experiment tested, then apply the same screening pipeline
(individual suppression → keep beneficial → combined suppression).

The only difference from our method is HOW features are selected:
  - Ours: quadrant analysis (statistical categorization)
  - Baseline: random selection from full SAE dictionary

Runs multiple seeds and reports mean ± std.

Usage:
    python experiments/random_suppression_baseline.py \
        --individual-csv results/sae_both_feature_selection_all_layers_p25/individual_suppression_selection.csv \
        --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
        --discovery-eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
        --n-seeds 5 \
        --output-dir results/sae_random_baseline_mmlu_p25/
"""

import csv
import torch
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse
import gc

from src.utils import seed_everything, load_eval_set, get_eval_letters
from src.model import load_model, get_letter_token_ids
from src.sae import SAESuppressionHook, load_sae
from src.evaluation import run_eval, summarise, compute_flips


SAE_SIZES = {
    'meta-llama/Llama-3.1-8B': 32768,
    'google/gemma-2-9b': 16384,
}


def get_per_layer_tested_counts(individual_csv):
    """Extract per-layer TESTED feature counts from the real experiment."""
    tested = defaultdict(list)
    with open(individual_csv) as f:
        for row in csv.DictReader(f):
            tested[int(row['layer'])].append(int(row['feature_idx']))
    return dict(tested)


def main():
    parser = argparse.ArgumentParser(
        description="Random Feature Suppression Baseline (with screening)"
    )
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--individual-csv', type=str, required=True,
                        help='CSV from the real experiment (to match per-layer tested counts)')
    parser.add_argument('--eval-set', type=str, required=True,
                        help='Validation eval set for final combined suppression')
    parser.add_argument('--discovery-eval-set', type=str, required=True,
                        help='Discovery eval set for individual screening')
    parser.add_argument('--discovery-pkl', type=str, default=None,
                        help='Feature extraction pkl for discovery set '
                             '(to sample from active features only)')
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--output-dir', type=str,
                        default='results/sae_random_baseline_p25/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--label', type=str, default='')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get per-layer tested counts from real experiment
    real_tested = get_per_layer_tested_counts(args.individual_csv)
    total_tested = sum(len(v) for v in real_tested.values())
    print(f"Real experiment tested: {total_tested} features across {len(real_tested)} layers")
    for l in sorted(real_tested):
        print(f"  L{l}: {len(real_tested[l])} features")

    # Build pool of active features per layer from discovery pkl
    if args.discovery_pkl:
        print(f"\nLoading active features from {args.discovery_pkl}...")
        with open(args.discovery_pkl, 'rb') as f:
            disc_data = pickle.load(f)
        active_features_per_layer = {}
        for layer_idx in real_tested.keys():
            active = set()
            for q in disc_data['per_question_data']:
                if layer_idx in q['sae_features']:
                    active.update(q['sae_features'][layer_idx]['indices'].tolist())
            active_features_per_layer[layer_idx] = sorted(active)
            print(f"  L{layer_idx}: {len(active)} active features")
        del disc_data
    else:
        sae_size = SAE_SIZES.get(args.model)
        if sae_size is None:
            raise ValueError(f"Unknown SAE size for {args.model}. Add to SAE_SIZES dict.")
        active_features_per_layer = {l: list(range(sae_size)) for l in real_tested.keys()}
        print(f"\nNo discovery pkl provided — sampling from all {sae_size} features")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, model_config = load_model(args.model, device=args.device)

    # Load discovery set (for individual screening)
    discovery_samples = load_eval_set(args.discovery_eval_set)
    print(f"Discovery set: {len(discovery_samples)} samples")

    # Load validation set (for final combined evaluation)
    validation_samples = load_eval_set(args.eval_set)
    print(f"Validation set: {len(validation_samples)} samples")

    letter_token_ids = get_letter_token_ids(
        tokenizer, get_eval_letters(discovery_samples + validation_samples))

    # ---- Phase 1: Discovery baseline ----
    print(f"\nDiscovery baseline (no suppression)")
    disc_baseline = run_eval(model, tokenizer, discovery_samples, letter_token_ids,
                             args.device, "Discovery baseline")
    disc_base_acc, disc_base_ent = summarise(disc_baseline)
    print(f"Discovery baseline: {disc_base_acc*100:.2f}%  entropy: {disc_base_ent:.4f}")

    # ---- Phase 2: Validation baseline ----
    print(f"\nValidation baseline (no suppression)")
    val_baseline = run_eval(model, tokenizer, validation_samples, letter_token_ids,
                            args.device, "Validation baseline")
    val_base_acc, val_base_ent = summarise(val_baseline)
    print(f"Validation baseline: {val_base_acc*100:.2f}%  entropy: {val_base_ent:.4f}")

    # Run random baselines
    all_results = []

    for seed_idx in range(args.n_seeds):
        rng = np.random.default_rng(seed_idx)
        print(f"\n{'='*60}")
        print(f"SEED {seed_idx} ({seed_idx+1}/{args.n_seeds})")
        print(f"{'='*60}")

        # ---- Step 1: Randomly select features from active pool ----
        random_features = {}
        for layer_idx, real_feats in real_tested.items():
            k = len(real_feats)
            pool = active_features_per_layer[layer_idx]
            if k > len(pool):
                print(f"  WARNING: L{layer_idx} needs {k} but only {len(pool)} active. Using all.")
                random_features[layer_idx] = pool
            else:
                chosen = rng.choice(len(pool), size=k, replace=False)
                random_features[layer_idx] = [pool[i] for i in chosen]

        # ---- Step 2: Individual screening on discovery set ----
        print(f"\nScreening {total_tested} random features individually...")
        beneficial = defaultdict(list)
        individual_results = []

        for layer_idx in sorted(random_features.keys()):
            sae = load_sae(model_config, layer_idx, device=args.device)

            for feat_idx in random_features[layer_idx]:
                hook = SAESuppressionHook(sae, [feat_idx], scale=args.scale)
                hook.register(model, model_config, layer_idx)

                res = run_eval(model, tokenizer, discovery_samples, letter_token_ids,
                               args.device, f"Screen L{layer_idx} feat {feat_idx}")
                hook.remove()

                acc, ent = summarise(res)
                acc_delta = (acc - disc_base_acc) * 100
                ent_delta = ent - disc_base_ent

                is_beneficial = acc_delta >= 0 and ent_delta < 0
                if is_beneficial:
                    beneficial[layer_idx].append(feat_idx)

                individual_results.append({
                    'layer': layer_idx,
                    'feature_idx': feat_idx,
                    'accuracy': acc,
                    'acc_delta': acc_delta,
                    'mean_entropy': ent,
                    'ent_delta': ent_delta,
                    'beneficial': is_beneficial,
                })

                tag = " [BENEFICIAL]" if is_beneficial else ""
                print(f"  L{layer_idx} feat {feat_idx:>5}: "
                      f"acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
                      f"ent={ent:.4f} ({ent_delta:+.4f}){tag}")

            del sae
            torch.cuda.empty_cache()
            gc.collect()

        n_beneficial = sum(len(v) for v in beneficial.values())
        print(f"\nScreening done: {n_beneficial}/{total_tested} passed")

        if n_beneficial == 0:
            print("No beneficial features found. Skipping combined suppression.")
            all_results.append({
                'seed': seed_idx,
                'n_tested': total_tested,
                'n_beneficial': 0,
                'accuracy': val_base_acc,
                'acc_delta': 0.0,
                'mean_entropy': val_base_ent,
                'ent_delta': 0.0,
                'flipped_correct': 0,
                'flipped_incorrect': 0,
                'random_features': dict(random_features),
                'beneficial_features': {},
                'individual_results': individual_results,
            })
            continue

        # ---- Step 3: Combined suppression on validation set ----
        print(f"\nCombined suppression ({n_beneficial} features) on validation set...")
        hooks = []
        saes = {}
        for layer_idx, feats in beneficial.items():
            sae = load_sae(model_config, layer_idx, device=args.device)
            saes[layer_idx] = sae
            h = SAESuppressionHook(sae, feats, scale=args.scale)
            h.register(model, model_config, layer_idx)
            hooks.append(h)

        res = run_eval(model, tokenizer, validation_samples, letter_token_ids,
                       args.device, f"Random seed {seed_idx} combined")
        for h in hooks:
            h.remove()

        acc, ent = summarise(res)
        flipped_correct, flipped_incorrect = compute_flips(val_baseline, res)
        acc_delta = (acc - val_base_acc) * 100
        ent_delta = ent - val_base_ent

        print(f"\n  Seed {seed_idx} result ({n_beneficial} beneficial features):")
        print(f"    Acc: {acc*100:.2f}% ({acc_delta:+.2f}%)")
        print(f"    Ent: {ent:.4f} ({ent_delta:+.4f})")
        print(f"    Flipped: ->correct={flipped_correct} ->wrong={flipped_incorrect}")

        all_results.append({
            'seed': seed_idx,
            'n_tested': total_tested,
            'n_beneficial': n_beneficial,
            'accuracy': acc,
            'acc_delta': acc_delta,
            'mean_entropy': ent,
            'ent_delta': ent_delta,
            'flipped_correct': flipped_correct,
            'flipped_incorrect': flipped_incorrect,
            'random_features': dict(random_features),
            'beneficial_features': dict(beneficial),
            'individual_results': individual_results,
        })

        for sae in saes.values():
            del sae
        del saes
        torch.cuda.empty_cache()
        gc.collect()

    # ---- Summary ----
    acc_deltas = [r['acc_delta'] for r in all_results]
    ent_deltas = [r['ent_delta'] for r in all_results]
    n_beneficials = [r['n_beneficial'] for r in all_results]

    print(f"\n{'='*60}")
    print(f"RANDOM BASELINE SUMMARY ({args.label})")
    print(f"  Seeds: {args.n_seeds}")
    print(f"  Features tested per run: {total_tested}")
    print(f"  Beneficial per run: {np.mean(n_beneficials):.1f} ± {np.std(n_beneficials):.1f}")
    print(f"  Acc delta: {np.mean(acc_deltas):+.3f}% ± {np.std(acc_deltas):.3f}%")
    print(f"  Ent delta: {np.mean(ent_deltas):+.4f} ± {np.std(ent_deltas):.4f}")
    print(f"  Per-seed acc: {[f'{d:+.2f}%' for d in acc_deltas]}")
    print(f"  Per-seed beneficial: {n_beneficials}")
    print(f"{'='*60}")

    # Save CSV
    csv_file = output_dir / 'random_baseline_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'seed', 'n_tested', 'n_beneficial', 'accuracy', 'acc_delta',
            'mean_entropy', 'ent_delta', 'flipped_correct', 'flipped_incorrect',
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: v for k, v in r.items()
                             if k not in ('beneficial_features', 'random_features', 'individual_results')})

    # Save pkl
    pkl_file = output_dir / 'random_baseline_results.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': vars(args),
            'real_per_layer_counts': {l: len(v) for l, v in real_tested.items()},
            'baseline_discovery': {'accuracy': disc_base_acc, 'mean_entropy': disc_base_ent},
            'baseline_validation': {'accuracy': val_base_acc, 'mean_entropy': val_base_ent},
            'results': all_results,
            'summary': {
                'acc_delta_mean': float(np.mean(acc_deltas)),
                'acc_delta_std': float(np.std(acc_deltas)),
                'ent_delta_mean': float(np.mean(ent_deltas)),
                'ent_delta_std': float(np.std(ent_deltas)),
                'n_beneficial_mean': float(np.mean(n_beneficials)),
                'n_beneficial_std': float(np.std(n_beneficials)),
            },
            'timestamp': datetime.now().isoformat(),
        }, f)

    print(f"\nSaved: {csv_file}")
    print(f"Saved: {pkl_file}")


if __name__ == '__main__':
    main()
