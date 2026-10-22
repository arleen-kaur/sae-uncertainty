"""
SAE Uncertainty Feature Discovery

Capture residual stream activations at specified layers, encode through pretrained SAEs,
and compute differential feature statistics between correct/incorrect MCQ predictions.

Usage:
    python experiments/extract_features.py \
        --model meta-llama/Llama-3.1-8B \
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_1094.json \
        --output-dir results/sae_uncertainty_1094/
"""

import csv
import json
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
from scipy import stats

from src.utils import seed_everything, load_eval_set, parse_layer_list, get_eval_letters
from src.model import load_model, get_letter_token_ids, get_default_sae_layers
from src.sae import ResidualStreamHook, load_sae
from src.evaluation import get_predicted_letter, compute_entropy


def main():
    parser = argparse.ArgumentParser(description="SAE Uncertainty Feature Discovery")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--eval-set', type=str,
                        default='data/eval_sets/eval_set_mcq_arc_challenge_500.json')
    parser.add_argument('--sae-layers', type=str, default=None,
                        help='Layers to capture SAE features at (default: from model config)')
    parser.add_argument('--top-k-features', type=int, default=100,
                        help='Number of top differential features to report per layer')
    parser.add_argument('--output-dir', type=str, default='results/sae_uncertainty/')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    seed_everything(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, model_config = load_model(
        args.model, device=args.device
    )

    # Determine SAE layers
    if args.sae_layers:
        sae_layers = parse_layer_list(args.sae_layers)
    else:
        sae_layers = get_default_sae_layers(model_config)
    print(f"SAE layers: {sae_layers}")

    # Load eval set
    samples = load_eval_set(args.eval_set)

    # Map answer letters to token IDs
    letter_token_ids = get_letter_token_ids(tokenizer, get_eval_letters(samples))
    print(f"Letter token IDs: {letter_token_ids}")

    # =========================================================================
    # Phase 1: Run inference and cache activations to CPU
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Running inference on {len(samples)} questions...")
    print(f"Capturing residual stream at layers: {sae_layers}")
    print(f"{'=' * 60}")

    hook = ResidualStreamHook()
    hook.register(model, model_config, sae_layers)

    per_question_data = []
    cached_activations = {li: [] for li in sae_layers}

    for q_idx, sample in enumerate(tqdm(samples, desc="Inference")):
        prompt = sample.get('mcq_prompt', sample.get('prompt', ''))
        correct_letter = sample.get('correct_letter', None)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1, :].detach()
        predicted = get_predicted_letter(logits, letter_token_ids)
        correct = (predicted == correct_letter) if correct_letter else None
        entropy = compute_entropy(logits, letter_token_ids)

        abcd_logits = torch.tensor([logits[tid].item() for tid in letter_token_ids.values()])
        abcd_probs = torch.softmax(abcd_logits, dim=0).tolist()

        for layer_idx in sae_layers:
            if layer_idx in hook.hidden_states:
                cached_activations[layer_idx].append(
                    hook.hidden_states[layer_idx].squeeze(0).cpu()
                )

        per_question_data.append({
            'q_idx': q_idx,
            'predicted_letter': predicted,
            'correct_letter': correct_letter,
            'correct': correct,
            'entropy': entropy,
            'abcd_probs': abcd_probs,
            'sae_features': {},
        })

    hook.remove_all()

    # Free model from GPU to make room for SAEs
    print("\nFreeing model from GPU memory...")
    model.cpu()
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # =========================================================================
    # Phase 2: Load SAEs one at a time and encode cached activations
    # =========================================================================
    d_sae = None

    for layer_idx in sae_layers:
        print(f"\nPhase 2: Loading SAE for layer {layer_idx}")
        sae = load_sae(model_config, layer_idx, device=args.device)
        if d_sae is None:
            d_sae = sae.cfg.d_sae
            print(f"SAE feature dimension: {d_sae}")

        activations = cached_activations[layer_idx]
        print(f"  Encoding {len(activations)} activations...")
        for q_idx, act in enumerate(tqdm(activations, desc=f"SAE L{layer_idx}", leave=False)):
            activation = act.unsqueeze(0).float().to(args.device)
            with torch.no_grad():
                feat_acts = sae.encode(activation)
            feat_acts = feat_acts.squeeze(0)

            nonzero_mask = feat_acts > 0
            nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            nonzero_values = feat_acts[nonzero_mask].float().cpu().numpy()

            per_question_data[q_idx]['sae_features'][layer_idx] = {
                'indices': nonzero_indices,
                'values': nonzero_values,
                'n_active': len(nonzero_indices),
                'total_activation': float(feat_acts.sum().item()),
                'max_activation': float(feat_acts.max().item()),
            }

        del sae
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  Layer {layer_idx} done, SAE unloaded.")

    del cached_activations
    gc.collect()

    # =========================================================================
    # Analysis: Find differential features
    # =========================================================================
    n_correct = sum(1 for d in per_question_data if d['correct'])
    n_incorrect = sum(1 for d in per_question_data if d['correct'] is False)
    n_total = len(per_question_data)
    print(f"\nBaseline accuracy: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")

    correct_qs = [d for d in per_question_data if d['correct']]
    incorrect_qs = [d for d in per_question_data if d['correct'] is False]

    correct_entropies = [d['entropy'] for d in correct_qs]
    incorrect_entropies = [d['entropy'] for d in incorrect_qs]
    print(f"Mean entropy — correct: {np.mean(correct_entropies):.4f}, "
          f"incorrect: {np.mean(incorrect_entropies):.4f}")

    all_entropies = [d['entropy'] for d in per_question_data]
    entropy_q25 = np.percentile(all_entropies, 25)
    entropy_q75 = np.percentile(all_entropies, 75)
    low_entropy_qs = [d for d in per_question_data if d['entropy'] <= entropy_q25]
    high_entropy_qs = [d for d in per_question_data if d['entropy'] >= entropy_q75]
    print(f"Entropy quartiles: Q25={entropy_q25:.4f}, Q75={entropy_q75:.4f}")
    print(f"Low entropy (Q1): {len(low_entropy_qs)} questions, "
          f"accuracy={sum(d['correct'] for d in low_entropy_qs)/len(low_entropy_qs)*100:.1f}%")
    print(f"High entropy (Q4): {len(high_entropy_qs)} questions, "
          f"accuracy={sum(d['correct'] for d in high_entropy_qs)/len(high_entropy_qs)*100:.1f}%")

    # Per-layer differential feature analysis
    csv_rows = []
    differential_features = {}

    for layer_idx in sae_layers:
        print(f"\n--- Layer {layer_idx} ---")

        active_features = set()
        for d in per_question_data:
            if layer_idx in d['sae_features']:
                active_features.update(d['sae_features'][layer_idx]['indices'].tolist())
        active_features = sorted(active_features)
        print(f"  Active features: {len(active_features)} / {d_sae}")

        if not active_features:
            continue

        feat_to_col = {f: i for i, f in enumerate(active_features)}
        n_active = len(active_features)

        correct_acts = np.zeros((len(correct_qs), n_active))
        incorrect_acts = np.zeros((len(incorrect_qs), n_active))

        for i, d in enumerate(correct_qs):
            if layer_idx in d['sae_features']:
                sf = d['sae_features'][layer_idx]
                for idx, val in zip(sf['indices'], sf['values']):
                    if idx in feat_to_col:
                        correct_acts[i, feat_to_col[idx]] = val

        for i, d in enumerate(incorrect_qs):
            if layer_idx in d['sae_features']:
                sf = d['sae_features'][layer_idx]
                for idx, val in zip(sf['indices'], sf['values']):
                    if idx in feat_to_col:
                        incorrect_acts[i, feat_to_col[idx]] = val

        layer_results = []
        for col_idx, feat_idx in enumerate(active_features):
            c_vals = correct_acts[:, col_idx]
            i_vals = incorrect_acts[:, col_idx]

            c_mean = c_vals.mean()
            i_mean = i_vals.mean()
            c_freq = (c_vals > 0).mean()
            i_freq = (i_vals > 0).mean()

            if c_freq < 0.01 and i_freq < 0.01:
                continue

            try:
                u_stat, p_value = stats.mannwhitneyu(c_vals, i_vals, alternative='two-sided')
            except ValueError:
                p_value = 1.0

            pooled_std = np.sqrt((c_vals.std() ** 2 + i_vals.std() ** 2) / 2)
            effect_size = (i_mean - c_mean) / pooled_std if pooled_std > 0 else 0.0

            layer_results.append({
                'feature_idx': feat_idx,
                'correct_mean_act': float(c_mean),
                'incorrect_mean_act': float(i_mean),
                'correct_freq': float(c_freq),
                'incorrect_freq': float(i_freq),
                'freq_diff': float(i_freq - c_freq),
                'mean_diff': float(i_mean - c_mean),
                'effect_size': float(effect_size),
                'p_value': float(p_value),
            })

        layer_results.sort(key=lambda x: abs(x['effect_size']), reverse=True)
        differential_features[layer_idx] = layer_results

        top_k = min(args.top_k_features, len(layer_results))
        sig_features = [r for r in layer_results if r['p_value'] < 0.05]
        print(f"  Significant features (p<0.05): {len(sig_features)}")

        uncertainty_feats = [r for r in layer_results if r['effect_size'] > 0]
        confidence_feats = [r for r in layer_results if r['effect_size'] < 0]
        print(f"  Uncertainty features (more active on incorrect): {len(uncertainty_feats)}")
        print(f"  Confidence features (more active on correct): {len(confidence_feats)}")

        if layer_results:
            top = layer_results[0]
            direction = "uncertainty" if top['effect_size'] > 0 else "confidence"
            print(f"  Top feature: #{top['feature_idx']} "
                  f"(effect={top['effect_size']:.3f}, p={top['p_value']:.4f}, "
                  f"type={direction})")

        for rank, r in enumerate(layer_results[:top_k]):
            csv_rows.append({
                'layer': layer_idx,
                'rank': rank,
                'feature_idx': r['feature_idx'],
                'correct_mean_act': r['correct_mean_act'],
                'incorrect_mean_act': r['incorrect_mean_act'],
                'correct_freq': r['correct_freq'],
                'incorrect_freq': r['incorrect_freq'],
                'freq_diff': r['freq_diff'],
                'mean_diff': r['mean_diff'],
                'effect_size': r['effect_size'],
                'p_value': r['p_value'],
                'direction': 'uncertainty' if r['effect_size'] > 0 else 'confidence',
            })

    # Entropy-based analysis
    print(f"\n{'=' * 60}")
    print("Entropy-based analysis (high vs low entropy)")
    print(f"{'=' * 60}")

    entropy_csv_rows = []
    for layer_idx in sae_layers:
        active_features = set()
        for d in per_question_data:
            if layer_idx in d['sae_features']:
                active_features.update(d['sae_features'][layer_idx]['indices'].tolist())
        active_features = sorted(active_features)
        if not active_features:
            continue

        feat_to_col = {f: i for i, f in enumerate(active_features)}
        n_active = len(active_features)

        low_acts = np.zeros((len(low_entropy_qs), n_active))
        high_acts = np.zeros((len(high_entropy_qs), n_active))

        for i, d in enumerate(low_entropy_qs):
            if layer_idx in d['sae_features']:
                sf = d['sae_features'][layer_idx]
                for idx, val in zip(sf['indices'], sf['values']):
                    if idx in feat_to_col:
                        low_acts[i, feat_to_col[idx]] = val

        for i, d in enumerate(high_entropy_qs):
            if layer_idx in d['sae_features']:
                sf = d['sae_features'][layer_idx]
                for idx, val in zip(sf['indices'], sf['values']):
                    if idx in feat_to_col:
                        high_acts[i, feat_to_col[idx]] = val

        layer_results = []
        for col_idx, feat_idx in enumerate(active_features):
            l_vals = low_acts[:, col_idx]
            h_vals = high_acts[:, col_idx]
            l_freq = (l_vals > 0).mean()
            h_freq = (h_vals > 0).mean()
            if l_freq < 0.01 and h_freq < 0.01:
                continue

            try:
                _, p_value = stats.mannwhitneyu(l_vals, h_vals, alternative='two-sided')
            except ValueError:
                p_value = 1.0

            l_mean = l_vals.mean()
            h_mean = h_vals.mean()
            pooled_std = np.sqrt((l_vals.std()**2 + h_vals.std()**2) / 2)
            effect_size = (h_mean - l_mean) / pooled_std if pooled_std > 0 else 0.0

            layer_results.append({
                'feature_idx': feat_idx,
                'low_entropy_mean': float(l_mean),
                'high_entropy_mean': float(h_mean),
                'low_entropy_freq': float(l_freq),
                'high_entropy_freq': float(h_freq),
                'effect_size': float(effect_size),
                'p_value': float(p_value),
            })

        layer_results.sort(key=lambda x: abs(x['effect_size']), reverse=True)
        sig = [r for r in layer_results if r['p_value'] < 0.05]
        print(f"  Layer {layer_idx}: {len(sig)} significant features (p<0.05)")

        for rank, r in enumerate(layer_results[:args.top_k_features]):
            entropy_csv_rows.append({
                'layer': layer_idx,
                'rank': rank,
                'feature_idx': r['feature_idx'],
                'low_entropy_mean': r['low_entropy_mean'],
                'high_entropy_mean': r['high_entropy_mean'],
                'low_entropy_freq': r['low_entropy_freq'],
                'high_entropy_freq': r['high_entropy_freq'],
                'effect_size': r['effect_size'],
                'p_value': r['p_value'],
                'direction': 'high_entropy' if r['effect_size'] > 0 else 'low_entropy',
            })

    # =========================================================================
    # Save results
    # =========================================================================
    model_short = args.model.split('/')[-1]

    csv_file = output_dir / f"differential_features_{model_short}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'layer', 'rank', 'feature_idx', 'direction',
            'correct_mean_act', 'incorrect_mean_act',
            'correct_freq', 'incorrect_freq', 'freq_diff',
            'mean_diff', 'effect_size', 'p_value',
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved differential features CSV: {csv_file}")

    entropy_csv_file = output_dir / f"entropy_features_{model_short}.csv"
    with open(entropy_csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'layer', 'rank', 'feature_idx', 'direction',
            'low_entropy_mean', 'high_entropy_mean',
            'low_entropy_freq', 'high_entropy_freq',
            'effect_size', 'p_value',
        ])
        writer.writeheader()
        writer.writerows(entropy_csv_rows)
    print(f"Saved entropy features CSV: {entropy_csv_file}")

    pkl_file = output_dir / f"sae_uncertainty_{model_short}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': {
                'model': args.model,
                'eval_set': args.eval_set,
                'sae_layers': sae_layers,
                'n_questions': len(samples),
                'timestamp': datetime.now().isoformat(),
            },
            'summary': {
                'n_correct': n_correct,
                'n_incorrect': n_incorrect,
                'accuracy': n_correct / n_total,
                'mean_entropy_correct': float(np.mean(correct_entropies)),
                'mean_entropy_incorrect': float(np.mean(incorrect_entropies)),
                'entropy_q25': float(entropy_q25),
                'entropy_q75': float(entropy_q75),
            },
            'per_question_data': per_question_data,
            'differential_features': differential_features,
        }, f)
    print(f"Saved pickle: {pkl_file}")
    print("Done!")


if __name__ == '__main__':
    main()
