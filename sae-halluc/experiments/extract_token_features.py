"""Token-level SAE feature extraction on RAGTruth QA (week-2 deliverable).

Feeds Llama the original RAGTruth QA prompt + a (hallucinated) response, samples hallucinated
tokens and matched faithful tokens from the same response, and records for each sampled token

    * SAE features at the position that PREDICTS the token (generation-time state) and at the
      token itself (detection ceiling), for several layers in the same forward pass
    * next-token entropy / surprisal over the full vocabulary

Het's loaders are imported read-only from his repo (--het-repo); nothing of his is modified.

Usage (on the GPU machine):
    python experiments/extract_token_features.py \
        --het-repo ~/thesis/sae-uncertainty \
        --split discovery --max-responses 200 --layers 8,16,24 \
        --output-dir results/token_audit_discovery200
"""

import argparse
import gc
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from halluc.data import load_qa_items                      # noqa: E402
from halluc.splits import split_of, DISCOVERY, VALIDATION, PILOT  # noqa: E402


def parse_layers(spec):
    return sorted({int(x) for x in spec.split(",") if x.strip()})


def main():
    ap = argparse.ArgumentParser(description="Token-level SAE feature extraction (RAGTruth QA)")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--het-repo", default=str(ROOT.parent / "sae-uncertainty"),
                    help="path to Het's sae-uncertainty checkout (for load_model / load_sae)")
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--split", choices=[DISCOVERY, VALIDATION, PILOT], default=DISCOVERY)
    ap.add_argument("--max-responses", type=int, default=None)
    ap.add_argument("--layers", default="8,16,24")
    ap.add_argument("--max-per-span", type=int, default=8)
    ap.add_argument("--pos-bins", type=int, default=5)
    ap.add_argument("--margin", type=int, default=2,
                    help="faithful tokens within N tokens of a labeled token are excluded from the pool")
    ap.add_argument("--min-run", type=int, default=3, help="run length that counts as a copy")
    ap.add_argument("--condition", choices=["rag", "no_rag"], default="rag",
                    help="rag = original prompt with passages; no_rag = passages removed, same response "
                         "tokens and same sampled pairs (run both with the same --seed/--max-responses)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, os.path.expanduser(args.het_repo))
    import torch                                            # noqa: E402
    from src.model import load_model, get_layers            # noqa: E402  (Het's code, read-only)
    from src.sae import load_sae                            # noqa: E402
    from halluc.extract import (                            # noqa: E402
        extract_phase1, encode_cache, build_pickles, select_items,
    )

    layers = parse_layers(args.layers)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_items = load_qa_items(args.dataset_dir)
    items = select_items(all_items, split_of, args.split, args.max_responses, args.seed)
    print("Split %s: %d hallucinated responses selected (of %d good QA responses)"
          % (args.split, len(items), len(all_items)))

    print("\nLoading model %s" % args.model)
    model, tokenizer, model_config = load_model(args.model, device=args.device)
    layer_modules = get_layers(model, model_config)

    print("\nPhase 1: forward passes, layers %s" % layers)
    records, cache, population, stats = extract_phase1(
        model, tokenizer, layer_modules, items, layers, args.device, seed=args.seed,
        min_run=args.min_run, margin=args.margin, max_per_span=args.max_per_span,
        pos_bins=args.pos_bins, condition=args.condition)
    print("Phase 1 done: %s" % stats)

    print("\nFreeing the LM before loading SAEs")
    model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    def free():
        torch.cuda.empty_cache()
        gc.collect()

    feats = encode_cache(cache, lambda l: load_sae(model_config, l, device=args.device),
                         layers, args.device, free=free)

    config = {
        "model": args.model, "split": args.split, "seed": args.seed, "layers": layers,
        "condition": args.condition,
        "max_per_span": args.max_per_span, "pos_bins": args.pos_bins, "margin": args.margin,
        "min_run": args.min_run, "max_responses": args.max_responses,
        "definition": {
            "unit": "response token", "correct": "token is faithful (not in a strict hallucinated span)",
            "entropy": "next-token entropy over the full vocabulary at the predicting position",
            "predict": "residual at position i-1 (predicts token i)",
            "self": "residual at position i",
        },
        **stats,
    }
    payloads = build_pickles(records, feats, layers, config)
    for kind, payload in payloads.items():
        path = out_dir / ("token_features_%s.pkl" % kind)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print("Saved %s" % path)
    with open(out_dir / "population.pkl", "wb") as f:
        pickle.dump({"config": config, "population": population}, f)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    print("Saved population stats and config. Done.")


if __name__ == "__main__":
    main()
