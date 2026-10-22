"""
Create a binary-MCQ "bridge" eval set from RAGTruth (QA task only).

Each RAGTruth QA response becomes one item:

    passages + question + response  ->  A/B: "fully supported" vs "unsupported/contradicted"

so Het's loop (extract -> quadrant -> screen -> suppress) runs on RAGTruth with
only the answer-letter generalization (`src.utils.get_eval_letters`).

IMPORTANT (interpretation): here the model is a *judge* of someone else's response.
"correct" = the model's verdict matches the human label; "uncertain" = the model is
unsure of its verdict. Features found this way relate to the model's *detection* of
hallucination, NOT to the model hallucinating while it generates. Treat this stage as
pipeline validation / a Het-comparable result.

Label handling
    hallucinated : >= 1 labeled span that is NOT `implicit_true`      (label-mode strict)
    clean        : no labeled spans at all
    ambiguous    : labeled spans, all `implicit_true`                  -> dropped (strict)
    `--label-mode any` treats ambiguous responses as hallucinated (sensitivity check).
    Only quality == "good" responses are used (drops refusals / truncations).

Confound control (defaults)
    --balance model   per-source-model class balance (removes the "which LLM wrote it"
                      shortcut: GPT-4/3.5 are mostly clean, Llama-2/Mistral mostly not)
    length matching   the larger class is subsampled to match the smaller class's
                      response word counts within each model (hallucinated responses
                      are longer: median ~146 vs ~102 words)
    option order      A/B text is randomized per item, so letter bias is separable
                      from content
    split             discovery/validation split BY QUESTION (source_id), never by response

Usage:
    python scripts/create_ragtruth_eval_set.py
    python scripts/create_ragtruth_eval_set.py --label-mode any --output-dir data/eval_sets
"""

import argparse
import json
import random
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = THESIS_ROOT / "ragtruth" / "dataset"

OPT_SUPPORTED = "Yes, fully supported"
OPT_UNSUPPORTED = "No, it contains unsupported or contradicted information"

PROMPT_TEMPLATE = (
    "Read the passages and the response. Decide whether the response is fully "
    "supported by the passages.\n\n"
    "Passages:\n{passages}\n\n"
    "Question: {question}\n\n"
    "Response: {response}\n\n"
    "Is the response fully supported by the passages?\n"
    "A. {opt_a}\n"
    "B. {opt_b}\n"
    "Answer:"
)


# ----------------------------------------------------------------------------
# Loading / labeling
# ----------------------------------------------------------------------------
def load_qa_items(dataset_dir: Path):
    """Return list of response dicts (quality=good, QA task) with source fields attached."""
    sources = {}
    with open(dataset_dir / "source_info.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["task_type"] == "QA":
                sources[row["source_id"]] = row

    items = []
    with open(dataset_dir / "response.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["source_id"] not in sources or r.get("quality") != "good":
                continue
            info = sources[r["source_id"]]["source_info"]
            strict_spans = [lb for lb in r["labels"] if not lb.get("implicit_true")]
            items.append({
                "response_id": r["id"],
                "source_id": r["source_id"],
                "source_model": r["model"],
                "ragtruth_split": r["split"],
                "question": info["question"],
                "passages": info["passages"],
                "response": r["response"],
                "n_spans": len(r["labels"]),
                "n_strict_spans": len(strict_spans),
                "label_types": sorted({lb["label_type"] for lb in r["labels"]}),
                "response_words": len(r["response"].split()),
                "passage_words": len(info["passages"].split()),
            })
    return items


def assign_label(item, label_mode: str):
    """Return 'hallucinated' | 'clean' | 'ambiguous'."""
    if item["n_spans"] == 0:
        return "clean"
    if item["n_strict_spans"] > 0:
        return "hallucinated"
    return "hallucinated" if label_mode == "any" else "ambiguous"


# ----------------------------------------------------------------------------
# Balancing
# ----------------------------------------------------------------------------
def pick_length_matched(pool, target_lengths, rng):
    """For each target length pick the unused pool item with the closest word count."""
    pool = list(pool)
    rng.shuffle(pool)
    chosen = []
    targets = list(target_lengths)
    rng.shuffle(targets)
    for t in targets:
        if not pool:
            break
        best = min(range(len(pool)), key=lambda i: abs(pool[i]["response_words"] - t))
        chosen.append(pool.pop(best))
    return chosen


def balance_group(hall, clean, rng, length_match):
    """Class-balance one group (a source model, or everything). Returns (hall, clean)."""
    n = min(len(hall), len(clean))
    if n == 0:
        return [], []
    if len(hall) <= len(clean):
        keep_h = list(hall)
        if length_match:
            keep_c = pick_length_matched(clean, [x["response_words"] for x in keep_h], rng)
        else:
            keep_c = rng.sample(clean, n)
    else:
        keep_c = list(clean)
        if length_match:
            keep_h = pick_length_matched(hall, [x["response_words"] for x in keep_c], rng)
        else:
            keep_h = rng.sample(hall, n)
    return keep_h, keep_c


def balance(items, mode, rng, length_match):
    hall = [x for x in items if x["label"] == "hallucinated"]
    clean = [x for x in items if x["label"] == "clean"]
    if mode == "none":
        return hall + clean
    if mode == "class":
        h, c = balance_group(hall, clean, rng, length_match)
        return h + c
    # mode == "model": balance classes within each source model
    out = []
    for m in sorted({x["source_model"] for x in items}):
        h, c = balance_group([x for x in hall if x["source_model"] == m],
                             [x for x in clean if x["source_model"] == m],
                             rng, length_match)
        out.extend(h + c)
    return out


# ----------------------------------------------------------------------------
# Sample construction
# ----------------------------------------------------------------------------
def build_sample(item, idx, rng):
    hallucinated = item["label"] == "hallucinated"
    correct_text = OPT_UNSUPPORTED if hallucinated else OPT_SUPPORTED
    wrong_text = OPT_SUPPORTED if hallucinated else OPT_UNSUPPORTED

    # Randomize which letter carries which text
    supported_first = rng.random() < 0.5
    opt_a, opt_b = (OPT_SUPPORTED, OPT_UNSUPPORTED) if supported_first else (OPT_UNSUPPORTED, OPT_SUPPORTED)
    options = {"A": opt_a, "B": opt_b}
    correct_letter = "A" if options["A"] == correct_text else "B"

    prompt = PROMPT_TEMPLATE.format(
        passages=item["passages"].strip(), question=item["question"].strip(),
        response=item["response"].strip(), opt_a=opt_a, opt_b=opt_b,
    )
    return {
        "id": f"ragtruth_qa_{item['response_id']}",
        "question": item["question"],
        "mcq_prompt": prompt,
        "options": options,
        "correct_answer": correct_text,
        "correct_letter": correct_letter,
        "distractors": [wrong_text],
        # --- metadata for confound checks / later analysis ---
        "source_id": item["source_id"],
        "source_model": item["source_model"],
        "hallucinated": hallucinated,
        "label_types": item["label_types"],
        "n_spans": item["n_spans"],
        "n_strict_spans": item["n_strict_spans"],
        "response_words": item["response_words"],
        "passage_words": item["passage_words"],
        "ragtruth_split": item["ragtruth_split"],
    }


def split_by_question(samples, discovery_frac, seed):
    ids = sorted({s["source_id"] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_disc = int(round(len(ids) * discovery_frac))
    disc_ids = set(ids[:n_disc])
    disc = [s for s in samples if s["source_id"] in disc_ids]
    val = [s for s in samples if s["source_id"] not in disc_ids]
    return disc, val


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def auc(pos, neg):
    """P(pos value > neg value), ties count half. 0.5 = no signal."""
    if not pos or not neg:
        return float("nan")
    ns = sorted(neg)
    total = 0.0
    for p in pos:
        lo, hi = bisect_left(ns, p), bisect_right(ns, p)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(neg))


def summarize(samples, name):
    n = len(samples)
    hall = [s for s in samples if s["hallucinated"]]
    clean = [s for s in samples if not s["hallucinated"]]
    by_model = defaultdict(lambda: [0, 0])  # [clean, hallucinated]
    for s in samples:
        by_model[s["source_model"]][1 if s["hallucinated"] else 0] += 1
    rep = {
        "n": n,
        "n_questions": len({s["source_id"] for s in samples}),
        "n_hallucinated": len(hall),
        "n_clean": len(clean),
        "correct_letter_counts": dict(Counter(s["correct_letter"] for s in samples)),
        "response_words_median_clean": statistics.median([s["response_words"] for s in clean]) if clean else None,
        "response_words_median_halluc": statistics.median([s["response_words"] for s in hall]) if hall else None,
        "passage_words_median_clean": statistics.median([s["passage_words"] for s in clean]) if clean else None,
        "passage_words_median_halluc": statistics.median([s["passage_words"] for s in hall]) if hall else None,
        # shortcut strength: 0.5 = none. Length AUC > 0.5 => longer responses look hallucinated.
        "length_auc_halluc_vs_clean": auc([s["response_words"] for s in hall],
                                          [s["response_words"] for s in clean]),
        # accuracy of "predict the majority class within each source model"
        "model_majority_baseline_acc": (
            sum(max(c, h) for c, h in by_model.values()) / n if n else None),
        "by_source_model": {m: {"clean": c, "hallucinated": h} for m, (c, h) in sorted(by_model.items())},
        "label_types": dict(Counter(t for s in hall for t in s["label_types"])),
    }
    print(f"\n--- {name} ---")
    print(f"  items: {n}  questions: {rep['n_questions']}  "
          f"hallucinated: {rep['n_hallucinated']}  clean: {rep['n_clean']}")
    print(f"  correct-letter counts: {rep['correct_letter_counts']}")
    print(f"  response words (median) clean/halluc: "
          f"{rep['response_words_median_clean']} / {rep['response_words_median_halluc']}  "
          f"length-only AUC: {rep['length_auc_halluc_vs_clean']:.3f}  (0.5 = no shortcut)")
    print(f"  passage words (median) clean/halluc: "
          f"{rep['passage_words_median_clean']} / {rep['passage_words_median_halluc']}")
    print(f"  per-source-model majority baseline acc: {rep['model_majority_baseline_acc']:.3f}  (0.5 = no shortcut)")
    for m, d in rep["by_source_model"].items():
        print(f"    {m:22s} clean {d['clean']:4d}  halluc {d['hallucinated']:4d}")
    return rep


def write_set(path, samples, metadata):
    with open(path, "w") as f:
        json.dump({"metadata": {**metadata, "num_samples": len(samples)}, "samples": samples}, f, indent=2)
    print(f"Saved {len(samples)} samples -> {path}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Create binary-MCQ bridge eval set from RAGTruth QA")
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--output-dir", type=Path, default=Path("data/eval_sets"))
    ap.add_argument("--label-mode", choices=["strict", "any"], default="strict",
                    help="strict: drop responses whose only spans are implicit_true; "
                         "any: count them as hallucinated (sensitivity check)")
    ap.add_argument("--balance", choices=["model", "class", "none"], default="model")
    ap.add_argument("--no-length-match", action="store_true",
                    help="random subsampling instead of response-length matching")
    ap.add_argument("--discovery-frac", type=float, default=0.6)
    ap.add_argument("--smoke-n", type=int, default=300,
                    help="also write a smoke-test subset of N items from the discovery split (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = load_qa_items(args.dataset_dir)
    for it in items:
        it["label"] = assign_label(it, args.label_mode)
    counts = Counter(it["label"] for it in items)
    print(f"Loaded {len(items)} good QA responses: {dict(counts)}  (label-mode={args.label_mode})")
    if args.label_mode == "strict":
        print(f"  dropped {counts['ambiguous']} ambiguous (implicit_true-only) responses")

    items = [it for it in items if it["label"] != "ambiguous"]
    kept = balance(items, args.balance, rng, length_match=not args.no_length_match)
    print(f"After balance={args.balance} (length-match={not args.no_length_match}): {len(kept)} items")

    kept.sort(key=lambda x: int(x["response_id"]))
    rng.shuffle(kept)
    samples = [build_sample(it, i, rng) for i, it in enumerate(kept)]

    disc, val = split_by_question(samples, args.discovery_frac, args.seed)
    overlap = {s["source_id"] for s in disc} & {s["source_id"] for s in val}
    assert not overlap, f"question leak across split: {sorted(overlap)[:5]}"

    report = {
        "all": summarize(samples, "ALL"),
        "discovery": summarize(disc, "DISCOVERY"),
        "validation": summarize(val, "VALIDATION"),
    }

    metadata = {
        "dataset": "ragtruth_qa",
        "format": "mcq",
        "task": "response_support_detection (model acts as judge of a given response)",
        "interpretation": "pipeline validation; features reflect detection, not generation-time hallucination",
        "letters": ["A", "B"],
        "num_options": 2,
        "label_mode": args.label_mode,
        "balance": args.balance,
        "length_match": not args.no_length_match,
        "discovery_frac": args.discovery_frac,
        "seed": args.seed,
    }
    stem = f"eval_set_mcq_ragtruth_{args.label_mode}_{len(samples)}"
    write_set(args.output_dir / f"{stem}.json", samples, metadata)
    write_set(args.output_dir / f"{stem}_discovery.json", disc, {**metadata, "split_half": "discovery"})
    write_set(args.output_dir / f"{stem}_validation.json", val, {**metadata, "split_half": "validation"})
    if args.smoke_n > 0:
        # Smoke subset comes from DISCOVERY only, so validation stays untouched.
        smoke = disc[:args.smoke_n]
        report["smoke"] = summarize(smoke, f"SMOKE ({len(smoke)})")
        write_set(args.output_dir / f"{stem}_smoke{len(smoke)}.json", smoke,
                  {**metadata, "split_half": "discovery_smoke_subset"})
    with open(args.output_dir / f"{stem}_report.json", "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "report": report}, f, indent=2)

    print("\nExample prompt:\n" + samples[0]["mcq_prompt"][-700:])
    print(f"Correct: {samples[0]['correct_letter']} ({samples[0]['correct_answer']})")


if __name__ == "__main__":
    main()
