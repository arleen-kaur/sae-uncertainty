"""
Build paired MCQ eval sets from FaithEval-Counterfactual (Ming et al., ICLR 2025; arXiv 2410.03727).

FaithEval-Counterfactual = ARC-Challenge questions + a long synthetic context that supports a
WRONG (counterfactual) option. Each item therefore has two objective answers:

    context_letter     the option the context supports   (the faithful answer, FaithEval's key)
    parametric_letter  the real ARC answer                (what the model "knows")

Two eval sets are written over the SAME questions and the SAME option order:

    ctx    : context + question + options. correct_letter = context_letter  ("faithful")
             incorrect = the model answers from memory (or anything else) instead of the context
    noctx  : question + options only (Het's usual ARC format). correct_letter = parametric_letter

so the RAG vs no-RAG comparison is item-for-item. Both are 60/40 discovery/validation split by
question id (same split for both), in Het's file format (`mcq_prompt`, `options`, `correct_letter`).

Only 4-option items are kept (Het's setup); items labelled 1-4 are relabelled A-D.

Input: raw FaithEval rows (data/faitheval/faitheval_counterfactual_raw.json, downloaded from
https://huggingface.co/datasets/Salesforce/FaithEval-counterfactual-v1.0) and Het's ARC eval set.

Usage:
    python scripts/create_faitheval_eval_set.py
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LETTERS = ["A", "B", "C", "D"]

INSTRUCTION = ("You are an expert in retrieval-based question answering. Please respond with the "
               "exact answer, using only the information provided in the context.")


def options_block(options):
    return "\n".join("%s. %s" % (l, o) for l, o in zip(LETTERS, options))


def prompt_noctx(question, options):
    return "Question: %s\n%s\nAnswer:" % (question, options_block(options))


def prompt_ctx(context, question, options, instruction=True):
    head = (INSTRUCTION + "\n\n") if instruction else ""
    return "%sContext: %s\n\nQuestion: %s\n%s\nAnswer:" % (
        head, context.strip(), question, options_block(options))


def build(rows, arc_by_id, instruction=True):
    ctx, noctx, dropped = [], [], Counter()
    for r in rows:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if len(texts) != 4:
            dropped["not_4_options"] += 1
            continue
        a = arc_by_id.get(r["id"])
        if a is None:
            dropped["not_in_arc_set"] += 1
            continue
        if r["question"].strip() != a["question"].strip():
            dropped["question_mismatch"] += 1
            continue
        key = str(r["answerKey"]).strip()
        if key in labels:
            ctx_idx = labels.index(key)
        elif key in LETTERS:                  # numeric option labels (1-4) but letter answer key
            ctx_idx = LETTERS.index(key)
        else:
            dropped["bad_answer_key"] += 1
            continue
        if texts[ctx_idx].strip() != r["answer"].strip():
            dropped["answer_text_mismatch"] += 1
            continue
        if a["correct_answer"] not in texts:
            dropped["arc_answer_not_in_options"] += 1
            continue
        par_idx = texts.index(a["correct_answer"])
        if par_idx == ctx_idx:
            dropped["context_answer_equals_true_answer"] += 1
            continue

        common = {
            "id": r["id"], "question": r["question"], "options": list(texts),
            "context_letter": LETTERS[ctx_idx], "parametric_letter": LETTERS[par_idx],
            "context_answer": texts[ctx_idx], "parametric_answer": texts[par_idx],
        }
        ctx.append({**common,
                    "mcq_prompt": prompt_ctx(r["context"], r["question"], texts, instruction),
                    "context": r["context"],
                    "correct_letter": LETTERS[ctx_idx], "correct_answer": texts[ctx_idx],
                    "distractors": [t for i, t in enumerate(texts) if i != ctx_idx]})
        noctx.append({**common,
                      "mcq_prompt": prompt_noctx(r["question"], texts),
                      "correct_letter": LETTERS[par_idx], "correct_answer": texts[par_idx],
                      "distractors": [t for i, t in enumerate(texts) if i != par_idx]})
    return ctx, noctx, dropped


def split_ids(ids, frac, seed):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    k = int(round(len(ids) * frac))
    return set(ids[:k])


def write(samples, name, out_dir, meta):
    path = out_dir / name
    with open(path, "w") as f:
        json.dump({"metadata": {**meta, "num_samples": len(samples)}, "samples": samples}, f, indent=2)
    print("wrote %s (%d)" % (path, len(samples)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(REPO / "data/faitheval/faitheval_counterfactual_raw.json"))
    ap.add_argument("--arc", default=str(REPO / "data/eval_sets/eval_set_mcq_arc_challenge_2577.json"))
    ap.add_argument("--output-dir", default=str(REPO / "data/eval_sets"))
    ap.add_argument("--discovery-frac", type=float, default=0.6)
    ap.add_argument("--no-instruction", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = json.load(open(args.raw))
    arc_by_id = {s["id"]: s for s in json.load(open(args.arc))["samples"]}
    ctx, noctx, dropped = build(rows, arc_by_id, instruction=not args.no_instruction)
    print("raw rows %d -> kept %d; dropped %s" % (len(rows), len(ctx), dict(dropped)))

    disc = split_ids([s["id"] for s in ctx], args.discovery_frac, args.seed)
    out = Path(args.output_dir)
    n = len(ctx)
    for tag, samples in (("ctx", ctx), ("noctx", noctx)):
        meta = {"dataset": "faitheval_counterfactual", "condition": tag, "format": "mcq",
                "num_options": 4, "seed": args.seed,
                "source": "Salesforce/FaithEval-counterfactual-v1.0 joined with ARC-Challenge by id",
                "label": "context_letter" if tag == "ctx" else "parametric_letter",
                "instruction": not args.no_instruction}
        base = "eval_set_mcq_faitheval_cf_%s_%d" % (tag, n)
        write(samples, base + ".json", out, {**meta, "split": "all"})
        write([s for s in samples if s["id"] in disc], base + "_discovery.json", out, {**meta, "split": "discovery"})
        write([s for s in samples if s["id"] not in disc], base + "_validation.json", out, {**meta, "split": "validation"})

    print("answer-letter balance (ctx key):", dict(Counter(s["correct_letter"] for s in ctx)))
    print("answer-letter balance (parametric key):", dict(Counter(s["parametric_letter"] for s in ctx)))


if __name__ == "__main__":
    main()
