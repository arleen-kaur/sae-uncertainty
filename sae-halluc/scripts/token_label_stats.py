"""Token-level label statistics for RAGTruth QA (week-1 deliverable).

Answers, before any GPU work:
  * Does the char->token alignment hold on every span? (integrity check)
  * How many hallucinated / faithful / masked tokens are there?
  * How do the two classes differ on the confounds we must control:
        copy status (vs passages and vs the whole prompt), token type, relative position?
  * How many hallucinated tokens can actually be MATCHED to faithful tokens on
    (token type, copy status, position bin)?  -> feasibility of the matched design.

Usage:
    python scripts/token_label_stats.py --tokenizer meta-llama/Llama-3.1-8B
    python scripts/token_label_stats.py --fake            # dry run, no transformers needed
    python scripts/token_label_stats.py --limit 300 --split train
"""

import argparse
import collections
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from halluc.alignment import HALL, FAITHFUL  # noqa: E402
from halluc.copy_features import STATUS_NAMES  # noqa: E402
from halluc.data import load_qa_items  # noqa: E402
from halluc.pipeline import process_item  # noqa: E402


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def quantiles(xs):
    if not xs:
        return None
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1)))]
    return {"p25": q(0.25), "median": q(0.5), "p75": q(0.75)}


def main():
    ap = argparse.ArgumentParser(description="Token-label statistics for RAGTruth QA")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--fake", action="store_true", help="use the fake word tokenizer (no transformers)")
    ap.add_argument("--split", choices=["train", "test"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="max responses (after category filter)")
    ap.add_argument("--min-run", type=int, default=3, help="run length that counts as a copy")
    ap.add_argument("--margin", type=int, default=0, help="mask faithful tokens within N tokens of a labeled token")
    ap.add_argument("--pos-bins", type=int, default=5)
    ap.add_argument("--output", default=os.path.join(ROOT, "outputs", "token_label_stats.json"))
    args = ap.parse_args()

    if args.fake:
        from fake_tokenizer import FakeTokenizer
        tok = FakeTokenizer("word_trimmed")
        tok_name = "fake_word_trimmed"
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tok_name = args.tokenizer

    # All good-quality responses are used: clean ones supply faithful tokens, hallucinated ones
    # supply both classes, and ambiguous ones still have usable tokens outside their soft spans.
    items = load_qa_items(split=args.split)
    if args.limit:
        items = items[:args.limit]
    print("Processing %d responses with tokenizer %s ..." % (len(items), tok_name))

    label_counts = collections.Counter()
    by_class = {c: {"ctx": collections.Counter(), "passage": collections.Counter(),
                    "type": collections.Counter(), "rel_pos": []} for c in (HALL, FAITHFUL)}
    cells = {c: collections.Counter() for c in (HALL, FAITHFUL)}
    hall_per_span = collections.Counter()
    span_failures = 0
    n_spans = 0
    spans_without_hall = 0
    resp_tokens = []
    bad_predict = 0

    for it in items:
        o = process_item(tok, it, min_run=args.min_run, margin_tokens=args.margin)
        span_failures += len(o["span_failures"])
        n_spans += o["n_strict_spans"]
        resp_tokens.append(o["n_resp_tokens"])
        seen = collections.Counter()
        for t in o["tokens"]:
            label_counts[t["label"]] += 1
            if t["predict_idx"] < 0:
                bad_predict += 1
            if t["label"] in (HALL, FAITHFUL):
                d = by_class[t["label"]]
                d["ctx"][t["ctx_status"]] += 1
                d["passage"][t["passage_status"]] += 1
                d["type"][t["type"]] += 1
                d["rel_pos"].append(t["rel_pos"])
                pb = min(args.pos_bins - 1, int(t["rel_pos"] * args.pos_bins))
                cells[t["label"]][(t["type"], t["ctx_status"], pb)] += 1
            if t["label"] == HALL and t["span_idx"] >= 0:
                seen[t["span_idx"]] += 1
        for k in range(o["n_strict_spans"]):
            if seen[k] == 0:
                spans_without_hall += 1
            else:
                hall_per_span[seen[k]] += 1

    total_resp_tokens = sum(resp_tokens)
    n_h, n_f = label_counts[HALL], label_counts[FAITHFUL]

    print("\n=== Alignment integrity ===")
    print("strict spans: %d   coverage failures: %d   spans with no hall token: %d   bad predict_idx: %d"
          % (n_spans, span_failures, spans_without_hall, bad_predict))

    print("\n=== Token labels (response tokens: %d across %d responses) ===" % (total_resp_tokens, len(items)))
    for lab, n in label_counts.most_common():
        print("  %-9s %9d  (%5.1f%%)" % (lab, n, pct(n, total_resp_tokens)))
    spans_tok = [k for k, v in hall_per_span.items() for _ in range(v)]
    if spans_tok:
        print("  hall tokens per span: %s" % quantiles(spans_tok))

    report = {
        "tokenizer": tok_name, "n_responses": len(items), "n_resp_tokens": total_resp_tokens,
        "label_counts": dict(label_counts), "n_strict_spans": n_spans,
        "span_coverage_failures": span_failures, "spans_without_hall_token": spans_without_hall,
        "min_run": args.min_run, "margin": args.margin, "classes": {},
    }

    print("\n=== Confounds: hallucinated vs faithful tokens ===")
    for kind, title in (("ctx", "copy status vs full prompt"), ("passage", "copy status vs passages only")):
        print("  %s:" % title)
        for lab in (HALL, FAITHFUL):
            n = sum(by_class[lab][kind].values())
            row = ", ".join("%s=%.1f%%" % (STATUS_NAMES[s], pct(by_class[lab][kind][s], n)) for s in (0, 1, 2))
            print("    %-9s %s" % (lab, row))
    print("  token type:")
    types = sorted({t for c in by_class.values() for t in c["type"]})
    print("    %-11s %s" % ("", "  ".join("%9s" % t for t in types)))
    for lab in (HALL, FAITHFUL):
        n = sum(by_class[lab]["type"].values())
        print("    %-11s %s" % (lab, "  ".join("%8.1f%%" % pct(by_class[lab]["type"][t], n) for t in types)))
    for lab in (HALL, FAITHFUL):
        print("  relative position, %-9s %s" % (lab, quantiles(by_class[lab]["rel_pos"])))

    # Matching feasibility: pair up tokens within (type, ctx copy status, position bin)
    matchable = sum(min(cells[HALL][c], cells[FAITHFUL][c]) for c in cells[HALL])
    matchable_novel = sum(min(cells[HALL][c], cells[FAITHFUL][c]) for c in cells[HALL] if c[1] == 0)
    hall_novel = sum(v for c, v in cells[HALL].items() if c[1] == 0)
    print("\n=== Matching feasibility (cells = token type x copy status x position bin) ===")
    print("  hallucinated tokens: %d   faithful tokens: %d" % (n_h, n_f))
    print("  matchable pairs (any copy status):   %d  (%.1f%% of hallucinated tokens)" % (matchable, pct(matchable, n_h)))
    print("  matchable pairs (non-copied only):   %d  (%.1f%% of hallucinated non-copied tokens, n=%d)"
          % (matchable_novel, pct(matchable_novel, hall_novel), hall_novel))

    for lab in (HALL, FAITHFUL):
        report["classes"][lab] = {
            "n": label_counts[lab],
            "ctx_status": {STATUS_NAMES[k]: v for k, v in by_class[lab]["ctx"].items()},
            "passage_status": {STATUS_NAMES[k]: v for k, v in by_class[lab]["passage"].items()},
            "type": dict(by_class[lab]["type"]),
            "rel_pos": quantiles(by_class[lab]["rel_pos"]),
        }
    report["matchable_pairs"] = matchable
    report["matchable_pairs_non_copied"] = matchable_novel
    report["hall_non_copied_tokens"] = hall_novel

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved %s" % args.output)


if __name__ == "__main__":
    main()
