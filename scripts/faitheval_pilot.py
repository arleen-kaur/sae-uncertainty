"""
Pilot on FaithEval-Counterfactual: does the base model follow the context or its memory?

Runs Het's letter-logit eval (no SAE) on the paired sets from scripts/create_faitheval_eval_set.py:

    noctx : closed-book ARC question           -> accuracy vs the real answer
    ctx   : counterfactual context + question  -> accuracy vs the CONTEXT answer ("faithful"),
                                                 plus how often it picks the real answer (memory)

Gate for continuing with Het's pipeline on this benchmark (see printed verdict):
  * closed-book accuracy well above chance (the model "knows" the answers),
  * context-faithful accuracy has headroom both ways (neither ~0% nor ~100%), so that the
    correct/incorrect quadrants are populated,
  * confident/uncertain entropy buckets (p25) each have enough items for Mann-Whitney power.

Usage (GPU machine):
    python scripts/faitheval_pilot.py --model meta-llama/Llama-3.1-8B \
        --ctx data/eval_sets/eval_set_mcq_faitheval_cf_ctx_995.json \
        --noctx data/eval_sets/eval_set_mcq_faitheval_cf_noctx_995.json \
        --output results/faitheval_pilot.json
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LETTERS = ["A", "B", "C", "D"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--noctx", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--entropy-percentile", type=float, default=25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from src.model import load_model, get_letter_token_ids
    from src.evaluation import run_eval

    ctx = json.load(open(args.ctx))["samples"][:args.limit]
    noc = json.load(open(args.noctx))["samples"][:args.limit]
    assert [s["id"] for s in ctx] == [s["id"] for s in noc], "sets must be item-aligned"

    model, tok, _ = load_model(args.model, device=args.device)
    ltid = get_letter_token_ids(tok, LETTERS)
    r_noc = run_eval(model, tok, noc, ltid, args.device, "closed-book")
    r_ctx = run_eval(model, tok, ctx, ltid, args.device, "with context")
    n = len(ctx)

    acc_noc = sum(r["correct"] for r in r_noc)
    faithful = sum(r["correct"] for r in r_ctx)
    memory = sum(r["predicted"] == s["parametric_letter"] for r, s in zip(r_ctx, ctx))
    other = n - faithful - memory
    # paired transitions: knows the answer closed-book, but does the context override it?
    knew = [i for i in range(n) if r_noc[i]["correct"]]
    knew_faithful = sum(r_ctx[i]["correct"] for i in knew)

    ent = np.array([r["entropy"] for r in r_ctx])
    lo, hi = np.percentile(ent, [args.entropy_percentile, 100 - args.entropy_percentile])
    cor = np.array([bool(r["correct"]) for r in r_ctx])
    conf, unc = ent <= lo, ent >= hi
    quad = {"confident_correct": int((conf & cor).sum()), "confident_incorrect": int((conf & ~cor).sum()),
            "uncertain_correct": int((unc & cor).sum()), "uncertain_incorrect": int((unc & ~cor).sum())}

    out = {
        "model": args.model, "n": n,
        "closed_book_acc": [acc_noc / n, wilson(acc_noc, n)],
        "ctx_faithful_acc": [faithful / n, wilson(faithful, n)],
        "ctx_follows_memory": [memory / n, wilson(memory, n)],
        "ctx_other": other / n,
        "knew_closed_book_n": len(knew),
        "faithful_given_knew": (knew_faithful / len(knew)) if knew else None,
        "pred_letter_dist_ctx": dict(Counter(r["predicted"] for r in r_ctx)),
        "pred_letter_dist_noctx": dict(Counter(r["predicted"] for r in r_noc)),
        "mean_entropy": {"noctx": float(np.mean([r["entropy"] for r in r_noc])), "ctx": float(ent.mean())},
        "quadrants_ctx_p%d" % args.entropy_percentile: quad,
    }
    verdict = []
    verdict.append("closed-book >> chance (25%%): %s" % ("OK" if acc_noc / n > 0.40 else "FAIL"))
    verdict.append("ctx faithful accuracy has headroom (5%%-95%%): %s"
                   % ("OK" if 0.05 < faithful / n < 0.95 else "FAIL"))
    smallest = min(quad.values())
    verdict.append("smallest quadrant >= 100 items: %s (%d)"
                   % ("OK" if smallest >= 100 else ("WARN" if smallest >= 30 else "FAIL"), smallest))
    out["gate"] = verdict

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": out,
               "items": [{"id": s["id"], "noctx": a, "ctx": b} for s, a, b in zip(ctx, r_noc, r_ctx)]},
              open(args.output, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
