"""On-policy pilot, step 2: label generated responses with free hallucination judges and report agreement.

Judges (add more later by writing another `run_<name>` function and registering it in JUDGES):
  lettuce : LettuceDetect (span-level, encoder).  response hallucinated  <=> at least one span >= min_confidence
  hhem    : Vectara HHEM-2.1-Open (response-level consistency score in [0,1]).  hallucinated <=> score < threshold

Writes a JSONL with the same schema as halluc.data.load_qa_items: `strict_spans` is filled with LettuceDetect's
spans, so the existing token-level pipeline can read it.  Per-judge raw outputs and the consensus label are
stored under `judge`.  Prints per-judge hallucination rates, pairwise agreement (kappa) and the consensus.

Usage (GPU machine, inside venv-sae; run scripts/judge_check.py first):
    python -u scripts/judge_responses.py --input outputs/onpolicy_gen.jsonl --output outputs/onpolicy_judged.jsonl
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

# ModernBERT (LettuceDetect) calls torch.compile, which needs Python.h (missing on this host): run eagerly instead.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from halluc.judging import agreement, consensus, spans_to_tuples  # noqa: E402


def run_lettuce(items, args):
    from lettucedetect.models.inference import HallucinationDetector
    det = HallucinationDetector(method="transformer", model_path=args.lettuce_model)
    spans = []
    for n, it in enumerate(items):
        if not it["response"].strip():
            spans.append(None)
            continue
        preds = det.predict(context=[it["passages"]], question=it["question"], answer=it["response"],
                            output_format="spans")
        # filter here (not via predict(min_confidence=...), which older lettucedetect versions lack)
        spans.append([s for s in preds if s.get("confidence", 1.0) >= args.lettuce_min_confidence])
        if (n + 1) % 50 == 0:
            print("  lettuce %d/%d" % (n + 1, len(items)))
    labels = [None if s is None else len(s) > 0 for s in spans]
    return labels, [{"spans": s} for s in spans]


def run_hhem(items, args):
    import torch
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("vectara/hallucination_evaluation_model",
                                                               trust_remote_code=True)
    model.eval()
    scores = []
    for n, it in enumerate(items):
        if not it["response"].strip():
            scores.append(None)
            continue
        with torch.no_grad():
            s = model.predict([(it["passages"], it["response"])])
        scores.append(float(s[0]))
        if (n + 1) % 50 == 0:
            print("  hhem %d/%d" % (n + 1, len(items)))
    labels = [None if s is None else s < args.hhem_threshold for s in scores]
    return labels, [{"score": s} for s in scores]


def run_lynx(items, args):
    from halluc.judge_models import load_lynx, lynx_generate, parse_lynx
    model, tok = load_lynx()
    idx = [i for i, it in enumerate(items) if it["response"].strip()]
    triples = [(items[i]["question"], items[i]["passages"], items[i]["response"]) for i in idx]
    texts = lynx_generate(model, tok, triples, batch_size=args.lynx_batch_size,
                          progress=lambda a, b: print("  lynx %d/%d" % (a, b)) if a % 40 == 0 or a == b else None)
    labels, raw = [None] * len(items), [{"verdict": None, "reasoning": ""} for _ in items]
    for i, text in zip(idx, texts):
        v = parse_lynx(text)
        labels[i] = None if v is None else v == "FAIL"
        raw[i] = {"verdict": v, "reasoning": text[:1500]}
    print("  lynx: %d answers with no parseable verdict" % sum(1 for i in idx if labels[i] is None))
    return labels, raw


def run_minicheck(items, args):
    from halluc.judge_models import load_minicheck, minicheck_answer
    scorer = load_minicheck(cache_dir=args.minicheck_cache)
    labels, raw = [], []
    for n, it in enumerate(items):
        if not it["response"].strip():
            labels.append(None)
            raw.append({"min_support": None, "sentences": [], "sentence_support": []})
            continue
        lab, r = minicheck_answer(scorer, it["passages"], it["response"], threshold=args.minicheck_threshold)
        labels.append(lab)
        raw.append(r)
        if (n + 1) % 50 == 0:
            print("  minicheck %d/%d" % (n + 1, len(items)))
    return labels, raw


JUDGES = {"lettuce": run_lettuce, "hhem": run_hhem, "lynx": run_lynx, "minicheck": run_minicheck}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--judges", default="lettuce,hhem")
    ap.add_argument("--lettuce-model", default="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    ap.add_argument("--lettuce-min-confidence", type=float, default=0.5)
    ap.add_argument("--hhem-threshold", type=float, default=0.5)
    ap.add_argument("--lynx-batch-size", type=int, default=8)
    ap.add_argument("--minicheck-threshold", type=float, default=0.5)
    ap.add_argument("--minicheck-cache", default=os.path.expanduser("~/.cache/minicheck"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.input)]
    if args.limit:
        items = items[:args.limit]
    print("%d responses to judge" % len(items))

    names = [j for j in args.judges.split(",") if j]
    labels, raw = {}, {}
    for name in names:
        # A judge already stored in the input (an earlier run) is reused, so only missing judges are run.
        if all(name in it.get("judge", {}) for it in items):
            print("reusing stored judge: %s" % name)
            labels[name] = [it["judge"][name]["hallucinated"] for it in items]
            raw[name] = [{k: v for k, v in it["judge"][name].items() if k != "hallucinated"} for it in items]
        else:
            print("running judge: %s" % name)
            labels[name], raw[name] = JUDGES[name](items, args)
        valid = [x for x in labels[name] if x is not None]
        print("  %s: hallucinated %d / %d (%.1f%%)" % (name, sum(valid), len(valid), 100.0 * sum(valid) / max(1, len(valid))))

    cons = consensus(labels) if len(names) > 1 else list(labels[names[0]])
    cons_two = consensus({k: labels[k] for k in ("lettuce", "hhem")}) if "lettuce" in names and "hhem" in names else None
    summary = {"n": len(items), "judges": names, "rates": {}, "pairwise": {}}
    for name in names:
        v = [x for x in labels[name] if x is not None]
        summary["rates"][name] = sum(v) / float(max(1, len(v)))
    for a, b in itertools.combinations(names, 2):
        summary["pairwise"]["%s|%s" % (a, b)] = agreement(labels[a], labels[b])
    decided = [c for c in cons if c is not None]
    summary["consensus"] = {"decided": len(decided), "undecided": len(cons) - len(decided),
                            "hallucinated": sum(decided), "clean": len(decided) - sum(decided)}

    with open(args.output, "w") as f:
        for i, it in enumerate(items):
            it["judge"] = {name: {"hallucinated": labels[name][i], **raw[name][i]} for name in names}
            it["judge"]["consensus_hallucinated"] = cons[i]
            if cons_two is not None:
                it["judge"]["consensus_lettuce_hhem"] = cons_two[i]  # the original 2-judge label, kept for comparison
            if "lettuce" in names and raw["lettuce"][i]["spans"]:
                it["strict_spans"] = [list(t) for t in spans_to_tuples(raw["lettuce"][i]["spans"], len(it["response"]))]
            f.write(json.dumps(it) + "\n")
    json.dump(summary, open(str(Path(args.output).with_suffix(".summary.json")), "w"), indent=1)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=1))
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
