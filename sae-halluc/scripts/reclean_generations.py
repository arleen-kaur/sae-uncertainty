"""Re-clean saved generations from their raw text (no GPU) and keep only usable answers.

    python scripts/reclean_generations.py --input outputs/onpolicy_gen.jsonl --output outputs/onpolicy_clean.jsonl

Non-answers (empty / echo of the question / too short) are dropped and counted, not judged. The raw model
text is untouched in `gen.raw`; the old single-line response is kept in `gen.response_v1`.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from halluc.cleaning import classify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.input)]
    reasons = collections.Counter()
    kept = []
    for r in rows:
        resp, reason = classify(r["gen"]["raw"], r["question"])
        reasons[reason or "kept"] += 1
        if reason:
            continue
        r["gen"]["response_v1"] = r["response"]
        r["response"] = resp
        kept.append(r)

    with open(args.output, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    per_q = collections.Counter(r["source_id"] for r in kept)
    n_q = len(set(r["source_id"] for r in rows))
    words = sorted(len(r["response"].split()) for r in kept)
    print("input %d | kept %d | %s" % (len(rows), len(kept),
          " | ".join("%s %d" % (k, v) for k, v in sorted(reasons.items()) if k != "kept")))
    print("questions with >=1 usable answer: %d of %d | usable answers per question (n_answers: n_questions): %s"
          % (len(per_q), n_q, dict(sorted(collections.Counter(per_q.values()).items()))))
    if words:
        print("words per kept response  min/median/90th/max: %d %d %d %d"
              % (words[0], words[len(words) // 2], words[int(len(words) * 0.9)], words[-1]))
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
