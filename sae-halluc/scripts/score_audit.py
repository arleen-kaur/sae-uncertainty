"""Score the judges against the human labels from the blind audit.

    python scripts/score_audit.py --sheet outputs/audit/audit_sheet.md --key outputs/audit/audit_key.json

Human label: S = supported -> "not hallucinated"; U or C -> "hallucinated".  Reports, per judge and for the
consensus: accuracy, precision, recall, and Cohen's kappa against the human, plus the S/U/C counts per stratum.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from halluc.judging import cohen_kappa  # noqa: E402


def parse_sheet(text):
    labels = {}
    for block in re.split(r"(?m)^### ITEM ", text)[1:]:
        m_id = re.search(r"\| id (\S+)", block)
        m_lab = re.search(r"(?m)^LABEL:[ \t]*([A-Za-z]?)", block)
        if m_id and m_lab and m_lab.group(1).upper() in ("S", "U", "C"):
            labels[m_id.group(1)] = m_lab.group(1).upper()
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--judged", default=None,
                    help="judged JSONL with more judges than the audit key (e.g. after adding lynx/minicheck); "
                         "their labels are looked up by response id")
    args = ap.parse_args()

    labels = parse_sheet(open(args.sheet, encoding="utf-8").read())
    key = json.load(open(args.key))
    names = ["lettuce", "hhem", "consensus"]
    if args.judged:
        rows = {}
        for line in open(args.judged):
            r = json.loads(line)
            rows[r.get("response_id") or r.get("id")] = r
        for i in key:
            j = rows[i]["judge"]
            judges = sorted(k for k in j if isinstance(j[k], dict))
            for k in judges:
                key[i][k] = j[k]["hallucinated"]
            votes = [key[i][k] for k in judges if key[i][k] is not None]
            key[i]["any_judge"] = any(votes) if votes else None
            key[i]["majority_all"] = j["consensus_hallucinated"]
            key[i]["consensus"] = j.get("consensus_lettuce_hhem", key[i].get("consensus"))
        names = judges + ["consensus", "majority_all", "any_judge"]
    n_items = len(key)
    ids = [i for i in key if i in labels]
    print("labelled %d of %d items" % (len(ids), n_items))
    if len(ids) < n_items:
        print("  (unlabelled items are skipped; finish them for the final numbers)")
    if not ids:
        return

    human = [labels[i] != "S" for i in ids]
    print("human: S %d | U %d | C %d  -> hallucinated (U or C) in %d of %d"
          % tuple([sum(labels[i] == k for i in ids) for k in "SUC"] + [sum(human), len(ids)]))
    by = collections.defaultdict(collections.Counter)
    for i in ids:
        by[key[i]["stratum"]][labels[i]] += 1
    for s in sorted(by):
        c = by[s]
        print("  %-13s S %2d  U %2d  C %2d" % (s, c["S"], c["U"], c["C"]))

    print("\njudge vs human (hallucinated = U or C)")
    print("  %-10s %4s %9s %10s %7s %7s" % ("judge", "n", "accuracy", "precision", "recall", "kappa"))
    for name in names:
        pairs = [(key[i][name], h) for i, h in zip(ids, human) if key[i][name] is not None]
        if not pairs:
            continue
        pred, ref = [p for p, _ in pairs], [h for _, h in pairs]
        tp = sum(1 for p, h in pairs if p and h)
        fp = sum(1 for p, h in pairs if p and not h)
        fn = sum(1 for p, h in pairs if not p and h)
        acc = sum(1 for p, h in pairs if p == h) / float(len(pairs))
        prec = tp / float(tp + fp) if tp + fp else float("nan")
        rec = tp / float(tp + fn) if tp + fn else float("nan")
        print("  %-10s %4d %9.2f %10.2f %7.2f %7.2f" % (name, len(pairs), acc, prec, rec, cohen_kappa(pred, ref)))
    print("\nNote: the strata over-sample disagreements, so these are NOT population rates; compare judges with each other.")


if __name__ == "__main__":
    main()
