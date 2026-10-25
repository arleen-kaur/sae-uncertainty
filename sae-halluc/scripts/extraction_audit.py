"""Audit an extraction run (CPU only). Usage:

    python scripts/extraction_audit.py --run-dir results/token_audit_discovery200
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from halluc import audit  # noqa: E402


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def show(title, d):
    print("\n== %s ==" % title)
    for k, v in d.items():
        if isinstance(v, float):
            print("  %-34s %.4f" % (k, v))
        else:
            print("  %-34s %s" % (k, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--percentile", type=float, default=25)
    ap.add_argument("--sae-C", type=float, default=0.05, help="inverse L2 strength for SAE logistic probe")
    ap.add_argument("--skip-predictive", action="store_true")
    args = ap.parse_args()
    d = Path(args.run_dir)

    pop = load(d / "population.pkl")
    pay = {k: load(d / ("token_features_%s.pkl" % k)) for k in ("predict", "self")}
    records = pay["predict"]["per_question_data"]
    layers = pay["predict"]["config"]["sae_layers"]
    print("Run: %s" % d)
    print("Config: split=%s layers=%s n_pairs=%s" % (
        pop["config"].get("split"), layers, pop["config"].get("n_pairs")))

    report = {}
    report["population"] = audit.population_report(pop["population"])
    pr = report["population"]
    show("All response tokens: entropy/surprisal separability (hall vs faithful)",
         {k: v for k, v in pr.items() if k != "entropy_auroc_by_type"})
    show("Entropy AUROC by token type", {k: v for k, v in pr["entropy_auroc_by_type"].items()})

    report["paired"] = audit.paired_report(records)
    show("Matched pairs (primary token-level contrast)", report["paired"])

    report["quadrants"] = audit.quadrant_sizes(records, args.percentile)
    show("Het-style quadrant sizes at p%g (min cell must be >= ~100 to be usable)" % args.percentile,
         report["quadrants"])

    if not args.skip_predictive:
        report["predictive"] = audit.predictive_report(records, pay, layers, args.sae_C)
        show("Group-CV AUROC, hall vs faithful (split by source question)", report["predictive"])
        print("\nRead this as: does SAE beat 'nuisance+entropy'? If not, the features add nothing beyond"
              " copy/type/position and uncertainty, and the thesis claim needs another route.")

    with open(d / "audit.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\nSaved %s" % (d / "audit.json"))


if __name__ == "__main__":
    main()
