"""Pool several blind audits into population-level estimates for every judge and for k-of-N voting rules.

    python scripts/pooled_audit.py --judged outputs/onpolicy_oneshot_judged4.jsonl \
        --audit outputs/audit/audit_sheet_human_corrected.md:outputs/audit/audit_key.json \
        --audit outputs/audit2/audit_sheet_claude.md:outputs/audit2/audit_key.json

Each audit over-samples some region, so raw accuracies are not comparable.  Here every answer belongs to one of
7 cells (defined from the judges' outputs); an audited item is weighted by (cell size) / (items audited in that
cell), which turns the pooled sample into an estimate for ALL answers.  95% intervals: bootstrap, resampling
audited items within each cell.  Human label: S = supported; U or C = hallucinated.
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from score_audit import parse_sheet  # noqa: E402

NAMES = ["lettuce", "hhem", "lynx", "minicheck"]


def cell(flags):
    lt, hh, ly, mc = (flags[n] for n in NAMES)
    if lt and hh:
        return "lettuce+hhem"
    if lt:
        return "lettuce_only"
    if hh:
        return "hhem_only"
    return "clean2_" + ("lynx+mc" if ly and mc else "mc_only" if mc else "lynx_only" if ly else "none")


def rules(flags):
    votes = sum(1 for n in NAMES if flags[n])
    out = {n: bool(flags[n]) for n in NAMES}
    out.update({"any>=1": votes >= 1, "votes>=2": votes >= 2, "votes>=3": votes >= 3, "votes=4": votes == 4})
    return out


def estimate(sample, sizes):
    """sample: {cell: [(rule_flags, hallucinated)]} -> {rule: (prevalence-free metrics)}"""
    tot = {"H": 0.0}
    acc = {}
    for c, items in sample.items():
        if not items:
            continue
        w = sizes[c] / float(len(items))
        for flags, hall in items:
            tot["H"] += w * hall
            for r, f in flags.items():
                a = acc.setdefault(r, {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0})
                a["tp" if f and hall else "fp" if f else "fn" if hall else "tn"] += w
    res = {"prevalence": tot["H"] / float(sum(sizes.values()))}
    for r, a in acc.items():
        p = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else float("nan")
        rc = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else float("nan")
        f1 = 2 * p * rc / (p + rc) if p == p and rc == rc and p + rc else float("nan")
        n = sum(a.values())
        res[r] = {"precision": p, "recall": rc, "f1": f1, "accuracy": (a["tp"] + a["tn"]) / n,
                  "flagged_rate": (a["tp"] + a["fp"]) / n}
    return res


def pct(vals, q):
    v = sorted(x for x in vals if x == x)
    return v[int(q * (len(v) - 1))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--audit", action="append", required=True, help="SHEET.md:KEY.json (repeatable)")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    rows = {}
    for line in open(args.judged):
        r = json.loads(line)
        rows[r["response_id"]] = {n: r["judge"][n]["hallucinated"] for n in NAMES}
    sizes = {}
    for flags in rows.values():
        sizes[cell(flags)] = sizes.get(cell(flags), 0) + 1

    sample, seen = {c: [] for c in sizes}, set()
    for spec in args.audit:
        sheet, key = spec.split(":")
        labels = parse_sheet(open(sheet, encoding="utf-8").read())
        for rid in json.load(open(key)):
            if rid in labels and rid not in seen:
                seen.add(rid)
                sample[cell(rows[rid])].append((rules(rows[rid]), int(labels[rid] != "S")))
    print("cell sizes (all %d answers) and audited items per cell, with how many were hallucinated:" % len(rows))
    for c in sorted(sizes):
        n = len(sample[c])
        h = sum(x[1] for x in sample[c])
        print("  %-18s size %3d   audited %2d   hallucinated %2d" % (c, sizes[c], n, h))
    empty = [c for c in sizes if not sample[c]]
    if empty:
        print("WARNING: no audited items in %s; they are ignored in the estimates" % empty)

    point = estimate(sample, sizes)
    rng = random.Random(0)
    boots = []
    for _ in range(args.boot):
        bs = {c: [rng.choice(v) for _ in v] if v else [] for c, v in sample.items()}
        boots.append(estimate(bs, sizes))

    def ci(get):
        vals = [get(b) for b in boots]
        return pct(vals, 0.025), pct(vals, 0.975)

    lo, hi = ci(lambda b: b["prevalence"])
    print("\nestimated share of ALL answers that are truly hallucinated: %.1f%%  (95%% CI %.1f-%.1f%%)  = about %d of %d"
          % (100 * point["prevalence"], 100 * lo, 100 * hi, round(point["prevalence"] * len(rows)), len(rows)))
    print("\n%-10s %-17s %-17s %-17s %s" % ("rule", "precision", "recall", "F1", "flags % of all"))
    for r in NAMES + ["any>=1", "votes>=2", "votes>=3", "votes=4"]:
        cells = []
        for m in ("precision", "recall", "f1"):
            a, b = ci(lambda x, r=r, m=m: x[r][m])
            cells.append("%.2f (%.2f-%.2f)" % (point[r][m], a, b))
        print("%-10s %-17s %-17s %-17s %.1f%%" % (r, cells[0], cells[1], cells[2], 100 * point[r]["flagged_rate"]))
    print("\nNote: labels are not all human (see the audit sheets); intervals reflect sampling error only.")


if __name__ == "__main__":
    main()
