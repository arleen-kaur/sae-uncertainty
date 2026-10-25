"""Build a BLIND human-audit sheet from judged on-policy answers.

    python scripts/make_audit_sheet.py --input outputs/onpolicy_oneshot_judged.jsonl --out-dir outputs/audit

Writes
    audit_sheet.md   what the human reads and labels (no judge information at all)
    audit_key.json   {response_id: {lettuce, hhem, consensus, stratum}}  -- do not open until labelling is done

Sampling: at most one answer per question (the 4 samples of a question are near duplicates), strata chosen so
that the judges' disagreements are well represented, order shuffled.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

STRATA = [("both_flagged", 15), ("lettuce_only", 8), ("hhem_only", 7), ("both_clean", 15)]
HEADER = """# Blind audit sheet

For each item write ONE letter after `LABEL:` (nothing else on that line):

- `S`  everything in the answer is supported by the passages
- `U`  the answer adds something the passages do not say (even if it is true in the real world)
- `C`  the answer contradicts the passages

Judge only against the PASSAGES, not against what you know. If unsure between S and U, ask: could I point to
the sentence in the passages that says this? If not, it is U.

"""


STRATA_EXTRA = [("lynx_and_minicheck", 8), ("minicheck_only", 14), ("lynx_only", 8), ("all_clean", 6)]


def stratum_extra(r):
    """Second audit: only answers that LettuceDetect AND HHEM both passed (the region the first audit barely saw),
    split by what Lynx and MiniCheck say. Anything flagged by lettuce or hhem returns None (not sampled)."""
    j = r["judge"]
    if j["lettuce"]["hallucinated"] or j["hhem"]["hallucinated"]:
        return None
    ly, mc = j["lynx"]["hallucinated"], j["minicheck"]["hallucinated"]
    if ly and mc:
        return "lynx_and_minicheck"
    if mc:
        return "minicheck_only"
    if ly:
        return "lynx_only"
    return "all_clean"


def stratum(r):
    lt, hh = r["judge"]["lettuce"]["hallucinated"], r["judge"]["hhem"]["hallucinated"]
    if lt and hh:
        return "both_flagged"
    if lt:
        return "lettuce_only"
    if hh:
        return "hhem_only"
    return "both_clean"


def squash(text):
    return re.sub(r"[ \t]+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--design", choices=["base", "extra"], default="base",
                    help="base: strata from lettuce/hhem (first audit); extra: lettuce+hhem-clean answers split by lynx/minicheck")
    ap.add_argument("--exclude-key", default=None,
                    help="an earlier audit_key.json: its questions are not sampled again")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.input)]
    rng = random.Random(args.seed)
    strata, strat_fn = (STRATA_EXTRA, stratum_extra) if args.design == "extra" else (STRATA, stratum)
    by_stratum = {name: [] for name, _ in strata}
    for r in rows:
        s = strat_fn(r)
        if s is not None:
            by_stratum[s].append(r)

    used_questions, picked = set(), []
    if args.exclude_key:
        old = set(json.load(open(args.exclude_key)))
        used_questions = {r["source_id"] for r in rows if r["response_id"] in old}
        print("excluding %d already-audited questions" % len(used_questions))
    for name, n in strata:
        pool = by_stratum[name][:]
        rng.shuffle(pool)
        taken = 0
        for r in pool:
            if taken == n:
                break
            if r["source_id"] in used_questions:
                continue
            used_questions.add(r["source_id"])
            picked.append((name, r))
            taken += 1
        print("%-13s available %3d  picked %2d" % (name, len(pool), taken))
    rng.shuffle(picked)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key, parts = {}, [HEADER]
    for i, (name, r) in enumerate(picked, 1):
        key[r["response_id"]] = {"lettuce": r["judge"]["lettuce"]["hallucinated"],
                                 "hhem": r["judge"]["hhem"]["hallucinated"],
                                 "consensus": r["judge"].get("consensus_lettuce_hhem", r["judge"].get("consensus_hallucinated")),
                                 "stratum": name}
        parts.append("### ITEM %d | id %s\n\n**Question:** %s\n\n**Answer:** %s\n\n**Passages:**\n\n%s\n\nLABEL: \n\n---\n\n"
                     % (i, r["response_id"], squash(r["question"]), squash(r["response"]),
                        re.sub(r"(passage \d:)", r"\n\1", squash(r["passages"]))))
    (out_dir / "audit_sheet.md").write_text("".join(parts), encoding="utf-8")
    (out_dir / "audit_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print("wrote %d items to %s" % (len(picked), out_dir))


if __name__ == "__main__":
    main()
