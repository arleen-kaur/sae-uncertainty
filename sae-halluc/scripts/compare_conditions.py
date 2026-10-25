"""Per-feature RAG vs no-RAG comparison of hallucinated-vs-faithful token features (CPU only).

For each layer, using the matched pairs (same response, same sampled tokens in both runs):

  1. d = activation(hallucinated token) - activation(matched faithful token), averaged over the
     pairs of one response (responses, not tokens, are the unit: tokens within a response are
     not independent),
  2. Wilcoxon signed-rank over responses per feature, Benjamini-Hochberg FDR across the features
     tested in that layer/condition,
  3. compare the two conditions:
       RAG-only      significant with passages, not without   (depends on the retrieved context)
       resistant     significant in both, same sign           (separates hall/faithful regardless)
       no-RAG-only   significant only without passages

Usage:
    python scripts/compare_conditions.py \
        --rag results/token_audit_discovery200 --no-rag results/token_audit_discovery200_norag \
        --position predict --output results/compare_rag_norag_predict.csv
"""

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def response_diffs(payload, layer, n_feat=32768):
    """-> (response_ids, D) with D[r, f] = mean over the response's pairs of act(hall) - act(faithful)."""
    rows = payload["per_question_data"]
    by_pair = defaultdict(dict)
    for i, r in enumerate(rows):
        by_pair[r["pair_id"]][r["role"]] = i
    resp_pairs = defaultdict(list)
    for pid, d in by_pair.items():
        if "hall" in d and "faithful" in d:
            resp_pairs[rows[d["hall"]]["response_id"]].append((d["hall"], d["faithful"]))

    def vec(i):
        f = rows[i]["sae_features"][layer]
        v = np.zeros(n_feat, dtype=np.float32)
        v[f["indices"]] = np.log1p(f["values"])
        return v

    ids, mats = [], []
    for rid in sorted(resp_pairs, key=str):
        diff = np.mean([vec(h) - vec(f) for h, f in resp_pairs[rid]], axis=0)
        ids.append(rid)
        mats.append(sparse.csr_matrix(diff))
    return ids, sparse.vstack(mats).tocsc()


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def per_feature(D, min_resp):
    """Wilcoxon signed-rank per feature with >= min_resp non-zero responses. -> {feat: (mean, p)}."""
    res = {}
    for f in range(D.shape[1]):
        col = D.getcol(f)
        nz = col.data
        if len(nz) < min_resp:
            continue
        dense = np.zeros(D.shape[0])
        dense[col.indices] = nz
        try:
            p = stats.wilcoxon(dense[dense != 0]).pvalue
        except ValueError:
            continue
        res[f] = (float(dense.mean()), float(p))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag", required=True)
    ap.add_argument("--no-rag", required=True)
    ap.add_argument("--position", choices=["predict", "self"], default="predict")
    ap.add_argument("--min-resp", type=int, default=15, help="min responses where the feature differs")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pr = load(Path(args.rag) / ("token_features_%s.pkl" % args.position))
    pn = load(Path(args.no_rag) / ("token_features_%s.pkl" % args.position))
    layers = pr["config"]["sae_layers"]
    out_rows, summary = [], []
    for l in layers:
        ids_r, Dr = response_diffs(pr, l)
        ids_n, Dn = response_diffs(pn, l)
        assert ids_r == ids_n, "runs must cover the same responses (same --seed/--max-responses)"
        a, b = per_feature(Dr, args.min_resp), per_feature(Dn, args.min_resp)
        # direct test of the difference between conditions (per response): d_rag - d_norag
        c = per_feature(Dr - Dn, args.min_resp)
        qc = dict(zip(sorted(c), bh([c[f][1] for f in sorted(c)]))) if c else {}
        feats = sorted(set(a) | set(b))
        qa = dict(zip(sorted(a), bh([a[f][1] for f in sorted(a)]))) if a else {}
        qb = dict(zip(sorted(b), bh([b[f][1] for f in sorted(b)]))) if b else {}
        n = {"rag_only": 0, "resistant": 0, "norag_only": 0}
        both_means = []
        for f in feats:
            ma, pa = a.get(f, (np.nan, np.nan))
            mb, pb = b.get(f, (np.nan, np.nan))
            sa, sb = qa.get(f, 1.0) < args.fdr, qb.get(f, 1.0) < args.fdr
            if sa and sb and np.sign(ma) == np.sign(mb):
                cat = "resistant"
            elif sa and not sb:
                cat = "rag_only"
            elif sb and not sa:
                cat = "norag_only"
            else:
                cat = ""
            if cat:
                n[cat if cat != "norag_only" else "norag_only"] += 1
            if f in a and f in b:
                both_means.append((ma, mb))
            out_rows.append({"layer": l, "feature": f, "mean_diff_rag": ma, "p_rag": pa, "q_rag": qa.get(f),
                             "mean_diff_norag": mb, "p_norag": pb, "q_norag": qb.get(f), "category": cat,
                             "gap_rag_minus_norag": c.get(f, (np.nan, np.nan))[0],
                             "q_gap": qc.get(f)})
        corr = float(np.corrcoef(np.array(both_means).T)[0, 1]) if len(both_means) > 3 else float("nan")
        n_gap = sum(1 for f in qc if qc[f] < args.fdr)
        n_gap_ragdep = sum(1 for f in qc if qc[f] < args.fdr and abs(a.get(f, (0, 1))[0]) > abs(b.get(f, (0, 1))[0]))
        summary.append((l, len(ids_r), len(a), len(b), n["rag_only"], n["resistant"], n["norag_only"], corr,
                        n_gap, n_gap_ragdep))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("layer  n_resp  tested_rag tested_norag | RAG-only resistant noRAG-only | corr(effect) |"
          " gap-sig  of which larger-with-passages")
    for s in summary:
        print("%5d %7d %10d %12d | %8d %9d %10d | %12.3f | %7d %10d" % s)
    print("\nFDR=%.2f, unit = response, position=%s. Saved %s" % (args.fdr, args.position, args.output))


if __name__ == "__main__":
    main()
