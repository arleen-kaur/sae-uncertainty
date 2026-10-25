"""CPU-only audit of an extraction run: is there signal, and is it more than copy/position/entropy?

All functions take the payloads written by experiments/extract_token_features.py.
scipy / scikit-learn are optional: sections that need them are skipped with a note.
"""

import numpy as np

try:
    from scipy import stats as _st
    from scipy import sparse as _sp
except Exception:  # noqa: BLE001
    _st = _sp = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
except Exception:  # noqa: BLE001
    LogisticRegression = GroupKFold = roc_auc_score = None

from .extract import LABEL_CODES, TYPE_CODES
from .tokens import ALL_TYPES

HALL, FAITHFUL = LABEL_CODES["hall"], LABEL_CODES["faithful"]


def auroc(pos, neg):
    """P(pos > neg) with ties as 1/2 (Mann-Whitney U / (n_pos n_neg))."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    # average ranks for ties
    i = 0
    r = np.empty(len(allv))
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    ranks[order] = r
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def population_report(population):
    """Entropy / surprisal separability over ALL response tokens (hall vs faithful)."""
    cat = {k: np.concatenate([p[k] for p in population])
           for k in ("label", "type", "ctx_status", "entropy", "surprisal")}
    h, f = cat["label"] == HALL, cat["label"] == FAITHFUL
    out = {"n_hall_tokens": int(h.sum()), "n_faithful_tokens": int(f.sum())}
    for name in ("entropy", "surprisal"):
        out["%s_auroc_all" % name] = auroc(cat[name][h], cat[name][f])
        m = cat["ctx_status"] == 0
        out["%s_auroc_novel_only" % name] = auroc(cat[name][h & m], cat[name][f & m])
        out["%s_mean_hall" % name] = float(cat[name][h].mean()) if h.any() else float("nan")
        out["%s_mean_faithful" % name] = float(cat[name][f].mean()) if f.any() else float("nan")
    out["copy_status_hall"] = np.bincount(cat["ctx_status"][h], minlength=3).tolist()
    out["copy_status_faithful"] = np.bincount(cat["ctx_status"][f], minlength=3).tolist()
    by_type = {}
    for t in ALL_TYPES:
        m = cat["type"] == TYPE_CODES[t]
        if (h & m).sum() >= 20 and (f & m).sum() >= 20:
            by_type[t] = {"n_hall": int((h & m).sum()),
                          "entropy_auroc": auroc(cat["entropy"][h & m], cat["entropy"][f & m])}
    out["entropy_auroc_by_type"] = by_type
    return out


def _pairs(records):
    by = {}
    for r in records:
        by.setdefault(r["pair_id"], {})[r["role"]] = r
    return [(v["hall"], v["faithful"]) for v in by.values() if "hall" in v and "faithful" in v]


def paired_report(records):
    """Matched hall-vs-faithful comparison: the primary token-level contrast."""
    pairs = _pairs(records)
    out = {"n_pairs": len(pairs), "tier_counts": {}}
    for h, f in pairs:
        out["tier_counts"][h["tier"]] = out["tier_counts"].get(h["tier"], 0) + 1
    for name in ("entropy", "surprisal"):
        d = np.array([h[name] - f[name] for h, f in pairs])
        out["%s_mean_diff" % name] = float(d.mean()) if len(d) else float("nan")
        out["%s_frac_hall_higher" % name] = float((d > 0).mean()) if len(d) else float("nan")
        if _st is not None and len(d) > 10 and np.any(d != 0):
            out["%s_wilcoxon_p" % name] = float(_st.wilcoxon(d).pvalue)
    # hall tokens that are novel (ctx 0) vs short copies etc.
    out["pairs_by_ctx_status"] = np.bincount([h["ctx_status"] for h, _ in pairs], minlength=3).tolist()
    return out


def quadrant_sizes(records, percentile=25):
    """Het-style 2x2: correct(=faithful)/incorrect(=hall) x confident/uncertain by entropy percentile."""
    ent = np.array([r["entropy"] for r in records])
    lo, hi = np.percentile(ent, percentile), np.percentile(ent, 100 - percentile)
    cor = np.array([r["correct"] for r in records])
    conf, unc = ent <= lo, ent >= hi
    q = {"A_correct_confident": int((cor & conf).sum()), "B_incorrect_confident": int((~cor & conf).sum()),
         "C_correct_uncertain": int((cor & unc).sum()), "D_incorrect_uncertain": int((~cor & unc).sum())}
    q["entropy_cutoffs"] = [float(lo), float(hi)]
    q["min_cell"] = min(q["A_correct_confident"], q["B_incorrect_confident"],
                        q["C_correct_uncertain"], q["D_incorrect_uncertain"])
    return q


def _nuisance(records):
    X = []
    for r in records:
        oh = [0.0] * len(ALL_TYPES)
        oh[TYPE_CODES[r["type"]]] = 1.0
        X.append(oh + [float(r["ctx_status"] == 1), float(r["ctx_status"] == 2),
                       r["rel_pos"], np.log1p(r["ctx_run"]), np.log1p(r["passage_run"])])
    return np.array(X, dtype=np.float32)


def _sparse_features(records, payload_rows, layer, min_count=5):
    rows, cols, vals = [], [], []
    for i, d in enumerate(payload_rows):
        f = d["sae_features"][layer]
        rows.extend([i] * len(f["indices"]))
        cols.extend(f["indices"].tolist())
        vals.extend(np.log1p(f["values"]).tolist())
    if not cols:
        return None
    n_feat = max(cols) + 1
    X = _sp.csr_matrix((vals, (rows, cols)), shape=(len(payload_rows), n_feat), dtype=np.float32)
    keep = np.flatnonzero(np.asarray((X > 0).sum(0)).ravel() >= min_count)
    return X[:, keep] if len(keep) else None


def _cv_auroc(X, y, groups, C=0.05, folds=5, standardize=False):
    if np.unique(groups).size < folds:
        folds = max(2, np.unique(groups).size)
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        if standardize:
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        clf = LogisticRegression(C=C, max_iter=500)
        clf.fit(Xtr, y[tr])
        scores[te] = clf.decision_function(Xte)
    return float(roc_auc_score(y, scores))


def predictive_report(records, payloads, layers, sae_C=0.05):
    """Group-CV AUROC (hall vs faithful) for: nuisance, entropy, nuisance+entropy, and SAE features
    at each position x layer, with SAE+nuisance+entropy to see what features add beyond the baselines."""
    if LogisticRegression is None or _sp is None:
        return {"skipped": "scikit-learn / scipy not available"}
    y = np.array([0 if r["correct"] else 1 for r in records])
    groups = np.array([r["source_id"] for r in records])
    nuis = _nuisance(records)
    ent = np.array([[r["entropy"], r["surprisal"]] for r in records], dtype=np.float32)
    out = {
        "nuisance_only": _cv_auroc(nuis, y, groups, C=1.0, standardize=True),
        "entropy_only": _cv_auroc(ent[:, :1], y, groups, C=1.0, standardize=True),
        "nuisance+entropy": _cv_auroc(np.hstack([nuis, ent[:, :1]]), y, groups, C=1.0, standardize=True),
    }
    base = np.hstack([nuis, ent[:, :1]])
    base = (base - base.mean(0)) / (base.std(0) + 1e-6)

    # Capacity-matched control: a probe that only sees WHICH token it is (bag of surface forms).
    # If the SAE probe does not clearly beat this, SAE features are mostly encoding token identity.
    surf = [r["text"].strip().lower() for r in records]
    counts = {}
    for s in surf:
        counts[s] = counts.get(s, 0) + 1
    vocab = {s: i for i, s in enumerate(sorted(s for s, c in counts.items() if c >= 5))}
    if vocab:
        r_, c_ = zip(*[(i, vocab[s]) for i, s in enumerate(surf) if s in vocab])
        lex = _sp.csr_matrix((np.ones(len(r_), dtype=np.float32), (r_, c_)), shape=(len(records), len(vocab)))
        out["lexical_only"] = _cv_auroc(lex, y, groups, C=sae_C)
        out["lexical+baselines"] = _cv_auroc(_sp.hstack([lex, _sp.csr_matrix(base)]).tocsr(), y, groups, C=sae_C)
        out["lexical_vocab_size"] = len(vocab)
    # Sanity: labels shuffled across pairs must give ~0.5 for the SAE probe (guards against leakage)
    y_shuf = np.random.RandomState(0).permutation(y)
    for kind, payload in payloads.items():
        rows = payload["per_question_data"]
        for l in layers:
            X = _sparse_features(records, rows, l)
            if X is None:
                out["sae_%s_L%d" % (kind, l)] = None
                continue
            out["sae_%s_L%d" % (kind, l)] = _cv_auroc(X, y, groups, C=sae_C)
            if kind == "predict" and l == layers[0]:
                out["sae_predict_L%d_SHUFFLED_LABELS" % l] = _cv_auroc(X, y_shuf, groups, C=sae_C)
            out["sae+baselines_%s_L%d" % (kind, l)] = _cv_auroc(
                _sp.hstack([X, _sp.csr_matrix(base)]).tocsr(), y, groups, C=sae_C)
    return out
