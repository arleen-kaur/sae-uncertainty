"""Judge-agreement utilities for the on-policy pilot (pure python, no model dependencies).

Labels are booleans: True = "hallucinated". `None` means a judge has no verdict / the consensus is undecided.
"""

import math


def cohen_kappa(a, b):
    """Cohen's kappa for two boolean label lists (nan when chance agreement is 1)."""
    assert len(a) == len(b) and len(a) > 0
    n = float(len(a))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(1 for x in a if x) / n, sum(1 for y in b if y) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


def agreement(a, b):
    """Summary of two judges' response-level labels (items where either is None are dropped)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    both_pos = sum(1 for x, y in pairs if x and y)
    both_neg = sum(1 for x, y in pairs if not x and not y)
    return {"n": n, "both_hallucinated": both_pos, "both_clean": both_neg,
            "only_first": sum(1 for x, y in pairs if x and not y),
            "only_second": sum(1 for x, y in pairs if y and not x),
            "agreement": (both_pos + both_neg) / float(n),
            "kappa": cohen_kappa(xs, ys),
            "rate_first": sum(xs) / float(n), "rate_second": sum(ys) / float(n)}


def consensus(labels_by_judge):
    """Majority vote per item over the non-None votes; a tie (e.g. two judges disagreeing) gives None.

    labels_by_judge: {judge_name: [bool|None, ...]} (all lists the same length)
    """
    names = sorted(labels_by_judge)
    n = len(labels_by_judge[names[0]])
    assert all(len(labels_by_judge[k]) == n for k in names)
    out = []
    for i in range(n):
        votes = [labels_by_judge[k][i] for k in names if labels_by_judge[k][i] is not None]
        pos = sum(1 for v in votes if v)
        neg = len(votes) - pos
        out.append(True if pos > neg else (False if neg > pos else None))
    return out


def spans_to_tuples(spans, response_len, label="lettuce"):
    """Judge spans [{'start','end',...}] -> sorted (start, end, label) tuples clipped to the response."""
    out = []
    for s in spans:
        a, b = max(0, int(s["start"])), min(int(response_len), int(s["end"]))
        if b > a:
            out.append((a, b, label))
    return sorted(out)
