"""Matched sampling of hallucinated vs faithful tokens *within the same response*.

Why match: faithful tokens are copied from the context far more often than hallucinated ones
(~33% vs ~8% copy tokens) and sit earlier in the response. Copied tokens are low-entropy, so
without matching the "confident + faithful" group would mostly be copy tokens.

For every sampled hallucinated token we pick one faithful token from the same response
(never reused), preferring the tightest match:

    tier 0   same (token type, copy status vs prompt, relative-position bin)
    tier 1   same (token type, copy status)
    tier 2   same copy status

and record the tier so analyses can restrict to tight matches. Hallucinated tokens are capped
per span (`max_per_span`): tokens within one span are strongly dependent, so a few random
tokens per span keep long spans from dominating.
"""

from collections import defaultdict

from .alignment import HALL, FAITHFUL


def _cell(tok, pos_bins):
    return (tok["type"], tok["ctx_status"], min(pos_bins - 1, int(tok["rel_pos"] * pos_bins)))


def sample_matched_tokens(tokens, rng, max_per_span=8, pos_bins=5):
    """Return (pairs, n_unmatched).

    `pairs` is a list of dicts {pair_id, tier, hall, faithful} where hall/faithful are token
    records (see pipeline.process_item). `rng` is a random.Random instance.
    """
    by_span = defaultdict(list)
    for t in tokens:
        if t["label"] == HALL and t["predict_idx"] >= 0:
            by_span[t["span_idx"]].append(t)

    chosen_hall = []
    for k in sorted(by_span):
        toks = by_span[k]
        if len(toks) > max_per_span:
            toks = rng.sample(toks, max_per_span)
        chosen_hall.extend(sorted(toks, key=lambda t: t["seq_idx"]))
    rng.shuffle(chosen_hall)

    pool = [t for t in tokens if t["label"] == FAITHFUL and t["predict_idx"] >= 0]
    idx0, idx1, idx2 = defaultdict(list), defaultdict(list), defaultdict(list)
    for t in pool:
        idx0[_cell(t, pos_bins)].append(t)
        idx1[(t["type"], t["ctx_status"])].append(t)
        idx2[t["ctx_status"]].append(t)

    used = set()
    pairs, unmatched = [], 0
    for h in chosen_hall:
        match, tier = None, None
        for tier_i, (index, key) in enumerate((
            (idx0, _cell(h, pos_bins)),
            (idx1, (h["type"], h["ctx_status"])),
            (idx2, h["ctx_status"]),
        )):
            cands = [t for t in index.get(key, ()) if t["seq_idx"] not in used]
            if cands:
                match, tier = rng.choice(cands), tier_i
                break
        if match is None:
            unmatched += 1
            continue
        used.add(match["seq_idx"])
        pairs.append({"pair_id": len(pairs), "tier": tier, "hall": h, "faithful": match})
    return pairs, unmatched
