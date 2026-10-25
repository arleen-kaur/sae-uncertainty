"""Character-span -> token alignment.

RAGTruth labels hallucinated spans as character offsets into the response. The model
works on tokens, so every token of the teacher-forced text gets a label:

    hall      >= min_hall_frac of the token's (non-whitespace) characters lie in a strict span
    boundary  0 < overlap < min_hall_frac   (BPE token straddles a span edge; exclude from both classes)
    ambig     overlaps an `implicit_true` span only (soft label; exclude from both classes)
    faithful  touches no labeled span at all
    near      faithful, but within `margin_tokens` of a labeled token (optional contamination guard)
    space     whitespace-only token (e.g. newlines)
    prompt    token belongs to the prompt, not the response
    special   zero-width token (BOS etc.)

The code is tokenizer-agnostic: it only needs `offsets`, a list of (start, end) character
offsets into `text`. It does not matter whether the tokenizer puts a leading space inside
a token's offsets or trims it: whitespace is stripped from each token's range before
overlaps are computed (when `text` is given).

Position convention (used downstream): the residual at sequence position i-1 is the state
that *predicts* token i, so the "generation-time" state for token i is `predict_idx = i-1`.
"""

from dataclasses import dataclass

HALL = "hall"
AMBIG = "ambig"
FAITHFUL = "faithful"
BOUNDARY = "boundary"
NEAR = "near"
SPACE = "space"
PROMPT = "prompt"
SPECIAL = "special"

LABELED = (HALL, BOUNDARY, AMBIG)


@dataclass
class TokenLabel:
    label: str
    span_idx: int      # index into strict_spans of the span with the largest overlap, or -1
    hall_frac: float   # fraction of the token's (non-whitespace) chars inside strict spans
    in_response: bool


def merge_intervals(intervals):
    """Union of [start, end) intervals -> sorted, non-overlapping list of (start, end)."""
    out = []
    for s, e in sorted((s, e) for s, e in intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _trim(text, s, e):
    """Shrink [s, e) so it neither starts nor ends on whitespace."""
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def label_tokens(offsets, resp_start, resp_end, strict_spans, ambiguous_spans=(),
                 text=None, min_hall_frac=0.5, margin_tokens=0):
    """Label every token. `*_spans` are (start, end[, type]) relative to the response start."""
    abs_strict = [(sp[0] + resp_start, sp[1] + resp_start) for sp in strict_spans]
    abs_amb = [(sp[0] + resp_start, sp[1] + resp_start) for sp in ambiguous_spans]
    merged_strict = merge_intervals(abs_strict)
    merged_amb = merge_intervals(abs_amb)

    labels = []
    for (s, e) in offsets:
        if e <= s:
            labels.append(TokenLabel(SPECIAL, -1, 0.0, False))
            continue
        if e <= resp_start or s >= resp_end:
            labels.append(TokenLabel(PROMPT, -1, 0.0, False))
            continue

        ts, te = _trim(text, s, e) if text is not None else (s, e)
        if te <= ts:  # whitespace-only token inside the response
            labels.append(TokenLabel(SPACE, -1, 0.0, True))
            continue

        length = te - ts
        hall_chars = sum(overlap(ts, te, a, b) for a, b in merged_strict)
        frac = hall_chars / length
        best, best_o = -1, 0
        for k, (a, b) in enumerate(abs_strict):
            o = overlap(ts, te, a, b)
            if o > best_o:
                best, best_o = k, o
        amb_chars = sum(overlap(ts, te, a, b) for a, b in merged_amb)

        if frac >= min_hall_frac:
            lab = HALL
        elif frac > 0:
            lab = BOUNDARY
        elif amb_chars > 0:
            lab = AMBIG
        else:
            lab = FAITHFUL
        labels.append(TokenLabel(lab, best, frac, True))

    if margin_tokens > 0:
        labeled_idx = [i for i, l in enumerate(labels) if l.label in LABELED]
        if labeled_idx:
            for i, l in enumerate(labels):
                if l.label == FAITHFUL and min(abs(i - j) for j in labeled_idx) <= margin_tokens:
                    l.label = NEAR
    return labels


def check_span_coverage(offsets, labels, text, resp_start, strict_spans):
    """Sanity check: for every strict span, the response tokens that overlap it must
    (a) exist and (b) together cover the span's characters.

    Returns a list of (span_idx, reason) for spans that fail; empty list = alignment OK.
    """
    failures = []
    for k, sp in enumerate(strict_spans):
        a, b = sp[0] + resp_start, sp[1] + resp_start
        a, b = _trim(text, a, b)
        if b <= a:
            failures.append((k, "empty_after_trim"))
            continue
        covering = [(s, e) for (s, e), l in zip(offsets, labels)
                    if l.in_response and overlap(s, e, a, b) > 0]
        if not covering:
            failures.append((k, "no_tokens"))
            continue
        lo = min(s for s, _ in covering)
        hi = max(e for _, e in covering)
        if not (lo <= a and hi >= b):
            failures.append((k, "tokens_do_not_cover_span"))
    return failures
