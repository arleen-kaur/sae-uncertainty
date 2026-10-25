"""Context-copy features.

Faithful response tokens are often copied from the context, and copied tokens are
low-entropy (induction / copy circuits). Hallucinated tokens are, by definition, not
supported by the passages. Without a control, "confident + faithful" would be mostly
copy tokens and top SAE features could be copy/induction features, not hallucination
features.

For each response token i we compute

    run[i]  = length of the longest contiguous run of response tokens ending at i that also
              appears contiguously in the reference token sequence
              (0 = token never occurs in the reference at all)

    status  = 0 'novel'  : run == 0              token not in the reference
              1 'short'  : 0 < run < min_run     token occurs, but not as part of a long copied run
              2 'copy'   : run >= min_run        induction-predictable copy
                                                  (the previous min_run-1 tokens matched a place in the
                                                  reference and token i is the next reference token)

The functions work on any hashable tokens (ids, or normalised strings).
"""

from collections import defaultdict

STATUS_NOVEL = 0
STATUS_SHORT = 1
STATUS_COPY = 2
STATUS_NAMES = {STATUS_NOVEL: "novel", STATUS_SHORT: "short", STATUS_COPY: "copy"}


def copy_run_lengths(resp_tokens, ref_tokens):
    """Longest contiguous match ending at each response token (see module docstring).

    Dynamic programme over reference positions: `prev[j]` is the match length ending at
    reference position j for the previous response token, so the run extends iff
    `prev` contains j-1. Cost is O(sum over tokens of their occurrences in the reference).
    """
    positions = defaultdict(list)
    for j, t in enumerate(ref_tokens):
        positions[t].append(j)

    out = []
    prev = {}
    for t in resp_tokens:
        cur = {}
        for j in positions.get(t, ()):
            cur[j] = prev.get(j - 1, 0) + 1
        out.append(max(cur.values()) if cur else 0)
        prev = cur
    return out


def copy_status(run_length, min_run=3):
    if run_length <= 0:
        return STATUS_NOVEL
    if run_length >= min_run:
        return STATUS_COPY
    return STATUS_SHORT
