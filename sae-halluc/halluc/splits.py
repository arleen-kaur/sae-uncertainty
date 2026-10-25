"""Question-level splits for the span-level experiments.

    pilot       RAGTruth's official test questions (150). Reserved for the on-policy steering
                pilot: never used for feature discovery or screening.
    discovery   60% of the remaining (train) questions, chosen by a seeded hash of source_id.
    validation  the other 40%.

Splitting is by *question* (source_id): all six responses to a question land in the same split,
so no passage text leaks between splits. The assignment is a pure function of
(source_id, seed), so every script agrees without sharing a file.
"""

import hashlib

DISCOVERY = "discovery"
VALIDATION = "validation"
PILOT = "pilot"


def split_of(source_id, ragtruth_split, seed=42, discovery_frac=0.6):
    if ragtruth_split == "test":
        return PILOT
    h = hashlib.sha256(("%d:%s" % (seed, source_id)).encode()).hexdigest()
    u = int(h[:12], 16) / float(16 ** 12)
    return DISCOVERY if u < discovery_frac else VALIDATION
