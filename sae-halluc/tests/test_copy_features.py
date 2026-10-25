import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from halluc.copy_features import (  # noqa: E402
    copy_run_lengths, copy_status, STATUS_NOVEL, STATUS_SHORT, STATUS_COPY,
)
from halluc.tokens import token_type  # noqa: E402


def naive_runs(resp, ref):
    """Brute force: longest L such that resp[i-L+1..i] is a contiguous subsequence of ref."""
    out = []
    for i in range(len(resp)):
        best = 0
        for L in range(1, i + 2):
            window = resp[i - L + 1:i + 1]
            found = any(ref[j:j + L] == window for j in range(len(ref) - L + 1))
            if found:
                best = L
            else:
                break   # a longer window containing a missing suffix cannot match
        out.append(best)
    return out


class CopyFeatureTests(unittest.TestCase):
    def test_hand_examples(self):
        self.assertEqual(copy_run_lengths([9, 2, 3, 4, 7], [1, 2, 3, 4, 5]), [0, 1, 2, 3, 0])
        self.assertEqual(copy_run_lengths([1, 2, 3], [1, 2, 1, 2, 3]), [1, 2, 3])
        self.assertEqual(copy_run_lengths([], [1, 2]), [])
        self.assertEqual(copy_run_lengths([1, 2], []), [0, 0])

    def test_run_resets_after_novel_token(self):
        # "2 3" copied, then a novel token, then "2" again: run restarts at 1
        self.assertEqual(copy_run_lengths([2, 3, 99, 2], [1, 2, 3, 4]), [1, 2, 0, 1])

    def test_matches_brute_force_on_random_sequences(self):
        rng = random.Random(0)
        for _ in range(300):
            vocab = rng.randint(2, 6)
            ref = [rng.randrange(vocab) for _ in range(rng.randint(0, 25))]
            resp = [rng.randrange(vocab + 1) for _ in range(rng.randint(0, 20))]
            self.assertEqual(copy_run_lengths(resp, ref), naive_runs(resp, ref), (resp, ref))

    def test_status_thresholds(self):
        self.assertEqual(copy_status(0), STATUS_NOVEL)
        self.assertEqual(copy_status(1), STATUS_SHORT)
        self.assertEqual(copy_status(2), STATUS_SHORT)
        self.assertEqual(copy_status(3), STATUS_COPY)
        self.assertEqual(copy_status(2, min_run=2), STATUS_COPY)

    def test_token_types(self):
        cases = {" the": "function", "The": "function", " Paris": "capitalized", " 1990": "numeric",
                 " $5": "numeric", ".": "punct", " (": "punct", "\n\n": "space", " running": "content",
                 "COVID": "capitalized", " a": "function"}
        for text, want in cases.items():
            self.assertEqual(token_type(text), want, text)


if __name__ == "__main__":
    unittest.main()
