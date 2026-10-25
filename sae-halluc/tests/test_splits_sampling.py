import collections
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from halluc.splits import split_of, DISCOVERY, VALIDATION, PILOT  # noqa: E402
from halluc.sampling import sample_matched_tokens  # noqa: E402


def tok(seq, label, typ="content", ctx=0, rel=0.5, span=-1):
    return {"seq_idx": seq, "predict_idx": seq - 1, "label": label, "type": typ,
            "ctx_status": ctx, "rel_pos": rel, "span_idx": span}


class SplitTests(unittest.TestCase):
    def test_official_test_questions_are_pilot(self):
        self.assertEqual(split_of("123", "test"), PILOT)

    def test_deterministic_and_roughly_60_40(self):
        ids = [str(i) for i in range(5000)]
        a = [split_of(i, "train") for i in ids]
        b = [split_of(i, "train") for i in ids]
        self.assertEqual(a, b)
        c = collections.Counter(a)
        self.assertEqual(set(c), {DISCOVERY, VALIDATION})
        self.assertAlmostEqual(c[DISCOVERY] / len(ids), 0.6, delta=0.03)

    def test_seed_changes_assignment(self):
        ids = [str(i) for i in range(500)]
        a = [split_of(i, "train", seed=1) for i in ids]
        b = [split_of(i, "train", seed=2) for i in ids]
        self.assertNotEqual(a, b)


class SamplingTests(unittest.TestCase):
    def make_tokens(self):
        toks, s = [], 10
        # one hallucinated span of 12 content tokens with ctx=0, and one of 3 numerics
        for _ in range(12):
            toks.append(tok(s, "hall", "content", 0, 0.7, span=0)); s += 1
        for _ in range(3):
            toks.append(tok(s, "hall", "numeric", 0, 0.8, span=1)); s += 1
        # faithful pool: plenty of matching content/ctx0, a few numerics, some of other kinds
        for _ in range(30):
            toks.append(tok(s, "faithful", "content", 0, 0.7)); s += 1
        for _ in range(5):
            toks.append(tok(s, "faithful", "numeric", 0, 0.1)); s += 1   # wrong pos bin -> tier 1
        for _ in range(10):
            toks.append(tok(s, "faithful", "function", 2, 0.5)); s += 1
        toks.append(tok(s, "ambig", "content", 0, 0.5)); s += 1
        toks.append(tok(s, "near", "content", 0, 0.5))
        return toks

    def test_cap_per_span_and_one_faithful_per_hall_token(self):
        pairs, un = sample_matched_tokens(self.make_tokens(), random.Random(0), max_per_span=8)
        per_span = collections.Counter(p["hall"]["span_idx"] for p in pairs)
        self.assertEqual(per_span[0], 8)
        self.assertEqual(per_span[1], 3)
        self.assertEqual(un, 0)
        used = [p["faithful"]["seq_idx"] for p in pairs]
        self.assertEqual(len(used), len(set(used)))     # no faithful token reused

    def test_only_hall_and_faithful_labels_are_used(self):
        pairs, _ = sample_matched_tokens(self.make_tokens(), random.Random(1))
        for p in pairs:
            self.assertEqual(p["hall"]["label"], "hall")
            self.assertEqual(p["faithful"]["label"], "faithful")

    def test_tiers(self):
        pairs, _ = sample_matched_tokens(self.make_tokens(), random.Random(2))
        for p in pairs:
            h, f = p["hall"], p["faithful"]
            self.assertEqual(h["ctx_status"], f["ctx_status"])      # copy status always matched
            if p["tier"] == 0:
                self.assertEqual((h["type"], int(h["rel_pos"] * 5)), (f["type"], int(f["rel_pos"] * 5)))
            if p["tier"] <= 1:
                self.assertEqual(h["type"], f["type"])
        # numerics sit at rel_pos 0.8 but faithful numerics are at 0.1 -> tier 1
        self.assertTrue(all(p["tier"] == 1 for p in pairs if p["hall"]["type"] == "numeric"))
        self.assertTrue(all(p["tier"] == 0 for p in pairs if p["hall"]["type"] == "content"))

    def test_unmatched_when_pool_exhausted(self):
        toks = [tok(10 + i, "hall", "content", 0, 0.5, span=0) for i in range(5)]
        toks += [tok(100 + i, "faithful", "content", 0, 0.5) for i in range(2)]
        pairs, un = sample_matched_tokens(toks, random.Random(0))
        self.assertEqual(len(pairs), 2)
        self.assertEqual(un, 3)

    def test_copy_status_is_never_crossed(self):
        toks = [tok(10, "hall", "content", 0, 0.5, span=0), tok(100, "faithful", "content", 2, 0.5)]
        pairs, un = sample_matched_tokens(toks, random.Random(0))
        self.assertEqual((len(pairs), un), (0, 1))

    def test_deterministic_given_seed(self):
        a, _ = sample_matched_tokens(self.make_tokens(), random.Random(7))
        b, _ = sample_matched_tokens(self.make_tokens(), random.Random(7))
        self.assertEqual([(p["hall"]["seq_idx"], p["faithful"]["seq_idx"]) for p in a],
                         [(p["hall"]["seq_idx"], p["faithful"]["seq_idx"]) for p in b])

    def test_first_response_token_is_allowed_predict_idx_zero_excluded(self):
        # a token with predict_idx < 0 cannot be sampled (no predicting state exists)
        toks = [tok(0, "hall", "content", 0, 0.5, span=0), tok(5, "faithful", "content", 0, 0.5)]
        pairs, un = sample_matched_tokens(toks, random.Random(0))
        self.assertEqual(len(pairs), 0)


if __name__ == "__main__":
    unittest.main()
