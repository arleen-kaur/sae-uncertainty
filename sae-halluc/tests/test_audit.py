import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from halluc import audit  # noqa: E402


class AurocTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(audit.auroc([3, 4], [1, 2]), 1.0)
        self.assertEqual(audit.auroc([1, 2], [3, 4]), 0.0)
        self.assertEqual(audit.auroc([1, 1], [1, 1]), 0.5)
        self.assertAlmostEqual(audit.auroc([1, 3], [2, 2]), 0.5)

    def test_matches_sklearn_if_present(self):
        if audit.roc_auc_score is None:
            self.skipTest("no sklearn")
        rng = np.random.RandomState(0)
        a, b = rng.randint(0, 6, 200), rng.randint(0, 6, 150)   # many ties
        y = np.r_[np.ones(200), np.zeros(150)]
        self.assertAlmostEqual(audit.auroc(a, b), audit.roc_auc_score(y, np.r_[a, b]), places=9)


def synthetic(n_pairs=240, signal=True, seed=0):
    """Hall tokens carry a planted SAE feature (idx 7) that entropy / nuisance cannot explain."""
    rng = np.random.RandomState(seed)
    records, rows = [], []
    for p in range(n_pairs):
        sid = "q%d" % (p // 4)
        ctx = int(rng.randint(0, 3))
        typ = ["content", "numeric", "function"][int(rng.randint(0, 3))]
        rel = float(rng.rand())
        for role in ("hall", "faithful"):
            r = {"source_id": sid, "pair_id": "p%d" % p, "tier": 0, "role": role,
                 "correct": role == "faithful", "entropy": float(rng.rand() * 3),
                 "surprisal": float(rng.rand() * 5), "type": typ, "ctx_status": ctx,
                 "rel_pos": rel, "ctx_run": ctx * 3, "passage_run": ctx * 3}
            records.append(r)
            idx = list(rng.choice(np.arange(20, 200), 8, replace=False))
            if signal and role == "hall":
                idx.append(7)
            idx = np.array(sorted(idx), dtype=np.int64)
            rows.append({"sae_features": {1: {"indices": idx, "values": np.ones(len(idx), np.float32) * 2}}})
    return records, {"predict": {"per_question_data": rows}}


class PredictiveTests(unittest.TestCase):
    def setUp(self):
        if audit.LogisticRegression is None or audit._sp is None:
            self.skipTest("needs sklearn + scipy")

    def test_planted_feature_found_and_baselines_at_chance(self):
        records, pay = synthetic(signal=True)
        out = audit.predictive_report(records, pay, [1])
        self.assertGreater(out["sae_predict_L1"], 0.95)
        for k in ("nuisance_only", "entropy_only", "nuisance+entropy"):
            self.assertLess(abs(out[k] - 0.5), 0.12, k)
        self.assertGreater(out["sae+baselines_predict_L1"], 0.95)

    def test_no_signal_gives_chance(self):
        records, pay = synthetic(signal=False, seed=3)
        out = audit.predictive_report(records, pay, [1])
        self.assertLess(abs(out["sae_predict_L1"] - 0.5), 0.12)


class ReportTests(unittest.TestCase):
    def test_paired_and_quadrants(self):
        records, _ = synthetic(signal=False)
        for r in records:
            if r["role"] == "hall":
                r["entropy"] += 1.0                      # plant an entropy shift
        rep = audit.paired_report(records)
        self.assertEqual(rep["n_pairs"], 240)
        self.assertGreater(rep["entropy_frac_hall_higher"], 0.6)
        q = audit.quadrant_sizes(records)
        self.assertEqual(q["A_correct_confident"] + q["B_incorrect_confident"] +
                         q["C_correct_uncertain"] + q["D_incorrect_uncertain"],
                         int((np.array([r["entropy"] for r in records]) <= q["entropy_cutoffs"][0]).sum()
                             + (np.array([r["entropy"] for r in records]) >= q["entropy_cutoffs"][1]).sum()))
        self.assertGreater(q["D_incorrect_uncertain"], q["B_incorrect_confident"])

    def test_population_report(self):
        from halluc.extract import LABEL_CODES, TYPE_CODES
        rng = np.random.RandomState(0)
        n = 4000
        label = rng.choice([LABEL_CODES["hall"], LABEL_CODES["faithful"]], n)
        ent = rng.rand(n) + (label == LABEL_CODES["hall"]) * 0.5
        pop = [{"label": label.astype(np.int8), "type": np.full(n, TYPE_CODES["content"], np.int8),
                "ctx_status": np.zeros(n, np.int8), "entropy": ent.astype(np.float32),
                "surprisal": rng.rand(n).astype(np.float32)}]
        rep = audit.population_report(pop)
        self.assertGreater(rep["entropy_auroc_all"], 0.7)
        self.assertLess(abs(rep["surprisal_auroc_all"] - 0.5), 0.05)


if __name__ == "__main__":
    unittest.main()
