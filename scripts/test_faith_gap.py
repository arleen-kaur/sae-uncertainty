"""Self-test for the pure-numpy parts of scripts/faith_gap.py (no torch/scipy needed).

    python scripts/test_faith_gap.py
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import faith_gap as fg  # noqa: E402


def test_norm():
    assert abs(fg.norm_isf(0.025) - 1.959964) < 1e-4
    assert abs(fg.norm_sf(0.0) - 0.5) < 1e-12
    assert abs(fg.z_critical(6) - fg.norm_isf(0.05 / 12)) < 1e-9
    assert fg.z_critical(6) > fg.z_critical(1)


def test_rank_ties():
    r = fg.rank_avg([3, 1, 1, 2])
    assert list(r) == [4.0, 1.5, 1.5, 3.0], r


def test_mannwhitney():
    auc, p = fg.mannwhitney([4, 5, 6, 7], [1, 2, 3, 3.5])
    assert auc == 1.0 and p < 0.05
    a2, _ = fg.mannwhitney([1, 2, 3, 3.5], [4, 5, 6, 7])
    assert abs(auc + a2 - 1.0) < 1e-12
    auc0, p0 = fg.mannwhitney([0, 0, 0, 0], [0, 0, 0])
    assert auc0 == 0.5 and p0 == 1.0               # all ties: no signal, no crash
    rs = np.random.RandomState(0)
    x, y = rs.randn(300), rs.randn(300)
    assert fg.mannwhitney(x, y)[1] > 0.01           # null data is not significant


def test_bh():
    q = fg.bh_qvalues([0.001, 0.04, 0.03, 0.5])
    assert all(0 <= v <= 1 for v in q)
    assert q[0] <= q[2] <= q[1] or q[2] <= q[1]
    assert list(fg.bh_qvalues([])) == []


def planted(n_f=60, n_m=60, F=200, seed=0):
    rs = np.random.RandomState(seed)
    n = n_f + n_m
    X = (rs.rand(n, F) < 0.3) * rs.rand(n, F)       # sparse non-negative activations
    labels = ["faithful"] * n_f + ["memory"] * n_m
    X[:n_f, 0] += 2.0                                # feature 0: higher on faithful
    X[:n_f, 1] += 1.5
    X[n_f:, 2] += 2.0                                # feature 2: higher on memory
    X[n_f:, 3] += 1.5
    return X, labels


def test_discover_and_select():
    X, labels = planted()
    rows = fg.discover(X, labels)
    by = {r["feature"]: r for r in rows}
    assert by[0]["auc_faith_gt_mem"] > 0.9 and by[0]["q"] < 0.01
    assert by[2]["auc_faith_gt_mem"] < 0.1 and by[2]["q"] < 0.01
    faith, mem, fa, ma = fg.select_sets(rows, k=5)
    assert faith[0] in (0, 1) and mem[0] in (2, 3)
    assert len(faith) == len(mem) <= 5
    assert 0 in faith and 1 in faith and 2 in mem and 3 in mem
    # a pure-noise dataset yields (almost) nothing significant after FDR
    rs = np.random.RandomState(1)
    Xn = (rs.rand(120, 300) < 0.3) * rs.rand(120, 300)
    rn = fg.discover(Xn, ["faithful"] * 60 + ["memory"] * 60)
    assert sum(r["q"] <= 0.05 for r in rn) <= 3


def test_other_items_ignored():
    X, labels = planted()
    labels = labels + ["other"] * 10
    X = np.vstack([X, np.random.RandomState(5).rand(10, X.shape[1]) * 5])
    rows = fg.discover(X, labels)
    assert {r["feature"] for r in rows} >= {0, 2}


def test_matched_random():
    rng = random.Random(0)
    freq = np.linspace(0.0, 1.0, 500)
    pool = list(range(500))
    target = [100, 250, 400]
    for _ in range(20):
        r = fg.matched_random(target, [f for f in pool if f not in target], freq, rng, window=10)
        assert len(set(r)) == 3
        for t, f in zip(sorted(target), r):
            assert abs(freq[t] - freq[f]) < 0.05, (t, f)
    sets = fg.random_sets(target, 3, pool, freq, rng, 7)
    assert len(sets) == 7 and all(len(s) == 3 for s in sets)


def test_compare_to_null():
    rs = np.random.RandomState(0)
    null = list(rs.randn(40) * 0.01)
    big = rs.randn(300) * 0.05 + 0.2                 # clearly above null, CI excludes 0
    r = fg.compare_to_null(big, null, +1)
    assert r["beyond_all"] and r["ci_excludes_0"] and r["z_dir"] > fg.z_critical(6)
    r2 = fg.compare_to_null(-big, null, +1)          # wrong direction never "passes"
    assert not r2["beyond_all"] and r2["z_dir"] < 0 and not r2["ci_excludes_0"]
    noise = rs.randn(300) * 0.05
    r3 = fg.compare_to_null(noise, null, +1)
    assert not (r3["z_dir"] >= fg.z_critical(6) and r3["beyond_all"] and r3["ci_excludes_0"])
    r4 = fg.compare_to_null(-big, null, -1)          # negative hypothesis, negative effect
    assert r4["beyond_all"] and r4["ci_excludes_0"] and r4["z_dir"] > 3


def test_by_letter():
    r = fg.by_letter([1.0, 3.0, -1.0, 2.0], ["A", "A", "B", "B"])
    assert r == {"A": (2.0, 2), "B": (0.5, 2)}


def test_label_item():
    assert fg.label_item("A", "A", "B") == "faithful"
    assert fg.label_item("B", "A", "B") == "memory"
    assert fg.label_item("C", "A", "B") == "other"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  %s" % t.__name__)
    print("%d tests passed" % len(tests))
