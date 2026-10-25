"""Tests for halluc.judging (no model dependencies).  python tests/test_judging.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from halluc import judging as J  # noqa: E402


def test_kappa_perfect_and_chance():
    a = [True, False, True, False]
    assert abs(J.cohen_kappa(a, a) - 1.0) < 1e-12
    assert abs(J.cohen_kappa(a, [not x for x in a]) + 1.0) < 1e-12
    # independent labels -> kappa 0
    x = [True, True, False, False]
    y = [True, False, True, False]
    assert abs(J.cohen_kappa(x, y)) < 1e-12
    assert math.isnan(J.cohen_kappa([True, True], [True, True]))   # no variation: undefined


def test_agreement_counts_and_nones():
    a = [True, True, False, False, None]
    b = [True, False, False, True, True]
    r = J.agreement(a, b)
    assert r["n"] == 4 and r["both_hallucinated"] == 1 and r["both_clean"] == 1
    assert r["only_first"] == 1 and r["only_second"] == 1 and abs(r["agreement"] - 0.5) < 1e-12
    assert J.agreement([None], [True]) == {"n": 0}


def test_consensus():
    got = J.consensus({"a": [True, True, False, None], "b": [True, False, False, True], "c": [False, True, False, None]})
    assert got == [True, True, False, True]
    two = J.consensus({"a": [True, False], "b": [False, False]})
    assert two == [None, False]                                      # two judges disagree -> undecided
    assert J.consensus({"a": [None], "b": [None]}) == [None]


def test_spans_to_tuples():
    spans = [{"start": 5, "end": 9}, {"start": -2, "end": 3}, {"start": 8, "end": 100}, {"start": 4, "end": 4}]
    assert J.spans_to_tuples(spans, 20) == [(0, 3, "lettuce"), (5, 9, "lettuce"), (8, 20, "lettuce")]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  %s" % t.__name__)
    print("%d tests passed" % len(tests))
