"""Compare named-set effects with their random (activity-matched) draws, across runs.

usage: python scripts/summarize_intervention.py results/intervene_L16_rag_matched.json \
           results/intervene_L16_rag_noboiler.json results/intervene_L16_norag_matched.json
For each named set: observed d, z vs the matching random draws, and the empirical
one-sided p = (1 + #random draws at least as extreme in the hypothesised direction) / (1 + N).
Hypothesised direction: suppress passage-dependent -> d > 0; amplify passage-dependent -> d < 0;
suppress hall -> d < 0; amplify hall -> d > 0.
"""
import json
import sys

import numpy as np

EXPECT = {"hall_suppress": -1, "hall_amplify": +1, "pass_suppress": +1, "pass_amplify": -1}

for path in sys.argv[1:]:
    c = json.load(open(path))["conditions"]
    print("\n== %s" % path)
    print("%-14s %9s %19s %7s %7s %6s" % ("set", "d", "95% CI", "z", "p_emp", "N"))
    for name, sign in EXPECT.items():
        if name not in c:
            continue
        kind = name.split("_")[1]
        rs = np.array([v["mean_d"] for k, v in c.items() if k.startswith("rand") and k.endswith("_" + kind)])
        d = c[name]["mean_d"]
        lo, hi = c[name]["ci95"]
        z = (d - rs.mean()) / rs.std() if rs.std() > 0 else float("nan")
        p = (1 + np.sum(sign * (rs - d) >= 0)) / (1.0 + len(rs))
        print("%-14s %+9.4f [%+.4f, %+.4f] %+7.2f %7.3f %6d" % (name, d, lo, hi, z, p, len(rs)))
