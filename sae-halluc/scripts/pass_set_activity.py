"""CPU-only: how often do the passage-dependent features fire with vs without passages?

If they are (almost) silent in the no_rag condition, suppressing them there can do nothing
by construction, so the A-vs-C drop in pass_suppress is expected and is NOT independent evidence.

usage: python scripts/pass_set_activity.py results/intervene_L16_rag_matched.json \
           results/token_discovery_full_rag/token_features_predict.pkl \
           results/token_discovery_full_no_rag/token_features_predict.pkl
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from intervene_teacher_forced import feature_freq  # noqa: E402

out = json.load(open(sys.argv[1]))
layer = out["config"]["layer"]
sets = {"pass_set": out["pass_set"], "hall_set": out["hall_set"]}
fr = feature_freq(sys.argv[2], layer)
fn = feature_freq(sys.argv[3], layer)
for name, fs in sets.items():
    a = np.array([fr.get(f, 0) for f in fs], dtype=float)
    b = np.array([fn.get(f, 0) for f in fs], dtype=float)
    print("\n%s: active-token counts, rag vs no_rag" % name)
    for f, x, y in zip(fs, a, b):
        print("  f%-6d rag %6d   no_rag %6d   ratio %.2f" % (f, x, y, y / max(x, 1)))
    print("  total rag %d, no_rag %d, ratio %.2f" % (a.sum(), b.sum(), b.sum() / max(a.sum(), 1)))
