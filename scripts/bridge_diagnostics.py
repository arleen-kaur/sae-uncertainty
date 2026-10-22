"""
Smoke-test / gate diagnostics for the RAGTruth bridge eval set.

Reads the pickle produced by experiments/extract_features.py plus the eval-set JSON it
was run on, and reports everything needed to decide whether the downstream quadrant /
screen / suppress steps are meaningful:

  1. Baseline accuracy (with 95% Wilson CI) vs the 50% chance level   <- the main gate
  2. Accuracy by source model, by correct letter, by response-length bin
  3. Letter bias (predicted-letter distribution, p(A) distribution)
  4. Entropy behaviour (binary entropy is a monotone function of |p(A) - 0.5|),
     correlation of entropy with response length
  5. The four quadrant bucket sizes under the same rule as scripts/analyze_quadrant.py
     (confident = bottom P% entropy, uncertain = top P%) and which comparisons would
     be underpowered

Usage:
    python scripts/bridge_diagnostics.py \
        --pickle results/ragtruth_smoke/sae_uncertainty_Llama-3.1-8B.pkl \
        --eval-set data/eval_sets/eval_set_mcq_ragtruth_strict_2638_discovery.json \
        --entropy-percentile 25
"""

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict

import numpy as np

try:
    from scipy import stats
except ImportError:  # bolt has no scipy; frappe's venv does
    stats = None

MIN_BUCKET_WARN = 100   # below this a bucket gives weak Mann-Whitney power
MIN_BUCKET_FAIL = 30    # below this the comparison is not usable


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def _avg_ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rho and p-value (scipy if available, else numpy rho with a t-approx p)."""
    if stats is not None:
        rho, p = stats.spearmanr(x, y)
        return float(rho), float(p)
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    if abs(rho) >= 1 or n < 3:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    # two-sided normal approximation to the t distribution (n is large here)
    p = math.erfc(abs(t) / math.sqrt(2))
    return rho, p


def fmt_acc(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:5.1f}%  (n={n}, 95% CI {100 * lo:.1f}-{100 * hi:.1f})"


def main():
    ap = argparse.ArgumentParser(description="Bridge smoke-test diagnostics")
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--entropy-percentile", type=float, default=25.0)
    ap.add_argument("--gate-acc", type=float, default=0.60,
                    help="accuracy needed for PASS (default 0.60)")
    ap.add_argument("--json-out", default=None, help="optional path to save the numbers")
    args = ap.parse_args()

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)
    pq = data["per_question_data"]
    with open(args.eval_set) as f:
        samples = json.load(f)["samples"]

    n_pq = len(pq)
    if n_pq > len(samples):
        raise SystemExit(f"pickle has {n_pq} items but eval set only {len(samples)}")
    if n_pq < len(samples):
        print(f"NOTE: pickle covers the first {n_pq} of {len(samples)} eval items "
              f"(smoke test on a subset); aligning by q_idx.")
    samples = samples[:n_pq]
    letters = sorted(samples[0]["options"].keys())

    correct = np.array([bool(d["correct"]) for d in pq])
    ent = np.array([d["entropy"] for d in pq])
    pred = np.array([d["predicted_letter"] for d in pq])
    true_letter = np.array([d["correct_letter"] for d in pq])
    # abcd_probs follows the order of `letters` (dict insertion order in extract_features)
    probs = np.array([d["abcd_probs"] for d in pq])
    max_ent = math.log(len(letters))

    print(f"Items: {n_pq}   options: {letters}   max entropy ln({len(letters)}) = {max_ent:.3f}")
    print(f"Hallucinated share in this subset: "
          f"{100 * np.mean([s['hallucinated'] for s in samples]):.1f}%")

    results = {}
    verdicts = []

    # 1. Overall accuracy ----------------------------------------------------
    k, n = int(correct.sum()), n_pq
    lo, hi = wilson(k, n)
    chance = 1 / len(letters)
    print(f"\n[1] BASELINE ACCURACY: {fmt_acc(k, n)}   chance = {100 * chance:.0f}%")
    results["accuracy"] = k / n
    results["accuracy_ci"] = [lo, hi]
    if lo > 0.55 and k / n >= args.gate_acc:
        verdicts.append(("accuracy", "PASS", f"{100 * k / n:.1f}% (CI lower bound {100 * lo:.1f}% > 55%)"))
    elif lo > chance:
        verdicts.append(("accuracy", "WARN", f"{100 * k / n:.1f}% is above chance but weak; "
                         f"correctness axis will be noisy"))
    else:
        verdicts.append(("accuracy", "FAIL", f"{100 * k / n:.1f}% is not distinguishable from chance"))

    # 2. Breakdowns ----------------------------------------------------------
    print("\n[2] ACCURACY BREAKDOWNS")
    by_model = defaultdict(list)
    for s, c in zip(samples, correct):
        by_model[s["source_model"]].append(c)
    print("  by source model (response author):")
    for m, v in sorted(by_model.items()):
        print(f"    {m:22s} {fmt_acc(int(sum(v)), len(v))}")

    print("  by true label:")
    for lab in (True, False):
        idx = [i for i, s in enumerate(samples) if s["hallucinated"] == lab]
        name = "hallucinated" if lab else "clean"
        if idx:
            print(f"    {name:12s} {fmt_acc(int(correct[idx].sum()), len(idx))}")

    print("  by correct letter:")
    acc_by_letter = {}
    for L in letters:
        idx = np.where(true_letter == L)[0]
        if len(idx):
            acc_by_letter[L] = correct[idx].mean()
            print(f"    correct={L}  {fmt_acc(int(correct[idx].sum()), len(idx))}")

    words = np.array([s["response_words"] for s in samples])
    edges = np.percentile(words, [0, 25, 50, 75, 100])
    print("  by response length (quartiles of words):")
    for a, b in zip(edges[:-1], edges[1:]):
        idx = np.where((words >= a) & (words <= b))[0]
        if len(idx):
            print(f"    {a:5.0f}-{b:5.0f} words  {fmt_acc(int(correct[idx].sum()), len(idx))}")

    # 3. Letter bias ---------------------------------------------------------
    print("\n[3] LETTER BIAS")
    pred_counts = Counter(pred.tolist())
    shares = {L: pred_counts.get(L, 0) / n_pq for L in letters}
    print("  predicted-letter share: " + "  ".join(f"{L}={100 * v:.1f}%" for L, v in shares.items()))
    print(f"  mean p({letters[0]}) = {probs[:, 0].mean():.3f}   "
          f"std = {probs[:, 0].std():.3f}")
    gap = max(acc_by_letter.values()) - min(acc_by_letter.values()) if acc_by_letter else 0
    print(f"  accuracy gap between correct letters: {100 * gap:.1f} points")
    results["pred_letter_share"] = shares
    if max(shares.values()) > 0.80 or gap > 0.25:
        verdicts.append(("letter_bias", "WARN",
                         f"max predicted-letter share {100 * max(shares.values()):.0f}%, "
                         f"accuracy gap {100 * gap:.0f} pts: quadrants may track the letter, not the content"))
    else:
        verdicts.append(("letter_bias", "PASS", "no dominant letter"))

    # 4. Entropy -------------------------------------------------------------
    print("\n[4] ENTROPY")
    print(f"  mean {ent.mean():.4f}  median {np.median(ent):.4f}  max {ent.max():.4f}  "
          f"(upper bound {max_ent:.4f})")
    print(f"  mean entropy correct / incorrect: "
          f"{ent[correct].mean():.4f} / {ent[~correct].mean():.4f}" if (~correct).any() else "  (no incorrect)")
    rho, p_rho = spearman(ent, words)
    print(f"  Spearman(entropy, response words) = {rho:+.3f} (p={p_rho:.3g})")
    results["entropy_length_spearman"] = float(rho)
    if abs(rho) > 0.3:
        verdicts.append(("entropy_length", "WARN", f"|rho|={abs(rho):.2f}: entropy tracks length"))
    else:
        verdicts.append(("entropy_length", "PASS", f"rho={rho:+.2f}"))

    # 5. Quadrants -----------------------------------------------------------
    pct = args.entropy_percentile
    low = np.percentile(ent, pct)
    high = np.percentile(ent, 100 - pct)
    conf = ent <= low
    unc = ent >= high
    A = int((correct & conf).sum())
    B = int((~correct & conf).sum())
    C = int((correct & unc).sum())
    D = int((~correct & unc).sum())
    print(f"\n[5] QUADRANT BUCKETS (entropy percentile {pct:g}: confident <= {low:.4f}, uncertain >= {high:.4f})")
    print(f"  A correct   + confident : {A}")
    print(f"  B incorrect + confident : {B}")
    print(f"  C correct   + uncertain : {C}")
    print(f"  D incorrect + uncertain : {D}")
    print(f"  used {A + B + C + D}/{n_pq} items")
    results["quadrants"] = {"A": A, "B": B, "C": C, "D": D}

    comps = [
        ("pure uncertainty   (A vs C)", A, C),
        ("pure incorrectness (A vs B)", A, B),
        ("interaction        (A vs D)", A, D),
    ]
    worst = "PASS"
    for name, g1, g2 in comps:
        m = min(g1, g2)
        status = "ok" if m >= MIN_BUCKET_WARN else ("UNDERPOWERED" if m >= MIN_BUCKET_FAIL else "UNUSABLE")
        print(f"  {name}: min group {m:4d}  -> {status}")
        if status == "UNUSABLE":
            worst = "FAIL"
        elif status == "UNDERPOWERED" and worst != "FAIL":
            worst = "WARN"
    verdicts.append(("buckets", worst,
                     f"A={A} B={B} C={C} D={D} (warn <{MIN_BUCKET_WARN}, fail <{MIN_BUCKET_FAIL} per group)"))

    # Verdict ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GATE SUMMARY")
    print("=" * 70)
    for name, status, msg in verdicts:
        print(f"  {status:5s}  {name:15s} {msg}")
    overall = ("FAIL" if any(v[1] == "FAIL" for v in verdicts)
               else "WARN" if any(v[1] == "WARN" for v in verdicts) else "PASS")
    print(f"\nOVERALL: {overall}")
    if overall == "FAIL":
        print("  -> Do not run the full loop. Change the prompt, or move to the span-level design.")
    elif overall == "WARN":
        print("  -> Look at the WARN lines before spending GPU time on all layers / the full set.")
    results["verdicts"] = verdicts
    results["overall"] = overall

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Saved {args.json_out}")


if __name__ == "__main__":
    main()
