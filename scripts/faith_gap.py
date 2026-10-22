"""
Faithfulness-gap discovery + intervention on FaithEval-Counterfactual (Het-style, per-dataset).

Question: can SAE features found on THIS benchmark be turned up/down to make the model follow the
context more (or its memory less)?  Het's tooling is reused (extract-style residual capture,
SAESuppressionHook with `scale`); the labelling and outcome differ from his quadrant pipeline because
FaithEval has too few confident-incorrect items (~20-30) for his discovery:

  label    : faithful = model answers the CONTEXT option;  memory = model answers the real ARC option.
  discovery: per-feature Mann-Whitney faithful vs memory on the DISCOVERY split (answer position,
             one token per item), BH-FDR per layer; features with q<=0.05 ranked by |AUC-0.5|.
  sets     : faith_set = top-n features higher on faithful items;  mem_set = top-n higher on memory
             items (n = min(K, sizes)).
  outcome  : continuous margin m = logit[context letter] - logit[memory letter] at the answer position
             (far more powerful than 0/1 accuracy), plus faithful-rate and flips, on the held-out
             VALIDATION split.  d = m(intervention) - m(baseline), per item, paired.
  null     : random feature sets matched to the selected features on firing frequency (suppress and
             amplify, `--n-random` draws each).

PRE-SPECIFIED success criterion (fixed before any validation run):
  Primary hypotheses, per layer:  H1 amplify faith_set  -> mean d > 0
                                  H2 suppress mem_set   -> mean d > 0
  A primary hypothesis PASSES at a layer iff ALL of
      (a) z vs the matching random null >= z_crit, z_crit = one-sided Bonferroni over
          (#layers x 2 hypotheses) at alpha=0.05,
      (b) the observed d is beyond EVERY random draw (in the hypothesised direction),
      (c) the item-bootstrap 95% CI of d excludes 0.
  Secondary (reported, not part of the criterion): the opposite-direction controls (suppress faith_set,
  amplify mem_set; expected d < 0) and, with --external-json, the RAGTruth-derived passage features.
  Results are reported whether or not anything passes.

Usage (GPU machine; layers can be split across GPUs with separate --layers and a shared --output-dir):
    CUDA_VISIBLE_DEVICES=0 python scripts/faith_gap.py --layers 8,12,16 --output-dir results/faith_gap/
    CUDA_VISIBLE_DEVICES=1 python scripts/faith_gap.py --layers 20,24,28 --output-dir results/faith_gap/
    python scripts/faith_gap.py --summarize-only --output-dir results/faith_gap/ --layers 8,12,16,20,24,28
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LETTERS = ["A", "B", "C", "D"]
MIN_NULL_DRAWS = 20   # fewer random draws than this cannot support a verdict

# hypothesis name -> (expected sign of d, which random null it is compared with, primary?)
HYPOTHESES = {
    "faith_amplify": (+1, "amplify", True),
    "mem_suppress": (+1, "suppress", True),
    "faith_suppress": (-1, "suppress", False),
    "mem_amplify": (-1, "amplify", False),
    # external (RAGTruth-derived) set has its own size, hence its own size-matched random null
    "ext_amplify": (+1, "ext_amplify", False),
    "ext_suppress": (-1, "ext_suppress", False),
}


# ----------------------------------------------------------------------------- pure statistics
def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_isf(p):
    """z such that P(Z > z) = p (bisection; no scipy needed)."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_sf(mid) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rank_avg(x):
    """1-based ranks with ties averaged."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    start = np.flatnonzero(np.concatenate(([True], xs[1:] != xs[:-1])))
    end = np.append(start[1:], n)
    avg = (start + 1 + end) / 2.0
    ranks = np.empty(n)
    ranks[order] = np.repeat(avg, end - start)
    return ranks


def mannwhitney(x, y):
    """(AUC = P(x > y) + 0.5 P(x == y), two-sided normal-approx p with tie correction)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    allv = np.concatenate([x, y])
    N = n1 + n2
    r = rank_avg(allv)
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    auc = u1 / (n1 * n2)
    _, counts = np.unique(allv, return_counts=True)
    tie = float((counts ** 3 - counts).sum())
    var = n1 * n2 / 12.0 * ((N + 1) - tie / (N * (N - 1.0)))
    if var <= 0:
        return float(auc), 1.0
    z = (u1 - n1 * n2 / 2.0) / math.sqrt(var)
    return float(auc), float(math.erfc(abs(z) / math.sqrt(2.0)))


def bh_qvalues(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    q = np.empty(n)
    if n == 0:
        return q
    order = np.argsort(p)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def label_item(pred, context_letter, parametric_letter):
    if pred == context_letter:
        return "faithful"
    if pred == parametric_letter:
        return "memory"
    return "other"


def discover(X, labels, min_freq=0.05):
    """Per-feature faithful-vs-memory test. X: (n_items, n_features). Returns list of row dicts (with q)."""
    lab = np.asarray(labels)
    fi = np.flatnonzero(lab == "faithful")
    mi = np.flatnonzero(lab == "memory")
    Xf, Xm = X[fi], X[mi]
    ff = (Xf > 0).mean(0)
    fm = (Xm > 0).mean(0)
    cand = np.flatnonzero((ff >= min_freq) | (fm >= min_freq))
    rows = []
    for j in cand:
        auc, p = mannwhitney(Xf[:, j], Xm[:, j])
        rows.append({"feature": int(j), "auc_faith_gt_mem": auc, "p": p,
                     "freq_faithful": float(ff[j]), "freq_memory": float(fm[j]),
                     "mean_faithful": float(Xf[:, j].mean()), "mean_memory": float(Xm[:, j].mean())})
    for r, q in zip(rows, bh_qvalues([r["p"] for r in rows])):
        r["q"] = float(q)
    return rows


def select_sets(rows, k, q_max=0.05):
    """faith_set / mem_set: significant features ranked by |AUC-0.5|, truncated to a common size n."""
    sig = [r for r in rows if r["q"] <= q_max]
    faith = sorted([r for r in sig if r["auc_faith_gt_mem"] > 0.5], key=lambda r: -r["auc_faith_gt_mem"])
    mem = sorted([r for r in sig if r["auc_faith_gt_mem"] < 0.5], key=lambda r: r["auc_faith_gt_mem"])
    n = min(k, len(faith), len(mem))
    return [r["feature"] for r in faith[:n]], [r["feature"] for r in mem[:n]], faith, mem


def matched_random(target, pool, freq, rng, window=30):
    """For every target feature pick a distinct random pool feature among the `window` nearest in firing frequency."""
    ranked = sorted(pool, key=lambda f: freq[f])
    rf = np.array([freq[f] for f in ranked])
    used, chosen = set(), []
    for t in target:
        i = int(np.argmin(np.abs(rf - freq[t])))
        cands = [f for f in ranked[max(0, i - window // 2): i + window // 2 + 1] if f not in used]
        if not cands:
            cands = [f for f in ranked if f not in used]
        f = rng.choice(cands)
        used.add(f)
        chosen.append(f)
    return sorted(chosen)


def random_sets(target_union, n, pool, freq, rng, n_draws):
    return [matched_random(rng.sample(target_union, n), pool, freq, rng) for _ in range(n_draws)]


def bootstrap_ci(x, B=2000, seed=0):
    x = np.asarray(x, dtype=float)
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare_to_null(item_deltas, null_means, sign):
    obs = float(np.mean(item_deltas))
    null = np.asarray(null_means, dtype=float)
    sd = float(null.std())
    z = (obs - null.mean()) / sd if sd > 0 else float("nan")
    z_dir = sign * z
    p_emp = (1.0 + np.sum(sign * (null - obs) >= 0)) / (1.0 + len(null))
    beyond = bool(np.all(sign * (null - obs) < 0))
    lo, hi = bootstrap_ci(item_deltas)
    ci_ok = (lo > 0) if sign > 0 else (hi < 0)
    return {"d": obs, "ci": [lo, hi], "null_mean": float(null.mean()), "null_sd": sd,
            "z_dir": float(z_dir), "p_emp": float(p_emp), "beyond_all": beyond, "ci_excludes_0": bool(ci_ok)}


def z_critical(n_layers, n_primary=2, alpha=0.05):
    return norm_isf(alpha / float(max(1, n_layers) * n_primary))


def by_letter(item_deltas, letters):
    """Mean d split by a per-item letter (e.g. the context letter). A real faithfulness effect should have
    the hypothesised sign for every letter; a letter-identity artefact flips sign across letters."""
    d = np.asarray(item_deltas, dtype=float)
    lt = np.asarray(letters)
    return {l: (float(d[lt == l].mean()), int((lt == l).sum())) for l in sorted(set(lt.tolist()))}


# ----------------------------------------------------------------------------- summary
def summarize_dir(output_dir, layers=None):
    out_dir = Path(output_dir)
    files = sorted(out_dir.glob("layer*.json"), key=lambda p: int(p.stem.replace("layer", "")))
    if layers is not None:
        files = [f for f in files if int(f.stem.replace("layer", "")) in layers]
    if not files:
        print("no layer*.json found in %s" % out_dir)
        return
    data = [json.load(open(f)) for f in files]
    n_layers = len(layers) if layers is not None else len(data)
    zc = z_critical(n_layers)
    print("\n=== faithfulness-gap interventions (validation) | layers run: %d | z_crit (Bonferroni over %d layers x 2) = %.2f ==="
          % (len(data), n_layers, zc))
    n_pass = 0
    n_primary = 0
    for L in data:
        print("\n-- layer %d | disc groups %s | val baseline faithful %.3f memory %.3f | set size %d | identity max|d| %.2g"
              % (L["layer"], L["disc_groups"], L["val_baseline"]["faithful_rate"],
                 L["val_baseline"]["memory_rate"], L["n_use"], L["identity_max_abs"]))
        if L["n_use"] == 0:
            print("   (no usable feature sets at this layer)")
            continue
        print("   %-15s %8s %20s %7s %7s %7s %s" % ("hypothesis", "d", "95% CI", "z_dir", "p_emp", "beyond", "verdict"))
        for name, (sign, kind, primary) in HYPOTHESES.items():
            if name not in L["named"]:
                continue
            nm = L["named"][name]
            if not L.get("null_" + kind):
                print("   %-15s (no size-matched null saved for this run; skipped)" % name)
                continue
            r = compare_to_null(nm["delta_margin"], L["null_" + kind], sign)
            ok = r["z_dir"] >= zc and r["beyond_all"] and r["ci_excludes_0"]
            enough = len(L["null_" + kind]) >= MIN_NULL_DRAWS
            tag = ""
            if primary:
                if enough:
                    n_primary += 1
                    n_pass += int(ok)
                    tag = "PASS" if ok else "fail"
                else:
                    tag = "n/a (null < %d draws: not a valid test)" % MIN_NULL_DRAWS
            print("   %-15s %+8.4f [%+8.4f,%+8.4f] %+7.2f %7.3f %7s %s   (faithful-rate %+.3f, flips +%d/-%d)"
                  % (name, r["d"], r["ci"][0], r["ci"][1], r["z_dir"], r["p_emp"], r["beyond_all"], tag,
                     nm["faithful_rate_delta"], nm["flips_to_faithful"], nm["flips_from_faithful"]))
            if L.get("val_context_letter") and name in ("faith_amplify", "mem_suppress", "ext_amplify"):
                bl = by_letter(nm["delta_margin"], L["val_context_letter"])
                print("   %15s   d by context letter: %s" % (
                    "", "  ".join("%s %+.3f (n=%d)" % (l, v[0], v[1]) for l, v in bl.items())))
        if L.get("disc_pred_by_group"):
            print("   discovery predicted-letter distribution by group: %s" % L["disc_pred_by_group"])
    print("\nPRIMARY hypotheses passing the pre-specified criterion: %d / %d" % (n_pass, n_primary))


# ----------------------------------------------------------------------------- GPU part
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--discovery-set", default="data/eval_sets/eval_set_mcq_faitheval_cf_ctx_995_discovery.json")
    ap.add_argument("--validation-set", default="data/eval_sets/eval_set_mcq_faitheval_cf_ctx_995_validation.json")
    ap.add_argument("--layers", default="8,12,16,20,24,28")
    ap.add_argument("--k", type=int, default=20, help="max features per set")
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--amplify", type=float, default=2.0)
    ap.add_argument("--min-group", type=int, default=30, help="min items in faithful/memory group to proceed")
    ap.add_argument("--min-freq", type=float, default=0.05,
                    help="a feature is tested only if active on >= this fraction of items in the faithful or memory group")
    ap.add_argument("--external-json", default=None,
                    help="intervention JSON from sae-halluc (has pass_set + config.layer): adds ext_* conditions")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize_dir(out_dir, layers)
        return

    todo = [L for L in layers if not (out_dir / ("layer%d.json" % L)).exists()]
    if not todo:
        print("all requested layers already done; use --summarize-only")
        summarize_dir(out_dir, layers)
        return

    import torch
    from src.model import load_model, get_letter_token_ids
    from src.sae import ResidualStreamHook, SAESuppressionHook, load_sae

    ext = None
    if args.external_json:
        e = json.load(open(args.external_json))
        ext = {"layer": int(e["config"]["layer"]), "features": [int(f) for f in e["pass_set"]]}
        print("external set: layer %d, %d features" % (ext["layer"], len(ext["features"])))

    disc = json.load(open(args.discovery_set))["samples"]
    val = json.load(open(args.validation_set))["samples"]
    print("discovery items %d | validation items %d | layers to run %s" % (len(disc), len(val), todo))

    model, tok, cfg = load_model(args.model, device=args.device)
    ltid = get_letter_token_ids(tok, LETTERS)
    letter_ids = [ltid[l] for l in LETTERS]
    li = {l: i for i, l in enumerate(LETTERS)}

    def enc(sample):
        return tok(sample.get("mcq_prompt", sample.get("prompt", "")), return_tensors="pt").to(args.device)

    def forward(inputs):
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        ll = out.logits[0, -1, letter_ids].float().cpu().numpy()
        return ll

    def margin_pred(ll, s):
        return float(ll[li[s["context_letter"]]] - ll[li[s["parametric_letter"]]]), LETTERS[int(np.argmax(ll))]

    # ---- discovery pass: baseline prediction + last-token residual at all requested layers
    hook = ResidualStreamHook()
    hook.register(model, cfg, todo)
    resid = {L: [] for L in todo}
    labels, disc_preds = [], []
    for s in disc:
        ll = forward(enc(s))
        _, pred = margin_pred(ll, s)
        labels.append(label_item(pred, s["context_letter"], s["parametric_letter"]))
        disc_preds.append(pred)
        for L in todo:
            resid[L].append(hook.hidden_states[L].float().cpu())
    hook.remove_all()
    groups = {k: labels.count(k) for k in ("faithful", "memory", "other")}
    disc_pred_by_group = {g: {l: sum(1 for lb, p in zip(labels, disc_preds) if lb == g and p == l) for l in LETTERS}
                          for g in ("faithful", "memory")}
    print("discovery groups:", groups)
    print("discovery predicted-letter distribution by group:", disc_pred_by_group)
    if min(groups["faithful"], groups["memory"]) < args.min_group:
        raise SystemExit("faithful/memory group smaller than --min-group (%d); aborting" % args.min_group)

    # ---- validation baseline (no hook)
    val_inputs = [enc(s) for s in val]
    base_margin, base_pred = [], []
    for s, inp in zip(val, val_inputs):
        m, p = margin_pred(forward(inp), s)
        base_margin.append(m)
        base_pred.append(p)
    base_margin = np.array(base_margin)
    base_faithful = np.array([p == s["context_letter"] for p, s in zip(base_pred, val)])
    base_memory = np.array([p == s["parametric_letter"] for p, s in zip(base_pred, val)])
    print("validation baseline: faithful %.3f, memory %.3f" % (base_faithful.mean(), base_memory.mean()))

    for L in todo:
        print("\n===== layer %d =====" % L)
        sae = load_sae(cfg, L, device=args.device)
        acts = torch.cat(resid[L], 0)
        X = []
        for i in range(0, len(acts), 64):
            with torch.no_grad():
                X.append(sae.encode(acts[i:i + 64].to(args.device)).float().cpu().numpy())
        X = np.concatenate(X, 0)
        rows = discover(X, labels, min_freq=args.min_freq)
        faith_set, mem_set, faith_all, mem_all = select_sets(rows, args.k)
        n_use = len(faith_set)
        print("tested %d features | significant faith-assoc %d, mem-assoc %d | using n=%d each"
              % (len(rows), len(faith_all), len(mem_all), n_use))
        result = {"layer": L, "n_use": n_use, "disc_groups": groups, "n_val": len(val),
                  "val_baseline": {"faithful_rate": float(base_faithful.mean()), "memory_rate": float(base_memory.mean())},
                  "faith_set": faith_set, "mem_set": mem_set, "n_tested": len(rows),
                  "top_faith": faith_all[:30], "top_mem": mem_all[:30],
                  "config": vars(args), "named": {}, "null_suppress": [], "null_amplify": [],
                  "null_ext_suppress": [], "null_ext_amplify": [],
                  "val_context_letter": [s["context_letter"] for s in val],
                  "val_memory_letter": [s["parametric_letter"] for s in val],
                  "val_base_pred": list(base_pred), "disc_pred_by_group": disc_pred_by_group,
                  "identity_max_abs": 0.0}
        if n_use < 3:
            print("fewer than 3 usable features per direction; skipping interventions at this layer")
            result["n_use"] = 0
            json.dump(result, open(out_dir / ("layer%d.json" % L), "w"), indent=1)
            del sae
            torch.cuda.empty_cache()
            continue

        freq = (X > 0).mean(0)
        chosen_sets = set(faith_set) | set(mem_set)
        pool = [int(f) for f in np.flatnonzero(freq >= 0.02) if int(f) not in chosen_sets]
        rng = random.Random(args.seed + L)
        rsets = random_sets(faith_set + mem_set, n_use, pool, freq, rng, args.n_random)
        result["random_sets"] = rsets
        ext_here = ext is not None and ext["layer"] == L
        ext_rsets = []
        if ext_here:
            # size-matched null for the external set (it has len(ext) features, not n_use)
            ext_pool = [f for f in pool if f not in set(ext["features"])]
            ext_rsets = [matched_random(ext["features"], ext_pool, freq, rng) for _ in range(args.n_random)]
            result["ext_random_sets"] = ext_rsets
        del X, acts

        conds = {"identity": (faith_set, 1.0),
                 "faith_amplify": (faith_set, args.amplify), "faith_suppress": (faith_set, 0.0),
                 "mem_amplify": (mem_set, args.amplify), "mem_suppress": (mem_set, 0.0)}
        if ext_here:
            conds["ext_amplify"] = (ext["features"], args.amplify)
            conds["ext_suppress"] = (ext["features"], 0.0)
        named = list(conds)
        for i, r in enumerate(rsets):
            conds["rand%03d_suppress" % i] = (r, 0.0)
            conds["rand%03d_amplify" % i] = (r, args.amplify)
        for i, r in enumerate(ext_rsets):
            conds["extrand%03d_suppress" % i] = (r, 0.0)
            conds["extrand%03d_amplify" % i] = (r, args.amplify)
        print("validation forwards: %d items x %d conditions = %d" % (len(val), len(conds), len(val) * len(conds)))

        shook = SAESuppressionHook(sae, [], 1.0)
        shook.register(model, cfg, L)
        dm = {c: np.zeros(len(val)) for c in conds}
        preds = {c: [] for c in named}
        for n_done, (s, inp) in enumerate(zip(val, val_inputs)):
            for c, (fe, sc) in conds.items():
                shook.feature_indices = torch.tensor(fe, dtype=torch.long)
                shook.scale = sc
                m, p = margin_pred(forward(inp), s)
                dm[c][n_done] = m - base_margin[n_done]
                if c in preds:
                    preds[c].append(p)
            if (n_done + 1) % 50 == 0:
                print("  layer %d: %d/%d validation items" % (L, n_done + 1, len(val)))
        shook.remove()

        result["identity_max_abs"] = float(np.abs(dm["identity"]).max())
        for c in named:
            if c == "identity":
                continue
            pf = np.array([p == s["context_letter"] for p, s in zip(preds[c], val)])
            result["named"][c] = {
                "delta_margin": dm[c].tolist(),
                "faithful_rate_delta": float(pf.mean() - base_faithful.mean()),
                "flips_to_faithful": int((~base_faithful & pf).sum()),
                "flips_from_faithful": int((base_faithful & ~pf).sum()),
            }
        result["null_suppress"] = [float(dm["rand%03d_suppress" % i].mean()) for i in range(len(rsets))]
        result["null_amplify"] = [float(dm["rand%03d_amplify" % i].mean()) for i in range(len(rsets))]
        result["null_ext_suppress"] = [float(dm["extrand%03d_suppress" % i].mean()) for i in range(len(ext_rsets))]
        result["null_ext_amplify"] = [float(dm["extrand%03d_amplify" % i].mean()) for i in range(len(ext_rsets))]
        json.dump(result, open(out_dir / ("layer%d.json" % L), "w"), indent=1)
        print("saved layer%d.json (identity max|d| = %.2g)" % (L, result["identity_max_abs"]))
        del sae
        torch.cuda.empty_cache()

    summarize_dir(out_dir, layers)


if __name__ == "__main__":
    main()
