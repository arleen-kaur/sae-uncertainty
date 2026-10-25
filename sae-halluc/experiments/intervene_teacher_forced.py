"""Teacher-forced causal test: does suppressing/amplifying a feature set move the model's
probability of hallucinated tokens relative to their matched faithful tokens?

Feature sets come from the DISCOVERY comparison (scripts/compare_conditions.py CSV); the test runs
on held-out VALIDATION responses with the same matched-pair sampling as the extraction.

    hall_set   more active on hallucinated tokens, same effect with and without passages
               (gap not significant)                               -> suppress / amplify
    pass_set   less active on hallucinated tokens, much more so with passages (gap significant)
                                                                   -> amplify / suppress
    random     same-size sets of other tested features, many draws -> null distribution

Metric per pair:  d = [log p(hall token) - log p(faithful token)]_intervened - [...]_baseline
(each log p is under the state at that token's predicting position). A set that makes the model
less prone to the hallucinated tokens gives d < 0.

The intervention follows Het's rule (encode -> scale features -> decode; add decode(mod) - decode(orig)
to the residual stream) but at EVERY response position from the first predicting state onward,
since the response is teacher-forced. `identity` (scale 1) must give d = 0: built-in sanity check.

Caveat: this measures probabilities of other models' tokens, not on-policy generation.

Usage (GPU):
    python experiments/intervene_teacher_forced.py --het-repo ~/thesis/sae-uncertainty \
        --compare-csv results/compare_rag_norag_predict.csv --layer 16 --condition rag \
        --split validation --max-responses 300 --output results/intervene_L16_rag.json
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from halluc.data import load_qa_items                         # noqa: E402
from halluc.splits import split_of, DISCOVERY, VALIDATION, PILOT  # noqa: E402


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def feature_freq(pickle_path, layer):
    """Number of sampled tokens on which each feature is active (from an extraction pickle)."""
    import pickle
    rows = pickle.load(open(pickle_path, "rb"))["per_question_data"]
    counts = {}
    for r in rows:
        for f in r["sae_features"][layer]["indices"].tolist():
            counts[f] = counts.get(f, 0) + 1
    return counts


def matched_random(target, pool, freq, rng, window=30):
    """For each target feature pick a distinct random pool feature among the `window` nearest in firing frequency."""
    ranked = sorted(pool, key=lambda f: freq.get(f, 0))
    pos = {f: i for i, f in enumerate(ranked)}
    chosen = set()
    for t in target:
        ft = freq.get(t, 0)
        i = min(range(len(ranked)), key=lambda k: abs(freq.get(ranked[k], 0) - ft))
        cands = [f for f in ranked[max(0, i - window // 2): i + window // 2 + 1] if f not in chosen]
        chosen.add(rng.choice(cands) if cands else rng.choice([f for f in ranked if f not in chosen]))
    return sorted(chosen)


def pick_sets(csv_path, layer, n, rng, n_random, q=0.05, freq=None):
    rows = [r for r in csv.DictReader(open(csv_path)) if int(r["layer"]) == layer]
    for r in rows:
        r["f"], r["a"], r["b"], r["g"] = int(r["feature"]), fl(r["mean_diff_rag"]), fl(r["mean_diff_norag"]), fl(r["q_gap"])
    hall = [r for r in rows if r["category"] == "resistant" and r["g"] >= q and r["a"] > 0 and r["b"] > 0]
    pas = [r for r in rows if r["g"] < q and r["a"] < 0 and abs(r["a"]) > abs(r["b"])]
    hall = [r["f"] for r in sorted(hall, key=lambda r: -r["a"])[:n]]
    pas = [r["f"] for r in sorted(pas, key=lambda r: r["a"])[:n]]
    pool = sorted(set(r["f"] for r in rows) - set(hall) - set(pas))
    if freq:
        # activity-matched null: mirrors the firing-frequency profile of hall_set + pass_set
        target = hall + pas
        rand = [matched_random(rng.sample(target, n), pool, freq, rng) for _ in range(n_random)]
    else:
        rand = [sorted(rng.sample(pool, n)) for _ in range(n_random)]
    return hall, pas, rand


class LayerIntervention:
    """Forward hook on one layer: scales `feats` of the SAE code at positions >= pos_from."""

    def __init__(self, sae, torch):
        self.sae, self.torch = sae, torch
        self.feats, self.scale, self.pos_from = None, 1.0, 0

    def __call__(self, module, inputs, output):
        if self.feats is None:
            return output
        torch = self.torch
        hidden = output[0] if isinstance(output, tuple) else output
        part = hidden[:, self.pos_from:, :].float()
        with torch.no_grad():
            acts = self.sae.encode(part)
            mod = acts.clone()
            mod[:, :, self.feats] *= self.scale
            delta = self.sae.decode(mod) - self.sae.decode(acts)
        new = hidden.clone()
        new[:, self.pos_from:, :] = hidden[:, self.pos_from:, :] + delta.to(hidden.dtype)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new


def boot_ci(per_resp, n=2000, seed=0):
    r = np.random.RandomState(seed)
    m = np.array(per_resp)
    means = [m[r.randint(0, len(m), len(m))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--het-repo", default=str(ROOT.parent / "sae-uncertainty"))
    ap.add_argument("--compare-csv", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--condition", choices=["rag", "no_rag"], default="rag")
    ap.add_argument("--split", choices=[DISCOVERY, VALIDATION, PILOT], default=VALIDATION)
    ap.add_argument("--max-responses", type=int, default=300)
    ap.add_argument("--n-feats", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--amplify", type=float, default=2.0)
    ap.add_argument("--freq-pickle", default=None,
                    help="extraction pickle (token_features_predict.pkl, discovery) -> activity-matched random sets")
    ap.add_argument("--exclude-words", default="",
                    help="comma-separated words; pairs where either token is one of them are dropped "
                         "(e.g. based,passage,passages,provided,according,given)")
    ap.add_argument("--pass-only", action="store_true",
                    help="only identity + pass_suppress + random suppress draws (cheap, for many random draws)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.het_repo).expanduser()))
    import torch                                                   # noqa: E402
    from src.model import load_model, get_layers                   # noqa: E402
    from src.sae import load_sae                                   # noqa: E402
    from halluc.extract import select_items                        # noqa: E402
    from halluc.pipeline import process_item, rebase_to_no_rag     # noqa: E402
    from halluc.sampling import sample_matched_tokens              # noqa: E402

    rng = random.Random(args.seed)
    freq = feature_freq(args.freq_pickle, args.layer) if args.freq_pickle else None
    hall, pas, rand = pick_sets(args.compare_csv, args.layer, args.n_feats, rng, args.n_random, freq=freq)
    excluded = {w.strip().lower() for w in args.exclude_words.split(",") if w.strip()}
    print("hall_set (%d): %s\npass_set (%d): %s" % (len(hall), hall, len(pas), pas))

    items = select_items(load_qa_items(), split_of, args.split, args.max_responses, args.seed)
    print("%s split: %d responses" % (args.split, len(items)))
    model, tok, cfg = load_model(args.model, device=args.device)
    sae = load_sae(cfg, args.layer, device=args.device)
    hook = LayerIntervention(sae, torch)
    handle = get_layers(model, cfg)[args.layer].register_forward_hook(hook)

    def feats(ids):
        return torch.tensor(ids, dtype=torch.long, device=args.device)

    conds = {"identity": (feats(hall), 1.0), "hall_suppress": (feats(hall), 0.0),
             "hall_amplify": (feats(hall), args.amplify), "pass_amplify": (feats(pas), args.amplify),
             "pass_suppress": (feats(pas), 0.0)}
    if args.pass_only:
        conds = {k: v for k, v in conds.items() if k in ("identity", "pass_suppress")}
    for i, r in enumerate(rand):
        conds["rand%03d_suppress" % i] = (feats(r), 0.0)
        if not args.pass_only:
            conds["rand%03d_amplify" % i] = (feats(r), args.amplify)

    per = {c: {"d": [], "dh": [], "df": [], "resp": []} for c in conds}
    ra = random.Random(args.seed)                                  # same RNG use as extract_phase1
    for n_done, it in enumerate(items):
        pipe = process_item(tok, it)
        if args.condition == "no_rag":
            pipe = rebase_to_no_rag(pipe, tok, it)
        pairs, _ = sample_matched_tokens(pipe["tokens"], ra, 8, 5)
        if excluded:   # filter AFTER sampling so the sampled pairs are identical to the unfiltered run
            pairs = [p for p in pairs if p["hall"]["text"].strip().lower() not in excluded
                     and p["faithful"]["text"].strip().lower() not in excluded]
        if not pairs:
            continue
        ids = torch.tensor([pipe["input_ids"]], device=args.device)
        pred_h = torch.tensor([p["hall"]["predict_idx"] for p in pairs], device=args.device)
        pred_f = torch.tensor([p["faithful"]["predict_idx"] for p in pairs], device=args.device)
        tgt_h = ids[0, [p["hall"]["seq_idx"] for p in pairs]]
        tgt_f = ids[0, [p["faithful"]["seq_idx"] for p in pairs]]
        hook.pos_from = int(min(pred_h.min(), pred_f.min()))

        def score():
            with torch.no_grad():
                lg = model(input_ids=ids).logits[0]
                lh = torch.log_softmax(lg[pred_h].float(), -1).gather(1, tgt_h[:, None])[:, 0]
                lf = torch.log_softmax(lg[pred_f].float(), -1).gather(1, tgt_f[:, None])[:, 0]
            return lh.cpu().numpy(), lf.cpu().numpy()

        hook.feats = None
        bh, bf = score()
        for name, (fe, sc) in conds.items():
            hook.feats, hook.scale = fe, sc
            ih, if_ = score()
            per[name]["d"].append(float(((ih - if_) - (bh - bf)).mean()))
            per[name]["dh"].append(float((ih - bh).mean()))
            per[name]["df"].append(float((if_ - bf).mean()))
        hook.feats = None
        if (n_done + 1) % 25 == 0:
            print("  %d/%d responses" % (n_done + 1, len(items)))
    handle.remove()

    out = {"config": vars(args), "hall_set": hall, "pass_set": pas, "random_sets": rand, "conditions": {}}
    for name, v in per.items():
        lo, hi = boot_ci(v["d"])
        out["conditions"][name] = {"n_responses": len(v["d"]), "mean_d": float(np.mean(v["d"])),
                                   "ci95": [lo, hi], "mean_d_hall_token": float(np.mean(v["dh"])),
                                   "mean_d_faithful_token": float(np.mean(v["df"]))}
    print("\ncondition               mean d (hall-faithful logp shift)   95%% CI (resp. bootstrap)   dHall   dFaith")
    for name in ["identity", "hall_suppress", "hall_amplify", "pass_amplify", "pass_suppress"]:
        if name not in out["conditions"]:
            continue
        c = out["conditions"][name]
        print("%-22s %+9.4f   [%+.4f, %+.4f]   %+.4f  %+.4f" % (
            name, c["mean_d"], c["ci95"][0], c["ci95"][1], c["mean_d_hall_token"], c["mean_d_faithful_token"]))
    for kind in (("suppress",) if args.pass_only else ("suppress", "amplify")):
        rs = [out["conditions"]["rand%03d_%s" % (i, kind)]["mean_d"] for i in range(len(rand))]
        print("random %-9s null: mean %+.4f  sd %.4f  min %+.4f  max %+.4f"
              % (kind, np.mean(rs), np.std(rs), min(rs), max(rs)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print("Saved %s" % args.output)


if __name__ == "__main__":
    main()
