"""Token-level extraction: residual capture at chosen positions + next-token statistics.

Het's loop is "last prompt token -> SAE -> A/B/C/D entropy". Here a *token* of a RAGTruth
response is the unit:

    position convention   the residual at sequence position i-1 is the state that PREDICTS token i.
        'predict'  features at position i-1   (generation-time state; the only place steering can act)
        'self'     features at position i     (the model has read token i; detection ceiling,
                                               partly lexical because it contains the token's identity)
    entropy               entropy of the full-vocabulary next-token distribution predicting token i
                          (same value for both positions, so the 2x2 is identical and only the
                          feature position differs)
    correct               token is faithful (False = inside a hallucinated span)

Two phases, like Het: (1) one forward pass per response capturing residuals for the sampled tokens
only; (2) free the LM, then encode the cached residuals with one SAE at a time.
"""

import random
import time
from datetime import datetime

import numpy as np
import torch

from .alignment import HALL, FAITHFUL
from .pipeline import process_item, rebase_to_no_rag
from .sampling import sample_matched_tokens
from .tokens import ALL_TYPES

LABEL_CODES = {"hall": 0, "faithful": 1, "ambig": 2, "boundary": 3, "near": 4, "space": 5}
TYPE_CODES = {t: i for i, t in enumerate(ALL_TYPES)}
KINDS = ("predict", "self")


class PositionCapture:
    """Forward hooks that keep the residual stream (block output) at chosen sequence positions."""

    def __init__(self):
        self.positions = None      # LongTensor of sequence positions to keep (set before each forward)
        self.store = {}            # layer -> [len(positions), d] tensor (on the model's device)
        self.handles = []

    def _make(self, layer):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.store[layer] = hidden[0, self.positions, :].detach()
        return hook

    def register(self, layer_modules, layer_indices):
        for l in layer_indices:
            self.handles.append(layer_modules[l].register_forward_hook(self._make(l)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles, self.store = [], {}


@torch.no_grad()
def token_stats(logits, ids, resp_seq_idx, chunk=128):
    """Entropy, surprisal and top-1 probability of the distribution that predicts each response token.

    logits: [T, V] for one sequence; ids: LongTensor [T]; resp_seq_idx: sequence indices of the
    response tokens. The distribution predicting token i is logits[i-1].
    """
    idx = torch.as_tensor(resp_seq_idx, dtype=torch.long, device=logits.device)
    ent, sur, top1 = [], [], []
    for a in range(0, len(idx), chunk):
        sel = idx[a:a + chunk]
        logp = torch.log_softmax(logits[sel - 1].float(), dim=-1)
        p = logp.exp()
        ent.append(-(p * logp).sum(-1))
        sur.append(-logp.gather(1, ids[sel].unsqueeze(1)).squeeze(1))
        top1.append(p.max(-1).values)
    cat = lambda xs: torch.cat(xs).float().cpu().numpy()
    return cat(ent), cat(sur), cat(top1)


def select_items(all_items, split_fn, split, max_responses=None, seed=42):
    """Hallucinated responses (>=1 strict span) belonging to `split`; optional seeded subsample."""
    items = [it for it in all_items
             if it["strict_spans"] and split_fn(it["source_id"], it["split"]) == split]
    items.sort(key=lambda it: int(it["response_id"]))
    if max_responses and len(items) > max_responses:
        items = random.Random(seed).sample(items, max_responses)
        items.sort(key=lambda it: int(it["response_id"]))
    return items


def extract_phase1(model, tokenizer, layer_modules, items, layer_indices, device, seed=42,
                   min_run=3, margin=2, max_per_span=8, pos_bins=5, log=print, condition="rag"):
    """Run the LM over `items`; return sampled-token records, cached residuals, population stats.

    condition: "rag"    -> original RAGTruth prompt (passages present)
               "no_rag" -> passages removed; identical response tokens, labels and sampled pairs
                           (the rebase happens before sampling and only shifts indices), so the
                           two runs are paired token-for-token via `pair_id` / `role`.
    """
    if condition not in ("rag", "no_rag"):
        raise ValueError("condition must be 'rag' or 'no_rag'")
    rng = random.Random(seed)
    cap = PositionCapture()
    cap.register(layer_modules, layer_indices)

    chunks = {l: {k: [] for k in KINDS} for l in layer_indices}
    records, population = [], []
    n_unmatched, n_skipped, tiers = 0, 0, {0: 0, 1: 0, 2: 0}
    t0 = time.time()

    try:
        for n_done, it in enumerate(items):
            pipe = process_item(tokenizer, it, min_run=min_run, margin_tokens=margin)
            if condition == "no_rag":
                pipe = rebase_to_no_rag(pipe, tokenizer, it)
            pairs, n_un = sample_matched_tokens(pipe["tokens"], rng, max_per_span, pos_bins)
            n_unmatched += n_un
            if not pairs:
                n_skipped += 1
                continue
            for p in pairs:
                tiers[p["tier"]] += 1

            ids = torch.tensor([pipe["input_ids"]], device=device)
            sampled = []
            for p in pairs:
                sampled.append((p, "hall", p["hall"]))
                sampled.append((p, "faithful", p["faithful"]))
            positions = sorted({t["predict_idx"] for _, _, t in sampled}
                               | {t["seq_idx"] for _, _, t in sampled})
            row_of = {pos: i for i, pos in enumerate(positions)}
            cap.positions = torch.tensor(positions, dtype=torch.long, device=device)

            with torch.no_grad():
                out = model(input_ids=ids)

            resp_seq = [t["seq_idx"] for t in pipe["tokens"]]
            ent, sur, top1 = token_stats(out.logits[0], ids[0], resp_seq)
            k_of = {s: k for k, s in enumerate(resp_seq)}
            for k, t in enumerate(pipe["tokens"]):
                t["entropy"], t["surprisal"], t["top1"] = float(ent[k]), float(sur[k]), float(top1[k])

            population.append(_population_row(it, pipe, ent, sur, top1))

            rows_pred = torch.tensor([row_of[t["predict_idx"]] for _, _, t in sampled], device=device)
            rows_self = torch.tensor([row_of[t["seq_idx"]] for _, _, t in sampled], device=device)
            for l in layer_indices:
                chunks[l]["predict"].append(cap.store[l][rows_pred].cpu())
                chunks[l]["self"].append(cap.store[l][rows_self].cpu())

            for p, role, t in sampled:
                k = k_of[t["seq_idx"]]
                span_type = it["strict_spans"][t["span_idx"]][2] if role == "hall" and t["span_idx"] >= 0 else None
                records.append({
                    "response_id": it["response_id"], "source_id": it["source_id"],
                    "source_model": it["source_model"],
                    "pair_id": "%s_%d" % (it["response_id"], p["pair_id"]), "tier": p["tier"],
                    "role": role, "correct": role == "faithful", "span_type": span_type,
                    "condition": condition,
                    "entropy": float(ent[k]), "surprisal": float(sur[k]), "top1": float(top1[k]),
                    "seq_idx": t["seq_idx"], "predict_idx": t["predict_idx"], "text": t["text"],
                    "type": t["type"], "rel_pos": t["rel_pos"], "span_idx": t["span_idx"],
                    "ctx_status": t["ctx_status"], "passage_status": t["passage_status"],
                    "ctx_run": t["ctx_run"], "passage_run": t["passage_run"],
                })
            if (n_done + 1) % 25 == 0:
                log("  %d/%d responses  %d sampled tokens  %.1fs"
                    % (n_done + 1, len(items), len(records), time.time() - t0))
    finally:
        cap.remove()

    cache = {l: {k: torch.cat(chunks[l][k]) if chunks[l][k] else torch.empty(0)
                 for k in KINDS} for l in layer_indices}
    stats = {"condition": condition, "n_responses": len(items), "n_skipped_no_pairs": n_skipped,
             "n_pairs": len(records) // 2, "n_unmatched_hall_tokens": n_unmatched,
             "match_tiers": tiers}
    return records, cache, population, stats


def _population_row(it, pipe, ent, sur, top1):
    toks = pipe["tokens"]
    return {
        "response_id": it["response_id"], "source_id": it["source_id"],
        "source_model": it["source_model"],
        "seq_idx": np.array([t["seq_idx"] for t in toks], dtype=np.int32),
        "label": np.array([LABEL_CODES.get(t["label"], 6) for t in toks], dtype=np.int8),
        "type": np.array([TYPE_CODES[t["type"]] for t in toks], dtype=np.int8),
        "ctx_status": np.array([t["ctx_status"] for t in toks], dtype=np.int8),
        "passage_status": np.array([t["passage_status"] for t in toks], dtype=np.int8),
        "rel_pos": np.array([t["rel_pos"] for t in toks], dtype=np.float32),
        "span_idx": np.array([t["span_idx"] for t in toks], dtype=np.int16),
        "entropy": ent.astype(np.float32), "surprisal": sur.astype(np.float32),
        "top1": top1.astype(np.float32),
    }


@torch.no_grad()
def encode_residuals(sae, residuals, device, batch=256):
    """Encode [N, d] residuals; return a list of (indices int64, values float32) of active features."""
    out = []
    for a in range(0, len(residuals), batch):
        x = residuals[a:a + batch].to(device).float()
        acts = sae.encode(x)
        mask = acts > 0
        counts = mask.sum(1).tolist()
        cols = mask.nonzero(as_tuple=True)[1].cpu().numpy().astype(np.int64)
        vals = acts[mask].float().cpu().numpy()
        o = 0
        for c in counts:
            out.append((cols[o:o + c], vals[o:o + c]))
            o += c
    return out


def encode_cache(cache, sae_loader, layer_indices, device, free=lambda: None, log=print):
    """Phase 2: one SAE at a time. Returns {kind: {layer: [(indices, values), ...]}}."""
    feats = {k: {} for k in KINDS}
    for l in layer_indices:
        log("Phase 2: SAE for layer %d" % l)
        sae = sae_loader(l)
        for k in KINDS:
            feats[k][l] = encode_residuals(sae, cache[l][k], device)
        del sae
        free()
    return feats


def build_pickles(records, feats, layer_indices, config):
    """One Het-compatible payload per capture position (`per_question_data` + `config.sae_layers`)."""
    payloads = {}
    n_h = sum(1 for r in records if r["role"] == "hall")
    n_f = len(records) - n_h
    for kind in KINDS:
        per_q = []
        for i, r in enumerate(records):
            d = dict(r)
            d["q_idx"] = i
            d["sae_features"] = {}
            for l in layer_indices:
                idx, val = feats[kind][l][i]
                d["sae_features"][l] = {
                    "indices": idx, "values": val, "n_active": len(idx),
                    "total_activation": float(val.sum()),
                    "max_activation": float(val.max()) if len(val) else 0.0,
                }
            per_q.append(d)
        payloads[kind] = {
            "config": {**config, "sae_layers": list(layer_indices), "position": kind,
                       "n_questions": len(records), "timestamp": datetime.now().isoformat()},
            "summary": {"n_hall_tokens": n_h, "n_faithful_tokens": n_f,
                        "mean_entropy_hall": float(np.mean([r["entropy"] for r in records if r["role"] == "hall"])),
                        "mean_entropy_faithful": float(np.mean([r["entropy"] for r in records if r["role"] != "hall"]))},
            "per_question_data": per_q,
        }
    return payloads
