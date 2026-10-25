"""Tokenise one RAGTruth QA item and attach everything the span-level analysis needs.

For each response token we record the hallucination label, the position that predicts it,
its relative position, its token type and its copy features (against both the passages and
the full prompt context).
"""

from .alignment import (
    HALL, FAITHFUL, label_tokens, check_span_coverage,
)
from .copy_features import copy_run_lengths, copy_status
from .data import build_teacher_forced_text
from .tokens import token_type


def process_item(tokenizer, item, separator=" ", min_run=3, min_hall_frac=0.5,
                 margin_tokens=0, add_special_tokens=True):
    """Tokenise prompt + response and compute per-response-token records.

    `tokenizer` must be callable like a HuggingFace fast tokenizer:
        tokenizer(text, return_offsets_mapping=True, add_special_tokens=...)
        -> mapping with "input_ids" and "offset_mapping".
    """
    full_text, (resp_start, resp_end), (pass_start, pass_end) = \
        build_teacher_forced_text(item, separator)

    enc = tokenizer(full_text, return_offsets_mapping=True,
                    add_special_tokens=add_special_tokens)
    ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]

    labels = label_tokens(
        offsets, resp_start, resp_end, item["strict_spans"], item["ambiguous_spans"],
        text=full_text, min_hall_frac=min_hall_frac, margin_tokens=margin_tokens,
    )
    span_failures = check_span_coverage(offsets, labels, full_text, resp_start, item["strict_spans"])

    # Reference token sequences for the copy features
    ctx_ids = [t for t, (s, e), l in zip(ids, offsets, labels) if not l.in_response and e > s]
    pass_ids = [t for t, (s, e) in zip(ids, offsets)
                if e > s and s < pass_end and e > pass_start]

    resp_idx = [i for i, l in enumerate(labels) if l.in_response and offsets[i][1] > offsets[i][0]]
    resp_ids = [ids[i] for i in resp_idx]
    pass_run = copy_run_lengths(resp_ids, pass_ids)
    ctx_run = copy_run_lengths(resp_ids, ctx_ids)

    n = len(resp_idx)
    tokens = []
    for k, i in enumerate(resp_idx):
        s, e = offsets[i]
        surface = full_text[s:e]
        tokens.append({
            "seq_idx": i,
            "predict_idx": i - 1,          # residual at i-1 predicts token i
            "resp_pos": k,
            "rel_pos": (k / (n - 1)) if n > 1 else 0.0,
            "text": surface,
            "type": token_type(surface),
            "label": labels[i].label,
            "span_idx": labels[i].span_idx,
            "hall_frac": labels[i].hall_frac,
            "passage_run": pass_run[k],
            "ctx_run": ctx_run[k],
            "passage_status": copy_status(pass_run[k], min_run),
            "ctx_status": copy_status(ctx_run[k], min_run),
        })

    return {
        "response_id": item["response_id"],
        "source_id": item["source_id"],
        "source_model": item["source_model"],
        "n_prompt_tokens": len(ids) - n,
        "n_resp_tokens": n,
        "input_ids": ids,
        "tokens": tokens,
        "span_failures": span_failures,
        "n_strict_spans": len(item["strict_spans"]),
    }


def rebase_to_no_rag(pipe, tokenizer, item, add_special_tokens=True):
    """Return a copy of `pipe` whose prompt has the passages removed (the "no-RAG" arm).

    Labels, token types, copy features and relative positions are kept as computed against the
    RAG prompt; only `input_ids` and the sequence indices change. The response token ids are
    reused exactly from the RAG tokenisation, so both arms score identical tokens.
    """
    from .data import build_no_rag_prompt

    toks = pipe["tokens"]
    if not toks:
        return {**pipe, "tokens": [], "input_ids": list(pipe["input_ids"])}
    first = toks[0]["seq_idx"]
    new_prompt = tokenizer(build_no_rag_prompt(item), add_special_tokens=add_special_tokens)["input_ids"]
    new_ids = list(new_prompt) + list(pipe["input_ids"][first:])
    shift = len(new_prompt) - first
    new_toks = []
    for t in toks:
        t2 = dict(t)
        t2["seq_idx"] = t["seq_idx"] + shift
        t2["predict_idx"] = t["predict_idx"] + shift
        new_toks.append(t2)
    return {**pipe, "input_ids": new_ids, "tokens": new_toks,
            "n_prompt_tokens": len(new_prompt), "condition": "no_rag"}


def class_of(token):
    """Collapse labels to the two classes of interest (None = exclude from both)."""
    if token["label"] == HALL:
        return "hall"
    if token["label"] == FAITHFUL:
        return "faithful"
    return None
