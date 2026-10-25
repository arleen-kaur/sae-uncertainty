"""Load RAGTruth QA items for span-level (token-level) analysis.

Each item carries the exact prompt the original LLMs saw (`source_info.prompt`), the
response, and the human span labels split into

    strict_spans     : spans NOT flagged `implicit_true`  -> used as "hallucinated"
    ambiguous_spans  : spans flagged `implicit_true`      -> masked out of both classes

Span offsets are character offsets into `response`. (Verified: every span's
`response[start:end] == text`; no zero-length spans; none run past the response.)
"""

import json
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = THESIS_ROOT / "ragtruth" / "dataset"


def load_qa_items(dataset_dir=None, split=None, quality="good", limit=None):
    """Return a list of QA response dicts (see module docstring).

    split   : None | "train" | "test"   (RAGTruth's official split, by question)
    quality : keep only responses with this `quality` (None keeps all)
    """
    dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR

    sources = {}
    with open(dataset_dir / "source_info.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["task_type"] == "QA":
                sources[row["source_id"]] = row

    items = []
    with open(dataset_dir / "response.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["source_id"] not in sources:
                continue
            if quality is not None and r.get("quality") != quality:
                continue
            if split is not None and r["split"] != split:
                continue
            src = sources[r["source_id"]]
            strict, ambiguous = [], []
            for lb in r["labels"]:
                span = (lb["start"], lb["end"], lb["label_type"])
                (ambiguous if lb.get("implicit_true") else strict).append(span)
            items.append({
                "response_id": r["id"],
                "source_id": r["source_id"],
                "source_model": r["model"],
                "split": r["split"],
                "prompt": src["prompt"],
                "question": src["source_info"]["question"],
                "passages": src["source_info"]["passages"],
                "response": r["response"],
                "strict_spans": sorted(strict),
                "ambiguous_spans": sorted(ambiguous),
            })
            if limit and len(items) >= limit:
                break
    return items


def item_category(item):
    """'hallucinated' | 'ambiguous' | 'clean' (same rule as the bridge adapter)."""
    if item["strict_spans"]:
        return "hallucinated"
    if item["ambiguous_spans"]:
        return "ambiguous"
    return "clean"


NO_RAG_TEMPLATE = "Briefly answer the following question:\n{question}\noutput:"


def build_no_rag_prompt(item):
    """Same task with the retrieved passages (and the passage-grounding instructions) removed.

    The response is left untouched, so the no-RAG arm reads the SAME response tokens as the
    RAG arm and only the context differs. Note the question text is taken verbatim from
    `source_info.question`; the instruction wording is the first line of the RAGTruth prompt.
    """
    return NO_RAG_TEMPLATE.format(question=item["question"].strip())


def build_teacher_forced_text(item, separator=" "):
    """Full text the model reads: original prompt + separator + response.

    The prompt ends in "output:" (all 989 QA prompts do), exactly where the original
    LLMs started writing. Returns
        full_text,
        (resp_start, resp_end)      character range of the response in full_text,
        (pass_start, pass_end)      character range of the passages in full_text.
    """
    prompt = item["prompt"].rstrip()
    pass_start = prompt.find(item["passages"])
    if pass_start < 0:
        raise ValueError("passages not found inside prompt for response %s" % item["response_id"])
    pass_end = pass_start + len(item["passages"])
    resp_start = len(prompt) + len(separator)
    full_text = prompt + separator + item["response"]
    return full_text, (resp_start, resp_start + len(item["response"])), (pass_start, pass_end)
