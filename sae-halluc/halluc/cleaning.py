"""Turn a base model's raw continuation into one answer, or say why it is not an answer.

A base model given RAGTruth's prompt (which ends in "output:") often emits nothing, repeats the question,
restates the task, or answers and then rambles into a new task. Rules, in order:

1. keep the first paragraph (cut at the first blank line): whole numbered lists survive, rambling after a
   blank line does not;
2. drop lines that only restate the task ("The following is the required output:");
3. if the text stops without final punctuation (it hit the token limit), trim back to the last full sentence;
4. non-answers are reported with a reason instead of being judged: empty, echo (just the question),
   prompt_leak (repeats the question or the prompt template / passages), too_short (< MIN_WORDS words).
"""
import re

MIN_WORDS = 3
META_PREFIXES = ("the following is", "the following are", "output:", "answer:", "here is the required",
                 "the required output")
_END = re.compile(r"[.!?][\"')\]]*")


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def clean_response(raw):
    """First paragraph of the continuation, task-restating lines removed, cut back to a full sentence."""
    text = raw.lstrip()
    paragraph = re.split(r"\n[ \t]*\n", text, maxsplit=1)[0]
    lines = [ln.strip() for ln in paragraph.split("\n") if ln.strip()]
    lines = [ln for ln in lines if not ln.lower().startswith(META_PREFIXES)]
    out = "\n".join(lines)
    nothing_after = len(paragraph) == len(text.rstrip())
    if out and nothing_after and not _END.search(out[-3:]):
        ends = [m.end() for m in _END.finditer(out)]
        if ends and len(out[:ends[-1]].split()) >= MIN_WORDS:
            out = out[:ends[-1]]
    return out.strip()


_TEMPLATE = re.compile(r"briefly answer the following question|bear in mind that your response|"
                       r"strictly based on the following|\bpassage\s*\d\s*:|the question is\s*:", re.I)


def classify(raw, question):
    """(cleaned answer, reason). reason is None for a usable answer, else
    'empty' | 'echo' | 'prompt_leak' | 'too_short'.

    prompt_leak = the model is repeating the prompt (starts with the question, or contains the template's own
    wording / "passage 1:"). Copied passages look perfectly "consistent", so these must never reach the judges.
    """
    resp = clean_response(raw)
    if not resp:
        return resp, "empty"
    q, r = _norm(question), _norm(resp)
    if r == q:
        return resp, "echo"
    if (q and r[:len(q)] == q) or _TEMPLATE.search(resp):
        return resp, "prompt_leak"
    if len(resp.split()) < MIN_WORDS:
        return resp, "too_short"
    return resp, None
