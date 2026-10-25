"""Wrappers for two extra free hallucination judges. Heavy imports (torch, transformers, minicheck) are lazy, so the
pure helpers (prompt, parsing, sentence splitting) can be tested anywhere.

  lynx      : PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct (CC BY-NC 4.0). An LLM judge: given QUESTION, DOCUMENT and
              ANSWER it writes JSON {"REASONING": ..., "SCORE": "PASS"|"FAIL"}. FAIL = not faithful = hallucinated.
              The prompt below is the one on the model card, unchanged.
  minicheck : lytang/MiniCheck-Flan-T5-Large (Tang et al., EMNLP 2024). Sentence-level: P(sentence supported by the
              document). The answer is split into sentences; hallucinated <=> its LEAST supported sentence < threshold.
"""
import re

LYNX_MODEL = "PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct"

LYNX_PROMPT = """
Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT. Show your reasoning.

--
QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
{question}

--
DOCUMENT:
{context}

--
ANSWER:
{answer}

--

Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
{{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}
"""

_SCORE = re.compile(r'"SCORE"\s*:\s*"?\s*(PASS|FAIL)', re.I)


def lynx_prompt(question, context, answer):
    return LYNX_PROMPT.format(question=question, context=context, answer=answer)


def parse_lynx(text):
    """'PASS' | 'FAIL' | None. Only the SCORE field counts (the reasoning may contain either word)."""
    found = _SCORE.findall(text)
    return found[-1].upper() if found else None


def _merge_short(sentences, min_words=3):
    """Join fragments such as '1.' or 'Step 2:' onto the sentence that follows them."""
    out, carry = [], ""
    for s in sentences:
        s = (carry + " " + s).strip() if carry else s
        carry = ""
        if len(s.split()) < min_words:
            carry = s
            continue
        out.append(s)
    if carry:
        if out:
            out[-1] = out[-1] + " " + carry
        else:
            out.append(carry)
    return out


def split_sentences(text):
    """Sentences of an answer (line breaks are hard boundaries); list markers are merged into their sentence."""
    sents = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        try:
            from nltk.tokenize import sent_tokenize
            sents.extend(s.strip() for s in sent_tokenize(block) if s.strip())
        except Exception:  # nltk or its punkt data unavailable: plain punctuation rule
            sents.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip())
    return _merge_short(sents)


# ---------------------------------------------------------------- Lynx

def load_lynx(model_name=LYNX_MODEL, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                 device_map={"": device}, low_cpu_mem_usage=True)
    model.eval()
    return model, tok


def lynx_generate(model, tok, triples, batch_size=8, max_new_tokens=600, progress=None):
    """triples: [(question, context, answer)]. Returns the raw generated text for each."""
    import torch
    stops = [tok.eos_token_id]
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0 and eot not in stops:
        stops.append(eot)
    outs = []
    for i in range(0, len(triples), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": lynx_prompt(*t)}],
                                         add_generation_prompt=True, tokenize=False)
                 for t in triples[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=stops)
        for g in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(g, skip_special_tokens=True))
        if progress:
            progress(min(i + batch_size, len(triples)), len(triples))
    return outs


# ---------------------------------------------------------------- MiniCheck

def load_minicheck(model_name="flan-t5-large", cache_dir=None):
    import nltk
    for res, path in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)
    from minicheck.minicheck import MiniCheck
    return MiniCheck(model_name=model_name, cache_dir=cache_dir)


def minicheck_answer(scorer, context, answer, threshold=0.5):
    """(hallucinated | None, raw). Scores every sentence against the whole context; the weakest sentence decides."""
    sents = split_sentences(answer)
    if not sents:
        return None, {"min_support": None, "sentences": [], "sentence_support": []}
    _, probs, _, _ = scorer.score(docs=[context] * len(sents), claims=sents)
    probs = [float(p) for p in probs]
    return min(probs) < threshold, {"min_support": min(probs), "sentences": sents, "sentence_support": probs}
