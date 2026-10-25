"""On-policy pilot, step 1: let Llama-3.1-8B (the model the SAEs belong to) answer RAGTruth QA questions.

Questions come from RAGTruth's 150 official TEST questions, which halluc.splits reserves for exactly this
pilot (never used for discovery/validation) and which LettuceDetect's RAGTruth training never saw.
The prompt is RAGTruth's own prompt (it ends in "output:"), so the base model answers the same task the
original models did.  Output items use the same schema as halluc.data.load_qa_items, with empty spans
(to be filled by scripts/judge_responses.py), so the existing token pipeline can read them unchanged.

Usage (GPU machine, inside venv-sae):
    python -u scripts/generate_on_policy.py --output outputs/onpolicy_gen.jsonl            # 150 q x 4 samples
    python -u scripts/generate_on_policy.py --samples 1 --temperature 0 --output outputs/onpolicy_greedy.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from halluc.data import load_qa_items  # noqa: E402


def clean_response(raw):
    """First line of the continuation (RAGTruth answers are single paragraphs); base models then ramble."""
    text = raw.lstrip()
    return text.split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--split", default="test", help="RAGTruth split; 'test' = the reserved pilot questions")
    ap.add_argument("--n-questions", type=int, default=150)
    ap.add_argument("--samples", type=int, default=4, help="responses per question")
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = greedy (then --samples should be 1)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.temperature == 0 and args.samples != 1:
        raise SystemExit("greedy decoding is deterministic: use --samples 1")

    # one entry per question (the prompt is identical for all six original responses)
    by_source = {}
    for it in load_qa_items(args.dataset_dir, split=args.split, quality=None):
        by_source.setdefault(it["source_id"], it)
    sources = sorted(by_source)
    random.Random(args.seed).shuffle(sources)
    sources = sources[:args.n_questions]
    print("%d %s-split questions selected" % (len(sources), args.split))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 device_map={"": args.device}, low_cpu_mem_usage=True)
    model.eval()
    torch.manual_seed(args.seed)

    jobs = [(sid, k) for sid in sources for k in range(args.samples)]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_trunc = n_empty = 0
    with open(str(out_path), "w") as f:
        for i in range(0, len(jobs), args.batch_size):
            batch = jobs[i:i + args.batch_size]
            prompts = [by_source[sid]["prompt"].rstrip() for sid, _ in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(args.device)
            gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
            if args.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            for (sid, k), toks in zip(batch, new_tokens):
                raw = tok.decode(toks, skip_special_tokens=True)
                resp = clean_response(raw)
                ntok = int((toks != tok.pad_token_id).sum())
                truncated = ntok >= args.max_new_tokens and "\n" not in raw.lstrip()
                n_trunc += int(truncated)
                n_empty += int(not resp)
                src = by_source[sid]
                f.write(json.dumps({
                    "response_id": "onp_%s_%d" % (sid, k), "source_id": sid, "source_model": "llama-3.1-8b-base",
                    "split": src["split"], "prompt": src["prompt"], "question": src["question"],
                    "passages": src["passages"], "response": resp, "strict_spans": [], "ambiguous_spans": [],
                    "gen": {"raw": raw, "sample": k, "temperature": args.temperature, "top_p": args.top_p,
                            "max_new_tokens": args.max_new_tokens, "truncated": truncated, "seed": args.seed},
                }) + "\n")
            f.flush()
            print("  %d/%d responses" % (min(i + args.batch_size, len(jobs)), len(jobs)))
    print("wrote %s | empty responses: %d | hit max-new-tokens without a newline: %d" % (out_path, n_empty, n_trunc))


if __name__ == "__main__":
    main()
