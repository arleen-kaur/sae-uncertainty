"""Install check for the free hallucination judges. Run this FIRST on the GPU machine.

    python scripts/judge_check.py                  # all four judges
    python scripts/judge_check.py lynx minicheck   # only the named ones (lettuce, hhem, lynx, minicheck)

Each judge is run on one supported and one unsupported answer. Expected: LettuceDetect flags a span in the
second answer; HHEM scores the first answer high (consistent) and the second low; Lynx says PASS then FAIL;
MiniCheck gives the second answer a low minimum sentence support.
"""
import os
import sys
import traceback
from pathlib import Path

# ModernBERT (LettuceDetect) calls torch.compile, which needs Python.h (missing on this host): run eagerly instead.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONTEXT = "France is a country in Europe. The capital of France is Paris. The population of France is 67 million."
QUESTION = "What is the capital of France? What is the population of France?"
GOOD = "The capital of France is Paris. The population of France is 67 million."
BAD = "The capital of France is Paris. The population of France is 69 million."


def check_lettuce():
    from lettucedetect.models.inference import HallucinationDetector
    det = HallucinationDetector(method="transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    for tag, ans in (("good", GOOD), ("bad", BAD)):
        print("  %-4s -> %s" % (tag, det.predict(context=[CONTEXT], question=QUESTION, answer=ans, output_format="spans")))


def check_hhem():
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained("vectara/hallucination_evaluation_model", trust_remote_code=True)
    s = m.predict([(CONTEXT, GOOD), (CONTEXT, BAD)])
    print("  good score %.3f | bad score %.3f   (higher = more consistent)" % (float(s[0]), float(s[1])))


def check_lynx():
    from halluc.judge_models import load_lynx, lynx_generate, parse_lynx
    model, tok = load_lynx()
    outs = lynx_generate(model, tok, [(QUESTION, CONTEXT, GOOD), (QUESTION, CONTEXT, BAD)], batch_size=2)
    for tag, text in zip(("good", "bad"), outs):
        print("  %-4s -> %s" % (tag, parse_lynx(text)))
        print("       %s" % text[:400].replace("\n", " "))


def check_minicheck():
    from halluc.judge_models import load_minicheck, minicheck_answer
    scorer = load_minicheck(cache_dir=os.path.expanduser("~/.cache/minicheck"))
    for tag, ans in (("good", GOOD), ("bad", BAD)):
        hall, raw = minicheck_answer(scorer, CONTEXT, ans)
        print("  %-4s -> hallucinated=%s  sentence support=%s" % (tag, hall, ["%.2f" % p for p in raw["sentence_support"]]))


CHECKS = [("lettuce", "LettuceDetect", check_lettuce), ("hhem", "HHEM-2.1-Open", check_hhem),
          ("lynx", "Lynx-8B", check_lynx), ("minicheck", "MiniCheck-Flan-T5-Large", check_minicheck)]

wanted = sys.argv[1:] or [c[0] for c in CHECKS]
ok = True
for key, title, fn in CHECKS:
    if key not in wanted:
        continue
    print("== %s" % title)
    try:
        fn()
    except Exception:
        ok = False
        traceback.print_exc()
        print("  %s FAILED (see above)" % title)

print("\nALL REQUESTED JUDGES OK" if ok else "\nAt least one judge failed; paste the traceback.")
sys.exit(0 if ok else 1)
