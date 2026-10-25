"""Install check for the free hallucination judges. Run this FIRST on the GPU machine.

    pip install lettucedetect          # if the LettuceDetect import below fails
    python scripts/judge_check.py

Each judge is run on one supported and one unsupported answer. Expected: LettuceDetect flags a span in the
second answer; HHEM scores the first answer high (consistent) and the second low.
"""
import os
import sys
import traceback

# ModernBERT (LettuceDetect) calls torch.compile, which needs Python.h (missing on this host): run eagerly instead.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

CONTEXT = "France is a country in Europe. The capital of France is Paris. The population of France is 67 million."
QUESTION = "What is the capital of France? What is the population of France?"
GOOD = "The capital of France is Paris. The population of France is 67 million."
BAD = "The capital of France is Paris. The population of France is 69 million."

ok = True

print("== LettuceDetect")
try:
    from lettucedetect.models.inference import HallucinationDetector
    det = HallucinationDetector(method="transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    for tag, ans in (("good", GOOD), ("bad", BAD)):
        print("  %-4s -> %s" % (tag, det.predict(context=[CONTEXT], question=QUESTION, answer=ans, output_format="spans")))
except Exception:
    ok = False
    traceback.print_exc()
    print("  LettuceDetect FAILED (see above)")

print("== HHEM-2.1-Open")
try:
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained("vectara/hallucination_evaluation_model", trust_remote_code=True)
    s = m.predict([(CONTEXT, GOOD), (CONTEXT, BAD)])
    print("  good score %.3f | bad score %.3f   (higher = more consistent)" % (float(s[0]), float(s[1])))
except Exception:
    ok = False
    traceback.print_exc()
    print("  HHEM FAILED (see above)")

print("\nALL JUDGES OK" if ok else "\nAt least one judge failed; paste the traceback.")
sys.exit(0 if ok else 1)
