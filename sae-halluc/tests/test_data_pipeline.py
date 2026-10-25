import collections
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from halluc.data import (  # noqa: E402
    load_qa_items, item_category, build_teacher_forced_text, DEFAULT_DATASET_DIR,
)
from halluc.pipeline import process_item  # noqa: E402
from halluc.alignment import HALL, FAITHFUL  # noqa: E402
from fake_tokenizer import FakeTokenizer  # noqa: E402

HAVE_DATA = (DEFAULT_DATASET_DIR / "response.jsonl").exists()


@unittest.skipUnless(HAVE_DATA, "RAGTruth dataset not found")
class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = load_qa_items()

    def test_category_counts_match_bridge_analysis(self):
        c = collections.Counter(item_category(it) for it in self.items)
        self.assertEqual(c["hallucinated"], 1319)
        self.assertEqual(c["ambiguous"], 387)
        self.assertEqual(c["clean"], 4061)

    def test_spans_are_valid_and_ordered(self):
        for it in self.items:
            n = len(it["response"])
            for s, e, _ in it["strict_spans"] + it["ambiguous_spans"]:
                self.assertTrue(0 <= s < e <= n)

    def test_teacher_forced_text_slices(self):
        for it in self.items[:200]:
            text, (rs, re_), (ps, pe) = build_teacher_forced_text(it)
            self.assertEqual(text[rs:re_], it["response"])
            self.assertEqual(text[ps:pe], it["passages"])
            self.assertTrue(text[:rs].rstrip().endswith("output:"))


@unittest.skipUnless(HAVE_DATA, "RAGTruth dataset not found")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        items = load_qa_items()
        hall = [i for i in items if item_category(i) == "hallucinated"][:150]
        clean = [i for i in items if item_category(i) == "clean"][:60]
        cls.sample = hall + clean

    def _run(self, mode, **kw):
        tok = FakeTokenizer(mode)
        return [process_item(tok, it, **kw) for it in self.sample]

    def test_no_alignment_failures_and_every_span_gets_hall_tokens(self):
        for mode in ("word_untrimmed", "word_trimmed"):
            outs = self._run(mode)
            n_spans = sum(o["n_strict_spans"] for o in outs)
            fails = sum(len(o["span_failures"]) for o in outs)
            self.assertEqual(fails, 0, mode)
            with_hall = 0
            for o in outs:
                got = {t["span_idx"] for t in o["tokens"] if t["label"] == HALL}
                with_hall += len(got)
            # overlapping spans share tokens, so allow a tiny shortfall
            self.assertGreaterEqual(with_hall / max(1, n_spans), 0.98, mode)

    def test_conventions_agree_on_labels(self):
        a = self._run("word_untrimmed")
        b = self._run("word_trimmed")
        for oa, ob in zip(a, b):
            self.assertEqual([t["label"] for t in oa["tokens"]], [t["label"] for t in ob["tokens"]])

    def test_clean_responses_have_only_faithful_like_tokens(self):
        for o in self._run("word_trimmed")[-60:]:
            self.assertEqual(o["n_strict_spans"], 0)
            self.assertFalse(any(t["label"] == HALL for t in o["tokens"]))

    def test_record_invariants(self):
        for o in self._run("word_trimmed"):
            toks = o["tokens"]
            self.assertTrue(toks)
            self.assertEqual([t["resp_pos"] for t in toks], list(range(len(toks))))
            for t in toks:
                self.assertEqual(t["predict_idx"], t["seq_idx"] - 1)
                self.assertGreaterEqual(t["predict_idx"], 0)
                self.assertTrue(0.0 <= t["rel_pos"] <= 1.0)
                self.assertGreaterEqual(t["ctx_run"], t["passage_run"])  # ctx is a superset reference

    def test_copy_features_are_informative(self):
        outs = self._run("word_trimmed")
        faithful_copy, hall_copy = [], []
        for o in outs:
            for t in o["tokens"]:
                if t["label"] == FAITHFUL:
                    faithful_copy.append(t["passage_status"] == 2)
                elif t["label"] == HALL:
                    hall_copy.append(t["passage_status"] == 2)
        fc = sum(faithful_copy) / len(faithful_copy)
        hc = sum(hall_copy) / len(hall_copy)
        # The confound the design must control: faithful tokens are copied far more often.
        self.assertGreater(fc, hc)


@unittest.skipUnless(HAVE_DATA, "RAGTruth dataset not found")
class RealTokenizerTests(unittest.TestCase):
    """Runs only when `transformers` and the tokenizer named in $HALLUC_TOKENIZER are available."""

    def test_real_tokenizer_alignment(self):
        name = os.environ.get("HALLUC_TOKENIZER")
        if not name:
            self.skipTest("set HALLUC_TOKENIZER (e.g. meta-llama/Llama-3.1-8B) to run")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # noqa: BLE001
            self.skipTest("cannot load tokenizer %s: %s" % (name, exc))
        items = load_qa_items()
        hall = [i for i in items if item_category(i) == "hallucinated"][:300]
        fails, spans, spans_with_hall = 0, 0, 0
        for it in hall:
            o = process_item(tok, it)
            fails += len(o["span_failures"])
            spans += o["n_strict_spans"]
            spans_with_hall += len({t["span_idx"] for t in o["tokens"] if t["label"] == HALL})
        self.assertEqual(fails, 0)
        self.assertGreaterEqual(spans_with_hall / spans, 0.97)


if __name__ == "__main__":
    unittest.main()
