import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from halluc.data import (  # noqa: E402
    load_qa_items, item_category, build_no_rag_prompt, DEFAULT_DATASET_DIR,
)
from halluc.pipeline import process_item, rebase_to_no_rag  # noqa: E402
from halluc.sampling import sample_matched_tokens  # noqa: E402
from fake_tokenizer import FakeTokenizer  # noqa: E402

HAVE_DATA = (DEFAULT_DATASET_DIR / "response.jsonl").exists()


@unittest.skipUnless(HAVE_DATA, "RAGTruth dataset not found")
class NoRagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        items = load_qa_items()
        cls.hall = [i for i in items if item_category(i) == "hallucinated"][:80]

    def test_prompt_has_no_passages(self):
        for it in self.hall[:30]:
            p = build_no_rag_prompt(it)
            self.assertNotIn(it["passages"][:40], p)
            self.assertTrue(p.endswith("output:"))
            self.assertIn(it["question"].strip(), p)

    def test_response_tokens_identical_and_indices_consistent(self):
        tok = FakeTokenizer("word_untrimmed")
        for it in self.hall:
            rag = process_item(tok, it, margin_tokens=2)
            no = rebase_to_no_rag(rag, tok, it)
            n = len(rag["tokens"])
            # same response ids at the tail, shorter prompt
            self.assertEqual(rag["input_ids"][-n:], no["input_ids"][-n:])
            self.assertLess(no["n_prompt_tokens"], rag["n_prompt_tokens"])
            self.assertEqual(len(no["input_ids"]), no["n_prompt_tokens"] + n)
            for a, b in zip(rag["tokens"], no["tokens"]):
                self.assertEqual(a["label"], b["label"])
                self.assertEqual(a["passage_status"], b["passage_status"])
                self.assertEqual(a["rel_pos"], b["rel_pos"])
                self.assertEqual(no["input_ids"][b["seq_idx"]], rag["input_ids"][a["seq_idx"]])
                self.assertEqual(b["predict_idx"], b["seq_idx"] - 1)

    def test_sampling_is_identical_across_conditions(self):
        tok = FakeTokenizer("word_untrimmed")
        for it in self.hall:
            rag = process_item(tok, it, margin_tokens=2)
            no = rebase_to_no_rag(rag, tok, it)
            p1, u1 = sample_matched_tokens(rag["tokens"], random.Random(7), 8, 5)
            p2, u2 = sample_matched_tokens(no["tokens"], random.Random(7), 8, 5)
            self.assertEqual(u1, u2)
            self.assertEqual([(p["pair_id"], p["tier"], p["hall"]["resp_pos"], p["faithful"]["resp_pos"]) for p in p1],
                             [(p["pair_id"], p["tier"], p["hall"]["resp_pos"], p["faithful"]["resp_pos"]) for p in p2])


@unittest.skipUnless(HAVE_DATA, "RAGTruth dataset not found")
class RealTokenizerNoRagTests(unittest.TestCase):
    def test_rebased_ids_equal_joint_tokenisation(self):
        """Prompt-only tokens + reused response ids must equal tokenising 'prompt + response' jointly."""
        name = os.environ.get("HALLUC_TOKENIZER")
        if not name:
            self.skipTest("set HALLUC_TOKENIZER (e.g. meta-llama/Llama-3.1-8B) to run")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # noqa: BLE001
            self.skipTest("cannot load tokenizer %s: %s" % (name, exc))
        items = [i for i in load_qa_items() if item_category(i) == "hallucinated"][:300]
        bad = 0
        for it in items:
            no = rebase_to_no_rag(process_item(tok, it), tok, it)
            joint = tok(build_no_rag_prompt(it) + " " + it["response"])["input_ids"]
            bad += int(joint != no["input_ids"])
        self.assertEqual(bad, 0, "%d/%d items differ between rebased and joint tokenisation" % (bad, len(items)))


if __name__ == "__main__":
    unittest.main()
