"""End-to-end CPU test of the extraction path: tiny random Llama + real Llama-3 tokenizer + stub SAE.

Skipped unless torch + transformers are installed and $HALLUC_TOKENIZER names a loadable tokenizer
(e.g. NousResearch/Meta-Llama-3.1-8B, an ungated mirror of the Llama-3.1 tokenizer).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False

from halluc.data import DEFAULT_DATASET_DIR  # noqa: E402

HAVE_DATA = (DEFAULT_DATASET_DIR / "response.jsonl").exists()


class StubSAE:
    """Random ReLU encoder: enough to exercise indices/values handling."""

    def __init__(self, d_model, d_sae=96, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(d_model, d_sae, generator=g) / d_model ** 0.5
        self.b = -0.05 * torch.ones(d_sae)

    def encode(self, x):
        return torch.relu(x @ self.W + self.b)


@unittest.skipUnless(HAVE and HAVE_DATA and os.environ.get("HALLUC_TOKENIZER"),
                     "needs torch, transformers, RAGTruth data and HALLUC_TOKENIZER")
class TinyExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from halluc.data import load_qa_items
        from halluc.extract import select_items
        from halluc.splits import split_of, DISCOVERY
        cls.tok = AutoTokenizer.from_pretrained(os.environ["HALLUC_TOKENIZER"])
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=len(cls.tok), hidden_size=64, intermediate_size=128,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=4096)
        cls.model = LlamaForCausalLM(cfg).eval()
        cls.items = select_items(load_qa_items(), split_of, DISCOVERY, max_responses=3, seed=1)

    def test_phase1_and_phase2(self):
        from halluc.extract import extract_phase1, encode_cache, build_pickles, KINDS
        layers = [1, 3]
        records, cache, population, stats = extract_phase1(
            self.model, self.tok, self.model.model.layers, self.items, layers, "cpu",
            seed=0, log=lambda *a: None)

        n = len(records)
        self.assertGreater(n, 0)
        self.assertEqual(n % 2, 0)
        self.assertEqual(sum(r["role"] == "hall" for r in records), n // 2)
        for l in layers:
            for k in KINDS:
                self.assertEqual(tuple(cache[l][k].shape), (n, 64))
        # predict-position and self-position residuals differ (different tokens)
        self.assertFalse(torch.equal(cache[1]["predict"], cache[1]["self"]))
        for r in records:
            self.assertGreaterEqual(r["entropy"], 0.0)
            self.assertLessEqual(r["entropy"], 12.0)        # ln(128256) ~ 11.76
            self.assertGreaterEqual(r["surprisal"], 0.0)
            self.assertEqual(r["correct"], r["role"] == "faithful")
        # copy status matched within every pair
        by_pair = {}
        for r in records:
            by_pair.setdefault(r["pair_id"], {})[r["role"]] = r
        for pr in by_pair.values():
            self.assertEqual(pr["hall"]["ctx_status"], pr["faithful"]["ctx_status"])

        # population stats cover every response token of every processed response
        self.assertEqual(len(population), stats["n_responses"] - stats["n_skipped_no_pairs"])

        # captured residual really is the block output at the requested position
        from halluc.pipeline import process_item
        it = self.items[0]
        pipe = process_item(self.tok, it, margin_tokens=2)
        ids = torch.tensor([pipe["input_ids"]])
        with torch.no_grad():
            hs = self.model(input_ids=ids, output_hidden_states=True).hidden_states
        first = records[0]
        # hidden_states[l+1] is the output of block l
        ref = hs[1 + 1][0, first["predict_idx"]]
        self.assertTrue(torch.allclose(cache[1]["predict"][0].float(), ref, atol=1e-5))

        feats = encode_cache(cache, lambda l: StubSAE(64, seed=l), layers, "cpu", log=lambda *a: None)
        for k in KINDS:
            for l in layers:
                self.assertEqual(len(feats[k][l]), n)
                idx, val = feats[k][l][0]
                self.assertEqual(len(idx), len(val))
                self.assertTrue((val > 0).all())
        # encode_residuals agrees with a dense encode
        dense = StubSAE(64, seed=1).encode(cache[1]["predict"].float())
        idx, val = feats["predict"][1][0]
        self.assertEqual(sorted(idx.tolist()), dense[0].nonzero().squeeze(-1).tolist())

        payloads = build_pickles(records, feats, layers, {"model": "tiny"})
        for kind in KINDS:
            p = payloads[kind]
            self.assertEqual(p["config"]["sae_layers"], layers)
            self.assertEqual(len(p["per_question_data"]), n)
            d0 = p["per_question_data"][0]
            for key in ("correct", "entropy", "sae_features", "q_idx"):
                self.assertIn(key, d0)
            self.assertIn("indices", d0["sae_features"][1])


if __name__ == "__main__":
    unittest.main()
