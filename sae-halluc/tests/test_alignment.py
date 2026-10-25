import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from halluc.alignment import (  # noqa: E402
    HALL, AMBIG, FAITHFUL, BOUNDARY, NEAR, PROMPT, SPECIAL, SPACE,
    label_tokens, check_span_coverage, merge_intervals,
)
from fake_tokenizer import FakeTokenizer  # noqa: E402

PROMPT_TEXT = "Question: where? output:"
RESPONSE = "The cat sat on the mat today."


def build(tokenizer, response=RESPONSE, sep=" "):
    text = PROMPT_TEXT + sep + response
    resp_start = len(PROMPT_TEXT) + len(sep)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    return text, resp_start, resp_start + len(response), [tuple(o) for o in enc["offset_mapping"]]


def span_of(sub, response=RESPONSE):
    s = response.index(sub)
    return (s, s + len(sub), "Evident Baseless Info")


def labels_by_text(text, offsets, labels):
    return [(text[s:e].strip(), l.label) for (s, e), l in zip(offsets, labels)]


class AlignmentTests(unittest.TestCase):
    def test_basic_labels(self):
        tok = FakeTokenizer("word_untrimmed")
        text, rs, re_, off = build(tok)
        labs = label_tokens(off, rs, re_, [span_of("sat on the mat")], text=text)
        got = dict()
        for (s, e), l in zip(off, labs):
            got.setdefault(text[s:e].strip(), []).append(l.label)
        self.assertEqual(labs[0].label, SPECIAL)               # BOS
        self.assertEqual(got["Question"], [PROMPT])
        self.assertEqual(got["output"], [PROMPT])
        self.assertEqual(got[":"], [PROMPT, PROMPT])           # "Question:" and "output:"
        self.assertEqual(got["The"], [FAITHFUL])
        self.assertEqual(got["cat"], [FAITHFUL])
        for w in ("sat", "on", "mat"):
            self.assertEqual(got[w], [HALL], w)
        self.assertEqual(got["the"], [HALL])                    # inside span ("the mat")
        self.assertEqual(got["today"], [FAITHFUL])
        self.assertEqual(got["."], [FAITHFUL])                  # period is outside the span

    def test_offset_convention_does_not_matter(self):
        spans = [span_of("sat on the mat")]
        results = []
        for mode in ("word_untrimmed", "word_trimmed"):
            tok = FakeTokenizer(mode)
            text, rs, re_, off = build(tok)
            labs = label_tokens(off, rs, re_, spans, text=text)
            results.append([l.label for l in labs])
        self.assertEqual(results[0], results[1])

    def test_leading_space_token_adjacent_to_span_is_not_hall(self):
        # span starts exactly at "sat": the token " sat" has its leading space OUTSIDE the span
        tok = FakeTokenizer("word_untrimmed")
        text, rs, re_, off = build(tok)
        labs = label_tokens(off, rs, re_, [span_of("sat")], text=text)
        d = {text[s:e].strip(): l.label for (s, e), l in zip(off, labs)}
        self.assertEqual(d["sat"], HALL)
        self.assertEqual(d["cat"], FAITHFUL)
        self.assertEqual(d["on"], FAITHFUL)

    def test_boundary_tokens_with_subword_split(self):
        tok = FakeTokenizer("chunk", chunk=3)
        resp = "Internationalization matters"
        text, rs, re_, off = build(tok, resp)
        # span covers "national" which cuts through the 3-char pieces "Int","ern","ati","ona","liz","ati","on"
        sp = span_of("national", resp)
        labs = label_tokens(off, rs, re_, [sp], text=text)
        kinds = {l.label for l in labs if l.in_response}
        self.assertIn(BOUNDARY, kinds)       # a piece straddles the span edge
        self.assertIn(HALL, kinds)
        self.assertIn(FAITHFUL, kinds)

    def test_ambiguous_only_when_no_strict_overlap(self):
        tok = FakeTokenizer("word_trimmed")
        text, rs, re_, off = build(tok)
        labs = label_tokens(off, rs, re_, [span_of("mat")], [span_of("cat sat"), span_of("mat")],
                            text=text)
        d = {text[s:e].strip(): l.label for (s, e), l in zip(off, labs)}
        self.assertEqual(d["cat"], AMBIG)
        self.assertEqual(d["sat"], AMBIG)
        self.assertEqual(d["mat"], HALL)     # strict beats ambiguous

    def test_overlapping_strict_spans_union_and_span_idx(self):
        tok = FakeTokenizer("word_trimmed")
        text, rs, re_, off = build(tok)
        a = span_of("cat sat on")
        b = span_of("on the mat")
        labs = label_tokens(off, rs, re_, [a, b], text=text)
        d = {text[s:e].strip(): l for (s, e), l in zip(off, labs)}
        for w in ("cat", "sat", "on", "the", "mat"):
            self.assertEqual(d[w].label, HALL)
        self.assertEqual(d["cat"].span_idx, 0)
        self.assertEqual(d["mat"].span_idx, 1)
        self.assertEqual(d["The"].label, FAITHFUL)
        self.assertEqual(check_span_coverage(off, labs, text, rs, [a, b]), [])

    def test_margin_marks_adjacent_faithful_tokens_near(self):
        tok = FakeTokenizer("word_trimmed")
        text, rs, re_, off = build(tok)
        labs = label_tokens(off, rs, re_, [span_of("sat")], text=text, margin_tokens=1)
        d = {text[s:e].strip(): l.label for (s, e), l in zip(off, labs)}
        self.assertEqual(d["cat"], NEAR)
        self.assertEqual(d["on"], NEAR)
        self.assertEqual(d["The"], FAITHFUL)
        self.assertEqual(d["today"], FAITHFUL)

    def test_whitespace_only_response_token_is_space(self):
        tok = FakeTokenizer("word_untrimmed")
        resp = "One\n\nTwo"
        text, rs, re_, off = build(tok, resp)
        labs = label_tokens(off, rs, re_, [], text=text)
        kinds = [l.label for l in labs if l.in_response]
        self.assertTrue(all(k in (FAITHFUL, SPACE) for k in kinds))

    def test_coverage_check_catches_corrupt_offsets(self):
        tok = FakeTokenizer("word_trimmed")
        text, rs, re_, off = build(tok)
        sp = span_of("sat on")
        labs = label_tokens(off, rs, re_, [sp], text=text)
        self.assertEqual(check_span_coverage(off, labs, text, rs, [sp]), [])
        # Corrupt: drop every token that overlaps the span
        bad_off = [(0, 0) if l.label == HALL else o for o, l in zip(off, labs)]
        bad_labs = label_tokens(bad_off, rs, re_, [sp], text=text)
        fails = check_span_coverage(bad_off, bad_labs, text, rs, [sp])
        self.assertEqual([k for k, _ in fails], [0])

    def test_merge_intervals(self):
        self.assertEqual(merge_intervals([(5, 9), (0, 3), (2, 6)]), [(0, 9)])
        self.assertEqual(merge_intervals([(0, 2), (4, 6)]), [(0, 2), (4, 6)])


if __name__ == "__main__":
    unittest.main()
