import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from halluc.judge_models import lynx_prompt, parse_lynx, split_sentences  # noqa: E402


def test_parse_lynx_reads_only_the_score_field():
    ok = '{"REASONING": ["the answer would FAIL if it said 70 million"], "SCORE": "PASS"}'
    assert parse_lynx(ok) == "PASS"
    assert parse_lynx('{"REASONING": ["adds new info"], "SCORE": "FAIL"}') == "FAIL"
    assert parse_lynx('{"REASONING": ["x"], "SCORE": FAIL}') == "FAIL"
    assert parse_lynx('{"REASONING": ["cut off mid-sentence and no score') is None
    assert parse_lynx("") is None


def test_lynx_prompt_survives_braces_in_the_text():
    p = lynx_prompt("q?", "doc with {braces} and {{double}}", 'answer {"a": 1}')
    assert "doc with {braces} and {{double}}" in p and 'answer {"a": 1}' in p
    assert p.rstrip().endswith('"SCORE": <your final score>}')


def test_split_sentences_merges_list_markers_and_respects_lines():
    s = split_sentences("1. Boil water.\n2. Add eggs. Wait ten minutes.")
    assert s == ["1. Boil water.", "2. Add eggs.", "Wait ten minutes."]
    assert split_sentences("") == []
    assert split_sentences("Paris is the capital. It is big.") == ["Paris is the capital.", "It is big."]
