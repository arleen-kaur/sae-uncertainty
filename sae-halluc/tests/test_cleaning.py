import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from halluc.cleaning import classify, clean_response  # noqa: E402

Q = "how to bake hard boiled egg"


def test_keeps_whole_list_cuts_after_blank_line():
    raw = " 1. Boil water.\n2. Add eggs.\n3. Wait ten minutes.\n\n2. 1. If you are not particular..."
    assert clean_response(raw) == "1. Boil water.\n2. Add eggs.\n3. Wait ten minutes."


def test_drops_task_restating_lines_but_keeps_the_answer():
    raw = "The following is the required output:\nThe following is the required output:\n1. Boil water.\n2. Add eggs."
    assert clean_response(raw) == "1. Boil water.\n2. Add eggs."


def test_trims_to_last_full_sentence_when_cut_by_token_limit():
    raw = " Paris is the capital of France. It lies on the Seine and has about two million peo"
    assert clean_response(raw) == "Paris is the capital of France."


def test_does_not_trim_when_it_ends_cleanly_or_text_follows():
    assert clean_response(" It is open daily.") == "It is open daily."
    raw = " It is open daily\n\nBriefly answer the following question: x"
    assert clean_response(raw) == "It is open daily"


def test_prompt_leaks_are_rejected_real_answers_are_not():
    q = "how to plan a trip to germany"
    assert classify("how to plan a trip to germany\npassage 1:Share. Budget Germany: Travel", q)[1] == "prompt_leak"
    assert classify("how to plan a trip to germany\nThe question is: how to plan a trip to germany.", q)[1] == "prompt_leak"
    q2 = "what is the proper way to dispose of a worn usa flag"
    leak = "Briefly answer the following question: " + q2 + ". Bear in mind that your response should be"
    assert classify(leak, q2)[1] == "prompt_leak"
    ok = "The proper way to dispose of a worn USA flag is to burn it respectfully."
    assert classify(ok, q2) == (ok, None)
    # an answer that merely mentions the topic is fine
    assert classify("Plan a trip to Germany by booking flights early.", q)[1] is None


def test_non_answers_get_a_reason():
    assert classify("", Q)[1] == "empty"
    assert classify("  \n\n", Q)[1] == "empty"
    assert classify(" how to bake hard boiled egg", Q)[1] == "echo"
    assert classify(" How to bake hard-boiled egg?", Q)[1] == "echo"
    assert classify(" Yes.", Q)[1] == "too_short"
    assert classify("The following is the required output:", Q)[1] == "empty"
    assert classify(" Boil the eggs for ten minutes.", Q) == ("Boil the eggs for ten minutes.", None)
