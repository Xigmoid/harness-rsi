"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Create a 500-word startup pitch for a new startup that will help people to find the best places to live. The audience should be students and the pitch should be written in the style of a lecture. Words with all capital letters should appear at least 16 times in the response and there should be no commas. The word batted should appear less than 2 times.'
INSTRUCTION_IDS = ['change_case:capital_word_frequency', 'punctuation:no_comma', 'keywords:frequency']
KWARGS = [{'capital_relation': 'at least', 'capital_frequency': 16}, {}, {'relation': 'less than', 'keyword': 'batted', 'frequency': 2}]


def test_response_follows_every_instruction():
    try:
        with open("/app/response.txt", encoding="utf-8") as handle:
            response = handle.read()
    except FileNotFoundError:
        raise AssertionError("/app/response.txt was not written by the agent")
    assert response.strip(), "/app/response.txt is empty"
    results = check_prompt(PROMPT, response, INSTRUCTION_IDS, KWARGS)
    failed = [k for k, ok in results.items() if not ok]
    assert not failed, f"instructions not followed: {failed} (all: {results})"
