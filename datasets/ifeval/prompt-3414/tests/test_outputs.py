"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Can you elaborate on "I froze when I was jogging"? Include some words in all capital letters. In particular, there should be 5 to 10 such capitalized words.'
INSTRUCTION_IDS = ['change_case:capital_word_frequency', 'change_case:capital_word_frequency']
KWARGS = [{'capital_relation': 'at least', 'capital_frequency': 5}, {'capital_relation': 'less than', 'capital_frequency': 11}]


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
