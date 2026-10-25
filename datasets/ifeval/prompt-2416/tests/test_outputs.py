"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a one week itinerary for a trip to the United States with a focus on overcoming the challenges faced by the country. Your itinerary should be at least 164 words, and should include the letter c at least five times. Your entire response should be in English, and in all capital letters.'
INSTRUCTION_IDS = ['length_constraints:number_words', 'keywords:letter_frequency', 'change_case:english_capital']
KWARGS = [{'relation': 'at least', 'num_words': 164}, {'let_relation': 'at least', 'letter': 'c', 'let_frequency': 5}, {}]


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
