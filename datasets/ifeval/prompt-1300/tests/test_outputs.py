"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a short proposal for a new research project that investigates how language evolves over time. I want to make it challenging, so:\n1. Do not include any commas in your response.\n2. Do not include the letter "c" anywhere in your response.\n3. Your response should contain at least 250 words.'
INSTRUCTION_IDS = ['punctuation:no_comma', 'keywords:letter_frequency', 'length_constraints:number_words']
KWARGS = [{}, {'let_relation': 'less than', 'letter': 'c', 'let_frequency': 1}, {'relation': 'at least', 'num_words': 250}]


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
