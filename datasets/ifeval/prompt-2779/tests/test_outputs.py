"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Make a list of ways to say the following sentence more professionally: "Hey bautista, let me know if you need any help with the graphs." Also, make sure the letter q appears less than 5 times in your response.'
INSTRUCTION_IDS = ['keywords:letter_frequency']
KWARGS = [{'let_relation': 'less than', 'letter': 'q', 'let_frequency': 5}]


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
