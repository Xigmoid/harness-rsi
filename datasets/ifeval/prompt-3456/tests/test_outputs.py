"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Generate a list of 100 random names. Make sure that no name is repeated and every name is unique. All letters in your entire response should be capitalized. Italicize 5 of your favorite names. For example:\n1. *FAVORITE NAME 1*\n2. *FAVORITE NAME 2*\n3. ...'
INSTRUCTION_IDS = ['detectable_format:number_highlighted_sections', 'change_case:english_capital']
KWARGS = [{'num_highlights': 5}, {}]


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
