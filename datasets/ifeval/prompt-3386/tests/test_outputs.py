"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Jennifer goes to the store to buy milk. She has 10 dollars in her pocket and milk costs 3 dollars per gallon. How many gallons of milk can she buy? Explain your thinking. Avoid the keywords: 'divide', 'answer'. Include the keyword 'remainder'."
INSTRUCTION_IDS = ['keywords:forbidden_words', 'keywords:existence']
KWARGS = [{'forbidden_words': ['divide', 'answer']}, {'keywords': ['remainder']}]


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
