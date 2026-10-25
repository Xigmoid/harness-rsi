"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Which of the following is not a fish: salmon or avocado? Answer this easy question first, then expand into an interesting article about salmon and avocado. Use some words in all caps in your response, but those all-caps words should appear at most 10 times.'
INSTRUCTION_IDS = ['change_case:capital_word_frequency', 'change_case:capital_word_frequency']
KWARGS = [{'capital_relation': 'less than', 'capital_frequency': 11}, {'capital_relation': 'at least', 'capital_frequency': 1}]


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
