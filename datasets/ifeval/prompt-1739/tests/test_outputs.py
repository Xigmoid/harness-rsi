"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Rewrite the following sentence in only Vietnamese, no other language is allowed, and refrain from using commas: "We may be able to improve our model for the next year. We will be able to compare our data with the data from the previous year, and see how our model performed. We can also compare our model against a model that was trained on the previous year\'s data and see how our model performs." No other language except Vietnamese is allowed to be used in your response.'
INSTRUCTION_IDS = ['punctuation:no_comma', 'language:response_language']
KWARGS = [{}, {'language': 'vi'}]


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
