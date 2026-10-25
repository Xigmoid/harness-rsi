"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'The hull of a ship is severely damaged in a storm. The ship has craters and some of its outer shell has been peeled off. How can I repair the hull? Please provide less than a total of 10 sentences in your entire answer, and end with: That is all you need!'
INSTRUCTION_IDS = ['startend:end_checker', 'length_constraints:number_sentences']
KWARGS = [{'end_phrase': 'That is all you need!'}, {'relation': 'less than', 'num_sentences': 10}]


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
