"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Students are travelling to UCI for 3 days. Create a hilarious itinerary for them. Do not use the word 'university'. Your entire response should have exactly 4 paragraphs.  Separate paragraphs with the markdown divider: ***"
INSTRUCTION_IDS = ['length_constraints:number_paragraphs', 'keywords:forbidden_words']
KWARGS = [{'num_paragraphs': 4}, {'forbidden_words': ['university']}]


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
