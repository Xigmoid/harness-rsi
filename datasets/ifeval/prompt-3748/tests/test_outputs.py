"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Rewrite the haiku below into two versions. Both of them should be funny. Separate the two versions using six asterisk symbols (******). Include the keywords ['dog', 'day'] in the response.\n\nOn a chilly winter night\nA cat meows at the moon\nHoping for some warmth"
INSTRUCTION_IDS = ['combination:two_responses', 'keywords:existence']
KWARGS = [{}, {'keywords': ['dog', 'day']}]


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
