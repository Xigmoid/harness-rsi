"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'May name is Naomi. Write a blog post in my name for the canucks hockey team about why they need to be more mindful about their environments. End the blog post with "Naomi thanks you for reading." No other words should follow this phrase. This phrase should be the very end of your entire response.'
INSTRUCTION_IDS = ['startend:end_checker']
KWARGS = [{'end_phrase': 'Naomi thanks you for reading.'}]


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
