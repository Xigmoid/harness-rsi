"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Hi, I'm looking for two Tamil movies to watch. Please recommend exactly two movies. Separated them by 6 asterisk symbols, like below:\n[Movie 1]\n[Description]\n******\n[Movie 2]\n[Description]\n\nYour entire response should be entirely in Tamil, no other language is allowed, and should not contain any commas. "
INSTRUCTION_IDS = ['combination:two_responses', 'language:response_language', 'punctuation:no_comma']
KWARGS = [{}, {'language': 'ta'}, {}]


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
