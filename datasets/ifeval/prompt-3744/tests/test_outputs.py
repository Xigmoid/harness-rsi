"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Give me a TLDR on the 2022 Google IO announcements, but the response must be entirely in the Bengali language, no other language is allowed, and have at least one placeholder such as [পণ্য].'
INSTRUCTION_IDS = ['detectable_content:number_placeholders', 'language:response_language']
KWARGS = [{'num_placeholders': 1}, {'language': 'bn'}]


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
