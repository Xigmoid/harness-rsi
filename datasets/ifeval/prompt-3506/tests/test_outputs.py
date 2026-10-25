"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a proposal for a research project that will determine whether pupils who have been exposed to a fast-paced environment are more likely to develop ADHD. Please wrap the entire output in JSON format. You can use markdown ticks like\n\n```JSON\n[json content]\n```\n\nPlease do not include the words "proposal" or "project" in the response.'
INSTRUCTION_IDS = ['detectable_format:json_format', 'keywords:forbidden_words']
KWARGS = [{}, {'forbidden_words': ['proposal', 'project']}]


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
