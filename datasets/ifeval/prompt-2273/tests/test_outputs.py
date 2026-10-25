"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = '11 results have an average of 51. The first five have an average of 49, and the last nine have an average of 52. What was the sixth result? Use weird language when explaining using mathematical notation. And add a postscript starting with P.P.S at the end.'
INSTRUCTION_IDS = ['detectable_content:postscript']
KWARGS = [{'postscript_marker': 'P.P.S'}]


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
