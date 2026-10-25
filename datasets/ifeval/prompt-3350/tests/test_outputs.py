"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "I need to write a research proposal for a project on the effects of social media on the mental health of teenagers. Please only write short sentences, and don't use any commas in your entire response. Include at least five placeholder represented by square brackets, such as [website]."
INSTRUCTION_IDS = ['punctuation:no_comma', 'detectable_content:number_placeholders']
KWARGS = [{}, {'num_placeholders': 5}]


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
