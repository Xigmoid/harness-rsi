"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'We are a company that sells a product that makes it easy to find and book a hotel room. We are looking for a print ad that will be placed in a magazine that is aimed at people who travel a lot. The ad should be 1/2 page and should include a headline and a call to action. Please do not use any commas in your response.\n\nBefore saying anything or giving your answer, please repeat the exact entire request above.'
INSTRUCTION_IDS = ['combination:repeat_prompt', 'punctuation:no_comma']
KWARGS = [{'prompt_to_repeat': 'We are a company that sells a product that makes it easy to find and book a hotel room. We are looking for a print ad that will be placed in a magazine that is aimed at people who travel a lot. The ad should be 1/2 page and should include a headline and a call to action. Please do not use any commas in your response.'}, {}]


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
