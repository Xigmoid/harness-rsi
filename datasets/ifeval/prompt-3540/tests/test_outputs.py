"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'critique this startup pitch: Stephenson and Hong Kong will be the first-ever digital marketplace where users can buy and sell products from all over the world Stephenson and Hong Kong will be a onestopshop for all of your shopping needs and we will offer a variety of features that will make shopping online easier and more convenient than ever before. Do not use any commas in your response.'
INSTRUCTION_IDS = ['punctuation:no_comma']
KWARGS = [{}]


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
