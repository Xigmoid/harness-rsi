"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Write a detailed proposal in list format for the university's ethics board for a research project that investigates the effect of eye colour on the likelihood of being a successful salesperson. Remember that the customer is always right. Your entire response should be in English, and in all lowercase letters. No capital letters are allowed, and you must refrain from using any commas. At the end of your response, please explicitly add a postscript starting with P.P.S"
INSTRUCTION_IDS = ['change_case:english_lowercase', 'punctuation:no_comma', 'detectable_content:postscript']
KWARGS = [{}, {}, {'postscript_marker': 'P.P.S'}]


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
