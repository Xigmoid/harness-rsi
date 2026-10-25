"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'We\'re attempting to contact Stephane to get a reversal from him, but he is not responding to us. Could you write this in a way that would seem more polite to moms? Please use the key "polite" to put your answer. Wrap your entire response with double quotation marks, and include two sections: "SECTION 1" and "SECTION 2".'
INSTRUCTION_IDS = ['startend:quotation', 'detectable_format:multiple_sections']
KWARGS = [{}, {'section_spliter': 'SECTION', 'num_sections': 2}]


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
