"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a casual blog post about how the outer solar system is different from the inner solar system, and what that means for the possibility of life. Wrap your entire response with double quotation marks. Your response should contain 17 or more sentences.'
INSTRUCTION_IDS = ['startend:quotation', 'length_constraints:number_sentences']
KWARGS = [{}, {'relation': 'at least', 'num_sentences': 17}]


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
