"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Can you write me a slogan for my video game project? The project aims to create a very funny fighting game. Please wrap your entire response with double quotation marks. Also, highlight at least three sections in your answer in markdown format using *highlighted text*. Such as: *Funny Fighting*.'
INSTRUCTION_IDS = ['startend:quotation', 'detectable_format:number_highlighted_sections']
KWARGS = [{}, {'num_highlights': 3}]


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
