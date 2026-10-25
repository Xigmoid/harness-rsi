"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Give me 5 Q and As, following the following format:\n\n"\nQ & A # 1\n***\nQ & A # 2\n***\nQ & A # 3\n***\nQ & A # 4\n***\nQ & A # 5\n"\n\nWrap your entire response with double quotation marks.'
INSTRUCTION_IDS = ['length_constraints:number_paragraphs', 'startend:quotation']
KWARGS = [{'num_paragraphs': 5}, {}]


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
