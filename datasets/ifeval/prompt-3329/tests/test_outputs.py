"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Hey! I need a rubric for evaluating the performance and price of a laptop. Can you create an example for me? Do not give me any bullet points, lists, or tables. Just write the rubric in plain English paragraphs. I'd like your response to be at least 30 sentences long."
INSTRUCTION_IDS = ['length_constraints:number_sentences']
KWARGS = [{'relation': 'at least', 'num_sentences': 30}]


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
