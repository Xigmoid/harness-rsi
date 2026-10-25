"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'A colt is 5 feet tall. It will grow 6 inches every month. How tall will it be in 3 years? Think step-by-step, then give your answer. Separate your thinking and the final answer by a line with just three "*" symbols: ***\nAt the end of your response, please explicitly add a postscript starting with P.P.S'
INSTRUCTION_IDS = ['length_constraints:number_paragraphs', 'detectable_content:postscript']
KWARGS = [{'num_paragraphs': 2}, {'postscript_marker': 'P.P.S'}]


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
