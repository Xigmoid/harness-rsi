"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Describe how to proof Cauchy–Schwarz inequality. Make sure your ENTIRE response is in only the Marathi language, no other languages allowed. Do not write long equations. Your response must be in the following exact format, and use *** as the section separator:\n\n[Section 1: Description of Cauchy–Schwarz inequality, in Marathi language]\n***\n[Section 2: Describe how to proof Cauchy–Schwarz inequality, in Marathi language, without equations]\n***\n[Section 3: Brief summary, in Marathi language]'
INSTRUCTION_IDS = ['length_constraints:number_paragraphs', 'language:response_language']
KWARGS = [{'num_paragraphs': 3}, {'language': 'mr'}]


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
