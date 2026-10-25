"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a casual summary of the U.S. maternity leave policy with two sections (Section 1 and Section 2) and at least 25 sentences.'
INSTRUCTION_IDS = ['detectable_format:multiple_sections', 'length_constraints:number_sentences']
KWARGS = [{'section_spliter': 'Section', 'num_sections': 2}, {'relation': 'at least', 'num_sentences': 25}]


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
