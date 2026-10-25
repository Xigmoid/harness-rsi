"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a funny, 150+ word ad for an attorney who helps poor people with their divorces. Highlight at least 3 text sections by italicize them with markdown (i.e. *highlighted section*).'
INSTRUCTION_IDS = ['detectable_format:number_highlighted_sections', 'length_constraints:number_words']
KWARGS = [{'num_highlights': 3}, {'relation': 'at least', 'num_words': 150}]


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
