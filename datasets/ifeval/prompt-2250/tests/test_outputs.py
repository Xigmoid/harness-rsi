"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write exactly 4 paragraphs about tips for installing a car seat for moms. Use 2 new lines to separate paragraphs. Start the 4th paragraph with the word "elm".'
INSTRUCTION_IDS = ['length_constraints:nth_paragraph_first_word']
KWARGS = [{'first_word': 'elm', 'num_paragraphs': 4, 'nth_paragraph': 4}]


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
