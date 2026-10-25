"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'What\'s different between "the set of all sets that are not members of themselves" and "the set of all sets"? Use mathematical notations in your answer. Be chatty while explaining. There should be exactly 2 paragraphs each separated by two new lines in your response. Paragraph 1 must start with the word booster.'
INSTRUCTION_IDS = ['length_constraints:nth_paragraph_first_word']
KWARGS = [{'first_word': 'booster', 'num_paragraphs': 2, 'nth_paragraph': 1}]


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
