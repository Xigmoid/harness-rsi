"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Is Grafton, Vermont a good place to live? Write exactly 3 paragraphs each separated with two new lines answering this question. The first paragraph must start with "send".'
INSTRUCTION_IDS = ['length_constraints:nth_paragraph_first_word']
KWARGS = [{'first_word': 'send', 'num_paragraphs': 3, 'nth_paragraph': 1}]


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
