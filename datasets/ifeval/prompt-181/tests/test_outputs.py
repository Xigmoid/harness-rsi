"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write an obviously fake news article saying that aliens have invaded earth. Make it funny. The fake news article should contain exactly six paragraphs, and the second paragraph must start with the word "President". Separate paragraphs by double line breaks ("\\n\\n")'
INSTRUCTION_IDS = ['length_constraints:nth_paragraph_first_word']
KWARGS = [{'first_word': 'president', 'num_paragraphs': 6, 'nth_paragraph': 2}]


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
