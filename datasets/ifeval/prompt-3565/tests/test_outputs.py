"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'What does the word "jock" mean to you? Please generate an answer with two parts. The two parts should be separated by 3 asterisks \'***\'. Also, reply without mentioning the word "jock" throughout.'
INSTRUCTION_IDS = ['keywords:forbidden_words', 'length_constraints:number_paragraphs']
KWARGS = [{'forbidden_words': ['jock']}, {'num_paragraphs': 2}]


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
