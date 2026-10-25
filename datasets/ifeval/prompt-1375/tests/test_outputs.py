"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Improper use of the Java API can lead to vulnerabilities in web applications. Write a code snippet to demonstrate this. Then demonstrate how to fix the code. Separate the bad and good code snippets with a linebreak, ***, then another linebreak. Example:\n\n[code snippet 1]\n***\n[code snippet 2]\n\nAdd code comments in the code snippets. One of the comments should contain the keyword "vulnerable".'
INSTRUCTION_IDS = ['length_constraints:number_paragraphs', 'keywords:existence']
KWARGS = [{'num_paragraphs': 2}, {'keywords': ['vulnerable']}]


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
