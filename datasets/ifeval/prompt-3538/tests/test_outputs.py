"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a 600+ word social media post for a startup that provides a platform for running realistic physics simulation. Make sure to include the word "bombs" at least once.'
INSTRUCTION_IDS = ['keywords:frequency', 'length_constraints:number_words']
KWARGS = [{'relation': 'at least', 'keyword': 'bombs', 'frequency': 1}, {'relation': 'at least', 'num_words': 600}]


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
