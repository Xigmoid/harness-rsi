"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a very short resume for a refinery operator who has 5 years of experience working in the chemical industry. Include the keywords "friends" and "hanson" in the resume. Make your entire output contain less than 50 words.'
INSTRUCTION_IDS = ['keywords:existence', 'length_constraints:number_words']
KWARGS = [{'keywords': ['friends', 'hanson']}, {'relation': 'less than', 'num_words': 50}]


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
