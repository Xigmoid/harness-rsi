"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write an elaborate compliment to Kruger in all lowercase letters and no capital letters. The word hughes should appear less than 2 times. Your response should be at least 100 words.'
INSTRUCTION_IDS = ['keywords:frequency', 'change_case:english_lowercase', 'length_constraints:number_words']
KWARGS = [{'relation': 'less than', 'keyword': 'hughes', 'frequency': 2}, {}, {'relation': 'at least', 'num_words': 100}]


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
