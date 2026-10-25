"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Who won the defamation case between Amber Heard and Johnny Depp? Write your answer as if you are writing to a group of elderly people. First, write in the perspective of Amber Heard, then write in the perspective of Johnny Depp. Separate those two version by 6 asterisk symbols ******. The entire response should have less than 300 words.'
INSTRUCTION_IDS = ['combination:two_responses', 'length_constraints:number_words']
KWARGS = [{}, {'relation': 'less than', 'num_words': 300}]


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
