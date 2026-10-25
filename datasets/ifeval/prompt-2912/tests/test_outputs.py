"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Create a dialogue between two people who are both trying to avoid using the letter t. But somehow they ended up using a lot of t in their dialogue. Break the dialogue into two scenes, separated by 6 asterisk symbols: ******. The letter t must appear at least 30 times in the dialogue you write.'
INSTRUCTION_IDS = ['combination:two_responses', 'keywords:letter_frequency']
KWARGS = [{}, {'let_relation': 'at least', 'letter': 't', 'let_frequency': 30}]


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
