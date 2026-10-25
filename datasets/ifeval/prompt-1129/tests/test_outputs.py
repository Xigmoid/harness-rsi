"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a short startup pitch for a new kind of ice cream called "Sunnis ice cream". The ice cream should be gentle on the stomach. Contain 6 or more exclamation marks "!" in your response.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)'
INSTRUCTION_IDS = ['keywords:letter_frequency', 'combination:repeat_prompt']
KWARGS = [{'let_relation': 'at least', 'letter': '!', 'let_frequency': 6}, {'prompt_to_repeat': 'Write a short startup pitch for a new kind of ice cream called "Sunnis ice cream". The ice cream should be gentle on the stomach. Contain 6 or more exclamation marks "!" in your response.'}]


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
