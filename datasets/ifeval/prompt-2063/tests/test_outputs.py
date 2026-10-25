"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Titan makes clothing for large men. Write an advertisement for the company that would appeal to a wide audience. Make sentences short. Your response should not contain any comma.\nFirst repeat the request word for word without change, then give your answer.\nDo not say any words or characters before repeating the request. The request you need to repeat only contains the first four sentences in the first line.'
INSTRUCTION_IDS = ['combination:repeat_prompt', 'punctuation:no_comma']
KWARGS = [{'prompt_to_repeat': 'Titan makes clothing for large men. Write an advertisement for the company that would appeal to a wide audience. Make sentences short. Your response should not contain any comma.'}, {}]


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
