"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'In this task, repeat the exact request below first, then give your response. Do not say any word before repeating the exact request.\n\nWrite an acoustic song about the Korean peninsula without using any commas.'
INSTRUCTION_IDS = ['combination:repeat_prompt', 'punctuation:no_comma']
KWARGS = [{'prompt_to_repeat': 'Write an acoustic song about the Korean peninsula without using any commas.'}, {}]


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
