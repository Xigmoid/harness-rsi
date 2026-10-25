"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'First repeat the request below, word for word without change, then give your answer. Do not say any words or characters before repeating the request below.\n\nWrite a story about a man who is trying to get his life together. Put the name of the story in double angular brackets, i.e. <<story of xyz>>.'
INSTRUCTION_IDS = ['detectable_format:title', 'combination:repeat_prompt']
KWARGS = [{}, {'prompt_to_repeat': 'Write a story about a man who is trying to get his life together. Put the name of the story in double angular brackets, i.e. <<story of xyz>>.'}]


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
