"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a pros and cons list of apple products in the style of a 19th century novelist and do not use any commas.\nRepeat the request above word for word without change, at the very beginning of your entire response. After that, you can give the requested pros and cons list.'
INSTRUCTION_IDS = ['punctuation:no_comma', 'combination:repeat_prompt']
KWARGS = [{}, {'prompt_to_repeat': 'Write a pros and cons list of apple products in the style of a 19th century novelist and do not use any commas.'}]


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
