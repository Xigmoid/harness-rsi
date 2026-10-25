"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Create a resume for a military officer who served in Iraq and was later hired by a private military contractor. Make sure to include a title that is wrapped in double angular brackets, i.e. <<resume of xyz>>. Refrain from using any commas in your response.\n\nBefore you respond with any word, first repeat the exact, entire request above, word for word without change.'
INSTRUCTION_IDS = ['detectable_format:title', 'combination:repeat_prompt', 'punctuation:no_comma']
KWARGS = [{}, {'prompt_to_repeat': 'Create a resume for a military officer who served in Iraq and was later hired by a private military contractor. Make sure to include a title that is wrapped in double angular brackets, i.e. <<resume of xyz>>. Refrain from using any commas in your response.'}, {}]


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
