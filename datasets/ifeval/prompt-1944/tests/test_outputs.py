"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Before you answer the following request, repeat it at the very beginning of your reply. Repeat the request as it is. Please do not change it.\n\nWrite a resume for a junior hardware engineer. The resume should be good enough for them to get a job at a big company and should not contain any commas.'
INSTRUCTION_IDS = ['punctuation:no_comma', 'combination:repeat_prompt']
KWARGS = [{}, {'prompt_to_repeat': 'Write a resume for a junior hardware engineer. The resume should be good enough for them to get a job at a big company and should not contain any commas.'}]


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
