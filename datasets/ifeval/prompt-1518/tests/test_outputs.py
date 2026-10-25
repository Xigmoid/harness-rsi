"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'I am a software engineer with 7 years of experience, and I am looking for a new job. Can you create a resume for me and explain each section?\n\nFirst repeat the exact request above, then give your answer. Do not say any word before repeating the exact request.'
INSTRUCTION_IDS = ['combination:repeat_prompt']
KWARGS = [{'prompt_to_repeat': 'I am a software engineer with 7 years of experience, and I am looking for a new job. Can you create a resume for me and explain each section?'}]


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
