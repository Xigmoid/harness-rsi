"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Elaborate on the following sentence into a formal story: "My dog is brown, and my cat is black." Your answer must contain a title, wrapped in double angular brackets, i.e. <<title>>, and should not contain any commas. In your response, the word flesh should appear less than 3 times.'
INSTRUCTION_IDS = ['detectable_format:title', 'punctuation:no_comma', 'keywords:frequency']
KWARGS = [{}, {}, {'relation': 'less than', 'keyword': 'flesh', 'frequency': 3}]


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
