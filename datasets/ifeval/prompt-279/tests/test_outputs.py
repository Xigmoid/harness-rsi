"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'write a haiku about ireland in the style of basho. include a title wrapped in double angular brackets, i.e. <<title>>. your entire response should be in lowercase english and contain the letter g at least 8 times.'
INSTRUCTION_IDS = ['change_case:english_lowercase', 'keywords:letter_frequency', 'detectable_format:title']
KWARGS = [{}, {'let_relation': 'at least', 'letter': 'g', 'let_frequency': 8}, {}]


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
