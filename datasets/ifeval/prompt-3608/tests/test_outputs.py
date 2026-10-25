"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a cover letter for a job application to a company which perhaps has a bad reputation. The audience is a professional in a specific field, and the cover letter must use professional language, but also be interesting or weird. The letter j should appear at least 20 times. Your entire response should be in English, and lowercase letters. No capital letters are allowed.'
INSTRUCTION_IDS = ['keywords:letter_frequency', 'change_case:english_lowercase']
KWARGS = [{'let_relation': 'at least', 'letter': 'j', 'let_frequency': 20}, {}]


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
