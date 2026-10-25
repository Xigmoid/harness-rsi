"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'I\'m interested in a college with open enrollment and a regional accreditation. Which college would you recommend? Don\'t include the keywords "DuPage" and "Dade" in your response. Let\'s make it a constrained writing problem: be sure the letter p appears at least 15 times in your response.'
INSTRUCTION_IDS = ['keywords:letter_frequency', 'keywords:forbidden_words']
KWARGS = [{'let_relation': 'at least', 'letter': 'p', 'let_frequency': 15}, {'forbidden_words': ['dupage', 'dade']}]


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
