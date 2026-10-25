"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Which one is a better brand for sneakers: Prada or Nike? Your entire response should be in English, and in all capital letters. At the end of your response, please explicitly add a postscript starting with P.S. The word sneaker should appear 10 or more times in your response.'
INSTRUCTION_IDS = ['change_case:english_capital', 'detectable_content:postscript', 'keywords:frequency']
KWARGS = [{}, {'postscript_marker': 'P.S.'}, {'relation': 'at least', 'keyword': 'sneaker', 'frequency': 10}]


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
