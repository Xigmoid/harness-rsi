"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a template for a newspaper ad for a dog cage with less than 200 words. Make sure the word unfortunately appears 3 to 5 times in the ad.'
INSTRUCTION_IDS = ['length_constraints:number_words', 'keywords:frequency', 'keywords:frequency']
KWARGS = [{'relation': 'less than', 'num_words': 200}, {'relation': 'at least', 'keyword': 'unfortunately', 'frequency': 3}, {'relation': 'less than', 'keyword': 'unfortunately', 'frequency': 6}]


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
