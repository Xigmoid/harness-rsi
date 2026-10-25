"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Compose a poem that has the word "land" and "river". It should be about nature and love. Also, the word "forest" should appear at least 3 times, and be written in English, with all letters lowercased.'
INSTRUCTION_IDS = ['keywords:frequency', 'change_case:english_lowercase', 'keywords:existence']
KWARGS = [{'relation': 'at least', 'keyword': 'forest', 'frequency': 3}, {}, {'keywords': ['land', 'river']}]


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
