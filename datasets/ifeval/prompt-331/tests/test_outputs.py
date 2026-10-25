"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a song about a man who rents a room in a house with a bunch of other people, and he absolutely hated it. Your song should contain at least 10 words in all capital letters that are adjectives or verbs. Commas are not allowed in the song.'
INSTRUCTION_IDS = ['change_case:capital_word_frequency', 'punctuation:no_comma']
KWARGS = [{'capital_relation': 'at least', 'capital_frequency': 10}, {}]


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
