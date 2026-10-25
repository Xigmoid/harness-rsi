"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "The Broncos have been so bad in the NRL this year, they're getting dust on the trophy cabinet. Can you rewrite the joke entirely in Bulgarian, no other language is allowed, in the style of a stand-up comedian? Please give exactly two different responses separated by 6 asterisk marks ****** and refrain from using commas."
INSTRUCTION_IDS = ['combination:two_responses', 'language:response_language', 'punctuation:no_comma']
KWARGS = [{}, {'language': 'bg'}, {}]


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
