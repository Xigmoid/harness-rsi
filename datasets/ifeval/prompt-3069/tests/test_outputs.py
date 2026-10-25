"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Can you create an itinerary for a 5 day trip to switzerland that includes exactly 3 bullet points in markdown format, in all lowercase letters, and a postscript at the end starting with P.S.?'
INSTRUCTION_IDS = ['detectable_format:number_bullet_lists', 'change_case:english_lowercase', 'detectable_content:postscript']
KWARGS = [{'num_bullets': 3}, {}, {'postscript_marker': 'P.S.'}]


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
