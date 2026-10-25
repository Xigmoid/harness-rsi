"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'List exactly 10 possible names for a new baby boy. Then add a postscript starting with P.P.S to the end of your response. Put the names into bullet points that are indicated by:\n* This is an example bullet'
INSTRUCTION_IDS = ['detectable_content:postscript', 'detectable_format:number_bullet_lists']
KWARGS = [{'postscript_marker': 'P.P.S'}, {'num_bullets': 10}]


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
