"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Name a new fashion company that young people might like, and give it a name with multiple meanings. Put the name in double angular brackets, such as <<name>>.\n\nLet's repeat the request above first, before you say anything or really respond to the request."
INSTRUCTION_IDS = ['combination:repeat_prompt', 'detectable_format:title']
KWARGS = [{'prompt_to_repeat': 'Name a new fashion company that young people might like, and give it a name with multiple meanings. Put the name in double angular brackets, such as <<name>>.'}, {}]


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
