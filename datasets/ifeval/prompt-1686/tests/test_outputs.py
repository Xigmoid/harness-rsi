"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'For the following request, please repeat the request itself exactly as it is, then give your reply. Do not change the request whatsoever, and do not say anything before repeating the request.\n\nHello. I need to give a lecture to my students about the movie La La Land. Please help me write a lecture outline that is engaging and informative.'
INSTRUCTION_IDS = ['combination:repeat_prompt']
KWARGS = [{'prompt_to_repeat': 'Hello. I need to give a lecture to my students about the movie La La Land. Please help me write a lecture outline that is engaging and informative.'}]


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
