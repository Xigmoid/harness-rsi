"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a haiku about foolish behavior in the form of a question, for an audience of young readers. It should include the topic of not studying. Give exactly two different responses, separated by 6 asterisk symbols (******), and include a title wrapped in double angular brackets, i.e. <<title>>. Do not use commas.'
INSTRUCTION_IDS = ['combination:two_responses', 'detectable_format:title', 'punctuation:no_comma']
KWARGS = [{}, {}, {}]


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
