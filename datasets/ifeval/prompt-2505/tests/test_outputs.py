"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Improve the following text, which is about how to learn a language. Also, provide two alternatives. The text is: "The best way to learn about a new culture is by living in it. Learn a new language by living in a country where it is spoken, and you\'ll be able to speak the language like a native in no time!". Finish your response with "Is there anything else I can help with?". No other words should follow this phrase.'
INSTRUCTION_IDS = ['startend:end_checker']
KWARGS = [{'end_phrase': 'Is there anything else I can help with?'}]


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
