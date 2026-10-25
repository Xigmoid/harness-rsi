"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a poem about a lonely Hue. The poem should be written for teenagers. In your poem, italicize at least one section in markdown, i.e *this is an italic text*, and include the word "singles" at least twice.'
INSTRUCTION_IDS = ['detectable_format:number_highlighted_sections', 'keywords:frequency']
KWARGS = [{'num_highlights': 1}, {'relation': 'at least', 'keyword': 'singles', 'frequency': 2}]


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
