"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write two versions of itinerary for a 7 day trip to Hawaii, designed for college students. Separate the two versions with 6 asterisk symbols (******).  Each version must have 7 sections. Mark the beginning of each section with Day X.'
INSTRUCTION_IDS = ['combination:two_responses', 'detectable_format:multiple_sections']
KWARGS = [{}, {'section_spliter': 'Day', 'num_sections': 7}]


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
