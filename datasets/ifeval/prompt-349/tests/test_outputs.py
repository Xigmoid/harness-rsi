"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'I want to start a garden for veggies and flowers in my small backyard. Can you give me some advice on how to water my plants? Have at least 3 italic text sections, such as: *italic text 1*, *italic text 2*, etc.\n\nMake your reply short -- the whole reply should contain less than 40 words.'
INSTRUCTION_IDS = ['detectable_format:number_highlighted_sections', 'length_constraints:number_words']
KWARGS = [{'num_highlights': 3}, {'relation': 'less than', 'num_words': 40}]


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
