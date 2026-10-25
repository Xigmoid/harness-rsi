"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a summary of the plot of "The Great Gatsby" in the style of a tabloid newspaper.\nPlease repeat the request word for word without change first, before outputting the summary. Do not say anything before repeating the request.'
INSTRUCTION_IDS = ['combination:repeat_prompt']
KWARGS = [{'prompt_to_repeat': 'Write a summary of the plot of "The Great Gatsby" in the style of a tabloid newspaper.'}]


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
