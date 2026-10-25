"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Please write a summary of the following advertiser page: "We insist on having the most talented team of web developers, content writers, graphic designers and online marketers in the industry. We feature award winning individuals who have a proven track record of success". Use markdowns and target moms. Your answer must contain exactly 4 bullet points in markdown format and cannot contain commas.'
INSTRUCTION_IDS = ['detectable_format:number_bullet_lists', 'punctuation:no_comma']
KWARGS = [{'num_bullets': 4}, {}]


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
