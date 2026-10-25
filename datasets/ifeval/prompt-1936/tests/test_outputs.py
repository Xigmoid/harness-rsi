"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Write a rubric, in the form of a list of bullet points, for evaluating the performance of a customer service representative. Your answer must not include keywords ['bad', 'underperform'] and must contain exactly 6 bullet points in the following form:\n* Bullet point 1\n* Bullet point 2\n* Bullet point 3\n* Bullet point 4\n* Bullet point 5\n* Bullet point 6"
INSTRUCTION_IDS = ['keywords:forbidden_words', 'detectable_format:number_bullet_lists']
KWARGS = [{'forbidden_words': ['bad', 'underperform']}, {'num_bullets': 6}]


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
