"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Rewrite the following blog as a list of exactly 4 bullet points: "The world is a beautiful place. The sun is shining, the birds are singing, and the flowers are blooming. It\'s a perfect day to go outside and enjoy all that nature has to offer." The bullet points should be in markdown such as:\n* Bullet point 1\n* Bullet point 2'
INSTRUCTION_IDS = ['detectable_format:number_bullet_lists']
KWARGS = [{'num_bullets': 4}]


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
