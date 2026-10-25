"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Can you tell me about the Hamptons? Your answer must be at least 300 words, must contain at least 3 placeholders represented by square brackets, such as [address] and exactly 2 bullet points using the markdown bullet points such as:\n* Bullet point 1\n* Bullet point 2'
INSTRUCTION_IDS = ['length_constraints:number_words', 'detectable_content:number_placeholders', 'detectable_format:number_bullet_lists']
KWARGS = [{'relation': 'at least', 'num_words': 300}, {'num_placeholders': 3}, {'num_bullets': 2}]


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
