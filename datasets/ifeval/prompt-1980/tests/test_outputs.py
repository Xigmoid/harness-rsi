"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a blog post about the sleek new magistrates with at least 300 words. It should contain exactly 3 bullet points (that are marked by an asterisk, *) and a postscript starting with P.S. at the end.'
INSTRUCTION_IDS = ['length_constraints:number_words', 'detectable_format:number_bullet_lists', 'detectable_content:postscript']
KWARGS = [{'relation': 'at least', 'num_words': 300}, {'num_bullets': 3}, {'postscript_marker': 'P.S.'}]


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
