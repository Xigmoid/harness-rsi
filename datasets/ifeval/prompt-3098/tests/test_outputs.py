"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a blog post about \'how to improve your writing skills\' with exactly 3 bullet points in markdown format, and exactly 4 sections.\n\nBullet points are indicated by "* ". For example:\n* Bullet 1\n* Bullet 2\n\nSections are separated by 3 asterisks: ***. For example:\n\nSection 1\n***\nSection 2\n\nYou should use words with all capital letters for at least 2 times.'
INSTRUCTION_IDS = ['detectable_format:number_bullet_lists', 'length_constraints:number_paragraphs', 'change_case:capital_word_frequency']
KWARGS = [{'num_bullets': 3}, {'num_paragraphs': 4}, {'capital_relation': 'at least', 'capital_frequency': 2}]


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
