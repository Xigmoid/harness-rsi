"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'A nucleus is a cluster of protons and neutrons. Elaborate on this. Write exactly 9 very short bullet points. Limit the number of words you use (less than 100 words). An example:\n* A nucleus is a cluster of protons and neutrons\n* A proton is ....\n\nPlease follow the format of the example above.'
INSTRUCTION_IDS = ['detectable_format:number_bullet_lists', 'length_constraints:number_words']
KWARGS = [{'num_bullets': 9}, {'relation': 'less than', 'num_words': 100}]


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
