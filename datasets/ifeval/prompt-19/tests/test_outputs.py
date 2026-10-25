"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Create a blog post for professionals in the field of computer science in the form of a funny riddle. Your entire reply should contain 600 to 700 words.'
INSTRUCTION_IDS = ['length_constraints:number_words', 'length_constraints:number_words']
KWARGS = [{'relation': 'at least', 'num_words': 600}, {'relation': 'less than', 'num_words': 701}]


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
