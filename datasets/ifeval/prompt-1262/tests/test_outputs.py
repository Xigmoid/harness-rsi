"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Can you elaborate on the following text: "Increased axle weight increased the damage to the road"? Your response must contain a title wrapped in double angular brackets, i.e. <<title>>. Use less than 5 sentences.'
INSTRUCTION_IDS = ['detectable_format:title', 'length_constraints:number_sentences']
KWARGS = [{}, {'relation': 'less than', 'num_sentences': 5}]


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
