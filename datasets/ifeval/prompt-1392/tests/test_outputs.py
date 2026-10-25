"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a blog post about the latest news in the US with a title in double angular brackets, i.e. <<title>>, and have less than 5 sentences (excluding 5). The sentences should be long, so that the total number of words in your response should be 250 or more.'
INSTRUCTION_IDS = ['detectable_format:title', 'length_constraints:number_sentences', 'length_constraints:number_words']
KWARGS = [{}, {'relation': 'less than', 'num_sentences': 5}, {'relation': 'at least', 'num_words': 250}]


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
