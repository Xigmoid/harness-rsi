"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'I work in the marketing department and I need your help. I need a template for an advertisement for a new product which is a portable camera. In the template, capitalize a few words to stress main points. Please limit the number of words with all capital letters to less than four. Your response should contain at least ten sentences.'
INSTRUCTION_IDS = ['change_case:capital_word_frequency', 'length_constraints:number_sentences']
KWARGS = [{'capital_relation': 'less than', 'capital_frequency': 4}, {'relation': 'at least', 'num_sentences': 10}]


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
