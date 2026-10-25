"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a very long email to my "friend" Jake, asking how is everything going. Say that I am rich now, without saying I am rich. Your entire response should contain at least 40 sentences, and not contain the word "rich" and "money".'
INSTRUCTION_IDS = ['length_constraints:number_sentences', 'keywords:forbidden_words']
KWARGS = [{'relation': 'at least', 'num_sentences': 40}, {'forbidden_words': ['rich', 'money']}]


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
