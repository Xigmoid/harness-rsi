"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Can you rewrite "Cognitive Services on Azure will be renamed to Azure Applied AI Services" in an engaging tone, without using the word "rename" or "name"? Keep it under 3 sentences (just 1 or 2 sentences, not 3).'
INSTRUCTION_IDS = ['keywords:forbidden_words', 'length_constraints:number_sentences']
KWARGS = [{'forbidden_words': ['name', 'rename']}, {'relation': 'less than', 'num_sentences': 3}]


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
