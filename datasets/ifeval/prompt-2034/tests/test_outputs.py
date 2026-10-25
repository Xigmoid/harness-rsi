"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a summary of the following text in a funny way: "The 2018 Nobel Prize in Chemistry has been awarded to Frances Arnold, George P. Smith and Gregory P. Winter for their work on directed evolution. Arnold was awarded half of the prize for her work on the directed evolution of enzymes, while Smith and Winter shared the other half for their work on the directed evolution of antibodies."\n\nDo not include "enzymes" and "antibodies" in your response.'
INSTRUCTION_IDS = ['keywords:forbidden_words']
KWARGS = [{'forbidden_words': ['enzymes', 'antibodies']}]


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
