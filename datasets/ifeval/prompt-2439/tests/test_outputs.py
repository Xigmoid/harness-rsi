"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Rewrite the following sentence to exactly 3 paragraphs, separated by two new lines and without using any commas: "Offences are not the only things that are grasped by the police.". Paragraph 1 must start with word punched. The response must contain at least 2 placeholders represented by square brackets, such as [address].'
INSTRUCTION_IDS = ['punctuation:no_comma', 'length_constraints:nth_paragraph_first_word', 'detectable_content:number_placeholders']
KWARGS = [{}, {'first_word': 'punched', 'num_paragraphs': 3, 'nth_paragraph': 1}, {'num_placeholders': 2}]


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
