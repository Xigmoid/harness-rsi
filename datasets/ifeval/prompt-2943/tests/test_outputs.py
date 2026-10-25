"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Darth Vader is angry with you, his server, because you assassinated a fake Jedi. Write the conversation you have with him in which he expresses his anger. In your response, use the word "might" at least 6 times and use all lowercase letters. Make sure to break the conversation down to 3 parts, separated by ***, such as:\n[conversation part 1]\n***\n[conversation part 2]\n***\n[conversation part 3]'
INSTRUCTION_IDS = ['keywords:frequency', 'change_case:english_lowercase', 'length_constraints:number_paragraphs']
KWARGS = [{'relation': 'at least', 'keyword': 'might', 'frequency': 6}, {}, {'num_paragraphs': 3}]


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
