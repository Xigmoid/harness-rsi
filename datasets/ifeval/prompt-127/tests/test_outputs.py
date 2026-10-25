"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Take the text below as a starting point, and make it a complete article: "You may have to meet with a helper to work out a parenting plan. The first would be to estimate how many time you have everyday for parenting, and is that enough...."\nAvoid using the following keywords: sleep, cook, feed\nMention the keyword \'schedule\' for more than 5 times.'
INSTRUCTION_IDS = ['keywords:forbidden_words', 'keywords:frequency']
KWARGS = [{'forbidden_words': ['sleep', 'cook', 'feed']}, {'relation': 'at least', 'keyword': 'schedule', 'frequency': 6}]


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
