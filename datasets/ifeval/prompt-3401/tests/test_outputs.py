"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Can you give me a zany, bullet point TLDR of this article: https://en.wikipedia.org/wiki/Dota_2\n\nMake it zany, but do not include the keywords 'icefrog', 'blizzard', 'lawsuit' in the response."
INSTRUCTION_IDS = ['keywords:forbidden_words']
KWARGS = [{'forbidden_words': ['icefrog', 'blizzard', 'lawsuit']}]


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
