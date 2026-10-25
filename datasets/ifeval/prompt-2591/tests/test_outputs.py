"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "I want you to act like a DnD dungeon master. I will be the sole player. Create a random class character sheet for me. Wrap the entire output in JSON format using markdown ticks. Include keywords 'medalist' and 'theta' in the response."
INSTRUCTION_IDS = ['detectable_format:json_format', 'keywords:existence']
KWARGS = [{}, {'keywords': ['medalist', 'theta']}]


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
