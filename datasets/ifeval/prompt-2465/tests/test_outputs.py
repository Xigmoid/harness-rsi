"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'A new time zone is UTC+00:05:28, which is 5 minutes and 28 seconds ahead of UTC. Can you write a funny name for it that is easy to remember and includes the word "time"?\nFirst, repeat the request word for word without change, then give your answer (Notes: 1. do NOT say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)'
INSTRUCTION_IDS = ['combination:repeat_prompt']
KWARGS = [{'prompt_to_repeat': 'A new time zone is UTC+00:05:28, which is 5 minutes and 28 seconds ahead of UTC. Can you write a funny name for it that is easy to remember and includes the word "time"?'}]


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
