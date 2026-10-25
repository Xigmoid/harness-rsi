"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Please rewrite the following sentence in a serious tone, similar to the president of the united states, and without an upward bias or inflation alarmism. Please also include the keywords "ink" and "memoirs" in the sentence: "The main points to consider are the following: First, the ECB’s own forecasts, which currently have a strong upward bias, must be revised downwards. Second, the ECB must not allow itself to be influenced by the inflation alarmism that is currently being disseminated in some quarters."'
INSTRUCTION_IDS = ['keywords:existence']
KWARGS = [{'keywords': ['ink', 'memoirs']}]


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
