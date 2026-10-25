"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Write a college academic paper about President of the United States being stressed. Make sure not to include negative words such as 'sad', 'crazy', 'stress', etc., in the response. Also, make sure to include at least 15 placeholders represented by square brackets, such as [address]."
INSTRUCTION_IDS = ['keywords:forbidden_words', 'detectable_content:number_placeholders']
KWARGS = [{'forbidden_words': ['sad', 'crazy', 'stress']}, {'num_placeholders': 15}]


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
