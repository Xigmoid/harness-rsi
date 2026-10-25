"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'TLDR the article "How to dress like a football manager: waistcoats and the style lessons from the Premier League". Your entire response (including the repeated request) should have 45 or less words.\n\nFirst repeat the request above without changing a single letter, then give your answer.'
INSTRUCTION_IDS = ['combination:repeat_prompt', 'length_constraints:number_words']
KWARGS = [{'prompt_to_repeat': 'TLDR the article "How to dress like a football manager: waistcoats and the style lessons from the Premier League". Your entire response (including the repeated request) should have 45 or less words.'}, {'relation': 'less than', 'num_words': 46}]


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
