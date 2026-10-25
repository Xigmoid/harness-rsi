"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Generate a forum thread about several people waiting to hear the latest local news. All sentences should be short. Refrain from using any commas. Use placeholders to represent different usernames. Use square brackets for placeholders, like [username1], [username2]. Please include at least 20 placeholders in the thread.'
INSTRUCTION_IDS = ['detectable_content:number_placeholders', 'punctuation:no_comma']
KWARGS = [{'num_placeholders': 20}, {}]


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
