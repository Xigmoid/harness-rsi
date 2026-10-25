"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Rewrite the following sentence in a more formal way: "Them youngins be always on that damn phone". Imagine you are the president of the United States. Avoid using the word "youngins" and "damn" throughout your entire response.\n\nPlease repleat the entire first line above, then give your rewrite. Do not add any word before that. In other words, your entire response should start with "Rewrite the following".'
INSTRUCTION_IDS = ['combination:repeat_prompt', 'keywords:forbidden_words']
KWARGS = [{'prompt_to_repeat': 'Rewrite the following sentence in a more formal way: "Them youngins be always on that damn phone". Imagine you are the president of the United States. Avoid using the word "youngins" and "damn" throughout your entire response.'}, {'forbidden_words': ['youngins', 'damn']}]


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
