"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'You visited a beach and a park. Which one is more likely to have a dog in it? Write at least 900 words. Do not include the words \'bark\' or \'run\' in your answer. Finish your response with the phrase "Does this make sense?"'
INSTRUCTION_IDS = ['length_constraints:number_words', 'keywords:forbidden_words', 'startend:end_checker']
KWARGS = [{'relation': 'at least', 'num_words': 900}, {'forbidden_words': ['bark', 'run']}, {'end_phrase': 'Does this make sense?'}]


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
