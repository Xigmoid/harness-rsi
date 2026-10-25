"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Which of the following is a better way to describe supporting ecological landscapes: (A) the practice of using the natural features of a landscape to create a more sustainable environment, or (B) the practice of using the natural features of a landscape to create a more aesthetically pleasing environment? Your response should be in English, all capital letters, contain no commas, and be fewer than 16 sentences.'
INSTRUCTION_IDS = ['change_case:english_capital', 'punctuation:no_comma', 'length_constraints:number_sentences']
KWARGS = [{}, {}, {'relation': 'less than', 'num_sentences': 16}]


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
