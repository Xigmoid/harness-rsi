"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "I want a weird poem that I can send to my friend Steve in Australia. It should be about why there are no kangaroo invasions in the US. The word robber should appear at least 2 times, and the poem must contain exactly 2 bullet point in markdown format, using the exact format below:\n* Point 1\n* Pont 2\nDo not include keywords ['kill', 'slaughter', 'occupy', 'invasion'] in the response."
INSTRUCTION_IDS = ['keywords:frequency', 'detectable_format:number_bullet_lists', 'keywords:forbidden_words']
KWARGS = [{'relation': 'at least', 'keyword': 'robber', 'frequency': 2}, {'num_bullets': 2}, {'forbidden_words': ['kill', 'slaughter', 'occupy', 'invasion']}]


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
