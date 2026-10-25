"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write a plot for a story about two people who swap fingerprints. Include a title wrapped in double angular brackets, i.e. <<title>>. In your response please avoid using commas.\n\nFirst, repeat the request above word for word without change.\nDo not say any words or characters before repeating the request above.\nAfter you repeated the request, you can give your response next.'
INSTRUCTION_IDS = ['combination:repeat_prompt', 'detectable_format:title', 'punctuation:no_comma']
KWARGS = [{'prompt_to_repeat': 'Write a plot for a story about two people who swap fingerprints. Include a title wrapped in double angular brackets, i.e. <<title>>. In your response please avoid using commas.'}, {}, {}]


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
