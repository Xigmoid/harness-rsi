"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Can you write a poem about the pros and cons of playing a lot of video games? Please make sure it's at least 40 sentences long (don't forget to add punctuations). You must highlight some words or phrases in your response, like *highlighted phrase*."
INSTRUCTION_IDS = ['detectable_format:number_highlighted_sections', 'length_constraints:number_sentences']
KWARGS = [{'num_highlights': 1}, {'relation': 'at least', 'num_sentences': 40}]


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
