"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Write an interesting and funny article about the biology of a banana peel. In your response, the word disappointed should appear at least 2 times, and at least six section should be highlighted with markdown,  i.e *banana peel*.'
INSTRUCTION_IDS = ['keywords:frequency', 'detectable_format:number_highlighted_sections']
KWARGS = [{'relation': 'at least', 'keyword': 'disappointed', 'frequency': 2}, {'num_highlights': 6}]


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
