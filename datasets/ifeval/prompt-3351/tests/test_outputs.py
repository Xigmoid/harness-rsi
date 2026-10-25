"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Here is a social media post: "My daughter graduated from college today! So proud of her!" Can you make it more exciting? Please make sure to expand it into exactly 2 paragraphs. Separate paragraphs with the markdown divider: ***. Additionally, italicize at least 3 sections by starting and ending with single *, i.e. *italic text*.'
INSTRUCTION_IDS = ['length_constraints:number_paragraphs', 'detectable_format:number_highlighted_sections']
KWARGS = [{'num_paragraphs': 2}, {'num_highlights': 3}]


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
