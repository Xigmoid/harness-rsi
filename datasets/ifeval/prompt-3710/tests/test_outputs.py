"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'I need a rap that tells professionals how to get babies to sleep through the night. Your answer must contain a title, wrapped in double angular brackets, such as <<title>>. Additionally, you need to highlight at least 2 sections with markdown, i.e. *highlighted section*. Finally, at the end of your response, please explicitly add a postscript starting with P.P.S'
INSTRUCTION_IDS = ['detectable_format:number_highlighted_sections', 'detectable_format:title', 'detectable_content:postscript']
KWARGS = [{'num_highlights': 2}, {}, {'postscript_marker': 'P.P.S'}]


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
