"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Explain Generative Adversarial Networks (GANs) to me using bullet points. Do not contain any commas in your response. End your response with a postscript indicated by P.P.S\nInclude the keywords "lacking", "model", "performance", "quality", "architecture".'
INSTRUCTION_IDS = ['punctuation:no_comma', 'detectable_content:postscript', 'keywords:existence']
KWARGS = [{}, {'postscript_marker': 'P.P.S'}, {'keywords': ['lacking', 'model', 'performance', 'quality', 'architecture']}]


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
