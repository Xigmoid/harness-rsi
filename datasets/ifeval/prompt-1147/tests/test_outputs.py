"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Rewrite the following statement to make it sound more formal, like a President of the United States:\n"Hi guys. The work was done to add in a fix for the issue that was observed in the field with the SSO. We are working with our collaborators closely. We will get it done. Thanks ya all."\nDo not include the following keywords: field, thanks, issue, collaborator.'
INSTRUCTION_IDS = ['keywords:forbidden_words']
KWARGS = [{'forbidden_words': ['field', 'thanks', 'issue', 'collaborator']}]


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
