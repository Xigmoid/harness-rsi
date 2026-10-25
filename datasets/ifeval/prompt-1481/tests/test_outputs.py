"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Write a casual, interesting, and weird resume for Antonia Maj who is applying for a job at a coffee company. They have experience in marketing, customer service, and sales. They are educated but not majored in anything related to coffee.\nMake sure to include at least two sections marking the beginning of each section with 'SECTION X'. In your entire response make sure to use exactly two bullet points in markdown format. Please use the following bullet point format:\n* Text for bullet 1\n* Text for bullet 2"
INSTRUCTION_IDS = ['detectable_format:multiple_sections', 'detectable_format:number_bullet_lists']
KWARGS = [{'section_spliter': 'SECTION', 'num_sections': 2}, {'num_bullets': 2}]


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
