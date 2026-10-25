"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = 'Can you re-create a story from a fictional newspaper with title: "A man mysteriously died in his house, and police are investigating"? Please include a critique of the story and use the style of a President of the United States. Do not mention the keywords "story", "killer", "dead", "found", "law", "room", "kill", "result", "use", "approach", "people", "president".'
INSTRUCTION_IDS = ['keywords:forbidden_words']
KWARGS = [{'forbidden_words': ['story', 'killer', 'dead', 'found', 'law', 'room', 'kill', 'result', 'use', 'approach', 'people', 'president']}]


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
