"""Verifier for one IFEval prompt (strict accuracy: every instruction must hold)."""

import sys

sys.path.insert(0, "/tests")

from rsi.ifeval_checkers import check_prompt  # noqa: E402

PROMPT = "Summarize the following paragraph. Use words in all capital letters at least 3 times to highlight key points.\n\nHow to get to 100% renewable energy in the US? Two researchers from Stanford University and the University of California, Berkeley studied the potential for solar and wind energy to power the US grid. They concluded that it is possible to get to 100% renewable energy by 2050 by building more solar and wind farms. The authors also pointed out that such a system would be less vulnerable to disruptions than the current grid, which is reliant on a few large power plants. The authors' plan would also create jobs and reduce pollution."
INSTRUCTION_IDS = ['change_case:capital_word_frequency']
KWARGS = [{'capital_relation': 'at least', 'capital_frequency': 3}]


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
