"""Differential test: ``rsi.ifeval_checkers`` vs. Google's reference checkers.

``tests/vendor/ifeval_official/`` holds a verbatim mirror of the reference
implementation (Apache-2.0).  This test runs both implementations over every
one of the 541 IFEval prompts and a corpus of responses and asserts that they
agree on **every** instruction, for **every** response.

A mismatch means the reward function of the ``instruction_following`` suite
would diverge from published IFEval numbers, so the test is deliberately
strict.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from vendor.ifeval_official import instructions_registry as official_registry  # noqa: E402

from rsi.ifeval_checkers import (  # noqa: E402
    INSTRUCTION_DICT,
    check_prompt,
    check_prompt_instances,
)

DATASET = ROOT / "tests" / "data" / "ifeval_input_data.jsonl"

# A deterministic corpus that exercises the edge cases of the checkers.
RESPONSES: list[str] = [
    "",
    "   \n  ",
    "Hello world.",
    "Hello, world! This is a test, with commas.",
    "ALL CAPS RESPONSE WITHOUT PUNCTUATION",
    "all lowercase response without punctuation",
    "This is Title Case Text.\n\nSecond paragraph here.",
    "First paragraph.\n***\nSecond paragraph.\n***\nThird paragraph.",
    "First para.\n\nSecond para starts here.",
    "* one\n* two\n* three",
    "- one\n- two",
    "* one\n- two",
    "<<A Title>>\nSome body text.",
    "Some *highlighted* and **bold** text.",
    '{"key": "value"}',
    '```json\n{"key": "value"}\n```',
    '```\n{"key": "value"}\n```',
    "{'key': 'value'}",
    "Dear [name], welcome to [city].",
    "Section 1\nalpha\nSection 2\nbeta",
    "Section 1\nalpha\nSection 2\nbeta\nSection 3\ngamma",
    "Audience 1\nalpha\nAudience 2\nbeta",
    "My answer is yes.",
    "My answer is no.",
    "Maybe, who knows?",
    "The body of the answer.\n\nP.S. one more thing.",
    "The body of the answer.\n\nP.P.S one more thing.",
    "The body of the answer.",
    'The whole answer in double quotes.',
    "First response.\n******\nSecond response.",
    "Same response.\n******\nSame response.",
    "Only one response.",
    "Peace!",
    "Here is my advice and then Peace!",
    "quiet queue with many q letters",
    "A story about a story.",
    "THE QUICK BROWN FOX",
    "the quick brown fox",
    "One two three four five six seven eight nine ten.",
    "First sentence. Second sentence! Third sentence?",
    "日本語のテキストです。",
    "This response contains a mom and a mother and intern and grow.",
    "This response mentions rock and sleep and cook.",
    " ".join(["word"] * 320),
    " ".join(["word"] * 12),
    "x" * 400,
    "\n".join(f"paragraph {i}" for i in range(5)),
    "A. B. C. D.",
    "Dr. Smith went to Washington. He arrived at 5 p.m. on Jan. 3.",
]


def _load_dataset() -> list[dict]:
    rows = []
    with DATASET.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


DATASET_ROWS = _load_dataset()


def _official_check(
    instruction_id: str, prompt: str, response: str, kwargs: dict
) -> bool:
    """Run the reference checker for a single instruction.

    Returns ``None`` when the reference implementation is *not deterministic*
    for the given kwargs -- it randomises a missing/invalid parameter.  The
    only such case in the dataset is ``keywords:letter_frequency`` with a
    non-alphabetic character (the reference only accepts ``a``-``z`` and
    then measures a random letter).
    """
    if instruction_id == "keywords:letter_frequency":
        letter = str(kwargs.get("letter", ""))
        if not (len(letter) == 1 and "a" <= letter <= "z"):
            return None
    cls = official_registry.INSTRUCTION_DICT[instruction_id]
    instruction = cls(instruction_id)
    instruction.build_description(**kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)
    return bool(instruction.check_following(response))


def test_dataset_is_complete() -> None:
    assert len(DATASET_ROWS) == 541
    ids = {
        i
        for row in DATASET_ROWS
        for i in row.get("instruction_id_list", [])
    }
    assert ids == set(official_registry.INSTRUCTION_DICT), "instruction id mismatch"
    assert ids <= set(INSTRUCTION_DICT), "our registry misses an instruction type"


def test_registry_covers_every_dataset_instruction() -> None:
    assert set(INSTRUCTION_DICT) == set(official_registry.INSTRUCTION_DICT)


@pytest.mark.parametrize("row", DATASET_ROWS, ids=[str(r.get("key")) for r in DATASET_ROWS])
def test_differential_agreement(row: dict) -> None:
    """Our checkers must agree with the reference on every response."""
    prompt = row["prompt"]
    ids = list(row.get("instruction_id_list", []))
    kwargs_list = list(row.get("kwargs", []))

    for response in RESPONSES:
        ours = check_prompt_instances(prompt, response, ids, kwargs_list)
        for index, instruction_id in enumerate(ids):
            kwargs = dict(kwargs_list[index]) if index < len(kwargs_list) else {}
            theirs = _official_check(instruction_id, prompt, response, kwargs)
            if theirs is None:  # the reference randomises here
                continue
            assert ours[index] == theirs, (
                f"instruction={instruction_id} response={response!r} "
                f"ours={ours[index]} reference={theirs}"
            )


def test_differential_agreement_on_generated_responses() -> None:
    """Fuzz a deterministic random corpus over every instruction type."""
    rng = random.Random(20241020)
    alphabet = list("abcXYZ .,!?*'\"[]()\n#-_")
    corpus = [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        for _ in range(300)
    ]
    for row in DATASET_ROWS[::7]:
        prompt = row["prompt"]
        ids = list(row.get("instruction_id_list", []))
        kwargs_list = list(row.get("kwargs", []))
        for response in corpus:
            ours = check_prompt_instances(prompt, response, ids, kwargs_list)
            for index, instruction_id in enumerate(ids):
                kwargs = dict(kwargs_list[index]) if index < len(kwargs_list) else {}
                theirs = _official_check(instruction_id, prompt, response, kwargs)
                if theirs is None:  # the reference randomises here
                    continue
                assert ours[index] == theirs, (
                    f"instruction={instruction_id} response={response!r} "
                    f"ours={ours[index]} reference={theirs}"
                )


def test_duplicate_instruction_ids_are_combined_with_and() -> None:
    """A prompt may repeat an instruction id; strict accuracy needs an AND."""
    ids = ["change_case:capital_word_frequency", "change_case:capital_word_frequency"]
    kwargs = [
        {"capital_relation": "less than", "capital_frequency": 10},
        {"capital_relation": "at least", "capital_frequency": 1},
    ]
    # "Hello world." has no capitalised word: "less than 10" holds, "at least 1"
    # does not -> the strict result must be False.
    results = check_prompt("p", "Hello world.", ids, kwargs)
    assert results["change_case:capital_word_frequency"] is False
    # "HELLO WORLD" satisfies both.
    results = check_prompt("p", "HELLO WORLD", ids, kwargs)
    assert results["change_case:capital_word_frequency"] is True


def test_reference_mirror_is_unmodified() -> None:
    """Guard against accidental edits of the vendored reference code."""
    source = (
        ROOT / "tests" / "vendor" / "ifeval_official" / "instructions.py"
    ).read_text(encoding="utf-8")
    assert "class ConstrainedResponseChecker" in source
    assert "class CapitalWordFrequencyChecker" in source
    # the only edit is the import line
    assert source.count("from . import instructions_util") == 1
    assert "from instruction_following_eval import" not in source
