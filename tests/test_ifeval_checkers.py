"""Positive / negative tests for the 25 IFEval instruction checkers.

``rsi.ifeval_checkers`` is a faithful port of Google's reference
implementation (``tests/vendor/ifeval_official/``).  This file documents the
*intent* of every instruction with a hand-written pair of responses -- one
that satisfies it and one that violates it -- and cross-checks both against
the reference implementation.

The exhaustive machine check lives in ``tests/test_ifeval_differential.py``
(all 541 dataset prompts x a response corpus).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from vendor.ifeval_official import instructions_registry as official_registry  # noqa: E402

from rsi.ifeval_checkers import INSTRUCTION_DICT, check_prompt  # noqa: E402

# (instruction_id, kwargs, satisfying response, violating response)
CASES: list[tuple[str, dict, str, str]] = [
    # -- punctuation ------------------------------------------------------- #
    (
        "punctuation:no_comma",
        {},
        "No commas here at all",
        "Yes, commas are here.",
    ),
    # -- length constraints ------------------------------------------------ #
    (
        "length_constraints:number_words",
        {"relation": "at least", "num_words": 5},
        "one two three four five six",
        "one two three",
    ),
    (
        "length_constraints:number_words",
        {"relation": "less than", "num_words": 5},
        "one two three",
        "one two three four five six",
    ),
    (
        "length_constraints:number_sentences",
        {"relation": "at least", "num_sentences": 2},
        "First one. Second one.",
        "Only one.",
    ),
    (
        "length_constraints:number_sentences",
        {"relation": "less than", "num_sentences": 2},
        "Only one.",
        "First one. Second one. Third one.",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "First paragraph.\n***\nSecond paragraph.",
        "First paragraph.\n***\nSecond paragraph.\n***\nThird paragraph.",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 1, "first_word": "weekend"},
        "Weekend is quiet.\n\nThe rest is loud.",
        "The weekend is quiet.\n\nThe rest is loud.",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "monday"},
        "First bit.\n\nMonday starts the week.",
        "First bit.\n\nTuesday starts the week.",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 1, "first_word": "weekend"},
        "Weekend is quiet.\n\nThe rest is loud.",
        "Weekend is quiet.\nThe rest is loud.",  # wrong paragraph count
    ),
    # -- keywords ---------------------------------------------------------- #
    (
        "keywords:existence",
        {"keywords": ["mom", "mother"]},
        "My mom is my mother.",
        "My mum is my mother.",
    ),
    (
        "keywords:forbidden_words",
        {"forbidden_words": ["rock"]},
        "Music is fine.",
        "Rock music is fine.",
    ),
    (
        "keywords:frequency",
        {"keyword": "story", "relation": "at least", "frequency": 2},
        "A story about a story.",
        "A single tale.",
    ),
    (
        "keywords:frequency",
        {"keyword": "story", "relation": "less than", "frequency": 2},
        "A single tale.",
        "A story about a story.",
    ),
    (
        "keywords:letter_frequency",
        {"letter": "q", "let_relation": "at least", "let_frequency": 2},
        "quiet queue",
        "quiet",
    ),
    (
        "keywords:letter_frequency",
        {"letter": "q", "let_relation": "less than", "let_frequency": 2},
        "quiet",
        "quiet queue",
    ),
    # -- detectable format -------------------------------------------------- #
    (
        "detectable_format:title",
        {},
        "<<My Title>>\n\nSome body text.",
        "Some body text without a title.",
    ),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 2},
        "This is *one* and this is *two*.",
        "This is *one* only.",
    ),
    (
        "detectable_format:number_bullet_lists",
        {"num_bullets": 2},
        "* first\n* second",
        "* first",
    ),
    (
        "detectable_format:number_bullet_lists",
        {"num_bullets": 2},
        "* first\n* second",
        "* first\n* second\n* third",
    ),
    (
        "detectable_format:number_bullet_lists",
        {"num_bullets": 2},
        "- first\n- second",
        "- first",
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "Dear [name], meet [place].",
        "Dear name, meet place.",
    ),
    (
        "detectable_format:json_format",
        {},
        '{"a": 1}',
        "{'a': 1}",
    ),
    (
        "detectable_format:json_format",
        {},
        '```json\n{"a": 1}\n```',
        "```json\nnot json\n```",
    ),
    (
        "detectable_format:multiple_sections",
        {"section_spliter": "Section", "num_sections": 2},
        "Section 1\nalpha\nSection 2\nbeta",
        "Section 1\nalpha",
    ),
    (
        "detectable_format:multiple_sections",
        {"section_spliter": "Audience", "num_sections": 2},
        "Audience 1\nalpha\nAudience 2\nbeta",
        "Audience 1\nalpha",
    ),
    (
        "detectable_format:constrained_response",
        {},
        "My answer is yes.",
        "Maybe?",
    ),
    # -- detectable content ------------------------------------------------- #
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 3},
        "[a] [b] [c]",
        "[a] [b]",
    ),
    (
        "detectable_content:postscript",
        {"postscript_marker": "P.S."},
        "The body.\n\nP.S. one more thing.",
        "The body.\n\nNo postscript here.",
    ),
    (
        "detectable_content:postscript",
        {"postscript_marker": "P.P.S"},
        "The body.\n\nP.P.S still one more thing.",
        "The body.",
    ),
    # -- start / end -------------------------------------------------------- #
    (
        "startend:quotation",
        {},
        '"The whole thing in quotes."',
        "The whole thing in quotes.",
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Peace!"},
        "Peace!",
        "Peace! And then some more text.",
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Peace!"},
        "Here is my advice. Peace!",
        "Here is my advice.",
    ),
    # -- change case --------------------------------------------------------- #
    (
        "change_case:english_lowercase",
        {},
        "all lowercase text",
        "Not All Lowercase",
    ),
    (
        "change_case:english_capital",
        {},
        "ALL UPPERCASE TEXT",
        "Not All Uppercase",
    ),
    (
        "change_case:capital_word_frequency",
        {"capital_relation": "at least", "capital_frequency": 2},
        "THIS is A test",
        "this is a test",
    ),
    (
        "change_case:capital_word_frequency",
        {"capital_relation": "less than", "capital_frequency": 2},
        "this is a test",
        "THIS is A test",
    ),
    # -- combination ---------------------------------------------------------- #
    (
        "combination:repeat_prompt",
        {"prompt_to_repeat": "Write a haiku about rain."},
        "Write a haiku about rain. Rain falls softly.",
        "Rain falls softly.",
    ),
    (
        "combination:two_responses",
        {},
        "First answer.\n******\nSecond answer.",
        "Only one answer.",
    ),
    (
        "combination:two_responses",
        {},
        "First answer.\n******\nSecond answer.",
        "Same answer.\n******\nSame answer.",
    ),
    # -- language ------------------------------------------------------------- #
    (
        "language:response_language",
        {"language": "en"},
        "This answer is written in plain English for the checker.",
        "これは日本語のテキストです。",
    ),
]


def _reference_check(instruction_id: str, response: str, kwargs: dict) -> bool:
    cls = official_registry.INSTRUCTION_DICT[instruction_id]
    instruction = cls(instruction_id)
    instruction.build_description(**kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt="prompt")
    return bool(instruction.check_following(response))


@pytest.mark.parametrize(
    "instruction_id,kwargs,positive,negative",
    CASES,
    ids=[f"{case[0]}-{i}" for i, case in enumerate(CASES)],
)
def test_checker_positive_and_negative(
    instruction_id: str, kwargs: dict, positive: str, negative: str
) -> None:
    assert instruction_id in INSTRUCTION_DICT, f"no checker for {instruction_id}"
    results = check_prompt("prompt", positive, [instruction_id], [kwargs])
    assert results[instruction_id] is True, f"{instruction_id} rejected a valid response"

    results = check_prompt("prompt", negative, [instruction_id], [kwargs])
    assert results[instruction_id] is False, f"{instruction_id} accepted a bad response"


@pytest.mark.parametrize(
    "instruction_id,kwargs,positive,negative",
    CASES,
    ids=[f"{case[0]}-{i}" for i, case in enumerate(CASES)],
)
def test_matches_reference_implementation(
    instruction_id: str, kwargs: dict, positive: str, negative: str
) -> None:
    """Our checker must agree with Google's reference on every hand-written case."""
    for response in (positive, negative):
        ours = check_prompt("prompt", response, [instruction_id], [kwargs])
        theirs = _reference_check(instruction_id, response, kwargs)
        assert ours[instruction_id] == theirs, (
            f"instruction={instruction_id} response={response!r} "
            f"ours={ours[instruction_id]} reference={theirs}"
        )


def test_every_dataset_instruction_type_has_a_case() -> None:
    """The case list must cover the 25 instruction types of the dataset."""
    covered = {case[0] for case in CASES}
    expected = set(official_registry.INSTRUCTION_DICT)
    assert expected <= covered, f"missing cases: {sorted(expected - covered)}"
    assert expected <= set(INSTRUCTION_DICT), "registry is missing an instruction type"


def test_check_prompt_is_strict() -> None:
    """Strict accuracy: one violated instruction fails the whole prompt."""
    results = check_prompt(
        "prompt",
        "no commas and no title",
        ["punctuation:no_comma", "detectable_format:title"],
        [{}, {}],
    )
    assert results == {
        "punctuation:no_comma": True,
        "detectable_format:title": False,
    }


def test_check_prompt_unknown_instruction_is_false() -> None:
    results = check_prompt("prompt", "hello", ["does:not:exist"], [{}])
    assert results["does:not:exist"] is False


def test_check_prompt_tolerates_missing_kwargs() -> None:
    """A missing required kwarg must fail the check instead of raising."""
    results = check_prompt(
        "prompt", "one two three", ["length_constraints:number_words"], []
    )
    assert results["length_constraints:number_words"] is False


def test_letter_frequency_counts_non_alphabetic_letters() -> None:
    """The reference randomises non-alphabetic letters; we count them."""
    kwargs = {"letter": "#", "let_relation": "at least", "let_frequency": 2}
    results = check_prompt("prompt", "a # b # c", ["keywords:letter_frequency"], [kwargs])
    assert results["keywords:letter_frequency"] is True
    results = check_prompt("prompt", "a # b", ["keywords:letter_frequency"], [kwargs])
    assert results["keywords:letter_frequency"] is False
