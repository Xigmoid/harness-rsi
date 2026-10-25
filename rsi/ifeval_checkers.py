"""Verifiable-instruction checkers for the IFEval instruction set.

The IFEval dataset (`google/IFEval`, Apache-2.0) ships 541 prompts together
with a list of *verifiable instruction ids* and their parameters.  This module
implements the checkers for all 25 instruction types of that dataset.

**Provenance.**  The checkers are a faithful port of the reference
implementation published by Google Research in
``google-research/instruction_following_eval`` (Apache-2.0).  The upstream
source is mirrored verbatim under ``tests/vendor/ifeval_official/`` and the
test suite runs a *differential* test of this module against that mirror, so
every semantic change here is caught.

Two deliberate differences, both required for a reproducible benchmark:

* the reference implementation *randomises* missing parameters (e.g. a random
  number of words); this module raises / fails the instruction instead, so a
  reward is a deterministic function of (response, instruction, kwargs);
* ``langdetect`` and ``nltk`` are optional here.  When they are missing the
  module degrades to deterministic approximations (script uniformity for the
  language checks, a regex sentence splitter for the sentence count).  The
  degraded path is marked with ``# fallback``.

The API mirrors the reference one closely enough to be used as a drop-in::

    instruction = INSTRUCTION_DICT[instruction_id](instruction_id)
    instruction.build_description(**kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)
    instruction.check_following(response_text)
"""

from __future__ import annotations

import functools
import json
import re
from typing import Any, Iterable, Sequence

__all__ = [
    "INSTRUCTION_DICT",
    "Instruction",
    "check_prompt",
    "check_prompt_instances",
    "count_sentences",
    "count_words",
]


# --------------------------------------------------------------------------- #
# optional dependencies
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - exercised implicitly
    import langdetect  # type: ignore

    # `langdetect` seeds its detector from the global RNG, so two calls on the
    # same text can disagree.  A benchmark reward has to be a deterministic
    # function of the response, so the seed is pinned here (this also makes
    # the module agree with the reference implementation).
    langdetect.DetectorFactory.seed = 0
except ImportError:  # pragma: no cover
    langdetect = None  # type: ignore

try:  # pragma: no cover - exercised implicitly
    import nltk  # type: ignore
    from nltk.tokenize import RegexpTokenizer  # type: ignore

    _WORD_TOKENIZER = RegexpTokenizer(r"\w+")
except ImportError:  # pragma: no cover
    nltk = None  # type: ignore
    _WORD_TOKENIZER = None


# --------------------------------------------------------------------------- #
# counting helpers (reference: instruction_following_eval/instructions_util.py)
# --------------------------------------------------------------------------- #


def count_words(text: str) -> int:
    """Number of word tokens (``nltk``'s ``RegexpTokenizer(r"\\w+")``)."""
    if _WORD_TOKENIZER is not None:
        return len(_WORD_TOKENIZER.tokenize(text))
    return len(re.findall(r"\w+", text))  # fallback: identical to the tokenizer


_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|"
    r"However\s|That\s|This\s|Wherever)"
)
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text: str) -> list[str]:
    """Split the text into sentences (reference implementation, verbatim)."""
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(
        _MULTIPLE_DOTS,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_sentences(text: str) -> int:
    """Number of sentences (``nltk``'s punkt tokenizer, with a fallback)."""
    if nltk is not None:
        try:
            tokenizer = _get_sentence_tokenizer()
            return len(tokenizer.tokenize(text))
        except Exception:
            pass  # punkt data is not installed -> fallback
    return len([s for s in split_into_sentences(text) if s.strip()])  # fallback


@functools.lru_cache(maxsize=None)
def _get_sentence_tokenizer():
    """The punkt tokenizer used by the reference implementation."""
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


def _word_tokens(text: str) -> list[str]:
    """Word tokens for the capital-word counter (``nltk.word_tokenize``)."""
    if nltk is not None:
        try:
            return nltk.tokenize.word_tokenize(text)
        except LookupError:
            pass  # punkt data is not installed -> fallback
        except Exception:
            pass
    # fallback: a close approximation of the treebank tokenizer
    return re.findall(r"\w+(?:[.'’-]\w+)*", text)


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #


class Instruction:
    """Base class for the verifiable instructions."""

    def __init__(self, instruction_id: str):
        self.id = instruction_id
        self._description = ""
        self._kwargs: dict[str, Any] = {}

    # -- description ------------------------------------------------------- #

    def build_description(self, **kwargs: Any) -> str:
        """Store the parameters and render the human-readable description."""
        self._kwargs = dict(kwargs)
        self._description = self._render(**kwargs)
        return self._description

    def _render(self, **kwargs: Any) -> str:
        return f"Follow the instruction: {self.id}."

    def get_instruction_args(self) -> dict[str, Any] | None:
        """The arguments the checker needs beyond the response itself."""
        return None

    def get_instruction_args_keys(self) -> list[str]:
        """The kwargs that *must* be present for the check to be meaningful."""
        return []

    # -- checking ---------------------------------------------------------- #

    def check_following(self, value: str) -> bool:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# language
# --------------------------------------------------------------------------- #


class ResponseLanguageChecker(Instruction):
    """The whole response must be written in the requested language."""

    def build_description(self, *, language: str | None = None, **_: Any) -> str:
        self._language = language
        self._description = (
            f"Your ENTIRE response should be in {language} language, "
            "no other language is allowed."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {"language": self._language}

    def get_instruction_args_keys(self) -> list[str]:
        return ["language"]

    def check_following(self, value: str) -> bool:
        if langdetect is None:
            # fallback: script uniformity instead of a language model
            return bool(value.strip()) and _single_script(value)
        try:
            return langdetect.detect(value) == self._language
        except Exception:
            # the reference implementation counts an undetectable language as
            # "followed"
            return True


def _single_script(text: str) -> bool:
    """A crude script-uniformity check (used when langdetect is missing)."""
    scripts = set()
    for char in text:
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF:
            scripts.add("han")
        elif 0x3040 <= code <= 0x30FF:
            scripts.add("kana")
        elif 0xAC00 <= code <= 0xD7AF:
            scripts.add("hangul")
        elif 0x0400 <= code <= 0x04FF:
            scripts.add("cyrillic")
        elif 0x0600 <= code <= 0x06FF:
            scripts.add("arabic")
        elif 0x0370 <= code <= 0x03FF:
            scripts.add("greek")
    return len(scripts) <= 1


def _looks_english(text: str) -> bool:
    """Fallback for ``langdetect.detect(text) == "en"``."""
    if langdetect is not None:
        try:
            return langdetect.detect(text) == "en"
        except Exception:
            return True
    # fallback: latin script without diacritics-heavy scripts
    return _single_script(text)


# --------------------------------------------------------------------------- #
# length constraints
# --------------------------------------------------------------------------- #


class NumberOfSentences(Instruction):
    """Number of sentences, with an ``at least`` / ``less than`` relation."""

    def build_description(
        self,
        *,
        num_sentences: int | None = None,
        relation: str | None = None,
        **_: Any,
    ) -> str:
        self._num_sentences_threshold = num_sentences
        if relation is None:
            relation = "at least"
        elif relation not in ("less than", "at least"):
            raise ValueError(
                "The supported relation for comparison must be in "
                f"('less than', 'at least'), but {relation} is given."
            )
        self._comparison_relation = relation
        self._description = (
            f"Your response should contain {relation} {num_sentences} sentences."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {
            "num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_sentences", "relation"]

    def check_following(self, value: str) -> bool:
        num_sentences = count_sentences(value)
        if self._comparison_relation == "less than":
            return num_sentences < self._num_sentences_threshold
        return num_sentences >= self._num_sentences_threshold


class ParagraphChecker(Instruction):
    """Exactly ``num_paragraphs`` paragraphs, separated by ``***``."""

    def build_description(self, *, num_paragraphs: int | None = None, **_: Any) -> str:
        self._num_paragraphs = num_paragraphs
        self._description = (
            f"There should be {num_paragraphs} paragraphs. "
            "Paragraphs are separated with the markdown divider: ***"
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {"num_paragraphs": self._num_paragraphs}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_paragraphs"]

    def check_following(self, value: str) -> bool:
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)
        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False
        return num_paragraphs == self._num_paragraphs


class ParagraphFirstWordCheck(Instruction):
    """The nth paragraph must start with ``first_word``.

    Paragraphs of this instruction are separated by *two newlines* (``\\n\\n``),
    not by the ``***`` divider.
    """

    def build_description(
        self,
        *,
        num_paragraphs: int | None = None,
        nth_paragraph: int | None = None,
        first_word: str | None = None,
        **_: Any,
    ) -> str:
        self._num_paragraphs = num_paragraphs
        self._nth_paragraph = nth_paragraph
        self._first_word = first_word.lower() if first_word is not None else None
        self._description = (
            f"There should be {num_paragraphs} paragraphs. "
            "Paragraphs and only paragraphs are separated with each other by "
            "two new lines as if it was '\\n\\n' in python. "
            f"Paragraph {nth_paragraph} must start with word {first_word}."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value: str) -> bool:
        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)
        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}
        word = paragraph.split()[0].strip()
        word = word.lstrip("'")
        word = word.lstrip('"')
        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return num_paragraphs == self._num_paragraphs and first_word == self._first_word


class NumberOfWords(Instruction):
    """Number of words, with an ``at least`` / ``less than`` relation."""

    def build_description(
        self,
        *,
        num_words: int | None = None,
        relation: str | None = None,
        **_: Any,
    ) -> str:
        self._num_words = num_words
        if relation is None:
            relation = "at least"
        elif relation not in ("less than", "at least"):
            raise ValueError(
                "The supported relation for comparison must be in "
                f"('less than', 'at least'), but {relation} is given."
            )
        self._comparison_relation = relation
        self._description = f"Answer with {relation} {num_words} words."
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_words", "relation"]

    def check_following(self, value: str) -> bool:
        num_words = count_words(value)
        if self._comparison_relation == "less than":
            return num_words < self._num_words
        return num_words >= self._num_words


# --------------------------------------------------------------------------- #
# keywords
# --------------------------------------------------------------------------- #


class KeywordChecker(Instruction):
    """All the required keywords must appear."""

    def build_description(self, *, keywords: Sequence[str] | None = None, **_: Any) -> str:
        self._keywords = sorted(keywords or [])
        self._description = f"Include keywords {self._keywords} in the response."
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {"keywords": self._keywords}

    def get_instruction_args_keys(self) -> list[str]:
        return ["keywords"]

    def check_following(self, value: str) -> bool:
        for keyword in self._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return False
        return True


class KeywordFrequencyChecker(Instruction):
    """A keyword must appear at least / less than N times."""

    def build_description(
        self,
        *,
        keyword: str | None = None,
        frequency: int | None = None,
        relation: str | None = None,
        **_: Any,
    ) -> str:
        self._keyword = keyword.strip() if keyword else keyword
        self._frequency = frequency
        if relation is None:
            relation = "at least"
        elif relation not in ("less than", "at least"):
            raise ValueError(
                "The supported relation for comparison must be in "
                f"('less than', 'at least'), but {relation} is given."
            )
        self._comparison_relation = relation
        self._description = (
            f"In your response, the word {self._keyword} should appear "
            f"{relation} {frequency} times."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {
            "keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self) -> list[str]:
        return ["keyword", "frequency", "relation"]

    def check_following(self, value: str) -> bool:
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))
        if self._comparison_relation == "less than":
            return actual_occurrences < self._frequency
        return actual_occurrences >= self._frequency


class ForbiddenWords(Instruction):
    """None of the forbidden words may appear."""

    def build_description(
        self, *, forbidden_words: Sequence[str] | None = None, **_: Any
    ) -> str:
        self._forbidden_words = sorted(set(forbidden_words or []))
        self._description = (
            f"Do not include keywords {self._forbidden_words} in the response."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {"forbidden_words": self._forbidden_words}

    def get_instruction_args_keys(self) -> list[str]:
        return ["forbidden_words"]

    def check_following(self, value: str) -> bool:
        for word in self._forbidden_words:
            if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
                return False
        return True


class LetterFrequencyChecker(Instruction):
    """A character must appear at least / less than N times.

    The reference implementation accepts only ``a``-``z`` and *randomises* the
    letter otherwise -- but the dataset contains prompts that ask for ``#``
    and ``!``, so the reference silently measures a random letter instead of
    the requested one.  This implementation counts the requested character,
    which is what the instruction says, and never randomises.
    """

    def build_description(
        self,
        *,
        letter: str | None = None,
        let_frequency: int | None = None,
        let_relation: str | None = None,
        **_: Any,
    ) -> str:
        if not letter or len(letter.strip()) != 1:
            raise ValueError(f"letter must be a single character, got {letter!r}")
        self._letter = letter.strip().lower()
        self._frequency = let_frequency
        if let_relation is None:
            let_relation = "at least"
        elif let_relation not in ("less than", "at least"):
            raise ValueError(
                "The supported relation for comparison must be in "
                f"('less than', 'at least'), but {let_relation} is given."
            )
        self._comparison_relation = let_relation
        self._description = (
            f"In your response, the letter {self._letter} should appear "
            f"{let_relation} {let_frequency} times."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self) -> list[str]:
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value: str) -> bool:
        value = value.lower()
        if self._comparison_relation == "less than":
            return value.count(self._letter) < self._frequency
        return value.count(self._letter) >= self._frequency


# --------------------------------------------------------------------------- #
# detectable format
# --------------------------------------------------------------------------- #


class BulletListChecker(Instruction):
    """Exactly ``num_bullets`` markdown bullet points."""

    def build_description(self, *, num_bullets: int | None = None, **_: Any) -> str:
        self._num_bullets = num_bullets
        self._description = (
            f"Your answer must contain exactly {num_bullets} bullet points. "
            "Use the markdown bullet points such as:\n"
            "* This is point 1. \n"
            "* This is point 2"
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {"num_bullets": self._num_bullets}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_bullets"]

    def check_following(self, value: str) -> bool:
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        return len(bullet_lists) + len(bullet_lists_2) == self._num_bullets


class ConstrainedResponseChecker(Instruction):
    """The response must contain one of the three allowed option phrases."""

    OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")

    def build_description(self, **_: Any) -> str:
        self._description = (
            f"Answer with one of the following options: {self.OPTIONS}"
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return None

    def get_instruction_args_keys(self) -> list[str]:
        return []

    def check_following(self, value: str) -> bool:
        value = value.strip()
        for constrained_response in self.OPTIONS:
            if constrained_response in value:
                return True
        return False


class HighlightSectionChecker(Instruction):
    """At least ``num_highlights`` ``*highlighted*`` sections."""

    def build_description(
        self, *, num_highlights: int | None = None, **_: Any
    ) -> str:
        self._num_highlights = num_highlights
        self._description = (
            f"Highlight at least {num_highlights} sections in your answer "
            "with markdown, i.e. *highlighted section*."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {"num_highlights": self._num_highlights}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_highlights"]

    def check_following(self, value: str) -> bool:
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1
        return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
    """At least ``num_sections`` sections marked with ``<splitter> N``."""

    def build_description(
        self,
        *,
        section_spliter: str | None = None,
        num_sections: int | None = None,
        **_: Any,
    ) -> str:
        self._section_spliter = (
            section_spliter.strip() if isinstance(section_spliter, str) else section_spliter
        )
        if self._section_spliter is None:
            self._section_spliter = "Section"
        self._num_sections = num_sections
        self._description = (
            f"Your response must have {num_sections} sections. Mark the beginning "
            f"of each section with {self._section_spliter} X, such as:\n"
            f"{self._section_spliter} 1\n"
            "[content of section 1]\n"
            f"{self._section_spliter} 2\n"
            "[content of section 2]"
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
        }

    def get_instruction_args_keys(self) -> list[str]:
        return ["section_spliter", "num_sections"]

    def check_following(self, value: str) -> bool:
        section_splitter_pattern = r"\s?" + self._section_spliter + r"\s?\d+\s?"
        sections = re.split(section_splitter_pattern, value)
        return len(sections) - 1 >= self._num_sections


class JsonFormat(Instruction):
    """The whole response must be valid JSON (markdown fences allowed)."""

    def build_description(self, **_: Any) -> str:
        self._description = (
            "Entire output should be wrapped in JSON format. "
            "You can use markdown ticks such as ```."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return None

    def get_instruction_args_keys(self) -> list[str]:
        return []

    def check_following(self, value: str) -> bool:
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError:
            return False
        return True


class TitleChecker(Instruction):
    """The response must contain a ``<<title>>``."""

    def build_description(self, **_: Any) -> str:
        self._description = (
            "Your answer must contain a title, wrapped in double angular "
            "brackets, such as <<poem of joy>>."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return None

    def get_instruction_args_keys(self) -> list[str]:
        return []

    def check_following(self, value: str) -> bool:
        titles = re.findall(r"<<[^\n]+>>", value)
        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


# --------------------------------------------------------------------------- #
# detectable content
# --------------------------------------------------------------------------- #


class PlaceholderChecker(Instruction):
    """At least ``num_placeholders`` ``[square bracket]`` placeholders."""

    def build_description(
        self, *, num_placeholders: int | None = None, **_: Any
    ) -> str:
        self._num_placeholders = num_placeholders
        self._description = (
            f"The response must contain at least {num_placeholders} placeholders "
            "represented by square brackets, such as [address]."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {"num_placeholders": self._num_placeholders}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_placeholders"]

    def check_following(self, value: str) -> bool:
        placeholders = re.findall(r"\[.*?\]", value)
        return len(placeholders) >= self._num_placeholders


class PostscriptChecker(Instruction):
    """The response must contain a postscript starting with the marker."""

    def build_description(
        self, *, postscript_marker: str | None = None, **_: Any
    ) -> str:
        self._postscript_marker = (
            postscript_marker.strip()
            if isinstance(postscript_marker, str)
            else postscript_marker
        )
        if self._postscript_marker is None:
            self._postscript_marker = "P.S."
        self._description = (
            "At the end of your response, please explicitly add a postscript "
            f"starting with {self._postscript_marker}"
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self) -> list[str]:
        return ["postscript_marker"]

    def check_following(self, value: str) -> bool:
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return bool(postscript)


# --------------------------------------------------------------------------- #
# start / end
# --------------------------------------------------------------------------- #


class QuotationChecker(Instruction):
    """The whole response must be wrapped in double quotation marks."""

    def build_description(self, **_: Any) -> str:
        self._description = "Wrap your entire response with double quotation marks."
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return None

    def get_instruction_args_keys(self) -> list[str]:
        return []

    def check_following(self, value: str) -> bool:
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'


class EndChecker(Instruction):
    """The response must end with the exact phrase."""

    def build_description(self, *, end_phrase: str | None = None, **_: Any) -> str:
        self._end_phrase = (
            end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        if self._end_phrase is None:
            self._end_phrase = "Is there anything else I can help with?"
        self._description = (
            f"Finish your response with this exact phrase {self._end_phrase}. "
            "No other words should follow this phrase."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self) -> list[str]:
        return ["end_phrase"]

    def check_following(self, value: str) -> bool:
        value = value.strip().strip("\"").lower()
        return value.endswith(self._end_phrase.strip().lower())


# --------------------------------------------------------------------------- #
# change case
# --------------------------------------------------------------------------- #


class CapitalLettersEnglishChecker(Instruction):
    """The response must be English and entirely uppercase."""

    def build_description(self, **_: Any) -> str:
        self._description = (
            "Your entire response should be in English, and in all capital letters."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return None

    def get_instruction_args_keys(self) -> list[str]:
        return []

    def check_following(self, value: str) -> bool:
        try:
            return value.isupper() and _looks_english(value)
        except Exception:
            return True


class LowercaseLettersEnglishChecker(Instruction):
    """The response must be English and entirely lowercase."""

    def build_description(self, **_: Any) -> str:
        self._description = (
            "Your entire response should be in English, and in all lowercase "
            "letters. No capital letters are allowed."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return None

    def get_instruction_args_keys(self) -> list[str]:
        return []

    def check_following(self, value: str) -> bool:
        try:
            return value.islower() and _looks_english(value)
        except Exception:
            return True


class CapitalWordFrequencyChecker(Instruction):
    """Number of fully-capitalised words, with an ``at least`` / ``less than`` relation."""

    def build_description(
        self,
        *,
        capital_frequency: int | None = None,
        capital_relation: str | None = None,
        **_: Any,
    ) -> str:
        self._frequency = capital_frequency
        if capital_relation is None:
            capital_relation = "at least"
        elif capital_relation not in ("less than", "at least"):
            raise ValueError(
                "The supported relation for comparison must be in "
                f"('less than', 'at least'), but {capital_relation} is given."
            )
        self._comparison_relation = capital_relation
        self._description = (
            "In your response, words with all capital letters should appear "
            f"{capital_relation} {capital_frequency} times."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self) -> list[str]:
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value: str) -> bool:
        words = _word_tokens(value)
        capital_words = len([word for word in words if word.isupper()])
        if self._comparison_relation == "less than":
            return capital_words < self._frequency
        return capital_words >= self._frequency


# --------------------------------------------------------------------------- #
# combination
# --------------------------------------------------------------------------- #


class TwoResponsesChecker(Instruction):
    """Two different responses separated by ``******``."""

    def build_description(self, **_: Any) -> str:
        self._description = (
            "Give two different responses. Responses and only responses should "
            "be separated by 6 asterisk symbols: ******."
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return None

    def get_instruction_args_keys(self) -> list[str]:
        return []

    def check_following(self, value: str) -> bool:
        valid_responses = []
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )


class RepeatPromptThenAnswer(Instruction):
    """The response must start with the request, verbatim."""

    def build_description(self, *, prompt_to_repeat: str | None = None, **_: Any) -> str:
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        self._prompt_to_repeat = prompt_to_repeat
        self._description = (
            "First repeat the request word for word without change, then give "
            "your answer (1. do not say any words or characters before "
            "repeating the request; 2. the request you need to repeat does not "
            "include this sentence)"
        )
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self) -> list[str]:
        return ["prompt_to_repeat"]

    def check_following(self, value: str) -> bool:
        return value.strip().lower().startswith(self._prompt_to_repeat.strip().lower())


# --------------------------------------------------------------------------- #
# punctuation
# --------------------------------------------------------------------------- #


class CommaChecker(Instruction):
    """No comma may appear anywhere in the response."""

    def build_description(self, **_: Any) -> str:
        self._description = "In your entire response, refrain from the use of any commas."
        return self._description

    def get_instruction_args(self) -> dict[str, Any] | None:
        return None

    def get_instruction_args_keys(self) -> list[str]:
        return []

    def check_following(self, value: str) -> bool:
        return not re.search(r"\,", value)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

INSTRUCTION_DICT: dict[str, type[Instruction]] = {
    "keywords:existence": KeywordChecker,
    "keywords:frequency": KeywordFrequencyChecker,
    "keywords:forbidden_words": ForbiddenWords,
    "keywords:letter_frequency": LetterFrequencyChecker,
    "language:response_language": ResponseLanguageChecker,
    "length_constraints:number_sentences": NumberOfSentences,
    "length_constraints:number_paragraphs": ParagraphChecker,
    "length_constraints:number_words": NumberOfWords,
    "length_constraints:nth_paragraph_first_word": ParagraphFirstWordCheck,
    "detectable_content:number_placeholders": PlaceholderChecker,
    "detectable_content:postscript": PostscriptChecker,
    "detectable_format:number_bullet_lists": BulletListChecker,
    "detectable_format:constrained_response": ConstrainedResponseChecker,
    "detectable_format:number_highlighted_sections": HighlightSectionChecker,
    "detectable_format:multiple_sections": SectionChecker,
    "detectable_format:json_format": JsonFormat,
    "detectable_format:title": TitleChecker,
    "combination:two_responses": TwoResponsesChecker,
    "combination:repeat_prompt": RepeatPromptThenAnswer,
    "startend:end_checker": EndChecker,
    "change_case:capital_word_frequency": CapitalWordFrequencyChecker,
    "change_case:english_capital": CapitalLettersEnglishChecker,
    "change_case:english_lowercase": LowercaseLettersEnglishChecker,
    "punctuation:no_comma": CommaChecker,
    "startend:quotation": QuotationChecker,
}


def check_prompt_instances(
    prompt: str,
    response: str,
    instruction_ids: Iterable[str],
    kwargs_list: Iterable[dict[str, Any]],
) -> list[bool]:
    """Check a response against every instruction instance, in order.

    A prompt may carry the *same* instruction id twice with different
    parameters (e.g. ``capital_word_frequency`` with "at least 1" and "less
    than 10"); every occurrence is a separate check and is reported
    separately here.
    """
    ids = list(instruction_ids)
    kwargs_items = list(kwargs_list)
    outcomes: list[bool] = []
    for index, instruction_id in enumerate(ids):
        cls = INSTRUCTION_DICT.get(instruction_id)
        if cls is None:
            outcomes.append(False)
            continue
        kwargs = dict(kwargs_items[index]) if index < len(kwargs_items) else {}
        instruction = cls(instruction_id)
        try:
            instruction.build_description(**kwargs)
            required = instruction.get_instruction_args_keys()
            if any(key not in kwargs for key in required):
                outcomes.append(False)
                continue
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)
            outcomes.append(bool(instruction.check_following(response)))
        except Exception:
            outcomes.append(False)
    return outcomes


def check_prompt(
    prompt: str,
    response: str,
    instruction_ids: Iterable[str],
    kwargs_list: Iterable[dict[str, Any]],
) -> dict[str, bool]:
    """Check a response against a list of instructions (the strict metric).

    A prompt is followed only when **every** one of its instructions is
    followed -- this is IFEval's *strict accuracy*.  When the same instruction
    id occurs more than once the results are combined with a logical AND, so
    the returned value stays meaningful for the strict metric.

    An instruction whose required kwargs are missing is reported as *not
    followed* (the reference implementation would randomise the missing
    parameter, which is not acceptable for a benchmark reward).

    Apart from that the behaviour is identical to the reference
    implementation, including for empty responses; the "the agent must write
    something" rule lives in the task verifier, not in the checker.
    """
    ids = list(instruction_ids)
    outcomes = check_prompt_instances(prompt, response, ids, kwargs_list)
    results: dict[str, bool] = {}
    for instruction_id, outcome in zip(ids, outcomes):
        results[instruction_id] = results.get(instruction_id, True) and outcome
    return results
