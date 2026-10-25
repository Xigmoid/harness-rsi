"""The row schema: which dataset, which model, which harness.

Every annotated rollout carries these three keys, because a trajectory is
only interpretable together with them:

``dataset``
    Which task set the rollout answers -- ``ClawGym`` or ``Openthought``.

``model``
    Which model produced it -- ``GLM-5.3-Flash`` or ``MiniMax-M3.1-Flash``.
    The run is served by more than one endpoint, and the two are not
    interchangeable: a dataset row that does not say which model wrote it
    cannot be filtered, sampled or audited.

``harness``
    Which agent harness produced it -- ``Mini`` (mini-swe-agent) or
    ``Opencode``.  The harness is the variable the whole experiment is
    about, so it is named explicitly rather than inferred from the shape
    of the messages.

The canonical names are short and stable.  The raw provider ids are kept
next to them (``model_provider``, ``model_id``) so that a row can be traced
back to the endpoint that answered, and so that a third model does not
need a new vocabulary.
"""

from __future__ import annotations

from typing import Any

#: The two task sets.
DATASET_CLAWGYM = "ClawGym"
DATASET_OPENTHOUGHT = "Openthought"

#: The two harnesses, in the words the experiment uses.
HARNESS_MINI = "Mini"
HARNESS_OPENCODE = "Opencode"

#: The two annotation models.
MODEL_GLM = "GLM-5.3-Flash"
MODEL_MINIMAX = "MiniMax-M3.1-Flash"

#: The keys every row carries.
REQUIRED_KEYS = ("dataset", "model", "harness", "messages", "conversations")


def canonical_harness(value: str | None) -> str:
    """``mini-swe-agent`` / ``opencode`` / ``Mini`` -> ``Mini`` / ``Opencode``."""
    text = (value or "").strip().lower()
    if "opencode" in text:
        return HARNESS_OPENCODE
    return HARNESS_MINI


def canonical_model(
    model_id: str | None = None, provider: str | None = None
) -> str:
    """The provider's own model name -> the canonical one.

    The id is matched on what it *is*, not on where it came from: a
    provider id of ``glm5.3-flash`` and a model name of
    ``GLM-5.3-Flash`` are the same model, and the row should not depend on
    which spelling the endpoint happened to use.
    """
    haystack = f"{model_id or ''} {provider or ''}".lower()
    if "minimax" in haystack or "m3.1" in haystack:
        return MODEL_MINIMAX
    if "glm" in haystack:
        return MODEL_GLM
    return (model_id or provider or "unknown").strip()


def canonical_dataset(family: str | None) -> str:
    """The registry's family name -> the dataset name."""
    text = (family or "").strip().lower()
    if "clawgym" in text or "arc" in text or "persona" in text:
        return DATASET_CLAWGYM
    return DATASET_OPENTHOUGHT


def provenance(
    dataset: str,
    harness: str,
    model: str,
    model_id: str | None = None,
    model_provider: str | None = None,
) -> dict[str, Any]:
    """The three keys, plus the raw provider ids.

    The three keys are what a trainer filters on; the raw ids are what an
    auditor follows.
    """
    return {
        "dataset": dataset,
        "harness": harness,
        "model": model,
        "model_id": model_id,
        "model_provider": model_provider,
    }
