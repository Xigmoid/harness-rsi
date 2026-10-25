"""Optimizer interface.

An optimizer proposes the *next* generation.  It never runs a benchmark itself
and it never writes a spec to disk: the caller owns the evaluation, the
acceptance decision and the ledger.  That separation is what makes the
"半自动,人工审阅每代 diff" protocol of the plan enforceable.

Every proposal is constrained by the **edit budget** (constraint C-5): a
generation may change at most ``max_edits`` fields, so that a score difference
can be attributed to a small, nameable set of changes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..schema import HarnessSpec

__all__ = ["Proposal", "Optimizer", "EditBudgetExceeded", "Observation"]


class EditBudgetExceeded(ValueError):
    """Raised when a proposal changes more fields than the budget allows."""


@dataclass
class Observation:
    """What the optimizer knows about one evaluated generation."""

    generation: int
    spec: HarnessSpec
    mean_score: float
    parent: int | None = None
    accepted: bool = True
    changed_fields: tuple[str, ...] = ()
    per_task: Mapping[str, float] = field(default_factory=dict)


@dataclass
class Proposal:
    """A candidate next generation."""

    spec: HarnessSpec
    parent: int
    changed_fields: tuple[str, ...]
    rationale: str = ""
    strategy: str = ""

    def __post_init__(self) -> None:
        if not self.changed_fields:
            raise ValueError("a proposal must change at least one field")


class Optimizer:
    """Base class of all search strategies."""

    name = "base"

    def __init__(self, *, max_edits: int = 2, seed: int = 0):
        if max_edits < 1:
            raise ValueError("max_edits must be >= 1")
        self.max_edits = max_edits
        self.rng = random.Random(seed)

    # -- to be implemented -------------------------------------------------- #

    def propose(
        self,
        spec: HarnessSpec,
        history: Sequence[Observation],
    ) -> Proposal:
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------ #

    def check_budget(self, changed: Sequence[str]) -> None:
        if len(changed) > self.max_edits:
            raise EditBudgetExceeded(
                f"proposal changes {len(changed)} fields ({list(changed)}), "
                f"budget is {self.max_edits} (constraint C-5)"
            )

    def mutate(
        self,
        spec: HarnessSpec,
        paths: Sequence[str],
        *,
        strategy: str = "",
        rationale: str = "",
        parent: int = 0,
    ) -> Proposal:
        """Apply a random edit to each of ``paths`` and return the proposal.

        A field that cannot be mutated into a *valid* spec (rewiring a hop
        usually breaks the workflow) is skipped rather than fatal: the search
        must not die because one knob happens to be locked.  If none of the
        requested paths is mutable, any other mutable field is used instead.
        """
        from ..fields import expand_fields, mutate

        self.check_budget(paths)
        current = spec
        applied: list[str] = []
        skipped: list[str] = []
        for path in paths:
            new_spec = None
            for _ in range(30):
                try:
                    new_spec = mutate(current, path, self.rng)
                    break
                except Exception:
                    continue
            if new_spec is None:
                skipped.append(path)
                continue
            current = new_spec
            applied.append(path)

        if not applied:
            # Fall back to any other mutable field.
            for candidate in expand_fields(spec):
                if not candidate.mutable or candidate.path in skipped:
                    continue
                for _ in range(10):
                    try:
                        current = mutate(spec, candidate.path, self.rng)
                        applied.append(candidate.path)
                        break
                    except Exception:
                        continue
                if applied:
                    break

        if not applied:
            raise EditBudgetExceeded(
                f"no admissible mutation found for {list(paths)}"
            )
        note = f" (skipped {', '.join(skipped)})" if skipped else ""
        return Proposal(
            spec=current,
            parent=parent,
            changed_fields=tuple(applied),
            rationale=f"{rationale}{note}".strip(),
            strategy=strategy or self.name,
        )
