"""L0: random search over the typed configuration space.

This is the *baseline* the whole study needs.  If random configurations already
raise the score, a sophisticated optimizer is not what produced the gain --
and the amount by which random search improves the harness calibrates how
large the configuration space effect is at all (contribution point 2 of the
plan).
"""

from __future__ import annotations

from typing import Sequence

from ..fields import expand_fields
from ..schema import HarnessSpec
from .base import Observation, Optimizer, Proposal

__all__ = ["RandomSearch"]


class RandomSearch(Optimizer):
    """Proposes a random edit of 1--2 fields, preferring high-priority groups."""

    name = "random_search"

    def __init__(self, *, max_edits: int = 2, seed: int = 0, priority_bias: float = 0.7):
        super().__init__(max_edits=max_edits, seed=seed)
        self.priority_bias = priority_bias

    def propose(
        self,
        spec: HarnessSpec,
        history: Sequence[Observation],
    ) -> Proposal:
        fields = [f for f in expand_fields(spec) if f.mutable]
        if not fields:
            raise ValueError("the spec has no mutable fields")

        n_edits = 1 if self.rng.random() < 0.4 else min(self.max_edits, 2)
        chosen: list[str] = []
        pool = list(fields)
        for _ in range(n_edits):
            if not pool:
                break
            paths = [f.path for f in pool]
            # The weights must be drawn from the *same* population as the
            # paths, otherwise the sampling fails.
            weights = [
                self.priority_bias if f.priority == 0 else 1.0 / (f.priority + 1) for f in pool
            ]
            path = self.rng.choices(paths, weights=weights, k=1)[0]
            chosen.append(path)
            pool = [f for f in pool if f.path != path]

        parent = history[-1].generation if history else spec.version
        proposal = self.mutate(
            spec,
            chosen,
            strategy=self.name,
            rationale=f"random edit of {len(chosen)} field(s)",
            parent=parent,
        )
        return proposal
