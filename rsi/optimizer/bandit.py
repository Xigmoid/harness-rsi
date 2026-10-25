"""L1: per-field bandit.

Every field of the registry is treated as an "arm".  When a generation
changed exactly one field, the score delta of that generation is a noisy
estimate of the value of that arm.  The bandit then decides which field to
turn next.

The nice property of this level is that it *is* the attribution table: the
per-field estimates are exactly the effect sizes the study reports.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from ..fields import expand_fields
from ..schema import HarnessSpec
from .base import Observation, Optimizer, Proposal

__all__ = ["Bandit", "ArmStat", "ThompsonBandit"]


@dataclass
class ArmStat:
    """Accumulated evidence about one field."""

    path: str
    trials: int = 0
    total_delta: float = 0.0
    total_delta_sq: float = 0.0
    wins: int = 0
    losses: int = 0

    @property
    def mean(self) -> float:
        return self.total_delta / self.trials if self.trials else 0.0

    @property
    def variance(self) -> float:
        if self.trials < 2:
            return 0.0
        mean = self.mean
        return max(0.0, self.total_delta_sq / self.trials - mean * mean)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "trials": self.trials,
            "mean_delta": self.mean,
            "sd": math.sqrt(self.variance),
            "wins": self.wins,
            "losses": self.losses,
        }


class Bandit(Optimizer):
    """UCB1 over the per-field effect sizes."""

    name = "bandit_ucb"

    def __init__(self, *, max_edits: int = 1, seed: int = 0, c: float = 1.4):
        super().__init__(max_edits=max_edits, seed=seed)
        self.c = c
        self.arms: dict[str, ArmStat] = {}

    # -- learning ----------------------------------------------------------- #

    def observe(self, history: Sequence[Observation]) -> None:
        """Update the arms from the observed generations.

        Only generations that changed **exactly one** field are used: with two
        changed fields the delta cannot be attributed to either of them.
        """
        by_generation = {obs.generation: obs for obs in history}
        for obs in history:
            if obs.parent is None or obs.parent not in by_generation:
                continue
            parent = by_generation[obs.parent]
            delta = obs.mean_score - parent.mean_score
            if len(obs.changed_fields) != 1:
                continue
            path = obs.changed_fields[0]
            stat = self.arms.setdefault(path, ArmStat(path=path))
            stat.trials += 1
            stat.total_delta += delta
            stat.total_delta_sq += delta * delta
            if delta > 0:
                stat.wins += 1
            elif delta < 0:
                stat.losses += 1

    def _ucb(self, stat: ArmStat, total: int) -> float:
        if stat.trials == 0:
            return float("inf")
        return stat.mean + self.c * math.sqrt(math.log(max(2, total)) / stat.trials)

    def values(self, spec: HarnessSpec) -> dict[str, float]:
        """Current UCB value of every mutable field of ``spec``."""
        total = sum(s.trials for s in self.arms.values()) + 1
        out: dict[str, float] = {}
        for f in expand_fields(spec):
            if not f.mutable:
                continue
            stat = self.arms.get(f.path, ArmStat(path=f.path))
            out[f.path] = self._ucb(stat, total)
        return out

    # -- proposing ---------------------------------------------------------- #

    def propose(
        self,
        spec: HarnessSpec,
        history: Sequence[Observation],
    ) -> Proposal:
        self.observe(history)
        values = self.values(spec)
        if not values:
            raise ValueError("the spec has no mutable fields")
        # Never propose a value the field already has: the bandit would spin.
        best_path = max(values, key=lambda p: (values[p], -len(p)))
        parent = history[-1].generation if history else spec.version
        return self.mutate(
            spec,
            [best_path],
            strategy=self.name,
            rationale=(
                f"UCB chose {best_path} (value={values[best_path]:.4f}, "
                f"trials={self.arms[best_path].trials if best_path in self.arms else 0})"
            ),
            parent=parent,
        )

    def effect_table(self) -> list[dict]:
        """The per-field effect sizes, sorted by mean effect."""
        rows = [s.to_dict() for s in self.arms.values() if s.trials > 0]
        rows.sort(key=lambda r: r["mean_delta"], reverse=True)
        return rows


class ThompsonBandit(Bandit):
    """Beta-Bernoulli bandit over the sign of the effect.

    Useful when the score deltas are dominated by noise: only the *direction*
    of an edit is trusted, not its magnitude.
    """

    name = "bandit_thompson"

    def __init__(self, *, max_edits: int = 1, seed: int = 0, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(max_edits=max_edits, seed=seed)
        self.alpha = alpha
        self.beta = beta

    def _ucb(self, stat: ArmStat, total: int) -> float:
        if stat.trials == 0:
            return float("inf")
        wins = stat.wins + self.alpha
        losses = stat.losses + self.beta
        return random.betavariate(wins, losses)
