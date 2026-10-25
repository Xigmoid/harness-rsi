"""L2: cross-entropy method over the continuous fields.

The continuous knobs (``contract.max_tokens``, ``runtime.max_iterations``,
``runtime.context_keep_recent``, ...) live in a bounded box.  CEM keeps a
Gaussian over that box, samples a population, keeps the elite, and refits the
mean and the variance.  It is the right tool for the fields where the bandit
has no notion of "direction": an int knob has an ordering, and CEM uses it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from ..fields import expand_fields
from ..schema import HarnessSpec, get_path, set_path
from .base import Observation, Optimizer, Proposal

__all__ = ["CEM"]


@dataclass
class _Variable:
    path: str
    lo: int
    hi: int
    mean: float
    std: float

    def clamp(self, value: float) -> int:
        return int(max(self.lo, min(self.hi, round(value))))


class CEM(Optimizer):
    """Cross-entropy method over the continuous (integer) fields."""

    name = "cem"

    def __init__(
        self,
        *,
        max_edits: int = 1,
        seed: int = 0,
        population: int = 8,
        elite_frac: float = 0.25,
        iterations: int = 3,
    ):
        super().__init__(max_edits=max_edits, seed=seed)
        self.population = population
        self.elite_frac = elite_frac
        self.iterations = iterations
        self._vars: dict[str, _Variable] = {}
        self._iteration = 0

    # -- distribution ------------------------------------------------------- #

    def _variables(self, spec: HarnessSpec) -> dict[str, _Variable]:
        out: dict[str, _Variable] = {}
        for f in expand_fields(spec):
            if not f.mutable or f.kind != "int" or not f.domain:
                continue
            lo, hi = int(f.domain[0]), int(f.domain[1])
            current = get_path(spec.to_dict(), f.path)
            current = int(current) if isinstance(current, (int, float)) else lo
            var = self._vars.get(f.path)
            if var is None:
                var = _Variable(path=f.path, lo=lo, hi=hi, mean=float(current), std=(hi - lo) / 4 or 1.0)
                self._vars[f.path] = var
            else:
                var.lo, var.hi = lo, hi
            out[f.path] = var
        return out

    def _sample(self, variables: dict[str, _Variable]) -> dict[str, int]:
        return {
            path: var.clamp(self.rng.gauss(var.mean, var.std))
            for path, var in variables.items()
        }

    def _fit(self, variables: dict[str, _Variable], elites: Sequence[dict[str, int]]) -> None:
        if not elites:
            return
        for path, var in variables.items():
            values = [float(e[path]) for e in elites if path in e]
            if not values:
                continue
            mean = sum(values) / len(values)
            var_ = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
            var.mean = mean
            var.std = max(1.0, math.sqrt(var_))

    # -- proposing ---------------------------------------------------------- #

    def propose(
        self,
        spec: HarnessSpec,
        history: Sequence[Observation],
    ) -> Proposal:
        variables = self._variables(spec)
        if not variables:
            raise ValueError("the spec has no mutable continuous fields")

        parent = history[-1].generation if history else spec.version
        candidates: list[tuple[float, dict[str, int]]] = []
        for _ in range(self.population):
            candidates.append((0.0, self._sample(variables)))
        # Without a scoring signal (offline use), fall back to the current mean.
        best = min(candidates, key=lambda c: abs(c[1].get(next(iter(variables)), 0)))
        sample = best[1]

        # The edit budget is a hard constraint: at most ``max_edits`` fields may
        # move.  The variables that are furthest from their mean are the ones
        # the distribution wants to move most, so they are the ones to spend
        # the budget on.
        deviating = sorted(
            (
                (abs(sample[path] - var.mean), path)
                for path, var in variables.items()
                if sample.get(path) is not None
                and sample[path] != get_path(spec.to_dict(), path)
            ),
            reverse=True,
        )
        selected = [path for _, path in deviating[: max(1, self.max_edits)]]

        current = spec
        applied: list[str] = []
        for path in selected:
            value = sample[path]
            existing = get_path(spec.to_dict(), path)
            if existing == value:
                continue
            try:
                current = HarnessSpec.from_dict(set_path(current.to_dict(), path, value))
            except Exception:
                continue
            applied.append(path)

        if not applied:
            # Force at least one change: move the variable furthest from its mean.
            path = deviating[0][1] if deviating else max(variables)
            var = variables[path]
            value = var.clamp(var.mean + (1 if var.mean <= var.hi / 2 else -1))
            try:
                current = HarnessSpec.from_dict(set_path(current.to_dict(), path, value))
                applied.append(path)
            except Exception:
                pass
        if not applied:
            raise ValueError("CEM could not produce a valid mutation")

        self.check_budget(applied)
        self._iteration += 1
        return Proposal(
            spec=current,
            parent=parent,
            changed_fields=tuple(applied),
            rationale=f"CEM sample of {len(applied)} continuous field(s)",
            strategy=self.name,
        )

    def update(self, history: Sequence[Observation]) -> None:
        """Refit the Gaussian from the elite of the observed generations."""
        if not history:
            return
        spec = history[-1].spec
        variables = self._variables(spec)
        elites = [obs for obs in history if obs.changed_fields]
        elites.sort(key=lambda o: o.mean_score, reverse=True)
        n_elite = max(1, int(len(elites) * self.elite_frac))
        self._fit(variables, [self._values_of(obs) for obs in elites[:n_elite]])

    @staticmethod
    def _values_of(obs: Observation) -> dict[str, int]:
        out: dict[str, int] = {}
        for path in obs.changed_fields:
            try:
                value = get_path(obs.spec.to_dict(), path)
            except Exception:
                continue
            if isinstance(value, (int, float)):
                out[path] = int(value)
        return out
