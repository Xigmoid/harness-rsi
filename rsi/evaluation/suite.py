"""Evaluating one generation on a whole suite of benchmarks.

A *suite* is a set of benchmarks that share a capability (or the union of
all of them).  Two objectives are supported:

``specialize``
    The optimizer's objective is the composite of one suite.  Running the
    loop once per capability family therefore produces one *specialised*
    harness per family.  This is the default, because a single objective
    keeps the paired statistics sharp and because RHI's own ablation suggests
    the optimal contract is task-dependent -- specialising is what makes that
    hypothesis testable.

``joint``
    The objective is the composite over the whole suite, i.e. one harness is
    optimized for every capability at once.  This is the *control* experiment:
    it answers how much a single generic harness can do, and it is the
    counterpart of the random-search baseline.

Either way the per-benchmark scores are always recorded separately, so a
joint run can be broken down per capability afterwards, and the transfer
matrix can be built from the same data.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..benchmarks import BenchmarkSpec, suite_benchmarks
from ..schema import HarnessSpec
from .runner import Backend, EvalResult
from .subsets import Subset

__all__ = ["SuiteResult", "run_suite", "composite_score"]


def composite_score(
    per_benchmark: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """The composite objective of a suite.

    The default is the *unweighted mean of the per-benchmark mean scores*.
    An unweighted mean is deliberate: weighting by task count would let the
    biggest benchmark decide the direction of the search, and weighting by
    inverse variance would reward the cheapest benchmark.  The noise of a
    composite is the sum of the per-benchmark variances, so the composite's
    resolution is always worse than any single member's -- another reason
    the specialised mode is the default.
    """
    if not per_benchmark:
        return 0.0
    if weights:
        total_weight = sum(weights.get(k, 0.0) for k in per_benchmark)
        if total_weight <= 0:
            return statistics.fmean(per_benchmark.values())
        return (
            sum(per_benchmark[k] * weights.get(k, 0.0) for k in per_benchmark)
            / total_weight
        )
    return statistics.fmean(per_benchmark.values())


@dataclass
class SuiteResult:
    """The outcome of evaluating one generation on a suite."""

    suite: str
    benchmark_results: dict[str, EvalResult] = field(default_factory=dict)
    generation: int | None = None
    spec_sha256: str = ""
    weights: dict[str, float] = field(default_factory=dict)

    @property
    def per_benchmark_score(self) -> dict[str, float]:
        return {k: v.mean_score for k, v in self.benchmark_results.items()}

    @property
    def composite(self) -> float:
        return composite_score(self.per_benchmark_score, weights=self.weights)

    @property
    def n_runs(self) -> int:
        return sum(v.n_runs for v in self.benchmark_results.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(v.total_cost_usd for v in self.benchmark_results.values())

    def per_capability(self, capabilities: Mapping[str, str]) -> dict[str, float]:
        """Aggregate the benchmark scores by capability."""
        buckets: dict[str, list[float]] = {}
        for key, result in self.benchmark_results.items():
            capability = capabilities.get(key, key)
            buckets.setdefault(capability, []).append(result.mean_score)
        return {k: statistics.fmean(v) for k, v in buckets.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "generation": self.generation,
            "spec_sha256": self.spec_sha256,
            "per_benchmark": self.per_benchmark_score,
            "composite": self.composite,
            "n_runs": self.n_runs,
            "total_cost_usd": self.total_cost_usd,
            "benchmarks": {k: v.to_dict() for k, v in self.benchmark_results.items()},
        }


def run_suite(
    spec: HarnessSpec,
    suite: str,
    subsets: Mapping[str, Subset],
    seeds: Sequence[int],
    backend: Backend,
    workdir: Path,
    *,
    generation: int | None = None,
    base_config: Mapping[str, Any] | None = None,
    weights: Mapping[str, float] | None = None,
    benchmark_backends: Mapping[str, Backend] | None = None,
) -> SuiteResult:
    """Evaluate ``spec`` on every benchmark of ``suite``.

    ``benchmark_backends`` allows a per-benchmark backend (e.g. Harbor for
    the container benchmarks and something cheaper for the light ones); by
    default the same backend is used everywhere.
    """
    result = SuiteResult(
        suite=suite,
        generation=generation,
        spec_sha256=spec.spec_sha256(),
        weights=dict(weights or {}),
    )
    for benchmark in suite_benchmarks(suite):
        subset = subsets.get(benchmark.key)
        if subset is None:
            raise KeyError(
                f"no subset for benchmark {benchmark.key!r}; run `rsi init --suite {suite}`"
            )
        chosen = (benchmark_backends or {}).get(benchmark.key, backend)
        result.benchmark_results[benchmark.key] = chosen.run(
            spec,
            subset,
            seeds,
            workdir / benchmark.key,
            generation=generation,
            base_config=base_config,
        )
    return result
