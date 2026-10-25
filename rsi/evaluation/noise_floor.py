"""Noise floor estimation.

Before any optimizer is written, the pipeline has to know how large a
generation-to-generation difference it can actually detect.  This module
answers that question empirically:

* run the *same* configuration with k different seeds -> the spread of the
  per-task score is pure sampling noise;
* the resulting standard error is the resolution of the measurement, and
  ``delta_min`` is the smallest improvement that could be called significant.

If ``delta_min`` is larger than the improvement a single generation is
expected to produce, the protocol needs more seeds or more tasks -- not a
better optimizer.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import HarnessSpec
from .runner import Backend, EvalResult
from .subsets import Subset

__all__ = ["NoiseFloor", "estimate_noise_floor"]


@dataclass
class NoiseFloor:
    """The measured resolution of the evaluation pipeline."""

    spec_sha256: str
    subset: str
    seeds: list[int]
    n_tasks: int
    per_task_score_sd: float
    mean_score: float
    score_sd_across_seeds: float
    se_of_mean_delta: float
    delta_min: float
    mcnemar_se: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_sha256": self.spec_sha256,
            "subset": self.subset,
            "seeds": list(self.seeds),
            "n_tasks": self.n_tasks,
            "per_task_score_sd": self.per_task_score_sd,
            "mean_score": self.mean_score,
            "score_sd_across_seeds": self.score_sd_across_seeds,
            "se_of_mean_delta": self.se_of_mean_delta,
            "delta_min": self.delta_min,
            "mcnemar_se": self.mcnemar_se,
            "detail": self.detail,
        }

    def summary(self) -> str:
        return (
            f"noise floor on {self.subset}: n={self.n_tasks} tasks x {len(self.seeds)} seeds | "
            f"mean={self.mean_score:.4f} | SE(Δ)={self.se_of_mean_delta:.4f} | "
            f"δ_min={self.delta_min:.4f} ({self.delta_min * 100:.2f}pp)"
        )


def estimate_noise_floor(
    spec: HarnessSpec,
    subset: Subset,
    seeds: Sequence[int],
    backend: Backend,
    workdir,
    *,
    base_config: Mapping[str, Any] | None = None,
    generation: int | None = None,
    alpha: float = 0.05,
) -> NoiseFloor:
    """Run the same spec with several seeds and measure the noise.

    The estimate is built from the *per-task score* distribution: for every
    task we have k binary outcomes, and the per-task score is the number of
    passes divided by k.
    """
    result: EvalResult = backend.run(
        spec,
        subset,
        seeds,
        workdir,
        generation=generation,
        base_config=base_config,
    )

    per_task = result.per_task_scores()
    counts = result.per_task_pass_counts()
    scores = np.array(list(per_task.values()), dtype=float)

    # Within-task variance of the multi-seed score: this is what a repeated
    # measurement of the *same* configuration looks like.
    if scores.size:
        per_task_sd = float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0
    else:
        per_task_sd = 0.0

    # Between-seed spread of the overall mean: the sampling error of the mean.
    seed_means: list[float] = []
    for seed in seeds:
        per_seed = [1.0 if r.passed else 0.0 for r in result.results if r.seed == seed]
        if per_seed:
            seed_means.append(sum(per_seed) / len(per_seed))
    score_sd_across_seeds = (
        float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
    )

    n = max(1, len(per_task))
    k = max(1, len(seeds))
    # Standard error of the *difference* of two independent measurements of the
    # same configuration (this is what a generation-to-generation delta looks
    # like when the configuration did not actually change).
    se = per_task_sd / math.sqrt(n) * math.sqrt(2.0)
    z = 1.959963984540054  # two-sided 95%
    delta_min = z * se

    # McNemar-flavoured resolution: sqrt(b + c) / n with a 10% discordance
    # rate is the classical planning number from the plan.
    discordant = 0.10 * n
    mcnemar_se = math.sqrt(discordant) / n / math.sqrt(k)

    return NoiseFloor(
        spec_sha256=spec.spec_sha256(),
        subset=subset.name,
        seeds=list(seeds),
        n_tasks=n,
        per_task_score_sd=per_task_sd,
        mean_score=result.mean_score,
        score_sd_across_seeds=score_sd_across_seeds,
        se_of_mean_delta=se,
        delta_min=delta_min,
        mcnemar_se=mcnemar_se,
        detail={
            "n_runs": result.n_runs,
            "per_seed_means": seed_means,
            "pass_at_1": result.pass_at_1,
            "synthetic": result.synthetic,
        },
    )
