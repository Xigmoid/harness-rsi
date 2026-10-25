"""Statistics for the generation-to-generation comparison.

The unit of analysis is the *per-task score*, which is the number of passed
seeds divided by the number of seeds for that task.  With k seeds the score
moves in steps of 1/k instead of 0/1, which is what makes a single generation's
improvement measurable at all (plan section 6.1).

The acceptance rule is deliberately strict: a generation is only accepted when
the paired bootstrap CI lower bound is above zero **and** the exact McNemar
test rejects independence.  When the two disagree the stricter one wins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "PairedResult",
    "paired_bootstrap",
    "mcnemar",
    "mcnemar_from_pairs",
    "continuity_corrected_chi2",
    "score_from_seeds",
    "paired_delta",
    "accept_generation",
    "holm_bonferroni",
    "power_check",
]


@dataclass(frozen=True)
class PairedResult:
    """Outcome of one paired comparison."""

    n: int
    mean_delta: float
    se: float
    ci_lo: float
    ci_hi: float
    b: int
    c: int
    mcnemar_p: float
    n_boot: int

    @property
    def significant(self) -> bool:
        """The acceptance criterion of the plan: CI lower bound > 0 and p < 0.05."""
        return self.ci_lo > 0 and self.mcnemar_p < 0.05

    def summary(self) -> str:
        return (
            f"Δ={self.mean_delta:+.4f} "
            f"CI95=[{self.ci_lo:+.4f},{self.ci_hi:+.4f}] "
            f"b={self.b} c={self.c} mcnemar_p={self.mcnemar_p:.4f}"
        )


def score_from_seeds(passed: Sequence[bool]) -> float:
    """Per-task score: the fraction of seeds that passed."""
    if not passed:
        raise ValueError("no seeds to score")
    return float(sum(1 for p in passed if p)) / float(len(passed))


def paired_bootstrap(
    diff: np.ndarray | Sequence[float],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of the paired differences."""
    arr = np.asarray(diff, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty difference vector")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {
        "mean": float(arr.mean()),
        "se": float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "lo": lo,
        "hi": hi,
        "n": int(arr.size),
        "n_boot": int(n_boot),
    }


def mcnemar(b: int, c: int) -> float:
    """Exact McNemar p-value for the discordant pairs.

    ``b`` = new passed / old failed, ``c`` = new failed / old passed.  The
    exact binomial test is used because the chi-square approximation is
    unreliable for the small discordant counts that are typical here.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        # Normal approximation with continuity correction.
        chi2 = (abs(b - c) - 1) ** 2 / n
        return float(math.erfc(math.sqrt(chi2 / 2)))


def continuity_corrected_chi2(b: int, c: int) -> float:
    """McNemar's chi-square with continuity correction (reported next to p)."""
    n = b + c
    if n == 0:
        return 0.0
    return float((abs(b - c) - 1) ** 2 / n)


def mcnemar_from_pairs(new: np.ndarray, old: np.ndarray) -> tuple[int, int]:
    """Count the discordant pairs of two binary score vectors."""
    new = np.asarray(new).astype(float)
    old = np.asarray(old).astype(float)
    if new.shape != old.shape:
        raise ValueError("score vectors must have the same shape")
    b = int(np.sum((new > 0) & (old == 0)))
    c = int(np.sum((new == 0) & (old > 0)))
    return b, c


def paired_delta(
    new_scores: Mapping[str, float],
    old_scores: Mapping[str, float],
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> PairedResult:
    """Compare two generations on the tasks they have in common."""
    common = sorted(set(new_scores) & set(old_scores))
    if not common:
        raise ValueError("the two generations have no task in common")
    new = np.array([new_scores[t] for t in common], dtype=float)
    old = np.array([old_scores[t] for t in common], dtype=float)
    diff = new - old
    boot = paired_bootstrap(diff, n_boot=n_boot, seed=seed)
    b, c = mcnemar_from_pairs(new, old)
    return PairedResult(
        n=len(common),
        mean_delta=boot["mean"],
        se=boot["se"],
        ci_lo=boot["lo"],
        ci_hi=boot["hi"],
        b=b,
        c=c,
        mcnemar_p=mcnemar(b, c),
        n_boot=boot["n_boot"],
    )


def paired_delta_binary(
    new_results: Mapping[str, Mapping[int, bool]],
    old_results: Mapping[str, Mapping[int, bool]],
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> PairedResult:
    """Compare two generations on the **per-(task, seed) binary** outcomes.

    Why this exists
    ---------------
    ``paired_delta`` takes a *continuous* score per task -- the fraction of
    seeds that passed -- and feeds it to McNemar.  But McNemar's test is
    defined on **binary** discordant pairs: a pair counts as discordant only
    when one side passed and the other failed.  Passing a score of 0.333
    makes ``new > 0`` true, so a task that went from 0/3 to 1/3 passes is
    counted as a *win* against a task that went from 3/3 to 2/3 -- neither
    of which is a discordant pair in the test's sense.  The result is a
    p-value that is not the p-value the plan asked for.

    This function does the test the way the plan means it: the unit of
    observation is one (task, seed) run, and the pair is discordant only if
    one run passed and the other failed.
    """
    common_tasks = sorted(set(new_results) & set(old_results))
    if not common_tasks:
        raise ValueError("the two generations have no task in common")
    new_vec: list[float] = []
    old_vec: list[float] = []
    for task in common_tasks:
        shared_seeds = sorted(set(new_results[task]) & set(old_results[task]))
        for s in shared_seeds:
            new_vec.append(1.0 if new_results[task][s] else 0.0)
            old_vec.append(1.0 if old_results[task][s] else 0.0)
    new = np.array(new_vec, dtype=float)
    old = np.array(old_vec, dtype=float)
    diff = new - old
    boot = paired_bootstrap(diff, n_boot=n_boot, seed=seed)
    b, c = mcnemar_from_pairs(new, old)
    return PairedResult(
        n=len(new),
        mean_delta=boot["mean"],
        se=boot["se"],
        ci_lo=boot["lo"],
        ci_hi=boot["hi"],
        b=b,
        c=c,
        mcnemar_p=mcnemar(b, c),
        n_boot=boot["n_boot"],
    )


def accept_generation(result: PairedResult, *, alpha: float = 0.05) -> tuple[bool, str]:
    """Apply the acceptance rule of the plan.

    Accept when the bootstrap CI lower bound is above zero *and* the exact
    McNemar test rejects; when the two disagree, the stricter one wins.
    """
    ci_ok = result.ci_lo > 0
    p_ok = result.mcnemar_p < alpha
    ok = ci_ok and p_ok
    reason = (
        f"Δ={result.mean_delta:+.4f} CI95=[{result.ci_lo:+.4f},{result.ci_hi:+.4f}] "
        f"b={result.b} c={result.c} mcnemar_p={result.mcnemar_p:.4f} "
        f"-> {'ACCEPT' if ok else 'REJECT'} "
        f"(bootstrap_ci>0: {ci_ok}, mcnemar_p<{alpha}: {p_ok})"
    )
    return ok, reason


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (for the per-field effect table)."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def power_check(
    n_tasks: int,
    n_seeds: int,
    discordant_rate: float = 0.08,
    effect: float = 0.02,
) -> dict[str, float]:
    """Rough power calculation used to size the protocol.

    ``discordant_rate`` is the fraction of tasks whose score differs between
    two generations, ``effect`` the expected improvement in mean score.
    """
    n_disc = max(1.0, n_tasks * discordant_rate)
    se = math.sqrt(n_disc) / n_tasks / math.sqrt(max(1, n_seeds))
    return {
        "n_tasks": float(n_tasks),
        "n_seeds": float(n_seeds),
        "discordant_rate": float(discordant_rate),
        "effect": float(effect),
        "se": float(se),
        "z": float(effect / se) if se else float("inf"),
        "detectable": float(1.96 * se),
    }
