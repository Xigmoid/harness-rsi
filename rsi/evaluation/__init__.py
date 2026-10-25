"""Evaluation: task subsets, benchmark runners, statistics, noise floor."""

from .noise_floor import NoiseFloor, estimate_noise_floor
from .runner import (
    Backend,
    EvalResult,
    HarborBackend,
    MiniBackend,
    SyntheticBackend,
    TaskResult,
    build_backend,
    render_harbor_job,
    verify_job_invariance,
)
from .stats import (
    PairedResult,
    accept_generation,
    mcnemar,
    paired_bootstrap,
    paired_delta,
)
from .subsets import Subset, list_subsets, load_subset, make_split, save_subset, verify_subset

__all__ = [
    "Backend",
    "EvalResult",
    "HarborBackend",
    "MiniBackend",
    "NoiseFloor",
    "PairedResult",
    "Subset",
    "SyntheticBackend",
    "TaskResult",
    "accept_generation",
    "build_backend",
    "estimate_noise_floor",
    "list_subsets",
    "load_subset",
    "make_split",
    "mcnemar",
    "paired_bootstrap",
    "paired_delta",
    "render_harbor_job",
    "save_subset",
    "verify_job_invariance",
    "verify_subset",
]
