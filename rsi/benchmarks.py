"""The benchmark registry.

The framework is deliberately *capability-agnostic*: the same harness (an
orchestrator, subagents, contracts) is evaluated on several very different
task families, and the question the study answers is whether the optimal
contract is task-dependent.

Every benchmark is described by a :class:`BenchmarkSpec`:

* ``capability`` -- the task family (``code``, ``math``, ``reasoning``,
  ``instruction_following``).  This is the axis along which harnesses are
  specialised.
* ``dataset`` / ``version`` -- the Harbor dataset reference.
* ``source`` -- ``registry`` (a dataset in Harbor's registry) or ``path``
  (a local Harbor task directory).
* ``dev_ratio`` -- the dev/held-out ratio.  The protocol of the plan fixes
  the ratio, the absolute sizes follow from the benchmark.
* ``cost_class`` -- ``heavy`` (a container per task, minutes) or ``light``
  (seconds).  This matters for the budget: a composite objective over
  benchmarks with a 100x cost spread is dominated by the expensive ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

__all__ = [
    "BenchmarkSpec",
    "BENCHMARKS",
    "SUITES",
    "STUDY_CHAINS",
    "benchmarks_in_suite",
    "suite_benchmarks",
    "suite_capabilities",
    "get_benchmark",
    "suite_definitions",
]


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark the framework can evaluate on."""

    key: str
    capability: str
    dataset: str
    version: str
    dev_ratio: float
    cost_class: str = "light"
    source: str = "registry"
    path: str | None = None
    n_tasks: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.source not in ("registry", "path"):
            raise ValueError(f"benchmark {self.key}: source must be 'registry' or 'path'")
        if not 0.0 < self.dev_ratio < 1.0:
            raise ValueError(f"benchmark {self.key}: dev_ratio must be in (0, 1)")
        if self.source == "path" and not self.path:
            raise ValueError(f"benchmark {self.key}: a path benchmark needs a path")
        if self.cost_class not in ("light", "heavy"):
            raise ValueError(f"benchmark {self.key}: cost_class must be 'light' or 'heavy'")

    @property
    def dataset_ref(self) -> dict[str, str]:
        """The ``datasets:`` entry of the Harbor job file."""
        if self.source == "path":
            return {"path": self.path or ""}
        return {"name": self.dataset, "version": self.version}

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "capability": self.capability,
            "dataset": self.dataset,
            "version": self.version,
            "dev_ratio": self.dev_ratio,
            "cost_class": self.cost_class,
            "source": self.source,
            "path": self.path,
            "n_tasks": self.n_tasks,
            "notes": self.notes,
        }


#: The benchmarks that are available in Harbor 0.24.0's registry, plus the
#: ones that have to be provided locally.
BENCHMARKS: dict[str, BenchmarkSpec] = {
    # ---------------- code ------------------------------------------------ #
    "sbv": BenchmarkSpec(
        key="sbv",
        capability="code",
        dataset="swebench-verified",
        version="1.0",
        dev_ratio=300 / 500,
        cost_class="heavy",
        n_tasks=500,
        notes="SWE-bench Verified -- the final validation benchmark of the plan.",
    ),
    "tb21": BenchmarkSpec(
        key="tb21",
        capability="code",
        dataset="terminal-bench",
        version="2.0",
        dev_ratio=300 / 445,
        cost_class="heavy",
        source="path",
        path="datasets/registry/tb21",
        n_tasks=89,
        notes=(
            "Terminal-Bench.  The plan assumed 2.1 with 445 tasks; Harbor's "
            "registry carries 2.0 with 89 tasks, so the dev/held-out ratio of "
            "the plan is applied to what is actually available.  Local copy "
            "of the registry dataset (toolchain-baked base images)."
        ),
    ),
    "livecode": BenchmarkSpec(
        key="livecode",
        capability="code",
        dataset="livecodebench",
        version="6.0",
        dev_ratio=0.7,
        cost_class="light",
        source="path",
        path="datasets/registry/livecode",
        n_tasks=100,
        notes=(
            "Competitive programming problems with runnable tests.  Local "
            "copy of the registry dataset (toolchain-baked base image)."
        ),
    ),
    # ---------------- math ------------------------------------------------ #
    "aime": BenchmarkSpec(
        key="aime",
        capability="math",
        dataset="aime",
        version="1.0",
        dev_ratio=0.7,
        cost_class="light",
        source="path",
        path="datasets/registry/aime",
        n_tasks=60,
        notes=(
            "AIME competition problems (hard).  A *local* copy of the "
            "registry dataset: the base image is rebuilt with the agent "
            "toolchain baked in, see scripts/prepare_registry_tasks.py."
        ),
    ),
    "ineqmath": BenchmarkSpec(
        key="ineqmath",
        capability="math",
        dataset="ineqmath",
        version="1.0",
        dev_ratio=0.7,
        cost_class="light",
        source="path",
        path="datasets/registry/ineqmath",
        n_tasks=100,
        notes=(
            "Inequality problems from the InequalityMath benchmark.  Local "
            "copy of the registry dataset (toolchain-baked base image)."
        ),
    ),
    # ---------------- reasoning ------------------------------------------- #
    "gpqa": BenchmarkSpec(
        key="gpqa",
        capability="reasoning",
        dataset="gpqa-diamond",
        version="1.0",
        dev_ratio=0.7,
        cost_class="light",
        source="path",
        path="datasets/registry/gpqa",
        n_tasks=198,
        notes=(
            "Graduate-level science questions (GPQA Diamond).  Local copy "
            "of the registry dataset (toolchain-baked base image)."
        ),
    ),
    "arcagi": BenchmarkSpec(
        key="arcagi",
        capability="reasoning",
        dataset="arc_agi_2",
        version="1.0",
        dev_ratio=0.7,
        cost_class="light",
        n_tasks=167,
        notes="ARC-AGI-2 abstract reasoning tasks.",
    ),
    "reagym": BenchmarkSpec(
        key="reagym",
        capability="reasoning",
        dataset="reasoning-gym-hard",
        version="parity",
        dev_ratio=0.7,
        cost_class="light",
        n_tasks=288,
        notes="Procedurally generated reasoning tasks (hard variants).",
    ),
    # ---------------- instruction following ------------------------------- #
    "ifeval": BenchmarkSpec(
        key="ifeval",
        capability="instruction_following",
        dataset="ifeval",
        version="1.0",
        dev_ratio=0.7,
        cost_class="light",
        source="path",
        path="datasets/ifeval",
        n_tasks=541,
        notes=(
            "IFEval (verifiable instruction following).  Not in Harbor's "
            "registry: build the task directory with "
            "`python scripts/build_ifeval_tasks.py`."
        ),
    ),
}

#: Named groups of benchmarks.  A *suite* is the unit an optimizer is pointed
#: at: specialising for ``math`` means the objective is the composite of the
#: math benchmarks.
SUITES: dict[str, tuple[str, ...]] = {
    "code": ("sbv", "tb21"),
    "math": ("aime", "ineqmath"),
    "reasoning": ("gpqa", "arcagi", "reagym"),
    "instruction_following": ("ifeval",),
    "all": ("sbv", "tb21", "aime", "ineqmath", "gpqa", "arcagi", "reagym", "ifeval"),
    # -- the single-benchmark variants used by the 4x3 self-evolution study ---
    # One benchmark per capability: the study has to afford 13 harness
    # evaluations, and a composite over several benchmarks multiplies the
    # cost without adding a capability.
    "code_only": ("livecode",),
    "math_only": ("ineqmath",),
    "reasoning_only": ("gpqa",),
    "if_only": ("ifeval",),
}

#: The four capability chains of the self-evolution study.  Each chain
#: optimises one of these suites and is evaluated on its own benchmark.
STUDY_CHAINS: dict[str, str] = {
    "code": "code_only",
    "math": "math_only",
    "reasoning": "reasoning_only",
    "instruction_following": "if_only",
}

#: Suites that are cheap enough to iterate on (a full evolution of one
#: generation over them is minutes, not hours).
LIGHT_SUITES: tuple[str, ...] = ("math", "reasoning", "instruction_following")


def get_benchmark(key: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown benchmark {key!r}; known: {sorted(BENCHMARKS)}"
        ) from exc


def suite_benchmarks(suite: str) -> tuple[BenchmarkSpec, ...]:
    """The benchmarks of a suite."""
    if suite in BENCHMARKS:  # a single benchmark is a suite of one
        return (BENCHMARKS[suite],)
    if suite not in SUITES:
        raise KeyError(f"unknown suite {suite!r}; known: {sorted(SUITES)}")
    return tuple(BENCHMARKS[key] for key in SUITES[suite])


def benchmarks_in_suite(suite: str) -> tuple[str, ...]:
    if suite in BENCHMARKS:
        return (suite,)
    return SUITES[suite]


def suite_capabilities(suite: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(b.capability for b in suite_benchmarks(suite)))


def suite_definitions() -> list[dict[str, Any]]:
    """A machine readable description of every suite (for the report)."""
    out = []
    for name, keys in SUITES.items():
        specs = tuple(BENCHMARKS[k] for k in keys)
        out.append(
            {
                "suite": name,
                "benchmarks": list(keys),
                "capabilities": list(dict.fromkeys(s.capability for s in specs)),
                "cost_class": "heavy" if any(s.cost_class == "heavy" for s in specs) else "light",
                "n_tasks": sum(s.n_tasks or 0 for s in specs),
            }
        )
    return out
