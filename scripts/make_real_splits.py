#!/usr/bin/env python3
"""Rebuild the dev / held-out splits with Harbor's *real* task names.

The splits written by ``rsi init`` use placeholder task ids
(``aime-task-0000``), because the framework does not know how a benchmark's
tasks are named until Harbor resolves them.  That is fine for the synthetic
backend, but it makes the splits useless for a real run: a Harbor job
filters its dataset with ``task_names``, and those names have to be the
ones Harbor uses.

This script asks Harbor for the real task list of every benchmark, splits it
with the protocol's fixed seed and the benchmark's fixed ratio, and writes:

* ``subsets/<key>_dev.json`` and ``subsets/<key>_heldout.json`` -- the
  frozen split, with the SHA256 that the framework re-checks on every run.
  The split is **ratio invariant** (e.g. 300/445 for Terminal-Bench,
  300/500 for SWE-bench Verified); the absolute sizes follow from what is
  actually installed.
* ``harbor/job.<key>.yaml`` -- the job template with the **dev** task names
  frozen into ``datasets[].task_names``, so the job runs exactly the dev
  tasks and the template stays byte-identical across generations (C-1).

The dev split is *not* truncated in the subset file -- the ratio invariant
is what the protocol protects.  The job template may however run a cost
capped *prefix* of the dev split (``--max-dev-tasks``); the tasks that were
actually evaluated are recorded per task in the run records, so every score
is computed on the set that really ran.

    python scripts/make_real_splits.py --benchmarks aime ineqmath gpqa ifeval

Harbor is needed (Python >= 3.12).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsi.benchmarks import BENCHMARKS, get_benchmark  # noqa: E402
from rsi.evaluation.subsets import (  # noqa: E402
    SPLIT_SEED,
    Subset,
    make_split_by_ratio,
    save_subset,
)
from rsi.paths import (  # noqa: E402,F401
    TOOLCHAIN_BUNDLE_DIR,
    TOOLCHAIN_MOUNT_TARGET,
    harbor_scratch_dir,
    toolchain_available,
)

#: Benchmarks whose Harbor dataset lives in the registry rather than on disk.
REGISTRY_KEYS = tuple(k for k, b in BENCHMARKS.items() if b.source == "registry")


def _dataset_config(benchmark) -> dict:
    """The ``datasets:`` entry, as a plain dict (never as YAML text)."""
    if benchmark.source == "path":
        # An absolute path: the Harbor subprocess may run with a different
        # working directory than the project root.
        return {"path": str((ROOT / benchmark.path).resolve())}
    return {"name": benchmark.dataset, "version": benchmark.version}


def _job_dict(benchmark, task_names, *, n_concurrent: int, model: str) -> dict:
    """The frozen job template for one benchmark."""
    dataset = _dataset_config(benchmark)
    dataset["task_names"] = list(task_names)
    environment: dict = {
        # force_build=False: a task that declares a prebuilt `docker_image` is
        # used as it is.  With force_build=True Harbor re-runs `docker compose
        # build` for every task of every trial, which turns a seconds-long
        # evaluation into a minutes-long one for no benefit.
        "type": "docker",
        "delete": True,
        "force_build": False,
    }
    return {
        # Harbor bind-mounts the trial's log directories into the task
        # container, so the job output has to live on a path the container
        # runtime can mount.  On macOS that is the literal /private/tmp; the
        # runner copies the trials back into the project afterwards.
        "jobs_dir": str(harbor_scratch_dir()),
        "n_attempts": 1,
        "n_concurrent_trials": n_concurrent,
        "timeout_multiplier": 1.0,
        "environment": environment,
        "retry": {
            "max_retries": 3,
            "include_exceptions": [
                "ApiRateLimitError",
                "ApiInternalServerError",
                "ApiOverloadedError",
                "ApiConnectionClosedError",
            ],
        },
        "agents": [
            {
                # Harbor's import path format is `module.path:ClassName`.
                "import_path": "rsi.harbor_agent:RsiMiniSweAgent",
                "name": "rsi-mini-swe-agent",
                "model_name": model,
                "kwargs": {
                    "version": "2.4.6",
                    "config": {"agent": {"agent_class": "rsi.rhi_agent.RhiAgent"}},
                },
            }
        ],
        "datasets": [dataset],
    }


async def _real_task_names(benchmark) -> list[str]:
    """The task names Harbor itself uses for this benchmark."""
    from harbor.models.job.config import DatasetConfig

    config = DatasetConfig.model_validate(_dataset_config(benchmark))
    configs = await config.get_task_configs(disable_verification=True)
    names: list[str] = []
    for task in configs:
        name = task.name or (task.path.name if task.path else None)
        if name:
            names.append(name)
    return sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--n-concurrent", type=int, default=16)
    parser.add_argument("--model", default="openai/glm5.3-flash")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--max-dev-tasks",
        type=int,
        default=None,
        help=(
            "cost cap: the job template runs only the first N tasks of the "
            "shuffled dev split. The subset files keep the full ratio split."
        ),
    )
    parser.add_argument(
        "--only",
        choices=("dev", "heldout", "both"),
        default="dev",
        help="which split to freeze into the job template",
    )
    args = parser.parse_args()

    subsets_dir = ROOT / "subsets"
    harbor_dir = ROOT / "harbor"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    harbor_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for key in args.benchmarks:
        benchmark = get_benchmark(key)
        try:
            names = asyncio.run(_real_task_names(benchmark))
        except Exception as exc:  # noqa: BLE001 -- report and continue
            print(f"! {key}: could not resolve tasks: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if not names:
            print(f"! {key}: Harbor returned no task names")
            failures += 1
            continue

        dev_ratio = benchmark.dev_ratio
        dev_subset, heldout_subset = make_split_by_ratio(
            key, names, dev_ratio=dev_ratio, seed=args.seed, source=benchmark.source
        )
        for subset in (dev_subset, heldout_subset):
            path = save_subset(subset, subsets_dir / f"{key}_{subset.split}.json")
            print(
                f"wrote {path.relative_to(ROOT)} ({subset.n} tasks, "
                f"sha256={subset.sha256()[:12]})"
            )

        if args.only == "both":
            chosen = list(dev_subset.task_ids) + list(heldout_subset.task_ids)
        else:
            chosen = list(dev_subset.task_ids)
        if args.max_dev_tasks:
            chosen = chosen[: args.max_dev_tasks]

        job = _job_dict(
            benchmark, chosen, n_concurrent=args.n_concurrent, model=args.model
        )
        job_path = harbor_dir / f"job.{key}.yaml"
        job_path.write_text(
            yaml.safe_dump(job, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        print(f"wrote {job_path.relative_to(ROOT)} ({len(chosen)} task names frozen)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
