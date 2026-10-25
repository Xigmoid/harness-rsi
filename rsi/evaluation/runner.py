"""Benchmark runners.

Three backends implement the same interface:

``HarborBackend``
    Runs Terminal-Bench / SWE-bench through Harbor.  The job file is written
    from a frozen template and *only* the ``agents[0].kwargs.config`` block is
    replaced per generation (constraint C-1), so a score difference between two
    generations can only come from the harness configuration.

``MiniBackend``
    Runs SWE-bench through mini-swe-agent's own batch runner
    (``mini-extra swebench``).  Used as a cross-check of the Harbor path.

``SyntheticBackend``
    A deterministic test double.  It is **not** a benchmark: it fabricates a
    reproducible score from the spec hash so that the statistics, ledger and
    optimizer plumbing can be exercised offline.  Every result it produces is
    marked ``synthetic=True`` so it can never be mistaken for a measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import yaml

from ..paths import harbor_scratch_dir
from ..schema import HarnessSpec
from .subsets import Subset

__all__ = [
    "TaskResult",
    "EvalResult",
    "Backend",
    "HarborBackend",
    "MiniBackend",
    "SyntheticBackend",
    "build_backend",
    "render_harbor_job",
    "verify_job_invariance",
]


@dataclass
class TaskResult:
    """The outcome of one (task, seed) run."""

    task_id: str
    seed: int
    passed: bool
    duration_s: float = 0.0
    n_llm_calls: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    trajectory_path: Path | None = None
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    n_cache_tokens: int = 0
    retries: int = 0
    synthetic: bool = False
    #: The raw reward, when the benchmark reports one.  A verifier that
    #: fails may write a *negative* reward (IneqMath writes -1.0), so the
    #: pass/fail decision is made on its sign, and the raw value is kept
    #: for auditing.
    reward: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "task_id": self.task_id,
            "seed": self.seed,
            "passed": self.passed,
            "duration_s": round(self.duration_s, 3),
            "n_llm_calls": self.n_llm_calls,
            "cost_usd": round(self.cost_usd, 6),
            "n_input_tokens": self.n_input_tokens,
            "n_output_tokens": self.n_output_tokens,
            "n_cache_tokens": self.n_cache_tokens,
            "retries": self.retries,
            "synthetic": self.synthetic,
            "reward": self.reward,
            "error": self.error,
            "trajectory_path": str(self.trajectory_path) if self.trajectory_path else None,
        }
        if self.extra:
            data["extra"] = self.extra
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskResult":
        tp = data.get("trajectory_path")
        return cls(
            task_id=data["task_id"],
            seed=int(data.get("seed", 0)),
            passed=bool(data.get("passed", False)),
            duration_s=float(data.get("duration_s", 0.0)),
            n_llm_calls=int(data.get("n_llm_calls", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            n_input_tokens=int(data.get("n_input_tokens", 0)),
            n_output_tokens=int(data.get("n_output_tokens", 0)),
            n_cache_tokens=int(data.get("n_cache_tokens", 0)),
            retries=int(data.get("retries", 0)),
            synthetic=bool(data.get("synthetic", False)),
            reward=float(data.get("reward", 0.0) or 0.0),
            error=data.get("error"),
            trajectory_path=Path(tp) if tp else None,
            extra=dict(data.get("extra", {}) or {}),
        )


@dataclass
class EvalResult:
    """The aggregated result of one (generation, benchmark, subset) evaluation."""

    benchmark: str
    subset: str
    seeds: list[int]
    results: list[TaskResult]
    spec_sha256: str = ""
    generation: int | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    synthetic: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    # -- scores ------------------------------------------------------------ #

    def per_task_scores(self) -> dict[str, float]:
        """Per-task score: passed seeds / total seeds (multi-seed continuisation)."""
        buckets: dict[str, list[bool]] = {}
        for r in self.results:
            buckets.setdefault(r.task_id, []).append(r.passed)
        return {task: sum(v) / len(v) for task, v in buckets.items()}

    def per_task_pass_counts(self) -> dict[str, list[bool]]:
        buckets: dict[str, list[bool]] = {}
        for r in self.results:
            buckets.setdefault(r.task_id, []).append(r.passed)
        return buckets

    def per_task_seed_passes(self) -> dict[str, dict[int, bool]]:
        """The pass/fail outcome of every (task, seed) run.

        This is the unit the paired statistical test needs: McNemar's test
        is defined on binary discordant pairs, and the seed is what makes
        two runs of the same task comparable.
        """
        buckets: dict[str, dict[int, bool]] = {}
        for r in self.results:
            buckets.setdefault(r.task_id, {})[r.seed] = r.passed
        return buckets

    @property
    def mean_score(self) -> float:
        scores = self.per_task_scores()
        return sum(scores.values()) / len(scores) if scores else 0.0

    @property
    def n_tasks(self) -> int:
        return len(self.per_task_scores())

    @property
    def n_runs(self) -> int:
        return len(self.results)

    @property
    def total_retries(self) -> int:
        return sum(r.retries for r in self.results)

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def total_llm_calls(self) -> int:
        return sum(r.n_llm_calls for r in self.results)

    @property
    def pass_at_1(self) -> float:
        """Fraction of individual runs that passed (the "one shot" number)."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    # -- serialization ----------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "subset": self.subset,
            "seeds": list(self.seeds),
            "generation": self.generation,
            "spec_sha256": self.spec_sha256,
            "synthetic": self.synthetic,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "mean_score": self.mean_score,
            "pass_at_1": self.pass_at_1,
            "n_tasks": self.n_tasks,
            "n_runs": self.n_runs,
            "total_retries": self.total_retries,
            "n_errors": self.n_errors,
            "total_cost_usd": self.total_cost_usd,
            "total_llm_calls": self.total_llm_calls,
            "per_task": self.per_task_scores(),
            "results": [r.to_dict() for r in self.results],
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalResult":
        return cls(
            benchmark=data["benchmark"],
            subset=data["subset"],
            seeds=[int(s) for s in data.get("seeds", [])],
            results=[TaskResult.from_dict(r) for r in data.get("results", [])],
            spec_sha256=data.get("spec_sha256", ""),
            generation=data.get("generation"),
            started_at=float(data.get("started_at", 0.0)),
            finished_at=float(data.get("finished_at", 0.0)),
            synthetic=bool(data.get("synthetic", False)),
            extra=dict(data.get("extra", {}) or {}),
        )


class Backend(Protocol):
    """Common interface of the evaluation backends."""

    name: str

    def run(
        self,
        spec: HarnessSpec,
        subset: Subset,
        seeds: Sequence[int],
        workdir: Path,
        *,
        generation: int | None = None,
        base_config: Mapping[str, Any] | None = None,
    ) -> EvalResult:
        ...


# --------------------------------------------------------------------------- #
# Harbor
# --------------------------------------------------------------------------- #


def _run_tag(generation: int | None) -> str:
    """A per-generation directory name for the Harbor job output.

    The job output directory is scanned for ``results.json`` files, so it
    **must not be shared between generations**: a later generation would
    pick up the previous generation's trials and score them as its own.
    """
    return f"gen{generation:03d}" if generation is not None else "gen---"


def render_harbor_job(
    base_job: Mapping[str, Any],
    agent_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the job config for one generation.

    Only ``agents[0].kwargs.config`` is replaced; every other key is copied
    verbatim so the job file is byte-identical across generations (C-1).
    """
    job = json.loads(json.dumps(base_job, default=str))
    agents = job.get("agents")
    if not agents:
        raise ValueError("harbor job config has no agents")
    agent = agents[0]
    kwargs = dict(agent.get("kwargs") or {})
    kwargs["config"] = json.loads(json.dumps(agent_config, default=str))
    agent["kwargs"] = kwargs
    return job


def verify_job_invariance(
    base_job: Mapping[str, Any],
    rendered_job: Mapping[str, Any],
) -> list[str]:
    """Return the places where the rendered job differs from the template.

    The rule is C-1: **only** ``agents[0].kwargs.config`` may change.  Every
    other key -- top level *and* inside ``agents`` -- has to be identical, or
    the generation-to-generation difference is not attributable to the
    harness.
    """
    differing: list[str] = []
    for key in sorted(set(base_job) | set(rendered_job)):
        if key == "agents":
            continue
        if base_job.get(key) != rendered_job.get(key):
            differing.append(key)

    base_agents = list(base_job.get("agents") or [])
    rendered_agents = list(rendered_job.get("agents") or [])
    if len(base_agents) != len(rendered_agents):
        differing.append("agents")
    else:
        for index, (base_agent, rendered_agent) in enumerate(
            zip(base_agents, rendered_agents)
        ):
            base_keys = set(base_agent or {})
            rendered_keys = set(rendered_agent or {})
            for key in sorted(base_keys | rendered_keys):
                if key == "kwargs" and index == 0:
                    continue  # checked below, field by field
                if (base_agent or {}).get(key) != (rendered_agent or {}).get(key):
                    differing.append(f"agents[{index}].{key}")
            base_kwargs = dict((base_agent or {}).get("kwargs") or {})
            rendered_kwargs = dict((rendered_agent or {}).get("kwargs") or {})
            for key in sorted(set(base_kwargs) | set(rendered_kwargs)):
                if index == 0 and key == "config":
                    continue  # the one field that is allowed to change
                if base_kwargs.get(key) != rendered_kwargs.get(key):
                    differing.append(f"agents[{index}].kwargs.{key}")
    return sorted(differing)


class HarborBackend:
    """Runs a benchmark through the Harbor CLI.

    The retry policy and the concurrency are written *into* the job file
    rather than passed on the command line, so that the rendered job file is
    self-contained and byte-identical across generations.  Harbor's default is
    ``max_retries: 0`` -- retries must be switched on explicitly, otherwise
    the failure rate is not comparable across generations (trap T-8).
    """

    name = "harbor"

    def __init__(
        self,
        job_template: str | Path,
        *,
        jobs_dir: str | Path = "runs/harbor",
        n_concurrent: int | None = None,
        max_retries: int | None = None,
        retry_include: Sequence[str] | None = None,
        retry_exclude: Sequence[str] | None = None,
        extra_args: Sequence[str] = (),
        dry_run: bool = False,
    ):
        self.job_template = Path(job_template)
        self.jobs_dir = Path(jobs_dir)
        self.n_concurrent = n_concurrent
        self.max_retries = max_retries
        self.retry_include = tuple(retry_include) if retry_include is not None else None
        self.retry_exclude = tuple(retry_exclude) if retry_exclude is not None else None
        self.extra_args = tuple(extra_args)
        self.dry_run = dry_run
        #: The tasks the frozen job template actually runs (see
        #: :meth:`_resolve_effective_tasks`).  Filled in by :meth:`run`.
        self._effective_task_ids: list[str] = []

    # -- helpers ----------------------------------------------------------- #

    def _base_job(self) -> dict[str, Any]:
        data = yaml.safe_load(self.job_template.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{self.job_template}: harbor job template must be a mapping")
        # Apply the overrides to the *job*, not to the command line: the job
        # file is the frozen artifact that has to stay identical across
        # generations.
        if self.n_concurrent is not None:
            data["n_concurrent_trials"] = self.n_concurrent
        if self.max_retries is not None or self.retry_include is not None:
            retry = dict(data.get("retry") or {})
            if self.max_retries is not None:
                retry["max_retries"] = self.max_retries
            if self.retry_include is not None:
                retry["include_exceptions"] = list(self.retry_include)
            if self.retry_exclude is not None:
                retry["exclude_exceptions"] = list(self.retry_exclude)
            data["retry"] = retry
        return data

    def _command(self, job_path: Path, out_dir: Path) -> list[str]:
        return [
            _harbor_executable(),
            "job",
            "start",
            "-c",
            str(job_path),
            "-o",
            str(out_dir),
            # Harbor asks for confirmation when a task pulls variables from
            # the host environment.  The study runs unattended, so the prompt
            # has to be answered; `--yes` is the only way to do that without
            # a TTY.  It does not change what the job *does*, only that the
            # confirmation is pre-answered.
            "--yes",
        ] + list(self.extra_args)

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        """The environment for the ``harbor`` subprocess.

        The job file names the agent class by import path
        (``rsi.harbor_agent:RsiMiniSweAgent``), so the *host* process has to
        be able to import the ``rsi`` package.  An editable install is not
        always visible to a console script (its ``sys.path[0]`` is the
        script's directory, not the project), so the package's own location
        is put on ``PYTHONPATH`` explicitly.
        """
        env = dict(os.environ)
        package_root = str(Path(__file__).resolve().parent.parent.parent)
        existing = env.get("PYTHONPATH", "")
        parts = [package_root] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def _collect(self, out_dir: Path, subset: Subset, seed: int) -> list[TaskResult]:
        """Parse every trial result under the job directory.

        A task that was retried by Harbor produced several trial results; the
        last attempt is the one that counts, and the number of extra attempts
        is recorded as the retry count.
        """
        by_task: dict[str, list[dict[str, Any]]] = {}
        for results_path in sorted(out_dir.rglob("result*.json")):
            try:
                data = json.loads(results_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            task_name = _harbor_task_name(data)
            if not task_name:
                continue
            data["__path"] = str(results_path)
            by_task.setdefault(task_name, []).append(data)

        found: dict[str, TaskResult] = {}
        for task_name, attempts in by_task.items():
            attempts.sort(key=lambda d: (d.get("started_at") or "", d["__path"]))
            data = attempts[-1]
            # A trial passes when its reward is *positive*.  Testing the
            # truthiness of the reward instead would count a negative reward
            # as a pass: `bool(-1.0)` is `True`, and a verifier that fails
            # writes exactly that (IneqMath writes -1.0 when its LLM judge
            # cannot be reached).  Such a trial would then be scored as a
            # *success*, and the mean would be silently inflated.
            passed = _harbor_reward(data) > 0.0
            reward = _harbor_reward(data)
            agent_result = data.get("agent_result") or {}
            exception = data.get("exception_info") or {}
            timings = data.get("agent_execution") or {}
            duration = 0.0
            if isinstance(timings, Mapping):
                duration = _duration_seconds(timings.get("started_at"), timings.get("finished_at"))
            trajectory = Path(data["__path"]).parent / "agent" / "mini-swe-agent.trajectory.json"
            found[task_name] = TaskResult(
                task_id=task_name,
                seed=seed,
                passed=passed,
                duration_s=duration,
                n_llm_calls=int(agent_result.get("n_api_calls", 0) or 0),
                cost_usd=float(agent_result.get("cost_usd", 0.0) or 0.0),
                n_input_tokens=int(agent_result.get("n_input_tokens", 0) or 0),
                n_output_tokens=int(agent_result.get("n_output_tokens", 0) or 0),
                n_cache_tokens=int(agent_result.get("n_cache_tokens", 0) or 0),
                retries=max(0, len(attempts) - 1),
                error=exception.get("exception_message") or exception.get("exception_type"),
                reward=reward,
                trajectory_path=trajectory if trajectory.exists() else None,
            )
        # Tasks without a result file count as failures -- never drop them
        # (trap T-9), otherwise the score would be inflated.  But only the
        # tasks that were *supposed* to run: the job template may run a
        # cost-capped prefix of the dev split, and the tasks it did not
        # select are not failures, they are tasks that never happened.
        results: list[TaskResult] = []
        for task_id in self._effective_task_ids:
            results.append(
                found.get(
                    task_id,
                    TaskResult(task_id=task_id, seed=seed, passed=False, error="no result file"),
                )
            )
        for task_id, res in found.items():
            if task_id not in self._effective_task_ids:
                results.append(res)
        return results

    def _resolve_effective_tasks(self, base_job: Mapping[str, Any], subset: Subset) -> list[str]:
        """The tasks this job actually runs.

        The job template's ``datasets[].task_names`` is the frozen artifact
        that decides what runs (constraint C-1); the subset is the split it
        has to be taken from.  A cost-capped job names a prefix of the dev
        split, and the score must be computed on exactly that set -- not on
        the whole split, and not on the tasks that were never selected.
        """
        selected: list[str] = []
        for dataset in base_job.get("datasets") or []:
            names = (dataset or {}).get("task_names")
            if names:
                selected.extend(str(n) for n in names)
        if not selected:
            return list(subset.task_ids)
        outside = [n for n in selected if n not in set(subset.task_ids)]
        if outside:
            raise ValueError(
                f"the job template names {len(outside)} task(s) that are not in "
                f"the {subset.name} split (e.g. {outside[:3]}); the job must run "
                "a subset of the split, never tasks outside it"
            )
        # Keep the split's order so that the effective set is a prefix of it.
        allowed = set(selected)
        return [t for t in subset.task_ids if t in allowed]

    # -- main -------------------------------------------------------------- #

    def run(
        self,
        spec: HarnessSpec,
        subset: Subset,
        seeds: Sequence[int],
        workdir: Path,
        *,
        generation: int | None = None,
        base_config: Mapping[str, Any] | None = None,
    ) -> EvalResult:
        from ..adapter import to_mini_config

        workdir.mkdir(parents=True, exist_ok=True)
        base_job = self._base_job()
        self._effective_task_ids = self._resolve_effective_tasks(base_job, subset)
        all_results: list[TaskResult] = []
        started = time.time()

        for seed in seeds:
            seed_spec = _spec_with_seed(spec, seed)
            agent_config = to_mini_config(seed_spec, base_config, spec_path=workdir / "spec.yaml")
            job = render_harbor_job(base_job, agent_config)
            differing = verify_job_invariance(base_job, job)
            if differing:
                raise ValueError(
                    f"harbor job template is not invariant across generations: {differing}. "
                    "Only agents[0].kwargs.config may change (constraint C-1)."
                )
            job_path = workdir / f"job.seed{seed}.yaml"
            job_path.write_text(
                yaml.safe_dump(job, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            # Harbor bind-mounts the trial's log directories into the task
            # container, so the job output has to live on a mountable path
            # (see ``rsi.paths.harbor_scratch_dir``).  It is copied back into
            # the project afterwards, so the trajectories are archived where
            # the rest of the study lives.
            #
            # The path is *unique per run*: Docker Desktop's file sharing
            # caches the host's directory tree, and a path that was deleted
            # and recreated stays stale in the VM -- the mount then fails
            # with "bind source path does not exist".  A path that has never
            # existed before is always visible.
            run_id = f"{_run_tag(generation)}-seed{seed}-{int(time.time() * 1000)}"
            out_dir = harbor_scratch_dir() / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = self._command(job_path, out_dir)
            (workdir / f"command.seed{seed}.txt").write_text(
                " ".join(shlex.quote(c) for c in cmd) + "\n", encoding="utf-8"
            )
            if self.dry_run:
                continue
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=self._subprocess_env()
            )
            (workdir / f"harbor.seed{seed}.log").write_text(
                (proc.stdout or "") + (proc.stderr or ""), encoding="utf-8"
            )
            if proc.returncode != 0:
                # A failed job still yields partial results; the missing tasks
                # are recorded as failures below.
                (workdir / f"harbor.seed{seed}.failed").write_text(
                    f"exit={proc.returncode}\n", encoding="utf-8"
                )
            archive_dir = workdir / "harbor" / f"seed{seed}"
            _copy_tree(out_dir, archive_dir)
            all_results.extend(self._collect(archive_dir, subset, seed))
            # The scratch tree is removed *after* the archive copy: the path
            # is never reused, so the file-sharing cache cannot go stale on
            # it (see the comment above).
            shutil.rmtree(out_dir, ignore_errors=True)

        return EvalResult(
            benchmark=subset.benchmark,
            subset=subset.name,
            seeds=list(seeds),
            results=all_results,
            spec_sha256=spec.spec_sha256(),
            generation=generation,
            started_at=started,
            finished_at=time.time(),
            synthetic=False,
        )


def _spec_with_seed(spec: HarnessSpec, seed: int) -> HarnessSpec:
    """Copy of the spec with a different runtime seed (the only knob that moves)."""
    from dataclasses import replace

    return replace(spec, runtime=replace(spec.runtime, seed=seed))


def _harbor_executable() -> str:
    """The ``harbor`` CLI to run.

    Harbor needs Python >= 3.12, the framework does not, so the two live in
    different virtual environments in this project.  The console script is
    therefore *not* on the PATH of the interpreter that runs the framework,
    and the runner has to find it: next to the running interpreter first,
    then in the sibling ``.venv312``, then on the PATH.
    """
    project = Path(__file__).resolve().parent.parent.parent
    candidates = [
        Path(sys.executable).resolve().parent / "harbor",
        project / ".venv312" / "bin" / "harbor",
        project / ".venv" / "bin" / "harbor",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which("harbor")
    if found:
        return found
    raise FileNotFoundError(
        "the `harbor` CLI was not found.  It needs Python >= 3.12 and lives in "
        f"{project / '.venv312' / 'bin' / 'harbor'}; install it with "
        "`pip install -e '.[harbor]'` in a 3.12 environment."
    )


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy a directory tree, merging into an existing destination."""
    if not src.exists():
        return
    for path in sorted(src.rglob("*")):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _harbor_task_name(data: Mapping[str, Any]) -> str:
    """The plain task name of a Harbor trial result.

    Harbor 0.24.0 writes three different spellings of the same task:

    * ``trial_name``  -- ``prompt-1000__UWTkVJy`` (name + a hash of the
      trial), which is the only one that is *not* qualified by the dataset;
    * ``task_name``   -- ``ifeval/prompt-1000`` (dataset-qualified);
    * ``task_id``     -- ``{"path": "datasets/ifeval/prompt-1000"}``.

    The framework's subsets are keyed by the plain directory name, so the
    collector has to strip the qualification -- otherwise every trial is
    scored as "no result file" and the whole study reads as 0.
    """
    trial = str(data.get("trial_name") or "")
    if trial:
        return trial.split("__", 1)[0]
    name = str(data.get("task_name") or "")
    if name:
        return name.rsplit("/", 1)[-1]
    task_id = data.get("task_id")
    if isinstance(task_id, Mapping):
        path = str(task_id.get("path") or "")
        if path:
            return path.rsplit("/", 1)[-1]
    return ""


def _harbor_reward(data: Mapping[str, Any]) -> float:
    """The reward of a Harbor trial result.

    0.24.0 nests it: ``verifier_result = {"rewards": {"reward": 1.0}}``.
    Older layouts put it directly under ``verifier_result``.  Both are read,
    because a missing reward must never silently become a *pass*.
    """
    verifier = data.get("verifier_result") or {}
    if not isinstance(verifier, Mapping):
        return 0.0
    rewards = verifier.get("rewards")
    if isinstance(rewards, Mapping) and "reward" in rewards:
        return float(rewards["reward"] or 0.0)
    if "reward" in verifier:
        return float(verifier["reward"] or 0.0)
    return 0.0


def _parse_timestamp(value: Any) -> float | None:
    """A unix timestamp from either a number or an ISO-8601 string.

    Harbor writes ISO-8601 strings (``2026-10-20T09:47:00.000000Z``); the
    framework's own records use numbers.  ``float()`` on the former raises,
    which used to leave every ``duration_s`` at 0.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _duration_seconds(start: Any, end: Any) -> float:
    """The wall-clock duration of a phase, or 0 when it cannot be determined."""
    t0 = _parse_timestamp(start)
    t1 = _parse_timestamp(end)
    if t0 is None or t1 is None:
        return 0.0
    return max(0.0, t1 - t0)
    verifier = data.get("verifier_result") or {}
    if not isinstance(verifier, Mapping):
        return 0.0
    rewards = verifier.get("rewards")
    if isinstance(rewards, Mapping) and "reward" in rewards:
        return float(rewards["reward"] or 0.0)
    if "reward" in verifier:
        return float(verifier["reward"] or 0.0)
    return 0.0


# --------------------------------------------------------------------------- #
# mini-swe-agent's own SWE-bench runner
# --------------------------------------------------------------------------- #


class MiniBackend:
    """Runs SWE-bench through ``mini-extra swebench``."""

    name = "mini"

    def __init__(
        self,
        *,
        dataset: str = "verified",
        n_concurrent: int = 16,
        extra_args: Sequence[str] = (),
        dry_run: bool = False,
    ):
        self.dataset = dataset
        self.n_concurrent = n_concurrent
        self.extra_args = tuple(extra_args)
        self.dry_run = dry_run

    def run(
        self,
        spec: HarnessSpec,
        subset: Subset,
        seeds: Sequence[int],
        workdir: Path,
        *,
        generation: int | None = None,
        base_config: Mapping[str, Any] | None = None,
    ) -> EvalResult:
        from ..adapter import to_mini_config, write_mini_config

        workdir.mkdir(parents=True, exist_ok=True)
        all_results: list[TaskResult] = []
        started = time.time()

        for seed in seeds:
            seed_spec = _spec_with_seed(spec, seed)
            config = to_mini_config(seed_spec, base_config, spec_path=workdir / "spec.yaml")
            config_path = write_mini_config(config, workdir / f"mini_config.seed{seed}.yaml")
            cmd = [
                "mini-extra",
                "swebench",
                "--subset",
                self.dataset,
                "--config",
                str(config_path),
                "--agent-class",
                "rsi.rhi_agent.RhiAgent",
                "-n",
                str(self.n_concurrent),
            ]
            cmd += list(self.extra_args)
            (workdir / f"command.seed{seed}.txt").write_text(
                " ".join(shlex.quote(c) for c in cmd) + "\n", encoding="utf-8"
            )
            if self.dry_run:
                continue
            proc = subprocess.run(cmd, capture_output=True, text=True)
            (workdir / f"mini.seed{seed}.log").write_text(
                (proc.stdout or "") + (proc.stderr or ""), encoding="utf-8"
            )
            all_results.extend(self._collect(workdir, subset, seed))

        return EvalResult(
            benchmark=subset.benchmark,
            subset=subset.name,
            seeds=list(seeds),
            results=all_results,
            spec_sha256=spec.spec_sha256(),
            generation=generation,
            started_at=started,
            finished_at=time.time(),
            synthetic=False,
        )

    def _collect(self, workdir: Path, subset: Subset, seed: int) -> list[TaskResult]:
        """Parse mini-swe-agent's own batch output (``*.json`` result files)."""
        found: dict[str, TaskResult] = {}
        for path in sorted(workdir.rglob("*.json")):
            if path.name in ("spec.yaml",) or "config" in path.name:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            instance_id = data.get("instance_id") or data.get("task_id")
            if not instance_id:
                continue
            resolved = bool(data.get("resolved", data.get("passed", False)))
            found[instance_id] = TaskResult(
                task_id=instance_id,
                seed=seed,
                passed=resolved,
                duration_s=float(data.get("duration", 0.0) or 0.0),
                error=data.get("error"),
            )
        results = [
            found.get(
                task_id,
                TaskResult(task_id=task_id, seed=seed, passed=False, error="no result file"),
            )
            for task_id in subset.task_ids
        ]
        return results


# --------------------------------------------------------------------------- #
# Synthetic
# --------------------------------------------------------------------------- #


class SyntheticBackend:
    """Deterministic offline stand-in for a real benchmark.

    The score of a (task, seed) pair is derived from a hash of the spec, so it
    is reproducible and *does* change when the harness changes -- which is
    exactly what the statistics and optimizer code needs in order to be tested.
    It is **not** a measurement of anything; every result is flagged
    ``synthetic=True``.
    """

    name = "synthetic"

    def __init__(self, *, base_rate: float = 0.4, signal: float = 0.25, noise: float = 0.35):
        self.base_rate = base_rate
        self.signal = signal
        self.noise = noise

    def _score(self, spec: HarnessSpec, task_id: str, seed: int) -> bool:
        digest = hashlib.sha256(
            f"{spec.spec_sha256()}|{task_id}|{seed}".encode("utf-8")
        ).digest()
        u = int.from_bytes(digest[:8], "big") / float(1 << 64)
        # The "quality" of the harness is a deterministic function of the spec.
        quality = self.signal * (_spec_quality(spec) - 0.5) * 2
        threshold = self.base_rate + self.noise * (u - 0.5) - quality
        return u < threshold

    def run(
        self,
        spec: HarnessSpec,
        subset: Subset,
        seeds: Sequence[int],
        workdir: Path,
        *,
        generation: int | None = None,
        base_config: Mapping[str, Any] | None = None,
    ) -> EvalResult:
        workdir.mkdir(parents=True, exist_ok=True)
        results: list[TaskResult] = []
        for seed in seeds:
            for task_id in subset.task_ids:
                passed = self._score(spec, task_id, seed)
                results.append(
                    TaskResult(
                        task_id=task_id,
                        seed=seed,
                        passed=passed,
                        duration_s=1.0 + (hash((task_id, seed)) % 500) / 100.0,
                        n_llm_calls=10 + (hash((task_id, seed, spec.spec_sha256())) % 20),
                        cost_usd=0.01 * (1 + (hash((task_id, seed)) % 30)),
                        synthetic=True,
                    )
                )
        return EvalResult(
            benchmark=subset.benchmark,
            subset=subset.name,
            seeds=list(seeds),
            results=results,
            spec_sha256=spec.spec_sha256(),
            generation=generation,
            synthetic=True,
        )


def _spec_quality(spec: HarnessSpec) -> float:
    """A deterministic, arbitrary "quality" proxy of a spec.

    The synthetic backend needs *some* monotone signal from the spec, otherwise
    no configuration could ever look better than another one.  In a real
    evaluation this role is played by the benchmark.
    """
    flat = json.dumps(spec.to_dict(), sort_keys=True)
    digest = hashlib.sha256(flat.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / float(1 << 32)


def build_backend(spec_: str, **kwargs: Any) -> Backend:
    """Factory used by the CLI."""
    kind = spec_.lower()
    if kind == "harbor":
        return HarborBackend(**kwargs)
    if kind == "mini":
        return MiniBackend(**kwargs)
    if kind == "synthetic":
        return SyntheticBackend(**kwargs)
    raise ValueError(f"unknown backend {spec_!r} (harbor | mini | synthetic)")
