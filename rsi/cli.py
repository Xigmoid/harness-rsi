"""Command line interface.

    rsi init                     scaffold the project (specs, job, subsets, ledger)
    rsi gen propose              ask the optimizer for the next generation
    rsi gen evaluate             run the benchmark for a generation
    rsi gen compare A B          paired statistics between two generations
    rsi analyze fields           per-field effect sizes
    rsi analyze homogenization   similarity curve between generations
    rsi noise-floor              measure the resolution of the pipeline
    rsi report                   render ledger/report.md
    rsi validate                 validate a spec file
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .adapter import default_base_config, to_mini_config, write_mini_config
from .benchmarks import (
    BENCHMARKS,
    SUITES,
    BenchmarkSpec,
    benchmarks_in_suite,
    get_benchmark,
    suite_benchmarks,
    suite_capabilities,
)
from .evaluation.noise_floor import estimate_noise_floor
from .evaluation.runner import (
    EvalResult,
    HarborBackend,
    MiniBackend,
    SyntheticBackend,
    build_backend,
    render_harbor_job,
    verify_job_invariance,
)
from .evaluation.stats import accept_generation, paired_delta
from .evaluation.subsets import (
    Subset,
    list_subsets,
    make_split_by_ratio,
    save_subset,
    verify_subset,
)
from .fields import expand_fields
from .ledger.diff import (
    build_report,
    diff_specs,
    effect_by_field,
    homogenization_curve,
    transfer_matrix,
)
from .ledger.ledger import Generation, Ledger
from .optimizer.bandit import Bandit, ThompsonBandit
from .optimizer.base import Observation, Optimizer
from .optimizer.cem import CEM
from .optimizer.random_search import RandomSearch
from .schema import HarnessSpec, load_spec, save_spec

__all__ = ["main", "Project"]

DEFAULT_CONFIG_NAME = "rsi.yaml"


# --------------------------------------------------------------------------- #
# Project
# --------------------------------------------------------------------------- #


@dataclass
class Project:
    """The on-disk layout of one RSI experiment."""

    root: Path

    @classmethod
    def load(cls, root: str | Path = ".") -> "Project":
        return cls(root=Path(root).resolve())

    # -- paths -------------------------------------------------------------- #

    @property
    def config_path(self) -> Path:
        return self.root / DEFAULT_CONFIG_NAME

    @property
    def harness_dir(self) -> Path:
        return self.root / self.config.get("paths", {}).get("harness", "harness")

    @property
    def harbor_dir(self) -> Path:
        return self.root / self.config.get("paths", {}).get("harbor", "harbor")

    @property
    def subsets_dir(self) -> Path:
        return self.root / self.config.get("paths", {}).get("subsets", "subsets")

    @property
    def runs_dir(self) -> Path:
        return self.root / self.config.get("paths", {}).get("runs", "runs")

    @property
    def ledger_dir(self) -> Path:
        return self.root / self.config.get("paths", {}).get("ledger", "ledger")

    @property
    def base_config_path(self) -> Path:
        return self.harness_dir / "base_config.yaml"

    # -- config ------------------------------------------------------------- #

    _config_cache: dict[str, Any] | None = None

    @property
    def config(self) -> dict[str, Any]:
        if self._config_cache is None:
            path = self.config_path
            data: dict[str, Any] = {}
            if path.exists():
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(loaded, Mapping):
                    data = dict(loaded)
            self._config_cache = data
        return self._config_cache

    def save_config(self, config: Mapping[str, Any]) -> None:
        self.config_path.write_text(
            yaml.safe_dump(dict(config), sort_keys=True, allow_unicode=True), encoding="utf-8"
        )
        self._config_cache = None

    # -- specs -------------------------------------------------------------- #

    def spec_path(self, gen: int) -> Path:
        return self.harness_dir / f"gen{gen:02d}.yaml"

    def load_spec(self, gen: int) -> HarnessSpec:
        path = self.spec_path(gen)
        if not path.exists():
            raise FileNotFoundError(f"no spec for generation {gen} at {path}")
        return load_spec(path)

    def list_generations(self) -> list[int]:
        out = []
        if not self.harness_dir.exists():
            return out
        for path in self.harness_dir.glob("gen*.yaml"):
            if path.name == "base_config.yaml":
                continue
            stem = path.stem
            if stem.startswith("gen") and stem[3:].isdigit():
                out.append(int(stem[3:]))
        return sorted(out)

    def base_mini_config(self) -> dict[str, Any]:
        if self.base_config_path.exists():
            return yaml.safe_load(self.base_config_path.read_text(encoding="utf-8"))
        endpoint = dict(self.config.get("model_endpoint", {}) or {})
        return default_base_config(
            model_name=self.config.get("model", "anthropic/claude-sonnet-4-6"),
            action_protocol=self.config.get("action_protocol", "text"),
            **endpoint,
        )

    # -- subsets ------------------------------------------------------------ #

    def subset(self, name: str) -> Subset:
        for subset in list_subsets(self.subsets_dir):
            if subset.name == name:
                return subset
        raise FileNotFoundError(f"no subset named {name!r} in {self.subsets_dir}")

    # -- ledger ------------------------------------------------------------- #

    def ledger(self) -> Ledger:
        return Ledger(self.ledger_dir).load()


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _default_spec(model: str) -> HarnessSpec:
    """The gen-0 baseline harness: an orchestrator plus two subagents."""
    from .schema import (
        AgentSpec,
        AuxiliaryRules,
        Contract,
        ContractFormat,
        Hop,
        InstructionSlots,
        Runtime,
    )

    return HarnessSpec(
        version=0,
        agents=(
            AgentSpec(
                id="orchestrator",
                role="coordinator",
                instruction_template="standard",
                instruction_slots=InstructionSlots(
                    goal="primary",
                    required_outputs=("summary",),
                    forbidden_outputs=("raw_logs",),
                    verification="self_check",
                ),
                contract=Contract(
                    out_fields=("plan", "summary", "status"),
                    in_fields=("task", "findings", "evidence", "patch", "test_results", "status"),
                    format=ContractFormat.JSON,
                    max_tokens=800,
                    free_text=True,
                ),
                model=model,
                reasoning_effort="high",
                tools=("read_file", "grep_file", "run_bash"),
                max_tokens=8192,
            ),
            AgentSpec(
                id="investigator",
                role="analyst",
                instruction_template="standard",
                instruction_slots=InstructionSlots(
                    goal="thorough",
                    required_outputs=("findings", "evidence"),
                    forbidden_outputs=("patch",),
                    verification="self_check",
                ),
                contract=Contract(
                    out_fields=("findings", "evidence"),
                    in_fields=("task", "plan"),
                    format=ContractFormat.JSON,
                    max_tokens=800,
                    free_text=True,
                ),
                model=model,
                reasoning_effort="high",
                tools=("read_file", "grep_file", "run_bash"),
                max_tokens=8192,
            ),
            AgentSpec(
                id="implementer",
                role="implementer",
                instruction_template="standard",
                instruction_slots=InstructionSlots(
                    goal="primary",
                    required_outputs=("patch", "test_results"),
                    forbidden_outputs=("questions",),
                    verification="self_check",
                ),
                contract=Contract(
                    out_fields=("patch", "test_results"),
                    in_fields=("task", "findings", "evidence"),
                    format=ContractFormat.JSON,
                    max_tokens=800,
                    free_text=True,
                ),
                model=model,
                reasoning_effort="high",
                tools=("read_file", "grep_file", "run_bash"),
                max_tokens=8192,
            ),
        ),
        workflow=(
            Hop(id="h1", source="orchestrator", target="investigator", pass_fields=("task", "plan")),
            Hop(id="h2", source="orchestrator", target="implementer", pass_fields=("task", "findings")),
            Hop(id="h3", source="investigator", target="orchestrator", pass_fields=("findings", "evidence")),
            Hop(id="h4", source="implementer", target="orchestrator", pass_fields=("patch", "test_results")),
        ),
        auxiliary_rules=AuxiliaryRules(
            acceptance_gates=(
                "the change is applied in the workspace",
                "the relevant tests pass",
            ),
            failure_fallback=("retry_once", "degrade_to_free_text"),
            recall_triggers=("the findings do not point at a concrete file",),
            communication_rules=("every handoff must be self-contained",),
        ),
        runtime=Runtime(
            max_iterations=40,
            observation_max_chars=10000,
            context_keep_recent=20,
            seed=1234,
            temperature=0.0,
            max_hops=6,
            hop_context_max_chars=6000,
        ),
    )


def cmd_init(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    for directory in (
        project.harness_dir,
        project.harbor_dir,
        project.subsets_dir,
        project.runs_dir,
        project.ledger_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    model = args.model or project.config.get("model", "anthropic/claude-sonnet-4-6")

    # Build (and validate) everything *before* touching the disk, so a failure
    # cannot leave a half-initialized project behind.
    endpoint = _endpoint_settings(project, args)
    spec = _default_spec(model)
    base = default_base_config(
        model_name=model, action_protocol=args.action_protocol, **endpoint
    )
    job = yaml.safe_load(_default_harbor_job(model))
    config = {
        "model": model,
        "backend": args.backend,
        "seeds": args.seeds,
        "n_concurrent": args.n_concurrent,
        "max_retries": args.max_retries,
        "model_endpoint": endpoint,
        "action_protocol": args.action_protocol,
        "paths": {
            "harness": "harness",
            "harbor": "harbor",
            "subsets": "subsets",
            "runs": "runs",
            "ledger": "ledger",
        },
    }

    spec_path = project.spec_path(0)
    if not spec_path.exists():
        save_spec(spec, spec_path)
        print(f"wrote {spec_path}")
    else:
        print(f"kept existing {spec_path}")

    if not project.base_config_path.exists():
        write_mini_config(base, project.base_config_path)
        print(f"wrote {project.base_config_path}")
    else:
        print(f"kept existing {project.base_config_path}")

    # One job template per benchmark.  Every template is frozen; only the
    # `agents[0].kwargs.config` block is replaced per generation (C-1).
    for key in benchmarks_in_suite(args.suite):
        job_path = project.harbor_dir / f"job.{key}.yaml"
        if not job_path.exists():
            job_path.write_text(_default_harbor_job(model, key), encoding="utf-8")
            print(f"wrote {job_path}")
        else:
            print(f"kept existing {job_path}")
    default_job = project.harbor_dir / "job.yaml"
    if not default_job.exists():
        first = benchmarks_in_suite(args.suite)[0]
        default_job.write_text(_default_harbor_job(model, first), encoding="utf-8")
        print(f"wrote {default_job}")

    if not project.config_path.exists():
        project.save_config(config)
        print(f"wrote {project.config_path}")
    else:
        print(f"kept existing {project.config_path}")

    _init_subsets(project, args)

    print(f"\nspec sha256 (gen0): {load_spec(spec_path).spec_sha256()}")
    print("next: rsi gen evaluate --gen 0 --subset tb21_dev")
    return 0


def _endpoint_settings(project: Project, args: argparse.Namespace) -> dict[str, Any]:
    """The endpoint settings of the model, if any."""
    endpoint = dict(project.config.get("model_endpoint", {}) or {})
    for key, attr in (
        ("api_base", "api_base"),
        ("api_key", "api_key"),
        ("reasoning_effort", "reasoning_effort"),
    ):
        value = getattr(args, attr, None)
        if value:
            endpoint[key] = value
    return endpoint


#: Harbor's own defaults retry nothing (``max_retries: 0``) and exclude the
#: timeout/verifier errors.  A fixed, explicit policy is required, otherwise
#: the failure rate is not comparable across generations (trap T-8).
HARBOR_RETRY_POLICY = {
    "max_retries": 3,
    "include_exceptions": [
        "ApiRateLimitError",
        "ApiInternalServerError",
        "ApiOverloadedError",
        "ApiConnectionClosedError",
    ],
}


def _default_harbor_job(model: str, benchmark: str = "tb21") -> str:
    spec = get_benchmark(benchmark)
    dataset = spec.dataset_ref
    job = {
        "jobs_dir": "runs/harbor",
        "n_attempts": 1,
        "n_concurrent_trials": 16,
        "timeout_multiplier": 1.0,
        "environment": {"type": "docker", "delete": True, "force_build": True},
        "retry": dict(HARBOR_RETRY_POLICY),
        "agents": [
            {
                # The pinning adapter keeps Harbor from overriding the
                # protocol fields of the generation's config (C-4); see
                # audit/harbor_mini_swe_agent.md.  Harbor's import path format
                # is `module.path:ClassName`.
                "import_path": "rsi.harbor_agent:RsiMiniSweAgent",
                "name": "rsi-mini-swe-agent",
                "model_name": model,
                # `config` is written into the container and layered over
                # mini-swe-agent's own `mini` config.  This block is the ONLY
                # part of the job file that differs between generations
                # (constraint C-1).  `version` pins the agent runtime that
                # Harbor installs in the container (constraint C-2).
                "kwargs": {
                    "version": "2.4.6",
                    "config": {"agent": {"agent_class": "rsi.rhi_agent.RhiAgent"}},
                },
            }
        ],
        "datasets": [dataset],
    }
    return yaml.safe_dump(job, sort_keys=False, allow_unicode=True)


#: Backwards-compatible alias: the benchmark registry lives in
#: :mod:`rsi.benchmarks`.


def _init_subsets(project: Project, args: argparse.Namespace) -> None:
    task_files: dict[str, str] = {}
    for item in args.tasks or []:
        if "=" not in item:
            print(f"! ignoring malformed --tasks entry {item!r} (expected BENCH=FILE)")
            continue
        bench, path = item.split("=", 1)
        task_files[bench] = path

    for key in benchmarks_in_suite(args.suite):
        spec = get_benchmark(key)
        source = "placeholder"
        if key in task_files:
            ids = [
                line.strip()
                for line in Path(task_files[key]).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
            source = task_files[key]
        elif spec.n_tasks:
            ids = [f"{key}-task-{i:04d}" for i in range(spec.n_tasks)]
            print(
                f"! {key}: no task list given, wrote {len(ids)} placeholder ids. "
                f"Replace them with `rsi init --tasks {key}=<file>` before reporting numbers."
            )
        else:
            print(f"! {key}: no task list and no known size; skipped")
            continue
        dev, heldout = make_split_by_ratio(
            key, ids, dev_ratio=spec.dev_ratio, source=source
        )
        for subset in (dev, heldout):
            path = project.subsets_dir / f"{subset.name}.json"
            if path.exists():
                print(f"kept existing {path}")
                continue
            save_subset(subset, path)
            print(f"wrote {path} ({subset.n} tasks, sha256={subset.sha256()[:12]}...)")


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    fields = expand_fields(spec)
    mutable = [f for f in fields if f.mutable]
    print(f"OK  {args.spec}")
    print(f"    spec_sha256 : {spec.spec_sha256()}")
    print(f"    version     : {spec.version}")
    print(f"    agents      : {', '.join(a.id for a in spec.agents)}")
    print(f"    orchestrator: {spec.orchestrator}")
    print(f"    hops        : {len(spec.workflow)}")
    print(f"    fields      : {len(fields)} ({len(mutable)} mutable)")
    return 0


def _make_optimizer(name: str, args: argparse.Namespace, seed: int | None = None) -> Optimizer:
    effective = args.seed if seed is None else seed
    if name == "random_search":
        return RandomSearch(max_edits=args.max_edits, seed=effective)
    if name == "bandit_ucb":
        return Bandit(max_edits=max(1, min(args.max_edits, 1)), seed=effective)
    if name == "bandit_thompson":
        return ThompsonBandit(max_edits=1, seed=effective)
    if name == "cem":
        return CEM(max_edits=1, seed=effective)
    raise SystemExit(f"unknown optimizer {name!r}")


def _chain_seed(seed: int, objective: str | None) -> int:
    """Derive a chain-specific seed from ``seed`` and the objective.

    The specialised mode runs one chain per capability, all of them from the
    same baseline.  If the seed were used as-is, every chain would draw the
    *same* edit from the same parent -- the "four specialised harnesses"
    would be one harness four times, and the transfer matrix would be
    measuring nothing.

    The mix is deterministic (crc32, not Python's salted ``hash``), so a
    chain is still reproducible: the same ``--seed`` and the same
    ``--objective`` always give the same proposal.
    """
    if not objective:
        return seed
    return (seed + zlib.crc32(objective.encode("utf-8"))) % (2**31)


def cmd_gen_propose(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    generations = project.list_generations()
    if not generations:
        raise SystemExit("no generations found; run `rsi init` first")
    parent_gen = args.parent if args.parent is not None else generations[-1]
    if parent_gen not in generations:
        raise SystemExit(
            f"no generation {parent_gen}; available: {generations}"
        )
    spec = project.load_spec(parent_gen)
    ledger = project.ledger()

    optimizer = _make_optimizer(
        args.optimizer, args, seed=_chain_seed(args.seed, args.objective)
    )
    # The objective decides *what* is being optimized: one capability family
    # (specialisation) or the whole suite at once (the joint control).
    #
    # The history is restricted to the **lineage of the parent**: a chain's
    # optimizer may only learn from its own ancestors.  Without this filter
    # the four specialised chains would read each other's measurements, and
    # the transfer evaluations of the other chains would leak into the
    # history of a proposal.
    lineage = {g.gen for g in ledger.chain(parent_gen)}
    history = ledger.observations(args.objective, lineage=lineage)
    proposal = optimizer.propose(spec, history)

    changes = diff_specs(spec, proposal.spec)
    # The generation *number* must never collide with an existing one: the
    # specialised mode runs one chain per capability, all of them branching
    # from the same baseline, so two chains legitimately propose at the same
    # time.  The chain is defined by the *parent*, not by the number --
    # numbering by ``parent + 1`` would make the second chain silently
    # overwrite the first.
    new_gen = max(generations) + 1
    # The lineage is stored *inside* the spec, so a generation is
    # self-describing: the ledger never has to guess who the parent was.
    new_spec = HarnessSpec.from_dict(
        {
            **proposal.spec.to_dict(),
            "meta": {
                **proposal.spec.meta,
                "parent": parent_gen,
                "optimizer": proposal.strategy,
                "objective": args.objective,
                "rationale": proposal.rationale,
                "changed_fields": [c.path for c in changes],
            },
        }
    )

    out = Path(args.out) if args.out else project.spec_path(new_gen)
    if out.exists() and not args.overwrite:
        raise SystemExit(
            f"{out} already exists; pass --overwrite to replace it"
        )
    save_spec(new_spec, out)
    print(f"proposed gen {new_gen} (parent {parent_gen}) via {proposal.strategy}")
    print(f"  objective  : {args.objective or '(first evaluation)'}")
    print(f"  rationale  : {proposal.rationale}")
    for change in changes:
        print(f"  {change.path}: {change.before} -> {change.after}")
    print(f"  written    : {out}")
    print(f"  sha256     : {new_spec.spec_sha256()}")
    return 0


def cmd_gen_evaluate(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    gen = args.gen
    spec = project.load_spec(gen)
    seeds = args.seeds or list(project.config.get("seeds", [1234]))

    if args.suite:
        backend = _build_backend(project, args.backend)
        return _evaluate_suite(project, gen, spec, seeds, backend, args)

    subset = project.subset(args.subset)
    # The job template is *per benchmark*: the dataset and the frozen task
    # list differ.  Evaluating a subset without picking its benchmark's
    # template silently runs a different benchmark's tasks.
    benchmark = _benchmark_of_subset(subset.name)
    backend = _build_backend(project, args.backend, benchmark=benchmark)
    # A tag separates two measurements of the *same* (generation, subset).
    # The transfer probe needs exactly that: it measures generation 0 on
    # every benchmark with a single seed, while the home measurement of
    # generation 0 uses three seeds.  Without a tag the two would share one
    # path, and the single-seed transfer result would silently overwrite the
    # three-seed home result -- taking the noise floor with it.
    tag = f".{args.tag}" if args.tag else ""
    workdir = project.runs_dir / f"gen{gen:03d}" / f"{subset.name}{tag}"
    result = backend.run(
        spec,
        subset,
        seeds,
        workdir,
        generation=gen,
        base_config=project.base_mini_config(),
    )
    print(
        f"gen {gen} on {subset.name}{tag}: mean={result.mean_score:.4f} "
        f"pass@1={result.pass_at_1:.4f} runs={result.n_runs} errors={result.n_errors}"
    )
    if result.synthetic:
        print("  (SYNTHETIC backend -- these numbers are a test double, not a measurement)")

    _write_result(project, gen, f"{subset.name}{tag}", result)
    _update_ledger(project, gen, f"{subset.name}{tag}", result)
    return 0


def _evaluate_suite(
    project: Project,
    gen: int,
    spec: HarnessSpec,
    seeds: list[int],
    backend: Any,
    args: argparse.Namespace,
) -> int:
    """Evaluate one generation on every benchmark of a suite."""
    from .evaluation.suite import composite_score

    benchmarks = suite_benchmarks(args.suite)
    scores: dict[str, float] = {}
    for benchmark in benchmarks:
        subset_name = f"{benchmark.key}_dev"
        try:
            subset = project.subset(subset_name)
        except FileNotFoundError:
            print(f"! skipping {benchmark.key}: no subset {subset_name} (run `rsi init --suite {args.suite}`)")
            continue
        benchmark_backend = _build_backend(project, args.backend, benchmark=benchmark)
        workdir = project.runs_dir / f"gen{gen:03d}" / subset_name
        result = benchmark_backend.run(
            spec,
            subset,
            seeds,
            workdir,
            generation=gen,
            base_config=project.base_mini_config(),
        )
        scores[benchmark.key] = result.mean_score
        print(
            f"  {benchmark.key:<12} ({benchmark.capability:<22}) mean={result.mean_score:.4f} "
            f"runs={result.n_runs} errors={result.n_errors}"
        )
        _write_result(project, gen, subset_name, result)
        _update_ledger(project, gen, subset_name, result)

    if not scores:
        raise SystemExit(f"no benchmark of suite {args.suite!r} could be evaluated")
    composite = composite_score(scores)
    print(
        f"gen {gen} on suite {args.suite}: composite={composite:.4f} "
        f"over {len(scores)} benchmark(s)"
    )
    return 0


def _benchmark_of_subset(name: str):
    """The benchmark a subset belongs to (``<key>_dev`` -> the benchmark).

    Returns ``None`` when the name does not follow the convention, so the
    caller can fall back to the default job template.
    """
    from .benchmarks import get_benchmark

    key = name.rsplit("_", 1)[0]
    try:
        return get_benchmark(key)
    except KeyError:
        return None


def _build_backend(project: Project, backend_name: str | None, benchmark=None) -> Any:
    name = backend_name or project.config.get("backend", "synthetic")
    if name == "harbor":
        if benchmark is not None:
            candidate = project.harbor_dir / f"job.{benchmark.key}.yaml"
            if not candidate.exists():
                raise SystemExit(
                    f"no harbor job template for {benchmark.key}: {candidate} is missing. "
                    "Every benchmark needs its own frozen template (the dataset and "
                    "the task list differ); generate it with "
                    f"`python scripts/make_real_splits.py --benchmarks {benchmark.key}`."
                )
            job = candidate
        else:
            job = project.harbor_dir / "job.yaml"
            if not job.exists():
                raise SystemExit(
                    f"no harbor job template: {job} is missing and no benchmark was "
                    "given.  Every benchmark needs its own frozen template."
                )
        return HarborBackend(
            job,
            jobs_dir=project.runs_dir / "harbor",
            n_concurrent=int(project.config.get("n_concurrent", 16)),
            max_retries=int(project.config.get("max_retries", 3)),
        )
    if name == "mini":
        return MiniBackend()
    if name == "synthetic":
        return SyntheticBackend()
    raise SystemExit(f"unknown backend {name!r}")


def _write_result(project: Project, gen: int, subset: str, result: EvalResult) -> Path:
    path = project.runs_dir / f"gen{gen:03d}" / subset / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _update_ledger(project: Project, gen: int, subset: str, result: EvalResult) -> None:
    ledger = project.ledger()
    spec = project.load_spec(gen)
    parent = spec.meta.get("parent")
    parent = int(parent) if parent is not None else None
    if gen in ledger.generations:
        record = ledger.generations[gen]
    else:
        record = Generation(gen=gen, spec=spec, parent=parent)
    # The changed fields are derived from the specs, not trusted from the
    # spec's metadata: the ledger must be able to reconstruct the edit even if
    # the proposal was written by hand.
    if parent is not None and parent in ledger.generations:
        record.changed_fields = tuple(
            diff_specs(ledger.generations[parent].spec, spec)
        )
    record.evaluations[subset] = result
    record.cost = {
        "usd": sum(ev.total_cost_usd for ev in record.evaluations.values()),
        "llm_calls": sum(ev.total_llm_calls for ev in record.evaluations.values()),
        "runs": sum(ev.n_runs for ev in record.evaluations.values()),
    }
    ledger.add_or_update(record)
    ledger.save()
    print(f"  ledger: {project.ledger_dir / f'gen{gen:03d}.yaml'}")


def cmd_gen_compare(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    ledger = project.ledger()
    new = ledger.get(args.new)
    old = ledger.get(args.old)
    key = args.subset or next(iter(sorted(new.evaluations)), None)
    if key is None:
        raise SystemExit("no evaluation recorded for the new generation")
    result = paired_delta(new.per_task(key), old.per_task(key), seed=args.seed)
    ok, reason = accept_generation(result)
    print(f"gen {args.new} vs gen {args.old} on {key}")
    print(f"  {reason}")
    print(f"  -> {'ACCEPT' if ok else 'REJECT'}")
    return 0 if ok else 1


def cmd_analyze_fields(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    ledger = project.ledger()
    rows = effect_by_field(ledger, args.subset)
    if not rows:
        print("no single-field generations recorded yet")
        return 0
    print(f"{'field':<52} {'group':<12} {'trials':>6} {'mean d':>9} {'sd':>8}")
    for row in rows:
        print(
            f"{row['path']:<52} {row['group']:<12} {row['n_trials']:>6} "
            f"{row['mean_delta']:>+9.4f} {row['sd_delta']:>8.4f}"
        )
    return 0


def _spec_resolver(project: Project) -> "Callable[[int], Any]":
    """Load a generation's spec from disk, even if it was never evaluated.

    The baseline of the specialised mode is often not evaluated on its own,
    but the lineage curve still needs its spec to measure the distance a
    generation moved away from it.
    """

    def resolve(gen: int) -> Any:
        try:
            return project.load_spec(gen)
        except (FileNotFoundError, ValueError):
            return None

    return resolve


def cmd_analyze_homogenization(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    ledger = project.ledger()
    curve = homogenization_curve(ledger, spec_resolver=_spec_resolver(project))
    if not curve:
        print("need at least two generations")
        return 0
    print(f"{'from':>5} {'to':>5} {'objective':>22} {'similarity':>11} {'changed':>8}")
    for point in curve:
        print(
            f"{point['from_gen']:>5} {point['to_gen']:>5} {point.get('objective', '-'):>22} "
            f"{point['similarity']:>11.4f} {point['n_changed_fields']:>8}"
        )
    return 0


def cmd_analyze_transfer(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    ledger = project.ledger()
    matrix = transfer_matrix(ledger, args.suites)
    benchmarks = matrix["benchmarks"]
    if not matrix["rows"]:
        print("no generations recorded yet")
        return 0
    header = f"{'gen':>4} {'objective':<24} " + " ".join(f"{b:>10}" for b in benchmarks)
    print(header)
    print("-" * len(header))
    for row in matrix["rows"]:
        cells = " ".join(
            f"{row['scores'][b]:>10.4f}" if row["scores"].get(b) is not None else f"{'-':>10}"
            for b in benchmarks
        )
        print(f"{row['gen']:>4} {str(row['objective']):<24} {cells}")
    return 0


def cmd_noise_floor(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    spec = project.load_spec(args.gen)
    subset = project.subset(args.subset)
    seeds = args.seeds or list(range(1234, 1234 + 5))
    backend = _build_backend(project, args.backend)
    workdir = project.runs_dir / "noise_floor" / f"gen{args.gen:03d}"
    floor = estimate_noise_floor(
        spec,
        subset,
        seeds,
        backend,
        workdir,
        base_config=project.base_mini_config(),
        generation=args.gen,
    )
    print(floor.summary())
    out = project.runs_dir / "noise_floor" / f"gen{args.gen:03d}" / "noise_floor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(floor.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"  written: {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    ledger = project.ledger()
    report = build_report(ledger, spec_resolver=_spec_resolver(project))
    path = project.ledger_dir / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(f"wrote {path}")
    return 0


def cmd_spec_show(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    spec = project.load_spec(args.gen)
    print(yaml.safe_dump(spec.to_dict(), sort_keys=True, allow_unicode=True))
    return 0


def cmd_spec_diff(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    old = project.load_spec(args.old)
    new = project.load_spec(args.new)
    changes = diff_specs(old, new)
    if not changes:
        print("no differences")
        return 0
    for change in changes:
        print(f"{change.path} [{change.group}]: {change.before} -> {change.after}")
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsi", description=__doc__)
    parser.add_argument("--root", default=".", help="project root (default: .)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="scaffold the project")
    p.add_argument("--model", default=None, help="model id, e.g. anthropic/claude-sonnet-4-6")
    p.add_argument("--backend", default="synthetic", choices=["synthetic", "harbor", "mini"])
    p.add_argument(
        "--suite",
        default="all",
        help="which benchmarks to prepare: a capability suite or a single benchmark key",
    )
    p.add_argument(
        "--action-protocol",
        default="text",
        choices=["toolcall", "text"],
        help="how the agent talks to the model: a code block (default) or OpenAI tool calls",
    )
    p.add_argument("--seeds", type=int, nargs="*", default=[1234, 1235, 1236])
    p.add_argument("--n-concurrent", type=int, default=16)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--tasks",
        action="append",
        help="BENCH=FILE with one task id per line (repeatable)",
    )
    p.add_argument("--api-base", default=None, help="OpenAI-compatible endpoint base URL")
    p.add_argument("--api-key", default=None, help="API key for the endpoint")
    p.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high"],
        help="reasoning effort forwarded to the endpoint (chat_template_kwargs)",
    )
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("validate", help="validate a spec file")
    p.add_argument("spec")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("spec", help="spec utilities")
    spec_sub = p.add_subparsers(dest="spec_command", required=True)

    q = spec_sub.add_parser("show")
    q.add_argument("gen", type=int)
    q.set_defaults(func=cmd_spec_show)

    q = spec_sub.add_parser("diff")
    q.add_argument("old", type=int)
    q.add_argument("new", type=int)
    q.set_defaults(func=cmd_spec_diff)

    p = sub.add_parser("gen", help="generation lifecycle")
    gen_sub = p.add_subparsers(dest="gen_command", required=True)

    q = gen_sub.add_parser("propose", help="propose the next generation")
    q.add_argument("--optimizer", default="random_search",
                   choices=["random_search", "bandit_ucb", "bandit_thompson", "cem"])
    q.add_argument("--max-edits", type=int, default=2)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--parent", type=int, default=None)
    q.add_argument("--out", default=None)
    q.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing spec file (off by default: a silent overwrite "
        "of one chain's proposal by another chain's is a data-loss bug)",
    )
    q.add_argument(
        "--objective",
        default=None,
        help="the suite the proposal optimizes (default: the first recorded evaluation). "
        "Use a capability suite to specialise the harness for that capability. "
        "Several chains can branch from the same --parent; each gets its own "
        "generation number.",
    )
    q.set_defaults(func=cmd_gen_propose)

    q = gen_sub.add_parser("evaluate", help="evaluate a generation")
    q.add_argument("--gen", type=int, required=True)
    q.add_argument("--subset", default=None, help="a single subset, e.g. sbv_dev")
    q.add_argument(
        "--suite",
        default=None,
        help="evaluate every benchmark of a suite (a capability family) instead of one subset",
    )
    q.add_argument("--seeds", type=int, nargs="*", default=None)
    q.add_argument("--backend", default=None, choices=[None, "synthetic", "harbor", "mini"])
    q.add_argument(
        "--tag",
        default=None,
        help=(
            "a suffix that separates this measurement from another one of the "
            "same (generation, subset); e.g. 'transfer' keeps a single-seed "
            "transfer probe from overwriting the three-seed home measurement"
        ),
    )
    q.set_defaults(func=cmd_gen_evaluate)

    q = gen_sub.add_parser("compare", help="paired comparison of two generations")
    q.add_argument("new", type=int)
    q.add_argument("old", type=int)
    q.add_argument("--subset", default=None)
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(func=cmd_gen_compare)

    p = sub.add_parser("analyze", help="analysis over the ledger")
    analyze_sub = p.add_subparsers(dest="analyze_command", required=True)

    q = analyze_sub.add_parser("fields", help="per-field effect sizes")
    q.add_argument("--subset", default=None)
    q.set_defaults(func=cmd_analyze_fields)

    q = analyze_sub.add_parser("homogenization", help="similarity curve")
    q.set_defaults(func=cmd_analyze_homogenization)

    q = analyze_sub.add_parser("transfer", help="cross-capability transfer matrix")
    q.add_argument("--suites", nargs="*", default=None)
    q.set_defaults(func=cmd_analyze_transfer)

    p = sub.add_parser("noise-floor", help="measure the pipeline's resolution")
    p.add_argument("--gen", type=int, default=0)
    p.add_argument("--subset", required=True)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--backend", default=None, choices=[None, "synthetic", "harbor", "mini"])
    p.set_defaults(func=cmd_noise_floor)

    p = sub.add_parser("report", help="render ledger/report.md")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
