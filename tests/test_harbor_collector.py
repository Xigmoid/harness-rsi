"""The Harbor collector must read the files Harbor actually writes.

Three things were wrong at the same time, and all three made a real
evaluation read as **0.0 without a single error**:

* the file is called ``result.json``, not ``results.json``;
* the reward is nested: ``verifier_result.rewards.reward``;
* the task is spelled three ways, and the one the framework's subsets are
  keyed by is ``trial_name`` without the trial hash (``prompt-1000__UWTkVJy``
  -> ``prompt-1000``), while ``task_name`` is dataset-qualified
  (``ifeval/prompt-1000``).

A collector that looks for the wrong name finds nothing, and "nothing" was
counted as "failed" -- so a perfectly working harness would have scored zero
on every benchmark of the study.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rsi.evaluation.runner import (
    HarborBackend,
    _harbor_reward,
    _harbor_task_name,
    render_harbor_job,
    verify_job_invariance,
)
from rsi.evaluation.subsets import Subset


def _trial(name: str, *, reward: float, trial_hash: str = "UWTkVJy", source: str = "ifeval"):
    """A trial result in the shape Harbor 0.24.0 writes it."""
    return {
        "trial_name": f"{name}__{trial_hash}",
        "task_name": f"{source}/{name}",
        "task_id": {"path": f"datasets/{source}/{name}"},
        "verifier_result": {"rewards": {"reward": reward}},
        "agent_result": {
            "n_api_calls": 3,
            "cost_usd": 0.01,
            "n_input_tokens": 100,
            "n_output_tokens": 20,
            "n_cache_tokens": 0,
        },
        "agent_execution": {
            "started_at": "2026-10-20T09:47:00.000000Z",
            "finished_at": "2026-10-20T09:47:05.000000Z",
        },
        "exception_info": None,
    }


def test_task_name_is_the_plain_name():
    assert _harbor_task_name(_trial("prompt-1000", reward=1.0)) == "prompt-1000"
    # A name that contains "__" inside it must survive.
    assert _harbor_task_name({"trial_name": "a__b__HASH"}) == "a"
    # Fallbacks: the dataset-qualified name and the path.
    assert _harbor_task_name({"task_name": "gpqa-diamond/12"}) == "12"
    assert _harbor_task_name({"task_id": {"path": "datasets/x/y"}}) == "y"
    assert _harbor_task_name({}) == ""


def test_reward_is_read_from_the_nested_place():
    assert _harbor_reward(_trial("t", reward=1.0)) == 1.0
    assert _harbor_reward(_trial("t", reward=0.0)) == 0.0
    # The older, flat layout still works.
    assert _harbor_reward({"verifier_result": {"reward": 0.5}}) == 0.5
    # A missing reward is a failure, never a pass.
    assert _harbor_reward({}) == 0.0
    assert _harbor_reward({"verifier_result": {}}) == 0.0


def test_collect_reads_harbor_result_json(tmp_path: Path):
    """The end-to-end shape: a job directory with Harbor's files in it."""
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "prompt-1000__UWTkVJy"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(_trial("prompt-1000", reward=1.0)), encoding="utf-8"
    )
    fail_dir = job_dir / "prompt-1001__q2ixGhy"
    fail_dir.mkdir(parents=True)
    (fail_dir / "result.json").write_text(
        json.dumps(_trial("prompt-1001", reward=0.0, trial_hash="q2ixGhy")),
        encoding="utf-8",
    )
    # The job-level summary must not be mistaken for a trial.
    (job_dir / "result.json").write_text(
        json.dumps({"n_total_trials": 2, "stats": {}}), encoding="utf-8"
    )

    subset = Subset(
        name="ifeval_dev",
        split="dev",
        benchmark="ifeval",
        task_ids=("prompt-1000", "prompt-1001"),
    )
    template = tmp_path / "job.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "agents": [{"import_path": "rsi.harbor_agent:RsiMiniSweAgent"}],
                "datasets": [
                    {"name": "ifeval", "version": "1.0", "task_names": list(subset.task_ids)}
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = HarborBackend(template)
    backend._effective_task_ids = list(subset.task_ids)
    results = backend._collect(job_dir, subset, seed=1234)

    by_id = {r.task_id: r for r in results}
    assert set(by_id) == {"prompt-1000", "prompt-1001"}
    assert by_id["prompt-1000"].passed is True
    assert by_id["prompt-1001"].passed is False
    assert by_id["prompt-1000"].duration_s == pytest.approx(5.0)
    assert by_id["prompt-1000"].n_llm_calls == 3


def test_collect_counts_a_missing_trial_as_a_failure(tmp_path: Path):
    """A task that was selected but produced no result is a failure, not a
    dropped task (trap T-9)."""
    subset = Subset(
        name="ifeval_dev", split="dev", benchmark="ifeval", task_ids=("a", "b")
    )
    backend = HarborBackend(tmp_path / "missing.yaml")
    backend._effective_task_ids = ["a", "b"]
    results = backend._collect(tmp_path / "empty", subset, seed=1234)
    assert len(results) == 2
    assert all(not r.passed for r in results)
    assert all(r.error == "no result file" for r in results)


def test_a_negative_reward_is_a_failure_not_a_pass(tmp_path: Path):
    """A verifier that fails writes a *negative* reward.

    IneqMath writes ``-1.0`` when its LLM judge cannot be reached.  Judging
    the pass by the truthiness of the reward would call that a pass, because
    ``bool(-1.0)`` is ``True`` -- and the mean would then be inflated by a
    run that in fact measured nothing.
    """
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "ineq-0__abc"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(_trial("ineq-0", reward=-1.0, trial_hash="abc")), encoding="utf-8"
    )
    subset = Subset(
        name="ineqmath_dev", split="dev", benchmark="ineqmath", task_ids=("ineq-0",)
    )
    backend = HarborBackend(tmp_path / "job.yaml")
    backend._effective_task_ids = ["ineq-0"]
    results = backend._collect(job_dir, subset, seed=1234)
    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].reward == pytest.approx(-1.0)


def test_effective_tasks_must_be_a_subset_of_the_split(tmp_path: Path):
    subset = Subset(
        name="ifeval_dev", split="dev", benchmark="ifeval", task_ids=("a", "b")
    )
    backend = HarborBackend(tmp_path / "job.yaml")
    job = {"datasets": [{"task_names": ["a", "zzz"]}]}
    with pytest.raises(ValueError, match="not in"):
        backend._resolve_effective_tasks(job, subset)
    assert backend._resolve_effective_tasks(
        {"datasets": [{"task_names": ["b"]}]}, subset
    ) == ["b"]
    # No task_names at all -> the whole split runs.
    assert backend._resolve_effective_tasks({"datasets": [{}]}, subset) == ["a", "b"]


def test_job_invariance_still_holds():
    base = {
        "agents": [{"import_path": "x:Y", "kwargs": {"version": "2.4.6", "config": {"a": 1}}}],
        "datasets": [{"name": "d", "version": "1.0"}],
    }
    rendered = render_harbor_job(base, {"a": 2})
    assert verify_job_invariance(base, rendered) == []
