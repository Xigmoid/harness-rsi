"""Subsets, optimizers, the ledger and the runners."""

from __future__ import annotations

import json
import random

import pytest

from rsi.evaluation.runner import (
    EvalResult,
    SyntheticBackend,
    TaskResult,
    render_harbor_job,
    verify_job_invariance,
)
from rsi.evaluation.subsets import (
    Subset,
    make_split,
    make_split_by_ratio,
    save_subset,
    verify_subset,
)
from rsi.ledger.diff import diff_specs, effect_by_field, homogenization_curve, spec_similarity
from rsi.ledger.ledger import Generation, Ledger
from rsi.optimizer.bandit import Bandit
from rsi.optimizer.base import EditBudgetExceeded, Observation
from rsi.optimizer.cem import CEM
from rsi.optimizer.random_search import RandomSearch


# --------------------------------------------------------------------------- #
# Subsets
# --------------------------------------------------------------------------- #


class TestSubsets:
    def test_the_split_is_deterministic(self):
        ids = [f"t{i}" for i in range(50)]
        a, b = make_split("x", ids, n_dev=30, n_heldout=20)
        c, d = make_split("x", ids, n_dev=30, n_heldout=20)
        assert a.task_ids == c.task_ids
        assert b.task_ids == d.task_ids

    def test_the_split_is_disjoint_and_complete(self):
        ids = [f"t{i}" for i in range(50)]
        dev, heldout = make_split("x", ids, n_dev=30, n_heldout=20)
        assert not set(dev.task_ids) & set(heldout.task_ids)
        assert len(dev.task_ids) == 30
        assert len(heldout.task_ids) == 20

    def test_the_ratio_split_follows_the_protocol(self):
        # 500 SWE-bench tasks -> 300/200 exactly as the plan states.
        ids = [f"t{i}" for i in range(500)]
        dev, heldout = make_split_by_ratio("sbv", ids, dev_ratio=300 / 500)
        assert len(dev.task_ids) == 300
        assert len(heldout.task_ids) == 200

    def test_the_ratio_split_adapts_to_a_smaller_benchmark(self):
        ids = [f"t{i}" for i in range(89)]
        dev, heldout = make_split_by_ratio("tb21", ids, dev_ratio=300 / 445)
        assert len(dev.task_ids) + len(heldout.task_ids) == 89
        assert len(dev.task_ids) == 60

    def test_a_tampered_subset_is_detected(self, tmp_path):
        subset = Subset(name="x_dev", split="dev", benchmark="x", task_ids=("a", "b"))
        path = save_subset(subset, tmp_path / "x_dev.json")
        verify_subset(path)  # no exception
        data = json.loads(path.read_text())
        data["task_ids"] = ["a", "c"]
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="was modified"):
            verify_subset(path)

    def test_duplicate_task_ids_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            Subset(name="x", split="dev", benchmark="x", task_ids=("a", "a"))


# --------------------------------------------------------------------------- #
# Optimizers
# --------------------------------------------------------------------------- #


class TestOptimizers:
    def test_random_search_respects_the_edit_budget(self, spec):
        optimizer = RandomSearch(max_edits=2, seed=0)
        for _ in range(20):
            proposal = optimizer.propose(spec, [])
            assert 1 <= len(proposal.changed_fields) <= 2
            assert proposal.spec.spec_sha256() != spec.spec_sha256()

    def test_random_search_is_reproducible(self, spec):
        a = RandomSearch(max_edits=1, seed=5).propose(spec, [])
        b = RandomSearch(max_edits=1, seed=5).propose(spec, [])
        assert a.changed_fields == b.changed_fields
        assert a.spec.spec_sha256() == b.spec.spec_sha256()

    def test_the_edit_budget_is_enforced(self, spec):
        optimizer = RandomSearch(max_edits=1, seed=0)
        with pytest.raises(EditBudgetExceeded):
            optimizer.check_budget(["a", "b", "c"])

    def test_the_bandit_learns_the_field_effects(self, spec):
        bandit = Bandit(max_edits=1, seed=0)
        history = [
            Observation(
                generation=0,
                spec=spec,
                mean_score=0.40,
                parent=None,
                changed_fields=(),
            ),
            Observation(
                generation=1,
                spec=spec,
                mean_score=0.55,
                parent=0,
                changed_fields=("agents[0].contract.max_tokens",),
            ),
        ]
        bandit.observe(history)
        arm = bandit.arms["agents[0].contract.max_tokens"]
        assert arm.trials == 1
        assert arm.mean == pytest.approx(0.15)
        assert arm.wins == 1

    def test_the_bandit_ignores_multi_field_generations(self, spec):
        bandit = Bandit(max_edits=1, seed=0)
        history = [
            Observation(generation=0, spec=spec, mean_score=0.40, parent=None, changed_fields=()),
            Observation(
                generation=1,
                spec=spec,
                mean_score=0.55,
                parent=0,
                changed_fields=("agents[0].contract.max_tokens", "runtime.max_iterations"),
            ),
        ]
        bandit.observe(history)
        assert bandit.arms == {}

    def test_the_bandit_proposes_a_valid_spec(self, spec):
        bandit = Bandit(max_edits=1, seed=0)
        proposal = bandit.propose(spec, [])
        assert len(proposal.changed_fields) == 1
        assert proposal.spec.orchestrator == spec.orchestrator

    def test_cem_produces_a_valid_mutation(self, spec):
        cem = CEM(max_edits=1, seed=0)
        proposal = cem.propose(spec, [])
        assert len(proposal.changed_fields) >= 1
        assert proposal.spec.spec_sha256() != spec.spec_sha256()


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def _generation(gen: int, spec, parent=None, score: float = 0.5, changed=()) -> Generation:
    """A ledger record with the given changed field paths."""
    result = EvalResult(
        benchmark="x",
        subset="x_dev",
        seeds=[1, 2, 3],
        results=[
            TaskResult(task_id=f"t{i}", seed=1, passed=(i % 2 == 0), synthetic=True)
            for i in range(10)
        ],
        spec_sha256=spec.spec_sha256(),
        generation=gen,
    )
    from rsi.ledger.ledger import FieldChange

    record = Generation(gen=gen, spec=spec, parent=parent)
    record.evaluations["x_dev"] = result
    record.changed_fields = tuple(
        FieldChange(path=path, before=None, after=None, group="test") for path in changed
    )
    return record


class TestLedger:
    def test_save_and_load_round_trip(self, spec, tmp_path):
        ledger = Ledger(tmp_path)
        ledger.add(_generation(0, spec))
        ledger.save()
        reloaded = Ledger(tmp_path).load()
        assert reloaded.get(0).spec.spec_sha256() == spec.spec_sha256()

    def test_the_spec_is_archived_next_to_the_record(self, spec, tmp_path):
        ledger = Ledger(tmp_path)
        ledger.add(_generation(0, spec))
        ledger.save()
        assert (tmp_path / "specs" / "gen000.yaml").exists()

    def test_adding_the_same_generation_twice_is_rejected(self, spec, tmp_path):
        ledger = Ledger(tmp_path)
        ledger.add(_generation(0, spec))
        with pytest.raises(ValueError, match="already in the ledger"):
            ledger.add(_generation(0, spec))

    def test_the_lineage_is_reconstructible(self, spec, tmp_path):
        ledger = Ledger(tmp_path)
        ledger.add(_generation(0, spec))
        ledger.add(_generation(1, spec, parent=0))
        chain = ledger.chain(1)
        assert [g.gen for g in chain] == [0, 1]

    def test_observations_are_derived_from_the_ledger(self, spec, tmp_path):
        ledger = Ledger(tmp_path)
        ledger.add(_generation(0, spec))
        ledger.add(_generation(1, spec, parent=0, changed=("runtime.max_iterations",)))
        observations = ledger.observations()
        assert len(observations) == 2
        assert observations[1].parent == 0


class TestDiff:
    def test_the_diff_names_the_changed_field(self, spec):
        from rsi.schema import set_path

        other = set_path(spec.to_dict(), "agents[0].contract.max_tokens", 1200)
        from rsi.schema import HarnessSpec

        changes = diff_specs(spec, HarnessSpec.from_dict(other))
        assert len(changes) == 1
        assert changes[0].path == "agents[0].contract.max_tokens"
        assert changes[0].before == 800
        assert changes[0].after == 1200
        assert changes[0].group == "contract"

    def test_identical_specs_have_an_empty_diff(self, spec):
        assert diff_specs(spec, spec) == []

    def test_similarity_is_one_for_identical_specs(self, spec):
        assert spec_similarity(spec, spec) == pytest.approx(1.0, abs=1e-6)

    def test_similarity_drops_for_different_specs(self, spec):
        from rsi.schema import HarnessSpec, set_path

        other = HarnessSpec.from_dict(
            set_path(spec.to_dict(), "runtime.max_iterations", 99)
        )
        assert spec_similarity(spec, other) < 1.0

    def test_similarity_is_sensitive_to_list_fields(self, spec):
        """Regression: a list field must move the similarity.

        The contract's ``in_fields`` / ``out_fields`` are lists, and they are
        exactly the fields the task-specificity hypothesis is about -- an
        encoding that ignores list changes would make the homogenization
        curve blind to the thing it is supposed to measure.
        """
        from rsi.schema import HarnessSpec, set_path

        other = HarnessSpec.from_dict(
            set_path(
                spec.to_dict(),
                "agents[0].contract.out_fields",
                ("plan", "summary", "evidence"),
            )
        )
        assert spec_similarity(spec, other) < 1.0
        assert diff_specs(spec, other)

    def test_similarity_is_sensitive_to_numeric_fields(self, spec):
        """A scalar change must move the similarity too."""
        from rsi.schema import HarnessSpec, set_path

        other = HarnessSpec.from_dict(
            set_path(spec.to_dict(), "agents[0].contract.max_tokens", 1200)
        )
        assert spec_similarity(spec, other) < 1.0

    def test_similarity_is_symmetric_and_bounded(self, spec):
        from rsi.schema import HarnessSpec, set_path

        other = HarnessSpec.from_dict(
            set_path(spec.to_dict(), "runtime.max_iterations", 99)
        )
        forward = spec_similarity(spec, other)
        backward = spec_similarity(other, spec)
        assert forward == pytest.approx(backward, abs=1e-9)
        assert 0.0 <= forward <= 1.0

    def test_the_effect_table_needs_single_field_generations(self, spec, tmp_path):
        ledger = Ledger(tmp_path)
        ledger.add(_generation(0, spec))
        ledger.add(
            _generation(1, spec, parent=0, changed=("runtime.max_iterations",))
        )
        rows = effect_by_field(ledger)
        assert rows
        assert rows[0]["path"] == "runtime.max_iterations"

    def test_the_homogenization_curve_needs_two_generations(self, spec, tmp_path):
        ledger = Ledger(tmp_path)
        ledger.add(_generation(0, spec))
        assert homogenization_curve(ledger) == []
        ledger.add(_generation(1, spec, parent=0))
        curve = homogenization_curve(ledger)
        assert len(curve) == 1
        assert curve[0]["from_gen"] == 0

    def test_the_homogenization_curve_follows_the_lineage(self, spec, tmp_path):
        """Parallel chains: two consecutive numbers are two *siblings*.

        The curve must compare every generation with its own parent, not with
        the generation that happens to have the next number.
        """
        ledger = Ledger(tmp_path)
        ledger.add(_generation(0, spec))
        ledger.add(_generation(1, spec, parent=0))
        ledger.add(_generation(2, spec, parent=0))  # a sibling, not a child
        curve = homogenization_curve(ledger)
        assert len(curve) == 2
        assert {(row["from_gen"], row["to_gen"]) for row in curve} == {(0, 1), (0, 2)}
        for row in curve:
            assert row["from_gen"] == 0

    def test_the_homogenization_curve_can_resolve_an_unevaluated_parent(
        self, spec, tmp_path
    ):
        """The baseline is often not evaluated on its own.

        Without a resolver the curve would be empty and the report would say
        "need at least two generations" even though four chains exist.
        """
        ledger = Ledger(tmp_path)
        ledger.add(_generation(1, spec, parent=0))
        assert homogenization_curve(ledger) == []
        curve = homogenization_curve(ledger, spec_resolver=lambda gen: spec)
        assert len(curve) == 1
        assert curve[0]["from_gen"] == 0
        assert curve[0]["to_gen"] == 1

    def test_the_report_flags_synthetic_data(self, spec, tmp_path):
        """A report built from the test double must say so, loudly."""
        from rsi.ledger.diff import build_report

        ledger = Ledger(tmp_path)
        ledger.add(_generation(0, spec))
        report = build_report(ledger)
        assert "SYNTHETIC DATA" in report

    def test_the_report_has_no_synthetic_banner_for_real_data(self, spec, tmp_path):
        from rsi.evaluation.runner import EvalResult, TaskResult
        from rsi.ledger.diff import build_report

        ledger = Ledger(tmp_path)
        record = _generation(0, spec)
        record.evaluations["x_dev"] = EvalResult(
            benchmark="x",
            subset="x_dev",
            seeds=[1],
            results=[TaskResult(task_id="t0", seed=1, passed=True, synthetic=False)],
            spec_sha256=spec.spec_sha256(),
            generation=0,
        )
        ledger.add(record)
        report = build_report(ledger)
        assert "SYNTHETIC DATA" not in report


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #


class TestRunners:
    def test_the_synthetic_backend_is_deterministic(self, spec, tmp_path):
        subset = Subset(name="x_dev", split="dev", benchmark="x", task_ids=("a", "b", "c"))
        backend = SyntheticBackend()
        first = backend.run(spec, subset, [1, 2], tmp_path / "a")
        second = backend.run(spec, subset, [1, 2], tmp_path / "b")
        assert first.per_task_scores() == second.per_task_scores()
        assert first.synthetic is True

    def test_the_synthetic_backend_scores_every_run(self, spec, tmp_path):
        subset = Subset(name="x_dev", split="dev", benchmark="x", task_ids=("a", "b", "c"))
        result = SyntheticBackend().run(spec, subset, [1, 2], tmp_path)
        assert result.n_runs == 6
        assert result.n_tasks == 3
        assert 0.0 <= result.mean_score <= 1.0

    def test_a_different_spec_scores_differently(self, spec, tmp_path):
        from rsi.schema import HarnessSpec, set_path

        subset = Subset(name="x_dev", split="dev", benchmark="x", task_ids=tuple(f"t{i}" for i in range(40)))
        other = HarnessSpec.from_dict(set_path(spec.to_dict(), "runtime.max_iterations", 99))
        a = SyntheticBackend().run(spec, subset, [1], tmp_path / "a")
        b = SyntheticBackend().run(other, subset, [1], tmp_path / "b")
        assert a.mean_score != b.mean_score

    def test_the_job_file_only_changes_the_agent_config(self):
        base = {
            "jobs_dir": "runs",
            "n_concurrent_trials": 4,
            "agents": [{"name": "mini-swe-agent", "kwargs": {"config": {"a": 1}}}],
            "datasets": [{"name": "terminal-bench", "version": "2.0"}],
        }
        rendered = render_harbor_job(base, {"a": 2})
        assert verify_job_invariance(base, rendered) == []
        assert rendered["agents"][0]["kwargs"]["config"] == {"a": 2}
        assert rendered["datasets"] == base["datasets"]

    def test_a_drifting_job_file_is_detected(self):
        base = {
            "agents": [{"name": "x", "kwargs": {}}],
            "datasets": [{"name": "a"}],
        }
        rendered = render_harbor_job(base, {"a": 1})
        rendered["datasets"] = [{"name": "b"}]
        assert verify_job_invariance(base, rendered) == ["datasets"]
