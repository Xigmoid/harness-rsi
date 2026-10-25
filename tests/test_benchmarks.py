"""The multi-benchmark layer: registry, suites, transfer matrix."""

from __future__ import annotations

import pytest

from rsi.benchmarks import (
    BENCHMARKS,
    SUITES,
    benchmarks_in_suite,
    get_benchmark,
    suite_benchmarks,
    suite_capabilities,
)
from rsi.evaluation.suite import composite_score, run_suite
from rsi.evaluation.runner import SyntheticBackend
from rsi.evaluation.subsets import Subset
from rsi.ledger.diff import transfer_matrix
from rsi.ledger.ledger import Generation, Ledger


class TestRegistry:
    def test_every_capability_has_benchmarks(self):
        capabilities = {b.capability for b in BENCHMARKS.values()}
        assert {"code", "math", "reasoning", "instruction_following"} <= capabilities

    def test_the_suite_keys_are_resolvable(self):
        for suite, keys in SUITES.items():
            for key in keys:
                assert key in BENCHMARKS, f"suite {suite} references unknown benchmark {key}"

    def test_the_swe_bench_split_matches_the_protocol(self):
        # 500 tasks, dev ratio 300/500 -> exactly 300/200.
        spec = get_benchmark("sbv")
        assert spec.n_tasks == 500
        assert round(spec.n_tasks * spec.dev_ratio) == 300

    def test_the_terminal_bench_note_records_the_discrepancy(self):
        # The plan assumed 2.1/445; Harbor's registry carries 2.0/89.
        spec = get_benchmark("tb21")
        assert spec.dataset == "terminal-bench"
        assert "89" in spec.notes or spec.n_tasks == 89

    def test_a_path_benchmark_needs_a_path(self):
        with pytest.raises(ValueError, match="needs a path"):
            from rsi.benchmarks import BenchmarkSpec

            BenchmarkSpec(
                key="x",
                capability="math",
                dataset="x",
                version="1",
                dev_ratio=0.7,
                source="path",
            )

    def test_the_dataset_ref_is_harbor_shaped(self):
        assert get_benchmark("sbv").dataset_ref == {
            "name": "swebench-verified",
            "version": "1.0",
        }
        assert get_benchmark("ifeval").dataset_ref == {"path": "datasets/ifeval"}

    def test_suite_capabilities(self):
        assert suite_capabilities("math") == ("math",)
        assert set(suite_capabilities("all")) >= {"code", "math", "reasoning"}

    def test_a_single_benchmark_is_a_suite_of_one(self):
        assert benchmarks_in_suite("aime") == ("aime",)
        assert len(suite_benchmarks("aime")) == 1

    def test_an_unknown_suite_raises(self):
        with pytest.raises(KeyError):
            suite_benchmarks("nope")


class TestComposite:
    def test_the_composite_is_the_unweighted_mean(self):
        assert composite_score({"a": 1.0, "b": 0.0}) == pytest.approx(0.5)

    def test_the_weights_are_applied(self):
        assert composite_score({"a": 1.0, "b": 0.0}, weights={"a": 3.0, "b": 1.0}) == pytest.approx(0.75)

    def test_an_empty_suite_scores_zero(self):
        assert composite_score({}) == 0.0


class TestSuiteRun:
    def test_a_suite_run_records_every_benchmark(self, spec, tmp_path):
        subsets = {
            "aime": Subset(name="aime_dev", split="dev", benchmark="aime", task_ids=("a", "b")),
            "ineqmath": Subset(
                name="ineqmath_dev", split="dev", benchmark="ineqmath", task_ids=("c",)
            ),
        }
        result = run_suite(
            spec, "math", subsets, [1, 2], SyntheticBackend(), tmp_path, generation=0
        )
        assert set(result.benchmark_results) == {"aime", "ineqmath"}
        assert result.composite == pytest.approx(
            composite_score(result.per_benchmark_score)
        )
        assert result.n_runs == 3 * 2

    def test_a_missing_subset_is_an_error(self, spec, tmp_path):
        with pytest.raises(KeyError, match="no subset"):
            run_suite(spec, "math", {}, [1], SyntheticBackend(), tmp_path)

    def test_the_capability_aggregation(self, spec, tmp_path):
        subsets = {
            "aime": Subset(name="aime_dev", split="dev", benchmark="aime", task_ids=("a",)),
            "ineqmath": Subset(
                name="ineqmath_dev", split="dev", benchmark="ineqmath", task_ids=("b",)
            ),
        }
        result = run_suite(
            spec, "math", subsets, [1], SyntheticBackend(), tmp_path, generation=0
        )
        per_capability = result.per_capability({"aime": "math", "ineqmath": "math"})
        assert set(per_capability) == {"math"}


class TestTransferMatrix:
    def _ledger(self, spec, tmp_path):
        from rsi.evaluation.runner import EvalResult, TaskResult

        ledger = Ledger(tmp_path)
        for gen, objective in ((0, "math"), (1, "math"), (2, "reasoning")):
            record = Generation(gen=gen, spec=spec, parent=None)
            record.spec = spec
            for key in ("aime_dev", "gpqa_dev"):
                record.evaluations[key] = EvalResult(
                    benchmark=key.split("_")[0],
                    subset=key,
                    seeds=[1],
                    results=[TaskResult(task_id="t", seed=1, passed=True, synthetic=True)],
                )
            record.spec = type(spec).from_dict(
                {**spec.to_dict(), "meta": {"objective": objective}}
            )
            ledger.add(record)
        return ledger

    def test_the_matrix_has_a_row_per_generation(self, spec, tmp_path):
        ledger = self._ledger(spec, tmp_path)
        matrix = transfer_matrix(ledger)
        assert len(matrix["rows"]) == 3
        assert "aime" in matrix["benchmarks"]
        assert "gpqa" in matrix["benchmarks"]

    def test_the_objective_is_recorded_per_row(self, spec, tmp_path):
        ledger = self._ledger(spec, tmp_path)
        matrix = transfer_matrix(ledger)
        assert matrix["rows"][0]["objective"] == "math"
        assert matrix["rows"][2]["objective"] == "reasoning"

    def test_the_cells_hold_the_scores(self, spec, tmp_path):
        ledger = self._ledger(spec, tmp_path)
        matrix = transfer_matrix(ledger)
        assert matrix["rows"][0]["scores"]["aime"] == 1.0

    def test_the_ledger_objective_uses_the_composite(self, spec, tmp_path):
        ledger = self._ledger(spec, tmp_path)
        record = ledger.get(0)
        score = ledger.objective_score(record, "math")
        assert score == pytest.approx(1.0)
        assert ledger.objective_score(record, None) == pytest.approx(1.0)
