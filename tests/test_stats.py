"""Statistics: bootstrap, McNemar, the acceptance rule, the noise floor."""

from __future__ import annotations

import numpy as np
import pytest

from rsi.evaluation.stats import (
    accept_generation,
    holm_bonferroni,
    mcnemar,
    mcnemar_from_pairs,
    paired_bootstrap,
    paired_delta,
    paired_delta_binary,
    power_check,
    score_from_seeds,
)


class TestBootstrap:
    def test_a_constant_difference_has_a_degenerate_ci(self):
        result = paired_bootstrap(np.ones(50), n_boot=2000, seed=0)
        assert result["mean"] == pytest.approx(1.0)
        assert result["lo"] == pytest.approx(1.0)
        assert result["hi"] == pytest.approx(1.0)

    def test_a_zero_difference_is_not_significant(self):
        result = paired_bootstrap(np.zeros(50), n_boot=2000, seed=0)
        assert result["lo"] <= 0 <= result["hi"]

    def test_the_bootstrap_is_reproducible(self):
        diff = np.random.default_rng(0).normal(0.1, 0.3, 200)
        a = paired_bootstrap(diff, n_boot=2000, seed=3)
        b = paired_bootstrap(diff, n_boot=2000, seed=3)
        assert a == b

    def test_an_empty_vector_is_rejected(self):
        with pytest.raises(ValueError):
            paired_bootstrap(np.array([]))


class TestMcNemar:
    def test_no_discordant_pairs_gives_p_one(self):
        assert mcnemar(0, 0) == 1.0

    def test_a_clear_imbalance_is_significant(self):
        assert mcnemar(30, 5) < 0.01

    def test_a_small_imbalance_is_not(self):
        assert mcnemar(8, 5) > 0.05

    def test_the_test_is_symmetric(self):
        assert mcnemar(30, 5) == pytest.approx(mcnemar(5, 30))

    def test_counting_the_discordant_pairs(self):
        new = np.array([1, 1, 0, 0, 1, 0])
        old = np.array([0, 1, 1, 0, 0, 0])
        b, c = mcnemar_from_pairs(new, old)
        assert (b, c) == (2, 1)


class TestAcceptance:
    def _result(self, delta, b, c, n=100):
        from rsi.evaluation.stats import PairedResult

        return PairedResult(
            n=n,
            mean_delta=delta,
            se=0.01,
            ci_lo=delta - 0.02,
            ci_hi=delta + 0.02,
            b=b,
            c=c,
            mcnemar_p=mcnemar(b, c),
            n_boot=1000,
        )

    def test_a_clear_improvement_is_accepted(self):
        ok, reason = accept_generation(self._result(0.05, 20, 5))
        assert ok
        assert "ACCEPT" in reason

    def test_a_null_result_is_rejected(self):
        ok, _ = accept_generation(self._result(0.0, 5, 5))
        assert not ok

    def test_the_stricter_of_the_two_criteria_wins(self):
        # A positive CI but a p-value above the threshold -> rejected.
        from rsi.evaluation.stats import PairedResult

        result = PairedResult(
            n=100,
            mean_delta=0.05,
            se=0.01,
            ci_lo=0.01,
            ci_hi=0.09,
            b=8,
            c=6,
            mcnemar_p=0.6,
            n_boot=1000,
        )
        ok, _ = accept_generation(result)
        assert not ok


class TestScores:
    def test_the_multi_seed_score_is_continuous(self):
        assert score_from_seeds([True, False, False]) == pytest.approx(1 / 3)
        assert score_from_seeds([True, True, True]) == 1.0
        assert score_from_seeds([False, False]) == 0.0

    def test_an_empty_seed_list_is_rejected(self):
        with pytest.raises(ValueError):
            score_from_seeds([])

    def test_paired_delta_uses_the_common_tasks(self):
        new = {"a": 1.0, "b": 0.5, "c": 0.0}
        old = {"a": 0.0, "b": 0.5, "c": 0.0, "d": 1.0}
        result = paired_delta(new, old, n_boot=1000, seed=0)
        assert result.n == 3
        assert result.mean_delta == pytest.approx(1 / 3)

    def test_disjoint_task_sets_are_rejected(self):
        with pytest.raises(ValueError):
            paired_delta({"a": 1.0}, {"b": 1.0})


class TestPairedDeltaBinary:
    """The paired test must run on binary (task, seed) outcomes.

    McNemar's test is defined on *discordant binary pairs*.  Feeding it the
    aggregated per-task score (the fraction of seeds that passed) makes
    ``new > 0`` true for a task that went from 0/3 to 1/3, so a partial
    improvement is counted as a full win against a partial regression.
    """

    def test_a_partial_improvement_is_not_a_discordant_win(self):
        # task "a": old failed all three seeds, new passed one of three.
        # task "b": old passed all three, new passed two of three.
        new = {"a": {1: False, 2: False, 3: True}, "b": {1: True, 2: True, 3: False}}
        old = {"a": {1: False, 2: False, 3: False}, "b": {1: True, 2: True, 3: True}}
        result = paired_delta_binary(new, old, n_boot=1000, seed=0)
        # Six (task, seed) pairs; exactly one is discordant in each direction.
        assert result.n == 6
        assert result.b == 1
        assert result.c == 1

    def test_the_aggregated_comparison_would_have_miscounted(self):
        """The same data through the continuous path gives a different b/c.

        Task "a" went 0/3 -> 1/3 and task "b" went 3/3 -> 2/3.  Neither is a
        discordant *run* pair: in "a" the changed seed is a win and the two
        unchanged seeds are both failures; in "b" the changed seed is a loss
        and the two unchanged seeds are both passes.  The binary path finds
        exactly one win and one loss; the aggregated path sees "a" as a
        win (0.333 > 0) and "b" as a win as well (0.667 > 0), so it misses
        the loss entirely.
        """
        new_scores = {"a": 1 / 3, "b": 2 / 3}
        old_scores = {"a": 0.0, "b": 1.0}
        b_agg, c_agg = mcnemar_from_pairs(
            np.array([new_scores["a"], new_scores["b"]]),
            np.array([old_scores["a"], old_scores["b"]]),
        )
        assert (b_agg, c_agg) == (1, 0)  # wrong: the loss is invisible
        new = {"a": {1: False, 2: False, 3: True}, "b": {1: True, 2: True, 3: False}}
        old = {"a": {1: False, 2: False, 3: False}, "b": {1: True, 2: True, 3: True}}
        result = paired_delta_binary(new, old, n_boot=1000, seed=0)
        assert (result.b, result.c) == (1, 1)  # right

    def test_identical_generations_have_no_discordant_pairs(self):
        runs = {"a": {1: True, 2: False}, "b": {1: True, 2: True}}
        result = paired_delta_binary(runs, runs, n_boot=1000, seed=0)
        assert result.b == 0
        assert result.c == 0
        assert result.mcnemar_p == 1.0
        assert result.mean_delta == pytest.approx(0.0)

    def test_only_shared_seeds_are_compared(self):
        new = {"a": {1: True, 2: True}}
        old = {"a": {1: False}}
        result = paired_delta_binary(new, old, n_boot=1000, seed=0)
        assert result.n == 1
        assert result.b == 1
        assert result.c == 0

    def test_disjoint_task_sets_are_rejected(self):
        with pytest.raises(ValueError):
            paired_delta_binary({"a": {1: True}}, {"b": {1: True}})


class TestMultiplicity:
    def test_holm_adjustment_is_monotone(self):
        adjusted = holm_bonferroni([0.01, 0.02, 0.04])
        assert adjusted[0] <= adjusted[1] <= adjusted[2]
        assert all(p <= 1.0 for p in adjusted)

    def test_holm_on_an_empty_list(self):
        assert holm_bonferroni([]) == []


class TestPower:
    def test_more_seeds_reduce_the_resolution(self):
        few = power_check(300, 1)
        many = power_check(300, 5)
        assert many["se"] < few["se"]
        assert many["detectable"] < few["detectable"]
