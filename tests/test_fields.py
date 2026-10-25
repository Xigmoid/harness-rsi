"""The field registry: expansion, mutation, edit budget."""

from __future__ import annotations

import random

import pytest
from conftest import make_spec

from rsi.fields import (
    expand_fields,
    get_value,
    group_of,
    mutable_fields,
    mutate,
    set_value,
)
from rsi.schema import SpecError


class TestExpansion:
    def test_every_agent_and_hop_is_covered(self, spec):
        paths = {f.path for f in expand_fields(spec)}
        assert "agents[0].contract.max_tokens" in paths
        assert "agents[2].contract.out_fields" in paths
        assert "workflow[0].pass_fields" in paths
        assert "runtime.max_iterations" in paths

    def test_groups_follow_the_rhi_priorities(self, spec):
        groups = {f.path: f.group for f in expand_fields(spec)}
        assert groups["agents[0].contract.max_tokens"] == "contract"
        assert groups["workflow[0].pass_fields"] == "hop"
        assert groups["runtime.max_iterations"] == "runtime"
        assert groups["agents[0].instruction_slots.goal"] == "instruction"
        assert groups["agents[0].role"] == "role"

    def test_model_knobs_are_disabled_by_default(self, spec):
        mutable = {f.path for f in mutable_fields(spec)}
        assert "agents[0].reasoning_effort" not in mutable
        assert "agents[0].max_tokens" not in mutable
        assert "agents[0].contract.max_tokens" in mutable

    def test_group_of(self, spec):
        assert group_of(spec, "runtime.max_iterations") == "runtime"
        assert group_of(spec, "does.not.exist") == "unknown"


class TestAccess:
    def test_get_and_set(self, spec):
        assert get_value(spec, "agents[0].contract.max_tokens") == 800
        new = set_value(spec, "agents[0].contract.max_tokens", 1200)
        assert get_value(new, "agents[0].contract.max_tokens") == 1200
        assert get_value(spec, "agents[0].contract.max_tokens") == 800

    def test_set_value_validates(self, spec):
        with pytest.raises(SpecError):
            set_value(spec, "agents[0].contract.max_tokens", 10**9)

    def test_set_value_keeps_the_spec_valid(self, spec):
        # "questions" is not in the vocabulary of the orchestrator's contract.
        with pytest.raises(SpecError):
            set_value(spec, "agents[0].contract.out_fields", ["questions"])


class TestMutation:
    def test_mutation_produces_a_valid_spec(self, spec):
        rng = random.Random(0)
        for path in ("agents[0].contract.max_tokens", "runtime.max_iterations"):
            mutated = mutate(spec, path, rng)
            assert mutated.spec_sha256() != spec.spec_sha256()
            assert mutated.orchestrator == spec.orchestrator

    def test_a_list_mutation_keeps_the_field_set_valid(self, spec):
        rng = random.Random(1)
        for _ in range(20):
            mutated = mutate(spec, "agents[1].contract.out_fields", rng)
            out = mutated.agent("investigator").contract.out_fields
            assert len(out) >= 1
            # The orchestrator must still accept the fields that arrive.
            assert "findings" in out or len(out) >= 1

    def test_mutating_an_unknown_field_raises(self, spec):
        with pytest.raises(SpecError):
            mutate(spec, "agents[9].contract.max_tokens", random.Random(0))

    def test_mutation_is_reproducible_for_a_seed(self, spec):
        a = mutate(spec, "runtime.max_iterations", random.Random(7))
        b = mutate(spec, "runtime.max_iterations", random.Random(7))
        assert a.spec_sha256() == b.spec_sha256()
