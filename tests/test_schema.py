"""The typed spec: validation, serialization and hashing."""

from __future__ import annotations

import pytest
from conftest import make_spec

from rsi.schema import (
    AgentSpec,
    Contract,
    HarnessSpec,
    Hop,
    SpecError,
    flatten_spec,
    get_path,
    load_spec,
    save_spec,
    set_path,
)


class TestValidation:
    def test_duplicate_agent_ids_are_rejected(self):
        spec = make_spec()
        with pytest.raises(SpecError, match="duplicate agent ids"):
            HarnessSpec(
                version=0,
                agents=(spec.agents[0], spec.agents[0], spec.agents[2]),
                workflow=spec.workflow,
                runtime=spec.runtime,
            )

    def test_unknown_hop_endpoint_is_rejected(self):
        spec = make_spec()
        workflow = (*spec.workflow[:-1], Hop(id="h9", source="investigator", target="ghost"))
        with pytest.raises(SpecError, match="unknown target agent"):
            HarnessSpec(
                version=0, agents=spec.agents, workflow=workflow, runtime=spec.runtime
            )

    def test_pass_fields_must_be_produced_by_the_source(self):
        spec = make_spec()
        workflow = (
            Hop(
                id="h1",
                source="orchestrator",
                target="investigator",
                pass_fields=("nonexistent_field",),
            ),
            *spec.workflow[1:],
        )
        with pytest.raises(SpecError, match="not produced by"):
            HarnessSpec(
                version=0, agents=spec.agents, workflow=workflow, runtime=spec.runtime
            )

    def test_pass_fields_must_be_accepted_by_the_target(self):
        spec = make_spec()
        # The implementer produces "patch", but the investigator only accepts
        # task and plan.
        workflow = (
            *spec.workflow[:3],
            Hop(
                id="h4",
                source="implementer",
                target="investigator",
                pass_fields=("patch",),
            ),
        )
        with pytest.raises(SpecError, match="does not declare in_fields"):
            HarnessSpec(
                version=0, agents=spec.agents, workflow=workflow, runtime=spec.runtime
            )

    def test_exactly_one_orchestrator_is_required(self):
        spec = make_spec()
        with pytest.raises(SpecError, match="exactly one orchestrator"):
            HarnessSpec(
                version=0,
                agents=spec.agents,
                workflow=spec.workflow[:1],
                runtime=spec.runtime,
            )

    def test_orchestrator_must_receive_the_task(self):
        spec = make_spec()
        agents = list(spec.agents)
        orchestrator = agents[0]
        contract = orchestrator.contract
        agents[0] = AgentSpec(
            id=orchestrator.id,
            role=orchestrator.role,
            instruction_template=orchestrator.instruction_template,
            instruction_slots=orchestrator.instruction_slots,
            contract=Contract(
                out_fields=contract.out_fields,
                in_fields=tuple(f for f in contract.in_fields if f != "task"),
                format=contract.format,
                max_tokens=contract.max_tokens,
                free_text=contract.free_text,
            ),
            model=orchestrator.model,
        )
        # The hops must not carry "task" either, otherwise the hop rule fires
        # first and the test would not exercise the rule it is named after.
        workflow = (
            Hop(id="h1", source="orchestrator", target="investigator", pass_fields=("plan",)),
            *spec.workflow[2:],
        )
        with pytest.raises(SpecError, match="must declare 'task'"):
            HarnessSpec(
                version=0, agents=tuple(agents), workflow=workflow, runtime=spec.runtime
            )

    def test_contract_fields_must_be_unique_and_well_formed(self):
        with pytest.raises(SpecError, match="duplicate contract field"):
            Contract(out_fields=("a", "a"), in_fields=("task",))
        with pytest.raises(SpecError, match="invalid contract field name"):
            Contract(out_fields=("not a field",), in_fields=("task",))

    def test_empty_out_fields_are_rejected(self):
        with pytest.raises(SpecError, match="must not be empty"):
            Contract(out_fields=(), in_fields=("task",))

    def test_runtime_bounds_are_enforced(self):
        with pytest.raises(SpecError, match="max_iterations out of range"):
            make_spec().runtime.__class__(max_iterations=0)
        with pytest.raises(SpecError, match="min_delegations out of range"):
            make_spec().runtime.__class__(min_delegations=99)


class TestSerialization:
    def test_round_trip_preserves_the_spec(self, spec):
        restored = HarnessSpec.from_dict(spec.to_dict())
        assert restored.to_dict() == spec.to_dict()
        assert restored.spec_sha256() == spec.spec_sha256()

    def test_yaml_round_trip(self, spec, tmp_path):
        path = save_spec(spec, tmp_path / "gen.yaml")
        restored = load_spec(path)
        assert restored.spec_sha256() == spec.spec_sha256()

    def test_hash_is_stable_under_key_order(self, spec):
        data = spec.to_dict()
        shuffled = {k: data[k] for k in reversed(list(data))}
        assert HarnessSpec.from_dict(shuffled).spec_sha256() == spec.spec_sha256()

    def test_hash_changes_when_a_field_changes(self, spec):
        other = set_path(spec.to_dict(), "agents[0].contract.max_tokens", 1200)
        assert HarnessSpec.from_dict(other).spec_sha256() != spec.spec_sha256()

    def test_enum_values_accept_plain_strings(self):
        spec = make_spec()
        data = spec.to_dict()
        data["agents"][0]["instruction_slots"]["goal"] = "thorough"
        data["agents"][0]["contract"]["format"] = "markdown"
        restored = HarnessSpec.from_dict(data)
        assert restored.agents[0].instruction_slots.goal.value == "thorough"
        assert restored.agents[0].contract.format.value == "markdown"

    def test_flatten_covers_every_searchable_field(self, spec):
        flat = flatten_spec(spec)
        assert "agents[0].contract.max_tokens" in flat
        assert "runtime.max_iterations" in flat
        assert "workflow[0].pass_fields" in flat
        assert flat["agents[1].contract.out_fields"] == ("findings", "evidence")


class TestPathAccess:
    def test_get_and_set(self, spec):
        assert get_path(spec.to_dict(), "agents[1].contract.max_tokens") == 800
        data = set_path(spec.to_dict(), "agents[1].contract.max_tokens", 1200)
        assert get_path(data, "agents[1].contract.max_tokens") == 1200
        assert get_path(spec.to_dict(), "agents[1].contract.max_tokens") == 800

    def test_invalid_path_raises(self, spec):
        with pytest.raises(SpecError):
            get_path(spec.to_dict(), "agents[99].contract.max_tokens")
        with pytest.raises(SpecError):
            get_path(spec.to_dict(), "agents[0]..bad")
