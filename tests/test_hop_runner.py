"""The hop runner: scheduling, contract rendering, payload extraction."""

from __future__ import annotations

import json

import pytest
from conftest import make_spec

from rsi.hop_runner import HopRunner, PayloadError, RunState
from rsi.schema import ContractFormat, HarnessSpec


@pytest.fixture()
def runner(spec):
    return HopRunner(spec)


@pytest.fixture()
def state(spec):
    return RunState(task="fix the bug", current_agent=spec.orchestrator)


class TestScheduling:
    def test_orchestrator_is_identified(self, spec):
        assert spec.orchestrator == "orchestrator"

    def test_next_hop_from_the_orchestrator(self, runner, state):
        hop = runner.next_hop(state)
        assert hop is not None
        assert hop.source == "orchestrator"
        assert hop.target in {"investigator", "implementer"}

    def test_a_delegation_wins_over_the_schedule(self, runner, state):
        state.pending_delegate = "implementer"
        hop = runner.next_hop(state)
        assert hop is not None
        assert hop.target == "implementer"

    def test_an_unknown_delegation_falls_back(self, runner, state):
        state.pending_delegate = "ghost"
        hop = runner.next_hop(state)
        assert hop is not None
        assert hop.source == "orchestrator"

    def test_control_returns_to_the_orchestrator(self, runner, spec):
        state = RunState(task="t", current_agent="investigator")
        hop = runner.transfer_target("investigator", state)
        assert hop is not None
        assert hop.target == "orchestrator"

    def test_a_subagent_without_an_outgoing_edge_has_no_target(self, runner, spec):
        state = RunState(task="t", current_agent="investigator")
        assert runner.transfer_target("ghost", state) is None


class TestRendering:
    def test_the_orchestrator_sees_the_whole_workflow(self, runner):
        text = runner.render_system("orchestrator")
        assert "investigator" in text
        assert "implementer" in text
        assert "h1" in text

    def test_a_subagent_only_sees_its_own_in_fields(self, runner, spec):
        text = runner.render_system("implementer")
        assert "findings" in text
        # The investigator's raw fields are not part of the implementer's view
        assert "### Available agents" not in text

    def test_the_context_is_filtered_by_the_contract(self, runner, spec, state):
        # The investigator produced findings and evidence ...
        state.payloads["investigator"] = {"findings": "f", "evidence": "e", "secret": "s"}
        hop = spec.edge("investigator", "orchestrator")
        state.cycle_payloads = dict(state.payloads)
        # ... the orchestrator declares task/findings/evidence as in_fields.
        context = runner.assemble_context("orchestrator", state, hop)
        assert "findings" in context
        assert "secret" not in context

    def test_the_context_respects_the_hop_limit(self, runner, spec, state):
        # The investigator produced more than the hop carries.
        state.payloads["investigator"] = {"findings": "f", "evidence": "e"}
        hop = spec.edge("investigator", "orchestrator")
        context = runner.assemble_context("orchestrator", state, hop)
        assert "findings" in context
        assert "evidence" in context  # both are on h3
        # "patch" is in the orchestrator's in_fields but not on this hop, so it
        # must not travel.
        state.payloads["implementer"] = {"patch": "p"}
        context = runner.assemble_context("orchestrator", state, hop)
        assert "patch" not in context

    def test_the_context_is_truncated(self, spec):
        small = HarnessSpec.from_dict({**spec.to_dict()})
        from rsi.schema import Runtime

        small = HarnessSpec.from_dict(
            {**small.to_dict(), "runtime": {**small.to_dict()["runtime"], "hop_context_max_chars": 300}}
        )
        runner = HopRunner(small)
        state = RunState(task="x" * 1000, current_agent="orchestrator")
        context = runner.assemble_context("orchestrator", state)
        assert len(context) <= 300

    def test_the_delegation_rule_is_stated(self, spec):
        runner = HopRunner(spec)
        text = runner.render_system("orchestrator")
        assert "<delegate>" in text
        assert "MUST delegate" in text


class TestPayloadExtraction:
    def test_json_payload(self, runner):
        raw = 'some text\n<handoff>\n{"findings": "f", "evidence": "e"}\n</handoff>'
        payload = runner.extract_payload("investigator", raw)
        assert payload == {"findings": "f", "evidence": "e"}

    def test_fenced_json_payload(self, runner):
        raw = '<handoff>\n```json\n{"findings": "f", "evidence": "e"}\n```\n</handoff>'
        payload = runner.extract_payload("investigator", raw)
        assert payload["findings"] == "f"

    def test_a_missing_block_raises(self, runner):
        with pytest.raises(PayloadError) as exc:
            runner.extract_payload("investigator", "no block here")
        assert exc.value.reason == "missing_block"

    def test_invalid_json_raises(self, runner):
        with pytest.raises(PayloadError) as exc:
            runner.extract_payload("investigator", "<handoff>{not json}</handoff>")
        assert exc.value.reason == "invalid_json"

    def test_missing_fields_raise(self, runner):
        with pytest.raises(PayloadError) as exc:
            runner.extract_payload("investigator", '<handoff>{"findings": "f"}</handoff>')
        assert exc.value.reason == "missing_fields"
        assert "evidence" in exc.value.missing

    def test_extra_fields_are_dropped(self, runner):
        raw = '<handoff>{"findings": "f", "evidence": "e", "extra": 1}</handoff>'
        payload = runner.extract_payload("investigator", raw)
        assert "extra" not in payload

    def test_markdown_payload(self, spec):
        from rsi.schema import set_path

        data = set_path(spec.to_dict(), "agents[1].contract.format", "markdown")
        md_spec = HarnessSpec.from_dict(data)
        runner = HopRunner(md_spec)
        payload = runner.extract_payload(
            "investigator", "<handoff>\nfindings: the bug is in foo.py\nevidence: traceback\n</handoff>"
        )
        assert payload["findings"] == "the bug is in foo.py"

    def test_degradation_keeps_the_raw_text(self, runner):
        payload = runner.degrade_to_free_text("investigator", "raw prose")
        assert payload["findings"] == "raw prose"
        assert payload["free_text"] == "raw prose"


class TestDelegation:
    def test_parse_delegate(self):
        assert HopRunner.parse_delegate("<delegate>investigator</delegate>") == "investigator"
        assert HopRunner.parse_delegate("<delegate> investigator </delegate>") == "investigator"

    def test_no_delegate(self):
        assert HopRunner.parse_delegate("just text") is None
        assert HopRunner.parse_delegate("<handoff>{}</handoff>") is None
