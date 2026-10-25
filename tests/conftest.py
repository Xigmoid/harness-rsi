"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsi.schema import (  # noqa: E402
    AgentSpec,
    AuxiliaryRules,
    Contract,
    ContractFormat,
    GoalStyle,
    HarnessSpec,
    Hop,
    InstructionSlots,
    Runtime,
    VerificationStyle,
)


def make_spec(**overrides) -> HarnessSpec:
    """A small but valid harness: one orchestrator, two subagents."""
    kwargs = dict(
        version=0,
        agents=(
            AgentSpec(
                id="orchestrator",
                role="coordinator",
                instruction_template="standard",
                instruction_slots=InstructionSlots(
                    goal=GoalStyle.PRIMARY,
                    required_outputs=("summary",),
                    forbidden_outputs=("raw_logs",),
                    verification=VerificationStyle.SELF_CHECK,
                ),
                contract=Contract(
                    out_fields=("plan", "summary"),
                    in_fields=("task", "findings", "evidence", "patch"),
                    format=ContractFormat.JSON,
                    max_tokens=800,
                    free_text=True,
                ),
                model="openai/glm5.3-flash",
            ),
            AgentSpec(
                id="investigator",
                role="analyst",
                instruction_template="standard",
                instruction_slots=InstructionSlots(
                    goal=GoalStyle.THOROUGH,
                    required_outputs=("findings", "evidence"),
                ),
                contract=Contract(
                    out_fields=("findings", "evidence"),
                    in_fields=("task", "plan"),
                    format=ContractFormat.JSON,
                    max_tokens=800,
                ),
                model="openai/glm5.3-flash",
            ),
            AgentSpec(
                id="implementer",
                role="implementer",
                instruction_template="standard",
                instruction_slots=InstructionSlots(
                    goal=GoalStyle.PRIMARY,
                    required_outputs=("patch",),
                ),
                contract=Contract(
                    out_fields=("patch",),
                    in_fields=("task", "findings"),
                    format=ContractFormat.JSON,
                    max_tokens=800,
                ),
                model="openai/glm5.3-flash",
            ),
        ),
        workflow=(
            Hop(id="h1", source="orchestrator", target="investigator", pass_fields=("task", "plan")),
            Hop(id="h2", source="orchestrator", target="implementer", pass_fields=("task", "findings")),
            Hop(id="h3", source="investigator", target="orchestrator", pass_fields=("findings", "evidence")),
            Hop(id="h4", source="implementer", target="orchestrator", pass_fields=("patch",)),
        ),
        auxiliary_rules=AuxiliaryRules(
            acceptance_gates=("the change is applied",),
            failure_fallback=("retry_once", "degrade_to_free_text"),
        ),
        runtime=Runtime(
            max_iterations=20,
            observation_max_chars=10000,
            context_keep_recent=10,
            seed=1234,
            temperature=0.0,
            max_hops=4,
            hop_context_max_chars=3000,
        ),
    )
    kwargs.update(overrides)
    return HarnessSpec(**kwargs)


@pytest.fixture()
def spec() -> HarnessSpec:
    return make_spec()
