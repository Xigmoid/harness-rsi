"""Typed harness specification for the RHI framework.

The spec is the *only* object an optimizer is allowed to mutate.  Everything else
(agent behaviour, prompts, hop scheduling) is derived from it, so a generation
diff is always a diff of this structure.

Design rules enforced here:

* every dataclass is frozen, so a spec can be treated as an immutable value;
* validation is fail-fast and happens at construction time (C-4/C-5 of the plan);
* (de)serialization is hand-written instead of framework-generated, because
  ``to_dict``-style helpers silently drop fields (trap T-12);
* a canonical SHA256 is derived from the *semantic* content, so two specs that
  differ only in dict ordering hash identically.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

__all__ = [
    "SpecError",
    "ContractFormat",
    "FailurePolicy",
    "GoalStyle",
    "VerificationStyle",
    "Contract",
    "InstructionSlots",
    "AgentSpec",
    "Hop",
    "AuxiliaryRules",
    "Runtime",
    "HarnessSpec",
    "INSTRUCTION_TEMPLATES",
    "load_spec",
    "save_spec",
    "spec_sha256",
    "get_path",
    "set_path",
    "flatten_spec",
]


class SpecError(ValueError):
    """Raised when a spec violates a structural or semantic rule."""


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class ContractFormat(str, Enum):
    """Wire format used to hand a payload from a subagent to the orchestrator."""

    JSON = "json"
    MARKDOWN = "markdown"


class FailurePolicy(str, Enum):
    """What to do when a contract payload cannot be parsed (trap T-7)."""

    RETRY_ONCE = "retry_once"
    DEGRADE_TO_FREE_TEXT = "degrade_to_free_text"
    ABORT = "abort"


class GoalStyle(str, Enum):
    """Slot values for the goal phrasing of an instruction."""

    PRIMARY = "primary"
    THOROUGH = "thorough"
    MINIMAL = "minimal"


class VerificationStyle(str, Enum):
    """Slot values for how much self-verification an agent is asked to do."""

    SELF_CHECK = "self_check"
    PEER_REVIEW = "peer_review"
    NONE = "none"


# --------------------------------------------------------------------------- #
# Instruction templates
# --------------------------------------------------------------------------- #

#: Instruction templates are *not* free text.  Each is a fixed scaffold with
#: named slots; the optimizer only ever moves the slot values, so that every
#: edit has a well defined semantic boundary (plan section 4.2).
INSTRUCTION_TEMPLATES: dict[str, str] = {
    "standard": (
        "You are {role}.\n"
        "Goal: {goal}\n"
        "You must produce: {required_outputs}\n"
        "You must not produce: {forbidden_outputs}\n"
        "Verification: {verification}\n"
    ),
    "concise": (
        "You are {role}. Focus on {goal}.\n"
        "Deliver: {required_outputs}\n"
        "Avoid: {forbidden_outputs}\n"
        "Verify: {verification}\n"
    ),
    "evidence_first": (
        "You are {role}. {goal}.\n"
        "Every claim must be backed by evidence in the workspace.\n"
        "Required: {required_outputs}\n"
        "Forbidden: {forbidden_outputs}\n"
        "Verification: {verification}\n"
    ),
}

_GOAL_TEXT: dict[GoalStyle, str] = {
    GoalStyle.PRIMARY: "solve the task correctly and completely",
    GoalStyle.THOROUGH: "solve the task correctly, covering edge cases and failure modes",
    GoalStyle.MINIMAL: "solve the task with the smallest sufficient change",
}

_VERIFICATION_TEXT: dict[VerificationStyle, str] = {
    VerificationStyle.SELF_CHECK: "re-read your own output and check it against the goal",
    VerificationStyle.PEER_REVIEW: "state the checks another agent should run on your output",
    VerificationStyle.NONE: "do not add extra verification steps",
}

_FIELD_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def _as_enum(enum_cls, value, where: str):
    """Coerce a string into an enum member (frozen dataclasses need this)."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:
        allowed = [member.value for member in enum_cls]
        raise SpecError(
            f"{where}: {value!r} is not a valid {enum_cls.__name__} (allowed: {allowed})"
        ) from exc


def _enum_value(value) -> str:
    """Return the plain string of an enum member, tolerating a raw string."""
    return value.value if hasattr(value, "value") else str(value)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise SpecError(f"expected a list of strings, got a bare string: {value!r}")
    return tuple(str(v) for v in value)


def _check_field_names(names: Iterable[str], where: str) -> None:
    seen: set[str] = set()
    for name in names:
        if not _FIELD_NAME_RE.match(name):
            raise SpecError(
                f"{where}: invalid contract field name {name!r} "
                "(must match [A-Za-z][A-Za-z0-9_]{0,63})"
            )
        if name in seen:
            raise SpecError(f"{where}: duplicate contract field name {name!r}")
        seen.add(name)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Contract:
    """The information interface between one agent and the orchestrator.

    ``out_fields`` are the fields the agent produces; ``in_fields`` are the
    fields it is willing to receive.  The RHI paper's core hypothesis is that
    restricting the interface to task-relevant fields is what makes a harness
    better, so these two tuples are the primary optimization target.
    """

    out_fields: tuple[str, ...]
    in_fields: tuple[str, ...]
    format: ContractFormat = ContractFormat.JSON
    max_tokens: int = 800
    free_text: bool = True

    def __post_init__(self) -> None:
        _check_field_names(self.out_fields, f"contract({self.out_fields})")
        _check_field_names(self.in_fields, f"contract({self.out_fields})")
        if not self.out_fields:
            raise SpecError("contract.out_fields must not be empty")
        if not (64 <= self.max_tokens <= 32000):
            raise SpecError(f"contract.max_tokens out of range: {self.max_tokens}")
        object.__setattr__(
            self, "format", _as_enum(ContractFormat, self.format, "contract.format")
        )

    @property
    def all_fields(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.out_fields, *self.in_fields)))


@dataclass(frozen=True)
class InstructionSlots:
    """Typed slots of an agent instruction.

    Free-form instruction text is deliberately *not* part of the search space:
    it cannot be attributed and it lets the optimizer drift into prose noise.
    """

    goal: GoalStyle = GoalStyle.PRIMARY
    required_outputs: tuple[str, ...] = ()
    forbidden_outputs: tuple[str, ...] = ()
    verification: VerificationStyle = VerificationStyle.SELF_CHECK

    def __post_init__(self) -> None:
        _check_field_names(self.required_outputs, "instruction_slots.required_outputs")
        _check_field_names(self.forbidden_outputs, "instruction_slots.forbidden_outputs")
        overlap = set(self.required_outputs) & set(self.forbidden_outputs)
        if overlap:
            raise SpecError(f"instruction slots both require and forbid: {sorted(overlap)}")
        object.__setattr__(
            self, "goal", _as_enum(GoalStyle, self.goal, "instruction_slots.goal")
        )
        object.__setattr__(
            self,
            "verification",
            _as_enum(
                VerificationStyle, self.verification, "instruction_slots.verification"
            ),
        )


@dataclass(frozen=True)
class AgentSpec:
    """One candidate agent of the harness."""

    id: str
    role: str
    instruction_template: str
    contract: Contract
    model: str
    reasoning_effort: str = "high"
    tools: tuple[str, ...] = ("read_file", "grep_file", "run_bash")
    max_tokens: int = 8192
    instruction_slots: InstructionSlots = field(default_factory=InstructionSlots)

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.id):
            raise SpecError(f"invalid agent id {self.id!r} (must match [a-z][a-z0-9_]{{0,31}})")
        if not self.role.strip():
            raise SpecError(f"agent {self.id!r}: role must not be empty")
        if not self.model.strip():
            raise SpecError(f"agent {self.id!r}: model must not be empty")
        if not (256 <= self.max_tokens <= 200000):
            raise SpecError(f"agent {self.id!r}: max_tokens out of range: {self.max_tokens}")
        if not self.tools:
            raise SpecError(f"agent {self.id!r}: at least one tool is required")
        if self.instruction_template not in INSTRUCTION_TEMPLATES:
            raise SpecError(
                f"agent {self.id!r}: unknown instruction_template {self.instruction_template!r}; "
                f"known: {sorted(INSTRUCTION_TEMPLATES)}"
            )

    def render_instruction(self) -> str:
        """Render the instruction template with the current slot values."""
        slots = self.instruction_slots
        required = ", ".join(slots.required_outputs) or "(anything relevant)"
        forbidden = ", ".join(slots.forbidden_outputs) or "(nothing in particular)"
        return INSTRUCTION_TEMPLATES[self.instruction_template].format(
            role=self.role,
            goal=_GOAL_TEXT[slots.goal],
            required_outputs=required,
            forbidden_outputs=forbidden,
            verification=_VERIFICATION_TEXT[slots.verification],
        )


@dataclass(frozen=True)
class Hop:
    """One edge of the orchestrator/subagent workflow."""

    id: str
    source: str
    target: str
    pass_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.id):
            raise SpecError(f"invalid hop id {self.id!r}")
        if not self.source or not self.target:
            raise SpecError(f"hop {self.id!r}: source and target are required")
        if self.source == self.target:
            raise SpecError(f"hop {self.id!r}: self-loops are not allowed")
        _check_field_names(self.pass_fields, f"hop({self.id}).pass_fields")


@dataclass(frozen=True)
class AuxiliaryRules:
    """Termination, fallback, recall and communication rules (RHI Figure 4)."""

    acceptance_gates: tuple[str, ...] = ()
    failure_fallback: tuple[str, ...] = (FailurePolicy.RETRY_ONCE.value,)
    recall_triggers: tuple[str, ...] = ()
    communication_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        known = {p.value for p in FailurePolicy}
        for policy in self.failure_fallback:
            if policy not in known:
                raise SpecError(
                    f"auxiliary_rules.failure_fallback: unknown policy {policy!r}; known: {sorted(known)}"
                )
        if not self.failure_fallback:
            raise SpecError("auxiliary_rules.failure_fallback must not be empty")


@dataclass(frozen=True)
class Runtime:
    """Execution budget and context-management knobs."""

    max_iterations: int = 40
    observation_max_chars: int = 10000
    context_keep_recent: int = 20
    seed: int = 1234
    temperature: float = 0.0
    max_hops: int = 6
    hop_context_max_chars: int = 6000
    min_delegations: int = 1

    def __post_init__(self) -> None:
        if not (1 <= self.max_iterations <= 200):
            raise SpecError(f"runtime.max_iterations out of range: {self.max_iterations}")
        if not (256 <= self.observation_max_chars <= 200000):
            raise SpecError(f"runtime.observation_max_chars out of range: {self.observation_max_chars}")
        if not (1 <= self.context_keep_recent <= 200):
            raise SpecError(f"runtime.context_keep_recent out of range: {self.context_keep_recent}")
        if not (1 <= self.max_hops <= 64):
            raise SpecError(f"runtime.max_hops out of range: {self.max_hops}")
        if not (256 <= self.hop_context_max_chars <= 200000):
            raise SpecError(f"runtime.hop_context_max_chars out of range: {self.hop_context_max_chars}")
        if not (0.0 <= self.temperature <= 2.0):
            raise SpecError(f"runtime.temperature out of range: {self.temperature}")
        if not (0 <= self.min_delegations <= 16):
            raise SpecError(f"runtime.min_delegations out of range: {self.min_delegations}")


@dataclass(frozen=True)
class HarnessSpec:
    """The complete, typed harness specification for one generation."""

    version: int
    agents: tuple[AgentSpec, ...]
    workflow: tuple[Hop, ...]
    auxiliary_rules: AuxiliaryRules = field(default_factory=AuxiliaryRules)
    runtime: Runtime = field(default_factory=Runtime)
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False)

    # -- construction ------------------------------------------------------- #

    def __post_init__(self) -> None:
        if self.version < 0:
            raise SpecError(f"version must be >= 0, got {self.version}")
        if not self.agents:
            raise SpecError("a spec needs at least one agent")

        ids = [a.id for a in self.agents]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise SpecError(f"duplicate agent ids: {dupes}")

        hop_ids = [h.id for h in self.workflow]
        if len(set(hop_ids)) != len(hop_ids):
            dupes = sorted({i for i in hop_ids if hop_ids.count(i) > 1})
            raise SpecError(f"duplicate hop ids: {dupes}")

        by_id = {a.id: a for a in self.agents}
        for hop in self.workflow:
            if hop.source not in by_id:
                raise SpecError(f"hop {hop.id!r}: unknown source agent {hop.source!r}")
            if hop.target not in by_id:
                raise SpecError(f"hop {hop.id!r}: unknown target agent {hop.target!r}")
            source, target = by_id[hop.source], by_id[hop.target]
            producible = set(source.contract.out_fields) | set(source.contract.in_fields)
            consumable = set(target.contract.in_fields)
            bad = [f for f in hop.pass_fields if f not in producible]
            if bad:
                raise SpecError(
                    f"hop {hop.id!r}: pass_fields {bad} are not produced by {source.id!r} "
                    f"(available: {sorted(producible)})"
                )
            bad = [f for f in hop.pass_fields if f not in consumable]
            if bad:
                raise SpecError(
                    f"hop {hop.id!r}: target {target.id!r} does not declare in_fields {bad}"
                )

        orchestrators = self.orchestrator_candidates()
        if len(orchestrators) != 1:
            raise SpecError(
                "the workflow must have exactly one orchestrator (the agent every "
                "other agent reports back to, and which delegates to the others); "
                f"candidates: {orchestrators}"
            )
        if not any(h.target == self.orchestrator for h in self.workflow):
            raise SpecError(
                f"the workflow must contain at least one edge back to the orchestrator "
                f"{self.orchestrator!r}"
            )

        first = self.agents[0]
        if "task" not in first.contract.in_fields:
            raise SpecError(
                f"the first agent {first.id!r} must declare 'task' in contract.in_fields "
                "(the task is the entry point of the information flow)"
            )
        if "task" not in self.agent(self.orchestrator).contract.in_fields:
            raise SpecError(
                f"the orchestrator {self.orchestrator!r} must declare 'task' in "
                "contract.in_fields: the task enters the harness through it"
            )

        for agent in self.agents:
            if agent.id == self.orchestrator:
                continue
            if not any(h.target == agent.id for h in self.workflow):
                raise SpecError(f"agent {agent.id!r} is never reached by any hop")

    # -- graph helpers ------------------------------------------------------ #

    def orchestrator_candidates(self) -> list[str]:
        """Agents that can act as the orchestrator of the workflow.

        The plan phrases the orchestrator as "the unique node with in-degree
        0", which cannot be literally true for a workflow that must also
        contain "at least one edge back to the orchestrator" -- the two rules
        contradict each other.  The rule implemented here is the one that
        actually captures the intent:

        an agent ``O`` is the orchestrator when

        1. every other agent has at least one hop whose target is ``O``
           (every agent can report back), and
        2. ``O`` has at least one hop whose target is another agent
           (the orchestrator really delegates).

        Exactly one agent may satisfy both conditions.
        """
        candidates: list[str] = []
        for candidate in self.agents:
            others = [a for a in self.agents if a.id != candidate.id]
            if not others:
                continue
            everyone_reports_back = all(
                any(h.source == other.id and h.target == candidate.id for h in self.workflow)
                for other in others
            )
            delegates = any(
                h.source == candidate.id and h.target != candidate.id for h in self.workflow
            )
            if everyone_reports_back and delegates:
                candidates.append(candidate.id)
        return candidates

    @property
    def orchestrator(self) -> str:
        return self.orchestrator_candidates()[0]

    def agent(self, agent_id: str) -> AgentSpec:
        for a in self.agents:
            if a.id == agent_id:
                return a
        raise KeyError(agent_id)

    def hops_from(self, agent_id: str) -> tuple[Hop, ...]:
        return tuple(h for h in self.workflow if h.source == agent_id)

    def hops_to(self, agent_id: str) -> tuple[Hop, ...]:
        return tuple(h for h in self.workflow if h.target == agent_id)

    def edge(self, source: str, target: str) -> Hop | None:
        for h in self.workflow:
            if h.source == source and h.target == target:
                return h
        return None

    def with_version(self, version: int) -> "HarnessSpec":
        return replace(self, version=version)

    def __hash__(self) -> int:  # frozen dataclass with a dict field is not hashable by default
        return int(self.spec_sha256()[:16], 16)

    # -- serialization ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Hand written serialization: no field may be silently dropped (T-12)."""
        return {
            "version": self.version,
            "agents": [
                {
                    "id": a.id,
                    "role": a.role,
                    "instruction_template": a.instruction_template,
                    "instruction_slots": {
                        "goal": _enum_value(a.instruction_slots.goal),
                        "required_outputs": list(a.instruction_slots.required_outputs),
                        "forbidden_outputs": list(a.instruction_slots.forbidden_outputs),
                        "verification": _enum_value(a.instruction_slots.verification),
                    },
                    "contract": {
                        "out_fields": list(a.contract.out_fields),
                        "in_fields": list(a.contract.in_fields),
                        "format": _enum_value(a.contract.format),
                        "max_tokens": a.contract.max_tokens,
                        "free_text": a.contract.free_text,
                    },
                    "model": a.model,
                    "reasoning_effort": a.reasoning_effort,
                    "tools": list(a.tools),
                    "max_tokens": a.max_tokens,
                }
                for a in self.agents
            ],
            "workflow": [
                {
                    "id": h.id,
                    "source": h.source,
                    "target": h.target,
                    "pass_fields": list(h.pass_fields),
                }
                for h in self.workflow
            ],
            "auxiliary_rules": {
                "acceptance_gates": list(self.auxiliary_rules.acceptance_gates),
                "failure_fallback": list(self.auxiliary_rules.failure_fallback),
                "recall_triggers": list(self.auxiliary_rules.recall_triggers),
                "communication_rules": list(self.auxiliary_rules.communication_rules),
            },
            "runtime": {
                "max_iterations": self.runtime.max_iterations,
                "observation_max_chars": self.runtime.observation_max_chars,
                "context_keep_recent": self.runtime.context_keep_recent,
                "seed": self.runtime.seed,
                "temperature": self.runtime.temperature,
                "max_hops": self.runtime.max_hops,
                "hop_context_max_chars": self.runtime.hop_context_max_chars,
                "min_delegations": self.runtime.min_delegations,
            },
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HarnessSpec":
        agents = []
        for raw in data.get("agents", []):
            slots = raw.get("instruction_slots", {}) or {}
            contract = raw["contract"]
            agents.append(
                AgentSpec(
                    id=raw["id"],
                    role=raw["role"],
                    instruction_template=raw.get("instruction_template", "standard"),
                    instruction_slots=InstructionSlots(
                        goal=GoalStyle(slots.get("goal", GoalStyle.PRIMARY.value)),
                        required_outputs=_as_tuple(slots.get("required_outputs")),
                        forbidden_outputs=_as_tuple(slots.get("forbidden_outputs")),
                        verification=VerificationStyle(
                            slots.get("verification", VerificationStyle.SELF_CHECK.value)
                        ),
                    ),
                    contract=Contract(
                        out_fields=_as_tuple(contract["out_fields"]),
                        in_fields=_as_tuple(contract.get("in_fields")),
                        format=ContractFormat(contract.get("format", ContractFormat.JSON.value)),
                        max_tokens=int(contract.get("max_tokens", 800)),
                        free_text=bool(contract.get("free_text", True)),
                    ),
                    model=raw["model"],
                    reasoning_effort=raw.get("reasoning_effort", "high"),
                    tools=_as_tuple(raw.get("tools")) or ("read_file", "grep_file", "run_bash"),
                    max_tokens=int(raw.get("max_tokens", 8192)),
                )
            )

        workflow = [
            Hop(
                id=h["id"],
                source=h["source"],
                target=h["target"],
                pass_fields=_as_tuple(h.get("pass_fields")),
            )
            for h in data.get("workflow", [])
        ]

        aux_raw = data.get("auxiliary_rules", {}) or {}
        aux = AuxiliaryRules(
            acceptance_gates=_as_tuple(aux_raw.get("acceptance_gates")),
            failure_fallback=_as_tuple(aux_raw.get("failure_fallback"))
            or (FailurePolicy.RETRY_ONCE.value,),
            recall_triggers=_as_tuple(aux_raw.get("recall_triggers")),
            communication_rules=_as_tuple(aux_raw.get("communication_rules")),
        )

        rt_raw = data.get("runtime", {}) or {}
        runtime = Runtime(
            max_iterations=int(rt_raw.get("max_iterations", 40)),
            observation_max_chars=int(rt_raw.get("observation_max_chars", 10000)),
            context_keep_recent=int(rt_raw.get("context_keep_recent", 20)),
            seed=int(rt_raw.get("seed", 1234)),
            temperature=float(rt_raw.get("temperature", 0.0)),
            max_hops=int(rt_raw.get("max_hops", 6)),
            hop_context_max_chars=int(rt_raw.get("hop_context_max_chars", 6000)),
            min_delegations=int(rt_raw.get("min_delegations", 1)),
        )

        return cls(
            version=int(data.get("version", 0)),
            agents=tuple(agents),
            workflow=tuple(workflow),
            auxiliary_rules=aux,
            runtime=runtime,
            meta=dict(data.get("meta", {}) or {}),
        )

    # -- hashing ------------------------------------------------------------ #

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def spec_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# YAML helpers
# --------------------------------------------------------------------------- #


def load_spec(path: str | Path) -> HarnessSpec:
    """Load a spec from a YAML file."""
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise SpecError(f"{p}: spec file must contain a mapping at the top level")
    return HarnessSpec.from_dict(data)


def save_spec(spec: HarnessSpec, path: str | Path) -> Path:
    """Write a spec to YAML, sorted by key so the file diff is stable."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=True, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    return p


def spec_sha256(spec: HarnessSpec) -> str:
    return spec.spec_sha256()


# --------------------------------------------------------------------------- #
# Path access (used by the optimizer and the ledger diff)
# --------------------------------------------------------------------------- #

_PATH_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?")


def _parse_path(path: str) -> list[tuple[str, int | None]]:
    """Parse a dotted path into a list of operations.

    An operation is either ``("key", name)`` or ``("index", i)``.  The index
    belongs to the *value* of the preceding key: ``agents[1]`` means "the
    ``agents`` value, element 1".
    """
    ops: list[tuple[str, int | None]] = []
    pos = 0
    while pos < len(path):
        m = _PATH_TOKEN_RE.match(path, pos)
        if not m:
            raise SpecError(f"invalid path {path!r} at offset {pos}")
        name, index = m.group(1), m.group(2)
        ops.append(("key", name))
        if index is not None:
            ops.append(("index", int(index)))
        pos = m.end()
        if pos < len(path):
            if path[pos] != ".":
                raise SpecError(f"invalid path {path!r} at offset {pos}")
            pos += 1
    return ops


def get_path(obj: Any, path: str) -> Any:
    """Read a value out of a nested structure addressed by a dotted path.

    ``agents[0].contract.max_tokens`` is the canonical addressing scheme.
    """
    current = obj
    for kind, value in _parse_path(path):
        if kind == "key":
            if isinstance(current, Mapping):
                if value not in current:
                    raise SpecError(f"path {path!r}: no key {value!r}")
                current = current[value]
            elif isinstance(current, (list, tuple)):
                raise SpecError(f"path {path!r}: expected an index for {value!r}")
            else:
                current = getattr(current, value)
        else:  # index
            if not isinstance(current, (list, tuple)):
                raise SpecError(f"path {path!r}: {value} is not an index into a list")
            if not (-len(current) <= value < len(current)):
                raise SpecError(f"path {path!r}: index {value} is out of range")
            current = current[value]
    return current


def set_path(obj: Any, path: str, value: Any) -> Any:
    """Return a copy of ``obj`` with ``path`` set to ``value``.

    Works on both dataclass instances and plain dicts, and never mutates the
    input.
    """
    ops = _parse_path(path)
    if not ops:
        raise SpecError("empty path")

    def _apply(node: Any, ops: list[tuple[str, int | None]], value: Any) -> Any:
        kind, key = ops[0]
        rest = ops[1:]
        if kind == "key":
            if isinstance(node, Mapping):
                if key not in node:
                    raise SpecError(f"path: no key {key!r}")
                new = dict(node)
                if rest:
                    new[key] = _apply(node[key], rest, value)
                else:
                    new[key] = value
                return new
            if isinstance(node, (list, tuple)):
                raise SpecError(f"path: expected an index for {key!r}")
            if not hasattr(node, key):
                raise SpecError(
                    f"path: no attribute {key!r} on {type(node).__name__}"
                )
            if rest:
                return replace(node, **{key: _apply(getattr(node, key), rest, value)})
            return replace(node, **{key: value})
        # index
        if not isinstance(node, (list, tuple)):
            raise SpecError(f"path: {key} is not an index into a list")
        new_list = list(node)
        if rest:
            new_list[key] = _apply(node[key], rest, value)
        else:
            new_list[key] = value
        return new_list

    return _apply(obj, ops, value)


def flatten_spec(spec: HarnessSpec) -> dict[str, Any]:
    """Flatten a spec into ``path -> value`` for diffing and hashing."""
    flat: dict[str, Any] = {}
    for i, agent in enumerate(spec.agents):
        for field_name, value in (
            ("role", agent.role),
            ("instruction_template", agent.instruction_template),
            ("model", agent.model),
            ("reasoning_effort", agent.reasoning_effort),
            ("max_tokens", agent.max_tokens),
        ):
            flat[f"agents[{i}].{field_name}"] = value
        # Lists are kept whole: a list is *one* field, so reordering or adding
        # an element must count as a single edit against the edit budget.
        flat[f"agents[{i}].tools"] = tuple(agent.tools)
        flat[f"agents[{i}].contract.out_fields"] = tuple(agent.contract.out_fields)
        flat[f"agents[{i}].contract.in_fields"] = tuple(agent.contract.in_fields)
        flat[f"agents[{i}].contract.format"] = _enum_value(agent.contract.format)
        flat[f"agents[{i}].contract.max_tokens"] = agent.contract.max_tokens
        flat[f"agents[{i}].contract.free_text"] = agent.contract.free_text
        flat[f"agents[{i}].instruction_slots.goal"] = _enum_value(agent.instruction_slots.goal)
        flat[f"agents[{i}].instruction_slots.verification"] = _enum_value(
            agent.instruction_slots.verification
        )
        flat[f"agents[{i}].instruction_slots.required_outputs"] = tuple(
            agent.instruction_slots.required_outputs
        )
        flat[f"agents[{i}].instruction_slots.forbidden_outputs"] = tuple(
            agent.instruction_slots.forbidden_outputs
        )
    for i, hop in enumerate(spec.workflow):
        flat[f"workflow[{i}].id"] = hop.id
        flat[f"workflow[{i}].source"] = hop.source
        flat[f"workflow[{i}].target"] = hop.target
        flat[f"workflow[{i}].pass_fields"] = tuple(hop.pass_fields)
    flat["auxiliary_rules.acceptance_gates"] = tuple(spec.auxiliary_rules.acceptance_gates)
    flat["auxiliary_rules.recall_triggers"] = tuple(spec.auxiliary_rules.recall_triggers)
    flat["auxiliary_rules.communication_rules"] = tuple(spec.auxiliary_rules.communication_rules)
    flat["auxiliary_rules.failure_fallback"] = tuple(spec.auxiliary_rules.failure_fallback)
    for name in (
        "max_iterations",
        "observation_max_chars",
        "context_keep_recent",
        "seed",
        "temperature",
        "max_hops",
        "hop_context_max_chars",
        "min_delegations",
    ):
        flat[f"runtime.{name}"] = getattr(spec.runtime, name)
    return flat


def apply_flattened(spec: HarnessSpec, flat: Mapping[str, Any]) -> HarnessSpec:
    """Inverse of :func:`flatten_spec` for the mutable subset of paths."""
    data = spec.to_dict()
    for path, value in flat.items():
        data = set_path(data, path, value)
    return HarnessSpec.from_dict(data)
