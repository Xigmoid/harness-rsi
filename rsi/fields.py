"""Registry of optimizable fields.

This module is the *only* source of truth for "which knobs are worth turning".
It is also the skeleton of the attribution table: every field carries a group
(``contract`` / ``hop`` / ``instruction`` / ``role`` / ``runtime`` / ``model``)
and a priority derived from the RHI ablation:

* P0 ``contract`` and ``hop`` -- RHI's component-level clustering shows that
  contracts change the most and cluster most clearly, hops come second.
* P1 ``runtime`` -- cheap to evaluate, directly visible in cost/quality.
* P2 ``instruction`` -- only *slots* are searchable, never free prose.
* P3 ``role`` / ``model`` -- roles are almost indistinguishable across tasks in
  RHI; models are an explicit "break the homogenization" knob, off by default.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Mapping, Sequence

from .schema import (
    ContractFormat,
    GoalStyle,
    HarnessSpec,
    SpecError,
    VerificationStyle,
    get_path,
    set_path,
)

__all__ = [
    "FieldSpec",
    "FieldTemplate",
    "FIELD_TEMPLATES",
    "FIELD_VOCABULARY",
    "ROLE_DOMAIN",
    "EFFORT_DOMAIN",
    "expand_fields",
    "mutable_fields",
    "get_value",
    "set_value",
    "mutate",
    "sample_value",
    "group_of",
]

Kind = Literal["int", "float", "str", "bool", "enum", "list"]

#: Vocabulary of contract field names.  A contract field is a *named slot* in
#: the information interface; the vocabulary is deliberately small and shared so
#: that two generations stay comparable (and so that "one more field" stays a
#: meaningful edit rather than a new concept).
#:
#: The vocabulary spans several capabilities on purpose.  If it only contained
#: code-flavoured names ("patch", "test_results"), a math harness would be
#: unable to express "derivation" and "answer", and the cross-capability
#: comparison would be rigged before it started.  A *shared* vocabulary lets
#: each capability select the fields it actually needs -- which is precisely
#: the RHI hypothesis under test.
FIELD_VOCABULARY: tuple[str, ...] = (
    # task / control
    "task",
    "plan",
    "status",
    "summary",
    "next_steps",
    "risks",
    "questions",
    # evidence
    "findings",
    "evidence",
    "artifacts",
    "metrics",
    "logs",
    "commands",
    # code
    "patch",
    "diff",
    "test_results",
    # math / reasoning
    "answer",
    "derivation",
    "check",
    "candidates",
    "constraint",
    "assumption",
    "alternative",
    "comparison",
)

#: Role labels are drawn from a closed vocabulary so that a role edit stays
#: typed and attributable (P3 priority).
ROLE_DOMAIN: tuple[str, ...] = (
    "coordinator",
    "planner",
    "analyst",
    "implementer",
    "reviewer",
    "verifier",
)

EFFORT_DOMAIN: tuple[str, ...] = ("low", "medium", "high", "xhigh")


@dataclass(frozen=True)
class FieldSpec:
    """A concrete, addressable knob of a specific spec."""

    path: str
    kind: Kind
    domain: tuple[Any, ...] | None = None
    mutable: bool = True
    group: str = ""
    priority: int = 3
    rationale: str = ""
    vocabulary: tuple[str, ...] = ()
    min_size: int = 0
    max_size: int = 64

    @property
    def continuous(self) -> bool:
        return self.kind in ("int", "float")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.path} [{self.group}/P{self.priority}]"


@dataclass(frozen=True)
class FieldTemplate:
    """A group-level description, expanded into concrete paths per spec."""

    template: str
    kind: Kind
    scope: Literal["agent", "hop", "global"] = "agent"
    domain: tuple[Any, ...] | None = None
    mutable: bool = True
    group: str = ""
    priority: int = 3
    rationale: str = ""
    min_size: int = 0
    max_size: int = 64


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

FIELD_TEMPLATES: tuple[FieldTemplate, ...] = (
    # ---------------- P0: contract (the information interface) ------------- #
    FieldTemplate(
        template="agents[{i}].contract.out_fields",
        kind="list",
        scope="agent",
        group="contract",
        priority=0,
        rationale="RHI 6.2.2: contracts show the clearest task-dependent clustering.",
        min_size=1,
        max_size=6,
    ),
    FieldTemplate(
        template="agents[{i}].contract.in_fields",
        kind="list",
        scope="agent",
        group="contract",
        priority=0,
        rationale="Restricting what a subagent may see is the core RHI hypothesis.",
        min_size=1,
        max_size=6,
    ),
    FieldTemplate(
        template="agents[{i}].contract.max_tokens",
        kind="int",
        scope="agent",
        domain=(128, 4000),
        group="contract",
        priority=0,
        rationale="Payload budget: too small truncates evidence, too large re-adds redundancy.",
    ),
    FieldTemplate(
        template="agents[{i}].contract.format",
        kind="enum",
        scope="agent",
        domain=tuple(f.value for f in ContractFormat),
        group="contract",
        priority=0,
        rationale="JSON is machine-checkable; markdown is more robust to model drift.",
    ),
    FieldTemplate(
        template="agents[{i}].contract.free_text",
        kind="bool",
        scope="agent",
        group="contract",
        priority=0,
        rationale="Whether the orchestrator may also read unstructured prose next to the fields.",
    ),
    # ---------------- P0: hop (the interaction structure) ------------------- #
    FieldTemplate(
        template="workflow[{i}].pass_fields",
        kind="list",
        scope="hop",
        group="hop",
        priority=0,
        rationale="RHI: hops also cluster by task; pass_fields are the concrete interface.",
        min_size=0,
        max_size=6,
    ),
    FieldTemplate(
        template="workflow[{i}].target",
        kind="enum",
        scope="hop",
        group="hop",
        priority=0,
        rationale="Rewiring an edge changes who sees which evidence.",
    ),
    # ---------------- P1: runtime (cost and context) ----------------------- #
    FieldTemplate(
        template="runtime.max_iterations",
        kind="int",
        scope="global",
        domain=(8, 120),
        group="runtime",
        priority=1,
        rationale="Total turn budget of the orchestrator loop.",
    ),
    FieldTemplate(
        template="runtime.observation_max_chars",
        kind="int",
        scope="global",
        domain=(1000, 50000),
        group="runtime",
        priority=1,
        rationale="How much command output survives truncation.",
    ),
    FieldTemplate(
        template="runtime.context_keep_recent",
        kind="int",
        scope="global",
        domain=(4, 60),
        group="runtime",
        priority=1,
        rationale="How many recent turns the orchestrator keeps verbatim.",
    ),
    FieldTemplate(
        template="runtime.max_hops",
        kind="int",
        scope="global",
        domain=(1, 16),
        group="runtime",
        priority=1,
        rationale="Maximum number of orchestrator->subagent->orchestrator cycles.",
    ),
    # ---------------- P2: instruction (slots only) ------------------------- #
    FieldTemplate(
        template="agents[{i}].instruction_slots.goal",
        kind="enum",
        scope="agent",
        domain=tuple(g.value for g in GoalStyle),
        group="instruction",
        priority=2,
        rationale="Goal phrasing changes how thorough the agent is, without prose freedom.",
    ),
    FieldTemplate(
        template="agents[{i}].instruction_slots.verification",
        kind="enum",
        scope="agent",
        domain=tuple(v.value for v in VerificationStyle),
        group="instruction",
        priority=2,
        rationale="Verification style trades cost for reliability.",
    ),
    FieldTemplate(
        template="agents[{i}].instruction_slots.required_outputs",
        kind="list",
        scope="agent",
        group="instruction",
        priority=2,
        rationale="What the agent must deliver, expressed as named slots.",
        min_size=0,
        max_size=4,
    ),
    FieldTemplate(
        template="agents[{i}].instruction_slots.forbidden_outputs",
        kind="list",
        scope="agent",
        group="instruction",
        priority=2,
        rationale="What the agent must not spend tokens on.",
        min_size=0,
        max_size=4,
    ),
    FieldTemplate(
        template="agents[{i}].instruction_template",
        kind="enum",
        scope="agent",
        domain=("standard", "concise", "evidence_first"),
        group="instruction",
        priority=2,
        rationale="Scaffold of the instruction; swapping it is one discrete, attributable edit.",
    ),
    # ---------------- P3: role (almost never moves) ------------------------ #
    FieldTemplate(
        template="agents[{i}].role",
        kind="enum",
        scope="agent",
        domain=ROLE_DOMAIN,
        group="role",
        priority=3,
        rationale="RHI: roles are the least separable component across tasks.",
    ),
    # ---------------- P3: model (homogenization breaker, off by default) --- #
    FieldTemplate(
        template="agents[{i}].reasoning_effort",
        kind="enum",
        scope="agent",
        domain=EFFORT_DOMAIN,
        mutable=False,
        group="model",
        priority=3,
        rationale="Explicit knob to break harness homogenization; disabled by default.",
    ),
    FieldTemplate(
        template="agents[{i}].max_tokens",
        kind="int",
        scope="agent",
        domain=(1024, 65536),
        mutable=False,
        group="model",
        priority=3,
        rationale="Per-agent generation budget; disabled by default to keep the budget small.",
    ),
)


# --------------------------------------------------------------------------- #
# Expansion
# --------------------------------------------------------------------------- #


def _pass_field_vocabulary(spec: HarnessSpec, hop_index: int) -> tuple[str, ...]:
    hop = spec.workflow[hop_index]
    source = spec.agent(hop.source)
    target = spec.agent(hop.target)
    producible = set(source.contract.out_fields) | set(source.contract.in_fields)
    consumable = set(target.contract.in_fields)
    return tuple(sorted(producible & consumable))


def expand_fields(spec: HarnessSpec) -> tuple[FieldSpec, ...]:
    """Expand the templates into concrete, addressable fields of ``spec``."""
    out: list[FieldSpec] = []
    for tpl in FIELD_TEMPLATES:
        if tpl.scope == "agent":
            for i, agent in enumerate(spec.agents):
                path = tpl.template.format(i=i)
                domain = tpl.domain
                if path.endswith(".role"):
                    domain = tpl.domain
                out.append(
                    FieldSpec(
                        path=path,
                        kind=tpl.kind,
                        domain=domain,
                        mutable=tpl.mutable,
                        group=tpl.group,
                        priority=tpl.priority,
                        rationale=tpl.rationale,
                        vocabulary=FIELD_VOCABULARY if tpl.kind == "list" else (),
                        min_size=tpl.min_size,
                        max_size=tpl.max_size,
                    )
                )
        elif tpl.scope == "hop":
            for i, hop in enumerate(spec.workflow):
                path = tpl.template.format(i=i)
                if path.endswith(".target"):
                    domain = tuple(a.id for a in spec.agents if a.id != hop.source)
                else:
                    domain = None
                out.append(
                    FieldSpec(
                        path=path,
                        kind=tpl.kind,
                        domain=domain,
                        mutable=tpl.mutable,
                        group=tpl.group,
                        priority=tpl.priority,
                        rationale=tpl.rationale,
                        vocabulary=_pass_field_vocabulary(spec, i) if tpl.kind == "list" else (),
                        min_size=tpl.min_size,
                        max_size=tpl.max_size,
                    )
                )
        else:  # global
            out.append(
                FieldSpec(
                    path=tpl.template,
                    kind=tpl.kind,
                    domain=tpl.domain,
                    mutable=tpl.mutable,
                    group=tpl.group,
                    priority=tpl.priority,
                    rationale=tpl.rationale,
                    vocabulary=FIELD_VOCABULARY if tpl.kind == "list" else (),
                    min_size=tpl.min_size,
                    max_size=tpl.max_size,
                )
            )
    return tuple(out)


def mutable_fields(spec: HarnessSpec) -> tuple[FieldSpec, ...]:
    return tuple(f for f in expand_fields(spec) if f.mutable)


def group_of(spec: HarnessSpec, path: str) -> str:
    for f in expand_fields(spec):
        if f.path == path:
            return f.group
    return "unknown"


# --------------------------------------------------------------------------- #
# Read / write / mutate
# --------------------------------------------------------------------------- #


def get_value(spec: HarnessSpec, path: str) -> Any:
    """Read a field value out of a spec (works on the dict form)."""
    return get_path(spec.to_dict(), path)


def set_value(spec: HarnessSpec, path: str, value: Any) -> HarnessSpec:
    """Return a new, validated spec with ``path`` set to ``value``."""
    data = set_path(spec.to_dict(), path, value)
    return HarnessSpec.from_dict(data)


def sample_value(field_spec: FieldSpec, rng: random.Random) -> Any:
    """Sample a value for a field from its domain."""
    kind = field_spec.kind
    if kind == "int":
        lo, hi = field_spec.domain or (0, 1)
        return rng.randint(int(lo), int(hi))
    if kind == "float":
        lo, hi = field_spec.domain or (0.0, 1.0)
        return round(rng.uniform(float(lo), float(hi)), 4)
    if kind == "bool":
        return rng.choice([True, False])
    if kind == "enum":
        domain = field_spec.domain or ()
        if not domain:
            raise SpecError(f"field {field_spec.path!r} has no domain to sample from")
        return rng.choice(list(domain))
    if kind == "str":
        domain = field_spec.domain or ()
        return rng.choice(list(domain)) if domain else "unknown"
    raise SpecError(f"field {field_spec.path!r}: cannot sample a value of kind {kind!r}")


def mutate(spec: HarnessSpec, path: str, rng: random.Random, *, max_attempts: int = 50) -> HarnessSpec:
    """Apply one random, validated edit to ``spec`` and return the new spec.

    The edit is validated: a mutation that would produce an invalid spec (for
    example dropping a contract field that a hop still transports) is retried
    with another sample.  :class:`SpecError` is raised only when no admissible
    mutation exists at all.
    """
    fields = {f.path: f for f in expand_fields(spec)}
    if path not in fields:
        raise SpecError(f"unknown field {path!r}")
    f = fields[path]

    last_error: Exception | None = None
    for _ in range(max(1, max_attempts)):
        current = get_value(spec, path)
        if f.kind == "list":
            current_list = list(current or [])
            vocab = [v for v in f.vocabulary if v not in current_list]
            options: list[tuple[str, ...]] = []
            if vocab and len(current_list) < f.max_size:
                for v in vocab:
                    options.append(tuple(sorted([*current_list, v])))
            if len(current_list) > f.min_size:
                for idx in range(len(current_list)):
                    options.append(tuple(v for j, v in enumerate(current_list) if j != idx))
            if not options:
                raise SpecError(f"field {path!r}: no admissible list mutation")
            new_value = rng.choice(options)
            payload: Any = list(new_value)
        else:
            # A mutation that does not change anything is not a mutation: draw
            # again, excluding the current value from the domain.
            effective = f
            if f.domain and current in f.domain and len(f.domain) > 1:
                effective = replace(
                    f, domain=tuple(v for v in f.domain if v != current)
                )
            payload = sample_value(effective, rng)
            if payload == current:
                continue

        try:
            return set_value(spec, path, payload)
        except SpecError as exc:
            last_error = exc
            continue

    raise SpecError(
        f"field {path!r}: no admissible mutation found "
        f"({last_error})"
    )
