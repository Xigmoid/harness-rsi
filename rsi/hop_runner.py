"""Hop scheduling and contract rendering.

This module implements the RHI information-flow hypothesis in code:

* a *contract* says which named fields an agent produces (``out_fields``) and
  which ones it is willing to receive (``in_fields``);
* a *hop* says which fields actually travel along one edge of the workflow;
* the orchestrator only ever sees the contract-filtered payload of a subagent,
  never the subagent's raw trajectory.

The module is deliberately free of any mini-swe-agent import so that it can be
unit tested (and reasoned about) on its own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .schema import AgentSpec, ContractFormat, FailurePolicy, HarnessSpec, Hop, SpecError

__all__ = [
    "PayloadError",
    "HopRecord",
    "RunState",
    "HopRunner",
    "DELEGATE_BLOCK",
    "HANDOFF_BLOCK",
]

DELEGATE_BLOCK = "delegate"
HANDOFF_BLOCK = "handoff"

_DELEGATE_RE = re.compile(rf"<{DELEGATE_BLOCK}>\s*([A-Za-z0-9_]+)\s*</{DELEGATE_BLOCK}>")
_HANDOFF_RE = re.compile(rf"<{HANDOFF_BLOCK}>\s*(.*?)\s*</{HANDOFF_BLOCK}>", re.DOTALL)
_MD_FIELD_RE = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*\*)?([A-Za-z][A-Za-z0-9_]{0,63})(?:\*\*)?\s*[:：]\s*(.*)$")


class PayloadError(Exception):
    """Raised when a contract payload cannot be extracted from a raw message."""

    def __init__(self, message: str, *, missing: Sequence[str] = (), reason: str = "invalid"):
        super().__init__(message)
        self.missing = tuple(missing)
        self.reason = reason


@dataclass
class HopRecord:
    """Bookkeeping for one executed hop."""

    hop_id: str
    source: str
    target: str
    fields: tuple[str, ...]
    n_turns: int = 0
    parse_failures: int = 0
    degraded: bool = False
    error: str | None = None


@dataclass
class RunState:
    """Mutable state of one agent run."""

    task: str
    current_agent: str
    cycle: int = 0
    hops: int = 0
    delegations: int = 0
    turns: int = 0
    payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    hop_log: list[HopRecord] = field(default_factory=list)
    parse_failures: int = 0
    n_calls: int = 0
    cost: float = 0.0
    finished: bool = False
    finish_reason: str = ""
    pending_delegate: str | None = None
    cycle_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record_payload(self, agent_id: str, payload: Mapping[str, Any]) -> None:
        self.payloads[agent_id] = dict(payload)
        self.cycle_payloads[agent_id] = dict(payload)


class HopRunner:
    """Schedules hops and renders the contract-limited context of each agent."""

    def __init__(self, spec: HarnessSpec, *, action_protocol: str = "text"):
        self.spec = spec
        self.orchestrator = spec.orchestrator
        if action_protocol not in ("text", "toolcall"):
            raise ValueError(
                f"action_protocol must be 'text' or 'toolcall', got {action_protocol!r}"
            )
        self.action_protocol = action_protocol

    # ------------------------------------------------------------------ #
    # scheduling
    # ------------------------------------------------------------------ #

    def next_hop(self, state: RunState) -> Hop | None:
        """Select the next edge to traverse out of the current agent.

        The scheduler is deterministic: a delegation request from the model wins,
        otherwise the current agent's outgoing edges are traversed in workflow
        order.  A deterministic scheduler is required for the generations to be
        comparable at all - a random router would add noise that no amount of
        seeds could average away.
        """
        if state.finished:
            return None
        current = state.current_agent

        if state.pending_delegate:
            target = state.pending_delegate
            state.pending_delegate = None
            hop = self.spec.edge(current, target)
            if hop is not None:
                return hop
            # The model asked for an agent that is not wired up: fall back to
            # the ordinary schedule instead of crashing the run.

        candidates = [h for h in self.spec.hops_from(current) if h.target != current]
        if not candidates:
            return None

        # Round-robin over the outgoing edges so that a fan-out workflow
        # actually visits every branch once per cycle.
        return candidates[state.cycle % len(candidates)]

    def transfer_target(self, agent_id: str, state: RunState) -> Hop | None:
        """Choose the edge along which control returns from ``agent_id``.

        The orchestrator is preferred as the target: a subagent that produced a
        payload normally hands control back, which is what makes the loop
        iterative rather than one-pass.
        """
        outgoing = list(self.spec.hops_from(agent_id))
        if not outgoing:
            return None
        back = [h for h in outgoing if h.target == self.orchestrator]
        if back:
            return back[0]
        return outgoing[state.cycle % len(outgoing)]

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #

    def render_system(self, agent_id: str, hop: Hop | None = None) -> str:
        """Render the system message of one agent.

        The orchestrator sees the *whole* workflow (it is the coordinator); a
        subagent only sees the fields it is allowed to receive.  This asymmetry
        is the RHI hypothesis implemented literally.
        """
        agent = self.spec.agent(agent_id)
        lines: list[str] = [agent.render_instruction(), ""]

        if agent_id == self.orchestrator:
            lines.append("## You are the orchestrator")
            lines.append(
                "You coordinate the other agents. You do not read their raw transcripts: "
                "you only receive the contract fields listed below."
            )
            lines.append("")
            lines.append("### Available agents")
            for other in self.spec.agents:
                if other.id == agent_id:
                    continue
                lines.append(
                    f"- `{other.id}` — role: {other.role}; "
                    f"receives: {', '.join(other.contract.in_fields) or '(none)'}; "
                    f"produces: {', '.join(other.contract.out_fields)}"
                )
            lines.append("")
            lines.append("### Workflow (hops)")
            for h in self.spec.workflow:
                passed = ", ".join(h.pass_fields) or "(nothing)"
                lines.append(f"- {h.id}: {h.source} -> {h.target} passing {passed}")
            lines.append("")
            if self.spec.auxiliary_rules.acceptance_gates:
                lines.append("### Acceptance gates (all must hold before you finish)")
                for gate in self.spec.auxiliary_rules.acceptance_gates:
                    lines.append(f"- {gate}")
                lines.append("")
            if self.spec.auxiliary_rules.recall_triggers:
                lines.append("### Recall triggers (delegate again when one of these holds)")
                for trigger in self.spec.auxiliary_rules.recall_triggers:
                    lines.append(f"- {trigger}")
                lines.append("")
        else:
            lines.append(f"## You are the subagent `{agent_id}`")
            lines.append(
                "You only receive the fields listed below — you do not see the other "
                "agents' transcripts. Work only from the information you were given."
            )
            lines.append("")
            lines.append("### Fields you may receive")
            lines.append(f"- {', '.join(agent.contract.in_fields) or '(nothing)'}")
            lines.append("")
            if hop is not None:
                lines.append("### Fields you receive on this hop")
                lines.append(
                    f"- {', '.join(hop.pass_fields) or '(nothing)'}"
                    if hop.pass_fields
                    else "- (nothing on this hop)"
                )
                lines.append("")
            lines.append("### Fields you must produce")
            lines.append(f"- {', '.join(agent.contract.out_fields)}")
            lines.append("")

        if self.spec.auxiliary_rules.communication_rules:
            lines.append("### Communication rules")
            for rule in self.spec.auxiliary_rules.communication_rules:
                lines.append(f"- {rule}")
            lines.append("")

        lines.append("### Tools")
        lines.append(f"- {', '.join(agent.tools)}")
        lines.append("")

        lines.append(self._protocol_block(agent_id))
        return "\n".join(lines)

    def _protocol_block(self, agent_id: str) -> str:
        agent = self.spec.agent(agent_id)
        others = ", ".join(a.id for a in self.spec.agents if a.id != agent_id)

        if self.action_protocol == "toolcall":
            action_rules = (
                "Every response must contain exactly one call to the `bash` tool, "
                "with a single command as its argument."
            )
            submit = "To finish the task, call `bash` with exactly this command:\n"
            handoff = (
                "When you are done, your final bash command must contain the handoff "
                "block, for example:\n"
                "cat <<'RHI_EOF' >/dev/null\n"
                "<handoff>\n"
                f'{{"{agent.contract.out_fields[0]}": "..."}}\n'
                "</handoff>\n"
                "RHI_EOF"
            )
        else:
            action_rules = (
                "Every response must contain exactly one bash code block with one command."
            )
            submit = "To finish the task, run exactly:\n"
            handoff = (
                "When you are done, add a handoff block to the same response (outside the "
                "code block), for example:\n"
                "<handoff>\n"
                f'{{"{agent.contract.out_fields[0]}": "..."}}\n'
                "</handoff>"
            )

        if agent_id == self.orchestrator:
            delegation_rule = ""
            if self.spec.runtime.min_delegations > 0:
                delegation_rule = (
                    f"\nYou MUST delegate the work to a subagent at least "
                    f"{self.spec.runtime.min_delegations} time(s) before you can finish. "
                    "Do not do the work yourself: your job is to plan, dispatch and verify.\n"
                )
            return (
                "## Protocol\n"
                f"{action_rules}\n"
                "To hand control to a subagent, add this line to the same response:\n"
                f"<{DELEGATE_BLOCK}>agent_id</{DELEGATE_BLOCK}>\n"
                f"Available agent ids: {others}\n"
                f"{delegation_rule}"
                f"{submit}"
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
                "(and nothing else in that command)."
            )

        fmt = (
            "a JSON object"
            if agent.contract.format is ContractFormat.JSON
            else "a markdown block with one `field: value` line per field"
        )
        return (
            "## Protocol\n"
            f"{action_rules}\n"
            f"{handoff}\n"
            f"The handoff must contain {fmt} with exactly these keys: "
            f"{', '.join(agent.contract.out_fields)}."
        )

    def assemble_context(self, agent_id: str, state: RunState, hop: Hop | None = None) -> str:
        """Assemble the contract-filtered context handed to ``agent_id``.

        Only fields the agent declared in ``in_fields`` are included, and only
        the ones that actually travelled on the hop.  This is the sparse
        "task-specific attention pattern" RHI argues for.
        """
        agent = self.spec.agent(agent_id)
        wanted = set(agent.contract.in_fields)
        chunks: list[str] = []

        if "task" in wanted:
            chunks.append(f"# Task\n{state.task}")

        received: dict[str, Any] = {}
        if hop is not None:
            source_payload = state.payloads.get(hop.source, {})
            for field_name in hop.pass_fields:
                if field_name not in wanted:
                    continue
                if field_name in source_payload:
                    received[field_name] = source_payload[field_name]
                    continue
                # The orchestrator has no payload of its own: it forwards the
                # evidence it collected from earlier subagents.
                for payload in state.cycle_payloads.values():
                    if field_name in payload:
                        received[field_name] = payload[field_name]
                        break
        else:
            for payload in state.cycle_payloads.values():
                for field_name, value in payload.items():
                    if field_name in wanted:
                        received.setdefault(field_name, value)

        if received:
            rendered = json.dumps(received, ensure_ascii=False, indent=2, sort_keys=True)
            chunks.append(f"# Received from other agents\n{rendered}")

        text = "\n\n".join(chunks)
        limit = self.spec.runtime.hop_context_max_chars
        if limit and len(text) > limit:
            text = text[: limit - 40] + "\n...[context truncated by harness]"
        return text

    # ------------------------------------------------------------------ #
    # payload extraction
    # ------------------------------------------------------------------ #

    def extract_payload(
        self,
        agent_id: str,
        raw: str,
        actions: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Extract the contract payload of ``agent_id`` from a raw message.

        The payload can arrive in two places, because mini-swe-agent supports
        two action protocols:

        * **text-based** -- the model writes a ``<handoff>`` block into the
          message text next to its code block;
        * **tool-call** -- the model puts the ``<handoff>`` block into the
          command of its final bash tool call (the message body is usually
          empty in that mode).

        Both are checked, in that order.  A missing payload is reported as a
        :class:`PayloadError`; the caller decides what to do with it.
        """
        agent = self.spec.agent(agent_id)
        body = self._handoff_body(raw)
        if body is None and actions:
            for action in actions:
                command = str(action.get("command", ""))
                body = self._handoff_body(command)
                if body is not None:
                    break
        if body is None:
            raise PayloadError(
                f"agent {agent_id!r} did not emit a <{HANDOFF_BLOCK}> block",
                reason="missing_block",
            )
        if agent.contract.format is ContractFormat.JSON:
            payload = self._parse_json_payload(agent, body)
        else:
            payload = self._parse_markdown_payload(agent, body)
        return payload

    @staticmethod
    def _handoff_body(text: str | None) -> str | None:
        if not text:
            return None
        match = _HANDOFF_RE.search(text)
        return match.group(1).strip() if match else None

    def _parse_json_payload(self, agent: AgentSpec, body: str) -> dict[str, Any]:
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            # Tolerate a fenced JSON block inside the handoff.
            fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", body, re.DOTALL)
            if fenced:
                try:
                    data = json.loads(fenced.group(1))
                except json.JSONDecodeError:
                    raise PayloadError(
                        f"agent {agent.id!r}: handoff is not valid JSON ({exc.msg})",
                        reason="invalid_json",
                    ) from exc
            else:
                raise PayloadError(
                    f"agent {agent.id!r}: handoff is not valid JSON ({exc.msg})",
                    reason="invalid_json",
                ) from exc
        if not isinstance(data, dict):
            raise PayloadError(
                f"agent {agent.id!r}: handoff must be a JSON object, got {type(data).__name__}",
                reason="not_an_object",
            )
        return self._check_fields(agent, data)

    def _parse_markdown_payload(self, agent: AgentSpec, body: str) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for line in body.splitlines():
            m = _MD_FIELD_RE.match(line)
            if not m:
                continue
            key, value = m.group(1), m.group(2).strip()
            if key in agent.contract.out_fields:
                data[key] = value
        if not data:
            raise PayloadError(
                f"agent {agent.id!r}: markdown handoff contains none of the expected fields "
                f"{list(agent.contract.out_fields)}",
                reason="no_fields",
            )
        return self._check_fields(agent, data)

    def _check_fields(self, agent: AgentSpec, data: dict[str, Any]) -> dict[str, Any]:
        missing = [f for f in agent.contract.out_fields if f not in data]
        if missing:
            raise PayloadError(
                f"agent {agent.id!r}: handoff is missing fields {missing}",
                missing=missing,
                reason="missing_fields",
            )
        return {k: v for k, v in data.items() if k in agent.contract.out_fields}

    # ------------------------------------------------------------------ #
    # delegation
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_delegate(raw: str) -> str | None:
        """Extract a requested delegate agent id from a raw message."""
        m = _DELEGATE_RE.search(raw or "")
        return m.group(1) if m else None

    def degrade_to_free_text(self, agent_id: str, raw: str) -> dict[str, Any]:
        """Fallback payload: keep the raw prose under a synthetic field."""
        agent = self.spec.agent(agent_id)
        return {f: raw.strip() for f in agent.contract.out_fields} | {"free_text": raw.strip()}

    # ------------------------------------------------------------------ #
    # statistics
    # ------------------------------------------------------------------ #

    def summary(self, state: RunState) -> dict[str, Any]:
        return {
            "cycles": state.cycle,
            "turns": state.turns,
            "n_calls": state.n_calls,
            "cost": state.cost,
            "parse_failures": state.parse_failures,
            "hops": [
                {
                    "hop_id": r.hop_id,
                    "source": r.source,
                    "target": r.target,
                    "fields": list(r.fields),
                    "n_turns": r.n_turns,
                    "parse_failures": r.parse_failures,
                    "degraded": r.degraded,
                    "error": r.error,
                }
                for r in state.hop_log
            ],
        }
