"""``RhiAgent`` -- a multi-agent, contract-driven harness on top of mini-swe-agent.

mini-swe-agent's ``DefaultAgent`` is a single-agent loop: one system prompt,
one message list, one action per turn.  RHI argues that the *harness* -- the
information interface between an orchestrator and its subagents -- is the
lever that matters.  This module implements that harness as a subclass:

* every agent gets its own message list, so a subagent never sees the
  orchestrator's (or another subagent's) transcript;
* the context handed to a subagent is filtered through its contract, so only
  the fields it declared in ``in_fields`` reach it;
* the orchestrator receives structured payloads, not raw trajectories.

Everything the agent does is derived from a :class:`~rsi.schema.HarnessSpec`,
which is what the optimizer searches over.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, Submitted

from .hop_runner import HopRunner, PayloadError, RunState
from .schema import FailurePolicy, HarnessSpec, Hop, load_spec

__all__ = ["RhiConfig", "RhiAgent"]

logger = logging.getLogger("rsi.agent")

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

#: Placeholder written into the mini-swe-agent templates.  The templates are
#: never rendered by mini itself -- RhiAgent owns the prompts (trap T-6).
RHI_TEMPLATE_PLACEHOLDER = "__RHI_RENDERED__"


class RhiConfig(AgentConfig):
    """mini-swe-agent config for the RHI agent.

    ``system_template`` / ``instance_template`` keep their placeholder values so
    that mini's own template machinery can never accidentally render a prompt
    that bypasses the contract logic.
    """

    system_template: str = RHI_TEMPLATE_PLACEHOLDER
    instance_template: str = RHI_TEMPLATE_PLACEHOLDER
    spec_path: Path = Path("harness/gen00.yaml")
    #: The spec, inlined.  The agent runs *inside the task container*, where
    #: the host's ``spec_path`` does not exist -- so the spec has to travel
    #: with the config (which Harbor writes to /tmp/mswea-config/custom.yaml
    #: inside the container).  ``spec_path`` stays as the fallback for local
    #: runs.
    spec: dict[str, Any] | None = None
    max_hops: int = 6
    hop_context_max_chars: int = 6000
    observation_max_chars: int = 10000
    context_keep_recent: int = 20
    seed: int = 1234
    temperature: float = 0.0


class RhiAgent(DefaultAgent):
    """Multi-agent, contract-driven agent."""

    def __init__(self, model, env, *, config_class: type = RhiConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        inline = getattr(self.config, "spec", None)
        if inline:
            self.spec: HarnessSpec = HarnessSpec.from_dict(inline)
        else:
            self.spec = load_spec(Path(self.config.spec_path))
        self.action_protocol = self._detect_action_protocol()
        self.hop_runner = HopRunner(self.spec, action_protocol=self.action_protocol)
        self.state: RunState | None = None
        self._agent_messages: dict[str, list[dict]] = {}
        self._order: list[tuple[str, dict]] = []
        self._agent_models: dict[str, Any] = {}
        self._current_hop: Hop | None = None
        self._retry_used: set[str] = set()
        self._hop_turns: dict[str, int] = {}

    def _detect_action_protocol(self) -> str:
        """Which action protocol the model speaks.

        mini-swe-agent's text-based model class is configured with an
        ``action_regex``; the tool-call class has none.  The rendered prompts
        have to describe the protocol the model actually uses, otherwise the
        agent asks for a code block from a model that only emits tool calls
        (or the other way round).

        The tool-call protocol additionally needs the ``mcp`` package:
        litellm imports its MCP handler as soon as ``tools`` is set, and
        mini-swe-agent always sets it.  Harbor installs mini-swe-agent with
        ``uv tool install mini-swe-agent --with litellm --with orjson
        --with fastapi`` -- without ``mcp`` -- so inside a task container the
        tool-call protocol cannot work.  The check below turns that into a
        clear error instead of an hour of exponential-backoff retries.
        """
        config = getattr(self.model, "config", None)
        regex = getattr(config, "action_regex", None) if config is not None else None
        protocol = "text" if regex else "toolcall"
        if protocol == "toolcall":
            import importlib.util

            if importlib.util.find_spec("mcp") is None:
                raise RuntimeError(
                    "the tool-call protocol needs the 'mcp' package: litellm imports "
                    "its MCP handler as soon as `tools` is set, and mini-swe-agent "
                    "always sets it. Install it (`pip install mcp`) or use the "
                    "text protocol (--action-protocol text), which has no such "
                    "dependency and is the only one that works inside a Harbor "
                    "task container, where mini-swe-agent is installed without "
                    "the mcp extra."
                )
        return protocol

    # ------------------------------------------------------------------ #
    # model handling
    # ------------------------------------------------------------------ #

    def _model_for(self, agent_id: str):
        """Return the model instance that serves ``agent_id``.

        The orchestrator uses the model that mini-swe-agent handed us; a
        subagent may use a different one (the explicit "break the
        homogenization" knob of the plan).
        """
        if agent_id in self._agent_models:
            return self._agent_models[agent_id]
        agent = self.spec.agent(agent_id)
        primary = self.spec.agent(self.hop_runner.orchestrator)
        if agent.model == primary.model and agent.reasoning_effort == primary.reasoning_effort:
            self._agent_models[agent_id] = self.model
            return self.model
        try:
            from minisweagent.models import get_model
        except ImportError:  # pragma: no cover - mini-swe-agent is a hard dependency
            self._agent_models[agent_id] = self.model
            return self.model
        base_config = dict(getattr(self.model.config, "model_dump", dict)())
        base_config.pop("model_name", None)
        kwargs = {
            **base_config,
            "model_name": agent.model,
            "model_kwargs": {
                **dict(base_config.get("model_kwargs", {})),
                "temperature": self.spec.runtime.temperature,
                "seed": self.spec.runtime.seed,
            },
        }
        self._agent_models[agent_id] = get_model(config=kwargs)
        return self._agent_models[agent_id]

    # ------------------------------------------------------------------ #
    # message bookkeeping
    # ------------------------------------------------------------------ #

    def _init_agents(self, task: str) -> None:
        self._agent_messages = {a.id: [] for a in self.spec.agents}
        self._order = []
        orchestrator = self.hop_runner.orchestrator
        agent = self.spec.agent(orchestrator)
        self._agent_messages[orchestrator] = [
            self.model.format_message(role="system", content=self.hop_runner.render_system(orchestrator)),
            self.model.format_message(role="user", content=self.hop_runner.assemble_context(orchestrator, self.state)),
        ]
        for msg in self._agent_messages[orchestrator]:
            self._order.append((orchestrator, msg))
        self._refresh_transcript()

    def _refresh_transcript(self) -> None:
        """Flatten the per-agent conversations into one chronological trajectory."""
        self.messages = []
        for agent_id, msg in self._order:
            item = dict(msg)
            item.setdefault("extra", {})
            item["extra"] = {**item["extra"], "rhi_agent_id": agent_id}
            self.messages.append(item)

    def _trim(self, agent_id: str) -> None:
        """Keep the context bounded: only the last N turns stay verbatim.

        This is the ``runtime.context_keep_recent`` lever, and it is what keeps
        a long orchestrator trajectory from drowning the contract payload.
        """
        keep = self.spec.runtime.context_keep_recent
        messages = self._agent_messages[agent_id]
        if keep <= 0 or len(messages) <= 2 * keep + 2:
            return
        head = messages[:2]
        tail = messages[-(2 * keep) :]
        notice = {
            "role": "user",
            "content": f"[{len(messages) - len(head) - len(tail)} earlier messages elided by the harness]",
        }
        self._agent_messages[agent_id] = head + [notice] + tail

    # ------------------------------------------------------------------ #
    # one turn
    # ------------------------------------------------------------------ #

    def _turn(self, agent_id: str) -> dict:
        """Run exactly one LLM turn (query + action execution) for ``agent_id``."""
        model = self._model_for(agent_id)
        messages = self._agent_messages[agent_id]

        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )

        self.n_calls += 1
        message = model.query(messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        messages.append(message)
        self._order.append((agent_id, message))
        self.state.n_calls = self.n_calls
        self.state.cost = self.cost

        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            action = self._guard_action(agent_id, action)
            outputs.append(self.env.execute(action))
        observations = model.format_observation_messages(message, outputs, self.get_template_vars())
        messages.extend(observations)
        for obs in observations:
            self._order.append((agent_id, obs))
        self._trim(agent_id)
        return message

    def _guard_action(self, agent_id: str, action: dict) -> dict:
        """Enforce the two structural rules of the harness.

        1. Only the orchestrator may submit the final answer.  A subagent that
           tries gets a harmless no-op instead, and the hop runner asks it for
           a proper handoff.
        2. The orchestrator may only submit after it has delegated at least
           ``runtime.min_delegations`` times.  Without this the model happily
           does the whole task itself and the contract/hop dimensions of the
           harness are never exercised -- a multi-agent harness that never
           delegates is a single-agent harness in disguise.  Setting
           ``min_delegations: 0`` turns the rule off, which is exactly the
           single-agent ablation.
        """
        command = str(action.get("command", ""))
        if agent_id != self.hop_runner.orchestrator:
            if SUBMIT_MARKER in command:
                logger.info("subagent %s tried to submit; converting to a handoff request", agent_id)
                return {
                    "command": "echo 'HANDOFF_REQUIRED: emit a <handoff> block with your fields'"
                }
            return action

        min_delegations = self.spec.runtime.min_delegations
        if SUBMIT_MARKER in command and self.state is not None:
            if self.state.delegations < min_delegations:
                agents = ", ".join(a.id for a in self.spec.agents if a.id != agent_id)
                logger.info(
                    "orchestrator tried to submit after %d delegation(s), %d required",
                    self.state.delegations,
                    min_delegations,
                )
                return {
                    "command": (
                        "echo 'DELEGATION_REQUIRED: you must delegate the work to a subagent "
                        f"before finishing. Available agents: {agents}. Add "
                        "<delegate>agent_id</delegate> to your response.'"
                    )
                }
        return action

    # ------------------------------------------------------------------ #
    # the run loop
    # ------------------------------------------------------------------ #

    def run(self, task: str = "", **kwargs) -> dict:
        """Run the multi-hop loop until the harness is done."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        orchestrator = self.hop_runner.orchestrator
        self.state = RunState(task=task, current_agent=orchestrator)
        self._current_hop = None
        self._retry_used = set()
        self._hop_turns = {}
        self._init_agents(task)

        current = orchestrator
        try:
            while not self.state.finished:
                if self.state.turns >= self.spec.runtime.max_iterations:
                    self.state.finished = True
                    self.state.finish_reason = "max_iterations"
                    break
                if self.state.cycle >= self.spec.runtime.max_hops:
                    self.state.finished = True
                    self.state.finish_reason = "max_hops"
                    break

                try:
                    message = self._turn(current)
                except FormatError as exc:
                    self.cost += exc.messages[0].get("extra", {}).get("cost", 0.0)
                    self._agent_messages[current].extend(exc.messages)
                    self.n_consecutive_format_errors += 1
                    if (
                        0
                        < self.config.max_consecutive_format_errors
                        <= self.n_consecutive_format_errors
                    ):
                        self._agent_messages[current].append(
                            {
                                "role": "exit",
                                "content": "RepeatedFormatError",
                                "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                            }
                        )
                        self.state.finished = True
                        self.state.finish_reason = "RepeatedFormatError"
                    continue
                except InterruptAgentFlow as exc:
                    self._agent_messages[current].extend(exc.messages)
                    self.state.finished = True
                    self.state.finish_reason = exc.messages[-1].get("extra", {}).get(
                        "exit_status", "Interrupted"
                    )
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    self._agent_messages[current].append(
                        self.model.format_message(
                            role="exit",
                            content=str(exc),
                            extra={
                                "exit_status": type(exc).__name__,
                                "submission": "",
                                "exception_str": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                        )
                    )
                    self.state.finished = True
                    self.state.finish_reason = type(exc).__name__
                    break

                self.n_consecutive_format_errors = 0
                self.state.turns += 1

                raw = message.get("content", "") or ""
                if current == orchestrator:
                    next_agent, hop = self._orchestrator_transfer(current, raw)
                else:
                    next_agent, hop = self._subagent_transfer(
                        current, raw, message.get("extra", {}).get("actions")
                    )

                if next_agent is None:
                    self._refresh_transcript()
                    self.save(self.config.output_path)
                    continue

                self.state.current_agent = next_agent
                current = next_agent
                if hop is not None:
                    self._current_hop = hop
                    self._enter_agent(current, hop)
                    self._hop_turns[hop.id] = self._hop_turns.get(hop.id, 0) + 1
                self._refresh_transcript()
                self.save(self.config.output_path)
        finally:
            self._refresh_transcript()
            self.save(self.config.output_path)

        if self.messages and self.messages[-1].get("role") == "exit":
            return self.messages[-1].get("extra", {})
        return {
            "exit_status": self.state.finish_reason or "Completed",
            "submission": self._submission(),
        }

    def _enter_agent(self, agent_id: str, hop: Hop) -> None:
        """Set up the message list of an agent that is about to take over."""
        if self._agent_messages.get(agent_id):
            return
        self._agent_messages[agent_id] = [
            self.model.format_message(
                role="system", content=self.hop_runner.render_system(agent_id, hop)
            ),
            self.model.format_message(
                role="user",
                content=self.hop_runner.assemble_context(agent_id, self.state, hop),
            ),
        ]
        for msg in self._agent_messages[agent_id]:
            self._order.append((agent_id, msg))

    def _orchestrator_transfer(self, current: str, raw: str) -> tuple[str | None, Hop | None]:
        """The orchestrator keeps acting until it delegates to a subagent."""
        delegate = self.hop_runner.parse_delegate(raw)
        if not delegate or delegate == current:
            return None, None
        edge = self.spec.edge(current, delegate)
        if edge is None:
            logger.warning("orchestrator delegated to unknown agent %r; ignoring", delegate)
            return None, None
        self.state.cycle += 1
        self.state.hops += 1
        self.state.delegations += 1
        return delegate, edge

    def _subagent_transfer(
        self, current: str, raw: str, actions: Sequence[dict] | None = None
    ) -> tuple[str | None, Hop | None]:
        """A subagent hands its payload back and control returns to the orchestrator."""
        try:
            payload = self.hop_runner.extract_payload(current, raw, actions)
            self.state.record_payload(current, payload)
            self._record_hop(self._current_hop, current, error=None, degraded=False)
        except PayloadError as exc:
            self.state.parse_failures += 1
            policy = self._failure_policy()
            logger.info("payload error on agent %s: %s (policy=%s)", current, exc, policy)
            if policy is FailurePolicy.RETRY_ONCE and current not in self._retry_used:
                self._retry_used.add(current)
                self._agent_messages[current].append(
                    {
                        "role": "user",
                        "content": (
                            f"Your handoff could not be parsed: {exc}. "
                            "Reply again with exactly the required fields inside a single "
                            "<handoff> block, together with your one action."
                        ),
                    }
                )
                self._record_hop(self._current_hop, current, error=str(exc), degraded=False)
                return current, None
            if policy is FailurePolicy.ABORT:
                self._record_hop(self._current_hop, current, error=str(exc), degraded=False)
                self.state.finished = True
                self.state.finish_reason = "PayloadAbort"
                return None, None
            payload = self.hop_runner.degrade_to_free_text(current, raw)
            self.state.record_payload(current, payload)
            self._record_hop(self._current_hop, current, error=str(exc), degraded=True)

        transfer = self.hop_runner.transfer_target(current, self.state)
        if transfer is None:
            self.state.finished = True
            self.state.finish_reason = "workflow_exhausted"
            return None, None
        self.state.hops += 1
        if transfer.target == self.hop_runner.orchestrator:
            self.state.cycle_payloads = dict(self.state.payloads)
        return transfer.target, transfer

    def _submission(self) -> str:
        for msg in reversed(self.messages):
            extra = msg.get("extra", {})
            if extra.get("submission"):
                return extra["submission"]
        return ""

    def _failure_policy(self) -> FailurePolicy:
        policies = self.spec.auxiliary_rules.failure_fallback
        for name in policies:
            try:
                return FailurePolicy(name)
            except ValueError:
                continue
        return FailurePolicy.RETRY_ONCE

    def _record_hop(
        self, hop: Hop | None, agent_id: str, *, error: str | None, degraded: bool
    ) -> None:
        from .hop_runner import HopRecord

        self.state.hop_log.append(
            HopRecord(
                hop_id=hop.id if hop else "<none>",
                source=hop.source if hop else "<none>",
                target=hop.target if hop else agent_id,
                fields=hop.pass_fields if hop else (),
                n_turns=self._hop_turns.get(hop.id, 0) if hop else 0,
                parse_failures=1 if error else 0,
                degraded=degraded,
                error=error,
            )
        )

    # ------------------------------------------------------------------ #
    # misc
    # ------------------------------------------------------------------ #

    def step(self) -> list[dict]:
        """Run a single turn of the current agent (mini-swe-agent contract)."""
        if self.state is None:
            self.state = RunState(task="", current_agent=self.hop_runner.orchestrator)
            self._init_agents("")
        agent_id = self.state.current_agent
        message = self._turn(agent_id)
        self.state.turns += 1
        self._refresh_transcript()
        return [message]

    def serialize(self, *extra_dicts) -> dict:
        data = super().serialize(*extra_dicts)
        if self.state is not None:
            data["rhi"] = {
                **self.hop_runner.summary(self.state),
                "spec_sha256": self.spec.spec_sha256(),
                "orchestrator": self.hop_runner.orchestrator,
                "finish_reason": self.state.finish_reason,
            }
        return data
