"""The adapter: spec -> mini-swe-agent config, and the C-4 expansion check."""

from __future__ import annotations

import pytest

from rsi.adapter import (
    FROZEN_OBSERVATION_TEMPLATE,
    RHI_AGENT_CLASS,
    TOLERANT_ACTION_REGEX,
    assert_full_expansion,
    default_base_config,
    missing_keys,
    to_harbor_agent_kwargs,
    to_mini_config,
)
from rsi.schema import SpecError


class TestBaseConfig:
    def test_the_placeholder_templates_are_used(self):
        base = default_base_config()
        assert base["agent"]["system_template"] == "__RHI_RENDERED__"
        assert base["agent"]["instance_template"] == "__RHI_RENDERED__"

    def test_the_text_protocol_is_the_default(self):
        # The text protocol is the only one that works inside a Harbor task
        # container: litellm imports its MCP handler as soon as `tools` is set.
        base = default_base_config()
        assert base["model"]["model_class"] == "litellm_textbased"
        assert base["model"]["action_regex"] == TOLERANT_ACTION_REGEX

    def test_the_toolcall_protocol_has_no_action_regex(self):
        base = default_base_config(action_protocol="toolcall")
        assert base["model"]["model_class"] == "litellm"
        assert "action_regex" not in base["model"]

    def test_the_text_protocol_uses_the_tolerant_regex(self):
        base = default_base_config(action_protocol="text")
        assert base["model"]["model_class"] == "litellm_textbased"
        assert base["model"]["action_regex"] == TOLERANT_ACTION_REGEX
        # The default fence of mini-swe-agent must not leak back in.
        assert "mswea_bash_command" not in base["model"]["action_regex"].replace(
            "(?:mswea_bash_command|bash|sh|shell)", ""
        )

    def test_an_unknown_protocol_is_rejected(self):
        with pytest.raises(ValueError, match="action_protocol"):
            default_base_config(action_protocol="smoke-signals")

    def test_the_endpoint_settings_are_forwarded(self):
        base = default_base_config(
            api_base="http://example:8000/v1",
            api_key="EMPTY",
            reasoning_effort="low",
        )
        kwargs = base["model"]["model_kwargs"]
        assert kwargs["api_base"] == "http://example:8000/v1"
        assert kwargs["api_key"] == "EMPTY"
        assert kwargs["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "low"


class TestMiniConfig:
    def test_the_agent_class_points_at_the_rhi_agent(self, spec):
        config = to_mini_config(spec)
        assert config["agent"]["agent_class"] == RHI_AGENT_CLASS

    def test_the_runtime_knobs_are_injected(self, spec):
        config = to_mini_config(spec)
        assert config["model"]["model_kwargs"]["seed"] == spec.runtime.seed
        assert config["model"]["model_kwargs"]["temperature"] == spec.runtime.temperature
        assert config["agent"]["max_hops"] == spec.runtime.max_hops
        assert config["agent"]["observation_max_chars"] == spec.runtime.observation_max_chars

    def test_the_base_config_is_fully_expanded(self, spec):
        base = default_base_config()
        config = to_mini_config(spec, base)
        assert missing_keys(config, base) == []

    def test_a_missing_base_key_is_detected(self, spec):
        base = default_base_config()
        # The base knows about a key the produced config does not have.
        extended = {**base, "model": {**base["model"], "unregistered_option": 1}}
        with pytest.raises(SpecError, match="does not fully expand"):
            assert_full_expansion(to_mini_config(spec, base), extended)

    def test_the_spec_path_is_recorded(self, spec, tmp_path):
        config = to_mini_config(spec, spec_path=tmp_path / "gen00.yaml")
        assert config["agent"]["spec_path"].endswith("gen00.yaml")

    def test_the_harbor_kwargs_carry_the_config(self, spec):
        kwargs = to_harbor_agent_kwargs(spec)
        assert set(kwargs) == {"config"}
        assert kwargs["config"]["agent"]["agent_class"] == RHI_AGENT_CLASS

    def test_the_observation_template_reads_the_configured_limit(self):
        # The template is frozen, but the limit it uses comes from the config.
        assert "observation_max_chars" in FROZEN_OBSERVATION_TEMPLATE
