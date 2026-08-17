from __future__ import annotations

import sys
import types

import pytest


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_none_toolset_sentinel_validates_to_explicit_empty_list():
    from hermes_cli.oneshot import _validate_explicit_toolsets

    assert _validate_explicit_toolsets("none") == ([], None)


def test_none_toolset_sentinel_cannot_be_mixed_with_other_toolsets():
    from hermes_cli.oneshot import _validate_explicit_toolsets

    toolsets, error = _validate_explicit_toolsets("none,web")

    assert toolsets is None
    assert error is not None
    assert "none cannot be combined" in error


@pytest.mark.parametrize(
    ("toolsets", "expected_toolsets"),
    [([], []), (None, None)],
)
def test_run_agent_preserves_explicit_tool_selection_without_config_fallback(
    monkeypatch, toolsets, expected_toolsets
):
    import hermes_cli.oneshot as oneshot

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]

        def run_conversation(self, _prompt):
            return {"final_response": "repaired", "completed": True, "failed": False}

        def shutdown_memory_provider(self, *_args):
            return None

        def close(self):
            return None

    def config_toolsets_must_not_be_read(*_args, **_kwargs):
        raise AssertionError("configured toolsets leaked into an explicit no-tools run")

    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module(
            "hermes_cli.config",
            load_config=lambda: {"model": {"default": "openai-codex/gpt-5.6-terra"}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "primary-key",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "provider": "openai-codex",
                "requested_provider": "openai-codex",
                "api_mode": "codex_responses",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module("hermes_cli.tools_config", _get_platform_tools=config_toolsets_must_not_be_read),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.mcp_startup",
        _module(
            "hermes_cli.mcp_startup",
            ensure_mcp_discovery_before_agent_build=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)

    response, _result = oneshot._run_agent(
        "repair",
        model="openai-codex/gpt-5.6-terra",
        provider="openai-codex",
        toolsets=toolsets,
        use_config_toolsets=False,
    )

    assert response == "repaired"
    assert captured["enabled_toolsets"] == expected_toolsets
