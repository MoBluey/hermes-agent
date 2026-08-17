from __future__ import annotations

import sys
import types

import pytest

from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE


PRIMARY_PROVIDER = "openai-codex"
PRIMARY_MODEL = "openai-codex/gpt-5.6-terra"
FALLBACK_PROVIDER = "anthropic"
FALLBACK_MODEL = "anthropic/claude-sonnet-5"


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_parser_accepts_explicit_oneshot_fallback_pair():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(
        [
            "-z",
            "weekly task",
            "--provider",
            PRIMARY_PROVIDER,
            "--model",
            PRIMARY_MODEL,
            "--fallback-provider",
            FALLBACK_PROVIDER,
            "--fallback-model",
            FALLBACK_MODEL,
        ]
    )

    assert args.fallback_provider == FALLBACK_PROVIDER
    assert args.fallback_model == FALLBACK_MODEL


@pytest.mark.parametrize(
    ("fallback_provider", "fallback_model"),
    [(FALLBACK_PROVIDER, None), (None, FALLBACK_MODEL)],
)
def test_run_oneshot_rejects_unpaired_fallback_flags(
    monkeypatch, capsys, fallback_provider, fallback_model
):
    import hermes_cli.oneshot as oneshot

    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(oneshot, "_run_agent", fail_if_called)

    rc = oneshot.run_oneshot(
        "weekly task",
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
    )

    assert rc == 2
    assert called is False
    assert "--fallback-provider and --fallback-model must be provided together" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("primary_error", "expected_reason"),
    [
        (
            AuthError(
                "Codex quota exhausted",
                provider=PRIMARY_PROVIDER,
                code=CODEX_RATE_LIMITED_CODE,
            ),
            "rate_limit",
        ),
        (
            AuthError(
                "Codex login expired",
                provider=PRIMARY_PROVIDER,
                code="invalid_grant",
                relogin_required=True,
            ),
            "auth_permanent",
        ),
    ],
)
def test_pre_agent_auth_failure_uses_explicit_fallback_and_consumes_chain(
    monkeypatch, primary_error, expected_reason
):
    import hermes_cli.oneshot as oneshot

    resolve_calls = []
    captured = {}

    def resolve_runtime_provider(**kwargs):
        resolve_calls.append(kwargs)
        if len(resolve_calls) == 1:
            raise primary_error
        return {
            "api_key": "fallback-key",
            "base_url": "https://api.anthropic.com",
            "provider": FALLBACK_PROVIDER,
            "requested_provider": FALLBACK_PROVIDER,
            "api_mode": "anthropic_messages",
            "credential_pool": None,
        }

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            captured["instance"] = self
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, _prompt):
            return {
                "final_response": "recovered",
                "completed": True,
                "failed": False,
                "model": self.model,
                "provider": self.provider,
            }

        def shutdown_memory_provider(self, *_args):
            return None

        def close(self):
            return None

    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module(
            "hermes_cli.config",
            load_config=lambda: {
                "model": {"default": PRIMARY_MODEL},
                "fallback_model": {"provider": "openrouter", "model": "legacy/model"},
            },
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
        _module("hermes_cli.runtime_provider", resolve_runtime_provider=resolve_runtime_provider),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
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

    response, result = oneshot._run_agent(
        "weekly task",
        model=PRIMARY_MODEL,
        provider=PRIMARY_PROVIDER,
        fallback_model=FALLBACK_MODEL,
        fallback_provider=FALLBACK_PROVIDER,
    )

    assert response == "recovered"
    assert [call["requested"] for call in resolve_calls] == [PRIMARY_PROVIDER, FALLBACK_PROVIDER]
    assert resolve_calls[1]["target_model"] == FALLBACK_MODEL
    assert captured["fallback_model"] is None
    assert getattr(captured["instance"], "_fallback_activated", False) is False
    assert captured["model"] == FALLBACK_MODEL
    assert captured["provider"] == FALLBACK_PROVIDER
    assert result["model"] == FALLBACK_MODEL
    assert result["provider"] == FALLBACK_PROVIDER
    assert result["fallback_used"] is True
    assert result["fallback_reason"] == expected_reason


@pytest.mark.parametrize(
    "primary_error",
    [
        AuthError("invalid provider", code="invalid_provider"),
        AuthError(
            "invalid provider",
            code="invalid_provider",
            relogin_required=True,
        ),
        AuthError("no provider configured", code="no_provider_configured"),
        AuthError(
            "Primary API key missing",
            provider="anthropic",
            code="missing_api_key",
        ),
        AuthError(
            "Codex auth service unavailable",
            provider=PRIMARY_PROVIDER,
            code="codex_refresh_failed",
        ),
        AuthError(
            "Codex refresh failed without terminal credential evidence",
            provider=PRIMARY_PROVIDER,
            code="codex_refresh_failed",
            relogin_required=True,
        ),
        AuthError(
            "Codex configuration failure",
            provider=PRIMARY_PROVIDER,
            code="configuration_error",
            relogin_required=True,
        ),
        AuthError(
            "non-Codex login expired",
            provider="anthropic",
            relogin_required=True,
        ),
        AuthError(
            "non-Codex tier denied",
            provider="xai-oauth",
            code="xai_oauth_tier_denied",
        ),
        AuthError(
            "non-Codex model catalogue unavailable",
            provider="nous",
            code="models_fetch_failed",
        ),
        AuthError(
            "non-Codex rate limit",
            provider="anthropic",
            code=CODEX_RATE_LIMITED_CODE,
        ),
        AuthError("generic configuration failure"),
        AuthError(
            "provider configuration is malformed",
            provider=PRIMARY_PROVIDER,
            code="configuration_error",
        ),
    ],
)
def test_pre_agent_unapproved_auth_error_does_not_fallback(monkeypatch, primary_error):
    import hermes_cli.oneshot as oneshot

    calls = []

    def resolve_runtime_provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise primary_error
        raise AssertionError("fallback resolver must not be called")

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: {"model": {"default": PRIMARY_MODEL}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module("hermes_cli.runtime_provider", resolve_runtime_provider=resolve_runtime_provider),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
    )
    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=object))

    with pytest.raises(AuthError) as exc_info:
        oneshot._run_agent(
            "weekly task",
            model=PRIMARY_MODEL,
            provider=PRIMARY_PROVIDER,
            fallback_model=FALLBACK_MODEL,
            fallback_provider=FALLBACK_PROVIDER,
        )

    assert exc_info.value is primary_error
    assert len(calls) == 1


def test_pre_agent_non_auth_resolution_error_does_not_fallback(monkeypatch):
    import hermes_cli.oneshot as oneshot

    calls = []

    def resolve_runtime_provider(**kwargs):
        calls.append(kwargs)
        raise ValueError("unknown model")

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: {"model": {"default": PRIMARY_MODEL}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module("hermes_cli.runtime_provider", resolve_runtime_provider=resolve_runtime_provider),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
    )
    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=object))

    with pytest.raises(ValueError, match="unknown model"):
        oneshot._run_agent(
            "weekly task",
            model=PRIMARY_MODEL,
            provider=PRIMARY_PROVIDER,
            fallback_model=FALLBACK_MODEL,
            fallback_provider=FALLBACK_PROVIDER,
        )

    assert len(calls) == 1


def test_primary_success_uses_explicit_chain_instead_of_profile_chain(monkeypatch):
    import hermes_cli.oneshot as oneshot

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]

        def run_conversation(self, _prompt):
            return {"final_response": "primary", "completed": True, "failed": False}

        def shutdown_memory_provider(self, *_args):
            return None

        def close(self):
            return None

    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module(
            "hermes_cli.config",
            load_config=lambda: {
                "model": {"default": PRIMARY_MODEL},
                "fallback_providers": [
                    {"provider": "openrouter", "model": "legacy/model"}
                ],
            },
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
                "provider": PRIMARY_PROVIDER,
                "requested_provider": PRIMARY_PROVIDER,
                "api_mode": "codex_responses",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
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

    _response, result = oneshot._run_agent(
        "weekly task",
        model=PRIMARY_MODEL,
        provider=PRIMARY_PROVIDER,
        fallback_model=FALLBACK_MODEL,
        fallback_provider=FALLBACK_PROVIDER,
    )

    assert captured["fallback_model"] == [
        {
            "provider": FALLBACK_PROVIDER,
            "model": FALLBACK_MODEL,
            "_fallback_policy": "strict_oneshot",
        }
    ]
    assert result["fallback_used"] is False
    assert result["fallback_reason"] is None


def test_pre_agent_fallback_auth_failure_reraises_primary_error(monkeypatch):
    import hermes_cli.oneshot as oneshot

    primary_error = AuthError(
        "primary Codex login expired",
        provider=PRIMARY_PROVIDER,
        code="invalid_token",
        relogin_required=True,
    )
    fallback_error = AuthError("fallback key missing", provider=FALLBACK_PROVIDER)
    calls = []

    def resolve_runtime_provider(**kwargs):
        calls.append(kwargs)
        raise primary_error if len(calls) == 1 else fallback_error

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: {"model": {"default": PRIMARY_MODEL}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module("hermes_cli.runtime_provider", resolve_runtime_provider=resolve_runtime_provider),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
    )
    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=object))

    with pytest.raises(AuthError) as exc_info:
        oneshot._run_agent(
            "weekly task",
            model=PRIMARY_MODEL,
            provider=PRIMARY_PROVIDER,
            fallback_model=FALLBACK_MODEL,
            fallback_provider=FALLBACK_PROVIDER,
        )

    assert exc_info.value is primary_error
    assert exc_info.value.__cause__ is fallback_error
    assert len(calls) == 2


def test_pre_agent_fallback_configuration_error_preserves_primary_error(monkeypatch):
    import hermes_cli.oneshot as oneshot

    primary_error = AuthError(
        "primary Codex login expired",
        provider=PRIMARY_PROVIDER,
        code="invalid_token",
        relogin_required=True,
    )
    calls = []

    def resolve_runtime_provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise primary_error
        raise ValueError("fallback model is not configured")

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: {"model": {"default": PRIMARY_MODEL}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module("hermes_cli.runtime_provider", resolve_runtime_provider=resolve_runtime_provider),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
    )
    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=object))

    with pytest.raises(AuthError) as exc_info:
        oneshot._run_agent(
            "weekly task",
            model=PRIMARY_MODEL,
            provider=PRIMARY_PROVIDER,
            fallback_model=FALLBACK_MODEL,
            fallback_provider=FALLBACK_PROVIDER,
        )

    assert exc_info.value is primary_error
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "fallback model is not configured"
    assert len(calls) == 2


def test_pre_agent_fallback_agent_init_failure_preserves_primary_error(monkeypatch):
    import hermes_cli.oneshot as oneshot

    primary_error = AuthError(
        "primary Codex login expired",
        provider=PRIMARY_PROVIDER,
        code="invalid_token",
        relogin_required=True,
    )
    calls = []

    def resolve_runtime_provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise primary_error
        return {
            "api_key": "fallback-key",
            "base_url": "https://api.anthropic.com",
            "provider": FALLBACK_PROVIDER,
            "requested_provider": FALLBACK_PROVIDER,
            "api_mode": "anthropic_messages",
            "credential_pool": None,
        }

    class FailingAgent:
        def __init__(self, **_kwargs):
            raise ValueError("fallback client setup failed")

    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=FailingAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: {"model": {"default": PRIMARY_MODEL}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module("hermes_cli.runtime_provider", resolve_runtime_provider=resolve_runtime_provider),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
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

    with pytest.raises(AuthError) as exc_info:
        oneshot._run_agent(
            "weekly task",
            model=PRIMARY_MODEL,
            provider=PRIMARY_PROVIDER,
            fallback_model=FALLBACK_MODEL,
            fallback_provider=FALLBACK_PROVIDER,
        )

    assert exc_info.value is primary_error
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "fallback client setup failed"
    assert len(calls) == 2


def test_run_and_exit_oneshot_forwards_explicit_fallback(monkeypatch):
    import hermes_cli.main as main_mod
    import hermes_cli.oneshot as oneshot

    captured = {}

    def fake_run_oneshot(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(oneshot, "run_oneshot", fake_run_oneshot)
    monkeypatch.setattr(main_mod, "_cleanup_oneshot_runtime", lambda: None)
    monkeypatch.setattr(
        main_mod,
        "_exit_after_oneshot",
        lambda rc: (_ for _ in ()).throw(SystemExit(rc)),
    )

    with pytest.raises(SystemExit) as exc_info:
        main_mod._run_and_exit_oneshot(
            "weekly task",
            model=PRIMARY_MODEL,
            provider=PRIMARY_PROVIDER,
            fallback_model=FALLBACK_MODEL,
            fallback_provider=FALLBACK_PROVIDER,
        )

    assert exc_info.value.code == 0
    assert captured["fallback_model"] == FALLBACK_MODEL
    assert captured["fallback_provider"] == FALLBACK_PROVIDER


@pytest.mark.parametrize(
    ("launcher_name", "launcher_patches"),
    [
        (
            "_try_fast_chat_launch",
            {
                "_wants_tui_early": lambda _argv: False,
            },
        ),
        (
            "_try_termux_fast_cli_launch",
            {
                "_is_termux_startup_environment": lambda: True,
                "_wants_tui_early": lambda _argv: False,
                "_is_termux_fast_version_argv": lambda _argv: False,
            },
        ),
    ],
)
def test_fast_oneshot_dispatches_forward_explicit_fallback(
    monkeypatch, launcher_name, launcher_patches
):
    import hermes_cli.config as config_mod
    import hermes_cli.main as main_mod

    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "-z",
            "weekly task",
            "--provider",
            PRIMARY_PROVIDER,
            "--model",
            PRIMARY_MODEL,
            "--fallback-provider",
            FALLBACK_PROVIDER,
            "--fallback-model",
            FALLBACK_MODEL,
        ],
    )
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: None)
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda _args: None)
    monkeypatch.setattr(main_mod, "_confirm_startup_expensive_model_override", lambda _args: None)
    for name, replacement in launcher_patches.items():
        monkeypatch.setattr(main_mod, name, replacement)

    def capture_dispatch(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_run_and_exit_oneshot", capture_dispatch)

    with pytest.raises(SystemExit) as exc_info:
        getattr(main_mod, launcher_name)()

    assert exc_info.value.code == 0
    assert captured["fallback_model"] == FALLBACK_MODEL
    assert captured["fallback_provider"] == FALLBACK_PROVIDER
