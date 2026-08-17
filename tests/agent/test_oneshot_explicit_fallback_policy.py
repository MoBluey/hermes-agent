from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import (
    _fallback_allowed_for_entry,
    try_activate_fallback,
)
from agent.error_classifier import ClassifiedError, FailoverReason, classify_api_error


STRICT_ENTRY = {
    "provider": "anthropic",
    "model": "anthropic/claude-sonnet-5",
    "_fallback_policy": "strict_oneshot",
}


@pytest.mark.parametrize(
    "reason",
    [
        FailoverReason.auth,
        FailoverReason.auth_permanent,
        FailoverReason.billing,
        FailoverReason.rate_limit,
        FailoverReason.upstream_rate_limit,
        FailoverReason.overloaded,
        FailoverReason.timeout,
    ],
)
def test_strict_oneshot_fallback_allows_only_approved_classifications(reason):
    classified = ClassifiedError(reason=reason)

    assert _fallback_allowed_for_entry(STRICT_ENTRY, classified) is True


@pytest.mark.parametrize(
    "reason",
    [
        FailoverReason.context_overflow,
        FailoverReason.payload_too_large,
        FailoverReason.image_too_large,
        FailoverReason.model_not_found,
        FailoverReason.provider_policy_blocked,
        FailoverReason.content_policy_blocked,
        FailoverReason.format_error,
        FailoverReason.invalid_encrypted_content,
        FailoverReason.multimodal_tool_content_unsupported,
        FailoverReason.thinking_signature,
        FailoverReason.long_context_tier,
        FailoverReason.oauth_long_context_beta_forbidden,
        FailoverReason.llama_cpp_grammar_pattern,
        FailoverReason.ssl_cert_verification,
        FailoverReason.unknown,
    ],
)
def test_strict_oneshot_fallback_denies_non_approved_classifications(reason):
    classified = ClassifiedError(reason=reason)

    assert _fallback_allowed_for_entry(STRICT_ENTRY, classified) is False


def test_strict_oneshot_fallback_denies_missing_classification():
    assert _fallback_allowed_for_entry(STRICT_ENTRY, None) is False


def test_strict_oneshot_fallback_requires_verified_http_5xx_for_server_error():
    unverified = ClassifiedError(reason=FailoverReason.server_error)
    verified = ClassifiedError(
        reason=FailoverReason.server_error,
        status_code=500,
        error_context={"verified_http_5xx": True},
    )

    assert _fallback_allowed_for_entry(STRICT_ENTRY, unverified) is False
    assert _fallback_allowed_for_entry(STRICT_ENTRY, verified) is True


@pytest.mark.parametrize(
    "error_context",
    [
        {"billing_unverified": True},
        {"possible_content_filter": True},
    ],
)
def test_strict_oneshot_fallback_denies_ambiguous_billing(error_context):
    classified = ClassifiedError(reason=FailoverReason.billing, error_context=error_context)

    assert _fallback_allowed_for_entry(STRICT_ENTRY, classified) is False


@pytest.mark.parametrize("status_code", [401, 402, 403, 404, 429, 500, 503, 504])
def test_ambiguous_billing_is_tagged_and_denied_across_statuses(status_code):
    error = RuntimeError(
        "You're out of extra usage. Add more at claude.ai/settings/usage and keep going."
    )
    error.status_code = status_code

    classified = classify_api_error(error, provider="anthropic")

    assert classified.reason is FailoverReason.billing
    assert classified.error_context.get("billing_unverified") is True
    assert classified.error_context.get("possible_content_filter") is True
    assert _fallback_allowed_for_entry(STRICT_ENTRY, classified) is False


@pytest.mark.parametrize(
    ("message", "status_code", "expected_reason"),
    [
        ("Invalid API key", 401, FailoverReason.auth),
        ("Payment required", 402, FailoverReason.billing),
        ("Too many requests", 429, FailoverReason.rate_limit),
        ("Internal server error", 500, FailoverReason.server_error),
        ("Service unavailable", 503, FailoverReason.overloaded),
    ],
)
def test_confirmed_provider_failures_remain_eligible(
    message, status_code, expected_reason
):
    error = RuntimeError(message)
    error.status_code = status_code

    classified = classify_api_error(error)

    assert classified.reason is expected_reason
    assert _fallback_allowed_for_entry(STRICT_ENTRY, classified) is True


@pytest.mark.parametrize("status_code", [None, 400, 500, 503])
def test_empty_provider_response_server_errors_are_tagged_and_denied(status_code):
    error = RuntimeError("provider returned an empty response despite retries")
    if status_code is not None:
        error.status_code = status_code

    classified = classify_api_error(error)

    assert classified.reason is FailoverReason.server_error
    assert classified.error_context.get("empty_provider_response") is True
    assert _fallback_allowed_for_entry(STRICT_ENTRY, classified) is False


def test_empty_stream_error_type_is_denied_even_with_opaque_message():
    from agent.errors import EmptyStreamError

    classified = classify_api_error(EmptyStreamError("unknown {}"))

    assert classified.error_context.get("empty_provider_response") is True
    assert _fallback_allowed_for_entry(STRICT_ENTRY, classified) is False


@pytest.mark.parametrize(
    "classified",
    [
        ClassifiedError(reason=FailoverReason.content_policy_blocked),
        ClassifiedError(
            reason=FailoverReason.billing,
            error_context={"billing_unverified": True},
        ),
    ],
)
def test_denied_strict_entry_does_not_advance_or_arm_cooldown(classified):
    agent = SimpleNamespace(
        _fallback_chain=[STRICT_ENTRY.copy()],
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _primary_runtime={"provider": "openai-codex"},
        provider="openai-codex",
    )
    assert try_activate_fallback(
        agent,
        reason=classified.reason,
        classified=classified,
    ) is False
    assert agent._fallback_index == 0
    assert agent._rate_limited_until == 0


def test_legacy_profile_entry_preserves_broad_fallback_policy():
    legacy_entry = {"provider": "anthropic", "model": "claude-sonnet"}
    classified = ClassifiedError(reason=FailoverReason.content_policy_blocked)

    assert _fallback_allowed_for_entry(legacy_entry, classified) is True
