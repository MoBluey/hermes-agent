"""Local-model stability guards (Ornith / Qwen on chat_completions).

* Visible announced-action stall: a reply ending "Let me write the script first, then run it." with
  no tool call re-prompts when intent-ack continuation is enabled for the model.
* Empty / no-op terminal and execute_code payloads are rejected before dispatch.
* Tool-loop halt recovery: the first halt blocks the halted call and grants one corrective
  iteration; a second halt in the same turn ends it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import visible_reply_announces_action
from agent.conversation_loop import _CODEX_ACK_CONTINUATION_NUDGE
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController
from agent.turn_tool_round import _GUARDRAIL_RECOVERY_NOTE


# ---------------------------------------------------------------------------
# Task 2: visible announced-action detector
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Let me write the parse script to a file first, then run it.",
    "I found the config. Now I need to check the gateway log.",
    "The build failed on step 3. Let me look at the stack trace:",
    "I'll run the migration and verify the row counts.",
    "Next, I'll patch the router config.",
])
def test_visible_detector_catches_announced_actions(text):
    assert visible_reply_announces_action(text)


@pytest.mark.parametrize("text", [
    "The answer is 42.",
    "Done - the file is saved at /tmp/x.txt. Let me know if you want changes.",
    "I'll wait for your confirmation before deleting anything.",
    "Fixed the bug in parser.py; all 12 tests pass.",
    "I can run the migration if you want.",
    "Should I proceed?",
    "Let me know if anything else breaks.",
    "I'll proceed once you confirm the branch name.",
    "Summary:\n- A\n- B\nI'll be here if you need more.",
    "I need to note that the backup is from yesterday.",
    "",
])
def test_visible_detector_ignores_final_answers(text):
    assert not visible_reply_announces_action(text)


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_degenerate_final_recovery)."""
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.model = "ornith-1.5-35b-jang-6m"
        agent.valid_tool_names = {"terminal", "web_search"}
        return agent


def _run(agent, stages, *, tool_result="ok"):
    agent.client.chat.completions.create.side_effect = stages
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call", return_value=tool_result) as dispatch,
    ):
        return agent.run_conversation("parse the export"), dispatch


def _tool_round(call_id, name="web_search", arguments='{"query": "x"}'):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call
    return _mock_response(
        content="", finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(name=name, arguments=arguments, call_id=call_id)],
    )


def _final(text):
    from tests.agent.test_run_agent import _mock_response
    return _mock_response(content=text, finish_reason="stop")


_STALL = "Let me write the parse script to a file first, then run it."


def test_visible_stall_reprompts_when_enabled_for_model(loop_agent):
    loop_agent._intent_ack_continuation = ["ornith", "qwen"]

    result, _ = _run(loop_agent, [_final(_STALL), _final("Parsed 120 rows.")])

    assert result["final_response"] == "Parsed 120 rows."
    assert loop_agent.client.chat.completions.create.call_count == 2
    contents = [m.get("content") for m in result["messages"]]
    assert contents.index(_STALL) + 1 == contents.index(_CODEX_ACK_CONTINUATION_NUDGE)


def test_visible_stall_not_reprompted_for_unlisted_model(loop_agent):
    loop_agent._intent_ack_continuation = ["qwen"]  # ornith not opted in

    result, _ = _run(loop_agent, [_final(_STALL)])

    assert result["final_response"] == _STALL
    assert loop_agent.client.chat.completions.create.call_count == 1


def test_visible_stall_nudges_are_capped_at_two(loop_agent):
    loop_agent._intent_ack_continuation = True

    result, _ = _run(loop_agent, [_final(_STALL), _final(_STALL), _final(_STALL)])

    assert loop_agent.client.chat.completions.create.call_count == 3
    assert result["final_response"] == _STALL


def test_genuine_final_answer_is_not_reprompted(loop_agent):
    loop_agent._intent_ack_continuation = ["ornith"]

    result, _ = _run(loop_agent, [_final("Parsed 120 rows. Let me know if you want a CSV.")])

    assert loop_agent.client.chat.completions.create.call_count == 1
    assert result["final_response"] == "Parsed 120 rows. Let me know if you want a CSV."


# ---------------------------------------------------------------------------
# Task 3: empty / no-op terminal and execute_code payloads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("tool", "args"), [
    ("terminal", {"command": ":"}),
    ("terminal", {"command": "  true "}),
    ("terminal", {"command": ""}),
    ("terminal", {"command": "   "}),
    ("terminal", {}),
    ("execute_code", {"code": ""}),
    ("execute_code", {"code": "\n  \n"}),
    ("execute_code", {"code": "pass"}),
])
def test_noop_payload_is_rejected_before_dispatch(tool, args):
    import model_tools

    with patch.object(model_tools, "_execute_tool") as execute:
        result = model_tools.handle_function_call(tool, dict(args), task_id="t", skip_pre_tool_call_hook=True)

    execute.assert_not_called()
    error = json.loads(result)["error"]
    assert "Re-emit the complete" in error
    assert tool in error


@pytest.mark.parametrize(("tool", "args"), [
    ("terminal", {"command": "ls -la"}),
    ("terminal", {"command": ": > /tmp/truncate-me"}),
    ("execute_code", {"code": "print(1)"}),
])
def test_real_payload_is_dispatched(tool, args):
    import model_tools

    with patch.object(model_tools, "_execute_tool", return_value='{"ok": true}') as execute:
        model_tools.handle_function_call(tool, dict(args), task_id="t", skip_pre_tool_call_hook=True)

    execute.assert_called_once()


# ---------------------------------------------------------------------------
# Task 4: halt recovery
# ---------------------------------------------------------------------------

def _hard_stop_controller():
    return ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))


def test_first_halt_grants_recovery_and_blocks_the_signature():
    controller = _hard_stop_controller()
    for _ in range(5):
        controller.observe_call("terminal", {"command": "ls"}, "same")
    halt = controller.halt_decision
    assert halt is not None and halt.code == "identical_call_streak_halt"

    assert controller.begin_halt_recovery(halt) is True
    assert controller.halt_decision is None
    blocked = controller.before_call("terminal", {"command": "ls"})
    assert blocked.action == "block" and blocked.code == "halted_call_block"
    assert controller.before_call("terminal", {"command": "ls -la"}).allows_execution
    # The recovery is spent: a second halt ends the turn.
    assert controller.begin_halt_recovery(blocked) is False


def test_recovery_resets_per_turn():
    controller = _hard_stop_controller()
    for _ in range(5):
        controller.observe_call("terminal", {"command": "ls"}, "same")
    assert controller.begin_halt_recovery(controller.halt_decision)
    controller.reset_for_turn()
    assert controller.before_call("terminal", {"command": "ls"}).allows_execution
    for _ in range(5):
        controller.observe_call("terminal", {"command": "ls"}, "same")
    assert controller.begin_halt_recovery(controller.halt_decision) is True


def test_first_halt_gives_the_model_one_recovery_iteration(loop_agent):
    loop_agent._tool_guardrails = _hard_stop_controller()
    stages = [_tool_round(f"call_{i}", name="terminal", arguments='{"command": "ls"}') for i in range(5)]
    stages.append(_final("The directory has 3 files."))

    result, dispatch = _run(loop_agent, stages)

    assert result["final_response"] == "The directory has 3 files."
    assert result.get("guardrail") is None
    assert dispatch.call_count == 5
    tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
    assert _GUARDRAIL_RECOVERY_NOTE.split("{")[0] in tool_rows[-1]["content"]


def test_second_halt_in_the_same_turn_ends_it(loop_agent):
    loop_agent._tool_guardrails = _hard_stop_controller()
    stages = [_tool_round(f"call_{i}", name="terminal", arguments='{"command": "ls"}') for i in range(6)]
    stages.append(_final("never reached"))

    result, dispatch = _run(loop_agent, stages)

    # The 6th identical call is blocked before dispatch, which is the second halt.
    assert dispatch.call_count == 5
    assert loop_agent.client.chat.completions.create.call_count == 6
    assert result["guardrail"]["code"] == "halted_call_block"
    assert "I stopped retrying" in result["final_response"]
