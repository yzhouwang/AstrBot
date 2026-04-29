"""Tests for fail_closed and timeout_seconds on LLM hook decorators."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from astrbot.api.event import filter as filter_decorators
from astrbot.core.exceptions import HookAbortError
from astrbot.core.pipeline.context_utils import call_event_hook
from astrbot.core.star.star import StarMetadata, star_map
from astrbot.core.star.star_handler import EventType, star_handlers_registry


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


def _make_test_event() -> MagicMock:
    """Build a MagicMock AstrMessageEvent with stateful is_stopped/stop_event.

    The conftest.py ``mock_event`` fixture supplies most of the surface area
    we need (``event.trace = MagicMock()``), but it does not wire is_stopped
    to stop_event. Tests in this module need that linkage so we build our
    own minimal stand-in.
    """
    event = MagicMock()
    state = {"stopped": False}
    event.is_stopped = MagicMock(side_effect=lambda: state["stopped"])
    event.stop_event = MagicMock(side_effect=lambda: state.update(stopped=True))
    event.plugins_name = ["*"]
    event.trace = MagicMock()
    event.send = AsyncMock()
    event.unified_msg_origin = "test_umo"
    return event


def _register_fake_plugin_for(module_path: str) -> None:
    """Register a fake activated plugin for ``module_path`` so handlers
    registered during a test pass the only_activated filter inside
    ``star_handlers_registry.get_handlers_by_event_type``."""
    if module_path not in star_map:
        star_map[module_path] = StarMetadata(
            name="_test_fail_closed_plugin",
            module_path=module_path,
            activated=True,
        )


@pytest.fixture
def isolated_registry():
    """Snapshot+restore the global handler registry and ``star_map``.

    The Star handler registry and the plugin-name map are global singletons.
    Tests in this file register temporary handlers and would otherwise leak
    into sibling tests (or fail because a handler with the same fully-
    qualified name already exists). This fixture clears both before each
    test and restores them afterwards.
    """
    saved_handlers = list(star_handlers_registry._handlers)
    saved_map = dict(star_handlers_registry.star_handlers_map)
    saved_star_map = dict(star_map)

    star_handlers_registry._handlers.clear()
    star_handlers_registry.star_handlers_map.clear()
    # Note: star_map is intentionally left populated; we add a fake entry
    # below for handlers defined in this test module.
    _register_fake_plugin_for(__name__)

    yield

    star_handlers_registry._handlers[:] = saved_handlers
    star_handlers_registry.star_handlers_map.clear()
    star_handlers_registry.star_handlers_map.update(saved_map)
    star_map.clear()
    star_map.update(saved_star_map)


# ---------------------------------------------------------------------------
# 1. fail_closed=True + raising handler -> pipeline aborts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_fail_closed_aborts_pipeline(isolated_registry):
    second_called = MagicMock()

    @filter_decorators.on_llm_request(fail_closed=True)
    async def fail_closed_raises(event: Any, req: Any) -> None:
        raise ValueError("redaction failed")

    @filter_decorators.on_llm_request()
    async def downstream_should_not_run(event: Any, req: Any) -> None:
        second_called()

    event = _make_test_event()
    req = MagicMock()

    with pytest.raises(HookAbortError) as exc_info:
        await call_event_hook(event, EventType.OnLLMRequestEvent, req)

    assert "redaction failed" in str(exc_info.value) or "fail_closed_raises" in str(
        exc_info.value
    )
    assert event.is_stopped() is True
    second_called.assert_not_called()

    # Audit was recorded with the stable error code.
    event.trace.record.assert_called()
    record_call = event.trace.record.call_args
    assert record_call.args[0] == "hook_failure"
    assert record_call.kwargs["error_code"] == "ASTRBOT_HOOK_FAIL_CLOSED"
    assert record_call.kwargs["hook_name"] == "OnLLMRequestEvent"
    assert record_call.kwargs["fail_closed"] is True
    assert record_call.kwargs["kind"] == "exception"
    assert record_call.kwargs["exception_type"] == "ValueError"
    assert record_call.kwargs["traceback"] is not None


# ---------------------------------------------------------------------------
# 2. fail_closed=False (default) + raising -> log & continue
# (regression guard for backwards compatibility)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_fail_open_continues_pipeline(isolated_registry):
    second_called = MagicMock()

    @filter_decorators.on_llm_request()
    async def fail_open_raises(event: Any, req: Any) -> None:
        raise ValueError("oops")

    @filter_decorators.on_llm_request()
    async def downstream_runs(event: Any, req: Any) -> None:
        second_called()

    event = _make_test_event()

    result = await call_event_hook(event, EventType.OnLLMRequestEvent, MagicMock())

    assert result is False  # event was not stopped
    second_called.assert_called_once()
    assert event.is_stopped() is False

    # Audit recorded with the fail_open code (kind=exception).
    event.trace.record.assert_called()
    record_call = event.trace.record.call_args_list[0]
    assert record_call.kwargs["error_code"] == "ASTRBOT_HOOK_FAIL_OPEN"
    assert record_call.kwargs["fail_closed"] is False


# ---------------------------------------------------------------------------
# 3. timeout + fail_closed=True -> abort cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_timeout_aborts_when_fail_closed(isolated_registry):
    @filter_decorators.on_llm_request(fail_closed=True, timeout_seconds=0.05)
    async def slow_handler(event: Any, req: Any) -> None:
        await asyncio.sleep(1.0)

    event = _make_test_event()

    with pytest.raises(HookAbortError) as exc_info:
        await call_event_hook(event, EventType.OnLLMRequestEvent, MagicMock())

    assert "timeout" in str(exc_info.value).lower()
    assert event.is_stopped() is True

    event.trace.record.assert_called()
    record_call = event.trace.record.call_args
    assert record_call.kwargs["kind"] == "timeout"
    assert record_call.kwargs["error_code"] == "ASTRBOT_HOOK_FAIL_CLOSED"


# ---------------------------------------------------------------------------
# 4. timeout + fail_closed=False -> warn & continue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_timeout_warns_when_fail_open(isolated_registry):
    second_called = MagicMock()

    @filter_decorators.on_llm_request(timeout_seconds=0.05)
    async def slow_handler_warn(event: Any, req: Any) -> None:
        await asyncio.sleep(1.0)

    @filter_decorators.on_llm_request()
    async def downstream_runs_after_timeout(event: Any, req: Any) -> None:
        second_called()

    event = _make_test_event()

    # The project uses loguru, which does not propagate to pytest's caplog.
    # Patch the logger used by the dispatch module directly and assert that
    # a structured ASTRBOT_HOOK_TIMEOUT warning was emitted.
    with patch(
        "astrbot.core.pipeline.context_utils.logger.warning"
    ) as warning_mock:
        result = await call_event_hook(
            event, EventType.OnLLMRequestEvent, MagicMock()
        )

    assert result is False
    second_called.assert_called_once()
    assert event.is_stopped() is False
    warning_mock.assert_called_once()
    warning_args = warning_mock.call_args.args
    assert any("ASTRBOT_HOOK_TIMEOUT" in str(arg) for arg in warning_args), (
        f"expected ASTRBOT_HOOK_TIMEOUT in warning args, got {warning_args!r}"
    )


# ---------------------------------------------------------------------------
# 5. on_llm_response abort -> NO message sent to user (suppression contract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_llm_response_failure_does_not_send_to_user(isolated_registry):
    @filter_decorators.on_llm_response(fail_closed=True)
    async def citation_validation_fails(event: Any, resp: Any) -> None:
        raise RuntimeError("citation invalid")

    event = _make_test_event()

    with pytest.raises(HookAbortError):
        await call_event_hook(
            event, EventType.OnLLMResponseEvent, MagicMock()
        )

    # call_event_hook itself never invokes event.send; the broader
    # suppression contract is that downstream stages (RespondStage) check
    # event.is_stopped() before sending, and the scheduler/internal sub-
    # stage catch HookAbortError without sending an error reply.
    assert event.is_stopped() is True
    event.send.assert_not_called()


# ---------------------------------------------------------------------------
# 6. KeyboardInterrupt always propagates (regardless of fail_closed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_closed", [True, False])
async def test_keyboard_interrupt_propagates(isolated_registry, fail_closed):
    if fail_closed:

        @filter_decorators.on_llm_request(fail_closed=True)
        async def kbi_handler(event: Any, req: Any) -> None:
            raise KeyboardInterrupt()
    else:

        @filter_decorators.on_llm_request()
        async def kbi_handler(event: Any, req: Any) -> None:
            raise KeyboardInterrupt()

    event = _make_test_event()

    with pytest.raises(KeyboardInterrupt):
        await call_event_hook(event, EventType.OnLLMRequestEvent, MagicMock())

    # KeyboardInterrupt must not have been recoded as a "hook_failure" — the
    # interpreter-exit path must NOT pass through audit logging.
    event.trace.record.assert_not_called()


# ---------------------------------------------------------------------------
# 7. HookAbortError escapes broad `except Exception` catches (regression for
# the runner-style try/except around agent_hooks.* in
# tool_loop_agent_runner.py — those catches must NOT swallow the abort).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_abort_escapes_exception_catch(isolated_registry):
    @filter_decorators.on_agent_begin(fail_closed=True)
    async def aborting_hook(event: Any, run_context: Any) -> None:
        raise ValueError("policy violation")

    event = _make_test_event()

    swallowed = False
    aborted = False
    try:
        try:
            await call_event_hook(
                event, EventType.OnAgentBeginEvent, MagicMock()
            )
        except Exception:  # noqa: BLE001 — mimics runner's broad catch
            swallowed = True
    except HookAbortError:
        aborted = True

    assert swallowed is False, (
        "HookAbortError was caught by `except Exception` — the runner's "
        "agent_hooks try/except blocks would silently swallow the abort. "
        "HookAbortError must subclass BaseException, not Exception."
    )
    assert aborted is True
    assert event.is_stopped() is True


# ---------------------------------------------------------------------------
# 8. asyncio.CancelledError propagates cooperatively (regardless of
# fail_closed). Without this, request-task cancellation during shutdown or
# user abort would be miscounted as a hook failure.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_closed", [True, False])
async def test_cancelled_error_propagates(isolated_registry, fail_closed):
    if fail_closed:

        @filter_decorators.on_llm_request(fail_closed=True)
        async def cancelled_handler(event: Any, req: Any) -> None:
            raise asyncio.CancelledError()
    else:

        @filter_decorators.on_llm_request()
        async def cancelled_handler(event: Any, req: Any) -> None:
            raise asyncio.CancelledError()

    event = _make_test_event()

    with pytest.raises(asyncio.CancelledError):
        await call_event_hook(event, EventType.OnLLMRequestEvent, MagicMock())

    # CancelledError must not have been recorded as a hook_failure — it is
    # the cooperative-cancellation channel, not an error.
    event.trace.record.assert_not_called()
