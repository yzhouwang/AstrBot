"""Tests for opt-in fail_closed + timeout semantics on LLM filter decorators.

These tests exercise ``call_event_hook`` directly with synthetic handlers
registered into ``star_handlers_registry``. We avoid spinning up the full
pipeline so the test suite stays fast and focused on the dispatch site.

See ``astrbot/core/pipeline/context_utils.py`` and
``astrbot/core/star/star_handler.py`` for the implementation.
"""

from __future__ import annotations

import asyncio

import pytest

from astrbot.core.exceptions import HookAbortError
from astrbot.core.pipeline.context_utils import call_event_hook
from astrbot.core.star.register.star_handler import (
    register_on_llm_request,
    register_on_llm_response,
)
from astrbot.core.star.star import StarMetadata, star_map
from astrbot.core.star.star_handler import EventType, star_handlers_registry


class _FakeResult:
    """Minimal MessageEventResult stand-in just for is_stopped/stop_event."""

    def __init__(self) -> None:
        self._stopped = False

    def stop_event(self) -> None:
        self._stopped = True

    def is_stopped(self) -> bool:
        return self._stopped


class _FakeTrace:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict]] = []

    def record(self, name: str, **payload) -> None:
        self.records.append((name, payload))


class _FakeEvent:
    """Minimal AstrMessageEvent stand-in for hook dispatch tests."""

    def __init__(self) -> None:
        self.plugins_name: list[str] = ["*"]
        self._result: _FakeResult | None = None
        self._extras: dict = {}
        self.trace = _FakeTrace()
        self.send_calls: list = []

    def stop_event(self) -> None:
        if self._result is None:
            self._result = _FakeResult()
        self._result.stop_event()

    def is_stopped(self) -> bool:
        if self._result is None:
            return False
        return self._result.is_stopped()

    def set_extra(self, key: str, value) -> None:
        self._extras[key] = value

    def get_extra(self, key: str | None = None, default=None):
        if key is None:
            return self._extras
        return self._extras.get(key, default)

    def get_platform_name(self) -> str:
        return "test_platform"

    async def send(self, *args, **kwargs) -> None:
        # Record any send so tests can assert it never fires.
        self.send_calls.append((args, kwargs))


@pytest.fixture(autouse=True)
def _reset_registry():
    """Snapshot and restore the global handler registry between tests."""
    saved_handlers = list(star_handlers_registry._handlers)  # noqa: SLF001
    saved_map = dict(star_handlers_registry.star_handlers_map)
    saved_star_map = dict(star_map)
    star_handlers_registry.clear()
    star_map.clear()
    yield
    star_handlers_registry.clear()
    for h in saved_handlers:
        star_handlers_registry.append(h)
    star_handlers_registry.star_handlers_map.update(saved_map)
    star_map.update(saved_star_map)


@pytest.fixture
def fake_event() -> _FakeEvent:
    return _FakeEvent()


def _register_plugin_for(handler) -> None:
    """Register a synthetic StarMetadata so call_event_hook treats the handler
    as belonging to an activated plugin."""
    star_map[handler.__module__] = StarMetadata(
        name="test_plugin",
        author="test",
        desc="",
        version="0.0.0",
        module_path=handler.__module__,
        activated=True,
        reserved=False,
    )


@pytest.mark.asyncio
async def test_hook_fail_closed_aborts_pipeline(fake_event: _FakeEvent) -> None:
    """A fail_closed hook that raises must (a) abort the pipeline, (b)
    prevent later handlers from running, and (c) raise HookAbortError to
    the caller so the runner / scheduler knows not to send a reply."""
    later_handler_invocations: list[str] = []

    @register_on_llm_request(fail_closed=True)
    async def first_throws(event, request):
        raise RuntimeError("policy violation: unredacted PII detected")

    @register_on_llm_request()  # legacy default: fail_closed=False
    async def second_should_not_run(event, request):
        later_handler_invocations.append("second")

    _register_plugin_for(first_throws)
    _register_plugin_for(second_should_not_run)

    with pytest.raises(HookAbortError) as exc_info:
        await call_event_hook(
            fake_event, EventType.OnLLMRequestEvent, object()
        )

    assert exc_info.value.hook_name == "OnLLMRequestEvent"
    assert exc_info.value.plugin_name == "test_plugin"
    assert exc_info.value.handler_name == "first_throws"
    assert isinstance(exc_info.value.original_exc, RuntimeError)
    assert exc_info.value.timed_out is False
    assert later_handler_invocations == [], (
        "Downstream handler ran despite earlier fail_closed abort."
    )
    assert fake_event.is_stopped(), "Event was not stopped on fail_closed abort."
    abort_payload = fake_event.get_extra("_hook_abort")
    assert abort_payload is not None
    assert abort_payload["plugin_name"] == "test_plugin"
    assert abort_payload["exception_type"] == "RuntimeError"
    assert "PII" in abort_payload["exception_message"]


@pytest.mark.asyncio
async def test_hook_fail_open_continues_pipeline(fake_event: _FakeEvent) -> None:
    """Backwards-compat regression guard: legacy handlers (no fail_closed)
    must still be tolerated — exception logged, later handlers run."""
    later_handler_invocations: list[str] = []

    @register_on_llm_request()  # default fail_closed=False
    async def first_throws_open(event, request):
        raise RuntimeError("legacy plugin had a bug")

    @register_on_llm_request()
    async def second_runs(event, request):
        later_handler_invocations.append("second")

    _register_plugin_for(first_throws_open)
    _register_plugin_for(second_runs)

    # Should NOT raise.
    stopped = await call_event_hook(
        fake_event, EventType.OnLLMRequestEvent, object()
    )

    assert stopped is False
    assert later_handler_invocations == ["second"], (
        "Backwards compat broken: second handler did not run after a "
        "fail-open exception in the first."
    )
    assert not fake_event.is_stopped()
    assert fake_event.get_extra("_hook_abort") is None


@pytest.mark.asyncio
async def test_hook_timeout_aborts_when_fail_closed(
    fake_event: _FakeEvent,
) -> None:
    """A fail_closed handler that exceeds timeout_seconds must abort the
    pipeline and raise HookAbortError(timed_out=True)."""
    later_handler_invocations: list[str] = []

    @register_on_llm_request(fail_closed=True, timeout_seconds=0.1)
    async def slow_handler(event, request):
        await asyncio.sleep(2)  # well above the 100ms budget

    @register_on_llm_request()
    async def should_not_run(event, request):
        later_handler_invocations.append("late")

    _register_plugin_for(slow_handler)
    _register_plugin_for(should_not_run)

    with pytest.raises(HookAbortError) as exc_info:
        await call_event_hook(
            fake_event, EventType.OnLLMRequestEvent, object()
        )

    assert exc_info.value.timed_out is True
    assert exc_info.value.handler_name == "slow_handler"
    assert exc_info.value.duration_ms is not None
    assert exc_info.value.duration_ms >= 100  # at least the timeout window
    assert later_handler_invocations == []
    assert fake_event.is_stopped()


@pytest.mark.asyncio
async def test_hook_timeout_warns_when_fail_open(
    fake_event: _FakeEvent,
) -> None:
    """A fail-open handler that times out must be cancelled, and the
    pipeline must continue to subsequent handlers."""
    later_handler_invocations: list[str] = []
    completed = False

    @register_on_llm_request(fail_closed=False, timeout_seconds=0.1)
    async def slow_open_handler(event, request):
        nonlocal completed
        try:
            await asyncio.sleep(2)
            completed = True
        except asyncio.CancelledError:
            # asyncio.wait_for cancels the wrapped coroutine on timeout;
            # observing the cancellation here proves it was actually cancelled.
            raise

    @register_on_llm_request()
    async def runs_after_timeout(event, request):
        later_handler_invocations.append("ran")

    _register_plugin_for(slow_open_handler)
    _register_plugin_for(runs_after_timeout)

    # Must NOT raise.
    stopped = await call_event_hook(
        fake_event, EventType.OnLLMRequestEvent, object()
    )

    assert stopped is False
    assert completed is False, "Slow handler ran to completion despite timeout."
    assert later_handler_invocations == ["ran"], (
        "Pipeline did not continue after a fail-open timeout."
    )
    # No abort payload should be recorded because fail_closed was False.
    assert fake_event.get_extra("_hook_abort") is None


@pytest.mark.asyncio
async def test_on_llm_response_failure_does_not_send_to_user(
    fake_event: _FakeEvent,
) -> None:
    """Critical safety property: when a fail_closed @on_llm_response handler
    throws, no platform send call may be issued for the current turn.

    We assert this at the dispatch layer: HookAbortError propagates and
    the event is marked stopped before any send path can run.
    """

    @register_on_llm_response(fail_closed=True)
    async def citation_validator(event, response):
        raise ValueError("citation faithfulness check failed")

    _register_plugin_for(citation_validator)

    with pytest.raises(HookAbortError):
        await call_event_hook(
            fake_event, EventType.OnLLMResponseEvent, object()
        )

    assert fake_event.send_calls == [], (
        "event.send was called even though a fail_closed on_llm_response "
        "hook aborted the pipeline."
    )
    assert fake_event.is_stopped()
    assert fake_event.get_extra("_hook_abort") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_closed", [True, False])
async def test_keyboard_interrupt_propagates(
    fake_event: _FakeEvent, fail_closed: bool
) -> None:
    """KeyboardInterrupt and SystemExit must NEVER be swallowed, regardless
    of fail_closed setting. They are process-level signals."""

    @register_on_llm_request(fail_closed=fail_closed)
    async def interrupted(event, request):
        raise KeyboardInterrupt

    _register_plugin_for(interrupted)

    with pytest.raises(KeyboardInterrupt):
        await call_event_hook(
            fake_event, EventType.OnLLMRequestEvent, object()
        )


@pytest.mark.asyncio
async def test_decorator_rejects_nonpositive_timeout() -> None:
    """timeout_seconds must be positive when set — a 0 or negative value is
    almost certainly a config bug, surface it at registration time."""
    with pytest.raises(ValueError):

        @register_on_llm_request(timeout_seconds=0)
        async def bad_timeout_zero(event, request):  # noqa: ARG001
            pass

    with pytest.raises(ValueError):

        @register_on_llm_request(timeout_seconds=-1)
        async def bad_timeout_neg(event, request):  # noqa: ARG001
            pass


@pytest.mark.asyncio
async def test_legacy_decorator_invocation_unchanged(
    fake_event: _FakeEvent,
) -> None:
    """Decorator usage without any new kwargs must produce a metadata entry
    with the legacy defaults — fail_closed=False, timeout_seconds=None."""

    @register_on_llm_request()
    async def legacy(event, request):
        pass

    _register_plugin_for(legacy)

    handlers = star_handlers_registry.get_handlers_by_event_type(
        EventType.OnLLMRequestEvent
    )
    assert len(handlers) == 1
    md = handlers[0]
    assert md.fail_closed is False
    assert md.timeout_seconds is None

    # Dispatch should be a no-op.
    stopped = await call_event_hook(
        fake_event, EventType.OnLLMRequestEvent, object()
    )
    assert stopped is False
    assert not fake_event.is_stopped()
