import asyncio
import inspect
import time
import traceback
import typing as T

from astrbot import logger
from astrbot.core.exceptions import HookAbortError
from astrbot.core.message.message_event_result import CommandResult, MessageEventResult
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.star.star import star_map
from astrbot.core.star.star_handler import (
    EventType,
    StarHandlerMetadata,
    star_handlers_registry,
)


async def call_handler(
    event: AstrMessageEvent,
    handler: T.Callable[..., T.Awaitable[T.Any] | T.AsyncGenerator[T.Any, None]],
    *args,
    **kwargs,
) -> T.AsyncGenerator[T.Any, None]:
    """执行事件处理函数并处理其返回结果

    该方法负责调用处理函数并处理不同类型的返回值。它支持两种类型的处理函数:
    1. 异步生成器: 实现洋葱模型，每次 yield 都会将控制权交回上层
    2. 协程: 执行一次并处理返回值

    Args:
        event (AstrMessageEvent): 事件对象
        handler (Awaitable): 事件处理函数

    Returns:
        AsyncGenerator[None, None]: 异步生成器，用于在管道中传递控制流

    """
    ready_to_call = None  # 一个协程或者异步生成器

    trace_ = None

    try:
        ready_to_call = handler(event, *args, **kwargs)
    except TypeError:
        logger.error("处理函数参数不匹配，请检查 handler 的定义。", exc_info=True)

    if not ready_to_call:
        return

    if inspect.isasyncgen(ready_to_call):
        _has_yielded = False
        try:
            async for ret in ready_to_call:
                # 这里逐步执行异步生成器, 对于每个 yield 返回的 ret, 执行下面的代码
                # 返回值只能是 MessageEventResult 或者 None（无返回值）
                _has_yielded = True
                if isinstance(ret, MessageEventResult | CommandResult):
                    # 如果返回值是 MessageEventResult, 设置结果并继续
                    event.set_result(ret)
                    yield
                else:
                    # 如果返回值是 None, 则不设置结果并继续
                    # 继续执行后续阶段
                    yield ret
            if not _has_yielded:
                # 如果这个异步生成器没有执行到 yield 分支
                yield
        except Exception as e:
            logger.error(f"Previous Error: {trace_}")
            raise e
    elif inspect.iscoroutine(ready_to_call):
        # 如果只是一个协程, 直接执行
        ret = await ready_to_call
        if isinstance(ret, MessageEventResult | CommandResult):
            event.set_result(ret)
            yield
        else:
            yield ret


_HOOK_FAIL_CLOSED_CODE = "ASTRBOT_HOOK_FAIL_CLOSED"
_HOOK_FAIL_OPEN_CODE = "ASTRBOT_HOOK_FAIL_OPEN"
_HOOK_TIMEOUT_CODE = "ASTRBOT_HOOK_TIMEOUT"
_DEFAULT_HOOK_FAILURE_METRIC = "astrbot.hook.fail_closed"


def _resolve_plugin_name(handler: StarHandlerMetadata) -> str:
    """Resolve the registered plugin name for a handler, falling back to the
    module path when the plugin record is missing (e.g. in unit tests)."""
    plugin = star_map.get(handler.handler_module_path)
    if plugin is not None:
        return plugin.name
    return handler.handler_module_path


def _resolve_hook_failure_metric_name() -> str:
    """Read the configured metric name for hook failure audits, with a safe
    default. Lazy-imported so this module loads cleanly during early
    bootstrap, before astrbot_config has been initialized."""
    try:
        from astrbot.core import astrbot_config

        pipeline_cfg = astrbot_config.get("pipeline", {}) or {}
        name = pipeline_cfg.get("hook_failure_metric_name")
        if isinstance(name, str) and name:
            return name
    except Exception:
        pass
    return _DEFAULT_HOOK_FAILURE_METRIC


def _record_hook_failure(
    event: AstrMessageEvent,
    handler: StarHandlerMetadata,
    plugin_name: str,
    hook_type: EventType,
    *,
    kind: str,
    exc: BaseException | None,
    duration_ms: float,
) -> None:
    """Emit a structured audit event + a stable error code so operators can
    alert on hook failures without parsing free-form log lines.

    ``kind`` is one of ``"exception"`` or ``"timeout"``. ``exc`` is the
    underlying error (None for timeouts).
    """
    error_code = _HOOK_FAIL_CLOSED_CODE if handler.fail_closed else _HOOK_FAIL_OPEN_CODE
    if kind == "timeout" and not handler.fail_closed:
        error_code = _HOOK_TIMEOUT_CODE

    fields: dict[str, T.Any] = {
        "hook_name": hook_type.name,
        "handler_name": handler.handler_name,
        "plugin_name": plugin_name,
        "kind": kind,
        "exception_type": type(exc).__name__ if exc is not None else None,
        "traceback": (
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            if exc is not None
            else None
        ),
        "duration_ms": duration_ms,
        "fail_closed": handler.fail_closed,
        "timeout_seconds": handler.timeout_seconds,
        "error_code": error_code,
        "metric_name": _resolve_hook_failure_metric_name(),
    }

    trace = getattr(event, "trace", None)
    if trace is not None:
        try:
            trace.record("hook_failure", **fields)
        except Exception:
            # Audit failure must never mask the real failure.
            logger.debug(
                "event.trace.record failed while recording hook_failure",
                exc_info=True,
            )

    logger.error(
        "%s plugin=%s hook=%s handler=%s kind=%s duration_ms=%.1f",
        error_code,
        plugin_name,
        hook_type.name,
        handler.handler_name,
        kind,
        duration_ms,
    )


async def call_event_hook(
    event: AstrMessageEvent,
    hook_type: EventType,
    *args,
    **kwargs,
) -> bool:
    """调用事件钩子函数

    Each handler runs in its own try/except. By default a handler exception
    is logged and the next handler runs (fail-open, the historical behavior).
    Handlers registered with ``fail_closed=True`` instead abort the pipeline:
    ``event.stop_event()`` is called and a :class:`HookAbortError` is raised
    so the agent sub-stage can suppress the user-facing reply and skip the
    history append.

    Handlers registered with ``timeout_seconds=N`` run inside
    ``asyncio.wait_for``. A timeout in a fail-open handler logs a warning
    and continues; in a fail-closed handler it aborts the pipeline.

    ``KeyboardInterrupt`` and ``SystemExit`` always propagate.

    Returns:
        bool: True if the event was stopped (either via ``event.stop_event()``
        from a handler or because a handler aborted the pipeline). On
        ``fail_closed`` handler failure this function raises
        :class:`HookAbortError` instead of returning.
    """
    handlers = star_handlers_registry.get_handlers_by_event_type(
        hook_type,
        plugins_name=event.plugins_name,
    )
    for handler in handlers:
        assert inspect.iscoroutinefunction(handler.handler)
        plugin_name = _resolve_plugin_name(handler)
        logger.debug(
            f"hook({hook_type.name}) -> {plugin_name} - {handler.handler_name}",
        )

        timeout = handler.timeout_seconds
        started_at = time.perf_counter()

        try:
            if timeout is not None:
                await asyncio.wait_for(
                    handler.handler(event, *args, **kwargs),
                    timeout=timeout,
                )
            else:
                await handler.handler(event, *args, **kwargs)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            # Always propagate. KeyboardInterrupt/SystemExit are interpreter
            # exit signals; CancelledError is asyncio's cooperative
            # cancellation channel and swallowing it breaks request-scoped
            # task cancellation and shutdown.
            raise
        except HookAbortError:
            # If a handler itself raised HookAbortError (e.g., a wrapper
            # plugin proxying another hook), respect the intent and
            # propagate without re-recording. event.stop_event() should
            # have already been set by whoever raised it.
            raise
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - started_at) * 1000
            _record_hook_failure(
                event,
                handler,
                plugin_name,
                hook_type,
                kind="timeout",
                exc=None,
                duration_ms=duration_ms,
            )
            if handler.fail_closed:
                event.stop_event()
                raise HookAbortError(
                    f"hook timeout: {plugin_name}.{handler.handler_name} "
                    f"exceeded {timeout}s",
                ) from None
            logger.warning(
                "%s plugin=%s hook=%s handler=%s timeout_seconds=%s",
                _HOOK_TIMEOUT_CODE,
                plugin_name,
                hook_type.name,
                handler.handler_name,
                timeout,
            )
        except BaseException as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            _record_hook_failure(
                event,
                handler,
                plugin_name,
                hook_type,
                kind="exception",
                exc=exc,
                duration_ms=duration_ms,
            )
            if handler.fail_closed:
                event.stop_event()
                raise HookAbortError(
                    f"hook failed: {plugin_name}.{handler.handler_name}: "
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            # fail_open + exception: preserve the historical log-and-continue.
            logger.error(traceback.format_exc())

        if event.is_stopped():
            logger.info(
                f"{plugin_name} - {handler.handler_name} 终止了事件传播。",
            )
            return True

    return event.is_stopped()
