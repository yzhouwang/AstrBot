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
from astrbot.core.star.star_handler import EventType, star_handlers_registry


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


def _resolve_plugin_name(handler) -> str:
    """Best-effort plugin display name for diagnostics."""
    plugin = star_map.get(handler.handler_module_path)
    if plugin and plugin.name:
        return plugin.name
    return handler.handler_module_path or "<unknown>"


async def call_event_hook(
    event: AstrMessageEvent,
    hook_type: EventType,
    *args,
    **kwargs,
) -> bool:
    """调用事件钩子函数

    Returns:
        bool: 如果事件被终止，返回 True

    Raises:
        HookAbortError: 当某个 ``fail_closed=True`` 的 handler 抛出异常或超时时
            抛出。调用方负责捕获该异常并跳过任何用户可见的消息发送。

    """
    handlers = star_handlers_registry.get_handlers_by_event_type(
        hook_type,
        plugins_name=event.plugins_name,
    )
    for handler in handlers:
        plugin_name = _resolve_plugin_name(handler)
        timeout = handler.timeout_seconds
        fail_closed = handler.fail_closed
        started_at = time.monotonic()

        try:
            assert inspect.iscoroutinefunction(handler.handler)
            logger.debug(
                f"hook({hook_type.name}) -> {plugin_name} - {handler.handler_name}"
                + (f" [fail_closed]" if fail_closed else "")
                + (f" [timeout={timeout}s]" if timeout is not None else "")
            )
            coro = handler.handler(event, *args, **kwargs)
            if timeout is not None:
                await asyncio.wait_for(coro, timeout=timeout)
            else:
                await coro
        except (KeyboardInterrupt, SystemExit):
            # Never swallow process-level signals — they must propagate.
            raise
        except asyncio.TimeoutError as exc:
            duration_ms = (time.monotonic() - started_at) * 1000.0
            logger.error(
                "Hook %s -> %s.%s timed out after %.1fms (limit=%.3fs).",
                hook_type.name,
                plugin_name,
                handler.handler_name,
                duration_ms,
                timeout if timeout is not None else -1.0,
            )
            if fail_closed:
                _record_hook_failure(
                    event,
                    hook_type=hook_type,
                    plugin_name=plugin_name,
                    handler_name=handler.handler_name,
                    exc=exc,
                    timed_out=True,
                    duration_ms=duration_ms,
                )
                event.stop_event()
                raise HookAbortError(
                    hook_name=hook_type.name,
                    plugin_name=plugin_name,
                    handler_name=handler.handler_name,
                    reason=f"timed out after {duration_ms:.1f}ms",
                    original_exc=exc,
                    timed_out=True,
                    duration_ms=duration_ms,
                ) from exc
            # fail-open: log and continue with the next handler.
            continue
        except BaseException as exc:
            duration_ms = (time.monotonic() - started_at) * 1000.0
            tb_text = traceback.format_exc()
            if fail_closed:
                logger.error(
                    "Hook %s -> %s.%s raised under fail_closed=True; aborting pipeline.\n%s",
                    hook_type.name,
                    plugin_name,
                    handler.handler_name,
                    tb_text,
                )
                _record_hook_failure(
                    event,
                    hook_type=hook_type,
                    plugin_name=plugin_name,
                    handler_name=handler.handler_name,
                    exc=exc,
                    timed_out=False,
                    duration_ms=duration_ms,
                )
                event.stop_event()
                raise HookAbortError(
                    hook_name=hook_type.name,
                    plugin_name=plugin_name,
                    handler_name=handler.handler_name,
                    reason=f"{type(exc).__name__}: {exc}",
                    original_exc=exc,
                    timed_out=False,
                    duration_ms=duration_ms,
                ) from exc
            # fail-open: legacy behaviour — log and keep going.
            logger.error(tb_text)

        if event.is_stopped():
            logger.info(
                f"{plugin_name} - {handler.handler_name} 终止了事件传播。",
            )
            return True

    return event.is_stopped()


def _record_hook_failure(
    event: AstrMessageEvent,
    *,
    hook_type: EventType,
    plugin_name: str,
    handler_name: str,
    exc: BaseException,
    timed_out: bool,
    duration_ms: float,
) -> None:
    """Best-effort audit record for fail-closed hook aborts.

    Records to the event trace if available, and stashes a structured payload
    in ``event._extras`` so platform adapters / runners can include it in their
    own audit streams. Never raises — failure to record must not mask the
    original abort.
    """
    payload = {
        "hook_name": hook_type.name,
        "plugin_name": plugin_name,
        "handler_name": handler_name,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }
    try:
        trace = getattr(event, "trace", None)
        if trace is not None and hasattr(trace, "record"):
            trace.record("hook_fail_closed_abort", **payload)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to record hook abort to event.trace", exc_info=True)
    try:
        event.set_extra("_hook_abort", payload)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to stash hook abort on event extras", exc_info=True)
