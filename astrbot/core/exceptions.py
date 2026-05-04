from __future__ import annotations

import traceback as _traceback


class AstrBotError(Exception):
    """Base exception for all AstrBot errors."""


class HookAbortError(AstrBotError):
    """Raised when a fail-closed event hook handler raises or times out.

    Carries enough context for downstream callers (runners, platform adapters)
    to log the failure and skip any user-facing message send.
    """

    def __init__(
        self,
        *,
        hook_name: str,
        plugin_name: str,
        handler_name: str,
        reason: str,
        original_exc: BaseException | None = None,
        timed_out: bool = False,
        duration_ms: float | None = None,
    ) -> None:
        message = (
            f"Hook {hook_name} aborted by {plugin_name}.{handler_name}: {reason}"
        )
        super().__init__(message)
        self.hook_name = hook_name
        self.plugin_name = plugin_name
        self.handler_name = handler_name
        self.reason = reason
        self.original_exc = original_exc
        self.timed_out = timed_out
        self.duration_ms = duration_ms

    @property
    def traceback_text(self) -> str:
        if self.original_exc is None:
            return ""
        return "".join(
            _traceback.format_exception(
                type(self.original_exc),
                self.original_exc,
                self.original_exc.__traceback__,
            )
        )


class ProviderNotFoundError(AstrBotError):
    """Raised when a specified provider is not found."""


class EmptyModelOutputError(AstrBotError):
    """Raised when the model response contains no usable assistant output."""


class KnowledgeBaseUploadError(AstrBotError):
    """Raised when knowledge base upload fails with a user-facing message."""

    def __init__(
        self,
        *,
        stage: str,
        user_message: str,
        details: dict | None = None,
    ) -> None:
        super().__init__(user_message)
        self.stage = stage
        self.user_message = user_message
        self.details = details or {}

    def __str__(self) -> str:
        return self.user_message
