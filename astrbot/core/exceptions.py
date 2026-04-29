from __future__ import annotations


class AstrBotError(Exception):
    """Base exception for all AstrBot errors."""


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


class HookAbortError(AstrBotError):
    """Raised by ``call_event_hook`` when a handler registered with
    ``fail_closed=True`` raises or exceeds its ``timeout_seconds``.

    Catchers (the agent sub-stages) must:
      * skip ``_save_to_history`` so the failed turn does not poison the
        conversation,
      * not call ``event.send`` so no partial reply reaches the user,
      * ensure ``event.stop_event()`` is set so downstream pipeline stages
        (result decoration, respond stage) bail out via ``is_stopped()``.

    ``call_event_hook`` already calls ``event.stop_event()`` before raising,
    so callers only need to handle the propagation path.
    """
