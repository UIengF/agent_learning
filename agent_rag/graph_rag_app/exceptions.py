from __future__ import annotations

from typing import Any


class ProviderError(RuntimeError):
    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}
        super().__init__(f"[{provider}] {message}")


class EmbeddingError(ProviderError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "dashscope-embedding",
            message,
            status_code=status_code,
            retryable=retryable,
            details=details,
        )


class SearchBackendError(ProviderError):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retryable: bool = True,
        detail: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = dict(details or {})
        if detail:
            merged_details["detail"] = detail
        self.code = message
        self.detail = str(merged_details.get("detail", ""))
        super().__init__(
            provider,
            message,
            status_code=status_code,
            retryable=retryable,
            details=merged_details,
        )
