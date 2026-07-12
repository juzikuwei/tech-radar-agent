"""Call an OpenAI-compatible LLM with bounded transient-error retries."""

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from config.model_settings import ModelSettings


@dataclass(frozen=True)
class RetryNotice:
    """Information a UI can display before the next request attempt."""

    next_attempt: int
    max_attempts: int
    wait_seconds: float
    reason: str


StatusCallback = Callable[[RetryNotice], None]


class LLMRequestError(RuntimeError):
    """A model request failed after retry handling was applied."""


def _is_retryable(error: Exception) -> bool:
    """Return whether repeating the same request may recover."""
    if isinstance(error, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    return isinstance(error, APIStatusError) and error.status_code >= 500


def create_client(settings: ModelSettings) -> OpenAI:
    """Create the official SDK client for an OpenAI-compatible endpoint."""
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        max_retries=0,
    )


def generate_text(
    messages: list[dict[str, str]],
    *,
    settings: ModelSettings,
    client: Any | None = None,
    max_attempts: int = 5,
    initial_delay_seconds: float = 1.0,
    on_retry: StatusCallback | None = None,
    sleep: Callable[[float], None] = time.sleep,
    response_format: dict[str, str] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Generate text, retrying only temporary connection or service failures."""
    if not messages:
        raise ValueError("messages must not be empty")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")
    if initial_delay_seconds < 0:
        raise ValueError("initial_delay_seconds must not be negative")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    if temperature is not None and temperature < 0:
        raise ValueError("temperature must not be negative")

    sdk_client = client or create_client(settings)
    for attempt in range(1, max_attempts + 1):
        try:
            request_options: dict[str, Any] = {
                "model": settings.model,
                "messages": messages,
            }
            if response_format is not None:
                request_options["response_format"] = response_format
            if max_tokens is not None:
                request_options["max_tokens"] = max_tokens
            if temperature is not None:
                request_options["temperature"] = temperature
            response = sdk_client.chat.completions.create(
                **request_options
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise LLMRequestError("Model returned an empty response")
            return content.strip()
        except AuthenticationError as error:
            raise LLMRequestError(
                "Model authentication failed; check LLM_API_KEY"
            ) from error
        except Exception as error:
            if not _is_retryable(error) or attempt == max_attempts:
                raise LLMRequestError(
                    f"Model request failed on attempt {attempt}/{max_attempts}: {error}"
                ) from error

            wait_seconds = initial_delay_seconds * (2 ** (attempt - 1))
            if on_retry is not None:
                on_retry(
                    RetryNotice(
                        next_attempt=attempt + 1,
                        max_attempts=max_attempts,
                        wait_seconds=wait_seconds,
                        reason=type(error).__name__,
                    )
                )
            sleep(wait_seconds)

    raise AssertionError("retry loop ended unexpectedly")
