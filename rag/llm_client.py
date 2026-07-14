"""Call an OpenAI-compatible LLM with bounded transient-error retries."""

from collections.abc import Callable
from dataclasses import dataclass
import json
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


@dataclass(frozen=True)
class ModelUsage:
    """Token usage reported by one or more model calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "ModelUsage") -> "ModelUsage":
        return ModelUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True)
class AgentToolCall:
    """One function call assembled from streamed model deltas."""

    call_id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str


@dataclass(frozen=True)
class AgentMessage:
    """One complete streamed assistant message."""

    content: str
    tool_calls: tuple[AgentToolCall, ...]
    usage: ModelUsage | None
    finish_reason: str | None


StatusCallback = Callable[[RetryNotice], None]
AssistantDeltaCallback = Callable[[str], None]


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


def generate_agent_message(
    messages: list[dict[str, Any]],
    *,
    settings: ModelSettings,
    tools: list[dict[str, Any]],
    client: Any | None = None,
    max_attempts: int = 5,
    initial_delay_seconds: float = 1.0,
    on_retry: StatusCallback | None = None,
    on_delta: AssistantDeltaCallback | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> AgentMessage:
    """Stream one assistant turn that may contain text or function calls."""
    if not messages:
        raise ValueError("messages must not be empty")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")
    if initial_delay_seconds < 0:
        raise ValueError("initial_delay_seconds must not be negative")

    sdk_client = client or create_client(settings)
    for attempt in range(1, max_attempts + 1):
        received_delta = False
        try:
            request_options: dict[str, Any] = {
                "model": settings.model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if tools:
                request_options.update(
                    {
                        "tools": tools,
                        "tool_choice": "auto",
                        "parallel_tool_calls": False,
                    }
                )

            stream = sdk_client.chat.completions.create(**request_options)
            content_parts: list[str] = []
            tool_parts: dict[int, dict[str, object]] = {}
            usage: ModelUsage | None = None
            finish_reason: str | None = None

            for chunk in stream:
                chunk_usage = _model_usage(getattr(chunk, "usage", None))
                if chunk_usage is not None:
                    usage = chunk_usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                choice_finish_reason = getattr(choice, "finish_reason", None)
                if choice_finish_reason is not None:
                    finish_reason = str(choice_finish_reason)
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                content = getattr(delta, "content", None)
                if content:
                    received_delta = True
                    text = str(content)
                    content_parts.append(text)
                    if on_delta is not None:
                        on_delta(text)

                for tool_delta in getattr(delta, "tool_calls", None) or []:
                    received_delta = True
                    index = int(getattr(tool_delta, "index", 0) or 0)
                    buffer = tool_parts.setdefault(
                        index,
                        {"id": "", "name": "", "arguments": []},
                    )
                    call_id = getattr(tool_delta, "id", None)
                    if call_id:
                        buffer["id"] = str(call_id)
                    function = getattr(tool_delta, "function", None)
                    if function is None:
                        continue
                    name = getattr(function, "name", None)
                    if name:
                        buffer["name"] = str(name)
                    arguments = getattr(function, "arguments", None)
                    if arguments:
                        argument_parts = buffer["arguments"]
                        if isinstance(argument_parts, list):
                            argument_parts.append(str(arguments))

            tool_calls = tuple(
                _agent_tool_call(index, value)
                for index, value in sorted(tool_parts.items())
            )
            content = "".join(content_parts).strip()
            if not content and not tool_calls:
                raise LLMRequestError("Model returned an empty response")
            return AgentMessage(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                finish_reason=finish_reason,
            )
        except AuthenticationError as error:
            raise LLMRequestError(
                "Model authentication failed; check LLM_API_KEY"
            ) from error
        except Exception as error:
            if isinstance(error, LLMRequestError):
                raise
            if received_delta or not _is_retryable(error) or attempt == max_attempts:
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


def _model_usage(value: object) -> ModelUsage | None:
    if value is None:
        return None
    return ModelUsage(
        prompt_tokens=int(getattr(value, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(value, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(value, "total_tokens", 0) or 0),
    )


def _agent_tool_call(index: int, value: dict[str, object]) -> AgentToolCall:
    argument_parts = value.get("arguments")
    raw_arguments = (
        "".join(str(item) for item in argument_parts)
        if isinstance(argument_parts, list)
        else ""
    )
    parsed_arguments: dict[str, Any] | None = None
    try:
        parsed = json.loads(raw_arguments or "{}")
        if isinstance(parsed, dict):
            parsed_arguments = parsed
    except json.JSONDecodeError:
        pass
    return AgentToolCall(
        call_id=str(value.get("id") or f"tool-call-{index}"),
        name=str(value.get("name") or ""),
        arguments=parsed_arguments,
        raw_arguments=raw_arguments,
    )
