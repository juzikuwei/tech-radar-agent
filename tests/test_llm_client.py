from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, AuthenticationError

from config.model_settings import ModelSettings
from rag.llm_client import (
    AgentMessage,
    LLMRequestError,
    ModelUsage,
    RetryNotice,
    generate_agent_message,
    generate_text,
)


class FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.last_options: dict[str, object] | None = None

    def create(self, **options: object) -> object:
        self.last_options = options
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        if options.get("stream"):
            return outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))]
        )


def make_client(outcomes: list[object]) -> object:
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        completions_spy=completions,
    )


def timeout_error() -> APITimeoutError:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return APITimeoutError(request=request)


def authentication_error() -> AuthenticationError:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(401, request=request)
    return AuthenticationError("invalid key", response=response, body=None)


SETTINGS = ModelSettings("test-key", "https://api.deepseek.com", "deepseek-chat")
MESSAGES = [{"role": "user", "content": "hello"}]


def test_transient_failure_retries_with_exponential_backoff() -> None:
    client = make_client([timeout_error(), timeout_error(), "answer"])
    notices: list[RetryNotice] = []
    sleeps: list[float] = []

    result = generate_text(
        MESSAGES,
        settings=SETTINGS,
        client=client,
        on_retry=notices.append,
        sleep=sleeps.append,
    )

    assert result == "answer"
    assert client.completions_spy.calls == 3
    assert sleeps == [1.0, 2.0]
    assert [notice.next_attempt for notice in notices] == [2, 3]


def test_transient_failure_stops_after_five_attempts() -> None:
    client = make_client([timeout_error() for _ in range(5)])

    with pytest.raises(LLMRequestError, match="attempt 5/5"):
        generate_text(
            MESSAGES,
            settings=SETTINGS,
            client=client,
            sleep=lambda _: None,
        )

    assert client.completions_spy.calls == 5


def test_authentication_failure_is_not_retried() -> None:
    client = make_client([authentication_error()])

    with pytest.raises(LLMRequestError, match="authentication failed"):
        generate_text(MESSAGES, settings=SETTINGS, client=client)

    assert client.completions_spy.calls == 1


def test_empty_model_response_is_not_retried() -> None:
    client = make_client(["   "])

    with pytest.raises(LLMRequestError, match="empty response"):
        generate_text(MESSAGES, settings=SETTINGS, client=client)

    assert client.completions_spy.calls == 1


def test_generate_text_reports_usage_to_the_callback() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        ),
    )

    class UsageCompletions:
        def create(self, **options: object) -> object:
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=UsageCompletions()))
    usages: list[ModelUsage] = []

    result = generate_text(
        MESSAGES,
        settings=SETTINGS,
        client=client,
        on_usage=usages.append,
    )

    assert result == "answer"
    assert usages == [
        ModelUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    ]


def test_optional_generation_controls_are_forwarded() -> None:
    client = make_client(["answer"])

    result = generate_text(
        MESSAGES,
        settings=SETTINGS,
        client=client,
        response_format={"type": "json_object"},
        max_tokens=300,
        temperature=0.0,
    )

    assert result == "answer"
    assert client.completions_spy.last_options is not None
    assert client.completions_spy.last_options["response_format"] == {
        "type": "json_object"
    }
    assert client.completions_spy.last_options["max_tokens"] == 300
    assert client.completions_spy.last_options["temperature"] == 0.0


def stream_chunk(
    *,
    content: str | None = None,
    tool_calls: list[object] | None = None,
    finish_reason: str | None = None,
    usage: object | None = None,
) -> object:
    return SimpleNamespace(
        choices=(
            [
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=content,
                        tool_calls=tool_calls or [],
                    ),
                    finish_reason=finish_reason,
                )
            ]
            if content is not None or tool_calls or finish_reason is not None
            else []
        ),
        usage=usage,
    )


def tool_delta(
    index: int,
    *,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> object:
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_agent_message_streams_text_and_usage() -> None:
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        total_tokens=16,
    )
    client = make_client(
        [[
            stream_chunk(content="你"),
            stream_chunk(content="好"),
            stream_chunk(finish_reason="stop"),
            stream_chunk(usage=usage),
        ]]
    )
    deltas: list[str] = []

    result = generate_agent_message(
        MESSAGES,
        settings=SETTINGS,
        tools=[],
        client=client,
        on_delta=deltas.append,
    )

    assert result == AgentMessage(
        content="你好",
        tool_calls=(),
        usage=ModelUsage(12, 4, 16),
        finish_reason="stop",
    )
    assert deltas == ["你", "好"]
    assert client.completions_spy.last_options["stream"] is True
    assert client.completions_spy.last_options["stream_options"] == {
        "include_usage": True
    }


def test_agent_message_assembles_streamed_tool_arguments() -> None:
    client = make_client(
        [[
            stream_chunk(
                tool_calls=[
                    tool_delta(
                        0,
                        call_id="call-1",
                        name="search_papers",
                        arguments='{"query":"agent',
                    )
                ]
            ),
            stream_chunk(
                tool_calls=[tool_delta(0, arguments='ic rag","top_k":3}')]
            ),
            stream_chunk(finish_reason="tool_calls"),
        ]]
    )

    result = generate_agent_message(
        MESSAGES,
        settings=SETTINGS,
        tools=[{"type": "function", "function": {"name": "search_papers"}}],
        client=client,
    )

    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].call_id == "call-1"
    assert result.tool_calls[0].name == "search_papers"
    assert result.tool_calls[0].arguments == {
        "query": "agentic rag",
        "top_k": 3,
    }
    assert client.completions_spy.last_options["parallel_tool_calls"] is False
