from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, AuthenticationError

from config.model_settings import ModelSettings
from rag.llm_client import (
    LLMRequestError,
    RetryNotice,
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
