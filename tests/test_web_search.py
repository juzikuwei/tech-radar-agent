import json
from pathlib import Path

import httpx
import pytest

from config.web_search_settings import WebSearchSettings
from rag.web_search import (
    MAX_WEB_RESULTS,
    SNIPPET_CHAR_LIMIT,
    TITLE_CHAR_LIMIT,
    TavilySearchClient,
    WebSearchError,
    load_web_search_client,
)


def make_client(handler) -> TavilySearchClient:
    return TavilySearchClient("test-key", transport=httpx.MockTransport(handler))


def test_search_returns_bounded_truncated_results() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "T" * 400,
                        "url": f"https://example.test/{index}",
                        "content": "S" * 900,
                    }
                    for index in range(8)
                ]
            },
        )

    results = make_client(handler).search("model context protocol", max_results=99)

    assert captured["authorization"] == "Bearer test-key"
    assert captured["payload"]["max_results"] == MAX_WEB_RESULTS  # type: ignore[index]
    assert len(results) == MAX_WEB_RESULTS
    assert all(len(item.title) <= TITLE_CHAR_LIMIT for item in results)
    assert all(len(item.snippet) <= SNIPPET_CHAR_LIMIT for item in results)


def test_search_maps_http_and_transport_failures() -> None:
    unauthorized = make_client(lambda request: httpx.Response(401, json={}))
    with pytest.raises(WebSearchError, match="401") as unauthorized_error:
        unauthorized.search("agent skills")
    assert unauthorized_error.value.error_type == "authentication"
    assert unauthorized_error.value.status_code == 401
    assert unauthorized_error.value.retryable is False

    unavailable = make_client(lambda request: httpx.Response(503, json={}))
    with pytest.raises(WebSearchError, match="503") as unavailable_error:
        unavailable.search("agent skills")
    assert unavailable_error.value.error_type == "server"
    assert unavailable_error.value.status_code == 503
    assert unavailable_error.value.retryable is True

    rate_limited = make_client(lambda request: httpx.Response(429, json={}))
    with pytest.raises(WebSearchError, match="429") as rate_limit_error:
        rate_limited.search("agent skills")
    assert rate_limit_error.value.error_type == "rate_limit"
    assert rate_limit_error.value.retryable is True

    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection timed out")

    with pytest.raises(WebSearchError, match="request failed") as timeout_error:
        make_client(raise_timeout).search("agent skills")
    assert timeout_error.value.error_type == "timeout"
    assert timeout_error.value.retryable is True


def test_search_rejects_invalid_payload_and_blank_query() -> None:
    missing = make_client(lambda request: httpx.Response(200, json={"answer": "x"}))
    with pytest.raises(WebSearchError, match="missing results"):
        missing.search("agent skills")

    ok = make_client(lambda request: httpx.Response(200, json={"results": []}))
    with pytest.raises(ValueError, match="query must not be empty"):
        ok.search("   ")


def test_client_is_disabled_without_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    missing_env = tmp_path / "missing.env"

    assert WebSearchSettings.from_env(missing_env) is None
    assert load_web_search_client(missing_env) is None


def test_client_loads_when_api_key_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    missing_env = tmp_path / "missing.env"

    settings = WebSearchSettings.from_env(missing_env)
    assert settings is not None
    assert settings.api_key == "tvly-test"
    assert load_web_search_client(missing_env) is not None
