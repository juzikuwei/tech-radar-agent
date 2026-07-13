"""Bounded external web search used only to refine local retrieval queries.

Web results are untrusted content. They never become answer evidence and are
never citable; the research agent may only use them to turn vague or very new
terminology into precise English queries for the local paper search.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from config.environment import DEFAULT_ENV_PATH
from config.web_search_settings import WebSearchSettings


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_WEB_RESULTS = 5
TITLE_CHAR_LIMIT = 150
SNIPPET_CHAR_LIMIT = 200
REQUEST_TIMEOUT_SECONDS = 15.0


class WebSearchError(RuntimeError):
    """A web-search call failed and should become a tool observation."""


@dataclass(frozen=True)
class WebSearchResult:
    """One bounded untrusted search hit used only for query refinement."""

    title: str
    url: str
    snippet: str


class WebSearchClient(Protocol):
    """Anything that can perform one bounded web search."""

    def search(
        self, query: str, *, max_results: int = MAX_WEB_RESULTS
    ) -> tuple[WebSearchResult, ...]: ...


class TavilySearchClient:
    """Call the Tavily REST API once per search with bounded output."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        clean_key = api_key.strip()
        if not clean_key:
            raise ValueError("api_key must not be empty")
        self._api_key = clean_key
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def search(
        self, query: str, *, max_results: int = MAX_WEB_RESULTS
    ) -> tuple[WebSearchResult, ...]:
        """Return at most five truncated title/snippet hits for one query."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must not be empty")
        bounded_count = max(1, min(max_results, MAX_WEB_RESULTS))

        try:
            with httpx.Client(
                timeout=self._timeout_seconds, transport=self._transport
            ) as client:
                response = client.post(
                    TAVILY_SEARCH_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "query": clean_query,
                        "max_results": bounded_count,
                        "search_depth": "basic",
                    },
                )
        except httpx.HTTPError as error:
            raise WebSearchError(f"web search request failed: {error}") from error

        if response.status_code != 200:
            raise WebSearchError(
                f"web search returned HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise WebSearchError("web search returned invalid JSON") from error
        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            raise WebSearchError("web search response is missing results")

        results: list[WebSearchResult] = []
        for item in raw_results[:bounded_count]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("content") or "").strip()
            if not title and not snippet:
                continue
            results.append(
                WebSearchResult(
                    title=title[:TITLE_CHAR_LIMIT],
                    url=url,
                    snippet=snippet[:SNIPPET_CHAR_LIMIT],
                )
            )
        return tuple(results)


def load_web_search_client(
    env_path: Path = DEFAULT_ENV_PATH,
) -> TavilySearchClient | None:
    """Build the default client when the optional API key is configured."""
    settings = WebSearchSettings.from_env(env_path)
    if settings is None:
        return None
    return TavilySearchClient(settings.api_key)
