import asyncio
from pathlib import Path

import httpx
import pytest

import mcp_server.tools as tools_module
from config.mcp_settings import McpSettings
from config.model_settings import ModelSettings
from mcp_server.auth import BearerTokenMiddleware
from mcp_server.server import create_mcp_server
from mcp_server.tools import KnowledgeBaseTools
from rag.runtime import RagRuntime
from rag.search import SearchResult


class FakeCollection:
    def count(self) -> int:
        return 42


def make_runtime(database_path: Path = Path("unused.db")) -> RagRuntime:
    return RagRuntime(
        collection=FakeCollection(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=object(),  # type: ignore[arg-type]
        settings=ModelSettings("key", "https://example.test", "model"),
        database_path=database_path,
    )


def test_query_tool_returns_only_compact_public_evidence(
    monkeypatch: object,
) -> None:
    long_abstract = "A" * 700
    monkeypatch.setattr(
        tools_module,
        "search_knowledge_base",
        lambda *args, **kwargs: [
            SearchResult(
                arxiv_id="2501.09136",
                title="Agentic RAG",
                document=f"Agentic RAG\n{long_abstract}",
                entry_url="https://arxiv.org/abs/2501.09136",
                primary_category="cs.AI",
                similarity=0.8,
                rerank_score=4.123456,
            )
        ],
    )
    tools = KnowledgeBaseTools(make_runtime)

    payload = tools.query_knowledge_base("agentic rag", top_k=1)
    result = payload["results"][0]  # type: ignore[index]

    assert set(payload) == {"results"}
    assert set(result) == {
        "arxiv_id",
        "title",
        "abstract_excerpt",
        "primary_category",
        "score",
        "entry_url",
    }
    assert result["score"] == 4.1235
    assert len(result["abstract_excerpt"]) == 600
    assert result["abstract_excerpt"].endswith("…")


def test_query_tool_rejects_more_than_five_results() -> None:
    tools = KnowledgeBaseTools(make_runtime)

    with pytest.raises(ValueError, match="between 1 and 5"):
        tools.query_knowledge_base("agentic rag", top_k=6)


def test_paper_and_stats_tools_use_minimal_contracts(monkeypatch: object) -> None:
    monkeypatch.setattr(
        tools_module,
        "load_papers_by_arxiv_ids",
        lambda *args, **kwargs: [
            {
                "arxiv_id": "2501.09136",
                "title": "Agentic RAG",
                "abstract": "Stored abstract",
                "primary_category": "cs.AI",
                "published_at": "2025-01-16T00:00:00+00:00",
                "entry_url": "https://arxiv.org/abs/2501.09136",
                "pdf_url": "https://arxiv.org/pdf/2501.09136",
                "updated_at": "2025-01-16T00:00:00+00:00",
            }
        ],
    )
    monkeypatch.setattr(tools_module, "get_paper_count", lambda path: 42)
    tools = KnowledgeBaseTools(make_runtime)

    paper = tools.get_paper_by_arxiv_id("2501.09136")
    stats = tools.get_knowledge_base_stats()

    assert set(paper) == {
        "arxiv_id",
        "title",
        "abstract",
        "primary_category",
        "published_at",
        "entry_url",
    }
    assert stats == {"paper_count": 42, "vector_count": 42}


def test_server_registers_exactly_three_read_only_tools() -> None:
    settings = McpSettings(auth_token="a-secure-local-token")
    server = create_mcp_server(settings, make_runtime)

    registered = asyncio.run(server.list_tools())

    assert {tool.name for tool in registered} == {
        "query_knowledge_base",
        "get_paper_by_arxiv_id",
        "get_knowledge_base_stats",
    }
    assert all(tool.annotations.readOnlyHint is True for tool in registered)


def test_bearer_middleware_blocks_missing_token_and_allows_health() -> None:
    async def downstream(scope, receive, send):
        body = b"ok"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"2")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    app = BearerTokenMiddleware(downstream, "a-secure-local-token")

    async def exercise() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            missing = await client.post("/mcp")
            authorized = await client.post(
                "/mcp",
                headers={"Authorization": "Bearer a-secure-local-token"},
            )
            health = await client.get("/health")
        return missing, authorized, health

    missing, authorized, health = asyncio.run(exercise())

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert authorized.status_code == 200
    assert health.status_code == 200
