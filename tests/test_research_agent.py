import json

import numpy as np

from config.model_settings import ModelSettings
from rag.llm_client import AgentMessage, AgentToolCall, ModelUsage
from rag.research_agent import MAX_TOOL_CALLS, run_research_agent
from rag.search import SearchResult
from rag.web_search import WebSearchError, WebSearchResult


SETTINGS = ModelSettings("key", "https://example.test", "model")


def paper(arxiv_id: str = "2501.09136") -> SearchResult:
    return SearchResult(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        document="Grounded paper evidence.",
        entry_url=f"https://arxiv.org/abs/{arxiv_id}",
        primary_category="cs.AI",
        similarity=0.8,
        rerank_score=0.7,
    )


def tool_call(name: str, arguments: dict[str, object], index: int = 1) -> AgentToolCall:
    raw = json.dumps(arguments)
    return AgentToolCall(
        call_id=f"call-{index}",
        name=name,
        arguments=arguments,
        raw_arguments=raw,
    )


def message(
    content: str = "",
    *,
    calls: tuple[AgentToolCall, ...] = (),
    usage: ModelUsage | None = None,
) -> AgentMessage:
    return AgentMessage(
        content=content,
        tool_calls=calls,
        usage=usage,
        finish_reason="tool_calls" if calls else "stop",
    )


class FakeReranker:
    model_name = "fake"

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        return np.ones(len(documents))


class FakeWebSearch:
    def __init__(
        self,
        *,
        results: tuple[WebSearchResult, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self.results = results
        self.error = error
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> tuple[WebSearchResult, ...]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.results[:max_results]


def test_casual_message_finishes_without_tools(
    monkeypatch: object,
) -> None:
    observed_tools: list[list[dict[str, object]]] = []

    def fake_generate(*args: object, **kwargs: object) -> AgentMessage:
        observed_tools.append(kwargs["tools"])  # type: ignore[arg-type]
        on_delta = kwargs["on_delta"]
        on_delta("谢谢！")  # type: ignore[operator]
        return message(
            "谢谢！",
            usage=ModelUsage(10, 3, 13),
        )

    monkeypatch.setattr("rag.research_agent.generate_agent_message", fake_generate)
    monkeypatch.setattr(
        "rag.research_agent.hybrid_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("casual message must not search")
        ),
    )
    deltas: list[str] = []

    result = run_research_agent(
        "做得好",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=SETTINGS,
        on_assistant_delta=deltas.append,
    )

    assert result.answer == "谢谢！"
    assert result.response_kind == "conversation"
    assert result.retrieval_attempts == 0
    assert result.usage == ModelUsage(10, 3, 13)
    assert deltas == ["谢谢！"]
    assert [tool["function"]["name"] for tool in observed_tools[0]] == [
        "search_papers"
    ]
    assert [(event.stage, event.status) for event in result.trace] == [
        ("model", "started"),
        ("model", "completed"),
    ]


def test_agent_wraps_hybrid_retrieval_as_one_public_tool(
    monkeypatch: object,
) -> None:
    responses = [
        message(calls=(tool_call("search_papers", {"query": "agentic rag", "top_k": 3}),)),
        message(
            "基于论文的回答 [2501.09136]。",
            usage=ModelUsage(20, 8, 28),
        ),
    ]
    internal_trace_callbacks: list[object] = []

    monkeypatch.setattr(
        "rag.research_agent.generate_agent_message",
        lambda *args, **kwargs: responses.pop(0),
    )

    def fake_search(*args: object, **kwargs: object) -> list[SearchResult]:
        internal_trace_callbacks.append(kwargs["trace"])
        return [paper()]

    monkeypatch.setattr("rag.research_agent.hybrid_search", fake_search)

    result = run_research_agent(
        "什么是 Agentic RAG？",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=SETTINGS,
    )

    assert result.answer == "基于论文的回答 [2501.09136]。"
    assert [item.arxiv_id for item in result.papers] == ["2501.09136"]
    assert result.retrieval_attempts == 1
    assert {event.stage for event in result.trace} == {"model", "tool"}
    tool_events = [event for event in result.trace if event.stage == "tool"]
    assert [event.status for event in tool_events] == ["started", "completed"]
    assert tool_events[-1].details["tool"] == "search_papers"
    assert tool_events[-1].details["output"] == {
        "result_count": 1,
        "papers": [{"arxiv_id": "2501.09136", "title": "Paper 2501.09136"}],
    }
    assert internal_trace_callbacks


def test_non_retryable_web_failure_removes_tool_then_model_uses_local_search(
    monkeypatch: object,
) -> None:
    responses = [
        message(calls=(tool_call("web_search", {"query": "MCP skill"}),)),
        message(calls=(tool_call("search_papers", {"query": "Model Context Protocol", "top_k": 3}, 2),)),
        message("MCP 是一种协议 [2501.09136]。"),
    ]
    available_names: list[list[str]] = []

    def fake_generate(*args: object, **kwargs: object) -> AgentMessage:
        available_names.append(
            [tool["function"]["name"] for tool in kwargs["tools"]]  # type: ignore[index]
        )
        return responses.pop(0)

    web = FakeWebSearch(
        error=WebSearchError(
            "web search returned HTTP 401",
            error_type="authentication",
            status_code=401,
            retryable=False,
        )
    )
    monkeypatch.setattr("rag.research_agent.generate_agent_message", fake_generate)
    monkeypatch.setattr("rag.research_agent.hybrid_search", lambda *args, **kwargs: [paper()])

    result = run_research_agent(
        "MCP 和 Skill 是什么？",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=SETTINGS,
        web_search_client=web,
    )

    assert web.queries == ["MCP skill"]
    assert available_names == [
        ["search_papers", "web_search"],
        ["search_papers"],
        ["search_papers"],
    ]
    failed = [event for event in result.trace if event.status == "failed"]
    assert len(failed) == 1
    output = failed[0].details["output"]
    assert output["error_type"] == "authentication"  # type: ignore[index]
    assert output["tool_available"] is False  # type: ignore[index]
    assert result.answer == "MCP 是一种协议 [2501.09136]。"


def test_model_cannot_execute_a_tool_removed_from_the_current_menu(
    monkeypatch: object,
) -> None:
    responses = [
        message(calls=(tool_call("web_search", {"query": "MCP"}, 1),)),
        message(calls=(tool_call("web_search", {"query": "MCP again"}, 2),)),
        message("网页工具不可用，请补充具体上下文。"),
    ]
    web = FakeWebSearch(
        error=WebSearchError(
            "web search returned HTTP 401",
            error_type="authentication",
            status_code=401,
            retryable=False,
        )
    )
    monkeypatch.setattr(
        "rag.research_agent.generate_agent_message",
        lambda *args, **kwargs: responses.pop(0),
    )

    result = run_research_agent(
        "MCP 是什么？",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=SETTINGS,
        web_search_client=web,
    )

    assert web.queries == ["MCP"]
    unavailable = [
        event
        for event in result.trace
        if event.details.get("output", {}).get("error_type") == "tool_unavailable"
    ]
    assert len(unavailable) == 1
    assert result.answer == "网页工具不可用，请补充具体上下文。"


def test_tool_execution_failure_is_returned_to_model(
    monkeypatch: object,
) -> None:
    observed_messages: list[list[dict[str, object]]] = []
    responses = [
        message(calls=(tool_call("search_papers", {"query": "broken", "top_k": 3}),)),
        message("论文工具暂时不可用，请稍后重试。"),
    ]

    def fake_generate(messages: list[dict[str, object]], **kwargs: object) -> AgentMessage:
        observed_messages.append(list(messages))
        return responses.pop(0)

    monkeypatch.setattr("rag.research_agent.generate_agent_message", fake_generate)
    monkeypatch.setattr(
        "rag.research_agent.hybrid_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("index offline")),
    )

    result = run_research_agent(
        "检索这个主题",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=SETTINGS,
    )

    tool_message = observed_messages[1][-1]
    payload = json.loads(tool_message["content"])  # type: ignore[arg-type]
    assert payload["ok"] is False
    assert payload["error_type"] == "tool_execution"
    assert result.answer == "论文工具暂时不可用，请稍后重试。"
    assert result.generation_error is None


def test_only_first_tool_call_in_one_assistant_turn_is_executed(
    monkeypatch: object,
) -> None:
    responses = [
        message(
            calls=(
                tool_call("web_search", {"query": "first query"}, 1),
                tool_call("web_search", {"query": "duplicate query"}, 2),
            )
        ),
        message("请补充你所指的具体术语。"),
    ]
    observed_messages: list[list[dict[str, object]]] = []

    def fake_generate(messages: list[dict[str, object]], **kwargs: object) -> AgentMessage:
        observed_messages.append(list(messages))
        return responses.pop(0)

    web = FakeWebSearch(results=())
    monkeypatch.setattr("rag.research_agent.generate_agent_message", fake_generate)

    result = run_research_agent(
        "一个模糊术语",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=SETTINGS,
        web_search_client=web,
    )

    assert web.queries == ["first query"]
    tool_payloads = [
        json.loads(item["content"])  # type: ignore[arg-type]
        for item in observed_messages[1]
        if item.get("role") == "tool"
    ]
    assert len(tool_payloads) == 2
    assert tool_payloads[1]["error_type"] == "parallel_tool_call_rejected"
    rejected = [
        event
        for event in result.trace
        if event.details.get("output", {}).get("error_type")
        == "parallel_tool_call_rejected"
    ]
    assert len(rejected) == 1


def test_five_tool_calls_are_followed_by_one_tool_free_final_call(
    monkeypatch: object,
) -> None:
    calls = [
        message(
            calls=(
                tool_call(
                    "search_papers",
                    {"query": f"query {index}", "top_k": 1},
                    index,
                ),
            )
        )
        for index in range(1, MAX_TOOL_CALLS + 1)
    ]
    calls.append(message("工具预算已用完，没有足够证据。"))
    tool_menus: list[list[str]] = []

    def fake_generate(*args: object, **kwargs: object) -> AgentMessage:
        tool_menus.append(
            [tool["function"]["name"] for tool in kwargs["tools"]]  # type: ignore[index]
        )
        return calls.pop(0)

    monkeypatch.setattr("rag.research_agent.generate_agent_message", fake_generate)
    monkeypatch.setattr("rag.research_agent.hybrid_search", lambda *args, **kwargs: [])

    result = run_research_agent(
        "需要多轮检索",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=SETTINGS,
    )

    assert len(tool_menus) == MAX_TOOL_CALLS + 1
    assert all(menu == ["search_papers"] for menu in tool_menus[:-1])
    assert tool_menus[-1] == []
    assert result.retrieval_attempts == MAX_TOOL_CALLS
    assert result.answer == "工具预算已用完，没有足够证据。"
