import numpy as np

from config.model_settings import ModelSettings
from rag.research_agent import run_research_agent
from rag.research_plan import (
    ResearchAction,
    ResearchDecision,
    ResearchSubquestion,
)
from rag.search import SearchResult
from rag.web_search import WebSearchError, WebSearchResult


def paper(arxiv_id: str, document: str) -> SearchResult:
    return SearchResult(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        document=document,
        entry_url=f"https://arxiv.org/abs/{arxiv_id}",
        primary_category="cs.AI",
        similarity=0.8,
        rerank_score=0.5,
    )


class FakeReranker:
    model_name = "fake"

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        weights = {"MCP evidence": 0.9, "Skill evidence": 0.8}
        return np.asarray([weights.get(item, 0.5) for item in documents])


class FakeWebSearch:
    def __init__(
        self,
        results: tuple[WebSearchResult, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self.queries: list[str] = []
        self._results = results
        self._error = error

    def search(
        self, query: str, *, max_results: int = 5
    ) -> tuple[WebSearchResult, ...]:
        self.queries.append(query)
        if self._error is not None:
            raise self._error
        return self._results


def test_research_agent_decomposes_comparison_and_combines_searches(
    monkeypatch: object,
) -> None:
    plans = [
        ResearchDecision(
            question_type="comparison",
            reason_summary="先查 MCP",
            subquestions=(
                ResearchSubquestion("sq1", "MCP 是什么？", "pending"),
                ResearchSubquestion("sq2", "Skill 是什么？", "pending"),
                ResearchSubquestion("sq3", "二者有什么差异？", "pending"),
            ),
            next_action=ResearchAction(
                "search_papers", "sq1", "Model Context Protocol", 3
            ),
        ),
        ResearchDecision(
            question_type="comparison",
            reason_summary="MCP 已覆盖，继续查 Skill",
            subquestions=(
                ResearchSubquestion("sq1", "MCP 是什么？", "covered"),
                ResearchSubquestion("sq2", "Skill 是什么？", "pending"),
                ResearchSubquestion("sq3", "二者有什么差异？", "pending"),
            ),
            next_action=ResearchAction(
                "search_papers", "sq2", "AI agent skills", 3
            ),
        ),
        ResearchDecision(
            question_type="comparison",
            reason_summary="两侧证据都已获得，可以综合比较",
            subquestions=(
                ResearchSubquestion("sq1", "MCP 是什么？", "covered"),
                ResearchSubquestion("sq2", "Skill 是什么？", "covered"),
                ResearchSubquestion("sq3", "二者有什么差异？", "covered"),
            ),
            next_action=ResearchAction("finish"),
        ),
    ]
    observed_remaining: list[int] = []

    def fake_decide(*args: object, **kwargs: object) -> ResearchDecision:
        observed_remaining.append(kwargs["remaining_searches"])  # type: ignore[arg-type]
        return plans.pop(0)

    queries: list[str] = []

    def fake_search(query: str, **kwargs: object) -> list[SearchResult]:
        queries.append(query)
        return [
            paper("MCP", "MCP evidence")
            if "Context" in query
            else paper("SKILL", "Skill evidence")
        ]

    monkeypatch.setattr("rag.research_agent.decide_research_action", fake_decide)
    monkeypatch.setattr("rag.research_agent.hybrid_search", fake_search)
    monkeypatch.setattr(
        "rag.research_agent.answer_from_results",
        lambda *args, **kwargs: "combined grounded answer",
    )

    result = run_research_agent(
        "MCP 和 Skill 有什么区别？",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
    )

    assert queries == ["Model Context Protocol", "AI agent skills"]
    assert observed_remaining == [4, 3, 2]
    assert result.answer == "combined grounded answer"
    assert result.retrieval_attempts == 2
    assert {item.arxiv_id for item in result.papers} == {"MCP", "SKILL"}
    stages = [event.stage for event in result.trace]
    assert stages.count("agent_tool_call") == 2
    assert stages.count("agent_tool_observation") == 2
    assert "agent_plan" in stages
    assert "agent_finish" in stages


def test_research_agent_responds_to_feedback_without_tools(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(
        "rag.research_agent.decide_research_action",
        lambda *args, **kwargs: ResearchDecision(
            question_type="conversation",
            reason_summary="用户在反馈上一轮回答",
            subquestions=(
                ResearchSubquestion("sq1", "回应用户反馈", "covered"),
            ),
            next_action=ResearchAction("respond"),
        ),
    )
    monkeypatch.setattr(
        "rag.research_agent.hybrid_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("feedback must not search papers")
        ),
    )
    monkeypatch.setattr(
        "rag.research_agent.generate_conversational_response",
        lambda *args, **kwargs: "你说得对。你希望我检查哪一个具体结论？",
    )

    result = run_research_agent(
        "感觉你在乱说",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
    )

    assert result.answer == "你说得对。你希望我检查哪一个具体结论？"
    assert result.papers == ()
    assert result.retrieval_attempts == 0
    assert result.response_kind == "conversation"
    assert [event.stage for event in result.trace] == [
        "agent_plan",
        "agent_respond",
        "conversation_response",
    ]


def test_research_agent_returns_grounded_refusal_after_search(
    monkeypatch: object,
) -> None:
    decisions = [
        ResearchDecision(
            question_type="single_fact",
            reason_summary="先搜索",
            subquestions=(ResearchSubquestion("sq1", "未知问题", "pending"),),
            next_action=ResearchAction("search_papers", "sq1", "unknown topic", 3),
        ),
        ResearchDecision(
            question_type="single_fact",
            reason_summary="没有检索到支持证据",
            subquestions=(ResearchSubquestion("sq1", "未知问题", "unresolved"),),
            next_action=ResearchAction("refuse"),
        ),
    ]
    monkeypatch.setattr(
        "rag.research_agent.decide_research_action",
        lambda *args, **kwargs: decisions.pop(0),
    )
    monkeypatch.setattr("rag.research_agent.hybrid_search", lambda *args, **kwargs: [])

    result = run_research_agent(
        "知识库外的问题",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
    )

    assert result.answer is not None
    assert "不足以可靠回答" in result.answer
    assert result.papers == ()
    assert result.trace[-1].stage == "agent_refusal"


def test_research_agent_refines_terminology_with_web_search(
    monkeypatch: object,
) -> None:
    subquestion = ResearchSubquestion("sq1", "Agent Skill 是什么？", "pending")
    decisions = [
        ResearchDecision(
            question_type="single_fact",
            reason_summary="skill 可能是产品术语，先用网页搜索确认学术叫法",
            subquestions=(subquestion,),
            next_action=ResearchAction("web_search", query="AI agent skill meaning"),
        ),
        ResearchDecision(
            question_type="single_fact",
            reason_summary="网页结果指向 tool use，用精确术语检索本地论文",
            subquestions=(subquestion,),
            next_action=ResearchAction(
                "search_papers", "sq1", "LLM agent tool use skill library", 3
            ),
        ),
        ResearchDecision(
            question_type="single_fact",
            reason_summary="论文证据已覆盖",
            subquestions=(ResearchSubquestion("sq1", "Agent Skill 是什么？", "covered"),),
            next_action=ResearchAction("finish"),
        ),
    ]
    observed_budgets: list[tuple[int, int]] = []
    observed_observations: list[list[dict[str, object]]] = []

    def fake_decide(*args: object, **kwargs: object) -> ResearchDecision:
        observed_budgets.append(
            (kwargs["remaining_searches"], kwargs["remaining_web_searches"])  # type: ignore[arg-type]
        )
        observed_observations.append(list(kwargs["observations"]))  # type: ignore[arg-type]
        return decisions.pop(0)

    web = FakeWebSearch(
        results=(
            WebSearchResult(
                title="Agent skills explained",
                url="https://example.test/skills",
                snippet="Agent skills package tool use instructions for LLM agents.",
            ),
        )
    )
    local_queries: list[str] = []

    def fake_search(query: str, **kwargs: object) -> list[SearchResult]:
        local_queries.append(query)
        return [paper("SKILL", "Skill evidence")]

    monkeypatch.setattr("rag.research_agent.decide_research_action", fake_decide)
    monkeypatch.setattr("rag.research_agent.hybrid_search", fake_search)
    monkeypatch.setattr(
        "rag.research_agent.answer_from_results",
        lambda *args, **kwargs: "grounded answer",
    )

    result = run_research_agent(
        "Agent 的 skill 是什么？",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
        web_search_client=web,
    )

    assert web.queries == ["AI agent skill meaning"]
    assert local_queries == ["LLM agent tool use skill library"]
    assert observed_budgets == [(4, 2), (3, 1), (2, 1)]
    web_observation = observed_observations[1][0]
    assert web_observation["tool"] == "web_search"
    assert web_observation["result_count"] == 1
    assert result.answer == "grounded answer"
    assert result.retrieval_attempts == 1
    assert {item.arxiv_id for item in result.papers} == {"SKILL"}
    labels = [event.label for event in result.trace]
    assert "Agent 调用网页搜索工具" in labels
    assert "Agent 观察网页搜索结果" in labels


def test_research_agent_treats_web_search_failure_as_observation(
    monkeypatch: object,
) -> None:
    subquestion = ResearchSubquestion("sq1", "MCP 是什么？", "pending")
    decisions = [
        ResearchDecision(
            question_type="single_fact",
            reason_summary="先确认术语",
            subquestions=(subquestion,),
            next_action=ResearchAction("web_search", query="model context protocol"),
        ),
        ResearchDecision(
            question_type="single_fact",
            reason_summary="网页搜索失败，直接用现有术语检索本地论文",
            subquestions=(subquestion,),
            next_action=ResearchAction(
                "search_papers", "sq1", "Model Context Protocol", 3
            ),
        ),
        ResearchDecision(
            question_type="single_fact",
            reason_summary="论文证据已覆盖",
            subquestions=(ResearchSubquestion("sq1", "MCP 是什么？", "covered"),),
            next_action=ResearchAction("finish"),
        ),
    ]
    observed_observations: list[list[dict[str, object]]] = []

    def fake_decide(*args: object, **kwargs: object) -> ResearchDecision:
        observed_observations.append(list(kwargs["observations"]))  # type: ignore[arg-type]
        return decisions.pop(0)

    monkeypatch.setattr("rag.research_agent.decide_research_action", fake_decide)
    monkeypatch.setattr(
        "rag.research_agent.hybrid_search",
        lambda *args, **kwargs: [paper("MCP", "MCP evidence")],
    )
    monkeypatch.setattr(
        "rag.research_agent.answer_from_results",
        lambda *args, **kwargs: "grounded answer",
    )

    result = run_research_agent(
        "MCP 是什么？",
        top_k=5,
        collection=object(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
        web_search_client=FakeWebSearch(
            error=WebSearchError("web search returned HTTP 500")
        ),
    )

    assert result.answer == "grounded answer"
    assert result.retrieval_attempts == 1
    failure_observation = observed_observations[1][0]
    assert failure_observation["tool"] == "web_search"
    assert "HTTP 500" in str(failure_observation["error"])
    failed_events = [
        event
        for event in result.trace
        if event.label == "网页搜索工具执行失败" and event.status == "failed"
    ]
    assert len(failed_events) == 1
