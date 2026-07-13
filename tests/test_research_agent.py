import numpy as np

from config.model_settings import ModelSettings
from rag.research_agent import run_research_agent
from rag.research_plan import (
    ResearchAction,
    ResearchDecision,
    ResearchSubquestion,
)
from rag.search import SearchResult


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
        return np.asarray([weights[item] for item in documents])


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
