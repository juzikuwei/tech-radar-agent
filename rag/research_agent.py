"""Bounded ReAct loop for planning and multi-query paper research."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection

from config.model_settings import ModelSettings
from rag.application import RagResult, answer_from_results
from rag.conversation import ConversationTurn, bounded_history
from rag.execution_trace import TraceEvent, TraceEventCallback, TraceRecorder, start_timer
from rag.hybrid_search import hybrid_search
from rag.keyword_search import DEFAULT_DATABASE_PATH
from rag.llm_client import LLMRequestError, StatusCallback
from rag.reranker import Reranker
from rag.research_plan import (
    ResearchDecisionError,
    ResearchSubquestion,
    decide_research_action,
)
from rag.search import QueryEmbedder, SearchResult
from rag.web_search import WebSearchClient, WebSearchError


MAX_RESEARCH_ACTIONS = 4
MAX_WEB_SEARCHES = 2
OBSERVATION_PAPER_LIMIT = 5
OBSERVATION_EXCERPT_LIMIT = 300


class ResearchAgentError(RuntimeError):
    """A planning or tool failure that should fall back to the pipeline."""

    def __init__(self, message: str, trace: tuple[TraceEvent, ...]) -> None:
        super().__init__(message)
        self.trace = trace


@dataclass(frozen=True)
class SearchObservation:
    """One search action and the papers it returned."""

    subquestion_id: str
    query: str
    papers: tuple[SearchResult, ...]


def run_research_agent(
    question: str,
    *,
    top_k: int,
    collection: Collection,
    embedder: QueryEmbedder,
    reranker: Reranker,
    settings: ModelSettings,
    database_path: Path = DEFAULT_DATABASE_PATH,
    client: Any | None = None,
    on_retry: StatusCallback | None = None,
    conversation_history: tuple[ConversationTurn, ...] = (),
    active_evidence: tuple[SearchResult, ...] = (),
    on_trace: TraceEventCallback | None = None,
    max_actions: int = MAX_RESEARCH_ACTIONS,
    web_search_client: WebSearchClient | None = None,
) -> RagResult:
    """Let the model plan bounded searches, then generate one grounded answer."""
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if max_actions <= 0:
        raise ValueError("max_actions must be greater than zero")

    trace = TraceRecorder(on_event=on_trace)
    history = bounded_history(conversation_history)
    evidence_by_id = {paper.arxiv_id: paper for paper in active_evidence}
    observations: list[SearchObservation] = []
    observation_payloads: list[dict[str, object]] = []
    plan: tuple[ResearchSubquestion, ...] = ()
    search_count = 0
    web_search_count = 0

    while True:
        remaining_actions = max_actions - search_count - web_search_count
        remaining_web_searches = min(
            MAX_WEB_SEARCHES - web_search_count, remaining_actions
        )
        started_at = start_timer()
        try:
            decision = decide_research_action(
                clean_question,
                history=history,
                evidence=tuple(evidence_by_id.values()),
                observations=list(observation_payloads),
                previous_plan=plan,
                remaining_searches=remaining_actions,
                search_count=search_count,
                remaining_web_searches=remaining_web_searches,
                web_search_available=web_search_client is not None,
                settings=settings,
                client=client,
                on_retry=on_retry,
            )
        except (LLMRequestError, ResearchDecisionError, ValueError) as error:
            trace.record(
                stage="agent_decision",
                label="研究 Agent 规划失败",
                status="failed",
                started_at=started_at,
                details={
                    "error": str(error),
                    "round": search_count + web_search_count + 1,
                },
            )
            raise ResearchAgentError(str(error), trace.events) from error

        is_initial_plan = not plan
        plan = decision.subquestions
        trace.record(
            stage="agent_plan" if is_initial_plan else "agent_decision",
            label="DeepSeek 生成研究计划" if is_initial_plan else "DeepSeek 决定下一步研究动作",
            started_at=started_at,
            details={
                "question_type": decision.question_type,
                "reason_summary": decision.reason_summary,
                "plan": [
                    {
                        "id": item.id,
                        "question": item.question,
                        "status": item.status,
                    }
                    for item in plan
                ],
                "next_action": decision.next_action.type,
                "target_subquestion_id": decision.next_action.target_subquestion_id,
                "remaining_searches": remaining_actions,
                "remaining_web_searches": remaining_web_searches,
            },
        )

        action = decision.next_action
        if action.type == "web_search":
            web_search_count += 1
            _run_web_search(
                action.query or "",
                web_search_client=web_search_client,
                observation_payloads=observation_payloads,
                trace=trace,
            )
            continue

        if action.type == "search_papers":
            trace.record(
                stage="agent_tool_call",
                label="Agent 调用论文检索工具",
                details={
                    "tool": "search_papers",
                    "target_subquestion_id": action.target_subquestion_id,
                    "query": action.query,
                    "top_k": action.top_k,
                },
            )
            tool_started_at = start_timer()
            try:
                papers = hybrid_search(
                    action.query or "",
                    top_k=action.top_k or top_k,
                    collection=collection,
                    embedder=embedder,
                    reranker=reranker,
                    database_path=database_path,
                    trace=trace,
                    retrieval_round=search_count + 1,
                )
            except Exception as error:
                trace.record(
                    stage="agent_tool_observation",
                    label="论文检索工具执行失败",
                    status="failed",
                    started_at=tool_started_at,
                    details={
                        "tool": "search_papers",
                        "query": action.query,
                        "error": str(error),
                    },
                )
                raise ResearchAgentError(str(error), trace.events) from error

            search_count += 1
            observation = SearchObservation(
                subquestion_id=action.target_subquestion_id or "",
                query=action.query or "",
                papers=tuple(papers),
            )
            observations.append(observation)
            observation_payloads.append(_observation_payload(observation))
            for paper in papers:
                evidence_by_id.setdefault(paper.arxiv_id, paper)
            trace.record(
                stage="agent_tool_observation",
                label="Agent 观察论文检索结果",
                started_at=tool_started_at,
                details={
                    "target_subquestion_id": observation.subquestion_id,
                    "query": observation.query,
                    "result_count": len(papers),
                    "top_arxiv_ids": [paper.arxiv_id for paper in papers],
                },
            )
            continue

        selected = _select_final_evidence(
            clean_question,
            evidence=tuple(evidence_by_id.values()),
            observations=observations,
            top_k=top_k,
            reranker=reranker,
        )
        if action.type == "refuse":
            trace.record(
                stage="agent_refusal",
                label="研究 Agent 因证据不足而停止",
                details={
                    "reason_summary": decision.reason_summary,
                    "search_count": search_count,
                    "paper_count": len(selected),
                },
            )
            return RagResult(
                question=clean_question,
                papers=tuple(selected),
                answer=(
                    "当前本地论文摘要不足以可靠回答这个问题。"
                    f"研究过程已执行 {search_count} 次检索，但仍缺少关键证据。"
                ),
                generation_error=None,
                retrieval_attempts=search_count,
                standalone_question=clean_question,
                trace=trace.events,
            )

        trace.record(
            stage="agent_finish",
            label="研究 Agent 确认证据覆盖完成",
            details={
                "reason_summary": decision.reason_summary,
                "search_count": search_count,
                "paper_count": len(selected),
                "selected_arxiv_ids": [paper.arxiv_id for paper in selected],
            },
        )
        return _generate_final_answer(
            clean_question,
            selected,
            settings=settings,
            client=client,
            on_retry=on_retry,
            history=history,
            search_count=search_count,
            trace=trace,
        )


def _generate_final_answer(
    question: str,
    papers: list[SearchResult],
    *,
    settings: ModelSettings,
    client: Any | None,
    on_retry: StatusCallback | None,
    history: tuple[ConversationTurn, ...],
    search_count: int,
    trace: TraceRecorder,
) -> RagResult:
    started_at = start_timer()
    try:
        answer = answer_from_results(
            question,
            papers,
            settings=settings,
            client=client,
            on_retry=on_retry,
            conversation_history=history,
            standalone_question=question,
        )
    except LLMRequestError as error:
        trace.record(
            stage="answer_generation",
            label="DeepSeek 最终回答生成",
            status="failed",
            started_at=started_at,
            details={"error": str(error), "paper_count": len(papers)},
        )
        return RagResult(
            question=question,
            papers=tuple(papers),
            answer=None,
            generation_error=str(error),
            retrieval_attempts=search_count,
            standalone_question=question,
            trace=trace.events,
        )

    trace.record(
        stage="answer_generation",
        label="DeepSeek 最终回答生成",
        started_at=started_at,
        details={"paper_count": len(papers), "answer_char_count": len(answer)},
    )
    return RagResult(
        question=question,
        papers=tuple(papers),
        answer=answer,
        generation_error=None,
        retrieval_attempts=search_count,
        standalone_question=question,
        trace=trace.events,
    )


def _select_final_evidence(
    question: str,
    *,
    evidence: tuple[SearchResult, ...],
    observations: list[SearchObservation],
    top_k: int,
    reranker: Reranker,
) -> list[SearchResult]:
    """Preserve one paper per searched subquestion, then fill by global rank."""
    if not evidence:
        return []
    scores = reranker.score(question, [paper.document for paper in evidence])
    if len(scores) != len(evidence):
        raise ValueError("reranker returned an unexpected number of scores")
    globally_ranked = [
        replace(paper, rerank_score=float(score))
        for paper, score in zip(evidence, scores)
    ]
    globally_ranked.sort(
        key=lambda paper: (-float(paper.rerank_score or 0.0), paper.arxiv_id)
    )
    by_id = {paper.arxiv_id: paper for paper in globally_ranked}

    selected: list[SearchResult] = []
    selected_ids: set[str] = set()
    represented_subquestions: set[str] = set()
    for observation in observations:
        if observation.subquestion_id in represented_subquestions:
            continue
        candidates = [
            by_id[paper.arxiv_id]
            for paper in observation.papers
            if paper.arxiv_id in by_id
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda paper: (-float(paper.rerank_score or 0.0), paper.arxiv_id)
        )
        representative = candidates[0]
        if representative.arxiv_id not in selected_ids:
            selected.append(representative)
            selected_ids.add(representative.arxiv_id)
        represented_subquestions.add(observation.subquestion_id)
        if len(selected) >= top_k:
            return selected

    for paper in globally_ranked:
        if paper.arxiv_id not in selected_ids:
            selected.append(paper)
            selected_ids.add(paper.arxiv_id)
        if len(selected) >= top_k:
            break
    selected.sort(
        key=lambda paper: (-float(paper.rerank_score or 0.0), paper.arxiv_id)
    )
    return selected


def _run_web_search(
    query: str,
    *,
    web_search_client: WebSearchClient | None,
    observation_payloads: list[dict[str, object]],
    trace: TraceRecorder,
) -> None:
    """Run one auxiliary web search whose failure is an observation, not an abort."""
    trace.record(
        stage="agent_tool_call",
        label="Agent 调用网页搜索工具",
        details={"tool": "web_search", "query": query},
    )
    tool_started_at = start_timer()
    if web_search_client is None:
        error_message = "web search tool is not configured"
        results = None
    else:
        try:
            results = web_search_client.search(query)
            error_message = None
        except (WebSearchError, ValueError) as error:
            results = None
            error_message = str(error)

    if results is None:
        observation_payloads.append(
            {
                "tool": "web_search",
                "query": query,
                "result_count": 0,
                "error": error_message,
            }
        )
        trace.record(
            stage="agent_tool_observation",
            label="网页搜索工具执行失败",
            status="failed",
            started_at=tool_started_at,
            details={
                "tool": "web_search",
                "query": query,
                "error": error_message,
            },
        )
        return

    observation_payloads.append(
        {
            "tool": "web_search",
            "query": query,
            "result_count": len(results),
            "results": [
                {"title": item.title, "snippet": item.snippet} for item in results
            ],
        }
    )
    trace.record(
        stage="agent_tool_observation",
        label="Agent 观察网页搜索结果",
        started_at=tool_started_at,
        details={
            "tool": "web_search",
            "query": query,
            "result_count": len(results),
            "result_titles": [item.title for item in results],
            "result_urls": [item.url for item in results],
        },
    )


def _observation_payload(observation: SearchObservation) -> dict[str, object]:
    return {
        "tool": "search_papers",
        "target_subquestion_id": observation.subquestion_id,
        "query": observation.query,
        "result_count": len(observation.papers),
        "papers": [
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract_excerpt": paper.document[:OBSERVATION_EXCERPT_LIMIT],
            }
            for paper in observation.papers[:OBSERVATION_PAPER_LIMIT]
        ],
    }
