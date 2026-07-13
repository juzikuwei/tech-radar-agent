"""Application-level orchestration shared by CLI and UI entry points."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection

from config.model_settings import ModelSettings
from rag.conversation import (
    ConversationDecision,
    ConversationDecisionError,
    ConversationTurn,
    MAX_ACTIVE_EVIDENCE,
    bounded_history,
    decide_conversation_action,
)
from rag.embedder import E5Embedder
from rag.execution_trace import (
    TraceEvent,
    TraceEventCallback,
    TraceRecorder,
    start_timer,
)
from rag.hybrid_search import hybrid_search
from rag.keyword_search import DEFAULT_DATABASE_PATH
from rag.llm_client import LLMRequestError, StatusCallback, generate_text
from rag.prompt_builder import build_rag_messages
from rag.reranker import Reranker
from rag.retrieval_judge import (
    RetrievalDecision,
    RetrievalDecisionError,
    judge_retrieval,
)
from rag.search import SearchResult, search_collection


@dataclass(frozen=True)
class RagResult:
    """One retrieval run, preserving evidence when generation fails."""

    question: str
    papers: tuple[SearchResult, ...]
    answer: str | None
    generation_error: str | None
    retrieval_attempts: int = 0
    standalone_question: str | None = None
    conversation_decision: ConversationDecision | None = None
    conversation_decision_error: str | None = None
    retrieval_decision: RetrievalDecision | None = None
    retrieval_decision_error: str | None = None
    trace: tuple[TraceEvent, ...] = ()


def answer_from_results(
    question: str,
    results: list[SearchResult],
    *,
    settings: ModelSettings,
    client: Any | None = None,
    on_retry: StatusCallback | None = None,
    conversation_history: tuple[ConversationTurn, ...] = (),
    standalone_question: str | None = None,
) -> str:
    """Generate a display-ready answer from already retrieved papers."""
    messages = build_rag_messages(
        question,
        results,
        conversation_history=conversation_history,
        standalone_question=standalone_question,
    )
    return generate_text(
        messages,
        settings=settings,
        client=client,
        on_retry=on_retry,
    )


def run_rag(
    question: str,
    *,
    top_k: int,
    collection: Collection,
    embedder: E5Embedder,
    settings: ModelSettings,
    reranker: Reranker | None = None,
    database_path: Path = DEFAULT_DATABASE_PATH,
    client: Any | None = None,
    on_retry: StatusCallback | None = None,
    conversation_history: tuple[ConversationTurn, ...] = (),
    active_evidence: tuple[SearchResult, ...] = (),
    on_trace: TraceEventCallback | None = None,
) -> RagResult:
    """Reuse conversational evidence or perform at most two new retrievals."""
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question must not be empty")

    trace = TraceRecorder(on_event=on_trace)
    history = bounded_history(conversation_history)
    previous_evidence = list(active_evidence[:MAX_ACTIVE_EVIDENCE])
    retrieval_attempts = 0
    standalone_question = clean_question
    conversation_decision: ConversationDecision | None = None
    conversation_decision_error: str | None = None
    retrieval_decision: RetrievalDecision | None = None
    retrieval_decision_error: str | None = None
    should_judge_retrieval = True

    if history and reranker is not None:
        started_at = start_timer()
        try:
            conversation_decision = decide_conversation_action(
                clean_question,
                history,
                previous_evidence,
                settings=settings,
                client=client,
                on_retry=on_retry,
            )
            standalone_question = conversation_decision.standalone_question
            trace.record(
                stage="conversation_evidence_decision",
                label="DeepSeek 对话证据动作判断",
                started_at=started_at,
                details={
                    "coverage": conversation_decision.coverage,
                    "next_action": conversation_decision.next_action,
                    "reason": conversation_decision.reason,
                    "standalone_question": standalone_question,
                    "reusable_arxiv_ids": list(
                        conversation_decision.reusable_arxiv_ids
                    ),
                    "missing_aspects": list(
                        conversation_decision.missing_aspects
                    ),
                    "retrieval_query": conversation_decision.retrieval_query,
                },
            )

            reusable_results = _select_reusable_evidence(
                previous_evidence,
                conversation_decision.reusable_arxiv_ids,
            )
            if conversation_decision.next_action == "answer_from_existing":
                results = _rerank_existing_evidence(
                    standalone_question,
                    reusable_results,
                    top_k=top_k,
                    reranker=reranker,
                    trace=trace,
                )
                should_judge_retrieval = False
            elif conversation_decision.next_action == "retrieve_missing":
                new_results = _run_retrieval(
                    conversation_decision.retrieval_query or standalone_question,
                    top_k=top_k,
                    collection=collection,
                    embedder=embedder,
                    reranker=reranker,
                    database_path=database_path,
                    trace=trace,
                    retrieval_round=1,
                )
                retrieval_attempts = 1
                results = _merge_and_rerank(
                    standalone_question,
                    reusable_results,
                    new_results,
                    top_k=top_k,
                    reranker=reranker,
                    trace=trace,
                )
            else:
                results = _run_retrieval(
                    conversation_decision.retrieval_query or standalone_question,
                    top_k=top_k,
                    collection=collection,
                    embedder=embedder,
                    reranker=reranker,
                    database_path=database_path,
                    trace=trace,
                    retrieval_round=1,
                )
                retrieval_attempts = 1
        except (LLMRequestError, ConversationDecisionError) as error:
            conversation_decision_error = str(error)
            trace.record(
                stage="conversation_evidence_decision",
                label="DeepSeek 对话证据动作判断",
                status="failed",
                started_at=started_at,
                details={"error": conversation_decision_error},
            )
            results = _run_retrieval(
                clean_question,
                top_k=top_k,
                collection=collection,
                embedder=embedder,
                reranker=reranker,
                database_path=database_path,
                trace=trace,
                retrieval_round=1,
            )
            retrieval_attempts = 1
    else:
        results = _run_retrieval(
            clean_question,
            top_k=top_k,
            collection=collection,
            embedder=embedder,
            reranker=reranker,
            database_path=database_path,
            trace=trace,
            retrieval_round=1,
        )
        retrieval_attempts = 1

    if not results:
        return RagResult(
            question=clean_question,
            papers=(),
            answer=None,
            generation_error=None,
            retrieval_attempts=retrieval_attempts,
            standalone_question=standalone_question,
            conversation_decision=conversation_decision,
            conversation_decision_error=conversation_decision_error,
            trace=trace.events,
        )

    if reranker is not None and should_judge_retrieval:
        started_at = start_timer()
        try:
            retrieval_decision = judge_retrieval(
                standalone_question,
                results,
                settings=settings,
                client=client,
                on_retry=on_retry,
            )
            trace.record(
                stage="retrieval_judgment",
                label="DeepSeek 检索充分性判断",
                started_at=started_at,
                details={
                    "sufficient": retrieval_decision.sufficient,
                    "reason": retrieval_decision.reason,
                    "missing_aspects": list(
                        retrieval_decision.missing_aspects
                    ),
                    "rewritten_query": retrieval_decision.rewritten_query,
                },
            )
            if not retrieval_decision.sufficient and retrieval_attempts < 2:
                rewritten_results = hybrid_search(
                    retrieval_decision.rewritten_query or standalone_question,
                    top_k=top_k,
                    collection=collection,
                    embedder=embedder,
                    reranker=reranker,
                    database_path=database_path,
                    trace=trace,
                    retrieval_round=retrieval_attempts + 1,
                )
                retrieval_attempts += 1
                results = _merge_and_rerank(
                    standalone_question,
                    results,
                    rewritten_results,
                    top_k=top_k,
                    reranker=reranker,
                    trace=trace,
                )
        except (LLMRequestError, RetrievalDecisionError) as error:
            retrieval_decision_error = str(error)
            trace.record(
                stage="retrieval_judgment",
                label="DeepSeek 检索充分性判断",
                status="failed",
                started_at=started_at,
                details={"error": retrieval_decision_error},
            )

    started_at = start_timer()
    try:
        answer = answer_from_results(
            clean_question,
            results,
            settings=settings,
            client=client,
            on_retry=on_retry,
            conversation_history=history,
            standalone_question=standalone_question,
        )
    except LLMRequestError as error:
        trace.record(
            stage="answer_generation",
            label="DeepSeek 最终回答生成",
            status="failed",
            started_at=started_at,
            details={"error": str(error), "paper_count": len(results)},
        )
        return RagResult(
            question=clean_question,
            papers=tuple(results),
            answer=None,
            generation_error=str(error),
            retrieval_attempts=retrieval_attempts,
            standalone_question=standalone_question,
            conversation_decision=conversation_decision,
            conversation_decision_error=conversation_decision_error,
            retrieval_decision=retrieval_decision,
            retrieval_decision_error=retrieval_decision_error,
            trace=trace.events,
        )

    trace.record(
        stage="answer_generation",
        label="DeepSeek 最终回答生成",
        started_at=started_at,
        details={
            "paper_count": len(results),
            "answer_char_count": len(answer),
        },
    )

    return RagResult(
        question=clean_question,
        papers=tuple(results),
        answer=answer,
        generation_error=None,
        retrieval_attempts=retrieval_attempts,
        standalone_question=standalone_question,
        conversation_decision=conversation_decision,
        conversation_decision_error=conversation_decision_error,
        retrieval_decision=retrieval_decision,
        retrieval_decision_error=retrieval_decision_error,
        trace=trace.events,
    )


def _run_retrieval(
    query: str,
    *,
    top_k: int,
    collection: Collection,
    embedder: E5Embedder,
    reranker: Reranker | None,
    database_path: Path,
    trace: TraceRecorder,
    retrieval_round: int,
) -> list[SearchResult]:
    """Run the configured retriever and record the actual retrieval round."""
    if reranker is not None:
        return hybrid_search(
            query,
            top_k=top_k,
            collection=collection,
            embedder=embedder,
            reranker=reranker,
            database_path=database_path,
            trace=trace,
            retrieval_round=retrieval_round,
        )

    started_at = start_timer()
    results = search_collection(
        query,
        top_k=top_k,
        collection=collection,
        embedder=embedder,
    )
    trace.record(
        stage="dense_retrieval",
        label="E5 向量召回",
        started_at=started_at,
        details={
            "round": retrieval_round,
            "query": query,
            "result_count": len(results),
            "top_arxiv_ids": [result.arxiv_id for result in results],
        },
    )
    return results


def _select_reusable_evidence(
    active_evidence: list[SearchResult],
    reusable_ids: tuple[str, ...],
) -> list[SearchResult]:
    """Keep reusable papers in the model-specified order."""
    by_id = {paper.arxiv_id: paper for paper in active_evidence}
    return [by_id[paper_id] for paper_id in reusable_ids]


def _rerank_existing_evidence(
    question: str,
    evidence: list[SearchResult],
    *,
    top_k: int,
    reranker: Reranker,
    trace: TraceRecorder,
) -> list[SearchResult]:
    """Rerank reusable evidence for the current follow-up without retrieval."""
    started_at = start_timer()
    scores = reranker.score(question, [paper.document for paper in evidence])
    if len(scores) != len(evidence):
        raise ValueError("reranker returned an unexpected number of scores")
    reranked = [
        replace(paper, rerank_score=float(score))
        for paper, score in zip(evidence, scores)
    ]
    reranked.sort(
        key=lambda result: (-float(result.rerank_score or 0.0), result.arxiv_id)
    )
    final_results = reranked[: min(top_k, len(reranked))]
    trace.record(
        stage="active_evidence_rerank",
        label="活动证据复用与重排",
        started_at=started_at,
        details={
            "candidate_count": len(evidence),
            "result_count": len(final_results),
            "top_arxiv_ids": [paper.arxiv_id for paper in final_results],
        },
    )
    return final_results


def _merge_and_rerank(
    question: str,
    initial_results: list[SearchResult],
    rewritten_results: list[SearchResult],
    *,
    top_k: int,
    reranker: Reranker,
    trace: TraceRecorder | None = None,
) -> list[SearchResult]:
    """Deduplicate two retrieval rounds and rerank them for the original question."""
    started_at = start_timer()
    candidates: dict[str, SearchResult] = {}
    for result in [*initial_results, *rewritten_results]:
        candidates.setdefault(result.arxiv_id, result)
    if not candidates:
        return []

    unique_results = list(candidates.values())
    scores = reranker.score(
        question,
        [result.document for result in unique_results],
    )
    if len(scores) != len(unique_results):
        raise ValueError("reranker returned an unexpected number of scores")

    reranked = [
        replace(result, rerank_score=float(score))
        for result, score in zip(unique_results, scores)
    ]
    reranked.sort(
        key=lambda result: (-float(result.rerank_score or 0.0), result.arxiv_id)
    )
    final_results = reranked[: min(top_k, len(reranked))]
    if trace is not None:
        trace.record(
            stage="final_union_rerank",
            label="已有证据与新候选合并重排",
            started_at=started_at,
            details={
                "first_round_count": len(initial_results),
                "second_round_count": len(rewritten_results),
                "existing_count": len(initial_results),
                "new_count": len(rewritten_results),
                "unique_candidate_count": len(unique_results),
                "result_count": len(final_results),
                "top_arxiv_ids": [
                    result.arxiv_id for result in final_results
                ],
            },
        )
    return final_results
