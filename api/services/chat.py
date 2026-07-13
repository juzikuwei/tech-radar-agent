"""Translate a validated chat request into one production RAG execution."""

import json
import logging
from dataclasses import replace
from pathlib import Path
from queue import Queue
from threading import Thread
from collections.abc import Iterator
from typing import Literal, TypeAlias

from fastapi import HTTPException

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDecisionResponse,
    PaperResponse,
    TraceEventResponse,
)
from ingestion.repository import load_papers_by_arxiv_ids
from rag.application import RagResult, run_rag
from rag.conversation import (
    ConversationDecision,
    ConversationTurn,
    MAX_ACTIVE_EVIDENCE,
    MAX_CONVERSATION_TURNS,
)
from rag.execution_trace import TraceEvent, TraceEventCallback
from rag.search import SearchResult
from rag.similarity import build_embedding_text
from rag.runtime import RagRuntime
from rag.research_agent import ResearchAgentError, run_research_agent


LOGGER = logging.getLogger(__name__)
StreamItem: TypeAlias = tuple[Literal["trace"], TraceEvent] | tuple[
    Literal["result"], ChatResponse
] | tuple[Literal["error"], str]


def execute_chat(
    payload: ChatRequest,
    runtime: RagRuntime,
    *,
    on_trace: TraceEventCallback | None = None,
) -> ChatResponse:
    """Validate bounded client state and run the shared RAG application."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    history = _build_history(payload)
    active_ids = _validate_active_ids(payload.active_evidence_ids)
    active_evidence = _load_active_evidence(runtime.database_path, active_ids)
    fallback_used = False
    if payload.mode == "react":
        try:
            result = run_research_agent(
                question,
                top_k=payload.top_k,
                collection=runtime.collection,
                embedder=runtime.embedder,
                reranker=runtime.reranker,
                database_path=runtime.database_path,
                settings=runtime.settings,
                conversation_history=history,
                active_evidence=active_evidence,
                on_trace=on_trace,
                web_search_client=runtime.web_search_client,
            )
        except Exception as error:
            LOGGER.warning("Research Agent failed; falling back to pipeline: %s", error)
            fallback_used = True
            research_trace = (
                error.trace if isinstance(error, ResearchAgentError) else ()
            )
            fallback_event = TraceEvent(
                stage="react_fallback",
                label="研究 Agent 降级到可靠管线",
                status="failed",
                duration_ms=0.0,
                details={"error": str(error)},
            )
            if on_trace is not None:
                on_trace(fallback_event)
            fallback_result = run_rag(
                question,
                top_k=payload.top_k,
                collection=runtime.collection,
                embedder=runtime.embedder,
                reranker=runtime.reranker,
                database_path=runtime.database_path,
                settings=runtime.settings,
                conversation_history=history,
                active_evidence=active_evidence,
                on_trace=on_trace,
            )
            result = replace(
                fallback_result,
                trace=(*research_trace, fallback_event, *fallback_result.trace),
            )
    else:
        result = run_rag(
            question,
            top_k=payload.top_k,
            collection=runtime.collection,
            embedder=runtime.embedder,
            reranker=runtime.reranker,
            database_path=runtime.database_path,
            settings=runtime.settings,
            conversation_history=history,
            active_evidence=active_evidence,
            on_trace=on_trace,
        )
    return build_chat_response(
        result,
        mode=payload.mode,
        fallback_used=fallback_used,
    )


def stream_chat(payload: ChatRequest, runtime: RagRuntime) -> Iterator[str]:
    """Yield completed trace stages followed by one complete chat result."""
    items: Queue[StreamItem | None] = Queue()

    def emit_trace(event: TraceEvent) -> None:
        items.put(("trace", event))

    def produce() -> None:
        try:
            result = execute_chat(payload, runtime, on_trace=emit_trace)
            items.put(("result", result))
        except HTTPException as error:
            items.put(("error", _stream_error_message(error.detail)))
        except Exception:
            LOGGER.exception("Unhandled error while streaming a chat request")
            items.put(("error", "处理请求时发生错误，请稍后重试。"))
        finally:
            items.put(None)

    Thread(target=produce, name="chat-trace-stream", daemon=True).start()
    yield _stream_line(
        {
            "type": "run_started",
            "question": payload.question.strip(),
            "mode": payload.mode,
        }
    )

    while True:
        item = items.get()
        if item is None:
            return
        item_type, value = item
        if item_type == "trace":
            yield _stream_line(
                {
                    "type": "trace",
                    "event": _trace_response(value).model_dump(mode="json"),
                }
            )
        elif item_type == "result":
            yield _stream_line(
                {
                    "type": "result",
                    "result": value.model_dump(mode="json"),
                }
            )
        else:
            yield _stream_line({"type": "error", "message": value})


def _stream_line(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _stream_error_message(detail: object) -> str:
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, ensure_ascii=False)


def _build_history(payload: ChatRequest) -> tuple[ConversationTurn, ...]:
    if len(payload.conversation_history) > MAX_CONVERSATION_TURNS:
        raise HTTPException(
            status_code=422,
            detail=(
                "conversation_history supports at most "
                f"{MAX_CONVERSATION_TURNS} turns"
            ),
        )
    history = tuple(
        ConversationTurn(
            user_message=turn.user_message.strip(),
            assistant_message=turn.assistant_message.strip(),
        )
        for turn in payload.conversation_history
    )
    if any(not turn.user_message or not turn.assistant_message for turn in history):
        raise HTTPException(
            status_code=422,
            detail="conversation messages must not be blank",
        )
    return history


def _validate_active_ids(active_evidence_ids: list[str]) -> list[str]:
    active_ids = [paper_id.strip() for paper_id in active_evidence_ids]
    if any(not paper_id for paper_id in active_ids):
        raise HTTPException(
            status_code=422,
            detail="active_evidence_ids must contain only non-empty strings",
        )
    if len(active_ids) > MAX_ACTIVE_EVIDENCE:
        raise HTTPException(
            status_code=422,
            detail=(
                "active_evidence_ids supports at most "
                f"{MAX_ACTIVE_EVIDENCE} IDs"
            ),
        )
    if len(set(active_ids)) != len(active_ids):
        raise HTTPException(
            status_code=422,
            detail="active_evidence_ids must not contain duplicates",
        )
    return active_ids


def _load_active_evidence(
    database_path: Path,
    active_ids: list[str],
) -> tuple[SearchResult, ...]:
    papers = load_papers_by_arxiv_ids(database_path, active_ids)
    found_ids = {paper["arxiv_id"] for paper in papers}
    missing_ids = [paper_id for paper_id in active_ids if paper_id not in found_ids]
    if missing_ids:
        raise HTTPException(
            status_code=422,
            detail={"unknown_active_evidence_ids": missing_ids},
        )
    return tuple(
        SearchResult(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            document=build_embedding_text(paper["title"], paper["abstract"]),
            entry_url=paper["entry_url"],
            primary_category=paper["primary_category"],
            similarity=None,
        )
        for paper in papers
    )


def build_chat_response(
    result: RagResult,
    *,
    mode: Literal["pipeline", "react"] = "pipeline",
    fallback_used: bool = False,
) -> ChatResponse:
    """Serialize the application result without exposing Python objects."""
    return ChatResponse(
        question=result.question,
        answer=result.answer,
        generation_error=result.generation_error,
        papers=[_paper_response(paper) for paper in result.papers],
        trace=[_trace_response(event) for event in result.trace],
        retrieval_attempts=result.retrieval_attempts,
        standalone_question=result.standalone_question,
        conversation_decision=(
            _conversation_decision_response(result.conversation_decision)
            if result.conversation_decision is not None
            else None
        ),
        mode=mode,
        fallback_used=fallback_used,
    )


def _paper_response(paper: SearchResult) -> PaperResponse:
    return PaperResponse(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        document=paper.document,
        entry_url=paper.entry_url,
        primary_category=paper.primary_category,
        similarity=paper.similarity,
        keyword_score=paper.keyword_score,
        fusion_score=paper.fusion_score,
        rerank_score=paper.rerank_score,
    )


def _trace_response(event: TraceEvent) -> TraceEventResponse:
    return TraceEventResponse(
        stage=event.stage,
        label=event.label,
        status=event.status,
        duration_ms=event.duration_ms,
        details=event.details,
    )


def _conversation_decision_response(
    decision: ConversationDecision,
) -> ConversationDecisionResponse:
    return ConversationDecisionResponse(
        coverage=decision.coverage,
        next_action=decision.next_action,
        reason=decision.reason,
        standalone_question=decision.standalone_question,
        reusable_arxiv_ids=list(decision.reusable_arxiv_ids),
        missing_aspects=list(decision.missing_aspects),
        retrieval_query=decision.retrieval_query,
    )
