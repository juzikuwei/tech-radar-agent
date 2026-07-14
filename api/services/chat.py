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
    MAX_STORED_TURNS,
    SAFE_CLARIFICATION_RESPONSE,
)
from rag.conversation_store import (
    ConversationNotFoundError,
    ConversationState,
    ConversationTurnLimitError,
    append_conversation_turn,
    load_conversation_state,
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
    conversation_id: str,
    payload: ChatRequest,
    runtime: RagRuntime,
    *,
    on_trace: TraceEventCallback | None = None,
) -> ChatResponse:
    """Load trusted conversation state, run RAG, and persist a completed turn."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    state = _load_chat_state(conversation_id, runtime)

    history = state.recent_turns
    active_evidence = _load_active_evidence(
        runtime.database_path,
        state.active_evidence_ids,
    )
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
            LOGGER.warning("Research Agent failed: %s", error)
            fallback_used = True
            research_trace = (
                error.trace if isinstance(error, ResearchAgentError) else ()
            )
            if isinstance(error, ResearchAgentError) and error.tool_calls == 0:
                clarification_event = TraceEvent(
                    stage="react_clarification",
                    label="研究 Agent 决策失败后请求澄清",
                    status="failed",
                    duration_ms=0.0,
                    details={"error": str(error)},
                )
                if on_trace is not None:
                    on_trace(clarification_event)
                result = RagResult(
                    question=question,
                    papers=(),
                    answer=SAFE_CLARIFICATION_RESPONSE,
                    generation_error=None,
                    retrieval_attempts=0,
                    standalone_question=question,
                    trace=(*research_trace, clarification_event),
                    response_kind="conversation",
                )
            else:
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
                    trace=(
                        *research_trace,
                        fallback_event,
                        *fallback_result.trace,
                    ),
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
    response = build_chat_response(
        result,
        mode=payload.mode,
        fallback_used=fallback_used,
    )
    if response.answer is not None:
        try:
            append_conversation_turn(
                runtime.database_path,
                conversation_id,
                user_message=question,
                assistant_message=response.answer,
                paper_ids=tuple(paper.arxiv_id for paper in response.papers),
                response_kind=response.response_kind,
                active_evidence_ids=(
                    None
                    if response.response_kind == "conversation"
                    else tuple(
                        paper.arxiv_id for paper in response.papers[:5]
                    )
                ),
            )
        except ConversationNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail="conversation not found",
            ) from error
        except ConversationTurnLimitError as error:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"该会话已达到 {MAX_STORED_TURNS} 轮上限，请新建会话后继续。"
                ),
            ) from error
    return response


def validate_chat_conversation(conversation_id: str, runtime: RagRuntime) -> None:
    """Reject a missing or full conversation before streaming headers start."""
    _load_chat_state(conversation_id, runtime)


def stream_chat(
    conversation_id: str,
    payload: ChatRequest,
    runtime: RagRuntime,
) -> Iterator[str]:
    """Yield completed trace stages followed by one complete chat result."""
    items: Queue[StreamItem | None] = Queue()

    def emit_trace(event: TraceEvent) -> None:
        items.put(("trace", event))

    def produce() -> None:
        try:
            result = execute_chat(
                conversation_id,
                payload,
                runtime,
                on_trace=emit_trace,
            )
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


def _load_chat_state(
    conversation_id: str,
    runtime: RagRuntime,
) -> ConversationState:
    try:
        state = load_conversation_state(runtime.database_path, conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail="conversation not found") from error
    if state.summary.turn_count >= MAX_STORED_TURNS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"该会话已达到 {MAX_STORED_TURNS} 轮上限，请新建会话后继续。"
            ),
        )
    return state


def _load_active_evidence(
    database_path: Path,
    active_ids: tuple[str, ...],
) -> tuple[SearchResult, ...]:
    papers = load_papers_by_arxiv_ids(database_path, active_ids)
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
        response_kind=result.response_kind,
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
