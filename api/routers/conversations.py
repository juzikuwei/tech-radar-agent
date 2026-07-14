"""Persistent conversation and conversation-scoped chat routes."""

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from api.runtime import runtime_from_request
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    ConversationSummaryResponse,
    ConversationTurnResponse,
    PaperResponse,
)
from api.services.chat import (
    execute_chat,
    stream_chat,
    validate_chat_conversation,
)
from ingestion.repository import load_papers_by_arxiv_ids
from rag.conversation_store import (
    ConversationNotFoundError,
    ConversationSummary,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
)
from rag.similarity import build_embedding_text


router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationSummaryResponse, status_code=201)
def create_persistent_conversation(request: Request) -> ConversationSummaryResponse:
    """Create an empty persistent conversation."""
    runtime = runtime_from_request(request)
    return _summary_response(create_conversation(runtime.database_path))


@router.get("", response_model=list[ConversationSummaryResponse])
def list_persistent_conversations(request: Request) -> list[ConversationSummaryResponse]:
    """List persistent conversations from most recent to least recent."""
    runtime = runtime_from_request(request)
    return [
        _summary_response(summary)
        for summary in list_conversations(runtime.database_path)
    ]


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_persistent_conversation(
    conversation_id: str,
    request: Request,
) -> ConversationResponse:
    """Return a conversation's complete text and citation history."""
    runtime = runtime_from_request(request)
    try:
        conversation = get_conversation(runtime.database_path, conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail="conversation not found") from error

    unique_paper_ids = tuple(
        dict.fromkeys(
            paper_id
            for turn in conversation.turns
            for paper_id in turn.paper_ids
        )
    )
    paper_records = load_papers_by_arxiv_ids(
        runtime.database_path,
        unique_paper_ids,
    )
    papers_by_id = {
        paper["arxiv_id"]: _stored_paper_response(paper)
        for paper in paper_records
    }
    summary = conversation.summary
    return ConversationResponse(
        conversation_id=summary.conversation_id,
        title=summary.title,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        turn_count=summary.turn_count,
        turns=[
            ConversationTurnResponse(
                turn_id=turn.turn_id,
                user_message=turn.user_message,
                assistant_message=turn.assistant_message,
                paper_ids=list(turn.paper_ids),
                papers=[
                    papers_by_id[paper_id]
                    for paper_id in turn.paper_ids
                    if paper_id in papers_by_id
                ],
                response_kind=turn.response_kind,
                created_at=turn.created_at,
            )
            for turn in conversation.turns
        ],
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_persistent_conversation(
    conversation_id: str,
    request: Request,
) -> Response:
    """Delete one conversation and all of its turns."""
    runtime = runtime_from_request(request)
    try:
        delete_conversation(runtime.database_path, conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail="conversation not found") from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{conversation_id}/chat", response_model=ChatResponse)
def chat(
    conversation_id: str,
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    """Run and persist one conversation-scoped RAG turn."""
    return execute_chat(
        conversation_id,
        payload,
        runtime_from_request(request),
    )


@router.post("/{conversation_id}/chat/stream", response_class=StreamingResponse)
def chat_stream(
    conversation_id: str,
    payload: ChatRequest,
    request: Request,
) -> StreamingResponse:
    """Stream Trace events and persist the completed turn before the result."""
    runtime = runtime_from_request(request)
    validate_chat_conversation(conversation_id, runtime)
    return StreamingResponse(
        stream_chat(
            conversation_id,
            payload,
            runtime,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


def _summary_response(summary: ConversationSummary) -> ConversationSummaryResponse:
    return ConversationSummaryResponse(
        conversation_id=summary.conversation_id,
        title=summary.title,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        turn_count=summary.turn_count,
    )


def _stored_paper_response(paper: dict[str, str]) -> PaperResponse:
    return PaperResponse(
        arxiv_id=paper["arxiv_id"],
        title=paper["title"],
        document=build_embedding_text(paper["title"], paper["abstract"]),
        entry_url=paper["entry_url"],
        primary_category=paper["primary_category"],
        similarity=None,
        keyword_score=None,
        fusion_score=None,
        rerank_score=None,
    )
