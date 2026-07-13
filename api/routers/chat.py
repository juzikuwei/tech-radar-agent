"""Chat HTTP route."""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.runtime import runtime_from_request
from api.schemas import ChatRequest, ChatResponse
from api.services.chat import execute_chat, stream_chat


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Run the production RAG pipeline with bounded client state."""
    return execute_chat(payload, runtime_from_request(request))


@router.post("/chat/stream", response_class=StreamingResponse)
def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """Stream completed trace stages, then send the complete answer."""
    return StreamingResponse(
        stream_chat(payload, runtime_from_request(request)),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
