"""Typed HTTP request and response contracts."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_TOP_K = 5
MAX_TOP_K = 10
MAX_QUESTION_CHARS = 8_000


class ChatRequest(BaseModel):
    """Client-controlled options for one conversation-scoped RAG request."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    mode: Literal["pipeline", "react"] = "pipeline"


class PaperResponse(BaseModel):
    """One paper returned as grounded answer evidence."""

    arxiv_id: str
    title: str
    document: str
    entry_url: str
    primary_category: str
    similarity: float | None
    keyword_score: float | None
    fusion_score: float | None
    rerank_score: float | None


class TraceEventResponse(BaseModel):
    """JSON representation of one application trace event."""

    stage: str
    label: str
    status: str
    duration_ms: float
    details: dict[str, Any]


class ModelUsageResponse(BaseModel):
    """Provider-reported token usage for one complete agent run."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ConversationDecisionResponse(BaseModel):
    """Validated evidence action selected for a follow-up question."""

    coverage: str
    next_action: str
    reason: str
    standalone_question: str
    reusable_arxiv_ids: list[str]
    missing_aspects: list[str]
    retrieval_query: str | None


class ChatResponse(BaseModel):
    """Display-ready answer, evidence, and observable execution state."""

    question: str
    answer: str | None
    generation_error: str | None
    papers: list[PaperResponse]
    trace: list[TraceEventResponse]
    retrieval_attempts: int
    standalone_question: str | None
    conversation_decision: ConversationDecisionResponse | None
    response_kind: Literal["research", "conversation"] = "research"
    mode: Literal["pipeline", "react"] = "pipeline"
    fallback_used: bool = False
    usage: ModelUsageResponse | None = None


class ConversationSummaryResponse(BaseModel):
    """List-ready metadata for one persistent conversation."""

    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int


class ConversationTurnResponse(BaseModel):
    """One persisted turn with current paper metadata but no old Trace."""

    turn_id: int
    user_message: str
    assistant_message: str
    paper_ids: list[str]
    papers: list[PaperResponse]
    response_kind: Literal["research", "conversation"]
    created_at: str


class ConversationResponse(ConversationSummaryResponse):
    """One persistent conversation and its complete history."""

    turns: list[ConversationTurnResponse]


class KnowledgeBaseStatsResponse(BaseModel):
    """Current SQLite and vector-index sizes."""

    paper_count: int
    vector_count: int
