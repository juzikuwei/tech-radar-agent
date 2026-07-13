"""Typed HTTP request and response contracts."""

from typing import Any, Literal

from pydantic import BaseModel, Field


DEFAULT_TOP_K = 5
MAX_TOP_K = 10


class ConversationTurnInput(BaseModel):
    """One completed conversation turn supplied by the client."""

    user_message: str = Field(min_length=1)
    assistant_message: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Bounded client state required for one RAG request."""

    question: str = Field(min_length=1)
    conversation_history: list[ConversationTurnInput] = Field(default_factory=list)
    active_evidence_ids: list[str] = Field(default_factory=list)
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
    mode: Literal["pipeline", "react"] = "pipeline"
    fallback_used: bool = False


class KnowledgeBaseStatsResponse(BaseModel):
    """Current SQLite and vector-index sizes."""

    paper_count: int
    vector_count: int
