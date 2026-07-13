"""Health and knowledge-base status routes."""

from fastapi import APIRouter, Request

from api.runtime import runtime_from_request
from api.schemas import KnowledgeBaseStatsResponse
from ingestion.repository import get_paper_count


router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    """Report that startup resources are ready."""
    return {"status": "ok"}


@router.get(
    "/knowledge-base/stats",
    response_model=KnowledgeBaseStatsResponse,
)
def knowledge_base_stats(request: Request) -> KnowledgeBaseStatsResponse:
    """Return source and derived-index counts for frontend status display."""
    runtime = runtime_from_request(request)
    return KnowledgeBaseStatsResponse(
        paper_count=get_paper_count(runtime.database_path),
        vector_count=runtime.collection.count(),
    )
