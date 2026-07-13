"""Bounded public payloads returned by MCP tools."""

from rag.search import SearchResult


MAX_ABSTRACT_EXCERPT_CHARS = 600


def search_result_payload(result: SearchResult) -> dict[str, object]:
    """Return only evidence fields useful to an external Agent."""
    abstract = _abstract_from_document(result.document)
    return {
        "arxiv_id": result.arxiv_id,
        "title": result.title,
        "abstract_excerpt": _truncate(abstract, MAX_ABSTRACT_EXCERPT_CHARS),
        "primary_category": result.primary_category,
        "score": (
            round(result.rerank_score, 4)
            if result.rerank_score is not None
            else None
        ),
        "entry_url": result.entry_url,
    }


def paper_payload(paper: dict[str, str]) -> dict[str, object]:
    """Return the minimal stored paper detail contract."""
    return {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "primary_category": paper["primary_category"],
        "published_at": paper["published_at"],
        "entry_url": paper["entry_url"],
    }


def _abstract_from_document(document: str) -> str:
    _, separator, abstract = document.partition("\n")
    return abstract.strip() if separator else document.strip()


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"
