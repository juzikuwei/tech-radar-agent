"""Read-only knowledge-base operations shared by external adapters."""

from rag.hybrid_search import hybrid_search
from rag.runtime import RagRuntime
from rag.search import SearchResult


def search_knowledge_base(
    query: str,
    *,
    top_k: int,
    runtime: RagRuntime,
) -> list[SearchResult]:
    """Return reranked local papers without invoking an answer model."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    return hybrid_search(
        clean_query,
        top_k=top_k,
        collection=runtime.collection,
        embedder=runtime.embedder,
        reranker=runtime.reranker,
        database_path=runtime.database_path,
    )
