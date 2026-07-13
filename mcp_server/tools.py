"""Read-only MCP tool implementations with bounded outputs."""

from collections.abc import Callable

from ingestion.repository import get_paper_count, load_papers_by_arxiv_ids
from mcp_server.serializers import paper_payload, search_result_payload
from rag.knowledge_base import search_knowledge_base
from rag.runtime import RagRuntime


DEFAULT_TOOL_TOP_K = 3
MAX_TOOL_TOP_K = 5


class KnowledgeBaseTools:
    """Expose safe knowledge-base operations to MCP clients."""

    def __init__(self, runtime_provider: Callable[[], RagRuntime]) -> None:
        self._runtime_provider = runtime_provider

    def query_knowledge_base(
        self,
        query: str,
        top_k: int = DEFAULT_TOOL_TOP_K,
    ) -> dict[str, object]:
        """Search local arXiv abstracts and return at most five compact results."""
        if not 1 <= top_k <= MAX_TOOL_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOOL_TOP_K}")
        runtime = self._runtime_provider()
        results = search_knowledge_base(query, top_k=top_k, runtime=runtime)
        return {"results": [search_result_payload(result) for result in results]}

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> dict[str, object]:
        """Return one paper's normalized abstract and public metadata."""
        clean_id = arxiv_id.strip()
        if not clean_id:
            raise ValueError("arxiv_id must not be empty")
        runtime = self._runtime_provider()
        papers = load_papers_by_arxiv_ids(runtime.database_path, [clean_id])
        if not papers:
            raise ValueError(f"paper not found: {clean_id}")
        return paper_payload(papers[0])

    def get_knowledge_base_stats(self) -> dict[str, int]:
        """Return only source and vector-index record counts."""
        runtime = self._runtime_provider()
        return {
            "paper_count": get_paper_count(runtime.database_path),
            "vector_count": runtime.collection.count(),
        }
